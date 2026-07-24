# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch
from rl_algorithms.amp_utils.motion_loader import AMPLoader
from rl_algorithms.amp_utils.normalizer import Normalizer
from rl_algorithms.rsl_rl.algorithms import (
    LocomotionPPO,
    LocomotionPPOEeErrorDelta,
    LocomotionPPOForce,
)
from rl_algorithms.rsl_rl.env import VecEnv
from rl_algorithms.rsl_rl.modules import (
    ActorCriticEncoder,
    AMPDiscriminator,
    VAEBlind,
    VAEBlindEeErrorDelta,
    VAEBlindForce,
)
from rl_algorithms.rsl_rl.utils import store_code_state


class LocomotionOnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cuda:0"):
        self.cfg = train_cfg
        self.policy_type = train_cfg["policy_type"]
        self.training_type = train_cfg["training_type"]
        self.module_cfg_dict = train_cfg["module_cfg_dict"]
        self.train_cfg_dict = train_cfg["train_cfg_dict"]
        self.amp_loader_cfg = train_cfg["amp_loader_cfg"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # check if use amp
        self.use_amp = self.train_cfg_dict["use_amp"] if "use_amp" in self.train_cfg_dict else False

        # check if use vae
        self.use_vae = self.train_cfg_dict["use_vae"] if "use_vae" in self.train_cfg_dict else False
        if self.use_vae:
            self.use_vae_exclusive_optimizer = self.train_cfg_dict["vae"]["use_exclusive_optimizer"] if "use_exclusive_optimizer" in self.train_cfg_dict["vae"] else False
        else:
            self.use_vae_exclusive_optimizer = False

        # get number of observations
        num_actions = self.env.num_actions

        # evaluate the policy class
        module_dict = {}
        # ActorCritic
        joint_pos_limits = self.env.unwrapped.scene["robot"].data.default_joint_pos_limits[0]
        joint_pos_action = self.env.unwrapped.action_manager._terms.get("joint_pos")
        action_joint_ids = getattr(joint_pos_action, "_joint_ids", None)
        if isinstance(action_joint_ids, slice):
            action_joint_pos_limits = joint_pos_limits[action_joint_ids]
        elif action_joint_ids is not None:
            action_joint_ids = torch.as_tensor(action_joint_ids, device=joint_pos_limits.device, dtype=torch.long)
            action_joint_pos_limits = joint_pos_limits[action_joint_ids]
        else:
            action_joint_pos_limits = joint_pos_limits
        if action_joint_pos_limits.shape[0] != num_actions:
            raise ValueError(
                "Action joint limit count does not match num_actions: "
                f"{action_joint_pos_limits.shape[0]} vs {num_actions}."
            )
        dof_range = (action_joint_pos_limits[:, 1] - action_joint_pos_limits[:, 0]).to(self.device)
        min_std = torch.tensor(self.module_cfg_dict['actor_critic']['min_normalized_std'], device=self.device) * (torch.abs(dof_range))
        self.module_cfg_dict['actor_critic']['min_normalized_std'] = min_std
        actor_critic: ActorCriticEncoder = eval(self.cfg["policy_type"]["actor_critic_type"])(
            self.module_cfg_dict['actor_critic']
        ).to(self.device)
        module_dict['actor_critic'] = actor_critic

        # AMP
        if self.use_amp:
            amp_data = AMPLoader(
                device,
                amp_loader_cfg=self.amp_loader_cfg ,
                time_between_frames=self.env.step_dt,
                num_preload_transitions=self.train_cfg_dict['amp']['num_preload_transitions'],
                motion_files=self.train_cfg_dict['amp']['motion_files'],
            )
            amp_normalizer = Normalizer(amp_data.observation_dim)
            amp_discriminator: AMPDiscriminator = AMPDiscriminator(
                amp_data.observation_dim * 2,
                self.module_cfg_dict['amp']['hidden_dims'],
                device,
            ).to(self.device)
            module_dict['amp_discriminator'] = amp_discriminator
            module_dict['amp_normalizer'] = amp_normalizer
            module_dict['amp_data'] = amp_data

        # VAE
        if self.use_vae:
            vae: VAEBlind = eval(self.cfg["policy_type"]["vae_type"])(self.module_cfg_dict['vae']).to(self.device)
            module_dict['vae'] = vae

        # initialize algorithm
        self.alg = eval(self.cfg["policy_type"].get("algo_type", "LocomotionPPO"))(
            module_dict=module_dict,
            train_cfg_dict=self.train_cfg_dict,
            device=self.device,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        obs_dict = self.env.get_observations()

        self.alg.init_storage(
            training_type=self.cfg["training_type"],
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            obs=obs_dict,
            action_shape=[num_actions],
            device=self.device,
        )

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [__file__]
        _ = self.env.reset()

    def init_logger(self):
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard, Neptune, Wandb or SwanLab summary writer(s), default: SwanLab.
            self.logger_type = self.cfg.get("logger", "swanlab")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg)
            elif self.logger_type == "swanlab":
                from rl_algorithms.rsl_rl.utils import SwanLabSummaryWriter

                self.writer = SwanLabSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg)
            elif self.logger_type == "wandb":
                from rl_algorithms.rsl_rl.utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb', 'swanlab' or 'tensorboard'.")

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False, profiler=None):  # noqa: C901
        # initialize writer
        self.init_logger()
        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs_dict = self.env.get_observations()
        obs_dict = obs_dict.to(self.device)
        next_amp_obs = obs_dict["amp_obs"].clone()
        amp_obs = next_amp_obs
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        episode_stats_enabled = (
            self.log_dir is not None
            and not self.disable_logs
            and not os.getenv("DISABLE_EP_STATS")
        )

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            # TODO: Do we need to synchronize empirical normalizers?
            #   Right now: No, because they all should converge to the same values "asymptotically".

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            completed_episode_stats = []

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    # AMP has its own replay buffer, while next_obs is replaced
                    # with the post-step target below.  Excluding both avoids
                    # cloning data that PPO immediately discards.
                    rollout_obs = (
                        obs_dict.exclude("amp_obs", "next_obs").detach().clone()
                    )
                    actions = self.alg.act(rollout_obs)
                    if self.use_amp:
                        amp_out = self.alg.amp_discriminator.discriminator_out(
                            amp_obs, next_amp_obs, normalizer=self.alg.amp_normalizer
                        )
                    else:
                        amp_out = None
                    # ``next_amp_obs`` is already an owned clone.  Rebinding keeps the
                    # previous transition alive after the variable is replaced below.
                    amp_obs = next_amp_obs
                    # Step the environment
                    (
                        obs_dict,
                        rewards,
                        dones,
                        infos,
                        reset_env_ids,
                        terminal_amp_states,
                        episode_reward,
                    ) = self.env.step(actions.to(self.device), amp_out)

                    # Move to device
                    obs_dict, rewards, dones = (
                        obs_dict.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    if 'amp_obs' in obs_dict:
                        next_amp_obs = obs_dict["amp_obs"].detach().clone()
                        if terminal_amp_states is not None:
                            next_amp_obs_with_term = next_amp_obs.clone()
                        else:
                            next_amp_obs_with_term = next_amp_obs
                    else:
                        next_amp_obs_with_term = None
                    if terminal_amp_states is not None:
                        next_amp_obs_with_term[reset_env_ids] = terminal_amp_states.detach()
                    # process_env_step() copies the transition into rollout
                    # storage before the environment can mutate this tensor.
                    next_actor_obs = obs_dict["next_obs"].detach()
                    # process the step
                    self.alg.process_env_step(
                        rewards,
                        dones,
                        infos,
                        amp_obs,
                        next_amp_obs_with_term,
                        next_actor_obs,
                    )

                    # book keeping
                    if self.log_dir is not None and not self.disable_logs:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        if episode_stats_enabled:
                            cur_reward_sum += rewards
                            cur_episode_length += 1
                            # env.step already computed the ordered reset ids.  Accumulate
                            # selected GPU values through the rollout, then perform one CPU
                            # transfer instead of two transfers plus another nonzero per step.
                            if reset_env_ids.numel() > 0:
                                completed_episode_stats.append(
                                    torch.stack(
                                        (
                                            cur_reward_sum[reset_env_ids],
                                            cur_episode_length[reset_env_ids],
                                        ),
                                        dim=-1,
                                    )
                                )
                                cur_reward_sum[reset_env_ids] = 0
                                cur_episode_length[reset_env_ids] = 0

                if completed_episode_stats:
                    stats_cpu = (
                        torch.cat(completed_episode_stats, dim=0)
                        .cpu()
                        .numpy()
                    )
                    rewbuffer.extend(stats_cpu[:, 0].tolist())
                    lenbuffer.extend(stats_cpu[:, 1].tolist())

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns
                self.alg.compute_returns(obs_dict)

            # update policy
            loss_dict = self.alg.update()
            self._synchronize_command_curriculum_state()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # 支持 save_interval 为单个 int 或 含有 int 的列表 / 元组
                if isinstance(self.save_interval, int):
                    if it % self.save_interval == 0:
                        self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
                elif isinstance(self.save_interval, (list, tuple)):
                    if it in self.save_interval:
                        self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if git_file_paths and hasattr(self.writer, "save_file"):
                    for path in git_file_paths:
                        self.writer.save_file(path)

            if profiler is not None:
                profiler.step()

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                # Some curriculum statistics are intentionally unavailable
                # until a complete episode has supplied a valid sample.  Do
                # not write NaN events: TensorBoard/SwanLab render those as
                # empty charts, which is indistinguishable from a broken log.
                finite_values = infotensor[torch.isfinite(infotensor)]
                if finite_values.numel() == 0:
                    continue
                value = torch.mean(finite_values)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.actor_critic.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            # everything else
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            # -- Losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # -- Rewards
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # -- episode info
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (
                               locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        # -- Save model
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "actor_critic_optimizer_state_dict": self.alg.optimizer_dict['actor_critic'].state_dict(),
            "iter": self.current_learning_iteration,
            # ``iter`` is the rollout index that has just completed.  Persist the next
            # index explicitly so resume neither repeats that rollout nor loses one
            # rollout when reconstructing time-driven schedules.
            "next_iter": self.current_learning_iteration + 1,
            "curriculum_state": self._capture_curriculum_state(),
            "training_schedule_state": self._capture_training_schedule_state(),
            "infos": infos,
        }

        if self.use_amp:
            saved_dict_amp = {
                "amp_discriminator_state_dict": self.alg.amp_discriminator.state_dict(),
                "amp_normalizer": self.alg.amp_normalizer,
            }
            saved_dict.update(saved_dict_amp)

        if self.use_vae:
            saved_dict_vae = {
                "vae_state_dict": self.alg.vae.state_dict(),
            }
            if self.use_vae_exclusive_optimizer:
                saved_dict_vae["vae_optimizer_state_dict"] = self.alg.optimizer_dict['vae'].state_dict()
            saved_dict.update(saved_dict_vae)

        # save model
        torch.save(saved_dict, path)

        # # upload model to external logging service
        # if not self.disable_logs:
        #     self.writer.save_model(path, self.current_learning_iteration)

    def _capture_curriculum_state(self) -> dict[str, object]:
        """Capture restorable curriculum state alongside the policy checkpoint."""
        env = self.env.unwrapped
        state: dict[str, object] = {
            "common_step_counter": int(
                getattr(env, "common_step_counter", 0)
            )
        }
        base_cmd = None
        try:
            base_cmd = env.command_manager.get_term("base_command")
            for attr in ("lin_x_level", "ang_z_level"):
                if hasattr(base_cmd.cfg, attr):
                    state[attr] = float(getattr(base_cmd.cfg, attr))
        except (AttributeError, KeyError, TypeError, ValueError):
            pass

        if base_cmd is not None:
            command_curriculum_hooks = self._get_command_curriculum_hooks(
                base_cmd
            )
            if command_curriculum_hooks is not None:
                command_state = command_curriculum_hooks[0]()
                if not isinstance(command_state, dict):
                    raise TypeError(
                        "base_command.curriculum_state_dict() must return a dict, "
                        f"got {type(command_state).__name__}."
                    )
                state["base_command"] = command_state
        if hasattr(env, "ee_force_curriculum_level"):
            state["ee_force_curriculum_level"] = float(
                env.ee_force_curriculum_level
            )
        return state

    @staticmethod
    def _get_command_curriculum_hooks(base_cmd):
        """Return the optional command-owned curriculum hooks when fully implemented."""
        hooks = tuple(
            getattr(base_cmd, name, None)
            for name in (
                "curriculum_state_dict",
                "load_curriculum_state_dict",
                "apply_curriculum_ranges",
            )
        )
        return hooks if all(callable(hook) for hook in hooks) else None

    @staticmethod
    def _get_command_curriculum_sync_hooks(base_cmd):
        """Return the optional fixed-tensor distributed synchronization hooks."""
        hooks = tuple(
            getattr(base_cmd, name, None)
            for name in (
                "distributed_curriculum_state_tensor",
                "load_distributed_curriculum_state_tensor",
            )
        )
        return hooks if all(callable(hook) for hook in hooks) else None

    def _synchronize_command_curriculum_state(self) -> None:
        """Broadcast opt-in command curriculum state from rank zero once per update."""
        if not self.is_distributed:
            return
        try:
            base_cmd = self.env.unwrapped.command_manager.get_term("base_command")
        except (AttributeError, KeyError, TypeError, ValueError):
            return
        sync_hooks = self._get_command_curriculum_sync_hooks(base_cmd)
        if sync_hooks is not None:
            # Use one small, fixed-layout NCCL tensor here to avoid a dozen
            # GPU-to-CPU scalar synchronizations plus Python object pickling on
            # every PPO update.  The command owns the layout, including all
            # EMA/counter fields needed to retain exact synchronization semantics.
            state = sync_hooks[0]()
            if not isinstance(state, torch.Tensor):
                raise TypeError(
                    "base_command.distributed_curriculum_state_tensor() must "
                    f"return a tensor, got {type(state).__name__}."
                )
            if state.device != torch.device(self.device):
                raise ValueError(
                    "Distributed base-command curriculum tensor must be on the "
                    f"runner device {self.device}, got {state.device}."
                )
            torch.distributed.broadcast(state, src=0)
            if self.gpu_global_rank != 0:
                sync_hooks[1](state)
            return

        hooks = self._get_command_curriculum_hooks(base_cmd)
        if hooks is None:
            return

        payload = [hooks[0]() if self.gpu_global_rank == 0 else None]
        torch.distributed.broadcast_object_list(
            payload,
            src=0,
            device=torch.device(self.device),
        )
        if not isinstance(payload[0], dict):
            raise TypeError(
                "Distributed base-command curriculum state must be a dict, "
                f"got {type(payload[0]).__name__}."
            )
        if self.gpu_global_rank != 0:
            hooks[1](payload[0])
            hooks[2]()

    def _capture_training_schedule_state(self) -> dict[str, float]:
        """Capture non-module/non-optimizer scalar schedules needed for exact resume."""
        state = {
            "learning_rate": float(self.alg.learning_rate),
        }
        for attr in ("vae_beta", "p_boot"):
            if hasattr(self.alg, attr):
                state[attr] = float(getattr(self.alg, attr))
        return state

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=False)
        # -- Load model
        resumed_training = self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if self.use_amp:
            self.alg.amp_discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"], strict=True)
            self.alg.amp_normalizer = loaded_dict["amp_normalizer"]
        if self.use_vae:
            self.alg.vae.load_state_dict(loaded_dict["vae_state_dict"], strict=True)
        # -- load optimizer if used
        if load_optimizer and resumed_training:
            # -- algorithm optimizer
            for key, _ in self.alg.optimizer_dict.items():
                self.alg.optimizer_dict[key].load_state_dict(loaded_dict[f"{key}_optimizer_state_dict"])
        # -- load current learning iteration
        if resumed_training:
            last_iteration = int(loaded_dict["iter"])
            next_iteration = int(
                loaded_dict.get("next_iter", last_iteration + 1)
            )
            self.current_learning_iteration = next_iteration

            saved_curriculum = loaded_dict.get("curriculum_state")
            if isinstance(saved_curriculum, dict):
                self._restore_curriculum_state(saved_curriculum)
            else:
                # Backward compatibility for existing checkpoints that only carry ``iter``.
                self._restore_curriculum_from_iter(last_iteration)

            saved_schedule = loaded_dict.get("training_schedule_state")
            if isinstance(saved_schedule, dict):
                self._restore_training_schedule_state(saved_schedule)
            else:
                # Old checkpoints keep the adaptive LR in the optimizer state, while
                # VAE beta/p_boot must still be reconstructed from completed updates.
                if load_optimizer:
                    self._restore_learning_rate_from_optimizer()
                self._restore_annealed_vae_from_updates(next_iteration)

            # The runner/environment are newly constructed.  Re-reset only for actual
            # training so the just-materialized command ranges/current force amplitude are
            # reflected in the initial commands instead of waiting for later episode resets.
            if load_optimizer and self.log_dir is not None:
                _ = self.env.reset()
                self._prime_ee_external_force_after_resume()
        return loaded_dict.get("infos")

    def _restore_training_schedule_state(self, state: dict) -> None:
        """Restore scalar schedules that are not part of module state_dicts."""
        if "learning_rate" in state:
            self.alg.learning_rate = float(state["learning_rate"])
        for attr in ("vae_beta", "p_boot"):
            if attr in state and hasattr(self.alg, attr):
                setattr(self.alg, attr, float(state[attr]))
        print(f"[INFO] Restored training schedule state: {state}")

    def _restore_learning_rate_from_optimizer(self) -> None:
        """Recover adaptive PPO LR from an old checkpoint's optimizer param groups."""
        optimizer = self.alg.optimizer_dict.get("actor_critic")
        if optimizer is None or not optimizer.param_groups:
            return
        actor_group = next(
            (
                group
                for group in optimizer.param_groups
                if group.get("name") == "actor_critic"
            ),
            optimizer.param_groups[0],
        )
        self.alg.learning_rate = float(actor_group["lr"])
        print(
            "[INFO] Restored adaptive learning rate from optimizer: "
            f"{self.alg.learning_rate}"
        )

    def _prime_ee_external_force_after_resume(self) -> None:
        """Start one restored-range EE force ramp instead of waiting a fresh interval."""
        env = self.env.unwrapped
        if float(
            getattr(env, "ee_force_curriculum_current_max", 0.0)
        ) <= 0.0:
            return
        event_cfg = None
        event_manager = getattr(env, "event_manager", None)
        if event_manager is not None and hasattr(
            event_manager, "get_term_cfg"
        ):
            try:
                event_cfg = event_manager.get_term_cfg(
                    "ee_external_force"
                )
            except (KeyError, ValueError):
                event_cfg = None
        if event_cfg is None:
            event_cfg = getattr(
                getattr(getattr(env, "cfg", None), "events", None),
                "ee_external_force",
                None,
            )
        event_func = getattr(event_cfg, "func", None)
        event_params = getattr(event_cfg, "params", None)
        if not callable(event_func) or not isinstance(event_params, dict):
            print(
                "[WARN] EE force curriculum was restored, but its interval "
                "event could not be primed."
            )
            return
        env_ids = torch.arange(env.num_envs, device=env.device)
        event_func(env, env_ids, **event_params)
        print(
            "[INFO] Primed EE external-force ramp from restored curriculum "
            f"range={event_params.get('force_x_range')}"
        )

    def _restore_curriculum_from_iter(self, iteration: int):
        """resume 时按 iter 反推课程进度，避免从 0 重新爬。

        - 速度命令课程（lin_x / ang_z）和末端外力课程是时间驱动的：由
          ``(iter + 1) * num_steps_per_env`` 反推 ``common_step_counter`` 与 level，
          因为 checkpoint 在该 iter 的 rollout/update 完成后才保存。
        - 地形课程按机器人表现逐 env 升降，无法由 iter 反推；resume 后会自行
          快速回升，故不在此处理。
        """
        steps_per_env = max(1, int(getattr(self, "num_steps_per_env", 1)))
        common_step_counter = (int(iteration) + 1) * steps_per_env
        self._restore_curriculum_state(
            {"common_step_counter": common_step_counter}, derived=True
        )

    def _restore_curriculum_state(
        self, state: dict[str, object], *, derived: bool = False
    ):
        """Restore and immediately materialize all persisted curricula."""
        from rl_sim_env.tasks.manager_based.common.mdp.curriculums import (
            apply_ee_external_force_curriculum_level,
            apply_velocity_command_curriculum_ranges,
            derive_ee_external_force_curriculum_level,
            derive_velocity_curriculum_level,
        )

        env = self.env.unwrapped
        common_step_counter = int(state.get("common_step_counter", 0))
        env.common_step_counter = common_step_counter
        restored: dict[str, int | float] = {
            "common_step_counter": common_step_counter
        }

        try:
            base_cmd = env.command_manager.get_term("base_command")
            command_curriculum_hooks = self._get_command_curriculum_hooks(
                base_cmd
            )
            has_velocity_curriculum = False
            for attr, max_attr in (
                ("lin_x_level", "max_lin_x_level"),
                ("ang_z_level", "max_ang_z_level"),
            ):
                if not hasattr(base_cmd.cfg, attr) or not hasattr(
                    base_cmd.cfg, max_attr
                ):
                    continue
                has_velocity_curriculum = True
                if attr in state:
                    level = float(state[attr])
                else:
                    level = derive_velocity_curriculum_level(
                        env,
                        max_level=float(getattr(base_cmd.cfg, max_attr)),
                        step_counter=common_step_counter,
                    )
                setattr(base_cmd.cfg, attr, level)
                restored[attr] = level

            if command_curriculum_hooks is not None:
                command_state = state.get("base_command")
                if command_state is None:
                    # Legacy checkpoints only contain the runner-owned scalar
                    # state.  Give command-owned curricula those restored levels
                    # and the global step so they can initialize any newer local
                    # statistics without discarding the old schedule position.
                    command_state = {
                        "common_step_counter": common_step_counter,
                        # cuVAETest inserted a stationary level zero ahead of the
                        # legacy moving schedule.  Opt-in loaders use this
                        # marker to migrate old level 0..5 to new level 1..6.
                        "legacy_time_driven_levels": True,
                        **{
                            attr: restored[attr]
                            for attr in ("lin_x_level", "ang_z_level")
                            if attr in restored
                        },
                    }
                elif not isinstance(command_state, dict):
                    raise TypeError(
                        "curriculum_state['base_command'] must be a dict, "
                        f"got {type(command_state).__name__}."
                    )
                command_curriculum_hooks[1](command_state)
                command_curriculum_hooks[2]()
                for attr in ("lin_x_level", "ang_z_level"):
                    if hasattr(base_cmd.cfg, attr):
                        restored[attr] = float(getattr(base_cmd.cfg, attr))
            elif has_velocity_curriculum:
                apply_velocity_command_curriculum_ranges(env)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            print(f"[WARN] Failed to restore velocity curriculum: {exc}")

        try:
            force_level = state.get("ee_force_curriculum_level")
            if force_level is None:
                force_level = derive_ee_external_force_curriculum_level(
                    env, step_counter=common_step_counter
                )
            force_level = apply_ee_external_force_curriculum_level(
                env, float(force_level)
            )
            if force_level > 0.0 or hasattr(
                env, "ee_force_curriculum_level"
            ):
                restored["ee_force_curriculum_level"] = force_level
                restored["ee_force_curriculum_current_max"] = float(
                    getattr(env, "ee_force_curriculum_current_max", 0.0)
                )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            print(f"[WARN] Failed to restore EE force curriculum: {exc}")

        source = "derived from legacy iter" if derived else "checkpoint"
        print(f"[INFO] Restored curriculum state ({source}): {restored}")

    def _restore_annealed_vae_from_updates(self, completed_updates: int):
        """resume 时按已完成 update 数重推 VAE 退火标量，避免从初始值重新退火。

        - ``vae_beta``（KL 权重）：从 ``beta`` 线性升到 ``beta_max``，全程
          ``beta_max_step`` 次 update；每次 update 加一个 ``vae_beta_step``。
        - ``p_boot``（adaboot 重构概率）：从 1.0 线性降到 0.0，全程
          ``adaboot_max_step`` 次 update。

        二者都是 update 次数的确定函数，存在算法实例上、不进 checkpoint，
        所以 resume 后回到初始值；这里一次性推到正确位置。
        """
        if not self.use_vae:
            return
        alg = self.alg
        updates = int(completed_updates)
        # __init__ 里 self.vae_beta 已经是初值，vae_beta_step 也已算好，
        # 直接加上 it 步即可（封顶到 beta_max）。
        if getattr(alg, "vae_beta_adaptive", False):
            alg.vae_beta = min(float(alg.vae_beta_max), float(alg.vae_beta) + float(alg.vae_beta_step) * updates)
        # p_boot 同理：初值 1.0，每次 update 减 p_boot_step，下限 0.0。
        if getattr(alg, "vae_use_adaboot", False):
            alg.p_boot = max(0.0, float(alg.p_boot) - float(alg.p_boot_step) * updates)
        print(f"[INFO] Derived VAE annealed scalars after updates={updates}: "
              f"vae_beta={getattr(alg, 'vae_beta', None)}, p_boot={getattr(alg, 'p_boot', None)}")

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        extra_dict = {}
        if device is not None:
            self.alg.actor_critic.to(device)
            if self.use_vae:
                self.alg.vae.to(device)
        policy = self.alg.actor_critic
        if self.use_vae:
            extra_dict["vae"] = self.alg.vae
        return policy, extra_dict

    def train_mode(self):
        # -- PPO
        self.alg.actor_critic.train()
        # -- AMP
        if self.use_amp:
            self.alg.amp_discriminator.train()
        # -- VAE
        if self.use_vae:
            self.alg.vae.train()

    def eval_mode(self):
        # -- PPO
        self.alg.actor_critic.eval()
        # -- AMP
        if self.use_amp:
            self.alg.amp_discriminator.eval()
        # -- VAE
        if self.use_vae:
            self.alg.vae.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    """
    Helper functions.
    """

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        # check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # if not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # get rank and world size
        self.gpu_local_rank_offset = int(os.getenv("JAX_LOCAL_RANK", "0"))
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0")) + self.gpu_local_rank_offset
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # rank of the main process
            "local_rank": self.gpu_local_rank,  # rank of the current process
            "world_size": self.gpu_world_size,  # total number of processes
        }

        # check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # validate multi-gpu configuration
        if (self.gpu_local_rank - self.gpu_local_rank_offset) >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)

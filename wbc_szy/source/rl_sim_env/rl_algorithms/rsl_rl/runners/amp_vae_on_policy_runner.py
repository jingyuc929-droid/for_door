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
from rl_algorithms.rsl_rl.algorithms import AMPVAEPPO
from rl_algorithms.rsl_rl.env import VecEnv
from rl_algorithms.rsl_rl.modules import VAE, ActorCritic, AMPDiscriminator
from rl_algorithms.rsl_rl.utils import store_code_state


class AMPVAEOnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cuda:0"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # get number of observations
        config_summary_env = self.env.cfg.config_summary.env
        num_actor_obs = config_summary_env.num_actor_obs
        num_critic_obs = config_summary_env.num_critic_obs
        num_amp_obs = config_summary_env.num_amp_obs
        num_vae_obs = config_summary_env.num_vae_obs
        obs_history_length = config_summary_env.obs_history_length
        cenet_in_dim = num_vae_obs * obs_history_length
        cenet_out_dim = config_summary_env.num_vae_out
        num_next_obs = 45
        num_actions = self.env.num_actions

        # evaluate the policy class
        # ActorCritic
        print("body_names", self.env.unwrapped.scene["robot"].data.body_names)
        dof_range = (
            self.env.unwrapped.scene["robot"].data.default_joint_pos_limits[0][:, 1]
            - self.env.unwrapped.scene["robot"].data.default_joint_pos_limits[0][:, 0]
        )
        min_std = torch.tensor(self.policy_cfg["min_normalized_std"], device=self.device) * (torch.abs(dof_range))
        actor_critic = ActorCritic(
            num_actor_obs=num_actor_obs + cenet_out_dim,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            min_std=min_std,
            **self.policy_cfg,
        ).to(self.device)

        # AMP
        config_summary_amp = self.env.cfg.config_summary.amp
        amp_data = AMPLoader(
            device,
            time_between_frames=self.env.step_dt,
            num_preload_transitions=config_summary_amp.num_preload_transitions,
            motion_files=config_summary_amp.motion_files,
        )
        amp_normalizer = Normalizer(amp_data.observation_dim)
        amp_discriminator: AMPDiscriminator = AMPDiscriminator(
            amp_data.observation_dim * 2,
            config_summary_amp.discr_hidden_dims,
            device,
        ).to(self.device)

        # VAE
        vae: VAE = VAE(cenet_in_dim, cenet_out_dim).to(self.device)

        # initialize algorithm
        self.alg = AMPVAEPPO(
            actor_critic,
            amp_discriminator,
            amp_data,
            amp_normalizer,
            vae,
            device=self.device,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
        self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_actor_obs],
            [num_actor_obs],
            [num_critic_obs],
            [num_amp_obs],
            [cenet_in_dim],
            [num_next_obs],
            [num_actions],
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
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "swanlab":
                from rsl_rl.utils.swanlab_utils import SwanLabSummaryWriter

                self.writer = SwanLabSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb', 'swanlab' or 'tensorboard'.")

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        self.init_logger()

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs_dict = self.env.get_observations()
        actor_obs = obs_dict["actor_obs"]
        critic_obs = obs_dict["critic_obs"]
        amp_obs = obs_dict["amp_obs"]
        next_amp_obs = amp_obs.clone()
        vae_obs = obs_dict["vae_obs"]
        actor_obs, critic_obs, amp_obs, vae_obs = (
            actor_obs.to(self.device),
            critic_obs.to(self.device),
            amp_obs.to(self.device),
            vae_obs.to(self.device),
        )
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            # TODO: Do we need to synchronize empirical normalizers?
            #   Right now: No, because they all should converge to the same values "asymptotically".

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        adaboot_p = 0.0
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(actor_obs, critic_obs, amp_obs, vae_obs)
                    amp_out = self.alg.amp_discriminator.discriminator_out(
                        amp_obs, next_amp_obs, normalizer=self.alg.amp_normalizer
                    )
                    amp_obs = torch.clone(next_amp_obs)
                    # Step the environment
                    (
                        obs_buf,
                        rewards,
                        dones,
                        infos,
                        reset_env_ids,
                        terminal_amp_states,
                        episode_reward,
                    ) = self.env.step(actions.to(self.device), amp_out.to(self.device))
                    actor_obs = obs_buf["actor_obs"]
                    critic_obs = obs_buf["critic_obs"]
                    next_amp_obs = obs_buf["amp_obs"]
                    vae_obs = obs_buf["vae_obs"]
                    # print("critic_obs", critic_obs)
                    # print("actor_obs", actor_obs)
                    # print("amp_obs", amp_obs)
                    # print("vae_obs", vae_obs)
                    # Move to device
                    actor_obs, critic_obs, next_amp_obs, vae_obs, rewards, dones = (
                        actor_obs.to(self.device),
                        critic_obs.to(self.device),
                        next_amp_obs.to(self.device),
                        vae_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    # perform normalization
                    actor_obs = self.obs_normalizer(actor_obs)
                    critic_obs = self.privileged_obs_normalizer(critic_obs)

                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    next_actor_obs = torch.clone(critic_obs.detach()[:, 3:48])

                    # process the step
                    self.alg.process_env_step(
                        rewards,
                        dones,
                        infos,
                        next_amp_obs_with_term,
                        next_actor_obs,
                        episode_reward,
                    )

                    # book keeping
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        # Update rewards
                        cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns
                adaboot_p = self.alg.compute_returns(critic_obs)

            # update policy
            loss_dict = self.alg.update()

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
                if git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

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
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.actor_critic.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))
        # -- Adaboot
        self.writer.add_scalar("Adaboot/adaboot_p", locs["adaboot_p"], locs["it"])
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
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "discriminator_state_dict": self.alg.amp_discriminator.state_dict(),
            "amp_normalizer": self.alg.amp_normalizer,
            "vae_state_dict": self.alg.vae.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }

        # save model
        torch.save(saved_dict, path)

        # # upload model to external logging service
        # if not self.disable_logs:
        #     self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=False)
        # -- Load model
        resumed_training = self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.amp_discriminator.load_state_dict(loaded_dict["discriminator_state_dict"], strict=True)
        self.alg.vae.load_state_dict(loaded_dict["vae_state_dict"], strict=True)
        self.alg.amp_normalizer = loaded_dict["amp_normalizer"]
        # -- load optimizer if used
        if load_optimizer and resumed_training:
            # -- algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        # -- load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
            self.alg.vae.to(device)
        policy = self.alg.actor_critic
        vae = self.alg.vae
        return policy, vae

    def train_mode(self):
        # -- PPO
        self.alg.actor_critic.train()
        # -- AMP
        self.alg.amp_discriminator.train()
        # -- VAE
        self.alg.vae.train()

    def eval_mode(self):
        # -- PPO
        self.alg.actor_critic.eval()
        # -- AMP
        self.alg.amp_discriminator.eval()
        # -- VAE
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

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import (
    CommandManager,
    CurriculumManager,
    RewardManager,
    TerminationManager,
    SceneEntityCfg,
)
from rl_sim_env.envs.terrain_aware_reward_manager import TerrainAwareRewardManager
from isaaclab.ui.widgets import ManagerLiveVisualizer
from isaaclab.utils import CircularBuffer
from isaacsim.core.version import get_version
from rl_sim_env.tasks.manager_based.locomotion.locomotion_base_env_cfg import LocomotionEnvCfg
import rl_sim_env.tasks.manager_based.common.mdp as mdp
from tensordict import TensorDict
from prettytable import PrettyTable


class LocomotionRLEnv(ManagerBasedEnv, gym.Env):
    """The superclass for the manager-based workflow reinforcement learning-based environments.

    This class inherits from :class:`ManagerBasedEnv` and implements the core functionality for
    reinforcement learning-based environments. It is designed to be used with any RL
    library. The class is designed to be used with vectorized environments, i.e., the
    environment is expected to be run in parallel with multiple sub-environments. The
    number of sub-environments is specified using the ``num_envs``.

    Each observation from the environment is a batch of observations for each sub-
    environments. The method :meth:`step` is also expected to receive a batch of actions
    for each sub-environment.

    While the environment itself is implemented as a vectorized environment, we do not
    inherit from :class:`gym.vector.VectorEnv`. This is mainly because the class adds
    various methods (for wait and asynchronous updates) which are not required.
    Additionally, each RL library typically has its own definition for a vectorized
    environment. Thus, to reduce complexity, we directly use the :class:`gym.Env` over
    here and leave it up to library-defined wrappers to take care of wrapping this
    environment for their agents.

    Note:
        For vectorized environments, it is recommended to **only** call the :meth:`reset`
        method once before the first call to :meth:`step`, i.e. after the environment is created.
        After that, the :meth:`step` function handles the reset of terminated sub-environments.
        This is because the simulator does not support resetting individual sub-environments
        in a vectorized environment.

    """

    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: LocomotionEnvCfg
    """Configuration for the environment."""

    def __init__(self, cfg: LocomotionEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.

        Args:
            cfg: The configuration for the environment.
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.
        """
        # -- counter for curriculum
        self.common_step_counter = 0

        # initialize the base class to setup the scene.
        super().__init__(cfg=cfg)
        # store the render mode
        self.render_mode = render_mode

        # initialize data and constants
        self.action_history_length = self.cfg.config_summary.env.action_history_length
        self.clip_obs = self.cfg.config_summary.env.clip_obs
        self.only_positive_reward = self.cfg.config_summary.reward.only_positive_reward

        # 初始化地形类型int，形状为 (num_envs, 1)
        self.terrain_types = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        if self.scene.terrain is not None and hasattr(self.scene.terrain, "terrain_types"):
            # 获取地形类型索引 (0-19)
            terrain_types_all = self.scene.terrain.terrain_types
            # 按照地形的比例
            num_cols = self.cfg.scene.terrain.terrain_generator.num_cols
            proportion_list = {}
            proportion_down_limit: int = 0
            proportion_up_limit: int = 0
            index = 0
            for key, value in self.cfg.scene.terrain.terrain_generator.sub_terrains.items():
                proportion_up_limit += value.proportion * num_cols
                proportion_list[index] = (proportion_down_limit, proportion_up_limit)
                proportion_down_limit = proportion_up_limit
                index += 1

            for env_id in range(self.num_envs):
                for key, (low, high) in proportion_list.items():
                    if low <= terrain_types_all[env_id] < high:
                        self.terrain_types[env_id] = key
                        break
            print("terrain_types", self.terrain_types)

            # 重要：在 terrain_types 设置好之后，重新设置地形相关的奖励权重
            # 因为 RewardManager 初始化时 terrain_types 还不存在
            if hasattr(self.reward_manager, '_setup_terrain_weights'):
                print("[INFO] 重新设置地形相关奖励权重（terrain_types 现在已可用）")
                self.reward_manager._setup_terrain_weights()

        to_drop = {
            "concatenate_terms",
            "concatenate_dim",
            "enable_corruption",
            "history_length",
            "flatten_history_dim",
        }
        term_dim_dict = {}
        self.delay_dict = {}
        self.obs_buffer_dict = {}
        for key, value in self.cfg.observations.__dict__.items():
            group_buffer = []
            for term_key, term_value in value.__dict__.items():
                if term_key in to_drop or term_value is None:
                    continue
                group_buffer.append(term_key)
                if 'params' in term_value.__dict__:
                    # 遍历所有参数，解析所有的 SceneEntityCfg
                    for param_name, param_value in term_value.params.items():
                        if isinstance(param_value, SceneEntityCfg):
                            param_value.resolve(self.scene)
                if term_key not in term_dim_dict:
                    term_dim_dict[term_key] = term_value.func(self, **term_value.params).shape[-1]
                else:
                    assert False, 'Repetitive term: ' + term_key
            if key != 'amp_obs':
                self.obs_buffer_dict[key] = group_buffer

        obs_term_dict = self.cfg.config_summary.observation.obs_term_dict
        for key, value in obs_term_dict.items():
            for term_key, term_value in value.items():
                if 'delay' in term_value:
                    self.delay_dict[term_key] = term_value['delay']

        self.policy_obs_dict = self.cfg.config_summary.observation.policy_obs_dict
        self.extra_obs_dict = self.cfg.config_summary.observation.extra_obs_dict
        obs_groups_dim_dict = {}
        self.history_dict = {}

        for key, value in self.policy_obs_dict.items():
            if 'history_length' not in value:
                history_length = 1
            else:
                history_length = value['history_length']
            if history_length > 1:
                self.history_dict[key] = history_length
            obs_groups_dim_dict[key] = sum(term_dim_dict[term] for term in value['terms']) * history_length

        for key, value in self.extra_obs_dict.items():
            obs_groups_dim_dict[key] = value

        # -- init buffers
        self.obs_tensor_dict = TensorDict(
            {key: torch.zeros(self.num_envs, value, device=self.device, dtype=torch.float32) for key, value in obs_groups_dim_dict.items()},
            batch_size=[self.num_envs],
            device=self.device,
        )

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_reward_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        self.obs_delay_dict = {}
        self.obs_delay_sample_dict = {}
        for key, value in self.delay_dict.items():
            self.obs_delay_dict[key] = CircularBuffer(value + 1, self.num_envs, self.device)
            self.obs_delay_sample_dict[key] = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.obs_history_dict = {}
        for key, value in self.history_dict.items():
            self.obs_history_dict[key] = CircularBuffer(value, self.num_envs, self.device)

        self.actions_history = CircularBuffer(
            self.action_history_length, self.num_envs, self.device
        )
        # -- set the framerate of the gym video recorder wrapper so that the playback speed of the produced video matches the simulation
        self.metadata["render_fps"] = 1 / self.step_dt

        self.event_apply_forces_torques_buf = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32)
        self.event_push_vel_buf = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)

        print('-------------------------------------------------------LocomotionRLEnv DOUBLE CHECK INFO--------------------------------------------------------------------------------')

        debug_info = ""
        for group_name, group_dim in self.observation_manager._group_obs_dim.items():
            # create table for term information
            table = PrettyTable()
            table.title = f"Active Observation Terms in Group: '{group_name}'"
            if self.observation_manager._group_obs_concatenate[group_name]:
                table.title += f" (shape: {group_dim})"
            table.field_names = ["Index", "Name", "Shape", "Scale", "Clip", "Noise", "Delay"]
            # set alignment of table columns
            table.align["Name"] = "l"
            # add info for each term
            obs_terms = zip(
                self.observation_manager._group_obs_term_names[group_name],
                self.observation_manager._group_obs_term_dim[group_name],
                self.observation_manager._group_obs_term_cfgs[group_name],
            )
            for index, (name, dims, cfg) in enumerate(obs_terms):
                # resolve inputs to simplify prints
                tab_dims = tuple(dims)
                if cfg.scale is not None:
                    tab_scale = cfg.scale.tolist()
                else:
                    tab_scale = None
                if name in self.delay_dict:
                    tab_delay = self.delay_dict[name]
                else:
                    tab_delay = None
                # add row
                table.add_row([index, name, tab_dims, tab_scale, cfg.clip, cfg.noise, tab_delay])
                print(index, name)
                print(cfg.params)

            # convert table to string
            debug_info += table.get_string()
            debug_info += "\n"

        print(debug_info)

        print("[INFO]: Completed setting up the environment...")
        print("[INFO] DOF Pos Limits: ", self.unwrapped.scene)

    """
    Properties.
    """

    def _get_obs_group(self, obs_dict, group_name) -> torch.Tensor:

        return self.obs_tensor_dict[group_name]

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in environment steps."""
        return math.ceil(self.max_episode_length_s / self.step_dt)

    @property
    def remaining_episode_time(self) -> torch.Tensor:
        """Remaining episode time in seconds."""
        return self.max_episode_length_s - self.episode_length_buf * self.step_dt

    """
    Operations - Setup.
    """

    def load_managers(self):
        # note: this order is important since observation manager needs to know the command and action managers
        # and the reward manager needs to know the termination manager
        # CLI/config overrides may change scene.num_envs after the config class was constructed.
        # Refresh terrain command ID partitions immediately before the command manager validates them.
        refresh_num_envs_cfg = getattr(self.cfg, "refresh_num_envs_dependent_cfg", None)
        base_command_cfg = getattr(self.cfg.commands, "base_command", None)
        if callable(refresh_num_envs_cfg) and hasattr(base_command_cfg, "command_ids"):
            refresh_num_envs_cfg()

        # -- command manager
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # call the parent class to load the managers for observations and actions.
        super().load_managers()

        # prepare the managers
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        
        # self.reward_manager = RewardManager(self.cfg.rewards, self)
        # -- reward manager (使用支持地形相关权重的自定义管理器)
        self.reward_manager = TerrainAwareRewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        # ObservationManager probes terms before startup randomization to infer
        # shapes.  Discard any static privileged-observation probes so their
        # first runtime value reflects the randomized physics properties.
        if hasattr(self, "_static_observation_cache"):
            self._static_observation_cache.clear()

    def setup_manager_visualizers(self):
        """Creates live visualizers for manager terms."""

        self.manager_visualizers = {
            "action_manager": ManagerLiveVisualizer(manager=self.action_manager),
            "observation_manager": ManagerLiveVisualizer(manager=self.observation_manager),
            "command_manager": ManagerLiveVisualizer(manager=self.command_manager),
            "termination_manager": ManagerLiveVisualizer(manager=self.termination_manager),
            "reward_manager": ManagerLiveVisualizer(manager=self.reward_manager),
            "curriculum_manager": ManagerLiveVisualizer(manager=self.curriculum_manager),
        }

    """
    Operations - MDP
    """

    def update_amp_out(self, amp_out: torch.Tensor | None = None):
        # process amp output
        if amp_out is not None:
            self.amp_out = amp_out

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """

        # update action history
        self.actions_history.append(action)
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for decimation_idx in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # update external push wrench (smooth ramp) before writing to sim
            mdp.update_push_wrench_base(self, dt=self.physics_dt)
            # update EE external force (smooth ramp) before writing to sim
            mdp.update_ee_external_force(self, dt=self.physics_dt)
            # update independently named external-force channels
            mdp.update_external_force_channels(self, dt=self.physics_dt)
            # set actions into simulator
            self.scene.write_data_to_sim()
            # print(self.scene.articulations['robot'].actuators['base_legs'].positions_delay_buffer.time_lags)
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)
            # update delay buffers
            if decimation_idx < self.cfg.decimation - 1:
                noise_and_delay_obs = self.observation_manager.compute_group("noise_and_delay_obs")
                for key, value in self.delay_dict.items():
                    self.obs_delay_dict[key].append(noise_and_delay_obs[key])

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.done_buf = self.termination_manager.dones
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        if self.only_positive_reward:
            self.reward_buf[:] = torch.clip(self.reward_buf[:], min=0.0)
        self.episode_reward_buf += self.reward_buf

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            obs_dict = self.observation_manager.compute()
            if self.clip_obs is not None:
                for term in obs_dict.values():
                    term[:] = torch.clip(
                        term, min=-self.clip_obs, max=self.clip_obs
                    )
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            if "amp_obs" in self.observation_manager.active_terms:
                terminal_amp_states = self.observation_manager.compute_group("amp_obs")[reset_env_ids]
            else:
                terminal_amp_states = None
        else:
            terminal_amp_states = None
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        self.event_push_vel_buf.zero_()
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        obs_dict = self.observation_manager.compute()

        noise_and_delay_obs = obs_dict["noise_and_delay_obs"]
        for key, value in self.delay_dict.items():
            self.obs_delay_dict[key].append(noise_and_delay_obs[key])

        obs_term_buffer = {}
        for group_key, value in self.obs_buffer_dict.items():
            for term_key in value:
                obs_term_buffer[term_key] = obs_dict[group_key][term_key]

        for group_key, value in self.policy_obs_dict.items():
            if group_key == 'amp_obs':
                continue
            group_buffer = []
            for term_key in value['terms']:
                if term_key in self.delay_dict:
                    group_buffer.append(self.obs_delay_dict[term_key][self.obs_delay_sample_dict[term_key]])
                else:
                    group_buffer.append(obs_term_buffer[term_key])
            if group_key in self.obs_history_dict:
                self.obs_history_dict[group_key].append(torch.cat(group_buffer, dim=-1))
                self.obs_tensor_dict[group_key] = self.obs_history_dict[group_key].buffer.reshape(self.num_envs, -1)
            else:
                self.obs_tensor_dict[group_key] = torch.cat(group_buffer, dim=-1)
            # print(group_key, self.obs_tensor_dict[group_key].shape)

        if 'amp_obs' in obs_dict:
            self.obs_tensor_dict['amp_obs'] = obs_dict['amp_obs']

        if self.clip_obs is not None:
            for term in self.obs_tensor_dict.values():
                term.clamp_(-self.clip_obs, self.clip_obs)

        # return observations, rewards, resets and extras
        return (
            self.obs_tensor_dict,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
            reset_env_ids,
            terminal_amp_states,
            self.episode_reward_buf
        )

    def render(self, recompute: bool = False) -> np.ndarray | None:
        """Run rendering without stepping through the physics.

        By convention, if mode is:

        - **human**: Render to the current display and return nothing. Usually for human consumption.
        - **rgb_array**: Return an numpy.ndarray with shape (x, y, 3), representing RGB values for an
          x-by-y pixel image, suitable for turning into a video.

        Args:
            recompute: Whether to force a render even if the simulator has already rendered the scene.
                Defaults to False.

        Returns:
            The rendered image as a numpy array if mode is "rgb_array". Otherwise, returns None.

        Raises:
            RuntimeError: If mode is set to "rgb_data" and simulation render mode does not support it.
                In this case, the simulation render mode must be set to ``RenderMode.PARTIAL_RENDERING``
                or ``RenderMode.FULL_RENDERING``.
            NotImplementedError: If an unsupported rendering mode is specified.
        """
        # run a rendering step of the simulator
        # if we have rtx sensors, we do not need to render again sin
        if not self.sim.has_rtx_sensors() and not recompute:
            self.sim.render()
        # decide the rendering mode
        if self.render_mode == "human" or self.render_mode is None:
            return None
        elif self.render_mode == "rgb_array":
            # check that if any render could have happened
            if self.sim.render_mode.value < self.sim.RenderMode.PARTIAL_RENDERING.value:
                raise RuntimeError(
                    f"Cannot render '{self.render_mode}' when the simulation render mode is"
                    f" '{self.sim.render_mode.name}'. Please set the simulation render mode to:"
                    f"'{self.sim.RenderMode.PARTIAL_RENDERING.name}' or '{self.sim.RenderMode.FULL_RENDERING.name}'."
                    " If running headless, make sure --enable_cameras is set."
                )
            # create the annotator if it does not exist
            if not hasattr(self, "_rgb_annotator"):
                import omni.replicator.core as rep

                # create render product
                self._render_product = rep.create.render_product(
                    self.cfg.viewer.cam_prim_path, self.cfg.viewer.resolution
                )
                # create rgb annotator -- used to read data from the render product
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])
            # obtain the rgb data
            rgb_data = self._rgb_annotator.get_data()
            # convert to numpy array
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # return the rgb data
            # note: initially the renerer is warming up and returns empty data
            if rgb_data.size == 0:
                return np.zeros((self.cfg.viewer.resolution[1], self.cfg.viewer.resolution[0], 3), dtype=np.uint8)
            else:
                return rgb_data[:, :, :3]
        else:
            raise NotImplementedError(
                f"Render mode '{self.render_mode}' is not supported. Please use: {self.metadata['render_modes']}."
            )

    def close(self):
        if not self._is_closed:
            # destructor is order-sensitive
            del self.command_manager
            del self.reward_manager
            del self.termination_manager
            del self.curriculum_manager
            # call the parent class to close the environment
            super().close()

    """
    Helper functions.
    """

    @staticmethod
    def _reset_command_manager_without_host_sync(manager, env_ids):
        """Reset commands while keeping metric reductions on the device."""
        if env_ids is None:
            env_ids = slice(None)
        extras = {}
        for name, term in manager._terms.items():
            hidden_metric_names = getattr(term, "_hidden_log_metrics", ())
            for metric_name, metric_value in term.metrics.items():
                if metric_name not in hidden_metric_names:
                    extras[f"Metrics/{name}/{metric_name}"] = torch.mean(
                        metric_value[env_ids]
                    )
                metric_value[env_ids] = 0.0
            term.command_counter[env_ids] = 0
            term._resample(env_ids)
        return extras

    @staticmethod
    def _reset_curriculum_manager_without_host_sync(manager, env_ids):
        """Snapshot curriculum logs without converting CUDA scalars to Python."""
        extras = {}
        for term_name, term_state in manager._curriculum_state.items():
            if term_state is None:
                continue
            if isinstance(term_state, dict):
                for key, value in term_state.items():
                    if isinstance(value, torch.Tensor):
                        value = value.clone()
                    extras[f"Curriculum/{term_name}/{key}"] = value
            else:
                if isinstance(term_state, torch.Tensor):
                    term_state = term_state.clone()
                extras[f"Curriculum/{term_name}"] = term_state
        for term_cfg in manager._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        return extras

    @staticmethod
    def _reset_termination_manager_without_host_sync(manager, env_ids):
        """Reset termination terms without per-term CUDA ``item()`` calls."""
        last_episode_done_stats = manager._term_dones.float().mean(dim=0)
        extras = {
            "Episode_Termination/" + key: last_episode_done_stats[i]
            for i, key in enumerate(manager._term_names)
        }
        for term_cfg in manager._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        return extras

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.single_observation_space = gym.spaces.Dict()
        for group_name, group_term_names in self.observation_manager.active_terms.items():
            # extract quantities about the group
            has_concatenated_obs = self.observation_manager.group_obs_concatenate[group_name]
            group_dim = self.observation_manager.group_obs_dim[group_name]
            # check if group is concatenated or not
            # if not concatenated, then we need to add each term separately as a dictionary
            if has_concatenated_obs:
                self.single_observation_space[group_name] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=group_dim)
            else:
                self.single_observation_space[group_name] = gym.spaces.Dict({
                    term_name: gym.spaces.Box(low=-np.inf, high=np.inf, shape=term_dim)
                    for term_name, term_dim in zip(group_term_names, group_dim)
                })
        # action space (unbounded since we don't impose any limits)
        action_dim = sum(self.action_manager.action_term_dim)
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        # update the curriculum for environments that need a reset
        self.curriculum_manager.compute(env_ids=env_ids)
        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)
        # stop any ongoing external pushes on reset envs (only if the push system was ever initialized)
        if hasattr(self, "event_push_force_state"):
            mdp.reset_push_force(self, env_ids)
        if hasattr(self, "event_push_yaw_torque_state"):
            mdp.reset_push_yaw_torque(self, env_ids)
        # stop any ongoing EE external-force ramp on reset envs
        if hasattr(self, "ee_force_ramp_state"):
            mdp.reset_ee_external_force(self, env_ids)
        # clear named force channels before reset events sample their first targets
        mdp.reset_external_force_channels(self, env_ids)
        # apply events such as randomizations for environments that need a reset
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        # -- observation manager
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- rewards manager
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- curriculum manager
        info = self._reset_curriculum_manager_without_host_sync(
            self.curriculum_manager, env_ids
        )
        self.extras["log"].update(info)
        # -- command manager
        info = self._reset_command_manager_without_host_sync(
            self.command_manager, env_ids
        )
        self.extras["log"].update(info)
        # -- event manager
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- termination manager
        info = self._reset_termination_manager_without_host_sync(
            self.termination_manager, env_ids
        )
        self.extras["log"].update(info)
        # -- recorder manager
        info = self.recorder_manager.reset(env_ids)
        self.extras["log"].update(info)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0
        self.episode_reward_buf[env_ids] = 0

        for key, value in self.delay_dict.items():
            self.obs_delay_sample_dict[key][env_ids] = torch.randint(low=0, high=value + 1, size=(len(env_ids),), dtype=torch.long, device=self.device)

        self.actions_history.reset(env_ids)
        for buf in self.obs_delay_dict.values():
            buf.reset(env_ids)
        for buf in self.obs_history_dict.values():
            buf.reset(env_ids)

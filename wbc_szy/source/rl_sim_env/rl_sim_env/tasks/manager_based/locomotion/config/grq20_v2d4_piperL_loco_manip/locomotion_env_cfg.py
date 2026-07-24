# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from rl_sim_env.tasks.manager_based.common.command.config import (
    create_uniform_velocity_height_pitch_command_cfg,
)
from rl_sim_env.tasks.manager_based.common.sensor.frame_transform_config import (
    create_body_frame_transform_cfg,
)
from rl_sim_env.tasks.manager_based.common.sensor.ray_caster_config import (
    BLIND_HEIGHT_SCANNER_CFG,
)
from rl_sim_env.tasks.manager_based.common.terrain.config import LOCOMOTION_ROUGH_ONLY_TERRAIN_CFG2d4
from rl_sim_env.tasks.manager_based.locomotion.locomotion_base_env_cfg import LocomotionEnvCfg

from .config_summary import (
    AMPDataCfg,
    ConfigSummary,
    ROBOT_BASE_LINK,
    ROBOT_CFG,
    ROBOT_FOOT_NAMES,
    ROBOT_JOINT_NAMES,
    ROBOT_THIGH_NAMES,
)


@configclass
class LocomotionPiperLLowerBodyEnvCfg(LocomotionEnvCfg):
    def __post_init__(self):
        # config summary
        self.config_summary = ConfigSummary
        self.amp_loader_cfg = AMPDataCfg()
        num_envs = self.config_summary.env.num_envs

        # general settings
        self.decimation = self.config_summary.general.decimation
        self.episode_length_s = self.config_summary.general.episode_length_s
        self.is_finite_horizon = self.config_summary.general.is_finite_horizon

        # scene settings
        self.scene.num_envs = num_envs
        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Only random rough terrain is used for this lower-body task; no stairs or slopes.
        self.scene.terrain = LOCOMOTION_ROUGH_ONLY_TERRAIN_CFG2d4

        # height scanner settings
        self.scene.height_scanner = BLIND_HEIGHT_SCANNER_CFG
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + ROBOT_BASE_LINK
        self.scene.height_scanner.update_period = self.decimation * self.config_summary.sim.dt

        # contact and frame transform settings
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=3,
            track_air_time=True,
        )
        self.scene.contact_forces.update_period = self.config_summary.sim.dt
        self.scene.frame_transform = create_body_frame_transform_cfg(ROBOT_BASE_LINK, ROBOT_FOOT_NAMES)

        # simulation settings
        self.sim.dt = self.config_summary.sim.dt
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # lower policy command: vx, vy, wz, trunk height, trunk pitch
        self.commands.base_command = create_uniform_velocity_height_pitch_command_cfg(
            rel_standing_envs=self.config_summary.command.standing_command_prob,
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=self.config_summary.command.heading_control_stiffness,
            lin_vel_x=self.config_summary.command.lin_vel_x,
            lin_vel_y=self.config_summary.command.lin_vel_y,
            ang_vel_z=self.config_summary.command.ang_vel_z,
            heading=self.config_summary.command.heading,
            base_height_cmd=self.config_summary.command.base_height_cmd,
            fixed_height_cmd=self.config_summary.command.fixed_height_cmd,
            probability_of_using_fixed_height_cmd=(
                self.config_summary.command.probability_of_using_fixed_height_cmd
            ),
            base_pitch_cmd=self.config_summary.command.base_pitch_cmd,
            fixed_pitch_cmd=self.config_summary.command.fixed_pitch_cmd,
            probability_of_using_fixed_pitch_cmd=(
                self.config_summary.command.probability_of_using_fixed_pitch_cmd
            ),
        )

        # action settings: policy only controls 12 leg joints
        self.actions.joint_pos.joint_names = ROBOT_JOINT_NAMES
        self.actions.joint_pos.scale = self.config_summary.action.scale

        # observations
        to_drop = {
            "concatenate_terms",
            "concatenate_dim",
            "enable_corruption",
            "history_length",
            "flatten_history_dim",
        }
        invalid_obs_group_keys = list(self.observations.__dict__.keys() - self.config_summary.observation.obs_term_dict.keys())
        for key in invalid_obs_group_keys:
            self.observations.__dict__[key] = None

        for group_key, group_value in self.config_summary.observation.obs_term_dict.items():
            invalid_obs_term_keys = list(self.observations.__dict__[group_key].__dict__.keys() - group_value.keys())
            invalid_obs_term_keys[:] = [x for x in invalid_obs_term_keys if x not in to_drop]
            for key in invalid_obs_term_keys:
                self.observations.__dict__[group_key].__dict__[key] = None

            for key, value in group_value.items():
                if "scale" in value:
                    self.observations.__dict__[group_key].__dict__[key].scale = value["scale"]
                if "noise" in value:
                    self.observations.__dict__[group_key].__dict__[key].noise = value["noise"]
                if "clip" in value:
                    self.observations.__dict__[group_key].__dict__[key].clip = value["clip"]
                if "params" in value:
                    for k, v in value["params"].items():
                        self.observations.__dict__[group_key].__dict__[key].params[k] = v

        # events
        invalid_events_keys = list(self.events.__dict__.keys() - self.config_summary.event.config_dict.keys())
        for key in invalid_events_keys:
            self.events.__dict__[key] = None
        for key, value in self.config_summary.event.config_dict.items():
            self.events.__dict__[key].mode = value["mode"]
            if "interval_range_s" in value:
                self.events.__dict__[key].interval_range_s = value["interval_range_s"]
            if "params" in value:
                for k, v in value["params"].items():
                    self.events.__dict__[key].params[k] = v

        # rewards. The config_summary reward whitelist is lower-body only; all EE/manip rewards are removed here.
        invalid_rewards_keys = list(self.rewards.__dict__.keys() - self.config_summary.reward.config_dict.keys())
        for key in invalid_rewards_keys:
            self.rewards.__dict__[key] = None
        for key, value in self.config_summary.reward.config_dict.items():
            weight = value["weight"]
            if isinstance(weight, list):
                self.rewards.__dict__[key].weight = 0.0
                self.rewards.__dict__[key].terrain_weights = weight
            else:
                self.rewards.__dict__[key].weight = weight

            if "params" in value:
                for k, v in value["params"].items():
                    self.rewards.__dict__[key].params[k] = v

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [ROBOT_BASE_LINK] + ROBOT_THIGH_NAMES
        self.terminations.bad_orientation.params["asset_cfg"].body_names = ROBOT_BASE_LINK

        # This task uses a fixed-range 5D lower-body command and rough-only terrain,
        # so the terrain/command curriculums from the generic locomotion base do not apply.
        self.curriculum.terrain_levels_vel = None
        self.curriculum.lin_vel_x_command_threshold = None
        self.curriculum.ang_vel_z_command_threshold = None
        self.curriculum.ee_external_force_threshold = None


@configclass
class LocomotionPiperLLowerBodyEnvCfg_PLAY(LocomotionPiperLLowerBodyEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.amp_obs.enable_corruption = False
        self.observations.ground_truth_obs.enable_corruption = False
        self.observations.noise_and_delay_obs.enable_corruption = False

        self.events.randomize_base_mass = None
        self.events.randomize_base_com = None
        self.events.randomize_physics_material = None
        self.events.randomize_joint_friction = None
        self.events.reset_actuator_gains = None
        self.events.reset_robot_joints = None
        self.events.push_robot = None
        self.events.reset_joint_offset = None
        self.events.randomize_passive_arm_targets = None
        self.rewards.amp_reward = None
        self.curriculum.ang_vel_z_command_threshold = None
        self.curriculum.lin_vel_x_command_threshold = None

        self.commands.base_command = create_uniform_velocity_height_pitch_command_cfg(
            rel_standing_envs=0.0,
            rel_heading_envs=1.0,
            heading_command=False,
            heading_control_stiffness=0.5,
            lin_vel_x=(0.8, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
            base_height_cmd=self.config_summary.command.base_height_cmd,
            fixed_height_cmd=self.config_summary.command.fixed_height_cmd,
            probability_of_using_fixed_height_cmd=(
                self.config_summary.command.probability_of_using_fixed_height_cmd
            ),
            base_pitch_cmd=(0.0, 0.0),
            fixed_pitch_cmd=0.0,
            probability_of_using_fixed_pitch_cmd=1.0,
        )


class LocomotionPiperLLowerBodyEnvCfg_ONNX_PLAY(LocomotionPiperLLowerBodyEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.terminations.bad_orientation = None
        self.terminations.time_out = None


@configclass
class LocomotionPiperLLowerBodyEnvCfg_DEBUG_PLAY(LocomotionPiperLLowerBodyEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.terminations.bad_orientation = None


class LocomotionPiperLLowerBodyEnvCfg_REPLAY_AMPDATA(LocomotionPiperLLowerBodyEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.rewards = None

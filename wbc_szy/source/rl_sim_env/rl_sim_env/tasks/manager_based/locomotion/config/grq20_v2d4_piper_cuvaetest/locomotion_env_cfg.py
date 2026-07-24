# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from rl_sim_env.tasks.manager_based.locomotion.locomotion_base_env_cfg import LocomotionEnvCfg
from .config_summary import (
    ROBOT_BASE_LINK,
    ROBOT_CALF_NAMES,
    ROBOT_CFG,
    ROBOT_EE_BODY_NAME,
    ROBOT_EE_TOOL_QUAT_OFFSET,
    ROBOT_FOOT_NAMES,
    ROBOT_THIGH_NAMES,
    ROBOT_WHOLE_BODY_JOINT_NAMES,
    ConfigSummary,
    AMPDataCfg,
)
from rl_sim_env.tasks.manager_based.common.command.config import (
    create_uniform_velocity_command_cfg,
)
from rl_sim_env.tasks.manager_based.common.manipulation.command.config import (
    create_projected_com_target_points_command_cfg,
)
from rl_sim_env.tasks.manager_based.common.sensor.frame_transform_config import (
    create_body_frame_transform_cfg,
)
from rl_sim_env.tasks.manager_based.common.sensor.ray_caster_config import (
    BLIND_HEIGHT_SCANNER_CFG,
    FOOT_SCANNER_CFG,
)

from rl_sim_env.tasks.manager_based.common.terrain.config import (
    LOCOMOTION_ROUGH_ONLY_TERRAIN_CFG2d4,
)
from .velocity_curriculum import (
    ang_vel_z_command_threshold as performance_ang_vel_z_command_threshold,
    create_performance_velocity_command_terrain_cfg,
    lin_vel_x_command_threshold as performance_lin_vel_x_command_threshold,
)


from isaaclab.sensors import ContactSensorCfg, patterns


def _terrain_command_ids_and_ranges(terrain_cfg, command_cfg, num_envs: int):
    command_ids = {}
    command_ranges = {}
    keys = list(terrain_cfg.terrain_generator.sub_terrains.keys())
    proportions = [float(terrain_cfg.terrain_generator.sub_terrains[key].proportion) for key in keys]
    total = sum(proportions)
    if total <= 0.0:
        raise ValueError("Terrain proportions must sum to a positive value.")

    expected_counts = [proportion / total * num_envs for proportion in proportions]
    counts = [int(count) for count in expected_counts]
    remaining = num_envs - sum(counts)
    order = sorted(range(len(keys)), key=lambda idx: expected_counts[idx] - counts[idx], reverse=True)
    for idx in order[:remaining]:
        counts[idx] += 1

    env_start = 0
    for key, count in zip(keys, counts):
        command_ids[key] = list(range(env_start, env_start + count))
        env_start += count
        command_ranges[key] = command_cfg.ranges[key]
    return command_ids, command_ranges


@configclass
class LocomotionWholeBodyVaeEnvCfg(LocomotionEnvCfg):
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
        # number of environments
        self.scene.num_envs = num_envs

        # robot settings
        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # terrain settings:
        # 仅保留 rough（random_rough），但铺成多块 tile，避免所有 env 重置到同一块 rough 上。
        # 具体生成器参数集中在 common/terrain/config.py
        self.scene.terrain = LOCOMOTION_ROUGH_ONLY_TERRAIN_CFG2d4

        # height scanner settings
        self.scene.height_scanner = BLIND_HEIGHT_SCANNER_CFG
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + ROBOT_BASE_LINK
        self.scene.height_scanner.update_period = self.decimation * self.config_summary.sim.dt

        # foot scanners settings (for foot_clearance / foot_scan style observations)
        # foot_clearance() only consumes the center ray, so keep one ray per foot.
        def make_center_foot_scanner_cfg():
            cfg = FOOT_SCANNER_CFG.copy()
            cfg.pattern_cfg = patterns.GridPatternCfg(resolution=0.05, size=(0.0, 0.0))
            return cfg

        self.scene.fl_foot_scanner = make_center_foot_scanner_cfg()
        self.scene.fl_foot_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + ROBOT_BASE_LINK
        self.scene.fl_foot_scanner.update_period = self.decimation * self.config_summary.sim.dt

        self.scene.fr_foot_scanner = make_center_foot_scanner_cfg()
        self.scene.fr_foot_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + ROBOT_BASE_LINK
        self.scene.fr_foot_scanner.update_period = self.decimation * self.config_summary.sim.dt

        self.scene.rl_foot_scanner = make_center_foot_scanner_cfg()
        self.scene.rl_foot_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + ROBOT_BASE_LINK
        self.scene.rl_foot_scanner.update_period = self.decimation * self.config_summary.sim.dt

        self.scene.rr_foot_scanner = make_center_foot_scanner_cfg()
        self.scene.rr_foot_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + ROBOT_BASE_LINK
        self.scene.rr_foot_scanner.update_period = self.decimation * self.config_summary.sim.dt
        # contact forces settings
        contact_sensor_body_names = [
            ROBOT_BASE_LINK,
            *ROBOT_FOOT_NAMES,
            *ROBOT_THIGH_NAMES,
            *ROBOT_CALF_NAMES,
        ]
        contact_sensor_body_regex = "(" + "|".join(contact_sensor_body_names) + ")"
        self.scene.contact_forces = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{contact_sensor_body_regex}",
            history_length=3,
            track_air_time=True,
        )
        self.scene.contact_forces.update_period = self.config_summary.sim.dt

        # frame transform settings
        self.scene.frame_transform = create_body_frame_transform_cfg(ROBOT_BASE_LINK, ROBOT_FOOT_NAMES)

        # simulation settings
        self.sim.dt = self.config_summary.sim.dt
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # command settings
        command_ids, command_ranges = _terrain_command_ids_and_ranges(
            self.scene.terrain, self.config_summary.command, num_envs
        )

        self.commands.base_command = create_performance_velocity_command_terrain_cfg(
            command_ids=command_ids,
            ranges=command_ranges,
            lin_x_level=self.config_summary.command.lin_x_level,
            ang_z_level=self.config_summary.command.ang_z_level,
            max_lin_x_level=self.config_summary.command.max_lin_x_level,
            max_ang_z_level=self.config_summary.command.max_ang_z_level,
            vel_curriculum_episode_mult=self.config_summary.command.vel_curriculum_episode_mult,
            heading_control_stiffness=self.config_summary.command.heading_control_stiffness,
            lin_x_level_ranges=self.config_summary.command.lin_x_level_ranges,
            ang_z_level_ranges=self.config_summary.command.ang_z_level_ranges,
            lin_y_active_range=self.config_summary.command.lin_y_active_range,
            lin_x_mae_thresholds=self.config_summary.command.lin_x_mae_thresholds,
            lin_y_mae_thresholds=self.config_summary.command.lin_y_mae_thresholds,
            ang_z_mae_thresholds=self.config_summary.command.ang_z_mae_thresholds,
            velocity_curriculum_min_level_steps=(
                self.config_summary.command.velocity_curriculum_min_level_steps
            ),
            velocity_curriculum_ema_alpha=(
                self.config_summary.command.velocity_curriculum_ema_alpha
            ),
            velocity_curriculum_min_survival_rate=(
                self.config_summary.command.velocity_curriculum_min_survival_rate
            ),
            velocity_curriculum_min_episode_length_ratio=(
                self.config_summary.command.velocity_curriculum_min_episode_length_ratio
            ),
            velocity_curriculum_min_completed_episodes=(
                self.config_summary.command.velocity_curriculum_min_completed_episodes
            ),
            velocity_curriculum_required_successes=(
                self.config_summary.command.velocity_curriculum_required_successes
            ),
            velocity_curriculum_evaluation_interval_steps=(
                self.config_summary.command.velocity_curriculum_evaluation_interval_steps
            ),
        )

        # cuVAETest owns its velocity progression.  Keep it independent from the
        # inherited terrain curriculum and from force-control's curriculum term.
        self.curriculum.terrain_levels_vel = None
        self.curriculum.lin_vel_x_command_threshold.func = (
            performance_lin_vel_x_command_threshold
        )
        self.curriculum.ang_vel_z_command_threshold.func = (
            performance_ang_vel_z_command_threshold
        )

        # ee target points command (9D) in projected COM yaw frame
        # 注意：末端刚体名称从 URDF/资产中确定；reward 侧会用到该名称，但 command 本身不依赖末端刚体。
        self.commands.ee_target_points = create_projected_com_target_points_command_cfg(
            asset_name="robot",
            trunk_body_name=ROBOT_BASE_LINK,
            terrain_sensor_name="height_scanner",
            ee_body_name=ROBOT_EE_BODY_NAME,
            ee_tool_quat_offset=ROBOT_EE_TOOL_QUAT_OFFSET,
            pos_r_range=self.config_summary.command.ee_target_pos_r_range,
            pos_theta_range=self.config_summary.command.ee_target_pos_theta_range,
            pos_z_range=self.config_summary.command.ee_target_pos_z_range,
            roll_range=self.config_summary.command.ee_target_roll_range,
            pitch_range=self.config_summary.command.ee_target_pitch_range,
            yaw_range=self.config_summary.command.ee_target_yaw_range,
            offset_distance=self.config_summary.command.ee_target_offset_distance,
            resampling_time_range=self.config_summary.command.ee_target_resampling_time_range,
            ramp_time_s=self.config_summary.command.ee_target_ramp_time_s,
            debug_vis=self.config_summary.command.ee_target_debug_vis,
            target_marker_radius=self.config_summary.command.ee_target_marker_radius,
            ee_pos_marker_radius=self.config_summary.command.ee_pos_marker_radius,
            use_episode_mae_metrics=True,
        )
        # self.commands.base_command = create_uniform_velocity_command_cfg(
        #     rel_standing_envs=0.0,
        #     rel_heading_envs=1.0,
        #     heading_command=True,
        #     heading_control_stiffness=0.5,
        #     lin_vel_x=(-1.0, 1.0),
        #     lin_vel_y=(-0.5, 0.5),
        #     ang_vel_z=(-1.5, 1.5),
        #     heading=(-1.57, 1.57),
        # )

        # scale
        self.actions.joint_pos.scale = self.config_summary.action.scale
        # 全身控制：动作关节顺序显式指定（腿12 + 臂6）
        self.actions.joint_pos.joint_names = ROBOT_WHOLE_BODY_JOINT_NAMES

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

        # event
        invalid_events_keys = list(self.events.__dict__.keys() - self.config_summary.event.config_dict.keys())
        for key in invalid_events_keys:
            self.events.__dict__[key] = None
        for key, value in self.config_summary.event.config_dict.items():

            self.events.__dict__[key].mode = value["mode"]
            # interval events: allow overriding trigger interval from config
            if "interval_range_s" in value:
                self.events.__dict__[key].interval_range_s = value["interval_range_s"]
            if "params" in value:
                for k, v in value["params"].items():
                    self.events.__dict__[key].params[k] = v

        # rewards
        invalid_rewards_keys = list(self.rewards.__dict__.keys() - self.config_summary.reward.config_dict.keys())
        for key in invalid_rewards_keys:
            self.rewards.__dict__[key] = None
        for key, value in self.config_summary.reward.config_dict.items():
            # 支持列表形式的权重（按地形类型）
            weight = value["weight"]
            if isinstance(weight, list):
                # 如果是列表形式，表示按地形类型配置不同权重
                # 将 weight 设为 1.0 作为占位符（实际权重由 TerrainAwareRewardManager 处理）
                self.rewards.__dict__[key].weight = 0.0
                # 存储完整的权重列表供 TerrainAwareRewardManager 使用
                self.rewards.__dict__[key].terrain_weights = weight
            else:
                # 标量形式：所有地形使用相同权重
                self.rewards.__dict__[key].weight = weight

            # 设置其他参数
            if "params" in value:
                for k, v in value["params"].items():
                    self.rewards.__dict__[key].params[k] = v

            # 与 command 配置对齐：避免在 config_summary 中重复写 offset_distance
            # - 旧版：track_ee_target_points_exp
            # - 新版：拆分为 track_ee_target_x_offset_exp / track_ee_target_y_offset_exp
            if hasattr(self.commands, "ee_target_points") and (
                "offset_distance" in self.rewards.__dict__[key].params
            ):
                if self.rewards.__dict__[key].params["offset_distance"] is None:
                    self.rewards.__dict__[key].params["offset_distance"] = (
                        self.commands.ee_target_points.offset_distance
                    )

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ROBOT_BASE_LINK
        self.terminations.bad_orientation.params["asset_cfg"].body_names = None

    def refresh_num_envs_dependent_cfg(self):
        if self.scene.terrain.terrain_generator is None:
            return
        command_ids, command_ranges = _terrain_command_ids_and_ranges(
            self.scene.terrain, self.config_summary.command, int(self.scene.num_envs)
        )
        self.commands.base_command.command_ids = command_ids
        self.commands.base_command.ranges = command_ranges


@configclass
class LocomotionWholeBodyVaeEnvCfg_PLAY(LocomotionWholeBodyVaeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.amp_obs.enable_corruption = False
        self.observations.ground_truth_obs.enable_corruption = False
        self.observations.noise_and_delay_obs.enable_corruption = False
        # remove random pushing event
        self.events.randomize_base_mass = None
        self.events.randomize_base_com = None
        self.events.randomize_physics_material = None
        self.events.randomize_joint_friction = None
        self.events.reset_actuator_gains = None
        self.events.reset_robot_joints = None
        self.events.push_robot = None
        self.events.reset_joint_offset = None
        self.rewards.amp_reward = None
        self.curriculum.ang_vel_z_command_threshold = None
        self.curriculum.lin_vel_x_command_threshold = None

        self.commands.base_command = create_uniform_velocity_command_cfg(
            rel_standing_envs=1.0,
            rel_heading_envs=0.0,
            heading_command=False,
            heading_control_stiffness=0.5,
            lin_vel_x=(0.8, 1),
            lin_vel_y=(-0.0, 0.0),
            ang_vel_z=(0.0, 0),
            heading=(0.0, 0.0),
        )

        self.commands.ee_target_points = create_projected_com_target_points_command_cfg(
            asset_name="robot",
            trunk_body_name=ROBOT_BASE_LINK,
            terrain_sensor_name="height_scanner",
            ee_body_name=ROBOT_EE_BODY_NAME,
            ee_tool_quat_offset=ROBOT_EE_TOOL_QUAT_OFFSET,
            # ---- 手动指定的末端目标（投影 COM yaw 系，极坐标）----
            pos_r_range=(0.3, 0.75),       # main 点半径 (m)
            pos_theta_range=(-1, 1),   # main 点方位角 (rad)，0 = 正前方
            pos_z_range=(0.1, 0.9),       # main 点高度 (m)

            #pos_r_range=(0.65, 0.65),       # main 点半径 (m)
            #pos_theta_range=(0.0, 0.0),   # main 点方位角 (rad)，0 = 正前方
            #pos_z_range=(0.65, 0.65),       # main 点高度 (m)

            # ---- 手动指定的末端姿态 (rad) ----

            #roll_range=(-0, 0),
            #pitch_range=(-0, 0),
            #yaw_range=(-0, 0),

            roll_range=(-1, 1),
            pitch_range=(-1, 1),
            yaw_range=(-0.5, 0.5),
            # ---- 其余参数与训练保持一致 ----
            offset_distance=self.config_summary.command.ee_target_offset_distance,
            # ---- PLAY 专用：更快更新手臂 cmd，便于观察效果（不影响 train）----
            # 训练仍用 config_summary.command.ee_target_resampling_time_range = (12, 12)
            resampling_time_range=(4.0, 4.0),
            # ramp 占比不宜过大，频率提高时同步调小（0 = 闪现到新目标）
            ramp_time_s=0.5,
            debug_vis=self.config_summary.command.ee_target_debug_vis,
            target_marker_radius=self.config_summary.command.ee_target_marker_radius,
            ee_pos_marker_radius=self.config_summary.command.ee_pos_marker_radius,
            use_episode_mae_metrics=True,
        )

class LocomotionWholeBodyVaeEnvCfg_ONNX_PLAY(LocomotionWholeBodyVaeEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.terminations.bad_orientation = None
        self.terminations.time_out = None
        # self.observations.lidar_obs.enable_corruption = True


@configclass
class LocomotionWholeBodyVaeEnvCfg_DEBUG_PLAY(LocomotionWholeBodyVaeEnvCfg_PLAY):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.terminations.bad_orientation = None
        # self.observations.lidar_obs.enable_corruption = True


class LocomotionWholeBodyVaeEnvCfg_REPLAY_AMPDATA(LocomotionWholeBodyVaeEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.rewards = None
        # no terrain curriculum
        self.curriculum.terrain_levels_vel = None

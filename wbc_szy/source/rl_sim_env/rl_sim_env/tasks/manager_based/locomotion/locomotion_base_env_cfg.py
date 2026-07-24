# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import rl_sim_env.tasks.manager_based.common.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, RayCasterCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # robots
    robot: ArticulationCfg = None

    # ground terrain
    terrain: TerrainImporterCfg = None

    # sensors
    height_scanner: RayCasterCfg = None
    contact_forces: ContactSensorCfg = None

    # foot scanners
    fl_foot_scanner: RayCasterCfg = None
    fr_foot_scanner: RayCasterCfg = None
    rl_foot_scanner: RayCasterCfg = None
    rr_foot_scanner: RayCasterCfg = None

    # frame transform
    frame_transform: FrameTransformerCfg = None

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_command: (
        mdp.UniformPose2dCommandCfg
        | mdp.UniformVelocityCommandCfg
        | mdp.UniformVelocityCommandTerrainCfg
        | mdp.UniformVelocityAndHeightCommandCfg
        | mdp.UniformVelocityAndHeightCommandTerrainCfg
        | mdp.UniformVelocityAndOrientationFlagCommandCfg
    ) = MISSING


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)
    joint_pos = mdp.JointPositionOffsetActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class GroundTruthObsCfg(ObsGroup):
        """Observations for ground truth."""

        # ground truth observation terms (order preserved)
        base_lin_vel_gt = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel_gt = ObsTerm(func=mdp.base_ang_vel)
        base_height_b_gt = ObsTerm(
            func=mdp.base_height_b,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "foot_sensor_cfg": SceneEntityCfg("frame_transform"),
                "contact_sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=".*_foot"
                ),
            },
        )
        projected_gravity_gt = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel_gt = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel_gt = ObsTerm(func=mdp.joint_vel_rel)
        actions_gt = ObsTerm(func=mdp.last_action)
        base_commands_gt = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_command"}
        )
        ee_target_points_gt = ObsTerm(
            func=mdp.command_9d, params={"command_name": "ee_target_points"}
        )
        ee_target_points_error_gt = ObsTerm(
            func=mdp.ee_target_points_error_9d,
            params={
                "command_name": "ee_target_points",
                "ee_asset_cfg": SceneEntityCfg("robot", body_names="link6"),
                "trunk_asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
                "offset_distance": 0.3,
            },
        )
        ee_target_points_error_delta_gt = ObsTerm(
            func=mdp.ee_target_points_error_delta_9d,
            params={
                "command_name": "ee_target_points",
                "ee_asset_cfg": SceneEntityCfg("robot", body_names="link6"),
                "trunk_asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
                "offset_distance": 0.3,
            },
        )
        pitch_command_gt = ObsTerm(
            func=mdp.pitch_command, params={"command_name": "pitch_command"}
        )

        push_vel_gt = ObsTerm(func=mdp.push_vel)
        # external push force in yaw-aligned trunk frame (2D: fx, fy)
        push_force_gt = ObsTerm(func=mdp.push_force)
        # external yaw torque about world Z (1D: tau_z)
        yaw_torque_gt = ObsTerm(func=mdp.push_yaw_torque)
        # end-effector external force in world axes (3D: Fx,Fy,Fz)
        ee_external_force_gt = ObsTerm(func=mdp.ee_external_forces_applied)
        random_material_gt = ObsTerm(
            func=mdp.random_material_obs,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
        )
        random_com_gt = ObsTerm(
            func=mdp.random_com_obs,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
        )
        random_mass_gt = ObsTerm(
            func=mdp.random_mass_obs,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
        )
        random_actuator_gains_gt = ObsTerm(
            func=mdp.randomize_actuator_gains_obs,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "kp_default": 70.0,
                "kd_default": 2.0,
            },
        )
        random_actuator_lag_gt = ObsTerm(func=mdp.randomize_actuator_lag_obs)
        random_joint_friction_gt = ObsTerm(
            func=mdp.joint_friction_obs,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        )
        height_map_yaw_gt = ObsTerm(
            func=mdp.height_scan_fix,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        leg_phase_gt = ObsTerm(func=mdp.phase_obs, params={"period": 0.7})
        foot_clearance_gt = ObsTerm(
            func=mdp.foot_clearance,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "foot_sensor_cfgs": {
                    "fl": SceneEntityCfg("fl_foot_scanner"),
                    "fr": SceneEntityCfg("fr_foot_scanner"),
                    "rl": SceneEntityCfg("rl_foot_scanner"),
                    "rr": SceneEntityCfg("rr_foot_scanner"),
                },
                "foot_tf_cfg": SceneEntityCfg("frame_transform"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class NoiseAndDelayObsCfg(ObsGroup):
        """Observations for noise and delay."""

        # real observation terms (order preserved)
        base_lin_vel_nad = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel_nad = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity_nad = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel_nad = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel_nad = ObsTerm(func=mdp.joint_vel_rel)
        height_map_yaw_nad = ObsTerm(
            func=mdp.height_scan_fix,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class AMPObsCfg(ObsGroup):
        """Observations for amp."""

        # amp observation terms (order preserved)
        base_lin_xy_vel_gt = ObsTerm(func=mdp.base_lin_xy_vel)
        base_ang_yaw_vel_gt = ObsTerm(func=mdp.base_ang_yaw_vel)
        joint_pos_abs_gt = ObsTerm(func=mdp.joint_pos)
        joint_vel_abs_gt = ObsTerm(func=mdp.joint_vel)
        foot_positions = ObsTerm(
            func=mdp.foot_positions,
            params={"sensor_cfg": SceneEntityCfg("frame_transform")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    ground_truth_obs: GroundTruthObsCfg = GroundTruthObsCfg()
    noise_and_delay_obs: NoiseAndDelayObsCfg = NoiseAndDelayObsCfg()
    amp_obs: AMPObsCfg = AMPObsCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    randomize_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.25, 1.75),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 64,
        },
    )

    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.05, 0.5),
            "operation": "abs",
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_joint_offset = EventTerm(
        func=mdp.reset_joint_offset,
        mode="reset",
        params={
            "randomization_params": (-0.06, 0.06),
            "operation": "add",
            "distribution": "uniform",
        },
    )  # 暂时只允许 add，其他 operation 需要修改

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.1, 0.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains_plus,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "kt_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity_obs_xy,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )

    # interval (external force push in trunk/body XY frame)
    push_force_robot = EventTerm(
        func=mdp.start_push_force_xy_base,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            # force sampling (preferred): sample fx/fy independently in the chosen force frame
            "force_xy_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
            "duration_range_s": (0.2, 0.2),
            "ramp_time_s": 0.1,
            # force direction frame:
            # - "yaw_horizontal": current behavior (horizontal push in yaw-aligned frame)
            # - "yaw_pitch": tilt the applied push with current base pitch (roll ignored)
            "force_frame": "yaw_horizontal",
            # extra pitch tilt (rad): sampled uniformly per event start (min, max)
            "pitch_offset_range_rad": (0.0, 0.0),
        },
    )

    # interval (external yaw torque about global Z on trunk link)
    push_yaw_torque_robot = EventTerm(
        func=mdp.start_push_yaw_torque_z_base,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "torque_magnitude_range": (0.0, 0.0),
            "duration_range_s": (0.2, 0.2),
            "ramp_time_s": 0.1,
        },
    )

    # interval (external force/torque in world frame, per body)
    # NOTE:
    # - Use `asset_cfg` to select which link/body receives the wrench.
    # - By default it is disabled (all ranges are zeros).
    apply_external_force_torque_3d = EventTerm(
        func=mdp.apply_external_force_torque_3d,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_x_range": (0.0, 0.0),
            "force_y_range": (0.0, 0.0),
            "force_z_range": (0.0, 0.0),
            "torque_x_range": (0.0, 0.0),
            "torque_y_range": (0.0, 0.0),
            "torque_z_range": (0.0, 0.0),
        },
    )

    # interval (slot): end-effector external wrench (world frame)
    # NOTE:
    # - Use the same generic wrench applier as other external-force events.
    # - For multi-body with different ranges, define multiple events with different `asset_cfg`.
    ee_external_force = EventTerm(
        func=mdp.start_ee_external_force_3d,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_x_range": (0.0, 0.0),
            "force_y_range": (0.0, 0.0),
            "force_z_range": (0.0, 0.0),
            "torque_x_range": (0.0, 0.0),
            "torque_y_range": (0.0, 0.0),
            "torque_z_range": (0.0, 0.0),
            # Linear ramp duration (s) to the newly sampled force target.
            # None = legacy step-and-hold via apply_external_force_torque_3d;
            # force_control overrides this to 1.0 via its config_dict.
            "ramp_duration_s": None,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    # pose tracking
    track_position_xy_yaw_frame = RewTerm(
        func=mdp.track_position_xy_yaw_frame,
        weight=0.0,
        params={"command_name": "base_command"},
    )
    track_heading = RewTerm(
        func=mdp.track_heading, weight=0.0, params={"command_name": "base_command"}
    )
    # velocity tracking
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=0.0,
        params={"command_name": "base_command", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.0,
        params={"command_name": "base_command", "std": math.sqrt(0.25)},
    )
    track_base_height_exp = RewTerm(
        func=mdp.track_base_height_exp,
        weight=0.0,
        params={"command_name": "base_command", "std": math.sqrt(0.25)},
    )
    track_lin_vel_xy_yaw_frame_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=0.0,
        params={"command_name": "base_command", "std": math.sqrt(0.25)},
    )
    # velocity tracking with external-force guidance (yaw frame force bias)
    track_lin_vel_xy_exp_force_bias = RewTerm(
        func=mdp.track_lin_vel_xy_base_frame_exp_force_bias,
        weight=0.0,
        params={
            "command_name": "base_command",
            "std": math.sqrt(0.25),
            "force_to_vel_scale": 0.0,
            # "components": use x/y force components; "norm": use planar force magnitude as a scalar.
            "force_mode": "components",
            "force_clip": None,
        },
    )
    track_ang_vel_z_world_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.0,
        params={"command_name": "base_command", "std": math.sqrt(0.25)},
    )
    # yaw angular velocity tracking with external-yaw-torque guidance
    track_ang_vel_z_exp_torque_bias = RewTerm(
        func=mdp.track_ang_vel_z_exp_torque_bias,
        weight=0.0,
        params={
            "command_name": "base_command",
            "std": math.sqrt(0.25),
            "torque_to_ang_vel_scale": 0.0,
            "torque_clip": None,
            # upright scaling: set to None to disable
            "upright_scale_max": 0.7,
        },
    )

    # pitch tracking (follow commanded pitch angle)
    track_pitch_exp = RewTerm(
        func=mdp.track_pitch_exp,
        weight=0.0,
        params={
            "command_name": "pitch_command",
            "std": math.sqrt(0.25),
            # upright scaling: set to None to disable
            "upright_scale_max": 0.7,
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )

    # -- penalties
    base_height_new_l2 = RewTerm(
        func=mdp.base_height_new_l2, weight=0.0, params={"command_name": "base_command"}
    )
    orientation_l2 = RewTerm(
        func=mdp.flat_orientation_new_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=0.0)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=0.0)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=0.0)
    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits, weight=0.0, params={"soft_ratio": 0.9}
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint")},
    )
    applied_torque_limits = RewTerm(func=mdp.applied_torque_limits, weight=0.0)
    base_acc_mix_l2 = RewTerm(
        func=mdp.base_acc_mix_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    )
    feet_lin_acc_l2 = RewTerm(
        func=mdp.feet_lin_acc_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*FOOT")},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0.0)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=0.0)
    joint_power = RewTerm(func=mdp.joint_power, weight=0.0)
    joint_power_distribution = RewTerm(func=mdp.joint_power_distribution, weight=0.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.0,
        params={
            "command_name": "base_command",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 0.35,
        },
    )
    stand_joint_deviation_l1 = RewTerm(
        func=mdp.stand_joint_deviation_l1,
        weight=0.0,
        params={
            "command_name": "base_command",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    feet_slide_base_frame = RewTerm(
        func=mdp.feet_slide_base_frame,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "body_asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )

    # -- safety
    feet_contact_forces = RewTerm(
        func=mdp.contact_forces_l2,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 700,
        },
    )
    dont_wait_unreached = RewTerm(
        func=mdp.dont_wait_unreached,
        weight=0.0,
        params={"command_name": "base_command"},
    )
    move_in_direction = RewTerm(
        func=mdp.move_in_direction, weight=0.0, params={"command_name": "base_command"}
    )
    reached_joint_deviation_l2 = RewTerm(
        func=mdp.reached_joint_deviation_l2,
        weight=0.0,
        params={"command_name": "base_command"},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"),
            "threshold": 0.1,
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT")},
    )

    # -- amp
    amp_reward = RewTerm(func=mdp.amp_reward, weight=0.0)

    base_height_l2_fix = RewTerm(
        func=mdp.base_height_l2_fix,
        weight=0.0,
        params={
            "target_height": 0.45,
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )

    hip_joint_penalty = RewTerm(
        func=mdp.hip_joint_penalty,
        weight=0.0,
        params={
            "command_name": "base_command",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
        },
    )

    hip_joint_penalty_lateral_velocity = RewTerm(
        func=mdp.hip_joint_penalty_lateral_velocity,
        weight=0.0,
        params={
            "command_name": "base_command",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
        },
    )

    foot_pos_y_penalty = RewTerm(
        func=mdp.foot_pos_y_penalty,
        weight=0.0,
        params={
            "command_name": "base_command",
            "sensor_cfg": SceneEntityCfg("frame_transform"),
            "target_y_offset": 0.18,
        },
    )

    # -- whole-body manipulation (end-effector tracking)
    track_ee_target_points_exp = RewTerm(
        func=mdp.track_ee_target_points_exp,
        weight=0.0,
        params={
            "command_name": "ee_target_points",
            "ee_asset_cfg": SceneEntityCfg("robot", body_names="link6"),
            "trunk_asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
            # offset_distance 将在具体 env_cfg/config_summary 中覆盖
            "offset_distance": 0.1,
            "std": 0.2,
            "main_weight": 1.0,
            "offset_weight": 0.5,
            "command_active_threshold": 1.0e-6,
        },
    )

    # 拆分：分别跟踪 main/x_offset/y_offset 三个点（便于单独调参）
    track_ee_target_main_exp = RewTerm(
        func=mdp.track_ee_target_main_exp,
        weight=0.0,
        params={
            "command_name": "ee_target_points",
            "ee_asset_cfg": SceneEntityCfg("robot", body_names="link6"),
            "trunk_asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
            "std": 0.2,
            "command_active_threshold": 1.0e-6,
        },
    )

    track_ee_target_x_offset_exp = RewTerm(
        func=mdp.track_ee_target_x_offset_exp,
        weight=0.0,
        params={
            "command_name": "ee_target_points",
            "ee_asset_cfg": SceneEntityCfg("robot", body_names="link6"),
            "trunk_asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
            # offset_distance 将在具体 env_cfg/config_summary 中覆盖
            "offset_distance": 0.1,
            "std": 0.2,
            "command_active_threshold": 1.0e-6,
        },
    )

    track_ee_target_y_offset_exp = RewTerm(
        func=mdp.track_ee_target_y_offset_exp,
        weight=0.0,
        params={
            "command_name": "ee_target_points",
            "ee_asset_cfg": SceneEntityCfg("robot", body_names="link6"),
            "trunk_asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
            # offset_distance 将在具体 env_cfg/config_summary 中覆盖
            "offset_distance": 0.1,
            "std": 0.2,
            "command_active_threshold": 1.0e-6,
        },
    )

    # 外力顺应版：不修改原有奖励项，新增 3 个独立 term（main / x_offset / y_offset）
    track_ee_target_main_exp_force_compliance = RewTerm(
        func=mdp.track_ee_target_main_exp_force_compliance,
        weight=0.0,
        params={
            "command_name": "ee_target_points",
            "ee_asset_cfg": SceneEntityCfg("robot", body_names="link6"),
            "trunk_asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
            "std": 0.2,
            "command_active_threshold": 1.0e-6,
        },
    )

    track_ee_target_x_offset_exp_force_compliance = RewTerm(
        func=mdp.track_ee_target_x_offset_exp_force_compliance,
        weight=0.0,
        params={
            "command_name": "ee_target_points",
            "ee_asset_cfg": SceneEntityCfg("robot", body_names="link6"),
            "trunk_asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
            # offset_distance 将在具体 env_cfg/config_summary 中覆盖
            "offset_distance": 0.1,
            "std": 0.2,
            "command_active_threshold": 1.0e-6,
        },
    )

    track_ee_target_y_offset_exp_force_compliance = RewTerm(
        func=mdp.track_ee_target_y_offset_exp_force_compliance,
        weight=0.0,
        params={
            "command_name": "ee_target_points",
            "ee_asset_cfg": SceneEntityCfg("robot", body_names="link6"),
            "trunk_asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
            # offset_distance 将在具体 env_cfg/config_summary 中覆盖
            "offset_distance": 0.1,
            "std": 0.2,
            "command_active_threshold": 1.0e-6,
        },
    )

    # -- whole-body manipulation
    # pitch cooperation with end-effector target height
    track_pitch_with_ee_target_height_exp = RewTerm(
        func=mdp.track_pitch_with_ee_target_height_exp,
        weight=0.0,
        params={
            "command_name": "ee_target_points",
            # default mapping (should be overridden per task/config)
            # Set to None to auto-resolve from command cfg (ranges.pos_z)
            "z_low": None,
            "z_high": None,
            "pitch_at_z_low": -0.3,
            "pitch_at_z_high": 0.3,
            "std": 0.5,
            "command_active_threshold": 1.0e-6,
            # upright scaling: set to None to disable
            "upright_scale_max": 0.7,
        },
    )

    # 外力偏移版本：姿态跟踪使用“偏移后”的末端目标高度
    track_pitch_with_ee_target_height_exp_force_compliance = RewTerm(
        func=mdp.track_pitch_with_ee_target_height_exp_force_compliance,
        weight=0.0,
        params={
            "command_name": "ee_target_points",
            "trunk_asset_cfg": SceneEntityCfg(
                "robot", body_names="base_link"
            ),
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
            "z_low": None,
            "z_high": None,
            "pitch_at_z_low": -0.3,
            "pitch_at_z_high": 0.3,
            "std": 0.5,
            "command_active_threshold": 1.0e-6,
            "upright_scale_max": 0.7,
        },
    )

    # -- handstand
    handstand_feet_height_exp = RewTerm(
        func=mdp.handstand_feet_height_exp,
        weight=0.0,
        params={
            "std": 0.25,
            "front_target_height": 0.6,
            "back_target_height": 0.6,
            "command_name": "base_command",
            "front_foot_asset_cfg": SceneEntityCfg("robot", body_names="F.*_foot"),
            "back_foot_asset_cfg": SceneEntityCfg("robot", body_names="R.*_foot"),
        },
    )
    handstand_feet_on_air = RewTerm(
        func=mdp.handstand_feet_on_air,
        weight=0.0,
        params={
            "front_foot_sensor_cfg": SceneEntityCfg("contact_forces", body_names="F.*_foot"),
            "back_foot_sensor_cfg": SceneEntityCfg("contact_forces", body_names="R.*_foot"),
            "command_name": "base_command",
        },
    )
    handstand_feet_air_time = RewTerm(
        func=mdp.handstand_feet_air_time,
        weight=0.0,
        params={
            "front_foot_sensor_cfg": SceneEntityCfg("contact_forces", body_names="F.*_foot"),
            "back_foot_sensor_cfg": SceneEntityCfg("contact_forces", body_names="R.*_foot"),
            "command_name": "base_command",
            "threshold": 0.35,
        },
    )
    handstand_orientation_l2 = RewTerm(
        func=mdp.handstand_orientation_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "command_name": "base_command",
        },
    )
    handstand_base_height_w_l2 = RewTerm(
        func=mdp.handstand_base_height_w_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "command_name": "base_command",
            "front_target_height": 0.6,
            "back_target_height": 0.6,
            "normal_target_height": 0.5,
        },
    )
    feet_air_time_positive_biped = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.0,
        params={
            "threshold": 0.35,
            "front_foot_sensor_cfg": SceneEntityCfg("contact_forces", body_names="F.*_foot"),
            "back_foot_sensor_cfg": SceneEntityCfg("contact_forces", body_names="R.*_foot"),
            "command_name": "base_command",
        },
    )

    handstand_hip_default_joint_pos_l2 = RewTerm(
        func=mdp.handstand_hip_default_joint_pos_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
            "command_name": "base_command",
        },
    )
    handstand_thigh_default_joint_pos_l2 = RewTerm(
        func=mdp.handstand_thigh_default_joint_pos_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_thigh_joint"),
            "command_name": "base_command",
        },
    )
    handstand_calf_default_joint_pos_l2 = RewTerm(
        func=mdp.handstand_calf_default_joint_pos_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_calf_joint"),
            "command_name": "base_command",
        },
    )
    foot_pos_x_forward = RewTerm(
        func=mdp.foot_pos_x_forward,
        weight=0.0,
        params={
            "command_name": "base_command",
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "foot_tf_cfg": SceneEntityCfg("frame_transform"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    track_base_height_exp_partial= RewTerm(
        func=mdp.track_base_height_exp_partial,
        weight=0.0,
        params={
            "height_target": 0.35,
            "std": math.sqrt(0.25),
            "foot_sensor_cfg": SceneEntityCfg("frame_transform"),
            "contact_sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=".*_foot"
                ),
        },
    )
    base_height_from_transform_l2 = RewTerm(
        func=mdp.base_height_from_transform_l2,
        weight=0.0,
        params={
            "foot_tf_cfg": SceneEntityCfg("frame_transform"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "base_height_target": 0.45,
        },
    )
    contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 300.0,
        },
    )
    gait_reward = RewTerm(
        func=mdp.GaitReward,
        weight=0.0,
        params={
            "std": 0.25,
            "command_name": "base_command",
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "synced_feet_pair_names": [["FL_foot", "RR_foot"], ["FR_foot", "RL_foot"]],
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    contact_reward = RewTerm(
        func=mdp.contact_reward,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    foot_clearance_reward = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.0,
        params={
            "std": 0.25,
            "tanh_mult": 5.0,
            "target_clearance_height": 0.15,
            "foot_asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    first_contact_foot_forces_penalty = RewTerm(
        func=mdp.first_contact_foot_forces_penalty,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 400.0,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"),
            "threshold": 1.0,
        },
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "limit_angle": 1.4,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # terrain_levels_pos = CurrTerm(func=mdp.terrain_levels_pos)
    terrain_levels_vel = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_x_command_threshold = CurrTerm(func=mdp.lin_vel_x_command_threshold)
    ang_vel_z_command_threshold = CurrTerm(func=mdp.ang_vel_z_command_threshold)
    ee_external_force_threshold = CurrTerm(func=mdp.ee_external_force_threshold)


##
# Environment configuration
##


@configclass
class LocomotionEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Locomotion environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=1500, env_spacing=0.1)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

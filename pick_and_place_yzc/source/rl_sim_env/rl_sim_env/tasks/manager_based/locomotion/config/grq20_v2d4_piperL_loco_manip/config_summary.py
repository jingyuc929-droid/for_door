import glob
import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from rl_algorithms.rsl_rl_wrapper import LocomotionOnPolicyRunnerCfg
from rl_sim_env import RL_SIM_ENV_ROOT_DIR

RL_SIM_ENV_ASSETS_DIR = os.path.join(RL_SIM_ENV_ROOT_DIR, "data/assets")
RL_SIM_ENV_DATASETS_DIR = os.path.join(RL_SIM_ENV_ROOT_DIR, "data/datasets")
RL_SIM_ENV_CONFIG_SUMMARY_DIR = os.path.abspath(os.path.dirname(__file__))

ROBOT_BASE_LINK = "base_link"
ROBOT_FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
ROBOT_JOINT_NAMES = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]
ROBOT_ARM_JOINT_NAMES = [
    "link1_joint",
    "link2_joint",
    "link3_joint",
    "link4_joint",
    "link5_joint",
    "link6_joint",
]
ROBOT_THIGH_NAMES = ["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]
ROBOT_CALF_NAMES = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]


@configclass
class AMPDataCfg:
    ROOT_POS_SIZE = 3
    ROOT_ROT_SIZE = 4
    ROOT_LINEAR_VEL_SIZE = 3
    ROOT_ANGULAR_VEL_SIZE = 3
    FRAME_POS_SIZE = 12
    FRAME_VEL_SIZE = 12
    JOINT_POS_SIZE = 12
    JOINT_VEL_SIZE = 12

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + ROOT_POS_SIZE
    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROOT_ROT_SIZE
    ROOT_LINEAR_VEL_START_IDX = ROOT_ROT_END_IDX
    ROOT_LINEAR_VEL_END_IDX = ROOT_LINEAR_VEL_START_IDX + ROOT_LINEAR_VEL_SIZE
    ROOT_ANGULAR_VEL_START_IDX = ROOT_LINEAR_VEL_END_IDX
    ROOT_ANGULAR_VEL_END_IDX = ROOT_ANGULAR_VEL_START_IDX + ROOT_ANGULAR_VEL_SIZE
    FRAME_POS_START_IDX = ROOT_ANGULAR_VEL_END_IDX
    FRAME_POS_END_IDX = FRAME_POS_START_IDX + FRAME_POS_SIZE
    FRAME_VEL_START_IDX = FRAME_POS_END_IDX
    FRAME_VEL_END_IDX = FRAME_VEL_START_IDX + FRAME_VEL_SIZE
    JOINT_POS_START_IDX = FRAME_VEL_END_IDX
    JOINT_POS_END_IDX = JOINT_POS_START_IDX + JOINT_POS_SIZE
    JOINT_VEL_START_IDX = JOINT_POS_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    TOTAL_SIZE = (
        ROOT_POS_SIZE
        + ROOT_ROT_SIZE
        + ROOT_LINEAR_VEL_SIZE
        + ROOT_ANGULAR_VEL_SIZE
        + FRAME_POS_SIZE
        + FRAME_VEL_SIZE
        + JOINT_POS_SIZE
        + JOINT_VEL_SIZE
    )

    root_keys = [
        "root_position_world",
        "root_quaternion_wxyz",
        "root_linear_velocity_base",
        "root_angular_velocity_base",
    ]
    frame_keys = ROBOT_FOOT_NAMES
    joint_keys = ROBOT_JOINT_NAMES
    frame_pos_keys = [frame + "_position_base" for frame in frame_keys]
    frame_vel_keys = [frame + "_velocity_base" for frame in frame_keys]
    joint_pos_keys = [joint + "_q" for joint in joint_keys]
    joint_vel_keys = [joint + "_dq" for joint in joint_keys]
    all_keys = root_keys + frame_pos_keys + frame_vel_keys + joint_pos_keys + joint_vel_keys

    root_position_world_indices = list(range(ROOT_POS_START_IDX, ROOT_POS_END_IDX))
    root_quaternion_indices = list(range(ROOT_ROT_START_IDX, ROOT_ROT_END_IDX))
    root_linear_velocity_base_indices = list(range(ROOT_LINEAR_VEL_START_IDX, ROOT_LINEAR_VEL_END_IDX))
    root_linear_velocity_xy_indices = list(range(ROOT_LINEAR_VEL_START_IDX, ROOT_LINEAR_VEL_END_IDX - 1))
    root_angular_velocity_base_indices = list(range(ROOT_ANGULAR_VEL_START_IDX, ROOT_ANGULAR_VEL_END_IDX))
    root_angular_velocity_yaw_indices = list(range(ROOT_ANGULAR_VEL_START_IDX + 2, ROOT_ANGULAR_VEL_END_IDX))
    frame_position_base_indices = list(range(FRAME_POS_START_IDX, FRAME_POS_END_IDX))
    frame_velocity_base_indices = list(range(FRAME_VEL_START_IDX, FRAME_VEL_END_IDX))
    joint_q_indices = list(range(JOINT_POS_START_IDX, JOINT_POS_END_IDX))
    joint_qd_indices = list(range(JOINT_VEL_START_IDX, JOINT_VEL_END_IDX))
    combined_indices = (
        root_linear_velocity_xy_indices
        + root_angular_velocity_yaw_indices
        + joint_q_indices
        + joint_qd_indices
        + frame_position_base_indices
    )


ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{RL_SIM_ENV_ASSETS_DIR}/robots/grq20_v2d4_piperL_front_mount/grq20_v2d4_piperL_front_mount.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.50),
        joint_pos={
            "FL_hip_joint": -0.05,
            "FL_thigh_joint": 0.75,
            "FL_calf_joint": -1.5,
            "FR_hip_joint": 0.05,
            "FR_thigh_joint": 0.75,
            "FR_calf_joint": -1.5,
            "RL_hip_joint": -0.05,
            "RL_thigh_joint": 0.75,
            "RL_calf_joint": -1.5,
            "RR_hip_joint": 0.05,
            "RR_thigh_joint": 0.75,
            "RR_calf_joint": -1.5,
            "link1_joint": 0.0,
            "link2_joint": 1.5,
            "link3_joint": -0.8,
            "link4_joint": 0.0,
            "link5_joint": 0.4,
            "link6_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit={".*_hip_joint": 120.0, ".*_thigh_joint": 120.0, ".*_calf_joint": 180.0},
            velocity_limit={".*_hip_joint": 20.0, ".*_thigh_joint": 20.0, ".*_calf_joint": 14.0},
            stiffness=75.0,
            damping=2.5,
            friction=0.05,
            armature=0.01,
            min_delay=0,
            max_delay=3,
        ),
        "arm": DelayedPDActuatorCfg(
            joint_names_expr=ROBOT_ARM_JOINT_NAMES,
            effort_limit=50.0,
            velocity_limit=15.0,
            stiffness=25.0,
            damping=1.0,
            friction=0.05,
            armature=0.01,
            min_delay=0,
            max_delay=3,
        ),
    },
)


@configclass
class ConfigSummary:
    class general:
        decimation = 4
        episode_length_s = 20.0
        render_interval = 4
        is_finite_horizon = False

    class sim:
        dt = 0.005

    class env:
        num_envs = 4000

        num_critic_obs = 51
        num_actor_obs = 47

        estimator_history_length = 5
        num_estimator_step_obs = 47
        num_estimator_obs = estimator_history_length * num_estimator_step_obs
        num_estimator_out = 23
        num_next_obs = 51

        num_privileged_obs = 43
        num_heightmap_obs = 187
        num_amp_obs = 39

        num_actions = 12
        action_history_length = 3

        clip_actions = 100.0
        clip_obs = 100.0

        hip_tor_limit = 80.0
        thigh_tor_limit = 80.0
        calf_tor_limit = 80.0

        RL_SIM_ENV_CONFIG_SUMMARY_DIR = RL_SIM_ENV_CONFIG_SUMMARY_DIR

        policy_type = {
            "actor_critic_type": "ActorCriticEncoder",
            "vae_type": "VAEBlind",
        }
        training_type = "rl"

        module_cfg_dict = {
            "amp": {
                "hidden_dims": [1024, 512],
            },
            "vae": {
                "encoder_in_dim": num_estimator_obs,
                "encoder_hidden_dims": [128],
                "encoder_out_dim": 64,
                "encoder_head_dim_dict": {"obs_vel": 3, "obs_com": 3, "obs_mass": 1, "obs_latent": 16},
                "decoder_in_dim": num_estimator_out,
                "decoder_hidden_dims": [64, 128],
                "decoder_out_dim": num_next_obs,
                "activation": "elu",
            },
            "actor_critic": {
                "actor": {
                    "num_actor_obs": num_actor_obs + num_estimator_out,
                    "num_actions": num_actions,
                    "actor_hidden_dims": [512, 256, 128],
                    "actor_obs_normalization": False,
                },
                "privileged_encoder": {
                    "num_privileged_obs": num_privileged_obs,
                    "privileged_encoder_hidden_dims": [64, 32],
                    "num_privileged_encoder_out": 16,
                },
                "heightmap_encoder": {
                    "num_heightmap_obs": num_heightmap_obs,
                    "heightmap_encoder_hidden_dims": [64, 32],
                    "num_heightmap_encoder_out": 32,
                },
                "critic": {
                    "num_critic_obs": num_critic_obs + 16 + 32,
                    "critic_hidden_dims": [512, 256, 128],
                    "critic_obs_normalization": False,
                },
                "init_noise_std": 1.0,
                "noise_std_type": "scalar",
                "activation": "elu",
                "min_normalized_std": [0.01, 0.01, 0.01] * 4,
            },
        }

        train_cfg_dict = {
            "use_amp": True,
            "use_vae": True,
            "ppo_algorithm": {
                "value_loss_coef": 0.5,
                "use_clipped_value_loss": True,
                "clip_param": 0.2,
                "entropy_coef": 0.01,
                "num_learning_epochs": 5,
                "num_mini_batches": 4,
                "learning_rate": 2.0e-5,
                "schedule": "adaptive",
                "gamma": 0.99,
                "lam": 0.95,
                "desired_kl": 0.01,
                "max_grad_norm": 1.0,
                "normalize_advantage_per_mini_batch": False,
            },
            "amp": {
                "amp_replay_buffer_size": 1000000,
                "amp_disc_grad_penalty": 5.0,
                "motion_files": glob.glob(f"{RL_SIM_ENV_DATASETS_DIR}/grq20_v2d3/diffheight/npz/*"),
                "num_preload_transitions": 2000000,
            },
            "vae": {
                "use_exclusive_optimizer": False,
                "use_exclusive_lr": False,
                "learning_rate": 1.0e-3,
                "beta_adaptive": True,
                "beta_max_step": 5000,
                "beta": 1.0e-4,
                "beta_max": 0.01,
                "free_bits": 0.5,
                "use_adaboot": True,
                "adaboot_max_step": 500,
            },
        }

    class action:
        scale = 0.25

    class observation:
        leg_joint_cfg = SceneEntityCfg("robot", joint_names=ROBOT_JOINT_NAMES)

        obs_term_dict = {
            "ground_truth_obs": {
                "base_lin_vel_gt": {"scale": 2.0},
                "base_ang_vel_gt": {"scale": 0.25},
                "base_height_b_gt": {"scale": 2.0},
                "projected_gravity_gt": {},
                "joint_pos_rel_gt": {"params": {"asset_cfg": leg_joint_cfg}},
                "joint_vel_rel_gt": {"scale": 0.05, "params": {"asset_cfg": leg_joint_cfg}},
                "actions_gt": {"scale": 0.25},
                "base_commands_gt": {"scale": (2.0, 2.0, 0.25, 2.0, 1.0)},
                "push_vel_gt": {},
                "random_material_gt": {
                    "params": {"asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_FOOT_NAMES)}
                },
                "random_com_gt": {
                    "scale": 5.0,
                    "params": {"asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_BASE_LINK)},
                },
                "random_mass_gt": {
                    "scale": 0.1,
                    "params": {"asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_BASE_LINK)},
                },
                "random_actuator_gains_gt": {
                    "scale": 0.1,
                    "params": {"asset_cfg": leg_joint_cfg, "kp_default": 75.0, "kd_default": 2.5},
                },
                "random_actuator_lag_gt": {},
                "height_map_yaw_gt": {"scale": 5.0, "clip": (-2.0, 2.0)},
            },
            "noise_and_delay_obs": {
                "base_ang_vel_nad": {"scale": 0.25, "noise": 0.3, "delay": 2},
                "projected_gravity_nad": {"noise": 0.05, "delay": 2},
                "joint_pos_rel_nad": {
                    "noise": 0.03,
                    "delay": 1,
                    "params": {"asset_cfg": leg_joint_cfg},
                },
                "joint_vel_rel_nad": {
                    "scale": 0.05,
                    "noise": 1.5,
                    "delay": 2,
                    "params": {"asset_cfg": leg_joint_cfg},
                },
            },
            "amp_obs": {
                "base_lin_xy_vel_gt": {},
                "base_ang_yaw_vel_gt": {},
                "joint_pos_abs_gt": {"params": {"asset_cfg": leg_joint_cfg}},
                "joint_vel_abs_gt": {"params": {"asset_cfg": leg_joint_cfg}},
                "foot_positions": {},
            },
        }

        policy_obs_dict = {
            "critic_obs": {
                "terms": [
                    "base_lin_vel_gt",
                    "base_ang_vel_gt",
                    "base_height_b_gt",
                    "projected_gravity_gt",
                    "joint_pos_rel_gt",
                    "joint_vel_rel_gt",
                    "actions_gt",
                    "base_commands_gt",
                ],
            },
            "actor_obs": {
                "terms": [
                    "base_ang_vel_nad",
                    "projected_gravity_nad",
                    "joint_pos_rel_nad",
                    "joint_vel_rel_nad",
                    "actions_gt",
                    "base_commands_gt",
                ],
            },
            "estimator_obs": {
                "history_length": 5,
                "terms": [
                    "base_ang_vel_nad",
                    "projected_gravity_nad",
                    "joint_pos_rel_nad",
                    "joint_vel_rel_nad",
                    "actions_gt",
                    "base_commands_gt",
                ],
            },
            "privileged_obs": {
                "terms": [
                    "push_vel_gt",
                    "random_mass_gt",
                    "random_com_gt",
                    "random_material_gt",
                    "random_actuator_gains_gt",
                    "random_actuator_lag_gt",
                ],
            },
            "gt_heightmap_obs": {"terms": ["height_map_yaw_gt"]},
            "amp_obs": {
                "terms": [
                    "base_lin_xy_vel_gt",
                    "base_ang_yaw_vel_gt",
                    "joint_pos_abs_gt",
                    "joint_vel_abs_gt",
                    "foot_positions",
                ],
            },
            "next_obs": {
                "terms": [
                    "base_lin_vel_gt",
                    "base_ang_vel_gt",
                    "base_height_b_gt",
                    "projected_gravity_gt",
                    "joint_pos_rel_gt",
                    "joint_vel_rel_gt",
                    "actions_gt",
                    "base_commands_gt",
                ],
            },
            "gt_vel_obs": {"terms": ["base_lin_vel_gt"]},
            "gt_mass_obs": {"terms": ["random_mass_gt"]},
            "gt_com_obs": {"terms": ["random_com_gt"]},
            "gt_base_height_obs": {"terms": ["base_height_b_gt"]},
        }

        extra_obs_dict = {"estimator_out": 23}

    class event:
        config_dict = {
            "randomize_base_mass": {
                "mode": "startup",
                "params": {
                    "asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_BASE_LINK),
                    "mass_distribution_params": (-1.0, 20.0),
                    "operation": "add",
                },
            },
            "randomize_base_com": {
                "mode": "startup",
                "params": {
                    "asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_BASE_LINK),
                    "com_range": {"x": (-0.05, 0.05), "y": (-0.03, 0.03), "z": (-0.03, 0.05)},
                },
            },
            "randomize_physics_material": {
                "mode": "startup",
                "params": {
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.25, 1.2),
                    "dynamic_friction_range": (0.25, 1.2),
                    "restitution_range": (0.0, 1.0),
                    "num_buckets": 64,
                },
            },
            "reset_base": {
                "mode": "reset",
                "params": {
                    "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-1.57, 1.57)},
                    "velocity_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (-0.5, 0.5),
                        "roll": (0.0, 0.0),
                        "pitch": (0.0, 0.0),
                        "yaw": (0.0, 0.0),
                    },
                },
            },
            "reset_robot_joints": {
                "mode": "reset",
                "params": {
                    "asset_cfg": SceneEntityCfg("robot", joint_names=ROBOT_JOINT_NAMES),
                    "position_range": (-0.5, 0.5),
                    "velocity_range": (0.0, 0.0),
                },
            },
            "reset_actuator_gains": {
                "mode": "reset",
                "params": {
                    "asset_cfg": SceneEntityCfg("robot", joint_names=ROBOT_JOINT_NAMES),
                    "stiffness_distribution_params": (0.8, 1.2),
                    "damping_distribution_params": (0.8, 1.2),
                    "kt_distribution_params": (0.8, 1.2),
                    "operation": "scale",
                    "distribution": "uniform",
                },
            },
            "push_robot": {
                "mode": "interval",
                "params": {"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
            },
            "randomize_passive_arm_targets": {
                "mode": "interval",
                "interval_range_s": (0.5, 1.5),
                "params": {
                    "asset_cfg": SceneEntityCfg("robot", joint_names=ROBOT_ARM_JOINT_NAMES),
                    "position_ranges": {
                        "link1_joint": (-0.8, 0.8),
                        "link2_joint": (0.2, 1.4),
                        "link3_joint": (-1.6, -0.2),
                        "link4_joint": (-0.8, 0.8),
                        "link5_joint": (-0.4, 1.2),
                        "link6_joint": (-1.0, 1.0),
                    },
                    "velocity_range": (-0.2, 0.2),
                    "hold_probability": 0.35,
                    "write_state": False,
                },
            },
        }

    class reward:
        only_positive_reward = True

        leg_joint_cfg = SceneEntityCfg("robot", joint_names=ROBOT_JOINT_NAMES)

        config_dict = {
            "track_lin_vel_xy_exp": {
                "weight": 1.5,
                "params": {"command_name": "base_command", "std": math.sqrt(0.25)},
            },
            "track_ang_vel_z_exp": {
                "weight": 0.5,
                "params": {"command_name": "base_command", "std": math.sqrt(0.25)},
            },
            "track_base_height_exp": {
                "weight": 0.8,
                "params": {"command_name": "base_command", "std": math.sqrt(0.008)},
            },
            "track_base_height_exp_partial": {
                "weight": 0.4,
                "params": {
                    "height_target": 0.426,
                    "std": math.sqrt(0.08),
                    "foot_sensor_cfg": SceneEntityCfg("frame_transform"),
                    "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=ROBOT_FOOT_NAMES),
                },
            },
            "track_pitch_exp": {
                "weight": 1.1,
                "params": {
                    "command_name": "base_command",
                    "command_index": 4,
                    "std": math.sqrt(0.015),
                    "upright_scale_max": 0.85,
                    "asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_BASE_LINK),
                },
            },
            "base_height_new_l2": {"weight": -0.0},
            "orientation_l2": {"weight": [-0.0, -0.05, -0.05, -0.05, -0.0, -0.0, -1.5]},
            "ang_vel_xy_l2": {"weight": -0.08},
            "lin_vel_z_l2": {"weight": -3.0},
            "dof_vel_l2": {"weight": -1.0e-4, "params": {"asset_cfg": leg_joint_cfg}},
            "dof_acc_l2": {"weight": -2.5e-7, "params": {"asset_cfg": leg_joint_cfg}},
            "dof_torques_l2": {"weight": -0.1e-4, "params": {"asset_cfg": leg_joint_cfg}},
            "action_rate_l2": {"weight": -0.01},
            "action_smoothness_l2": {"weight": -0.01},
            "joint_power": {"weight": -2.0e-5, "params": {"asset_cfg": leg_joint_cfg}},
            "joint_power_distribution": {"weight": -1.0e-5, "params": {"asset_cfg": leg_joint_cfg}},
            "feet_air_time": {"weight": 0.35, "params": {"threshold": 0.25}},
            "undesired_contacts": {
                "weight": -0.1,
                "params": {
                    "threshold": 0.1,
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ROBOT_THIGH_NAMES + ROBOT_CALF_NAMES),
                },
            },
            "stand_joint_deviation_l1": {
                "weight": -0.08,
                "params": {"command_name": "base_command", "asset_cfg": leg_joint_cfg},
            },
            "feet_slide": {"weight": -0.15},
            "feet_slide_base_frame": {"weight": -0.05},
            "amp_reward": {"weight": 0.5},
            "base_height_l2_fix": {"weight": -0.2, "params": {"target_height": 0.426}},
            "hip_joint_penalty": {
                "weight": [-0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.3],
                "params": {
                    "command_name": "base_command",
                    "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
                },
            },
            "foot_pos_y_penalty": {
                "weight": -0.0,
                "params": {"command_name": "base_command", "target_y_offset": 0.18},
            },
            "dof_pos_limits": {"weight": -0.0, "params": {"asset_cfg": leg_joint_cfg}},
            "feet_contact_forces": {
                "weight": -5.0e-4,
                "params": {
                    "threshold": 350.0,
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ROBOT_FOOT_NAMES),
                },
            },
            "contact_forces": {
                "weight": -3.0e-4,
                "params": {
                    "threshold": 450.0,
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ROBOT_FOOT_NAMES),
                },
            },
        }

    class command:
        heading_control_stiffness = 0.5
        standing_command_prob = 0.15
        lin_vel_x = (-1.0, 1.0)
        lin_vel_y = (-0.5, 0.5)
        ang_vel_z = (-1.0, 1.0)
        heading = (-math.pi / 2, math.pi / 2)
        base_height_cmd = (0.35, 0.5)
        fixed_height_cmd = 0.42
        probability_of_using_fixed_height_cmd = 0.35
        base_pitch_cmd = (-0.2, 0.5)
        fixed_pitch_cmd = 0.0
        probability_of_using_fixed_pitch_cmd = 0.35


@configclass
class LocomotionPiperLLowerBodyPPORunnerCfg(LocomotionOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 1000
    experiment_name = "grq20_v2d4_piperL_loco_manip_lower"
    swanlab_project = "grq20_v2d4_piperL_loco_manip_lower"

    policy_type = ConfigSummary.env.policy_type
    training_type = ConfigSummary.env.training_type

    module_cfg_dict = ConfigSummary.env.module_cfg_dict
    train_cfg_dict = ConfigSummary.env.train_cfg_dict
    amp_loader_cfg = AMPDataCfg()

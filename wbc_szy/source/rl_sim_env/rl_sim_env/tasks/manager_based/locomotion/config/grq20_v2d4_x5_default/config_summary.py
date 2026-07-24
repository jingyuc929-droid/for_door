import math
import os

from rl_sim_env import RL_SIM_ENV_ROOT_DIR
from rl_sim_env.tasks.manager_based.common.mdp import UniformVelocityCommandTerrainCfg
from isaaclab.managers import SceneEntityCfg

RL_SIM_ENV_ASSETS_DIR = os.path.join(RL_SIM_ENV_ROOT_DIR, "data/assets")
RL_SIM_ENV_DATASETS_DIR = os.path.join(RL_SIM_ENV_ROOT_DIR, "data/datasets")
RL_SIM_ENV_CONFIG_SUMMARY_DIR = os.path.abspath(os.path.dirname(__file__))

import glob

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass
from rl_algorithms.rsl_rl_wrapper import (
    LocomotionOnPolicyRunnerCfg,
)

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
# 机械臂（X5）关节：用于全身控制（腿+臂）。AMP/腿相关保持使用 ROBOT_JOINT_NAMES（仅腿12DOF）
ROBOT_ARM_JOINT_NAMES = [
    "link1_joint",
    "link2_joint",
    "link3_joint",
    "link4_joint",
    "link5_joint",
    "link6_joint",
]
ROBOT_WHOLE_BODY_JOINT_NAMES = ROBOT_JOINT_NAMES + ROBOT_ARM_JOINT_NAMES

# 末端执行器刚体名（从 URDF/资产中读取后填写）
# 说明：这里的名字必须与 articulation 的 body name 匹配（用于 reward/obs）。
ROBOT_EE_BODY_NAME = "link6"
ROBOT_THIGH_NAMES = ["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]
ROBOT_CALF_NAMES = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
ROBOT_HIP_NAMES = ["FL_hip", "FR_hip", "RL_hip", "RR_hip"]


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
        # grq20_v2d4_x5: 四足 + X5机械臂
        usd_path=f"{RL_SIM_ENV_ASSETS_DIR}/robots/grq20_v2d4_x5/grq20_v2d4_x5.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
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
            # 机械臂默认姿态（占位）：后续任务可按需要调整
            "link1_joint": 0.0,
            "link2_joint": 0.6,
            "link3_joint": 0.6,
            "link4_joint": 0.0,
            "link5_joint": 0.0,
            "link6_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            # effort_limit=120.0,
            # velocity_limit=20.0,
            effort_limit={".*_hip_joint": 120.0, ".*_thigh_joint": 120.0, ".*_calf_joint": 180.0},
            velocity_limit={".*_hip_joint": 20.0, ".*_thigh_joint": 20.0, ".*_calf_joint": 14.0},
            stiffness=75.0,
            damping=2.5,
            friction=0.05,
            armature=0.01,
            min_delay=0,
            max_delay=3,
        ),
        # 机械臂 actuator：逐关节拆分（每个关节参数可能不同）
        "arm_link1": DelayedPDActuatorCfg(
            joint_names_expr=["link1_joint"],
            effort_limit=50.0,
            velocity_limit=15.0,
            stiffness=20.0,
            damping=0.8,
            friction=0.05,
            armature=0.01,
            min_delay=0,
            max_delay=3,
        ),
        "arm_link2": DelayedPDActuatorCfg(
            joint_names_expr=["link2_joint"],
            effort_limit=50.0,
            velocity_limit=15.0,
            stiffness=20.0,
            damping=0.8,
            friction=0.05,
            armature=0.01,
            min_delay=0,
            max_delay=3,
        ),
        "arm_link3": DelayedPDActuatorCfg(
            joint_names_expr=["link3_joint"],
            effort_limit=50.0,
            velocity_limit=15.0,
            stiffness=20.0,
            damping=0.8,
            friction=0.05,
            armature=0.01,
            min_delay=0,
            max_delay=3,
        ),
        "arm_link4": DelayedPDActuatorCfg(
            joint_names_expr=["link4_joint"],
            effort_limit=50.0,
            velocity_limit=15.0,
            stiffness=20.0,
            damping=0.8,
            friction=0.05,
            armature=0.01,
            min_delay=0,
            max_delay=3,
        ),
        "arm_link5": DelayedPDActuatorCfg(
            joint_names_expr=["link5_joint"],
            effort_limit=50.0,
            velocity_limit=15.0,
            stiffness=15.0,
            damping=0.6,
            friction=0.05,
            armature=0.01,
            min_delay=0,
            max_delay=3,
        ),
        "arm_link6": DelayedPDActuatorCfg(
            joint_names_expr=["link6_joint"],
            effort_limit=50.0,
            velocity_limit=15.0,
            stiffness=15.0,
            damping=0.6,
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

        # 全身 DOF（腿12 + 机械臂6）
        num_dof = len(ROBOT_WHOLE_BODY_JOINT_NAMES)
        num_actions = num_dof

        # 机械臂目标点命令观测维度（ee_target_points_gt）
        ee_target_points_dim = 9

        # 维度与 observation.policy_obs_dict 对齐：
        # - critic_obs: 3(base_lin_vel)+3(base_ang_vel)+3(projected_gravity)+num_dof(joint_pos)+num_dof(joint_vel)
        #              +num_actions(actions)+3(base_commands)+ee_target_points_dim
        # - actor_obs : 3(base_ang_vel)+3(projected_gravity)+num_dof(joint_pos)+num_dof(joint_vel)
        #              +num_actions(actions)+3(base_commands)+ee_target_points_dim
        # 注意：本任务不再显式使用 pitch_command，因此不再计入 1 维 pitch 指令观测
        num_critic_obs = 12 + 2 * num_dof + num_actions + ee_target_points_dim
        num_actor_obs = 9 + 2 * num_dof + num_actions + ee_target_points_dim

        estimator_history_length = 5
        num_estimator_step_obs = num_actor_obs
        num_estimator_obs = estimator_history_length * num_estimator_step_obs
        num_estimator_out = 23
        num_next_obs = num_critic_obs

        # privileged_obs dims: push_vel(2) + mass(1) + com(3) + material(12) + actuator_gains(24) + lag(1) + foot_clearance(4) = 47
        num_privileged_obs = 47
        num_heightmap_obs = 187

        # AMP：仅腿部（按你说的 3 + 24 + 12 保持不变）
        num_amp_obs = 39
        action_history_length = 3

        clip_actions = 100.0
        clip_obs = 100.0

        hip_tor_limit = 100.0
        thigh_tor_limit = 100.0
        calf_tor_limit = 140.0

        RL_SIM_ENV_CONFIG_SUMMARY_DIR=RL_SIM_ENV_CONFIG_SUMMARY_DIR

        policy_type = {
            'actor_critic_type': "ActorCriticEncoder",
            'vae_type': "VAEBlind",
        }

        training_type = "rl"

        module_cfg_dict = {
            'amp': {
                'hidden_dims': [1024, 512],
            },
            "vae": {
                'encoder_in_dim': num_estimator_obs,
                'encoder_hidden_dims': [128],
                'encoder_out_dim': 64,
                'encoder_head_dim_dict': {'obs_vel': 3, 'obs_com': 3, 'obs_mass': 1, 'obs_latent': 16},
                'decoder_in_dim': num_estimator_out,
                'decoder_hidden_dims': [64, 128],
                'decoder_out_dim': num_next_obs,
                'activation': "elu",
            },
            "actor_critic": {
                "actor": {
                    'num_actor_obs': num_actor_obs + num_estimator_out,
                    'num_actions': num_actions,
                    'actor_hidden_dims': [512, 256, 128],
                    'actor_obs_normalization': False,
                },
                'privileged_encoder': {
                    'num_privileged_obs': num_privileged_obs,
                    'privileged_encoder_hidden_dims': [64, 32],
                    'num_privileged_encoder_out': 16,
                },
                'heightmap_encoder': {
                    'num_heightmap_obs': num_heightmap_obs,
                    'heightmap_encoder_hidden_dims': [64, 32],
                    'num_heightmap_encoder_out': 32,
                },
                "critic": {
                    'num_critic_obs': num_critic_obs + 16 + 32,
                    'critic_hidden_dims': [512, 256, 128],
                    'critic_obs_normalization': False,
                },
                'init_noise_std': 1.0,
                'noise_std_type': "scalar",
                'activation': "elu",
                # runner 里会与 dof_range（长度=num_actions）逐元素相乘，这里必须与 num_actions 对齐
                'min_normalized_std': [0.01] * num_actions,

            }
        }

        train_cfg_dict = {
            'use_amp': True,
            'use_vae': True,
            'ppo_algorithm':
            {
                'value_loss_coef' : 0.5,
                'use_clipped_value_loss' : True,
                'clip_param' : 0.2,
                'entropy_coef' : 0.01,
                'num_learning_epochs' : 5,
                'num_mini_batches' : 4,
                'learning_rate' : 2.0e-5,
                'schedule' : "adaptive",
                'gamma' : 0.99,
                'lam' : 0.95,
                'desired_kl' : 0.01,
                'max_grad_norm' : 1.0,
                'normalize_advantage_per_mini_batch' : False,
            },
            'amp' :
            {
                'amp_replay_buffer_size' : 1000000,
                'amp_disc_grad_penalty' : 5.0,
                'motion_files' : glob.glob(f"{RL_SIM_ENV_DATASETS_DIR}/grq20_v2d4_default/new/npz/*"),
                'num_preload_transitions': 2000000,
            },
            'vae' : {
                'use_exclusive_optimizer' : False,
                'use_exclusive_lr' : False,
                'learning_rate' : 1.0e-3,
                'beta_adaptive' : True,
                'beta_max_step' : 5000,
                'beta' : 1.0e-4,
                'beta_max' : 0.01,
                'free_bits' : 0.5,
                'use_adaboot' : True,
                'adaboot_max_step' : 500,
            },

        }

    class action:
        scale = 0.25

    class observation:
        obs_term_dict = {
            'ground_truth_obs': {
                # ground truth observation terms
                'base_lin_vel_gt': {'scale': 2.0, },
                'base_ang_vel_gt': {'scale': 0.25, },
                'projected_gravity_gt': {},
                # 全身关节观测（腿+臂）
                'joint_pos_rel_gt': {
                    'params': {
                        "asset_cfg": SceneEntityCfg("robot", joint_names=ROBOT_WHOLE_BODY_JOINT_NAMES),
                    }
                },
                'joint_vel_rel_gt': {
                    'scale': 0.05,
                    'params': {
                        "asset_cfg": SceneEntityCfg("robot", joint_names=ROBOT_WHOLE_BODY_JOINT_NAMES),
                    }
                },
                'actions_gt': {'scale': 0.25, },
                'base_commands_gt': {'scale': (2.0, 2.0, 0.25), },
                'ee_target_points_gt': {
                    'params': {'command_name': 'ee_target_points'},
                },
                # push velocity in world XY frame (2D)
                'push_vel_gt': {},
                'random_material_gt': {
                    'params': {
                        "asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_FOOT_NAMES)
                    }
                },
                'random_com_gt': {'scale': 5.0,
                                  'params': {
                                      "asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_BASE_LINK)
                                  }},
                'random_mass_gt': {'scale': 0.1,
                                   'params': {
                                       "asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_BASE_LINK)
                                   }},
                'random_actuator_gains_gt': {'scale': 0.1,
                                             'params': {
                                                 "kp_default": 75.0,
                                                 "kd_default": 3.0
                                             }},
                'random_actuator_lag_gt': {},
                'height_map_yaw_gt': {'scale': 5.0, 'clip': (-2.0, 2.0)},
                'foot_clearance_gt': {'scale': (5.0)},
            },
            'noise_and_delay_obs': {
                # noise and delay observation terms
                'base_ang_vel_nad': {'scale': 0.25, 'noise': 0.3, 'delay': 2},
                'projected_gravity_nad': {'noise': 0.05, 'delay': 2},
                # 全身关节观测（腿+臂）
                'joint_pos_rel_nad': {
                    'noise': 0.03,
                    'delay': 1,
                    'params': {
                        "asset_cfg": SceneEntityCfg("robot", joint_names=ROBOT_WHOLE_BODY_JOINT_NAMES),
                    }
                },
                'joint_vel_rel_nad': {
                    'scale': 0.05,
                    'noise': 1.5,
                    'delay': 2,
                    'params': {
                        "asset_cfg": SceneEntityCfg("robot", joint_names=ROBOT_WHOLE_BODY_JOINT_NAMES),
                    }
                },
            },
            'amp_obs': {
                # amp observation terms
                'base_lin_xy_vel_gt': {},
                'base_ang_yaw_vel_gt': {},
                # AMP：仅腿部关节（机械臂不参与 AMP）
                'joint_pos_abs_gt': {
                    'params': {
                        "asset_cfg": SceneEntityCfg("robot", joint_names=ROBOT_JOINT_NAMES),
                    }
                },
                'joint_vel_abs_gt': {
                    'params': {
                        "asset_cfg": SceneEntityCfg("robot", joint_names=ROBOT_JOINT_NAMES),
                    }
                },
                'foot_positions': {},
            }
        }

        policy_obs_dict = {
            'critic_obs': {
                'terms': ['base_lin_vel_gt',
                          'base_ang_vel_gt',
                          'projected_gravity_gt',
                          'joint_pos_rel_gt',
                          'joint_vel_rel_gt',
                          'actions_gt',
                          'base_commands_gt',
                          'ee_target_points_gt',
                          ],
            },
            'actor_obs': {
                'terms': [
                    'base_ang_vel_nad',
                    'projected_gravity_nad',
                    'joint_pos_rel_nad',
                    'joint_vel_rel_nad',
                    'actions_gt',
                    'base_commands_gt',
                    'ee_target_points_gt',
                ],
            },
            'estimator_obs': {
                'history_length': 5,
                'terms': [
                    'base_ang_vel_nad',
                    'projected_gravity_nad',
                    'joint_pos_rel_nad',
                    'joint_vel_rel_nad',
                    'actions_gt',
                    'base_commands_gt',
                    'ee_target_points_gt',
                ],
            },
            'privileged_obs': {
                'terms': ['push_vel_gt',
                          'random_mass_gt',
                          'random_com_gt',
                          'random_material_gt',
                          'random_actuator_gains_gt',
                          'random_actuator_lag_gt',
                          'foot_clearance_gt',
                          ],
            },
            'gt_heightmap_obs': {
                'terms': ['height_map_yaw_gt',
                          ],
            },
            'amp_obs': {
                'terms': ['base_lin_xy_vel_gt',
                          'base_ang_yaw_vel_gt',
                          'joint_pos_abs_gt',
                          'joint_vel_abs_gt',
                          'foot_positions',
                          ],
            },
            'next_obs': {
                'terms': ['base_lin_vel_gt',
                          'base_ang_vel_gt',
                          'projected_gravity_gt',
                          'joint_pos_rel_gt',
                          'joint_vel_rel_gt',
                          'actions_gt',
                          'base_commands_gt',
                          'ee_target_points_gt',
                          ],
            },
            'gt_vel_obs': {
                'terms': ['base_lin_vel_gt',
                          ],
            },
            'gt_mass_obs': {
                'terms': ['random_mass_gt',
                          ],
            },
            'gt_com_obs': {
                'terms': ['random_com_gt',
                          ],
            },
        }

        extra_obs_dict = {
            'estimator_out': 23,
        }

    class event:
        config_dict = {
            'randomize_base_mass': {
                'mode': "startup",
                'params': {
                    'asset_cfg': SceneEntityCfg("robot", body_names=ROBOT_BASE_LINK),
                    'mass_distribution_params': (-1.0, 30.0),
                    'operation': "add",
                }
            },
            'randomize_base_com': {
                'mode': "startup",
                'params': {
                    'asset_cfg': SceneEntityCfg("robot", body_names=ROBOT_BASE_LINK),
                    'com_range': {"x": (-0.05, 0.05), "y": (-0.03, 0.03), "z": (-0.03, 0.05)},
                }
            },
            'randomize_joint_friction': {
                'mode': "startup",
                'params': {
                    'asset_cfg': SceneEntityCfg("robot", joint_names=".*"),
                    'friction_distribution_params': (0.01, 0.10),
                    'operation': "abs",
                }
            },
            'randomize_physics_material': {
                'mode': "startup",
                'params': {
                    'asset_cfg': SceneEntityCfg("robot", body_names=".*"),
                    'static_friction_range': (0.25, 1.2),
                    'dynamic_friction_range': (0.25, 1.2),
                    'restitution_range': (0.0, 0.1),
                    'num_buckets': 64,
                }
            },
            'reset_joint_offset': {
                'mode': "startup",
                'params': {
                    'randomization_params': (-0.02, 0.02),
                    'operation': "add",
                    'distribution': "uniform",
                }
            },
            'reset_base': {
                'mode': "reset",
                'params': {
                    'pose_range': {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-1.57, 1.57)},
                    'velocity_range': {
                        'x': (-0.5, 0.5),
                        'y': (-0.5, 0.5),
                        'z': (-0.5, 0.5),
                        'roll': (0.0, 0.0),
                        'pitch': (0.0, 0.0),
                        'yaw': (0.0, 0.0),
                    }
                }
            },
            'reset_robot_joints': {
                'mode': "reset",
                'params': {
                    'position_range': (-0.4, 0.6),
                    'velocity_range': (0.0, 0.0),
                }
            },
            'reset_actuator_gains': {
                'mode': "reset",
                'params': {
                    'asset_cfg': SceneEntityCfg("robot", joint_names=".*"),
                    'stiffness_distribution_params': (0.8, 1.2),
                    'damping_distribution_params': (0.8, 1.2),
                    'kt_distribution_params': (0.8, 1.2),
                    'operation': "scale",
                    'distribution': "uniform",
                }
            },
            'push_robot': {
                'mode': "interval",
                'params': {
                    'velocity_range': {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
                }
            },
        }

    class reward:   
        only_positive_reward = True

        config_dict = {
            'track_lin_vel_xy_exp': {
                # disabled: replaced by force-biased velocity tracking
                'weight': 1.5,
                'params': {
                    'command_name': 'base_command',
                    'std': math.sqrt(0.25)
                }
            },
            # 'track_lin_vel_xy_exp_force_bias': {
            #     'weight': 1.5,
            #     'params': {
            #         'command_name': 'base_command',
            #         'std': math.sqrt(0.25),
            #         # v_target_xy = v_cmd_xy + force_to_vel_scale * f_xy
            #         # f_xy convention (tilted terrain ready):
            #         # - f_x: projection of applied world force onto trunk forward (yaw+pitch, roll ignored)
            #         # - f_y: keep yaw-horizontal y component (for now)
            #         # force_to_vel_scale can be a scalar or tuple (scale_x, scale_y) for different scaling per axis
            #         'force_to_vel_scale': (0.015, 0.01),  # (x_scale, y_scale)
            #         'force_clip': 20.0,
            #     }
            # },
            'track_ang_vel_z_exp': {
                # disabled: replaced by yaw-torque-biased yaw-rate tracking
                'weight': 0.5,
                'params': {
                    'command_name': 'base_command',
                    'std': math.sqrt(0.25)
                }
            },
            # 'track_ang_vel_z_exp_torque_bias': {
            #     'weight': 0.5,
            #     'params': {
            #         'command_name': 'base_command',
            #         'std': math.sqrt(0.25),
            #         # ω_target_z = ω_cmd_z + torque_to_ang_vel_scale * τ_z
            #         'torque_to_ang_vel_scale': 0.04,
            #         'torque_clip': 20.0,
            #         # upright scaling: set to None to disable
            #         'upright_scale_max': 0.7,
            #     }
            # },
            'track_base_height_exp_partial': {
                # 'weight': 0.2,
                'weight': 0.3,
                'params':{
                    'height_target': 0.42,
                    'std': math.sqrt(0.1),
                    "foot_sensor_cfg": SceneEntityCfg("frame_transform"),
                    "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=ROBOT_FOOT_NAMES),
                }
            },
            # NOTE: disable generic orientation penalty; replaced by pitch tracking reward
            'orientation_l2': {
                'weight': -0.0,
            },
            'ang_vel_xy_l2': {
                # 'weight': -0.05
                'weight': -0.1
            },
            'lin_vel_z_l2': {
                # 'weight': -2.0
                'weight': -4.0
            }, 
            'dof_vel_l2': {
                'weight': -2.0e-4
            },
            'dof_acc_l2': {
                'weight': -2.5e-7
            },
            'dof_torques_l2': {
                'weight': -3e-5
            },
            'action_rate_l2': {
                'weight': -0.01
            },
            'action_smoothness_l2': {
                'weight': -0.01
            },
            'joint_power': {
                'weight': -2.0e-6
            },
            'joint_power_distribution': {
                'weight': -1.0e-5
            },
            'feet_air_time': {
                'weight': 1.0,
                'params': {
                    'threshold': 0.35
                }
            },
            'undesired_contacts': {
                'weight': -0.1,
                'params': {
                    'threshold': 0.1,
                    'sensor_cfg': SceneEntityCfg("contact_forces", body_names=ROBOT_THIGH_NAMES + ROBOT_CALF_NAMES)
                }
            },
            'stand_joint_deviation_l1': {
                # 'weight': -0.1
                'weight': -0.1,
                'params': {
                    # 仅对腿部关节施加“站立回默认姿态”惩罚，避免机械臂在站立时被拉回默认姿态
                    'asset_cfg': SceneEntityCfg("robot", joint_names=ROBOT_JOINT_NAMES),
                },
            },
            'feet_slide': {
                'weight': -0.1
            },
            'feet_slide_base_frame': {
                # 'weight': -0.0
                'weight': [-0.0, -0.0, -0.0, -0.0, -0.1, -0.1, -0.1]
            },
            'amp_reward': {
                'weight': 0.4
            },
            'base_height_l2_fix': {
                # 'weight': -0,
                'weight': -0.5,
                'params': {
                    'target_height': 0.42
                }
            },
            'hip_joint_penalty': {
                # 'weight':-0.05,
                'weight':[-0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3],
                'params': {
                    'command_name': 'base_command',
                    'asset_cfg': SceneEntityCfg("robot", joint_names=".*_hip_joint")
                }
            },
            'hip_joint_penalty_lateral_velocity':{
                'weight':-0.0,
                'params': {
                    'command_name': 'base_command',
                    'asset_cfg': SceneEntityCfg("robot", joint_names=".*_hip_joint"),
                }
            },
            'foot_pos_y_penalty':{
                'weight':-0.0,
                'params': {
                    'command_name': 'base_command',
                    'target_y_offset': 0.18,
                }
            },
            'dof_pos_limits':{
                'weight':-0.0,
            },
            'contact_forces': {
                # 'weight': -1e-3,
                'weight': 0.0,
                'params': {
                    'threshold': 400.0,
                    'sensor_cfg': SceneEntityCfg("contact_forces", body_names=".*_foot"),
                }
            },
            'foot_clearance_reward': {
                # 'weight': 0.5,
                'weight': [0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05],
                'params': {
                    'std': 0.22,
                    'tanh_mult': 5.0,
                    'target_clearance_height': 0.15,
                    'foot_asset_cfg': SceneEntityCfg("robot", body_names=ROBOT_FOOT_NAMES),
                },
            },
            # 机械臂末端跟随（投影COM参考系下的 9D 目标点）
            # 注意：该奖励依赖于 commands.ee_target_points 输出的 9D 命令
            # 拆分：分别对应 3 个点的跟随奖励（main / x_offset / y_offset）
            'track_ee_target_main_exp': {
                'weight': 0.5,
                'params': {
                    'command_name': 'ee_target_points',
                    'ee_asset_cfg': SceneEntityCfg("robot", body_names=ROBOT_EE_BODY_NAME),
                    'trunk_asset_cfg': SceneEntityCfg("robot", body_names=ROBOT_BASE_LINK),
                    'terrain_sensor_cfg': SceneEntityCfg("height_scanner"),
                    'std': 0.15,
                    'command_active_threshold': 1.0e-6,
                },
            },
            'track_ee_target_x_offset_exp': {
                'weight': 1.0,
                'params': {
                    'command_name': 'ee_target_points',
                    'ee_asset_cfg': SceneEntityCfg("robot", body_names=ROBOT_EE_BODY_NAME),
                    'trunk_asset_cfg': SceneEntityCfg("robot", body_names=ROBOT_BASE_LINK),
                    'terrain_sensor_cfg': SceneEntityCfg("height_scanner"),
                    # offset_distance 将在 env_cfg 中与 command 的 offset_distance 对齐
                    'offset_distance': None,
                    'std': 0.10,
                    'command_active_threshold': 1.0e-6,
                },
            },
            'track_ee_target_y_offset_exp': {
                'weight': 1.0,
                'params': {
                    'command_name': 'ee_target_points',
                    'ee_asset_cfg': SceneEntityCfg("robot", body_names=ROBOT_EE_BODY_NAME),
                    'trunk_asset_cfg': SceneEntityCfg("robot", body_names=ROBOT_BASE_LINK),
                    'terrain_sensor_cfg': SceneEntityCfg("height_scanner"),
                    # offset_distance 将在 env_cfg 中与 command 的 offset_distance 对齐
                    'offset_distance': None,
                    'std': 0.10,
                    'command_active_threshold': 1.0e-6,
                },
            },
            # 末端目标高度协同机身俯仰：
            # - 末端目标点更低（z 接近 z_low）时，鼓励更"低头"的 pitch（pitch_at_z_low）
            # - 末端目标点更高（z 接近 z_high）时，鼓励更"抬头"的 pitch（pitch_at_z_high）
            # - 新特性：使用平滑过渡函数，区间内 pitch 引导很小，区间外平滑增强（无硬截断）
            # 说明：如果发现低头/抬头方向反了，交换 pitch_at_z_low / pitch_at_z_high 即可。
            'track_pitch_with_ee_target_height_exp': {
                # 'weight': 0.3,
                'weight': 0.4,
                'params': {
                    'command_name': 'ee_target_points',
                    # 中性区间（0点 / deadband）：用上下限定义
                    # 在 [neutral_z_low, neutral_z_high] 内 pitch 引导接近 neutral_pitch（很小）
                    # 区间外平滑地过渡到完整的 z->pitch 线性映射（连续、无硬截断）
                    'neutral_z_low': 0.5,
                    'neutral_z_high': 0.85,
                    'neutral_pitch': 0.0,
                    # 软引导角度（rad），不要太大，避免"硬对应"
                    'pitch_at_z_low': 0.4,
                    'pitch_at_z_high': -0.2,
                    # 越大越"软"，越不强制某个 z 对应某个 pitch
                    'std': 0.150,
                    'command_active_threshold': 1.0e-6,
                    # 直立缩放：倒地/大倾斜时衰减该奖励；设为 None 可关闭
                    'upright_scale_max': 0.7,
                },
            },
        }

    class command:
        lin_x_level: float = 0.0
        max_lin_x_level: float = 5.0
        ang_z_level: float = 0.0
        max_ang_z_level: float = 5.0
        vel_curriculum_episode_mult: float = 8.0

        heading_control_stiffness = 0.5

        # ----------------------------
        # base_command 的“模式”采样比例（每次 command resample 时按概率为部分 env 置零速度等）
        # 说明：
        # - yaw_command_prob: 原地转向/转圈（线速度清零，仅保留 yaw 角速度）
        # - standing_command_prob: 原地静止（线速度和角速度都清零）
        # 训练时当前默认使用 rough-only 地形（`random_rough`），因此把可调参数集中放这里便于调参。
        # ----------------------------
        yaw_command_prob: float = 0.3
        standing_command_prob: float = 0.0

        # ee target points command (9D) in projected COM yaw frame
        ee_target_resampling_time_range = (12, 12)
        # ee target command smooth ramp time (seconds)
        # - 0.0: jump instantly to new targets (legacy behavior)
        # - >0: interpolate to new targets within this duration
        ee_target_ramp_time_s = 0.5
        ee_target_debug_vis = True
        # debug marker size (meters)
        # - ee_target_marker_radius: 目标点球大小（main/x/y 三个点）
        # - ee_pos_marker_radius: 末端实际位置球大小（用于对比重合情况）
        ee_target_marker_radius = 0.03
        ee_pos_marker_radius = 0.02
        ee_target_offset_distance = 0.30
        ee_target_pos_r_range = (0.30, 0.70)
        ee_target_pos_theta_range = (-1.3, 1.3)
        ee_target_pos_z_range = (0.10, 0.85)
        ee_target_roll_range = (-1.57, 1.57)
        ee_target_pitch_range = (-1.57, 1.57)
        ee_target_yaw_range = (-0.5, 0.5)

        ranges = {
            "pyramid_stairs": UniformVelocityCommandTerrainCfg.Ranges(
                lin_vel_x=(-0.5, 0.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.25, 0.25),
                heading=(-math.pi / 2, math.pi / 2),
                heading_command_prob=1.0,
                yaw_command_prob=0.0,
                standing_command_prob=0.0,
                start_curriculum_lin_x=(-0.5, 0.5),
                start_curriculum_ang_z=(-0.25, 0.25),
                max_curriculum_lin_x=(-1.0, 1.0),
                max_curriculum_ang_z=(-1.0, 1.0),
            ),
            "pyramid_stairs_inv": UniformVelocityCommandTerrainCfg.Ranges(
                lin_vel_x=(-0.0, 0.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.25, 0.25),
                heading=(-math.pi / 2, math.pi / 2),
                heading_command_prob=1.0,
                yaw_command_prob=0.0,
                standing_command_prob=0.0,
                start_curriculum_lin_x=(-0.0, 0.5),
                start_curriculum_ang_z=(-0.25, 0.25),
                max_curriculum_lin_x=(-0.0, 1.0),
                max_curriculum_ang_z=(-1.0, 1.0),
            ),
            "boxes": UniformVelocityCommandTerrainCfg.Ranges(
                lin_vel_x=(-0.5, 0.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.25, 0.25),
                heading=(-math.pi / 2, math.pi / 2),
                heading_command_prob=1.0,
                yaw_command_prob=0.0,
                standing_command_prob=0.0,
                start_curriculum_lin_x=(-0.5, 0.5),
                start_curriculum_ang_z=(-0.25, 0.25),
                max_curriculum_lin_x=(-1.0, 1.0),
                max_curriculum_ang_z=(-1.0, 1.0),
            ),
            "random_rough": UniformVelocityCommandTerrainCfg.Ranges(
                lin_vel_x=(-0.5, 0.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.25, 0.25),
                heading=(-math.pi / 2, math.pi / 2),
                heading_command_prob=1.0,
                yaw_command_prob=yaw_command_prob,
                standing_command_prob=standing_command_prob,
                start_curriculum_lin_x=(-0.5, 0.5),
                start_curriculum_ang_z=(-0.25, 0.25),
                max_curriculum_lin_x=(-1.0, 1.0),
                max_curriculum_ang_z=(-1.0, 1.0),
            ),
            "hf_pyramid_slope": UniformVelocityCommandTerrainCfg.Ranges(
                lin_vel_x=(-0.5, 0.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.25, 0.25),
                heading=(-math.pi / 2, math.pi / 2),
                heading_command_prob=1.0,
                yaw_command_prob=0.0,
                standing_command_prob=0.0,
                start_curriculum_lin_x=(-0.5, 0.5),
                start_curriculum_ang_z=(-0.25, 0.25),
                max_curriculum_lin_x=(-1.0, 1.0),
                max_curriculum_ang_z=(-1.0, 1.0),
            ),
            "hf_pyramid_slope_inv": UniformVelocityCommandTerrainCfg.Ranges(
                lin_vel_x=(-0.5, 0.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.25, 0.25),
                heading=(-math.pi / 2, math.pi / 2),
                heading_command_prob=1.0,
                yaw_command_prob=0.0,
                standing_command_prob=0.0,
                start_curriculum_lin_x=(-0.5, 0.5),
                start_curriculum_ang_z=(-0.25, 0.25),
                max_curriculum_lin_x=(-1.0, 1.0),
                max_curriculum_ang_z=(-1.0, 1.0),
            ),
            # "mesh_gap": UniformVelocityCommandTerrainCfg.Ranges(
            #     lin_vel_x=(-1.0, 1.0),
            #     lin_vel_y=(0.0, 0.0),
            #     ang_vel_z=(0.0, 0.0),
            #     heading=(-math.pi / 4, math.pi / 4),
            #     heading_command_prob=0.0,
            #     yaw_command_prob=0.0,
            #     standing_command_prob=0.0,
            #     start_curriculum_lin_x=(-1.0, 1.0),
            #     start_curriculum_ang_z=(-1.5, 1.5),
            #     max_curriculum_lin_x=(-1.5, 1.5),
            #     max_curriculum_ang_z=(-1.5, 1.5),
            # ),
            # "mesh_pit": UniformVelocityCommandTerrainCfg.Ranges(
            #     lin_vel_x=(0.0, 1.0),
            #     lin_vel_y=(0.0, 0.0),
            #     ang_vel_z=(0.0, 0.0),
            #     heading=(-math.pi / 4, math.pi / 4),
            #     heading_command_prob=0.0,
            #     yaw_command_prob=0.0,
            #     standing_command_prob=0.0,
            #     start_curriculum_lin_x=(-1.0, 1.0),
            #     start_curriculum_ang_z=(-1.5, 1.5),
            #     max_curriculum_lin_x=(-1.5, 1.5),
            #     max_curriculum_ang_z=(-1.5, 1.5),
            # ),
            # "mesh_box": UniformVelocityCommandTerrainCfg.Ranges(
            #     lin_vel_x=(-1.0, 1.0),
            #     lin_vel_y=(0.0, 0.0),
            #     ang_vel_z=(0.0, 0.0),
            #     heading=(-math.pi / 4, math.pi / 4),
            #     heading_command_prob=0.0,
            #     yaw_command_prob=0.0,
            #     standing_command_prob=0.0,
            #     start_curriculum_lin_x=(-1.0, 1.0),
            #     start_curriculum_ang_z=(-1.5, 1.5),
            #     max_curriculum_lin_x=(-1.5, 1.5),
            #     max_curriculum_ang_z=(-1.5, 1.5),
            # ),
            "plane_run": UniformVelocityCommandTerrainCfg.Ranges(
                lin_vel_x=(-0.5, 0.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.25, 0.25),
                heading=(-math.pi / 2, math.pi / 2),
                heading_command_prob=0.3,
                yaw_command_prob=0.3,
                standing_command_prob=0.2,
                start_curriculum_lin_x=(-0.5, 0.5),
                start_curriculum_ang_z=(-0.25, 0.25),
                max_curriculum_lin_x=(-1.5, 1.5),
                max_curriculum_ang_z=(-1.5, 1.5),
            ),
            # "plane_yaw": UniformVelocityCommandTerrainCfg.Ranges(
            #     lin_vel_x=(0.0, 0.0),
            #     lin_vel_y=(0.0, 0.0),
            #     ang_vel_z=(-0.25, 0.25),
            #     heading=(-math.pi / 2, math.pi / 2),
            #     heading_command_prob=0.0,
            #     yaw_command_prob=0.05,
            #     standing_command_prob=0.0,
            #     start_curriculum_lin_x=(0.0, 0.0),
            #     start_curriculum_ang_z=(-0.25, 0.25),
            #     max_curriculum_lin_x=(0.0, 0.0),
            #     max_curriculum_ang_z=(-1.5, 1.5),
            # ),
            # "plane_stand": UniformVelocityCommandTerrainCfg.Ranges(
            #     lin_vel_x=(0.0, 0.0),
            #     lin_vel_y=(0.0, 0.0),
            #     ang_vel_z=(0.0, 0.0),
            #     heading=(-math.pi / 2, math.pi / 2),
            #     heading_command_prob=0.0,
            #     yaw_command_prob=0.05,
            #     standing_command_prob=0.0,
            #     start_curriculum_lin_x=(0.0, 0.0),
            #     start_curriculum_ang_z=(0.0, 0.0),
            #     max_curriculum_lin_x=(0.0, 0.0),
            #     max_curriculum_ang_z=(0.0, 0.0),
            # ),
        }


@configclass
class LocomotionWholeBodyPPORunnerCfg(LocomotionOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = [0, 500, 1000, 2000, 3000, 5000, 7000, 10000, 12000, 15000, 20000, 30000, 50000, 70000, 100000]
    experiment_name = "grq20_v2d4_x5_default_locomotion"
    swanlab_project = "grq20_v2d4_x5_default_locomotion"

    policy_type = ConfigSummary.env.policy_type
    training_type = ConfigSummary.env.training_type

    module_cfg_dict = ConfigSummary.env.module_cfg_dict
    train_cfg_dict = ConfigSummary.env.train_cfg_dict
    amp_loader_cfg = AMPDataCfg()

"""Basic configuration for the pull-door task.

Stage-0 grasping and Stage-1 unlocking rewards are shared with the push-door
task. Stage-2 uses a pull-specific base reward that follows the handle's
counter-clockwise circular path.
"""

import os

import door_env
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
import isaaclab.envs.mdp as mdp_std
from .door_env_env_cfg import (
    ActionsCfg,
    DOORWAY_CENTER_XY,
    DOORWAY_FORWARD_AXIS,
    DoorEnvSceneCfg,
    EventCfg,
    ObservationsCfg,
    _HOOK_CONTACT_KEEP_PARAMS,
    _HOOK_KEEP_PARAMS,
    _STAGE0_REWARD_TERMS,
)


@configclass
class PullDoorEnvSceneCfg(DoorEnvSceneCfg):
    """Scene assets and initial poses for pulling the door."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.door.spawn.usd_path = (
            f"{os.path.dirname(door_env.__file__)}/assets/Door_description/usd/Door_pull.usd"
        )

        # Disable the push task's spring-to-closed behavior. Damping and
        # friction remain active, while the handle-controlled lock event is
        # kept for the Stage-1 unlocking interaction.
        self.door.actuators["door_joint"].stiffness = 0.0
        self.door.actuators["handle_joint"].dynamic_friction = 0.0
        self.door.actuators["handle_joint"].viscous_friction = 0.0



@configclass
class PullActionsCfg(ActionsCfg):
    """Pull-door actions; initially identical to the push-door controller."""

    pass


@configclass
class PullObservationsCfg(ObservationsCfg):
    """Pull-door observations with Pull-only teacher privileged state."""

    @configclass
    class PullPrivilegedStateCfg(ObservationsCfg.PrivilegedStateCfg):
        zone_one_hot = ObsTerm(
            func=mdp.pull_zone_one_hot,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
                "door_closed_pos": 0.0,
                "panel_rotation_sign": -1.0,
                "closed_panel_axis_d": (1.0, 0.0),
                "hinge_offset_d": (0.0, 0.0),
                "door_length": 1.2,
                "radius": 3.0,
            },
        )
        traverse_state_one_hot = ObsTerm(func=mdp.pull_traverse_state_one_hot)
        traverse_started = ObsTerm(func=mdp.pull_traverse_started)
        traverse_cheat_given = ObsTerm(func=mdp.pull_traverse_cheat_given)

    privileged_state: PullPrivilegedStateCfg = PullPrivilegedStateCfg()

    def __post_init__(self) -> None:
        self.privileged_state.door_open_direction.params["door_open_sign"] = -1.0


@configclass
class PullEventCfg(EventCfg):
    """Door mechanism events using the pull task's negative opening direction."""

    def __post_init__(self) -> None:
        self.door_mechanism.params["door_open_sign"] = -1.0


@configclass
class PullRewardsCfg:
    """Shared grasp/unlock rewards plus pull-specific base following."""

    stage_gated_door_reward = RewTerm(
        func=mdp.stage_gated_door_reward,
        weight=1.0,
        params={
            "enable_stage_gated_reward": True,
            "stage0_only_reward": False,
            "pre_grasp_cap": 0.0,
            "stage0_reward_terms": _STAGE0_REWARD_TERMS,

            # One-shot Stage 0 -> 1 transition reward per episode.
            "grasp_success_weight": 30.0,
            "grasp_success_params": {
                "handle_joint_cfg": SceneEntityCfg("door", joint_names=["handle_joint"]),
                **_HOOK_KEEP_PARAMS,
                "distance_threshold": 0.05,
                "force_threshold": 0.25,
                "hold_steps": 10,
                "bonus": 20.0,
                "require_wrap": True,
                "require_ee_above_target": True,
                "minimum_height_delta": 0.0,
                "archive_cap": 512,
                "relax_near_after_handle_pos": -0.05,
                "less_than": True,
            },

            "press_handle_weight": 2.0,
            "press_handle_params": {
                "handle_joint_cfg": SceneEntityCfg("door", joint_names=["handle_joint"]),
                **_HOOK_CONTACT_KEEP_PARAMS,
                "less_than": True,
                "vel_deadzone": 0.01,
                "vel_scale": 0.05,
                "opposite_penalty": 0.2,
                "clip": 1.0,
                "pos_deadzone": 1.0e-4,
                "pos_scale": 2.0e-3,
                "use_vel_ema": True,
                "vel_ema_alpha": 0.25,
            },

            "keep_handle_after_press_weight": 1.5,
            "keep_handle_after_press_params": {
                "handle_joint_cfg": SceneEntityCfg("door", joint_names=["handle_joint"]),
                "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
                **_HOOK_CONTACT_KEEP_PARAMS,
                "handle_start_pos": 0.0,
                "handle_threshold": -0.30,
                "activate_progress": 0.20,
                "use_unlock_success_latch": True,
                "door_closed_pos": 0.0,
                "door_open_sign": -1.0,
                "push_enter_open": 0.02,
                "door_open_threshold": 0.35,
                "max_keep_steps_after_unlock": 24,
                "keep_until_door_open": True,
                "hold_reward": 0.01,
                "progress_boost": 0.04,
                "release_event_penalty": 0.20,
                "lost_penalty": 0.02,
                "auto_open_penalty": 0.05,
            },

            "stall_after_grasp_weight": 0.5,
            "stall_after_grasp_params": {
                "handle_joint_cfg": SceneEntityCfg("door", joint_names=["handle_joint"]),
                **_HOOK_CONTACT_KEEP_PARAMS,
                "stall_pos": -0.10,
                "pos_scale": 0.03,
                "penalty": 0.02,
                "recent_window_steps": 200,
                "grace_steps": 10,
                "less_than": True,
            },

            "stall_after_press_weight": 0.5,
            "stall_after_press_params": {
                "handle_joint_cfg": SceneEntityCfg("door", joint_names=["handle_joint"]),
                "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
                "enter_depth": 0.25,
                "exit_depth": 0.22,
                "grace_steps": 60,
                "ramp_steps": 60,
                "door_closed_pos": 0.0,
                "door_open_sign": -1.0,
                "door_progress_threshold": 0.01,
                "max_penalty": 0.2,
                "less_than": True,
            },

            "unlock_progress_weight": 4.5,
            "unlock_progress_params": {
                "handle_joint_cfg": SceneEntityCfg("door", joint_names=["handle_joint"]),
                **_HOOK_CONTACT_KEEP_PARAMS,
                "require_grasp_success": True,
                "handle_start_pos": 0.0,
                "reward_stop_pos": -0.30,
                "delta_power": 1.2,
                "ema_alpha": 0.7,
                "deadzone": 5.0e-5,
                "backtrack_penalty": 0.01,
                "delta_gain": 1.2,
                "abs_power": 1.8,
                "abs_gain": 0.35,
                "hold_start_ratio": 0.35,
                "hold_power": 1.6,
                "hold_gain": 0.25,
                "clip": 2.0,
            },

            # One-shot Stage 1 -> 2 transition reward per episode.
            "unlock_transition_weight": 30.0,
            "unlock_transition_params": {"bonus": 60.0},

            # Stage-2 pull progress uses the same smoothed delta and absolute
            # progress design as push-door, with negative opening direction.
            "push_door_weight": 8.0,
            "push_door_params": {
                "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
                **_HOOK_CONTACT_KEEP_PARAMS,
                "require_unlock_success_latch": True,
                "require_gate": True,
                "contact_threshold": 1.0,
                "distance_threshold": 0.1,
                "align_threshold": 0.5,
                "door_open_sign": -1.0,
                "door_closed_pos": 0.0,
                "door_open_target": 0.8,
                "delta_scale": 1.0,
                "abs_scale": 0.2,
                "ema_alpha": 0.25,
                "deadzone": 1.0e-4,
                # Reward only new episode-best opening and penalize rollback.
                "backtrack_penalty": 0.5,
                "use_best_open_progress": True,
                "rollback_scale": 0.10,
                "clip": 1.0,
                "premature_release_penalty": 0.2,
            },
        },
    )

    base_pull_follow = RewTerm(
        func=mdp.base_pull_follow_reward,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "handle_cfg": SceneEntityCfg("door", body_names=["handle_1"]),
            "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
            "door_closed_pos": 0.0,
            "door_open_sign": -1.0,
            "pull_start_joint_pos": 0.0,
            "pull_end_joint_pos": -0.8,
            "door_progress_scale": 0.02,
            "backtrack_penalty": 0.5,
            "use_best_open_progress": True,
            "stance_std": 0.25,
            "hinge_offset_d": (0.0, 0.0),
            "tangent_velocity_scale": 0.4,
            "max_planar_speed": 0.6,
            "overspeed_margin": 0.2,
            "yaw_gain": 2.0,
            "reach_deadzone": 0.10,
            "reach_scale": 0.20,
            "door_progress_weight": 8.0,
            "door_angle_weight": 0.5,
            "stance_weight": 3.0,
            "tangent_velocity_weight": 2.0,
            "face_handle_weight": 1.0,
            "reach_penalty_weight": 1.0,
            "overspeed_penalty_weight": 2.0,
        },
    )

    pull_target_reached_bonus = RewTerm(
        func=mdp.door_open_target_reached_bonus,
        weight=1.0,
        params={
            "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
            "door_closed_pos": 0.0,
            "door_open_sign": -1.0,
            "door_open_target": 0.8,
            "bonus": 25.0,
            "require_unlock_success_latch": True,
        },
    )

    release_handle_after_pull = RewTerm(
        func=mdp.release_handle_after_door_open,
        weight=2.0,
        params={
            "gripper_cfg": SceneEntityCfg("robot", body_names=["gripper_grasp_center"]),
            "handle_cfg": SceneEntityCfg("door", body_names=["handle_1"]),
            "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
            "handle_joint_cfg": SceneEntityCfg("door", joint_names=["handle_joint"]),
            "contact_sensor_name": "hook_contact",
            "door_closed_pos": 0.0,
            "door_open_sign": -1.0,
            "release_start_angle": 0.8,
            "contact_threshold": 0.2,
            "distance_saturation": 0.1,
            "handle_pressed_pos": -0.50,
            "handle_released_pos": 0.0,
            "handle_return_weight": 0.5,
        },
    )

    base_pull_traverse = RewTerm(
        func=mdp.base_pull_traverse_reward,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
            "activation_joint_pos": -0.8,
            "door_closed_pos": 0.0,
            "panel_rotation_sign": -1.0,
            "closed_panel_axis_d": (1.0, 0.0),
            "hinge_offset_d": (0.0, 0.0),
            "door_length": 1.2,
            "radius": 3.0,
            "safe_radius": 1.4,
            "z1_hold_steps": 3,
            "z1_transition_bonus": 25.0,
            "z2_transition_bonus": 60.0,
            "direct_z3_to_z2_penalty": 80.0,
            # Dense guidance: first round the frozen panel endpoint into Z1,
            # then move through the closed-frame doorway to the final pass point.
            "doorway_center_xy": DOORWAY_CENTER_XY,
            "doorway_forward_axis": (0.0, -1.0),
            "pass_distance": 0.9,
            "z1_waypoint_clearance": 0.40,
            "z1_progress_scale": 15.0,
            "z2_progress_scale": 20.0,
            "progress_step_clip": 0.03,
            "velocity_scale": 0.4,
            "velocity_reward_weight": 0.5,
            "stall_speed_threshold": 0.05,
            "stall_progress_threshold": 0.002,
            # At 50 Hz this gives roughly 0.5 s for release/posture adjustment.
            "stall_delay_steps": 25,
            "stall_penalty": 0.15,
        },
    )

    base_traverse_success_bonus = RewTerm(
        func=mdp.base_traverse_success_bonus,
        weight=30.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
            # The doorway frame is fixed to the closed door frame. Passing is
            # measured 0.9 m from its center along world/local -Y.
            "doorway_center_xy": DOORWAY_CENTER_XY,
            "doorway_forward_axis": (0.0, -1.0),
            "door_closed_pos": 0.0,
            "door_open_sign": -1.0,
            "required_door_angle": 0.8,
            "pass_distance": 0.9,
            "num_steps": 3,
            "bonus": 80.0,
            "exclude_pull_traverse_cheat": True,
            "require_pull_traverse_complete": True,
        },
    )

    bad_arm_pose_penalty = RewTerm(
        func=mdp.bad_arm_pose_penalty,
        weight=30.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "gripper_cfg": SceneEntityCfg("robot", body_names=["gripper_grasp_center"]),
            "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
            "doorway_center_xy": DOORWAY_CENTER_XY,
            "doorway_forward_axis": DOORWAY_FORWARD_AXIS,
            "pass_distance": 0.2,
            "bonus": -20.0,
        },
    )

    base_safety = RewTerm(
        func=mdp.base_safety_reward,
        weight=1.5,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
            "body_door_sensor_name": "body_door_contact",
            "leg_door_sensor_name": None,
            "body_frame_sensor_name": "body_door_frame_contact",
            "leg_frame_sensor_name": None,
            # Normalize contact penalties at the relaxed 50 N reference force.
            "force_ref": 50.0,
            "default_height": 0.43,
            "door_closed_pos": 0.0,
            "door_open_sign": -1.0,
            "stage3_start_angle": 0.10,
            "stage4_start_angle": 0.70,
            "early_body_door_weight": 1.0,
            "early_leg_door_weight": 0.5,
            "early_body_frame_weight": 1.0,
            "early_leg_frame_weight": 0.5,
            "late_body_door_weight": 0.5,
            "late_leg_door_weight": 0.2,
            "late_body_frame_weight": 0.5,
            "late_leg_frame_weight": 0.2,
            "cmd_rate_weight": 0.05,
            "height_pitch_weight": 2.0,
        },
    )


@configclass
class PullTerminationsCfg:
    time_out = DoneTerm(func=mdp_std.time_out, time_out=True)

    base_traverse_success = DoneTerm(
        func=mdp.base_traverse_success,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
            "doorway_center_xy": DOORWAY_CENTER_XY,
            "doorway_forward_axis": (0.0, -1.0),
            "door_closed_pos": 0.0,
            "door_open_sign": -1.0,
            "required_door_angle": 0.8,
            "pass_distance": 0.9,
            "num_steps": 3,
            "exclude_pull_traverse_cheat": True,
            "require_pull_traverse_complete": True,
        },
    )

    base_fall = DoneTerm(
        func=mdp.base_fall,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "min_base_height": 0.20,
            "max_base_height": 0.70,
        },
    )

    base_out_of_hinge_radius = DoneTerm(
        func=mdp.base_out_of_hinge_radius,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
            "door_closed_pos": 0.0,
            "panel_rotation_sign": -1.0,
            "closed_panel_axis_d": (1.0, 0.0),
            "hinge_offset_d": (0.0, 0.0),
            "door_length": 1.2,
            "radius": 3.0,
        },
    )

    bad_arm_pose = DoneTerm(
        func=mdp.bad_arm_pose,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "gripper_cfg": SceneEntityCfg("robot", body_names=["gripper_grasp_center"]),
            "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
            "doorway_center_xy": DOORWAY_CENTER_XY,
            "doorway_forward_axis": (0.0, -1.0),
            "pass_distance": 0.2,
        },
    )


@configclass
class PullDoorEnvCfg(ManagerBasedRLEnvCfg):
    """Pull-door environment with pull-follow shaping and task terminations."""

    scene: PullDoorEnvSceneCfg = PullDoorEnvSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: PullObservationsCfg = PullObservationsCfg()
    actions: PullActionsCfg = PullActionsCfg()
    events: PullEventCfg = PullEventCfg()

    rewards: PullRewardsCfg = PullRewardsCfg()
    terminations: PullTerminationsCfg = PullTerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 8
        self.episode_length_s = 20.0

        self.viewer.eye = (7.0, -0.5, 4.0)

        self.sim.dt = 0.0025
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 2**19

import isaaclab.envs.mdp as mdp_std

from .mit_actions import PiperHookMITJointActionCfg
from .hierarchical_actions import HighLevelDoorOpenActionCfg

# -------- rewards --------
from .rewards import (
    # helpers
    _body_pose_w,
    fingertip_mid_to_handle_distance,

    # arm posture after unlock
    arm_nominal_pose_after_unlock_l2,

    # shaping
    approach_handle_inv_square,
    align_grasp_pose_v2,

    # grasp + success
    grasp_handle_reward,
    grasp_handle_reward_preunlock_only,
    grasp_success_bonus,
    anti_release_after_press_to_open,

    #unlock progress
    press_handle_after_grasp_vel,
    stall_penalty_after_grasp_pos,
    near_unlock_stall_penalty,
    unlock_handle_progress_mixed,
    physical_unlock_transition_bonus,

    # push door progress after unlock
    push_door_progress_after_unlock,
    push_door_progress_after_unlock_success_only,
    release_handle_after_door_open,
    door_open_target_reached_bonus,
    pull_release_handle_reward,
    stage_gated_door_reward,
    base_to_pick_stance,
    ee_to_handle,
    ee_to_target_shaped,
    approach_progress,
    approach_reached_nearly,
    ee_handle_contact,
    ee_to_handle_shaped,
    align_hook_near_handle,
    ee_above_handle_target,
    grasping_success_shaped,
    grasp_stable_progress,

    # quadruped base rewards
    base_hold_reward,
    base_push_follow_reward,
    base_pull_follow_reward,
    base_pull_traverse_reward,
    compute_door_zone_masks,
    base_traverse_reward,
    base_safety_reward,
    body_door_collision_penalty,
    leg_door_collision_penalty,
    body_frame_collision_penalty,
    leg_frame_collision_penalty,
    base_traverse_success,
    base_traverse_success_bonus,
    bad_arm_pose_penalty,
)

# -------- terminations --------
from .terminations import (
    base_bad_orientation,
    base_fall,
    base_out_of_hinge_radius,
    bad_arm_pose,
)
# -------- stage ---------------
from .stage import (
    reset_root_state_to_default,
    staged_reset_from_archive,
)

# -------- observations --------
from .observations import (
    ee_pos_in_handle_frame,
    ee_quat_error_handle_frame,
    gripper_opening,
    ee_tcp_pose_w,
    ee_pos_in_noisy_handle_frame,
    ee_quat_error_noisy_handle_frame,
    finger_contact_norms,
    gripper_width,
    last_action,
    last_applied_arm_delta,
    arm_q_des_error,
    last_high_base_action,
    last_arm_action,
    high_base_command_3d,
    base_velocity_b,
    projected_gravity_b,
    base_height,
    base_to_doorway_center_b_xy,
    doorway_forward_axis_b_xy,
    handle_target_point_b,
    ee_to_handle_target_b,
    door_panel_forward_axis_b_xy,
    joint_position_scalar,
    joint_velocity_scalar,
    door_open_direction,
    pull_zone_one_hot,
    pull_traverse_state_one_hot,
    pull_traverse_started,
    pull_traverse_cheat_given,
    unlock_state,
    stage_id,
    body_door_contact_force_norm,
    leg_door_contact_force_norm,
    body_door_frame_contact_force_norm,
    leg_door_frame_contact_force_norm,
)

from ..low_level_locomotion.observations import (
    high_level_base_command,
    high_level_previous_action,
    low_level_last_action,
)

# -------- events --------
from .events import (
    update_door_lock_hysteresis_delayed_release,
    visualize_doorway_debug,
)

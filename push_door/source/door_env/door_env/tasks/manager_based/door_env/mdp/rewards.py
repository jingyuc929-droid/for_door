from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _physical_door_unlocked(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if hasattr(env, "_door_lock_mode"):
        return env._door_lock_mode == 2
    if hasattr(env, "_door_unlocked"):
        return env._door_unlocked.to(dtype=torch.bool)
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def arm_nominal_pose_after_unlock_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    door_joint_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    doorway_center_xy: tuple[float, float] = (0.0, 0.0),
    doorway_forward_axis: tuple[float, float] = (1.0, 0.0),
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
    strong_door_angle: float = 1.0,
    strong_pass_distance: float = 0.5,
    strong_scale: float = 200.0,
) -> torch.Tensor:
    """Penalize arm deviation with weak/strong post-unlock stages.

    The pre-unlock manipulation phase is intentionally left unconstrained so
    that reaching, hooking, and pressing the handle do not compete with a
    nominal-pose objective.  After unlock, the caller's base weight is used
    while both the door angle and traverse distance are below their thresholds.
    Crossing either threshold multiplies that weight by ``strong_scale``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    door: Articulation = env.scene[door_joint_cfg.name]
    joint_error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    penalty = torch.sum(torch.square(joint_error), dim=-1)

    door_joint_pos = door.data.joint_pos[:, door_joint_cfg.joint_ids[0]]
    door_angle = torch.clamp(
        float(door_open_sign) * (door_joint_pos - float(door_closed_pos)),
        min=0.0,
    )
    _, _, _, pass_distance, _, _ = _doorway_geometry(
        env,
        robot.data.root_pos_w[:, :2],
        doorway_center_xy,
        doorway_forward_axis,
        door_joint_cfg,
    )
    strong_stage = (door_angle >= float(strong_door_angle)) | (
        pass_distance >= float(strong_pass_distance)
    )
    scale = torch.where(
        strong_stage,
        torch.full_like(penalty, float(strong_scale)),
        torch.ones_like(penalty),
    )
    unlocked = _physical_door_unlocked(env).to(dtype=penalty.dtype)
    return penalty * scale * unlocked


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack((w, -x, -y, -z), dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector(s) v by quaternion(s) q (wxyz)."""
    qv = torch.cat((torch.zeros_like(v[..., :1]), v), dim=-1)
    return quat_mul(quat_mul(q, qv), quat_conjugate(q))[..., 1:]


def _body_pose_w(env: "ManagerBasedRLEnv", cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    asset: Articulation = env.scene[cfg.name]
    bid = cfg.body_ids[0]
    pos = asset.data.body_pos_w[:, bid, :]
    quat = asset.data.body_quat_w[:, bid, :]  # wxyz
    return pos, quat


# -----------------------------------------------------------------------------
# Geometry primitives
# -----------------------------------------------------------------------------


def fingertip_mid_to_handle_distance(
    env: "ManagerBasedRLEnv",
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    handle_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """|| (pL+pR)/2 - pHandle ||."""
    pL, _ = _body_pose_w(env, left_finger_cfg)
    pR, _ = _body_pose_w(env, right_finger_cfg)
    pTip = 0.5 * (pL + pR)
    pH, _ = _body_pose_w(env, handle_cfg)
    return torch.linalg.norm(pTip - pH, dim=-1)


def fingertip_mid_to_handle_grasp_point_distance(
    env: "ManagerBasedRLEnv",
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    handle_cfg: SceneEntityCfg,
    handle_offset_h: tuple[float, float, float] =  (-0.08, 0.04, -0.005),  # 在 handle frame 下的偏移
) -> torch.Tensor:
    # fingertips mid in world
    pL, _ = _body_pose_w(env, left_finger_cfg)
    pR, _ = _body_pose_w(env, right_finger_cfg)
    pTip = 0.5 * (pL + pR)

    # handle pose in world
    pH, qH = _body_pose_w(env, handle_cfg)

    # offset point in world: pG = pH + R_handle * offset_h
    off = torch.tensor(handle_offset_h, device=pH.device, dtype=pH.dtype).unsqueeze(0).repeat(pH.shape[0], 1)
    pG = pH + quat_rotate(qH, off)

    return torch.linalg.norm(pTip - pG, dim=-1)


def ee_tcp_to_handle_grasp_point_distance(
    env: "ManagerBasedRLEnv",
    hand_cfg: SceneEntityCfg,
    handle_cfg: SceneEntityCfg,
    ee_offset_pos: tuple[float, float, float] = (0.1523, 0.0, 0.0),
    handle_offset_h: tuple[float, float, float] = (-0.08, 0.04, -0.005),
) -> torch.Tensor:
    pHand, qHand = _body_pose_w(env, hand_cfg)
    pH, qH = _body_pose_w(env, handle_cfg)

    ee_off = torch.tensor(ee_offset_pos, device=pHand.device, dtype=pHand.dtype).unsqueeze(0).repeat(pHand.shape[0], 1)
    h_off = torch.tensor(handle_offset_h, device=pH.device, dtype=pH.dtype).unsqueeze(0).repeat(pH.shape[0], 1)

    pTcp = pHand + quat_rotate(qHand, ee_off)
    pG = pH + quat_rotate(qH, h_off)
    return torch.linalg.norm(pTcp - pG, dim=-1)


# -----------------------------------------------------------------------------
# Handle-only contact force (filtered) utilities
# Requires ContactSensorCfg(filter_prim_paths_expr=["{ENV_REGEX_NS}/Door/handle_1"])
# -----------------------------------------------------------------------------
def filtered_contact_force_vec_w(sensor: ContactSensor) -> torch.Tensor:
    """Filtered contact force vector in world frame, per env: [N,3].

    Uses sensor.data.force_matrix_w. We DO NOT fall back to net_forces_w to avoid self-collision noise.
    """
    fm = getattr(sensor.data, "force_matrix_w", None)
    if fm is None:
        return torch.zeros((sensor.data.net_forces_w.shape[0], 3), device=sensor.data.net_forces_w.device)

    # shapes commonly:
    #  [N, B, K, 3]  (B tracked bodies; K filtered prims)
    #  [N, K, 3]
    if fm.ndim == 4:
        vec = fm.sum(dim=2)  # [N, B, 3]
        # select the body with max norm (B usually 1)
        mag = torch.linalg.norm(vec, dim=-1)  # [N, B]
        idx = torch.argmax(mag, dim=1)
        return vec[torch.arange(vec.shape[0], device=vec.device), idx, :]
    if fm.ndim == 3:
        vec = fm.sum(dim=1)  # [N, 3]
        return vec

    raise RuntimeError(f"Unexpected force_matrix_w shape: {tuple(fm.shape)}")


#-----------------align grasp helpers-----------------
def filtered_contact_force_norm(sensor: ContactSensor) -> torch.Tensor:
    v = filtered_contact_force_vec_w(sensor)
    return torch.linalg.norm(v, dim=-1)


def _local_vec_to_world(q: torch.Tensor, v_local: tuple[float, float, float]) -> torch.Tensor:
    v = torch.tensor(v_local, device=q.device, dtype=q.dtype).unsqueeze(0).repeat(q.shape[0], 1)
    return quat_rotate(q, v)


def _axis_idx_to_local_vec(axis_idx: int, device, dtype) -> torch.Tensor:
    e = torch.zeros((1, 3), device=device, dtype=dtype)
    e[0, int(axis_idx)] = 1.0
    return e


def _handle_axis_world(qH: torch.Tensor, axis_idx: int) -> torch.Tensor:
    e = _axis_idx_to_local_vec(axis_idx, qH.device, qH.dtype).repeat(qH.shape[0], 1)
    return quat_rotate(qH, e)


def _safe_unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)


# -----------------------------------------------------------------------------
# ----------------Helper-------------------------------------------------------
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# (1) Approach + (2) Wrap alignment
# -----------------------------------------------------------------------------
def approach_handle_inv_square(
    env: "ManagerBasedRLEnv",
    hand_cfg: SceneEntityCfg,
    handle_cfg: SceneEntityCfg,
    ee_offset_pos: tuple[float, float, float] = (0.1523, 0.0, 0.0),
    handle_offset_h =  (-0.08, 0.04, -0.005),
    eps: float = 1e-4,
    scale: float = 0.10,
    clip: float = 1.0,
) -> torch.Tensor:
    d = ee_tcp_to_handle_grasp_point_distance(
        env,
        hand_cfg,
        handle_cfg,
        ee_offset_pos=ee_offset_pos,
        handle_offset_h=handle_offset_h,
    )
    r = scale / (d * d + eps)
    return torch.clamp(r, max=clip)

def align_grasp_pose_v2_terms(
    env: "ManagerBasedRLEnv",
    handle_cfg: SceneEntityCfg,
    hand_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg | None = None,
    right_finger_cfg: SceneEntityCfg | None = None,
    hook_approach_axis_hand: tuple[float, float, float] = (1.0, 0.0, 0.0),
    hook_mouth_axis_hand: tuple[float, float, float] = (0.0, 1.0, 0.0),
    handle_approach_axis: int = 1,
    expected_approach_sign: float = 1.0,
    world_down_axis: tuple[float, float, float] = (0.0, 0.0, -1.0),
    approach_weight: float = 0.70,
    mouth_down_weight: float = 0.30,
    # Deprecated two-finger parameters are accepted for config compatibility.
    grasp_axis: int = 2,
    min_sep: float = 0.010,
    sep_scale: float = 0.010,
    symmetry_scale: float = 0.015,
    gripper_open_axis_hand: tuple[float, float, float] | None = None,
    gripper_approach_axis_hand: tuple[float, float, float] | None = None,
    side_weight: float = 0.0,
    open_weight: float = 0.0,
):
    _, qH = _body_pose_w(env, handle_cfg)
    _, qG = _body_pose_w(env, hand_cfg)

    hook_approach_w = _safe_unit(_local_vec_to_world(qG, hook_approach_axis_hand))
    h_app_w = _safe_unit(_handle_axis_world(qH, handle_approach_axis))
    app_dot = torch.sum(hook_approach_w * h_app_w, dim=-1)
    approach_align = (float(expected_approach_sign) * app_dot).clamp(0.0, 1.0)

    hook_mouth_w = _safe_unit(_local_vec_to_world(qG, hook_mouth_axis_hand))
    world_down = torch.tensor(world_down_axis, device=qG.device, dtype=qG.dtype).unsqueeze(0).repeat(qG.shape[0], 1)
    world_down = _safe_unit(world_down)
    mouth_dot = torch.sum(hook_mouth_w * world_down, dim=-1)
    mouth_down_align = mouth_dot.clamp(0.0, 1.0)

    wsum = float(approach_weight + mouth_down_weight)
    score = (
        float(approach_weight) * approach_align
        + float(mouth_down_weight) * mouth_down_align
    ) / max(wsum, 1e-6)
    score = score.clamp(0.0, 1.0)

    zero = torch.zeros_like(score)
    one = torch.ones_like(score)
    return {
        "score": score,
        "side_score": one,
        "open_align": mouth_down_align,
        "approach_align": approach_align,
        "mouth_down_align": mouth_down_align,
        "open_dot": mouth_dot,
        "mouth_dot": mouth_dot,
        "approach_dot": app_dot,
        "l": zero,
        "r": zero,
        "sep": zero,
        "sym_err": zero,
    }

def align_grasp_pose_v2(
    env: "ManagerBasedRLEnv",
    handle_cfg: SceneEntityCfg,
    hand_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg | None = None,
    right_finger_cfg: SceneEntityCfg | None = None,
    hook_approach_axis_hand: tuple[float, float, float] = (1.0, 0.0, 0.0),
    hook_mouth_axis_hand: tuple[float, float, float] = (0.0, 1.0, 0.0),
    handle_approach_axis: int = 1,
    expected_approach_sign: float = 1.0,
    world_down_axis: tuple[float, float, float] = (0.0, 0.0, -1.0),
    approach_weight: float = 0.70,
    mouth_down_weight: float = 0.30,
    # Deprecated two-finger parameters are accepted for config compatibility.
    grasp_axis: int = 2,
    min_sep: float = 0.010,
    sep_scale: float = 0.010,
    symmetry_scale: float = 0.015,
    gripper_open_axis_hand: tuple[float, float, float] | None = None,
    gripper_approach_axis_hand: tuple[float, float, float] | None = None,
    side_weight: float = 0.0,
    open_weight: float = 0.0,
) -> torch.Tensor:
    terms = align_grasp_pose_v2_terms(
        env=env,
        handle_cfg=handle_cfg,
        hand_cfg=hand_cfg,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
        hook_approach_axis_hand=hook_approach_axis_hand,
        hook_mouth_axis_hand=hook_mouth_axis_hand,
        handle_approach_axis=handle_approach_axis,
        expected_approach_sign=expected_approach_sign,
        world_down_axis=world_down_axis,
        approach_weight=approach_weight,
        mouth_down_weight=mouth_down_weight,
        grasp_axis=grasp_axis,
        min_sep=min_sep,
        sep_scale=sep_scale,
        symmetry_scale=symmetry_scale,
        gripper_open_axis_hand=gripper_open_axis_hand,
        gripper_approach_axis_hand=gripper_approach_axis_hand,
        side_weight=side_weight,
        open_weight=open_weight,
    )
    return terms["score"]


# -----------------------------------------------------------------------------
# (3) Hook grasp reward: grasp-center geometry + hook/handle contact
# -----------------------------------------------------------------------------
def grasp_handle_reward(
    env: "ManagerBasedRLEnv",
    handle_cfg: SceneEntityCfg,
    hand_cfg: SceneEntityCfg,
    contact_sensor_name: str = "hook_contact",
    left_finger_cfg: SceneEntityCfg | None = None,
    right_finger_cfg: SceneEntityCfg | None = None,
    gripper_cfg: SceneEntityCfg | None = None,
    left_sensor_name: str = "left_finger_contact",
    right_sensor_name: str = "right_finger_contact",
    # gates
    distance_threshold: float = 0.08,
    ee_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    handle_offset_h =  (-0.08, 0.04, -0.005),
    hook_approach_axis_hand: tuple[float, float, float] = (1.0, 0.0, 0.0),
    hook_mouth_axis_hand: tuple[float, float, float] = (0.0, 1.0, 0.0),
    handle_approach_axis: int = 1,
    expected_approach_sign: float = 1.0,
    world_down_axis: tuple[float, float, float] = (0.0, 0.0, -1.0),
    approach_weight: float = 0.70,
    mouth_down_weight: float = 0.30,
    align_threshold: float = 0.30,
    # force shaping
    force_threshold: float = 0.25,
    force_scale: float = 2.0,
    # anti-hacking
    hold_steps: int = 6,
    hold_tau: float | None = None,
    hold_power: float = 1.0,
    hold_decay: float = 0.0,
    # Deprecated two-finger params are accepted for config compatibility.
    grasp_axis: int = 2,
    min_sep: float = 0.0050,
    sep_scale: float = 0.0010,
    symmetry_scale: float = 0.015,
    gripper_open_axis_hand: tuple[float, float, float] | None = None,
    gripper_approach_axis_hand: tuple[float, float, float] | None = None,
    align_side_weight: float = 0.0,
    align_open_weight: float = 0.0,
    align_approach_weight: float = 0.0,
    side_scale: float = 10.0,
    side_weight: float = 0.0,
    open_width: float = 0.09,
    min_closedness: float = 0.0,
    close_scale: float = 0.0,
    close_power: float = 1.0,
    balance_eps: float = 1e-6,
    balance_power: float = 1.0,
    finger_speed_std: float = 0.05,
) -> torch.Tensor:
    dist = ee_tcp_to_handle_grasp_point_distance(
        env,
        hand_cfg,
        handle_cfg,
        ee_offset_pos=ee_offset_pos,
        handle_offset_h=handle_offset_h,
    )
    near_ok = dist < float(distance_threshold)
    near_score = torch.exp(-torch.square(dist / float(max(distance_threshold, 1e-6))))

    align = align_grasp_pose_v2(
        env=env,
        handle_cfg=handle_cfg,
        hand_cfg=hand_cfg,
        hook_approach_axis_hand=hook_approach_axis_hand,
        hook_mouth_axis_hand=hook_mouth_axis_hand,
        handle_approach_axis=handle_approach_axis,
        expected_approach_sign=expected_approach_sign,
        world_down_axis=world_down_axis,
        approach_weight=approach_weight,
        mouth_down_weight=mouth_down_weight,
    )
    align_ok = align >= float(align_threshold)

    force = filtered_contact_force_norm(env.scene[contact_sensor_name])
    contact_ok = force > float(force_threshold)
    contact_score = torch.tanh(torch.clamp(force - float(force_threshold), min=0.0) / float(max(force_scale, 1e-6)))

    gate = near_ok & align_ok & contact_ok

    N = dist.shape[0]
    device = dist.device
    if (not hasattr(env, "_grasp_hold_counter")) or (env._grasp_hold_counter.shape[0] != N):
        env._grasp_hold_counter = torch.zeros(N, device=device, dtype=torch.int32)
    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf == 0
        env._grasp_hold_counter = torch.where(reset_mask, torch.zeros_like(env._grasp_hold_counter), env._grasp_hold_counter)

    if hold_decay > 0.0:
        decayed = (env._grasp_hold_counter.float() * float(hold_decay)).to(torch.int32)
        env._grasp_hold_counter = torch.where(gate, env._grasp_hold_counter + 1, decayed)
    else:
        env._grasp_hold_counter = torch.where(gate, env._grasp_hold_counter + 1, torch.zeros_like(env._grasp_hold_counter))

    tau = float(hold_tau) if hold_tau is not None else float(max(1.0, hold_steps / 2.0))
    hold_prog = 1.0 - torch.exp(-env._grasp_hold_counter.float() / tau)
    hold_prog = torch.clamp(hold_prog, 0.0, 1.0) ** float(hold_power)

    rew = hold_prog * contact_score
    rew = rew * (0.2 + 0.8 * near_score.clamp(0.0, 1.0))
    rew = rew * (0.2 + 0.8 * align.clamp(0.0, 1.0))

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["hook_grasp/contact_force_mean"] = force.mean().detach()
    env.extras["log"]["hook_grasp/contact_ok_ratio"] = contact_ok.float().mean().detach()
    env.extras["log"]["hook_grasp/near_ok_ratio"] = near_ok.float().mean().detach()
    env.extras["log"]["hook_grasp/align_mean"] = align.mean().detach()
    return rew

def grasp_handle_reward_preunlock_only(
    env: "ManagerBasedRLEnv",
    # new control params
    handle_joint_cfg: SceneEntityCfg,
    unlock_enter_pos: float = -0.03,
    fade_width: float = 0.02,
    less_than: bool = True,

    # --- original grasp_handle_reward params ---
    handle_cfg: SceneEntityCfg = SceneEntityCfg("door", body_names=["handle_1"]),
    handle_offset_h: tuple[float, float, float] = (-0.08, 0.04, -0.005),
    left_finger_cfg: SceneEntityCfg | None = None,
    right_finger_cfg: SceneEntityCfg | None = None,
    hand_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper_grasp_center"]),
    gripper_cfg: SceneEntityCfg | None = None,
    contact_sensor_name: str = "hook_contact",
    left_sensor_name: str = "left_finger_contact",
    right_sensor_name: str = "right_finger_contact",

    distance_threshold: float = 0.10,
    ee_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    grasp_axis: int = 2,
    min_sep: float = 0.005,
    sep_scale: float = 0.010,
    symmetry_scale: float = 0.015,
    gripper_open_axis_hand: tuple[float, float, float] | None = None,
    gripper_approach_axis_hand: tuple[float, float, float] | None = None,
    hook_approach_axis_hand: tuple[float, float, float] = (1.0, 0.0, 0.0),
    hook_mouth_axis_hand: tuple[float, float, float] = (0.0, 1.0, 0.0),
    handle_approach_axis: int = 1,
    expected_approach_sign: float = 1.0,
    world_down_axis: tuple[float, float, float] = (0.0, 0.0, -1.0),
    approach_weight: float = 0.70,
    mouth_down_weight: float = 0.30,
    align_side_weight: float = 0.70,
    align_open_weight: float = 0.10,
    align_approach_weight: float = 0.20,
    align_threshold: float = 0.30,

    force_threshold: float = 0.5,
    force_scale: float = 10.0,
    side_scale: float = 10.0,
    side_weight: float = 0.3,

    open_width: float = 0.09,
    min_closedness: float = 0.3,
    close_scale: float = 1.0,
    close_power: float = 2.0,

    hold_steps: int = 2,
    hold_tau: float = 2.5,
    hold_power: float = 1.0,
    hold_decay: float = 0.6,
    balance_power: float = 2.0,
    finger_speed_std: float = 0.08,
) -> torch.Tensor:
    """
    Use grasp_handle_reward only before the press/unlock phase.
    Once handle_pos goes sufficiently into press direction, fade this reward out.
    """

    base = grasp_handle_reward(
        env=env,
        handle_cfg=handle_cfg,
        handle_offset_h=handle_offset_h,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
        hand_cfg=hand_cfg,
        gripper_cfg=gripper_cfg,
        contact_sensor_name=contact_sensor_name,
        left_sensor_name=left_sensor_name,
        right_sensor_name=right_sensor_name,
        distance_threshold=distance_threshold,
        ee_offset_pos=ee_offset_pos,
        grasp_axis=grasp_axis,
        min_sep=min_sep,
        sep_scale=sep_scale,
        symmetry_scale=symmetry_scale,
        gripper_open_axis_hand=gripper_open_axis_hand,
        gripper_approach_axis_hand=gripper_approach_axis_hand,
        hook_approach_axis_hand=hook_approach_axis_hand,
        hook_mouth_axis_hand=hook_mouth_axis_hand,
        handle_approach_axis=handle_approach_axis,
        expected_approach_sign=expected_approach_sign,
        world_down_axis=world_down_axis,
        approach_weight=approach_weight,
        mouth_down_weight=mouth_down_weight,
        align_side_weight=align_side_weight,
        align_open_weight=align_open_weight,
        align_approach_weight=align_approach_weight,
        align_threshold=align_threshold,
        force_threshold=force_threshold,
        force_scale=force_scale,
        side_scale=side_scale,
        side_weight=side_weight,
        open_width=open_width,
        min_closedness=min_closedness,
        close_scale=close_scale,
        close_power=close_power,
        hold_steps=hold_steps,
        hold_tau=hold_tau,
        hold_power=hold_power,
        hold_decay=hold_decay,
        balance_power=balance_power,
        finger_speed_std=finger_speed_std,
    )

    door: Articulation = env.scene[handle_joint_cfg.name]
    jids = handle_joint_cfg.joint_ids
    handle_pos = (
        door.data.joint_pos[:, jids[0]]
        if len(jids) == 1
        else door.data.joint_pos[:, jids].mean(dim=-1)
    )

    if less_than:
        # e.g. unlock_enter_pos=-0.03, fade_width=0.02:
        # handle_pos >= -0.03 => scale=1
        # handle_pos <= -0.05 => scale=0
        fade_end = float(unlock_enter_pos) - float(fade_width)
        scale = torch.clamp(
            (handle_pos - fade_end) / float(max(fade_width, 1e-6)),
            0.0, 1.0
        )
    else:
        fade_end = float(unlock_enter_pos) + float(fade_width)
        scale = torch.clamp(
            (fade_end - handle_pos) / float(max(fade_width, 1e-6)),
            0.0, 1.0
        )

    return base * scale


# -----------------------------------------------------------------------------
# (5) Sparse success bonus (one-shot per episode)
# -----------------------------------------------------------------------------
def grasp_success_bonus(
    env: "ManagerBasedRLEnv",
    handle_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg | None = None,
    right_finger_cfg: SceneEntityCfg | None = None,
    hand_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper_grasp_center"]),
    gripper_cfg: SceneEntityCfg | None = None,
    contact_sensor_name: str = "hook_contact",
    left_sensor_name: str = "left_finger_contact",
    right_sensor_name: str = "right_finger_contact",
    distance_threshold: float = 0.1,
    ee_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    force_threshold: float = 0.25,
    hold_steps: int = 5,
    open_width: float = 0.09,
    min_closedness: float = 0.5,
    bonus: float = 20.0,
    require_wrap: bool = True,
    require_any_finger_contact: bool = False,
    use_force_norm: bool = True,
    near_mode: str = "min_finger",
    use_grasp_point: bool = True,
    handle_offset_h: tuple[float, float, float] =  (-0.08, 0.04, -0.005),
    archive_cap: int = 512,
    handle_joint_cfg: SceneEntityCfg | None = None,
    relax_near_after_handle_pos: float = -0.05,
    less_than: bool = True,
    # --- hook pose gate ---
    hook_approach_axis_hand: tuple[float, float, float] = (1.0, 0.0, 0.0),
    hook_mouth_axis_hand: tuple[float, float, float] = (0.0, 1.0, 0.0),
    expected_approach_sign: float = 1.0,
    world_down_axis: tuple[float, float, float] = (0.0, 0.0, -1.0),
    approach_weight: float = 0.70,
    mouth_down_weight: float = 0.30,
    # Deprecated two-finger params are accepted for config compatibility.
    grasp_axis: int = 2,
    min_sep: float = 0.010,
    sep_scale: float = 0.010,
    symmetry_scale: float = 0.015,
    gripper_open_axis_hand: tuple[float, float, float] = (0.0, 1.0, 0.0),
    gripper_approach_axis_hand: tuple[float, float, float] = (1.0, 0.0, 0.0),
    handle_approach_axis: int = 1,
    align_side_weight: float = 0.55,
    align_open_weight: float = 0.25,
    align_approach_weight: float = 0.20,
    align_threshold: float = 0.30,
    require_ee_above_target: bool = False,
    minimum_height_delta: float = 0.0,
) -> torch.Tensor:
    dist = ee_tcp_to_handle_grasp_point_distance(
        env,
        hand_cfg,
        handle_cfg,
        ee_offset_pos=ee_offset_pos,
        handle_offset_h=handle_offset_h,
    )

    near_ok = dist < float(distance_threshold)

    if require_wrap:
        align = align_grasp_pose_v2(
            env=env,
            handle_cfg=handle_cfg,
            hand_cfg=hand_cfg,
            hook_approach_axis_hand=hook_approach_axis_hand,
            hook_mouth_axis_hand=hook_mouth_axis_hand,
            handle_approach_axis=handle_approach_axis,
            expected_approach_sign=expected_approach_sign,
            world_down_axis=world_down_axis,
            approach_weight=approach_weight,
            mouth_down_weight=mouth_down_weight,
        )
        align_ok = align >= float(align_threshold)
    else:
        align = torch.ones_like(dist)
        align_ok = torch.ones_like(near_ok, dtype=torch.bool)

    force = filtered_contact_force_norm(env.scene[contact_sensor_name])
    contact_ok = force > float(force_threshold)

    if require_ee_above_target:
        p_hand, q_hand = _body_pose_w(env, hand_cfg)
        p_handle, q_handle = _body_pose_w(env, handle_cfg)
        ee_offset = torch.as_tensor(ee_offset_pos, device=p_hand.device, dtype=p_hand.dtype).view(1, 3)
        handle_offset = torch.as_tensor(handle_offset_h, device=p_handle.device, dtype=p_handle.dtype).view(1, 3)
        p_ee = p_hand + quat_rotate(q_hand, ee_offset.expand_as(p_hand))
        p_target = p_handle + quat_rotate(q_handle, handle_offset.expand_as(p_handle))
        height_ok = (p_ee[:, 2] - p_target[:, 2]) >= float(minimum_height_delta)
    else:
        height_ok = torch.ones_like(near_ok, dtype=torch.bool)

    if handle_joint_cfg is not None:
        door: Articulation = env.scene[handle_joint_cfg.name]
        jids = handle_joint_cfg.joint_ids
        handle_pos = door.data.joint_pos[:, jids[0]] if len(jids) == 1 else door.data.joint_pos[:, jids].mean(dim=-1)

        if less_than:
            relax_phase = handle_pos < float(relax_near_after_handle_pos)
        else:
            relax_phase = handle_pos > float(relax_near_after_handle_pos)
    else:
        handle_pos = torch.zeros_like(dist)
        relax_phase = torch.zeros_like(near_ok, dtype=torch.bool)

    gate_strict = near_ok & align_ok & contact_ok
    gate_relaxed = align_ok & contact_ok
    gate = torch.where(relax_phase, gate_relaxed, gate_strict) & height_ok

    N = dist.shape[0]
    device = dist.device
    if (not hasattr(env, "_grasp_success_counter")) or (env._grasp_success_counter.shape[0] != N):
        env._grasp_success_counter = torch.zeros(N, device=device, dtype=torch.int32)
    if (not hasattr(env, "_grasp_success_given")) or (env._grasp_success_given.shape[0] != N):
        env._grasp_success_given = torch.zeros(N, device=device, dtype=torch.bool)

    if not hasattr(env, "_grasp_success_init_cleared"):
        env._grasp_success_counter.zero_()
        env._grasp_success_given.zero_()
        env._grasp_success_init_cleared = True

    reset_mask = None
    if hasattr(env, "reset_buf"):
        try:
            reset_mask = env.reset_buf.to(dtype=torch.bool)
        except Exception:
            reset_mask = None

    if reset_mask is None and hasattr(env, "episode_length_buf"):
        if (not hasattr(env, "_prev_ep_len_grasp_success")) or (env._prev_ep_len_grasp_success.shape[0] != N):
            env._prev_ep_len_grasp_success = env.episode_length_buf.clone()
        reset_mask = (env.episode_length_buf < env._prev_ep_len_grasp_success) | (env.episode_length_buf == 0)
        env._prev_ep_len_grasp_success = env.episode_length_buf.clone()

    if reset_mask is not None:
        env._grasp_success_counter = torch.where(reset_mask, torch.zeros_like(env._grasp_success_counter), env._grasp_success_counter)
        env._grasp_success_given = torch.where(reset_mask, torch.zeros_like(env._grasp_success_given), env._grasp_success_given)

    env._grasp_success_counter = torch.where(gate, env._grasp_success_counter + 1, torch.zeros_like(env._grasp_success_counter))
    success_now = env._grasp_success_counter >= int(max(1, hold_steps))

    give = success_now & (~env._grasp_success_given)
    env._grasp_success_given = env._grasp_success_given | give

    idx = torch.nonzero(give).squeeze(-1)
    if idx.numel() > 0:
        from .stage import push_archive_from_env
        push_archive_from_env(
            env=env,
            name="_archive_grasp",
            env_ids=idx,
            cap=archive_cap,
            robot_cfg=SceneEntityCfg("robot"),
            door_cfg=SceneEntityCfg("door"),
            store_unlock_flag=False,
        )

    return give.float() * float(bonus)

# ---------------------encourage press handle---------------
def press_handle_after_grasp_vel(
    env: "ManagerBasedRLEnv",
    handle_joint_cfg: SceneEntityCfg,
    hand_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper_grasp_center"]),
    handle_cfg: SceneEntityCfg | None = None,
    gripper_cfg: SceneEntityCfg | None = None,
    contact_sensor_name: str = "hook_contact",
    left_sensor_name: str = "left_finger_contact",
    right_sensor_name: str = "right_finger_contact",
    contact_threshold: float = 0.25,
    require_any_contact: bool = True,
    open_width: float = 0.08,
    min_closedness: float = 0.35,
    distance_threshold: float = 0.12,
    ee_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    handle_offset_h: tuple[float, float, float] = (-0.08, 0.04, -0.005),
    hook_approach_axis_hand: tuple[float, float, float] = (1.0, 0.0, 0.0),
    hook_mouth_axis_hand: tuple[float, float, float] = (0.0, 1.0, 0.0),
    handle_approach_axis: int = 1,
    expected_approach_sign: float = 1.0,
    world_down_axis: tuple[float, float, float] = (0.0, 0.0, -1.0),
    approach_weight: float = 0.70,
    mouth_down_weight: float = 0.30,
    align_threshold: float = 0.30,
    # 方向/抗抖
    less_than: bool = True,
    vel_deadzone: float = 0.01,
    vel_scale: float = 0.05,
    opposite_penalty: float = 0.2,
    clip: float = 1.0,
    # --- NEW: anti-spike by requiring net position progress ---
    pos_deadzone: float = 1e-4,      # 忽略极小位移（数值噪声）
    pos_scale: float = 2e-3,         # Δpos -> [0,1] 的平滑尺度（越小越“硬门控”）
    use_vel_ema: bool = True,
    vel_ema_alpha: float = 0.25,     # EMA 平滑系数（0~1）
) -> torch.Tensor:
    # 只在本 episode 已经 grasp_success 后激活
    if not hasattr(env, "_grasp_success_given"):
        return torch.zeros(env.num_envs, device=env.device)
    phase = env._grasp_success_given  # [N] bool

    door: Articulation = env.scene[handle_joint_cfg.name]
    jids = handle_joint_cfg.joint_ids
    if len(jids) == 1:
        jid = jids[0]
        handle_pos = door.data.joint_pos[:, jid]
        handle_vel_raw = door.data.joint_vel[:, jid]
    else:
        handle_pos = door.data.joint_pos[:, jids].mean(dim=-1)
        handle_vel_raw = door.data.joint_vel[:, jids].mean(dim=-1)

    force = filtered_contact_force_norm(env.scene[contact_sensor_name])
    contact_ok = force > float(contact_threshold)
    if handle_cfg is not None:
        dist = ee_tcp_to_handle_grasp_point_distance(
            env,
            hand_cfg,
            handle_cfg,
            ee_offset_pos=ee_offset_pos,
            handle_offset_h=handle_offset_h,
        )
        near_ok = dist < float(distance_threshold)
        align = align_grasp_pose_v2(
            env=env,
            handle_cfg=handle_cfg,
            hand_cfg=hand_cfg,
            hook_approach_axis_hand=hook_approach_axis_hand,
            hook_mouth_axis_hand=hook_mouth_axis_hand,
            handle_approach_axis=handle_approach_axis,
            expected_approach_sign=expected_approach_sign,
            world_down_axis=world_down_axis,
            approach_weight=approach_weight,
            mouth_down_weight=mouth_down_weight,
        )
        align_ok = align >= float(align_threshold)
    else:
        near_ok = torch.ones_like(contact_ok, dtype=torch.bool)
        align_ok = torch.ones_like(contact_ok, dtype=torch.bool)
    gate = contact_ok & near_ok & align_ok

    # --- robust reset detection for buffers ---
    N = handle_pos.shape[0]
    reset_mask = None
    if hasattr(env, "reset_buf"):
        try:
            reset_mask = env.reset_buf.to(dtype=torch.bool)
        except Exception:
            reset_mask = None
    if reset_mask is None and hasattr(env, "episode_length_buf"):
        if (not hasattr(env, "_prev_ep_len_press")) or (env._prev_ep_len_press.shape[0] != N):
            env._prev_ep_len_press = env.episode_length_buf.clone()
        reset_mask = (env.episode_length_buf < env._prev_ep_len_press) | (env.episode_length_buf == 0)
        env._prev_ep_len_press = env.episode_length_buf.clone()

    # --- cache prev handle_pos to measure net progress ---
    if (not hasattr(env, "_press_prev_handle_pos")) or (env._press_prev_handle_pos.shape[0] != N):
        env._press_prev_handle_pos = handle_pos.clone()
    if reset_mask is not None:
        env._press_prev_handle_pos = torch.where(reset_mask, handle_pos, env._press_prev_handle_pos)

    # Δpos in desired direction (positive means progressing toward unlock)
    # less_than=True means handle_pos should decrease (become more negative)
    dpos = (env._press_prev_handle_pos - handle_pos) if less_than else (handle_pos - env._press_prev_handle_pos)
    env._press_prev_handle_pos = handle_pos.clone()

    # smooth progress gate in [0,1] (0 if no net progress)
    prog_gate = torch.tanh(torch.clamp(dpos - float(pos_deadzone), min=0.0) / float(pos_scale))

    # --- EMA smooth velocity to reduce impulsive spikes ---
    if use_vel_ema:
        if (not hasattr(env, "_press_vel_ema")) or (env._press_vel_ema.shape[0] != N):
            env._press_vel_ema = torch.zeros_like(handle_vel_raw)
        if reset_mask is not None:
            env._press_vel_ema = torch.where(reset_mask, torch.zeros_like(env._press_vel_ema), env._press_vel_ema)
        env._press_vel_ema = (1.0 - float(vel_ema_alpha)) * env._press_vel_ema + float(vel_ema_alpha) * handle_vel_raw
        handle_vel = env._press_vel_ema
    else:
        handle_vel = handle_vel_raw

    # velocity shaping (same as before)
    desired = (-handle_vel) if less_than else (handle_vel)
    pos = torch.clamp(desired - float(vel_deadzone), min=0.0)
    neg = torch.clamp(-desired - float(vel_deadzone), min=0.0)

    # KEY FIX:
    #  - positive reward is multiplied by prog_gate (needs net position progress)
    #  - negative (wrong-direction) still penalized to discourage oscillation spikes
    r_pos = torch.tanh(pos / float(vel_scale)) * prog_gate
    r_neg = torch.tanh(neg / float(vel_scale))
    r = r_pos - float(opposite_penalty) * r_neg

    r = torch.clamp(r, -float(clip), float(clip))
    return (phase & gate).float() * r

def stall_penalty_after_grasp_pos(
    env: "ManagerBasedRLEnv",
    handle_joint_cfg: SceneEntityCfg,
    hand_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper_grasp_center"]),
    handle_cfg: SceneEntityCfg | None = None,
    contact_sensor_name: str = "hook_contact",
    left_sensor_name: str = "left_finger_contact",
    right_sensor_name: str = "right_finger_contact",
    contact_threshold: float = 0.25,
    require_any_contact: bool = False,
    distance_threshold: float = 0.12,
    ee_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    handle_offset_h: tuple[float, float, float] = (-0.08, 0.04, -0.005),
    hook_approach_axis_hand: tuple[float, float, float] = (1.0, 0.0, 0.0),
    hook_mouth_axis_hand: tuple[float, float, float] = (0.0, 1.0, 0.0),
    handle_approach_axis: int = 1,
    expected_approach_sign: float = 1.0,
    world_down_axis: tuple[float, float, float] = (0.0, 0.0, -1.0),
    approach_weight: float = 0.70,
    mouth_down_weight: float = 0.30,
    align_threshold: float = 0.30,
    stall_pos: float = -0.10,
    pos_scale: float = 0.03,
    penalty: float = 0.02,
    recent_window_steps: int = 200,
    grace_steps: int = 10,
    less_than: bool = True,
) -> torch.Tensor:
    if not hasattr(env, "_grasp_success_given"):
        return torch.zeros(env.num_envs, device=env.device)

    N = env.num_envs
    device = env.device

    reset_mask = None
    if hasattr(env, "reset_buf"):
        try:
            reset_mask = env.reset_buf.to(dtype=torch.bool)
        except Exception:
            reset_mask = None

    if reset_mask is None and hasattr(env, "episode_length_buf"):
        if (not hasattr(env, "_prev_ep_len_stall_pos")) or (env._prev_ep_len_stall_pos.shape[0] != N):
            env._prev_ep_len_stall_pos = env.episode_length_buf.clone()
        reset_mask = (env.episode_length_buf < env._prev_ep_len_stall_pos) | (env.episode_length_buf == 0)
        env._prev_ep_len_stall_pos = env.episode_length_buf.clone()

    if (not hasattr(env, "_grasp_recent_ttl")) or (env._grasp_recent_ttl.shape[0] != N):
        env._grasp_recent_ttl = torch.zeros(N, device=device, dtype=torch.int32)
    if (not hasattr(env, "_grasp_recent_grace")) or (env._grasp_recent_grace.shape[0] != N):
        env._grasp_recent_grace = torch.zeros(N, device=device, dtype=torch.int32)
    if (not hasattr(env, "_prev_grasp_success_given")) or (env._prev_grasp_success_given.shape[0] != N):
        env._prev_grasp_success_given = env._grasp_success_given.clone()

    if reset_mask is not None:
        env._grasp_recent_ttl = torch.where(reset_mask, torch.zeros_like(env._grasp_recent_ttl), env._grasp_recent_ttl)
        env._grasp_recent_grace = torch.where(reset_mask, torch.zeros_like(env._grasp_recent_grace), env._grasp_recent_grace)
        env._prev_grasp_success_given = torch.where(reset_mask, torch.zeros_like(env._prev_grasp_success_given), env._prev_grasp_success_given)

    newly = env._grasp_success_given & (~env._prev_grasp_success_given)
    env._prev_grasp_success_given = env._grasp_success_given.clone()

    env._grasp_recent_ttl = torch.where(
        newly,
        torch.full_like(env._grasp_recent_ttl, int(recent_window_steps)),
        torch.clamp(env._grasp_recent_ttl - 1, min=0),
    )
    env._grasp_recent_grace = torch.where(
        newly,
        torch.full_like(env._grasp_recent_grace, int(grace_steps)),
        torch.clamp(env._grasp_recent_grace - 1, min=0),
    )

    phase = env._grasp_recent_ttl > 0
    grace_ok = env._grasp_recent_grace == 0

    force = filtered_contact_force_norm(env.scene[contact_sensor_name])
    contact_ok = force > float(contact_threshold)
    if handle_cfg is not None:
        dist = ee_tcp_to_handle_grasp_point_distance(
            env,
            hand_cfg,
            handle_cfg,
            ee_offset_pos=ee_offset_pos,
            handle_offset_h=handle_offset_h,
        )
        near_ok = dist < float(distance_threshold)
        align = align_grasp_pose_v2(
            env=env,
            handle_cfg=handle_cfg,
            hand_cfg=hand_cfg,
            hook_approach_axis_hand=hook_approach_axis_hand,
            hook_mouth_axis_hand=hook_mouth_axis_hand,
            handle_approach_axis=handle_approach_axis,
            expected_approach_sign=expected_approach_sign,
            world_down_axis=world_down_axis,
            approach_weight=approach_weight,
            mouth_down_weight=mouth_down_weight,
        )
        align_ok = align >= float(align_threshold)
    else:
        near_ok = torch.ones_like(contact_ok, dtype=torch.bool)
        align_ok = torch.ones_like(contact_ok, dtype=torch.bool)

    door: Articulation = env.scene[handle_joint_cfg.name]
    jids = handle_joint_cfg.joint_ids
    handle_pos = door.data.joint_pos[:, jids[0]] if len(jids) == 1 else door.data.joint_pos[:, jids].mean(dim=-1)

    if less_than:
        bad_depth = torch.clamp((handle_pos - float(stall_pos)) / float(pos_scale), min=0.0)
    else:
        bad_depth = torch.clamp((float(stall_pos) - handle_pos) / float(pos_scale), min=0.0)

    bad = phase & grace_ok & contact_ok & near_ok & align_ok
    return -float(penalty) * bad.float() * torch.tanh(bad_depth)

def anti_release_after_press_to_open(
    env: "ManagerBasedRLEnv",
    handle_joint_cfg: SceneEntityCfg,
    door_joint_cfg: SceneEntityCfg,
    gripper_cfg: SceneEntityCfg | None = None,
    hand_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper_grasp_center"]),
    handle_cfg: SceneEntityCfg | None = None,
    left_finger_cfg: SceneEntityCfg | None = None,
    right_finger_cfg: SceneEntityCfg | None = None,
    contact_sensor_name: str = "hook_contact",
    left_sensor_name: str = "left_finger_contact",
    right_sensor_name: str = "right_finger_contact",
    # --- keep gate ---
    contact_threshold: float = 0.3,
    require_any_contact: bool = False,
    open_width: float = 0.09,
    min_closedness: float = 0.25,
    distance_threshold: float = 0.13,
    ee_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    handle_offset_h: tuple[float, float, float] = (-0.08, 0.04, -0.005),
    hook_approach_axis_hand: tuple[float, float, float] = (1.0, 0.0, 0.0),
    hook_mouth_axis_hand: tuple[float, float, float] = (0.0, 1.0, 0.0),
    handle_approach_axis: int = 1,
    expected_approach_sign: float = 1.0,
    world_down_axis: tuple[float, float, float] = (0.0, 0.0, -1.0),
    approach_weight: float = 0.70,
    mouth_down_weight: float = 0.30,
    align_threshold: float = 0.30,
    # --- phase start ---
    handle_start_pos: float = 0.0,
    handle_threshold: float = -0.3,
    activate_progress: float = 0.25,
    use_unlock_success_latch: bool = True,
    # --- door open definition ---
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
    push_enter_open: float = 0.02,
    door_open_threshold: float = 0.35,
    # --- transition window only ---
    max_keep_steps_after_unlock: int = 8,
    keep_until_door_open: bool = True,
    # --- reward scales ---
    hold_reward: float = 0.005,
    progress_boost: float = 0.03,
    release_event_penalty: float = 0.20,
    lost_penalty: float = 0.01,
    auto_open_penalty: float = 0.05,
) -> torch.Tensor:
    N = env.num_envs
    device = env.device

    if not hasattr(env, "_grasp_success_given"):
        return torch.zeros(N, device=device)

    # ------------------------------------------------------------
    # 1) keep gate = hook/handle contact + grasp-center near target + hook pose
    # ------------------------------------------------------------
    force = filtered_contact_force_norm(env.scene[contact_sensor_name])
    contact_ok = force > float(contact_threshold)

    if handle_cfg is not None:
        dist = ee_tcp_to_handle_grasp_point_distance(
            env,
            hand_cfg,
            handle_cfg,
            ee_offset_pos=ee_offset_pos,
            handle_offset_h=handle_offset_h,
        )
        near_ok = dist < float(distance_threshold)
        align = align_grasp_pose_v2(
            env=env,
            handle_cfg=handle_cfg,
            hand_cfg=hand_cfg,
            hook_approach_axis_hand=hook_approach_axis_hand,
            hook_mouth_axis_hand=hook_mouth_axis_hand,
            handle_approach_axis=handle_approach_axis,
            expected_approach_sign=expected_approach_sign,
            world_down_axis=world_down_axis,
            approach_weight=approach_weight,
            mouth_down_weight=mouth_down_weight,
        )
        align_ok = align >= float(align_threshold)
    else:
        dist = torch.zeros(N, device=device)
        near_ok = torch.ones(N, dtype=torch.bool, device=device)
        align_ok = torch.ones(N, dtype=torch.bool, device=device)
    keep_ok = contact_ok & near_ok & align_ok

    # ------------------------------------------------------------
    # 2) handle progress
    # ------------------------------------------------------------
    handle_asset: Articulation = env.scene[handle_joint_cfg.name]
    hjids = handle_joint_cfg.joint_ids
    handle_pos = handle_asset.data.joint_pos[:, hjids[0]] if len(hjids) == 1 else handle_asset.data.joint_pos[:, hjids].mean(dim=-1)

    denom = float(handle_threshold - handle_start_pos)
    if abs(denom) < 1e-9:
        handle_prog = torch.zeros_like(handle_pos)
    else:
        handle_prog = (handle_pos - float(handle_start_pos)) / denom
        handle_prog = torch.clamp(handle_prog, 0.0, 1.0)

    # ------------------------------------------------------------
    # 3) door open
    # ------------------------------------------------------------
    door_asset: Articulation = env.scene[door_joint_cfg.name]
    djids = door_joint_cfg.joint_ids
    door_pos = door_asset.data.joint_pos[:, djids[0]] if len(djids) == 1 else door_asset.data.joint_pos[:, djids].mean(dim=-1)

    door_open = float(door_open_sign) * (door_pos - float(door_closed_pos))
    door_open = torch.clamp(door_open, min=0.0)

    # ------------------------------------------------------------
    # 4) reset detection
    # ------------------------------------------------------------
    reset_mask = None
    if hasattr(env, "reset_buf"):
        try:
            reset_mask = env.reset_buf.to(dtype=torch.bool)
        except Exception:
            reset_mask = None
    if reset_mask is None and hasattr(env, "episode_length_buf"):
        if (not hasattr(env, "_prev_ep_len_keep_transition")) or (env._prev_ep_len_keep_transition.shape[0] != N):
            env._prev_ep_len_keep_transition = env.episode_length_buf.clone()
        reset_mask = (env.episode_length_buf < env._prev_ep_len_keep_transition) | (env.episode_length_buf == 0)
        env._prev_ep_len_keep_transition = env.episode_length_buf.clone()

    # ------------------------------------------------------------
    # 5) unlock/open phase latch
    # ------------------------------------------------------------
    unlock_latch = torch.zeros(N, dtype=torch.bool, device=device)
    if use_unlock_success_latch and hasattr(env, "_unlock_success_given"):
        unlock_latch = env._unlock_success_given

    phase_raw = env._grasp_success_given & (
        (handle_prog > float(activate_progress))
        | unlock_latch
        | (door_open > float(push_enter_open))
    )

    if (not hasattr(env, "_keep_transition_phase")) or (env._keep_transition_phase.shape[0] != N):
        env._keep_transition_phase = torch.zeros(N, dtype=torch.bool, device=device)
    if (not hasattr(env, "_keep_transition_ttl")) or (env._keep_transition_ttl.shape[0] != N):
        env._keep_transition_ttl = torch.zeros(N, dtype=torch.int32, device=device)
    if (not hasattr(env, "_keep_transition_prev_keep_ok")) or (env._keep_transition_prev_keep_ok.shape[0] != N):
        env._keep_transition_prev_keep_ok = torch.zeros(N, dtype=torch.bool, device=device)

    if reset_mask is not None:
        env._keep_transition_phase = torch.where(reset_mask, torch.zeros_like(env._keep_transition_phase), env._keep_transition_phase)
        env._keep_transition_ttl = torch.where(reset_mask, torch.zeros_like(env._keep_transition_ttl), env._keep_transition_ttl)
        env._keep_transition_prev_keep_ok = torch.where(reset_mask, torch.zeros_like(env._keep_transition_prev_keep_ok), env._keep_transition_prev_keep_ok)

    newly = phase_raw & (~env._keep_transition_phase)
    env._keep_transition_phase = env._keep_transition_phase | phase_raw
    env._keep_transition_ttl = torch.where(
        newly,
        torch.full_like(env._keep_transition_ttl, int(max_keep_steps_after_unlock)),
        torch.clamp(env._keep_transition_ttl - 1, min=0),
    )

    phase = env._keep_transition_ttl > 0
    if keep_until_door_open:
        phase = phase & (door_open < float(door_open_threshold))

    # ------------------------------------------------------------
    # 6) release event inside short window only
    # ------------------------------------------------------------
    prev_keep = env._keep_transition_prev_keep_ok
    env._keep_transition_prev_keep_ok = phase & keep_ok

    release_event = phase & prev_keep & (~keep_ok)
    lost = phase & (~keep_ok)

    # ------------------------------------------------------------
    # 7) reward
    # ------------------------------------------------------------
    # only weak positive reward in the transition window
    hold_rew = (phase & keep_ok).float() * (float(hold_reward) + float(progress_boost) * handle_prog)

    event_pen = float(release_event_penalty) * release_event.float()
    lost_pen = float(lost_penalty) * lost.float()

    # if released and door still moves in this short window, penalize a bit
    auto_open = phase & (~keep_ok) & (door_open > float(push_enter_open))
    auto_pen = float(auto_open_penalty) * auto_open.float()

    return hold_rew - event_pen - lost_pen - auto_pen



def unlock_handle_progress_mixed(
    env: "ManagerBasedRLEnv",
    handle_joint_cfg: SceneEntityCfg,
    gripper_cfg: SceneEntityCfg | None = None,
    hand_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper_grasp_center"]),
    handle_cfg: SceneEntityCfg | None = None,
    contact_sensor_name: str = "hook_contact",
    left_sensor_name: str = "left_finger_contact",
    right_sensor_name: str = "right_finger_contact",
    # --- gate ---
    contact_threshold: float = 0.25,
    require_any_contact: bool = True,
    open_width: float = 0.09,
    min_closedness: float = 0.20,
    require_grasp_success: bool = False,
    distance_threshold: float = 0.12,
    ee_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    handle_offset_h: tuple[float, float, float] = (-0.08, 0.04, -0.005),
    hook_approach_axis_hand: tuple[float, float, float] = (1.0, 0.0, 0.0),
    hook_mouth_axis_hand: tuple[float, float, float] = (0.0, 1.0, 0.0),
    handle_approach_axis: int = 1,
    expected_approach_sign: float = 1.0,
    world_down_axis: tuple[float, float, float] = (0.0, 0.0, -1.0),
    approach_weight: float = 0.70,
    mouth_down_weight: float = 0.30,
    align_threshold: float = 0.30,
    # --- progress definition ---
    handle_start_pos: float = 0.0,
    reward_stop_pos: float = -0.30,
    # --- delta branch: still encourage "go deeper" ---
    delta_power: float = 1.2,
    ema_alpha: float = 0.7,
    deadzone: float = 5e-5,
    backtrack_penalty: float = 0.01,
    delta_gain: float = 1.2,
    # --- absolute branch: deep is always worth more ---
    abs_power: float = 1.8,
    abs_gain: float = 0.35,
    # --- hold branch: once deep enough, tiny-motion deep press should still score ---
    hold_start_ratio: float = 0.35,   # from here on, give deep-hold reward
    hold_power: float = 1.6,
    hold_gain: float = 0.25,
    # --- final clip ---
    clip: float = 2.0,
) -> torch.Tensor:
    """
    Mixed unlock reward:
      reward = gate * (
          delta_gain * delta_progress_reward
        + abs_gain   * absolute_depth_reward
        + hold_gain  * deep_hold_reward
      )

    - delta_progress_reward:
        rewards newly achieved deeper handle progress, using EMA + backtrack penalty.
    - absolute_depth_reward:
        rewards current depth itself; deeper is worth more.
    - deep_hold_reward:
        once the handle is already in a sufficiently deep region, keep a stable positive reward
        even if the instantaneous motion becomes very small.
    """

    # ------------------------------------------------------------
    # 1) read handle joint pos
    # ------------------------------------------------------------
    door: Articulation = env.scene[handle_joint_cfg.name]
    jids = handle_joint_cfg.joint_ids
    handle_pos = (
        door.data.joint_pos[:, jids[0]]
        if len(jids) == 1
        else door.data.joint_pos[:, jids].mean(dim=-1)
    )

    # ------------------------------------------------------------
    # 2) normalize progress to [0, 1]
    # ------------------------------------------------------------
    denom = float(reward_stop_pos - handle_start_pos)
    if abs(denom) < 1e-9:
        prog = torch.zeros_like(handle_pos)
    else:
        prog = (handle_pos - float(handle_start_pos)) / denom
        prog = torch.clamp(prog, 0.0, 1.0)

    # ------------------------------------------------------------
    # 3) gate: hook/handle contact + grasp-center near target + hook pose
    # ------------------------------------------------------------
    force = filtered_contact_force_norm(env.scene[contact_sensor_name])
    contact_ok = force > float(contact_threshold)
    if handle_cfg is not None:
        dist = ee_tcp_to_handle_grasp_point_distance(
            env,
            hand_cfg,
            handle_cfg,
            ee_offset_pos=ee_offset_pos,
            handle_offset_h=handle_offset_h,
        )
        near_ok = dist < float(distance_threshold)
        align = align_grasp_pose_v2(
            env=env,
            handle_cfg=handle_cfg,
            hand_cfg=hand_cfg,
            hook_approach_axis_hand=hook_approach_axis_hand,
            hook_mouth_axis_hand=hook_mouth_axis_hand,
            handle_approach_axis=handle_approach_axis,
            expected_approach_sign=expected_approach_sign,
            world_down_axis=world_down_axis,
            approach_weight=approach_weight,
            mouth_down_weight=mouth_down_weight,
        )
        align_ok = align >= float(align_threshold)
    else:
        near_ok = torch.ones_like(contact_ok, dtype=torch.bool)
        align_ok = torch.ones_like(contact_ok, dtype=torch.bool)

    if require_grasp_success:
        if hasattr(env, "_grasp_success_given"):
            grasp_ok = env._grasp_success_given
        else:
            grasp_ok = torch.zeros_like(contact_ok, dtype=torch.bool)
    else:
        grasp_ok = torch.ones_like(contact_ok, dtype=torch.bool)

    gate = contact_ok & near_ok & align_ok & grasp_ok

    # ------------------------------------------------------------
    # 4) robust reset detection
    # ------------------------------------------------------------
    N = handle_pos.shape[0]
    device = handle_pos.device

    reset_mask = None
    if hasattr(env, "reset_buf"):
        try:
            reset_mask = env.reset_buf.to(dtype=torch.bool)
        except Exception:
            reset_mask = None

    if reset_mask is None and hasattr(env, "episode_length_buf"):
        if (not hasattr(env, "_prev_ep_len_unlock_mixed")) or (env._prev_ep_len_unlock_mixed.shape[0] != N):
            env._prev_ep_len_unlock_mixed = env.episode_length_buf.clone()
        reset_mask = (env.episode_length_buf < env._prev_ep_len_unlock_mixed) | (env.episode_length_buf == 0)
        env._prev_ep_len_unlock_mixed = env.episode_length_buf.clone()

    # ------------------------------------------------------------
    # 5) EMA buffer for delta branch
    # ------------------------------------------------------------
    if (not hasattr(env, "_unlock_prog_ema")) or (env._unlock_prog_ema.shape[0] != N):
        env._unlock_prog_ema = torch.zeros(N, device=device, dtype=prog.dtype)

    if reset_mask is not None:
        env._unlock_prog_ema = torch.where(reset_mask, prog.detach(), env._unlock_prog_ema)

    prev = env._unlock_prog_ema
    new = (1.0 - float(ema_alpha)) * prev + float(ema_alpha) * prog
    env._unlock_prog_ema = new

    # ------------------------------------------------------------
    # 6) delta branch
    # ------------------------------------------------------------
    shaped_prev = torch.clamp(prev, 0.0, 1.0) ** float(delta_power)
    shaped_new = torch.clamp(new, 0.0, 1.0) ** float(delta_power)

    pos = torch.clamp(shaped_new - shaped_prev - float(deadzone), min=0.0)
    neg = torch.clamp(shaped_prev - shaped_new - float(deadzone), min=0.0)
    delta_rew = pos - float(backtrack_penalty) * neg

    # ------------------------------------------------------------
    # 7) absolute branch
    # ------------------------------------------------------------
    abs_rew = torch.clamp(prog, 0.0, 1.0) ** float(abs_power)

    # ------------------------------------------------------------
    # 8) deep-hold branch
    #    once progress is already reasonably deep, keep a stable positive reward
    # ------------------------------------------------------------
    hold_ratio = torch.clamp((prog - float(hold_start_ratio)) / float(max(1e-6, 1.0 - hold_start_ratio)), 0.0, 1.0)
    hold_rew = hold_ratio ** float(hold_power)

    # ------------------------------------------------------------
    # 9) final mixed reward
    # ------------------------------------------------------------
    rew = (
        float(delta_gain) * delta_rew
        + float(abs_gain) * abs_rew
        + float(hold_gain) * hold_rew
    )

    rew = gate.float() * torch.clamp(rew, -float(clip), float(clip))
    return rew



def push_door_progress_after_unlock(
    env: "ManagerBasedRLEnv",
    door_joint_cfg: SceneEntityCfg,
    # --- unlock gate ---
    handle_joint_cfg: SceneEntityCfg | None = None,
    handle_threshold: float = -0.3,
    handle_less_than: bool = True,
    use_unlock_success_latch: bool = True,
    # --- door progress definition ---
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
    door_open_target: float = 0.35,
    # --- reward scales ---
    delta_scale: float = 1.0,
    abs_scale: float = 0.2,
    # --- anti-jitter ---
    ema_alpha: float = 0.25,
    deadzone: float = 1e-4,
    backtrack_penalty: float = 0.2,
    clip: float = 1.0,
) -> torch.Tensor:
    # ------------------------------------------------------------------
    # 1) read door joint position
    # ------------------------------------------------------------------
    door: Articulation = env.scene[door_joint_cfg.name]
    jids = door_joint_cfg.joint_ids
    door_pos = door.data.joint_pos[:, jids[0]] if len(jids) == 1 else door.data.joint_pos[:, jids].mean(dim=-1)

    door_open = float(door_open_sign) * (door_pos - float(door_closed_pos))
    door_open = torch.clamp(door_open, min=0.0)

    # ------------------------------------------------------------------
    # 2) unlock phase gate only
    # ------------------------------------------------------------------
    if use_unlock_success_latch and hasattr(env, "_unlock_success_given"):
        unlocked_phase = env._unlock_success_given
    else:
        if handle_joint_cfg is None:
            unlocked_phase = torch.ones_like(door_open, dtype=torch.bool)
        else:
            handle_asset: Articulation = env.scene[handle_joint_cfg.name]
            hjids = handle_joint_cfg.joint_ids
            handle_pos = handle_asset.data.joint_pos[:, hjids[0]] if len(hjids) == 1 else handle_asset.data.joint_pos[:, hjids].mean(dim=-1)
            unlocked_phase = (handle_pos < handle_threshold) if handle_less_than else (handle_pos > handle_threshold)

    # ------------------------------------------------------------------
    # 3) reset detection
    # ------------------------------------------------------------------
    N = door_open.shape[0]
    device = door_open.device
    dtype = door_open.dtype

    reset_mask = None
    if hasattr(env, "reset_buf"):
        try:
            reset_mask = env.reset_buf.to(dtype=torch.bool)
        except Exception:
            reset_mask = None

    if reset_mask is None and hasattr(env, "episode_length_buf"):
        if (not hasattr(env, "_prev_ep_len_open_door_free")) or (env._prev_ep_len_open_door_free.shape[0] != N):
            env._prev_ep_len_open_door_free = env.episode_length_buf.clone()
        reset_mask = (env.episode_length_buf < env._prev_ep_len_open_door_free) | (env.episode_length_buf == 0)
        env._prev_ep_len_open_door_free = env.episode_length_buf.clone()

    # ------------------------------------------------------------------
    # 4) EMA buffer
    # ------------------------------------------------------------------
    if (not hasattr(env, "_door_open_ema_free")) or (env._door_open_ema_free.shape[0] != N):
        env._door_open_ema_free = torch.zeros(N, device=device, dtype=dtype)

    if reset_mask is not None:
        env._door_open_ema_free = torch.where(reset_mask, door_open.detach(), env._door_open_ema_free)

    prev = env._door_open_ema_free
    new = (1.0 - float(ema_alpha)) * prev + float(ema_alpha) * door_open
    env._door_open_ema_free = new

    d = new - prev
    pos = torch.clamp(d - float(deadzone), min=0.0)
    neg = torch.clamp(-d - float(deadzone), min=0.0)

    delta_rew = pos - float(backtrack_penalty) * neg
    delta_rew = torch.clamp(delta_rew, -float(clip), float(clip))

    abs_open = torch.clamp(new / float(max(1e-6, door_open_target)), 0.0, 1.0)

    rew = float(delta_scale) * delta_rew + float(abs_scale) * abs_open
    return unlocked_phase.float() * rew


def near_unlock_stall_penalty(
    env: "ManagerBasedRLEnv",
    handle_joint_cfg: SceneEntityCfg,
    door_joint_cfg: SceneEntityCfg,
    enter_depth: float = 0.25,
    exit_depth: float = 0.22,
    grace_steps: int = 60,
    ramp_steps: int = 60,
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
    door_progress_threshold: float = 0.01,
    max_penalty: float = 1.0,
    less_than: bool = True,
) -> torch.Tensor:
    door: Articulation = env.scene[handle_joint_cfg.name]
    hjids = handle_joint_cfg.joint_ids
    handle_pos = door.data.joint_pos[:, hjids[0]] if len(hjids) == 1 else door.data.joint_pos[:, hjids].mean(dim=-1)

    djids = door_joint_cfg.joint_ids
    door_pos = door.data.joint_pos[:, djids[0]] if len(djids) == 1 else door.data.joint_pos[:, djids].mean(dim=-1)
    door_open = float(door_open_sign) * (door_pos - float(door_closed_pos))
    door_open = torch.clamp(door_open, min=0.0)

    N = handle_pos.shape[0]
    device = handle_pos.device

    if hasattr(env, "_grasp_success_given"):
        grasp_ok = env._grasp_success_given.to(dtype=torch.bool)
    else:
        grasp_ok = torch.zeros(N, dtype=torch.bool, device=device)
    physical_unlocked = _physical_door_unlocked(env)

    press_depth = -handle_pos if less_than else handle_pos
    enter = press_depth >= float(enter_depth)
    remain = press_depth >= float(exit_depth)

    if (not hasattr(env, "_near_unlock_stall_active")) or (env._near_unlock_stall_active.shape[0] != N):
        env._near_unlock_stall_active = torch.zeros(N, dtype=torch.bool, device=device)
    if (not hasattr(env, "_near_unlock_stall_counter")) or (env._near_unlock_stall_counter.shape[0] != N):
        env._near_unlock_stall_counter = torch.zeros(N, dtype=torch.int32, device=device)

    reset_mask = None
    if hasattr(env, "reset_buf"):
        try:
            reset_mask = env.reset_buf.to(dtype=torch.bool)
        except Exception:
            reset_mask = None
    if reset_mask is None and hasattr(env, "episode_length_buf"):
        if (not hasattr(env, "_prev_ep_len_near_unlock_stall")) or (
            env._prev_ep_len_near_unlock_stall.shape[0] != N
        ):
            env._prev_ep_len_near_unlock_stall = env.episode_length_buf.clone()
        reset_mask = (
            (env.episode_length_buf < env._prev_ep_len_near_unlock_stall)
            | (env.episode_length_buf == 0)
        )
        env._prev_ep_len_near_unlock_stall = env.episode_length_buf.clone()

    active = grasp_ok & (~physical_unlocked) & (enter | (env._near_unlock_stall_active & remain))

    if reset_mask is not None:
        active = active & (~reset_mask)
        env._near_unlock_stall_active = torch.where(
            reset_mask, torch.zeros_like(env._near_unlock_stall_active), active
        )
        env._near_unlock_stall_counter = torch.where(
            reset_mask, torch.zeros_like(env._near_unlock_stall_counter), env._near_unlock_stall_counter
        )
    else:
        env._near_unlock_stall_active = active

    clear_counter = physical_unlocked | (~active)
    env._near_unlock_stall_counter = torch.where(
        clear_counter,
        torch.zeros_like(env._near_unlock_stall_counter),
        env._near_unlock_stall_counter + 1,
    )

    door_not_opening = door_open < float(door_progress_threshold)
    over_steps = torch.clamp(env._near_unlock_stall_counter.float() - float(grace_steps), min=0.0)
    stall_scale = torch.tanh(over_steps / float(max(int(ramp_steps), 1)))
    penalty = -float(max_penalty) * stall_scale * active.float() * door_not_opening.float()

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["stall_after_press/deep_press_active_ratio"] = active.float().mean().detach()
    env.extras["log"]["stall_after_press/counter_mean"] = env._near_unlock_stall_counter.float().mean().detach()
    env.extras["log"]["stall_after_press/counter_max"] = env._near_unlock_stall_counter.float().max().detach()
    env.extras["log"]["stall_after_press/penalty_mean"] = penalty.mean().detach()
    env.extras["log"]["stall_after_press/penalty_active_ratio"] = (penalty < 0.0).float().mean().detach()
    env.extras["log"]["handle_debug/handle_pos_mean"] = handle_pos.mean().detach()
    env.extras["log"]["handle_debug/handle_pos_min"] = handle_pos.min().detach()
    env.extras["log"]["handle_debug/near_unlock_ratio_025_030"] = (
        ((handle_pos <= -0.25) & (handle_pos > -0.30)).float().mean().detach()
    )
    env.extras["log"]["handle_debug/cross_unlock_threshold_ratio"] = (handle_pos <= -0.30).float().mean().detach()

    return penalty


def physical_unlock_transition_bonus(
    env: "ManagerBasedRLEnv",
    bonus: float = 10.0,
) -> torch.Tensor:
    physical_unlocked = _physical_door_unlocked(env)
    N = physical_unlocked.shape[0]
    device = physical_unlocked.device

    if (not hasattr(env, "_unlock_success_given")) or (env._unlock_success_given.shape[0] != N):
        env._unlock_success_given = torch.zeros(N, dtype=torch.bool, device=device)

    newly_unlocked = physical_unlocked & (~env._unlock_success_given.to(dtype=torch.bool))
    reward = newly_unlocked.float() * float(bonus)
    env._unlock_success_given = env._unlock_success_given | newly_unlocked

    idx = torch.nonzero(newly_unlocked).squeeze(-1)
    if idx.numel() > 0:
        from .stage import push_archive_from_env
        push_archive_from_env(
            env=env,
            name="_archive_unlock",
            env_ids=idx,
            cap=512,
            robot_cfg=SceneEntityCfg("robot"),
            door_cfg=SceneEntityCfg("door"),
            store_unlock_flag=True,
        )

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["transition/physical_unlocked_ratio"] = physical_unlocked.float().mean().detach()
    env.extras["log"]["transition/newly_unlocked_ratio"] = newly_unlocked.float().mean().detach()
    env.extras["log"]["transition/unlock_bonus_mean"] = reward.mean().detach()

    return reward


def push_door_progress_after_unlock_success_only(
    env: "ManagerBasedRLEnv",
    door_joint_cfg: SceneEntityCfg,
    # 只在 unlock_success 之后生效
    require_unlock_success_latch: bool = True,
    # gate: hook/handle contact + grasp-center near target + hook pose
    gripper_cfg: SceneEntityCfg | None = None,
    hand_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper_grasp_center"]),
    handle_cfg: SceneEntityCfg | None = None,
    contact_sensor_name: str = "hook_contact",
    left_sensor_name: str = "left_finger_contact",
    right_sensor_name: str = "right_finger_contact",
    contact_threshold: float = 0.25,
    require_any_contact: bool = True,
    open_width: float = 0.09,
    min_closedness: float = 0.50,
    require_gate: bool = True,
    distance_threshold: float = 0.14,
    ee_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    handle_offset_h: tuple[float, float, float] = (-0.08, 0.04, -0.005),
    hook_approach_axis_hand: tuple[float, float, float] = (1.0, 0.0, 0.0),
    hook_mouth_axis_hand: tuple[float, float, float] = (0.0, 1.0, 0.0),
    handle_approach_axis: int = 1,
    expected_approach_sign: float = 1.0,
    world_down_axis: tuple[float, float, float] = (0.0, 0.0, -1.0),
    approach_weight: float = 0.70,
    mouth_down_weight: float = 0.30,
    align_threshold: float = 0.20,
    # 门开度定义
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
    door_open_target: float = 0.35,
    # 奖励
    delta_scale: float = 1.0,
    abs_scale: float = 0.2,
    # 平滑/防抖
    ema_alpha: float = 0.25,
    deadzone: float = 1e-4,
    backtrack_penalty: float = 0.2,
    use_best_open_progress: bool = False,
    rollback_scale: float = 0.10,
    clip: float = 1.0,
    premature_release_penalty: float = 0.0,
) -> torch.Tensor:
    door: Articulation = env.scene[door_joint_cfg.name]
    jids = door_joint_cfg.joint_ids
    door_pos = door.data.joint_pos[:, jids[0]] if len(jids) == 1 else door.data.joint_pos[:, jids].mean(dim=-1)

    door_open = float(door_open_sign) * (door_pos - float(door_closed_pos))
    door_open = torch.clamp(door_open, min=0.0)

    N = door_open.shape[0]
    device = door_open.device
    dtype = door_open.dtype

    # -----------------------------
    # gate: ONLY after unlock_success
    # -----------------------------
    if require_unlock_success_latch:
        if hasattr(env, "_unlock_success_given"):
            unlock_gate = env._unlock_success_given.to(dtype=torch.bool)
        else:
            unlock_gate = torch.zeros(N, dtype=torch.bool, device=device)
    else:
        unlock_gate = torch.ones(N, dtype=torch.bool, device=device)

    gate = unlock_gate
    contact_ok = torch.ones(N, dtype=torch.bool, device=device)

    if require_gate:
        force = filtered_contact_force_norm(env.scene[contact_sensor_name])
        contact_ok = force >= float(contact_threshold)
        if handle_cfg is not None:
            dist = ee_tcp_to_handle_grasp_point_distance(
                env,
                hand_cfg,
                handle_cfg,
                ee_offset_pos=ee_offset_pos,
                handle_offset_h=handle_offset_h,
            )
            near_ok = dist < float(distance_threshold)
            align = align_grasp_pose_v2(
                env=env,
                handle_cfg=handle_cfg,
                hand_cfg=hand_cfg,
                hook_approach_axis_hand=hook_approach_axis_hand,
                hook_mouth_axis_hand=hook_mouth_axis_hand,
                handle_approach_axis=handle_approach_axis,
                expected_approach_sign=expected_approach_sign,
                world_down_axis=world_down_axis,
                approach_weight=approach_weight,
                mouth_down_weight=mouth_down_weight,
            )
            align_ok = align >= float(align_threshold)
        else:
            near_ok = torch.ones_like(contact_ok, dtype=torch.bool)
            align_ok = torch.ones_like(contact_ok, dtype=torch.bool)
        gate = gate & contact_ok & near_ok & align_ok

    # -----------------------------
    # robust reset detection
    # -----------------------------
    reset_mask = None
    if hasattr(env, "reset_buf"):
        try:
            reset_mask = env.reset_buf.to(dtype=torch.bool)
        except Exception:
            reset_mask = None

    if reset_mask is None and hasattr(env, "episode_length_buf"):
        if (not hasattr(env, "_prev_ep_len_open_after_unlock_success")) or (
            env._prev_ep_len_open_after_unlock_success.shape[0] != N
        ):
            env._prev_ep_len_open_after_unlock_success = env.episode_length_buf.clone()
        reset_mask = (
            (env.episode_length_buf < env._prev_ep_len_open_after_unlock_success)
            | (env.episode_length_buf == 0)
        )
        env._prev_ep_len_open_after_unlock_success = env.episode_length_buf.clone()

    # Penalize the contact-loss event before reaching the target.  Distance
    # and alignment are deliberately excluded: they gate positive shaping but
    # do not by themselves count as releasing the handle.
    if (not hasattr(env, "_door_pull_prev_contact_ok")) or (
        env._door_pull_prev_contact_ok.shape[0] != N
    ):
        env._door_pull_prev_contact_ok = torch.zeros(N, dtype=torch.bool, device=device)

    if reset_mask is not None:
        env._door_pull_prev_contact_ok = torch.where(
            reset_mask,
            torch.zeros_like(env._door_pull_prev_contact_ok),
            env._door_pull_prev_contact_ok,
        )

    active_contact = unlock_gate & contact_ok
    premature_release = (
        unlock_gate
        & env._door_pull_prev_contact_ok
        & (~contact_ok)
        & (door_open < float(door_open_target))
    )
    if reset_mask is not None:
        premature_release = premature_release & (~reset_mask)
    env._door_pull_prev_contact_ok = active_contact

    # -----------------------------
    # Monotonic best-open progress + EMA buffer
    # -----------------------------
    reward_open = torch.clamp(door_open, max=float(door_open_target))
    if (not hasattr(env, "_door_best_open_after_unlock_success")) or (
        env._door_best_open_after_unlock_success.shape[0] != N
    ):
        env._door_best_open_after_unlock_success = reward_open.detach().clone()

    if reset_mask is not None:
        env._door_best_open_after_unlock_success = torch.where(
            reset_mask,
            reward_open.detach(),
            env._door_best_open_after_unlock_success,
        )

    if use_best_open_progress:
        # Only gate-valid opening can establish a new record. Closing and then
        # reopening to an old angle therefore cannot replay positive progress.
        record_candidate = torch.where(
            gate, reward_open.detach(), env._door_best_open_after_unlock_success
        )
        env._door_best_open_after_unlock_success = torch.maximum(
            env._door_best_open_after_unlock_success, record_candidate
        )
        progress_open = env._door_best_open_after_unlock_success
        rollback_cost = torch.clamp(
            torch.relu(progress_open - reward_open)
            / max(float(rollback_scale), 1.0e-6),
            0.0,
            1.0,
        )
    else:
        progress_open = reward_open
        rollback_cost = torch.zeros_like(door_open)

    if (not hasattr(env, "_door_open_ema_after_unlock_success")) or (
        env._door_open_ema_after_unlock_success.shape[0] != N
    ):
        env._door_open_ema_after_unlock_success = torch.zeros(N, device=device, dtype=dtype)

    if reset_mask is not None:
        env._door_open_ema_after_unlock_success = torch.where(
            reset_mask,
            torch.clamp(door_open.detach(), max=float(door_open_target)),
            env._door_open_ema_after_unlock_success,
        )

    prev = env._door_open_ema_after_unlock_success
    new = (1.0 - float(ema_alpha)) * prev + float(ema_alpha) * progress_open
    env._door_open_ema_after_unlock_success = new

    d = new - prev
    pos = torch.clamp(d - float(deadzone), min=0.0)
    neg = torch.clamp(-d - float(deadzone), min=0.0)

    delta_rew = pos - float(backtrack_penalty) * (rollback_cost if use_best_open_progress else neg)
    delta_rew = torch.clamp(delta_rew, -float(clip), float(clip))

    abs_open = torch.clamp(new / float(max(1e-6, door_open_target)), 0.0, 1.0)

    rew = float(delta_scale) * delta_rew + float(abs_scale) * abs_open
    release_penalty = float(premature_release_penalty) * premature_release.float()

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["pull_door/gate_ratio"] = gate.float().mean().detach()
    env.extras["log"]["pull_door/contact_ratio"] = contact_ok.float().mean().detach()
    env.extras["log"]["pull_door/premature_release_ratio"] = premature_release.float().mean().detach()
    env.extras["log"]["pull_door/best_open_mean"] = (
        env._door_best_open_after_unlock_success.mean().detach()
    )
    env.extras["log"]["pull_door/rollback_cost_mean"] = rollback_cost.mean().detach()

    return gate.float() * rew - release_penalty


def door_open_target_reached_bonus(
    env: "ManagerBasedRLEnv",
    door_joint_cfg: SceneEntityCfg,
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
    door_open_target: float = 0.8,
    bonus: float = 25.0,
    require_unlock_success_latch: bool = True,
) -> torch.Tensor:
    """Give one reward when the door first reaches its opening target."""
    door: Articulation = env.scene[door_joint_cfg.name]
    jids = door_joint_cfg.joint_ids
    door_pos = (
        door.data.joint_pos[:, jids[0]]
        if len(jids) == 1
        else door.data.joint_pos[:, jids].mean(dim=-1)
    )
    door_open = torch.clamp(
        float(door_open_sign) * (door_pos - float(door_closed_pos)), min=0.0
    )
    n = door_open.shape[0]
    reset_mask = _base_reset_mask(env, "door_open_target_reached_bonus", n)
    given = getattr(env, "_door_open_target_bonus_given", None)
    if given is None or given.shape[0] != n:
        given = torch.zeros(n, dtype=torch.bool, device=env.device)
    given = torch.where(reset_mask, torch.zeros_like(given), given)
    if require_unlock_success_latch:
        unlocked = getattr(env, "_unlock_success_given", None)
        if unlocked is None or unlocked.shape[0] != n:
            unlocked = torch.zeros(n, dtype=torch.bool, device=env.device)
        else:
            unlocked = unlocked.to(dtype=torch.bool)
    else:
        unlocked = torch.ones(n, dtype=torch.bool, device=env.device)
    reached = unlocked & (door_open >= float(door_open_target))
    give = reached & (~given) & (~reset_mask)
    env._door_open_target_bonus_given = given | give
    reward = float(bonus) * give.float()
    _base_log(env, "door_open_target_bonus", reward)
    _base_log(env, "door_open_target_bonus_given", give.float())
    return reward


def release_handle_after_door_open(
    env: "ManagerBasedRLEnv",
    gripper_cfg: SceneEntityCfg,
    handle_cfg: SceneEntityCfg,
    door_joint_cfg: SceneEntityCfg,
    handle_joint_cfg: SceneEntityCfg,
    contact_sensor_name: str = "hook_contact",
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
    release_start_angle: float = 1.0,
    contact_threshold: float = 0.2,
    distance_saturation: float = 0.1,
    handle_pressed_pos: float = -0.50,
    handle_released_pos: float = 0.0,
    handle_return_weight: float = 0.5,
) -> torch.Tensor:
    """Reward handle return, contact release, and gripper retreat after opening."""
    door: Articulation = env.scene[door_joint_cfg.name]
    gripper: Articulation = env.scene[gripper_cfg.name]
    handle: Articulation = env.scene[handle_cfg.name]

    door_pos = door.data.joint_pos[:, door_joint_cfg.joint_ids[0]]
    door_open = torch.clamp(
        float(door_open_sign) * (door_pos - float(door_closed_pos)), min=0.0
    )
    handle_pos = door.data.joint_pos[:, handle_joint_cfg.joint_ids[0]]
    no_contact = _contact_indicator_from_sensor(
        env, contact_sensor_name, float(contact_threshold)
    ) <= 0.0
    distance = torch.linalg.norm(
        gripper.data.body_pos_w[:, gripper_cfg.body_ids[0], :]
        - handle.data.body_pos_w[:, handle_cfg.body_ids[0], :],
        dim=-1,
    )
    distance_reward = torch.clamp(
        distance / max(float(distance_saturation), 1.0e-6), 0.0, 1.0
    )
    n = door_open.shape[0]
    reset_mask = _base_reset_mask(env, "release_handle_after_open", n)
    opened_once = getattr(env, "_release_handle_door_opened_once", None)
    if opened_once is None or opened_once.shape[0] != n:
        opened_once = torch.zeros(n, dtype=torch.bool, device=env.device)
    opened_once = torch.where(reset_mask, torch.zeros_like(opened_once), opened_once)
    opened_once = opened_once | (door_open >= float(release_start_angle))
    env._release_handle_door_opened_once = opened_once

    handle_range = float(handle_released_pos) - float(handle_pressed_pos)
    if abs(handle_range) < 1.0e-6:
        handle_progress = torch.zeros_like(handle_pos)
    else:
        handle_progress = torch.clamp(
            (handle_pos - float(handle_pressed_pos)) / handle_range, 0.0, 1.0
        )
    # Smoothstep gives zero slope at both the fully pressed and released ends.
    handle_return_reward = handle_progress.square() * (3.0 - 2.0 * handle_progress)

    retreat_reward = no_contact.float() * distance_reward
    reward = opened_once.float() * (
        retreat_reward + float(handle_return_weight) * handle_return_reward
    )
    _base_log(env, "release_handle_after_open", reward)
    _base_log(env, "release_handle_after_open_latch", opened_once.float())
    _base_log(env, "release_handle_after_open_no_contact", no_contact.float())
    _base_log(env, "release_handle_after_open_distance", distance_reward)
    _base_log(env, "release_handle_after_open_retreat", retreat_reward)
    _base_log(env, "release_handle_after_open_handle_pos", handle_pos)
    _base_log(env, "release_handle_after_open_handle_progress", handle_progress)
    _base_log(env, "release_handle_after_open_handle_return", handle_return_reward)
    return reward


def pull_release_handle_reward(
    env: "ManagerBasedRLEnv",
    gripper_cfg: SceneEntityCfg,
    handle_cfg: SceneEntityCfg,
    door_joint_cfg: SceneEntityCfg,
    contact_sensor_name: str = "hook_contact",
    door_closed_pos: float = 0.0,
    door_open_sign: float = -1.0,
    release_start_angle: float = 0.8,
    activation_contact_threshold: float = 0.1,
    release_contact_threshold: float = 0.1,
    release_distance: float = 0.1,
    ee_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    handle_offset_h: tuple[float, float, float] = (-0.08, 0.04, -0.005),
    release_hold_steps: int = 3,
    contact_loss_bonus: float = 0.5,
    distance_progress_scale: float = 10.0,
    distance_progress_clip: float = 0.01,
    success_bonus: float = 5.0,
) -> torch.Tensor:
    """Guide a stable, one-shot handle release after completing the pull.

    The release phase is armed only when the target door angle is reached
    while handle contact is still present.  Afterwards it rewards the first
    contact-loss event, positive EE-to-handle separation progress up to the
    target distance, and a one-shot success after the final condition is held.
    """
    door: Articulation = env.scene[door_joint_cfg.name]
    gripper: Articulation = env.scene[gripper_cfg.name]
    handle: Articulation = env.scene[handle_cfg.name]

    jids = door_joint_cfg.joint_ids
    door_pos = (
        door.data.joint_pos[:, jids[0]]
        if len(jids) == 1
        else door.data.joint_pos[:, jids].mean(dim=-1)
    )
    door_open = torch.clamp(
        float(door_open_sign) * (door_pos - float(door_closed_pos)), min=0.0
    )
    contact_force = filtered_contact_force_norm(env.scene[contact_sensor_name])
    contact_present = contact_force > float(release_contact_threshold)
    no_contact = contact_force <= float(release_contact_threshold)
    distance = ee_tcp_to_handle_grasp_point_distance(
        env,
        gripper_cfg,
        handle_cfg,
        ee_offset_pos=ee_offset_pos,
        handle_offset_h=handle_offset_h,
    )

    n = door_open.shape[0]
    device = door_open.device
    reset_mask = _base_reset_mask(env, "pull_release_handle", n)
    if (not hasattr(env, "_pull_release_armed")) or env._pull_release_armed.shape[0] != n:
        env._pull_release_armed = torch.zeros(n, dtype=torch.bool, device=device)
        env._pull_release_prev_contact = torch.zeros(n, dtype=torch.bool, device=device)
        env._pull_release_contact_loss_given = torch.zeros(n, dtype=torch.bool, device=device)
        env._pull_release_success_counter = torch.zeros(n, dtype=torch.int32, device=device)
        env._pull_release_success_given = torch.zeros(n, dtype=torch.bool, device=device)
        env._pull_release_prev_distance = distance.detach().clone()

    env._pull_release_armed = torch.where(
        reset_mask, torch.zeros_like(env._pull_release_armed), env._pull_release_armed
    )
    env._pull_release_prev_contact = torch.where(
        reset_mask, torch.zeros_like(env._pull_release_prev_contact), env._pull_release_prev_contact
    )
    env._pull_release_contact_loss_given = torch.where(
        reset_mask,
        torch.zeros_like(env._pull_release_contact_loss_given),
        env._pull_release_contact_loss_given,
    )
    env._pull_release_success_counter = torch.where(
        reset_mask,
        torch.zeros_like(env._pull_release_success_counter),
        env._pull_release_success_counter,
    )
    env._pull_release_success_given = torch.where(
        reset_mask,
        torch.zeros_like(env._pull_release_success_given),
        env._pull_release_success_given,
    )
    env._pull_release_prev_distance = torch.where(
        reset_mask, distance.detach(), env._pull_release_prev_distance
    )

    can_arm = (
        (door_open >= float(release_start_angle))
        & (contact_force > float(activation_contact_threshold))
    )
    newly_armed = can_arm & (~env._pull_release_armed)
    env._pull_release_armed = env._pull_release_armed | can_arm
    env._pull_release_prev_distance = torch.where(
        newly_armed, distance.detach(), env._pull_release_prev_distance
    )

    contact_loss = (
        env._pull_release_armed
        & env._pull_release_prev_contact
        & no_contact
        & (~env._pull_release_contact_loss_given)
    )
    env._pull_release_contact_loss_given = env._pull_release_contact_loss_given | contact_loss

    capped_distance = torch.clamp(distance, max=float(release_distance))
    capped_previous = torch.clamp(env._pull_release_prev_distance, max=float(release_distance))
    distance_delta = torch.clamp(
        capped_distance - capped_previous,
        min=0.0,
        max=float(distance_progress_clip),
    )
    distance_progress = (
        env._pull_release_armed
        & no_contact
        & (~env._pull_release_success_given)
    ).float() * float(distance_progress_scale) * distance_delta

    success_condition = (
        env._pull_release_armed
        & no_contact
        & (distance >= float(release_distance))
    )
    env._pull_release_success_counter = torch.where(
        success_condition,
        env._pull_release_success_counter + 1,
        torch.zeros_like(env._pull_release_success_counter),
    )
    success_now = env._pull_release_success_counter >= int(max(1, release_hold_steps))
    give_success = success_now & (~env._pull_release_success_given)
    env._pull_release_success_given = env._pull_release_success_given | give_success

    reward = (
        float(contact_loss_bonus) * contact_loss.float()
        + distance_progress
        + float(success_bonus) * give_success.float()
    )
    env._pull_release_prev_contact = env._pull_release_armed & contact_present
    env._pull_release_prev_distance = distance.detach()

    _base_log(env, "pull_release", reward)
    _base_log(env, "pull_release_armed", env._pull_release_armed.float())
    _base_log(env, "pull_release_contact_force", contact_force)
    _base_log(env, "pull_release_contact_loss", contact_loss.float())
    _base_log(env, "pull_release_distance", distance)
    _base_log(env, "pull_release_distance_progress", distance_progress)
    _base_log(env, "pull_release_success", give_success.float())
    return reward


def _weighted_reward(
    func,
    env: "ManagerBasedRLEnv",
    weight: float,
    params: dict | None,
) -> torch.Tensor:
    if float(weight) == 0.0:
        return torch.zeros(env.num_envs, device=env.device)
    if params is None:
        params = {}
    params = _resolve_nested_scene_entity_cfgs(env, dict(params))
    return float(weight) * func(env, **params)


def _scene_entity_cfg_needs_resolve(cfg: object) -> bool:
    for attr_name in ("body_ids", "joint_ids", "fixed_tendon_ids", "object_collection_ids"):
        if hasattr(cfg, attr_name) and isinstance(getattr(cfg, attr_name), slice):
            return True
    return False


def _resolve_nested_scene_entity_cfgs(env: "ManagerBasedRLEnv", value):
    if isinstance(value, dict):
        return {key: _resolve_nested_scene_entity_cfgs(env, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_nested_scene_entity_cfgs(env, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_nested_scene_entity_cfgs(env, item) for item in value)
    if hasattr(value, "resolve") and _scene_entity_cfg_needs_resolve(value):
        value.resolve(env.scene)
    return value


def _log_stage_reward_scalar(env: "ManagerBasedRLEnv", name: str, value: torch.Tensor) -> None:
    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"][f"stage_gated_reward/{name}"] = value.mean().detach()


def _stage0_points(env, ee_cfg, handle_cfg, handle_offset_h):
    """Shared per-reward-call geometry for all hook Stage-0 terms."""
    cache = getattr(env, "_stage0_reward_step_cache", None)
    key = (ee_cfg.name, repr(ee_cfg.body_ids), handle_cfg.name, repr(handle_cfg.body_ids), tuple(handle_offset_h))
    if cache is not None and key in cache:
        return cache[key]
    ee_pos_w, _ = _body_pose_w(env, ee_cfg)
    handle_pos_w, handle_quat_w = _body_pose_w(env, handle_cfg)
    offset = torch.as_tensor(handle_offset_h, device=env.device, dtype=handle_pos_w.dtype).view(1, 3)
    target_pos_w = handle_pos_w + quat_rotate(handle_quat_w, offset.expand(env.num_envs, 3))
    result = (ee_pos_w, handle_pos_w, target_pos_w, handle_quat_w)
    if cache is not None:
        cache[key] = result
    return result


def _stage0_target_distance(env, ee_cfg, handle_cfg, handle_offset_h):
    cache = getattr(env, "_stage0_reward_step_cache", None)
    key = ("distance", ee_cfg.name, repr(ee_cfg.body_ids), handle_cfg.name, repr(handle_cfg.body_ids), tuple(handle_offset_h))
    if cache is not None and key in cache:
        return cache[key]
    ee, _, target, _ = _stage0_points(env, ee_cfg, handle_cfg, handle_offset_h)
    value = torch.linalg.norm(ee - target, dim=-1)
    if cache is not None:
        cache[key] = value
    return value


def _stage0_contact_norm(env, sensor_name):
    cache = getattr(env, "_stage0_reward_step_cache", None)
    key = ("contact", sensor_name)
    if cache is not None and key in cache:
        return cache[key]
    value = filtered_contact_force_norm(env.scene[sensor_name])
    if cache is not None:
        cache[key] = value
    return value


def base_to_pick_stance(env, robot_cfg, handle_cfg, handle_offset_h=(-0.08, 0.04, -0.005),
                        stance_offset_w=(-0.3, 0.3, -1.0), std=0.6):
    robot: Articulation = env.scene[robot_cfg.name]
    handle_pos_w, handle_quat_w = _body_pose_w(env, handle_cfg)
    offset = torch.as_tensor(handle_offset_h, device=env.device, dtype=handle_pos_w.dtype).view(1, 3)
    target = handle_pos_w + quat_rotate(handle_quat_w, offset.expand(env.num_envs, 3))
    stance = target + torch.as_tensor(stance_offset_w, device=env.device, dtype=target.dtype).view(1, 3)
    dist = torch.linalg.norm(robot.data.root_pos_w[:, :2] - stance[:, :2], dim=-1)
    return torch.exp(-torch.square(dist / max(float(std), 1.0e-6)))


def ee_to_handle(env, ee_cfg, handle_cfg, handle_offset_h=(-0.08, 0.04, -0.005), std=0.15):
    """Hook center to handle-offset target (not the handle rigid-body origin)."""
    dist = _stage0_target_distance(env, ee_cfg, handle_cfg, handle_offset_h)
    return torch.exp(-torch.square(dist / max(float(std), 1.0e-6)))


def _double_scale_distance_reward(dist, k_fast=2.0, k_slow=0.5, fast_gain=25.0, slow_gain=1.0):
    return float(k_fast) * torch.exp(-float(fast_gain) * dist.square()) + float(k_slow) * torch.exp(-float(slow_gain) * dist.square())


def ee_to_target_shaped(env, ee_cfg, handle_cfg, handle_offset_h=(-0.08, 0.04, -0.005), **shape):
    return _double_scale_distance_reward(_stage0_target_distance(env, ee_cfg, handle_cfg, handle_offset_h), **shape)


def approach_progress(env, ee_cfg, handle_cfg, handle_offset_h=(-0.08, 0.04, -0.005), max_progress=0.25):
    dist = _stage0_target_distance(env, ee_cfg, handle_cfg, handle_offset_h)
    prev = getattr(env, "_stage0_prev_approach_dist", None)
    if prev is None or prev.shape != dist.shape:
        prev = torch.full_like(dist, float("nan"))
        env._stage0_prev_approach_dist = prev
    reset = _base_reset_mask(env, "stage0_approach_progress", env.num_envs)
    reward = torch.where(torch.isfinite(prev) & (~reset), prev - dist, torch.zeros_like(dist))
    prev.copy_(dist.detach())
    return reward.clamp(-float(max_progress), float(max_progress))


def approach_reached_nearly(env, ee_cfg, handle_cfg, handle_offset_h=(-0.08, 0.04, -0.005), distance_threshold=0.05):
    return (_stage0_target_distance(env, ee_cfg, handle_cfg, handle_offset_h) < float(distance_threshold)).float()


def ee_handle_contact(env, ee_cfg, handle_cfg, contact_sensor_name="hook_contact",
                      handle_offset_h=(-0.08, 0.04, -0.005), distance_threshold=0.05, contact_threshold=0.25):
    near = _stage0_target_distance(env, ee_cfg, handle_cfg, handle_offset_h) < float(distance_threshold)
    contact = _stage0_contact_norm(env, contact_sensor_name) > float(contact_threshold)
    return (near | contact).float()


def ee_to_handle_shaped(env, ee_cfg, handle_cfg, handle_offset_h=(-0.08, 0.04, -0.005), **shape):
    # This is the separate Phase-0 physical-handle shaping term. ee_to_handle,
    # ee_to_target_shaped and progress consistently use the offset grasp point.
    ee, handle, _, _ = _stage0_points(env, ee_cfg, handle_cfg, handle_offset_h)
    return _double_scale_distance_reward(torch.linalg.norm(ee - handle, dim=-1), **shape)


def align_hook_near_handle(
    env,
    ee_cfg,
    handle_cfg,
    handle_offset_h=(-0.08, 0.04, -0.005),
    activation_distance=0.20,
    transition_width=0.04,
    **align_params,
):
    """Enable hook-pose alignment only after the EE is close to the grasp target.

    The distance gate ramps from zero at ``activation_distance + transition_width``
    to one at ``activation_distance``. This avoids a strong global orientation
    signal during early exploration while remaining continuous near 0.2 m.
    """
    dist = _stage0_target_distance(env, ee_cfg, handle_cfg, handle_offset_h)
    width = max(float(transition_width), 1.0e-6)
    gate = ((float(activation_distance) + width - dist) / width).clamp(0.0, 1.0)
    # Smoothstep removes the slope discontinuity at both ends of the gate.
    gate = gate.square() * (3.0 - 2.0 * gate)
    alignment = align_grasp_pose_v2(env=env, hand_cfg=ee_cfg, handle_cfg=handle_cfg, **align_params)
    return gate * alignment


def ee_above_handle_target(
    env,
    ee_cfg,
    handle_cfg,
    handle_offset_h=(-0.08, 0.04, -0.005),
    activation_distance=0.20,
    distance_ramp_width=0.04,
    close_distance=0.05,
    height_cap=0.07,
):
    """Guide the grasp center to approach the handle-offset target from above.

    Above-target guidance is active inside ``activation_distance``. In the
    final ``close_distance`` region, approaching below the target is penalized.
    Above is rewarded by the smooth distance gate and equal height is zero.
    """
    ee_pos_w, _, target_pos_w, _ = _stage0_points(env, ee_cfg, handle_cfg, handle_offset_h)
    dist = _stage0_target_distance(env, ee_cfg, handle_cfg, handle_offset_h)

    distance_gate = ((float(activation_distance) - dist) / max(float(distance_ramp_width), 1.0e-6)).clamp(0.0, 1.0)
    distance_gate = distance_gate.square() * (3.0 - 2.0 * distance_gate)

    height_delta = ee_pos_w[:, 2] - target_pos_w[:, 2]
    signed_height = torch.clamp(
        height_delta / max(float(height_cap), 1.0e-6), min=-1.0, max=1.0
    )

    above_reward = distance_gate * torch.clamp(signed_height, min=0.0)
    below_penalty = (dist < float(close_distance)).float() * torch.clamp(signed_height, max=0.0)
    return above_reward + below_penalty


def _stage0_grasp_quality(env, ee_cfg, handle_cfg, contact_sensor_name, handle_offset_h,
                          distance_threshold, contact_threshold, align_threshold, align_params):
    cache = getattr(env, "_stage0_reward_step_cache", None)
    key = ("quality", contact_sensor_name, float(distance_threshold), float(contact_threshold),
           float(align_threshold), repr(sorted(align_params.items())))
    if cache is not None and key in cache:
        return cache[key]
    dist = _stage0_target_distance(env, ee_cfg, handle_cfg, handle_offset_h)
    near = torch.exp(-torch.square(dist / max(float(distance_threshold), 1.0e-6)))
    align = align_grasp_pose_v2(env=env, hand_cfg=ee_cfg, handle_cfg=handle_cfg, **align_params)
    contact = (_stage0_contact_norm(env, contact_sensor_name) / max(float(contact_threshold), 1.0e-6)).clamp(0.0, 1.0)
    result = (near * align * contact,
              (dist < float(distance_threshold)) & (align >= float(align_threshold)) & (contact >= 1.0))
    if cache is not None:
        cache[key] = result
    return result


def grasping_success_shaped(env, ee_cfg, handle_cfg, contact_sensor_name="hook_contact",
                            handle_offset_h=(-0.08, 0.04, -0.005), distance_threshold=0.10,
                            contact_threshold=0.25, align_threshold=0.30,
                            above_std=0.10, below_score=0.02, **align_params):
    quality = _stage0_grasp_quality(
        env, ee_cfg, handle_cfg, contact_sensor_name, handle_offset_h,
        distance_threshold, contact_threshold, align_threshold, align_params,
    )[0]
    ee_pos_w, _, target_pos_w, _ = _stage0_points(env, ee_cfg, handle_cfg, handle_offset_h)
    dz = ee_pos_w[:, 2] - target_pos_w[:, 2]
    above_score = torch.where(
        dz >= 0.0,
        torch.exp(-torch.square(dz / max(float(above_std), 1.0e-6))),
        torch.full_like(dz, float(below_score)),
    )
    return quality * above_score


def grasp_stable_progress(env, ee_cfg, handle_cfg, contact_sensor_name="hook_contact",
                          handle_offset_h=(-0.08, 0.04, -0.005), distance_threshold=0.10,
                          contact_threshold=0.25, align_threshold=0.30, stable_time_s=0.08,
                          minimum_height_delta=0.0, maximum_height_delta=0.03, **align_params):
    stable = _stage0_grasp_quality(env, ee_cfg, handle_cfg, contact_sensor_name, handle_offset_h,
                                   distance_threshold, contact_threshold, align_threshold, align_params)[1]
    ee_pos_w, _, target_pos_w, _ = _stage0_points(env, ee_cfg, handle_cfg, handle_offset_h)
    dz = ee_pos_w[:, 2] - target_pos_w[:, 2]
    above = (dz >= float(minimum_height_delta)) & (dz <= float(maximum_height_delta))
    stable = stable & above
    counter = getattr(env, "_stage0_hook_grasp_stable_counter", None)
    if counter is None or counter.shape != stable.shape:
        counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
        env._stage0_hook_grasp_stable_counter = counter
    reset = _base_reset_mask(env, "stage0_hook_grasp_stable", env.num_envs)
    counter.copy_(torch.where(stable & (~reset), counter + 1, torch.zeros_like(counter)))
    required = max(1, int(math.ceil(float(stable_time_s) / float(env.step_dt))))
    return (counter.float() / float(required)).clamp(0.0, 1.0)


def _read_handle_unlock_flag(
    env: "ManagerBasedRLEnv",
    handle_joint_cfg: SceneEntityCfg | None,
    handle_threshold: float,
    less_than: bool,
) -> torch.Tensor:
    if handle_joint_cfg is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    handle_joint_cfg = _resolve_nested_scene_entity_cfgs(env, handle_joint_cfg)
    door: Articulation = env.scene[handle_joint_cfg.name]
    jids = handle_joint_cfg.joint_ids
    handle_pos = door.data.joint_pos[:, jids[0]] if len(jids) == 1 else door.data.joint_pos[:, jids].mean(dim=-1)
    return (handle_pos < float(handle_threshold)) if less_than else (handle_pos > float(handle_threshold))


def stage_gated_door_reward(
    env: "ManagerBasedRLEnv",
    enable_stage_gated_reward: bool = True,
    stage0_only_reward: bool = False,
    pre_grasp_cap: float = 1.0,
    stage0_reward_terms: dict[str, dict] | None = None,
    align_grasp_weight: float = 0.0,
    align_grasp_params: dict | None = None,
    approach_handle_weight: float = 0.0,
    approach_handle_params: dict | None = None,
    grasp_handle_weight: float = 0.0,
    grasp_handle_params: dict | None = None,
    grasp_success_weight: float = 0.0,
    grasp_success_params: dict | None = None,
    press_handle_weight: float = 0.0,
    press_handle_params: dict | None = None,
    keep_handle_after_press_weight: float = 0.0,
    keep_handle_after_press_params: dict | None = None,
    stall_after_grasp_weight: float = 0.0,
    stall_after_grasp_params: dict | None = None,
    stall_after_press_weight: float = 0.0,
    stall_after_press_params: dict | None = None,
    unlock_progress_weight: float = 0.0,
    unlock_progress_params: dict | None = None,
    unlock_transition_weight: float = 0.0,
    unlock_transition_params: dict | None = None,
    push_door_weight: float = 0.0,
    push_door_params: dict | None = None,
) -> torch.Tensor:
    """Aggregate DoorBot rewards with explicit stage masks."""

    if stage0_reward_terms:
        env._stage0_reward_step_cache = {}
        stage0_components = {
            name: _weighted_reward(term["func"], env, term.get("weight", 0.0), term.get("params"))
            for name, term in stage0_reward_terms.items()
        }
        stage0_reward = sum(stage0_components.values(), torch.zeros(env.num_envs, device=env.device))
        align_reward = approach_reward = grasp_handle_reward_term = torch.zeros_like(stage0_reward)
    else:
        stage0_components = {}
        align_reward = _weighted_reward(align_grasp_pose_v2, env, align_grasp_weight, align_grasp_params)
        approach_reward = _weighted_reward(approach_handle_inv_square, env, approach_handle_weight, approach_handle_params)
        grasp_handle_reward_term = _weighted_reward(grasp_handle_reward_preunlock_only, env, grasp_handle_weight, grasp_handle_params)
        stage0_reward = align_reward + approach_reward + grasp_handle_reward_term

    if stage0_only_reward:
        zero_reward = torch.zeros_like(stage0_reward)
        one_mask = torch.ones_like(stage0_reward)
        zero_mask = torch.zeros_like(stage0_reward)

        for name, value in stage0_components.items():
            _log_stage_reward_scalar(env, name, value)
        _log_stage_reward_scalar(env, "align", align_reward)
        _log_stage_reward_scalar(env, "approach", approach_reward)
        _log_stage_reward_scalar(env, "grasp_handle", grasp_handle_reward_term)
        _log_stage_reward_scalar(env, "press_handle", zero_reward)
        _log_stage_reward_scalar(env, "keep_handle_after_press", zero_reward)
        _log_stage_reward_scalar(env, "stall_after_grasp", zero_reward)
        _log_stage_reward_scalar(env, "stall_after_press", zero_reward)
        _log_stage_reward_scalar(env, "unlock_progress", zero_reward)
        _log_stage_reward_scalar(env, "unlock_transition", zero_reward)
        _log_stage_reward_scalar(env, "push_door", zero_reward)
        _log_stage_reward_scalar(env, "stage0_reward", stage0_reward)
        _log_stage_reward_scalar(env, "stage1_reward", zero_reward)
        _log_stage_reward_scalar(env, "stage2_reward", zero_reward)
        _log_stage_reward_scalar(env, "stage0_mask_ratio", one_mask)
        _log_stage_reward_scalar(env, "stage1_mask_ratio", zero_mask)
        _log_stage_reward_scalar(env, "stage2_mask_ratio", zero_mask)

        return stage0_reward

    if float(grasp_success_weight) == 0.0:
        grasp_success_reward = torch.zeros(env.num_envs, device=env.device)
    else:
        if grasp_success_params is None:
            grasp_success_params = {}
        grasp_success_call_params = _resolve_nested_scene_entity_cfgs(env, dict(grasp_success_params))
        grasp_success_reward = float(grasp_success_weight) * grasp_success_bonus(env, **grasp_success_call_params)

    if hasattr(env, "_grasp_success_given"):
        grasp_ok = env._grasp_success_given.to(dtype=torch.bool)
    else:
        grasp_ok = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    unlock_transition_reward = _weighted_reward(
        physical_unlock_transition_bonus, env, unlock_transition_weight, unlock_transition_params
    )
    unlock_flag = _physical_door_unlocked(env)

    press_handle_reward = _weighted_reward(
        press_handle_after_grasp_vel, env, press_handle_weight, press_handle_params
    )
    keep_handle_after_press_reward = _weighted_reward(
        anti_release_after_press_to_open, env, keep_handle_after_press_weight, keep_handle_after_press_params
    )
    stall_after_grasp_reward = _weighted_reward(
        stall_penalty_after_grasp_pos, env, stall_after_grasp_weight, stall_after_grasp_params
    )
    stall_after_press_reward = _weighted_reward(
        near_unlock_stall_penalty, env, stall_after_press_weight, stall_after_press_params
    )
    unlock_progress_reward = _weighted_reward(
        unlock_handle_progress_mixed, env, unlock_progress_weight, unlock_progress_params
    )

    push_door_reward = _weighted_reward(
        push_door_progress_after_unlock_success_only, env, push_door_weight, push_door_params
    )

    stage1_reward = (
        keep_handle_after_press_reward
        + press_handle_reward
        + stall_after_grasp_reward
        + stall_after_press_reward
        + unlock_progress_reward
    )
    stage2_reward = push_door_reward + 0.5 * keep_handle_after_press_reward

    stage0_mask = ~grasp_ok
    stage1_mask = grasp_ok & (~unlock_flag)
    stage2_mask = unlock_flag

    if enable_stage_gated_reward:
        masked_stage_reward = (
            stage0_mask.float() * stage0_reward
            + stage1_mask.float() * stage1_reward
            + stage2_mask.float() * stage2_reward
        )
        total_reward = masked_stage_reward + grasp_success_reward + unlock_transition_reward
    else:
        total_reward = (
            stage0_reward
            + grasp_success_reward
            + press_handle_reward
            + keep_handle_after_press_reward
            + stall_after_grasp_reward
            + stall_after_press_reward
            + unlock_progress_reward
            + unlock_transition_reward
            + push_door_reward
        )

    for name, value in stage0_components.items():
        _log_stage_reward_scalar(env, name, value)
    _log_stage_reward_scalar(env, "align", align_reward)
    _log_stage_reward_scalar(env, "approach", approach_reward)
    _log_stage_reward_scalar(env, "grasp_handle", grasp_handle_reward_term)
    _log_stage_reward_scalar(env, "press_handle", press_handle_reward)
    _log_stage_reward_scalar(env, "press_handle_raw", press_handle_reward)
    _log_stage_reward_scalar(env, "keep_handle_after_press", keep_handle_after_press_reward)
    _log_stage_reward_scalar(env, "stall_after_grasp", stall_after_grasp_reward)
    _log_stage_reward_scalar(env, "stall_after_press", stall_after_press_reward)
    _log_stage_reward_scalar(env, "unlock_progress", unlock_progress_reward)
    _log_stage_reward_scalar(env, "unlock_progress_raw", unlock_progress_reward)
    _log_stage_reward_scalar(env, "grasp_success_transition", grasp_success_reward)
    _log_stage_reward_scalar(env, "unlock_transition", unlock_transition_reward)
    _log_stage_reward_scalar(env, "push_door", push_door_reward)
    _log_stage_reward_scalar(env, "stage0_reward", stage0_reward)
    _log_stage_reward_scalar(env, "stage1_reward", stage1_reward)
    _log_stage_reward_scalar(env, "stage2_reward", stage2_reward)
    _log_stage_reward_scalar(env, "stage0_mask_ratio", stage0_mask.float())
    _log_stage_reward_scalar(env, "stage1_mask_ratio", stage1_mask.float())
    _log_stage_reward_scalar(env, "stage2_mask_ratio", stage2_mask.float())

    return total_reward


# -----------------------------------------------------------------------------
# Quadruped base rewards
# -----------------------------------------------------------------------------

def _base_log(env: "ManagerBasedRLEnv", name: str, value: torch.Tensor) -> None:
    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"][f"base_reward/{name}"] = value.mean().detach()


def _base_reset_mask(env: "ManagerBasedRLEnv", key: str, n: int) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        return torch.zeros(n, dtype=torch.bool, device=env.device)
    prev_name = f"_prev_ep_len_{key}"
    if (not hasattr(env, prev_name)) or (getattr(env, prev_name).shape[0] != n):
        setattr(env, prev_name, env.episode_length_buf.clone())
    prev = getattr(env, prev_name)
    mask = (env.episode_length_buf < prev) | (env.episode_length_buf == 0)
    setattr(env, prev_name, env.episode_length_buf.clone())
    return mask.to(dtype=torch.bool)


def _yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _unit_xy(vec: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    return vec / torch.clamp(torch.linalg.norm(vec, dim=-1, keepdim=True), min=eps)


def _const_xy(value: tuple[float, float], device, dtype) -> torch.Tensor:
    return torch.tensor(value, device=device, dtype=dtype).view(1, 2)


def compute_door_zone_masks(
    env: "ManagerBasedRLEnv",
    point_xy: torch.Tensor,
    door_joint_cfg: SceneEntityCfg = SceneEntityCfg("door", joint_names=["door_joint"]),
    door_closed_pos: float = 0.0,
    panel_rotation_sign: float = -1.0,
    closed_panel_axis_d: tuple[float, float] = (1.0, 0.0),
    hinge_offset_d: tuple[float, float] = (0.0, 0.0),
    door_length: float = 1.2,
    radius: float = 3.0,
    hinge_xy_override: torch.Tensor | None = None,
    d0_override: torch.Tensor | None = None,
    d_override: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Classify batched XY points into the pull-door Z1/Z2/Z3 regions.

    H is the hinge, d0 is the closed H-to-panel-end vector, and d is its
    current counterpart. The half-disk opposite the opening side of closed ray
    d0 is always Z2. Only the opening-side half-disk is split further: endpoint
    chord d-d0 separates outside Z1 from the hinge-facing inside half-plane,
    while Z3 is the continuing-opening side of d excluding triangle H,d0,d.
    The triangle and every remaining point are Z2. All masks are false outside
    the configured 3 m radius.

    ``panel_rotation_sign=-1`` is the calibrated Door_pull.usd convention: a
    negative joint position produces a positive geometric opening angle.
    """
    door: Articulation = env.scene[door_joint_cfg.name]
    jids = door_joint_cfg.joint_ids
    joint_pos = (
        door.data.joint_pos[:, jids[0]]
        if len(jids) == 1
        else door.data.joint_pos[:, jids].mean(dim=-1)
    )
    n = joint_pos.shape[0]
    dtype = door.data.root_pos_w.dtype
    device = door.data.root_pos_w.device

    hinge_offset = torch.tensor(
        (hinge_offset_d[0], hinge_offset_d[1], 0.0), dtype=dtype, device=device
    ).view(1, 3).expand(n, 3)
    hinge_xy = (door.data.root_pos_w + quat_rotate(door.data.root_quat_w, hinge_offset))[:, :2]

    closed_axis = torch.tensor(
        (closed_panel_axis_d[0], closed_panel_axis_d[1], 0.0), dtype=dtype, device=device
    ).view(1, 3).expand(n, 3)
    d0_unit = _unit_xy(quat_rotate(door.data.root_quat_w, closed_axis)[:, :2])
    d0 = float(door_length) * d0_unit

    angle = float(panel_rotation_sign) * (joint_pos - float(door_closed_pos))
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    d = torch.stack(
        (
            cos_a * d0[:, 0] - sin_a * d0[:, 1],
            sin_a * d0[:, 0] + cos_a * d0[:, 1],
        ),
        dim=-1,
    )

    # Traverse freezes these three tensors when the door first reaches its
    # activation angle. Other callers simply use the live geometry above.
    if hinge_xy_override is not None:
        hinge_xy = hinge_xy_override
    if d0_override is not None:
        d0 = d0_override
    if d_override is not None:
        d = d_override

    rel = point_xy - hinge_xy
    radial_distance = torch.linalg.norm(rel, dim=-1)
    in_radius = radial_distance <= float(radius)

    chord = d - d0
    chord_norm = torch.clamp(torch.linalg.norm(chord, dim=-1), min=1.0e-6)
    hinge_chord_side = chord[:, 0] * (-d0[:, 1]) - chord[:, 1] * (-d0[:, 0])
    point_from_closed_end = rel - d0
    point_chord_side = chord[:, 0] * point_from_closed_end[:, 1] - chord[:, 1] * point_from_closed_end[:, 0]
    hinge_side_sign = torch.where(
        hinge_chord_side >= 0.0,
        torch.ones_like(hinge_chord_side),
        -torch.ones_like(hinge_chord_side),
    )
    signed_inside_chord_distance = point_chord_side * hinge_side_sign / chord_norm
    chord_inside = signed_inside_chord_distance >= 0.0

    triangle_den = d0[:, 0] * d[:, 1] - d0[:, 1] * d[:, 0]
    safe_triangle_den = torch.where(
        triangle_den.abs() > 1.0e-6,
        triangle_den,
        torch.ones_like(triangle_den),
    )
    coeff_d0 = (rel[:, 0] * d[:, 1] - rel[:, 1] * d[:, 0]) / safe_triangle_den
    coeff_d = (d0[:, 0] * rel[:, 1] - d0[:, 1] * rel[:, 0]) / safe_triangle_den
    in_triangle = (
        (triangle_den.abs() > 1.0e-6)
        & (coeff_d0 >= 0.0)
        & (coeff_d >= 0.0)
        & ((coeff_d0 + coeff_d) <= 1.0)
    )

    geometric_cross = d0[:, 0] * d[:, 1] - d0[:, 1] * d[:, 0]
    opening_sign = torch.where(
        geometric_cross >= 0.0,
        torch.ones_like(geometric_cross),
        -torch.ones_like(geometric_cross),
    )
    opening_side_distance = opening_sign * (d[:, 0] * rel[:, 1] - d[:, 1] * rel[:, 0])
    opening_side_distance = opening_side_distance / max(float(door_length), 1.0e-6)
    on_opening_side_of_d = opening_side_distance >= 0.0

    closed_opening_side_distance = opening_sign * (d0[:, 0] * rel[:, 1] - d0[:, 1] * rel[:, 0])
    closed_opening_side_distance = closed_opening_side_distance / max(float(door_length), 1.0e-6)
    on_opening_halfplane = closed_opening_side_distance >= 0.0

    z1 = in_radius & on_opening_halfplane & (~chord_inside)
    z3 = in_radius & on_opening_halfplane & chord_inside & on_opening_side_of_d & (~in_triangle)
    z2 = in_radius & (~z1) & (~z3)

    return {
        "z1": z1,
        "z2": z2,
        "z3": z3,
        "in_radius": in_radius,
        "in_triangle": in_triangle & in_radius,
        "hinge_xy": hinge_xy,
        "d0": d0,
        "d": d,
        "closed_endpoint_xy": hinge_xy + d0,
        "current_endpoint_xy": hinge_xy + d,
        "radial_distance": radial_distance,
        "signed_inside_chord_distance": signed_inside_chord_distance,
        "opening_side_distance": opening_side_distance,
        "closed_opening_side_distance": closed_opening_side_distance,
        "on_opening_halfplane": on_opening_halfplane & in_radius,
        "joint_pos": joint_pos,
    }


def _door_angle(
    env: "ManagerBasedRLEnv",
    door_joint_cfg: SceneEntityCfg,
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
) -> torch.Tensor:
    door: Articulation = env.scene[door_joint_cfg.name]
    jids = door_joint_cfg.joint_ids
    door_pos = door.data.joint_pos[:, jids[0]] if len(jids) == 1 else door.data.joint_pos[:, jids].mean(dim=-1)
    return torch.clamp(float(door_open_sign) * (door_pos - float(door_closed_pos)), min=0.0)


def _base_door_opened_once(
    env: "ManagerBasedRLEnv",
    door_open: torch.Tensor,
    threshold: float,
    state_name: str,
) -> torch.Tensor:
    """Latch whether each environment crossed a door angle during this episode."""
    n = door_open.shape[0]
    attr_name = f"_{state_name}"
    opened_once = getattr(env, attr_name, None)
    if opened_once is None or opened_once.shape[0] != n:
        opened_once = torch.zeros(n, dtype=torch.bool, device=env.device)
    reset_mask = _base_reset_mask(env, state_name, n)
    opened_once = torch.where(reset_mask, torch.zeros_like(opened_once), opened_once)
    opened_once |= door_open > float(threshold)
    setattr(env, attr_name, opened_once)
    return opened_once


def _base_pose_vel(env: "ManagerBasedRLEnv", robot_cfg: SceneEntityCfg):
    robot: Articulation = env.scene[robot_cfg.name]
    base_xy = robot.data.root_pos_w[:, :2]
    base_vel_xy = robot.data.root_lin_vel_w[:, :2]
    yaw = _yaw_from_quat_wxyz(robot.data.root_quat_w)
    return robot, base_xy, base_vel_xy, yaw


def _base_cmd(env: "ManagerBasedRLEnv") -> torch.Tensor:
    cmd = getattr(env, "high_level_base_command", None)
    if cmd is None:
        return torch.zeros((env.num_envs, 5), device=env.device)
    return cmd


def _requested_base_cmd(env: "ManagerBasedRLEnv") -> torch.Tensor:
    cmd = getattr(env, "high_level_requested_base_command", None)
    if cmd is None:
        return _base_cmd(env)
    return cmd


def _ensure_base_init(env: "ManagerBasedRLEnv", base_xy: torch.Tensor, yaw: torch.Tensor):
    n = base_xy.shape[0]
    if (not hasattr(env, "_base_reward_init_xy")) or (env._base_reward_init_xy.shape != base_xy.shape):
        env._base_reward_init_xy = base_xy.detach().clone()
        env._base_reward_init_yaw = yaw.detach().clone()
    reset_mask = _base_reset_mask(env, "base_reward_init", n)
    env._base_reward_init_xy = torch.where(reset_mask.unsqueeze(-1), base_xy.detach(), env._base_reward_init_xy)
    env._base_reward_init_yaw = torch.where(reset_mask, yaw.detach(), env._base_reward_init_yaw)
    return env._base_reward_init_xy, env._base_reward_init_yaw


def _base_stage_masks(
    door_open: torch.Tensor,
    stage3_start_angle: float = 0.10,
    stage4_start_angle: float = 0.70,
):
    hold_mask = door_open < float(stage3_start_angle)
    push_follow_mask = (door_open >= float(stage3_start_angle)) & (door_open < float(stage4_start_angle))
    traverse_mask = door_open >= float(stage4_start_angle)
    return hold_mask, push_follow_mask, traverse_mask


def _doorway_geometry(
    env: "ManagerBasedRLEnv",
    base_xy: torch.Tensor,
    doorway_center_xy: tuple[float, float],
    doorway_forward_axis: tuple[float, float],
    door_cfg: SceneEntityCfg = SceneEntityCfg("door"),
):
    door: Articulation = env.scene[door_cfg.name]
    center_d = torch.tensor(
        (doorway_center_xy[0], doorway_center_xy[1], 0.0),
        dtype=door.data.root_pos_w.dtype,
        device=door.data.root_pos_w.device,
    ).view(1, 3).expand(base_xy.shape[0], 3)
    forward_d = torch.tensor(
        (doorway_forward_axis[0], doorway_forward_axis[1], 0.0),
        dtype=door.data.root_pos_w.dtype,
        device=door.data.root_pos_w.device,
    ).view(1, 3).expand(base_xy.shape[0], 3)
    center_w = door.data.root_pos_w + quat_rotate(door.data.root_quat_w, center_d)
    forward_w = quat_rotate(door.data.root_quat_w, forward_d)
    center = center_w[:, :2].to(dtype=base_xy.dtype, device=base_xy.device)
    forward = _unit_xy(forward_w[:, :2].to(dtype=base_xy.dtype, device=base_xy.device))
    to_center = center - base_xy
    to_center_u = _unit_xy(to_center)
    lateral = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
    signed_forward = torch.sum((base_xy - center) * forward, dim=-1)
    lateral_error = torch.sum((base_xy - center) * lateral, dim=-1)
    doorway_yaw = torch.atan2(forward[:, 1], forward[:, 0])
    return center, forward, to_center_u, signed_forward, lateral_error, doorway_yaw


def _contact_indicator_from_sensor(
    env: "ManagerBasedRLEnv",
    sensor_name: str | None,
    threshold: float,
) -> torch.Tensor:
    if not sensor_name:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    try:
        sensor: ContactSensor = env.scene[sensor_name]
    except Exception:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    fm = getattr(sensor.data, "force_matrix_w", None)
    if fm is not None:
        if fm.ndim == 4:
            force = torch.linalg.norm(fm.sum(dim=2), dim=-1).amax(dim=1)
        elif fm.ndim == 3:
            force = torch.linalg.norm(fm.sum(dim=1), dim=-1)
        else:
            force = torch.zeros(env.num_envs, device=env.device)
    else:
        net = getattr(sensor.data, "net_forces_w", None)
        if net is None:
            force = torch.zeros(env.num_envs, device=env.device)
        elif net.ndim == 3:
            force = torch.linalg.norm(net, dim=-1).amax(dim=1)
        else:
            force = torch.linalg.norm(net, dim=-1)
    return (force > float(threshold)).float()


def _contact_force_norm_from_sensor(
    env: "ManagerBasedRLEnv",
    sensor_name: str | None,
) -> torch.Tensor:
    if not sensor_name:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    try:
        sensor: ContactSensor = env.scene[sensor_name]
    except Exception:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    fm = getattr(sensor.data, "force_matrix_w", None)
    if fm is not None:
        if fm.ndim == 4:
            return torch.linalg.norm(fm.sum(dim=2), dim=-1).amax(dim=1)
        if fm.ndim == 3:
            return torch.linalg.norm(fm.sum(dim=1), dim=-1)
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    net = getattr(sensor.data, "net_forces_w", None)
    if net is None:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if net.ndim == 3:
        return torch.linalg.norm(net, dim=-1).amax(dim=1)
    return torch.linalg.norm(net, dim=-1)


def _normalized_contact_penalty(
    env: "ManagerBasedRLEnv",
    sensor_name: str | None,
    force_ref: float,
    door_joint_cfg: SceneEntityCfg,
    door_closed_pos: float,
    door_open_sign: float,
    stage3_start_angle: float,
    stage4_start_angle: float,
    early_scale: float,
    stage3_scale: float,
    stage4_scale: float,
) -> torch.Tensor:
    force = _contact_force_norm_from_sensor(env, sensor_name)
    penalty = torch.clamp(force / max(float(force_ref), 1.0e-6), 0.0, 1.0)
    door_open = _door_angle(env, door_joint_cfg, door_closed_pos, door_open_sign)
    scale = torch.full_like(penalty, float(early_scale))
    scale = torch.where(door_open >= float(stage3_start_angle), torch.full_like(scale, float(stage3_scale)), scale)
    scale = torch.where(door_open >= float(stage4_start_angle), torch.full_like(scale, float(stage4_scale)), scale)
    return penalty * scale


def body_door_collision_penalty(
    env: "ManagerBasedRLEnv",
    sensor_name: str = "body_door_contact",
    force_ref: float = 50.0,
    door_joint_cfg: SceneEntityCfg = SceneEntityCfg("door", joint_names=["door_joint"]),
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
    stage3_start_angle: float = 0.10,
    stage4_start_angle: float = 0.70,
    early_scale: float = 1.0,
    stage3_scale: float = 0.8,
    stage4_scale: float = 0.8,
) -> torch.Tensor:
    return _normalized_contact_penalty(
        env, sensor_name, force_ref, door_joint_cfg, door_closed_pos, door_open_sign,
        stage3_start_angle, stage4_start_angle, early_scale, stage3_scale, stage4_scale
    )


def leg_door_collision_penalty(
    env: "ManagerBasedRLEnv",
    sensor_name: str = "leg_door_contact",
    force_ref: float = 50.0,
    door_joint_cfg: SceneEntityCfg = SceneEntityCfg("door", joint_names=["door_joint"]),
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
    stage3_start_angle: float = 0.10,
    stage4_start_angle: float = 0.70,
    early_scale: float = 0.7,
    stage3_scale: float = 0.25,
    stage4_scale: float = 0.2,
) -> torch.Tensor:
    return _normalized_contact_penalty(
        env, sensor_name, force_ref, door_joint_cfg, door_closed_pos, door_open_sign,
        stage3_start_angle, stage4_start_angle, early_scale, stage3_scale, stage4_scale
    )


def body_frame_collision_penalty(
    env: "ManagerBasedRLEnv",
    sensor_name: str = "body_door_frame_contact",
    force_ref: float = 50.0,
    door_joint_cfg: SceneEntityCfg = SceneEntityCfg("door", joint_names=["door_joint"]),
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
    stage3_start_angle: float = 0.10,
    stage4_start_angle: float = 0.70,
    early_scale: float = 1.0,
    stage3_scale: float = 0.9,
    stage4_scale: float = 0.9,
) -> torch.Tensor:
    return _normalized_contact_penalty(
        env, sensor_name, force_ref, door_joint_cfg, door_closed_pos, door_open_sign,
        stage3_start_angle, stage4_start_angle, early_scale, stage3_scale, stage4_scale
    )


def leg_frame_collision_penalty(
    env: "ManagerBasedRLEnv",
    sensor_name: str = "leg_door_frame_contact",
    force_ref: float = 50.0,
    door_joint_cfg: SceneEntityCfg = SceneEntityCfg("door", joint_names=["door_joint"]),
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
    stage3_start_angle: float = 0.10,
    stage4_start_angle: float = 0.70,
    early_scale: float = 0.7,
    stage3_scale: float = 0.25,
    stage4_scale: float = 0.2,
) -> torch.Tensor:
    return _normalized_contact_penalty(
        env, sensor_name, force_ref, door_joint_cfg, door_closed_pos, door_open_sign,
        stage3_start_angle, stage4_start_angle, early_scale, stage3_scale, stage4_scale
    )


def base_hold_reward(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    door_joint_cfg: SceneEntityCfg = SceneEntityCfg("door", joint_names=["door_joint"]),
    stage3_start_angle: float = 0.10,
    stage4_start_angle: float = 0.70,
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
    cmd_penalty_scale: float = 0.5,
    pos_penalty_scale: float = 1.0,
    yaw_penalty_scale: float = 1.0,
    cmd_deadzone: float = 0.0,
    pos_deadzone: float = 0.0,
    yaw_deadzone: float = 0.0,
) -> torch.Tensor:
    _, base_xy, _, yaw = _base_pose_vel(env, robot_cfg)
    init_xy, init_yaw = _ensure_base_init(env, base_xy, yaw)
    cmd = _base_cmd(env)
    # Base hold belongs exclusively to stage 1: grasp latched, not unlocked.
    grasped = getattr(env, "_grasp_success_given", torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
    unlocked = _physical_door_unlocked(env)
    hold_mask = grasped.to(torch.bool) & (~unlocked)

    cmd_mag = torch.linalg.norm(cmd[:, :3], dim=-1)
    pos_err = torch.linalg.norm(base_xy - init_xy, dim=-1)
    yaw_err = torch.abs(_wrap_to_pi(yaw - init_yaw))

    cmd_cost = torch.clamp(cmd_mag - float(cmd_deadzone), min=0.0) ** 2
    pos_cost = torch.clamp(pos_err - float(pos_deadzone), min=0.0) ** 2
    yaw_cost = torch.clamp(yaw_err - float(yaw_deadzone), min=0.0) ** 2

    reward = (
        -float(cmd_penalty_scale) * cmd_cost
        -float(pos_penalty_scale) * pos_cost
        -float(yaw_penalty_scale) * yaw_cost
    )
    reward = hold_mask.float() * reward
    _base_log(env, "hold", reward)
    _base_log(env, "hold_mask", hold_mask.float())
    _base_log(env, "hold_cmd_cost", cmd_cost)
    _base_log(env, "hold_pos_cost", pos_cost)
    _base_log(env, "hold_yaw_cost", yaw_cost)
    return reward


def base_push_follow_reward(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    door_joint_cfg: SceneEntityCfg = SceneEntityCfg("door", joint_names=["door_joint"]),
    doorway_center_xy: tuple[float, float] = (0.0, 0.0),
    doorway_forward_axis: tuple[float, float] = (1.0, 0.0),
    stage3_start_angle: float = 0.10,
    stage4_start_angle: float = 0.70,
    door_reward_start_angle: float = 0.20,
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
    contact_sensor_name: str = "hook_contact",
    contact_threshold: float = 0.25,
    k_yaw: float = 2.0,
    progress_vel_scale: float = 0.5,
) -> torch.Tensor:
    _, base_xy, base_vel_xy, yaw = _base_pose_vel(env, robot_cfg)
    door_open = _door_angle(env, door_joint_cfg, door_closed_pos, door_open_sign)
    unlocked = _physical_door_unlocked(env)
    handle_contact = _contact_indicator_from_sensor(
        env, contact_sensor_name, float(contact_threshold)
    ) > 0.0
    stage3_mask = (
        (door_open >= float(stage3_start_angle))
        & (door_open <= float(stage4_start_angle))
        & unlocked
        & handle_contact
    )
    _, _, to_center_u, _, _, doorway_yaw = _doorway_geometry(
        env, base_xy, doorway_center_xy, doorway_forward_axis, door_joint_cfg
    )

    n = door_open.shape[0]
    if (not hasattr(env, "_base_reward_prev_door_open")) or (env._base_reward_prev_door_open.shape[0] != n):
        env._base_reward_prev_door_open = door_open.detach().clone()
    reset_mask = _base_reset_mask(env, "base_reward_door_open", n)
    env._base_reward_prev_door_open = torch.where(reset_mask, door_open.detach(), env._base_reward_prev_door_open)
    door_progress_reward = torch.relu(door_open - env._base_reward_prev_door_open)
    env._base_reward_prev_door_open = door_open.detach()

    door_angle_reward = torch.clamp(
        (door_open - float(door_reward_start_angle)) / float(max(stage4_start_angle - door_reward_start_angle, 1e-6)),
        0.0,
        1.0,
    )
    base_to_doorway_reward = torch.clamp(
        torch.sum(base_vel_xy * to_center_u, dim=-1) / float(max(progress_vel_scale, 1e-6)),
        0.0,
        1.0,
    )
    yaw_align_reward = torch.exp(-float(k_yaw) * _wrap_to_pi(yaw - doorway_yaw) ** 2)
    reward = stage3_mask.float() * (
        8.0 * door_angle_reward
        + 8.0 * door_progress_reward
        + 2.0 * base_to_doorway_reward
        + yaw_align_reward
    )
    _base_log(env, "push_follow", reward)
    _base_log(env, "push_follow_mask", stage3_mask.float())
    _base_log(env, "push_follow_unlocked", unlocked.float())
    _base_log(env, "push_follow_handle_contact", handle_contact.float())
    _base_log(env, "push_follow_door_angle", door_angle_reward)
    _base_log(env, "push_follow_door_progress", door_progress_reward)
    _base_log(env, "push_follow_to_doorway", base_to_doorway_reward)
    return reward


def base_pull_follow_reward(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    handle_cfg: SceneEntityCfg = SceneEntityCfg("door", body_names=["handle_1"]),
    door_joint_cfg: SceneEntityCfg = SceneEntityCfg("door", joint_names=["door_joint"]),
    door_closed_pos: float = 0.0,
    door_open_sign: float = -1.0,
    pull_start_joint_pos: float = 0.0,
    pull_end_joint_pos: float = -0.7,
    door_progress_scale: float = 0.02,
    backtrack_penalty: float = 1.0,
    use_best_open_progress: bool = False,
    stance_std: float = 0.25,
    hinge_offset_d: tuple[float, float] = (0.0, 0.0),
    tangent_velocity_scale: float = 0.4,
    max_planar_speed: float = 0.6,
    overspeed_margin: float = 0.2,
    yaw_gain: float = 2.0,
    reach_deadzone: float = 0.10,
    reach_scale: float = 0.20,
    door_progress_weight: float = 8.0,
    door_angle_weight: float = 0.5,
    stance_weight: float = 3.0,
    tangent_velocity_weight: float = 2.0,
    face_handle_weight: float = 1.0,
    reach_penalty_weight: float = 1.0,
    overspeed_penalty_weight: float = 2.0,
) -> torch.Tensor:
    """Follow the handle's circular path while pulling a negative-angle door.

    Raw joint positions gate the pull stage (normally ``-0.7 <= q <= 0``),
    while ``door_open_sign`` converts them to a positive opening-progress
    coordinate.  At physical unlock, the base-to-handle stance is captured and
    subsequently rotated by the door's opening angle to form a moving target.
    """
    robot, base_xy, base_vel_xy, yaw = _base_pose_vel(env, robot_cfg)
    door: Articulation = env.scene[door_joint_cfg.name]
    handle_asset: Articulation = env.scene[handle_cfg.name]
    jids = door_joint_cfg.joint_ids
    joint_pos = door.data.joint_pos[:, jids[0]] if len(jids) == 1 else door.data.joint_pos[:, jids].mean(dim=-1)
    door_open = torch.clamp(float(door_open_sign) * (joint_pos - float(door_closed_pos)), min=0.0)
    handle_xy = handle_asset.data.body_pos_w[:, handle_cfg.body_ids[0], :2]
    unlocked = _physical_door_unlocked(env)

    lower = min(float(pull_start_joint_pos), float(pull_end_joint_pos))
    upper = max(float(pull_start_joint_pos), float(pull_end_joint_pos))
    pull_mask = unlocked & (joint_pos >= lower) & (joint_pos <= upper)
    # Hand control over to the frozen traverse controller once the door has
    # reached its pull target.  Without this latch, the stationary stance,
    # facing, and absolute-angle terms can make waiting in Z3 profitable.
    traverse_started = getattr(env, "_pull_traverse_started", None)
    if traverse_started is not None and traverse_started.shape[0] == joint_pos.shape[0]:
        pull_mask = pull_mask & (~traverse_started.to(dtype=torch.bool))

    n = joint_pos.shape[0]
    reset_mask = _base_reset_mask(env, "base_pull_follow", n)
    if (not hasattr(env, "_base_pull_prev_unlocked")) or (env._base_pull_prev_unlocked.shape[0] != n):
        env._base_pull_prev_unlocked = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._base_pull_anchor_base_xy = base_xy.detach().clone()
        env._base_pull_anchor_handle_xy = handle_xy.detach().clone()
        env._base_pull_anchor_open = door_open.detach().clone()
        env._base_pull_prev_open = door_open.detach().clone()
        env._base_pull_best_open = door_open.detach().clone()
    if (not hasattr(env, "_base_pull_best_open")) or (env._base_pull_best_open.shape[0] != n):
        env._base_pull_best_open = door_open.detach().clone()

    env._base_pull_prev_unlocked = torch.where(
        reset_mask, torch.zeros_like(env._base_pull_prev_unlocked), env._base_pull_prev_unlocked
    )
    newly_unlocked = unlocked & (~env._base_pull_prev_unlocked)
    anchor_mask = reset_mask | newly_unlocked
    env._base_pull_anchor_base_xy = torch.where(
        anchor_mask.unsqueeze(-1), base_xy.detach(), env._base_pull_anchor_base_xy
    )
    env._base_pull_anchor_handle_xy = torch.where(
        anchor_mask.unsqueeze(-1), handle_xy.detach(), env._base_pull_anchor_handle_xy
    )
    env._base_pull_anchor_open = torch.where(anchor_mask, door_open.detach(), env._base_pull_anchor_open)
    env._base_pull_prev_open = torch.where(anchor_mask, door_open.detach(), env._base_pull_prev_open)
    env._base_pull_best_open = torch.where(
        anchor_mask, door_open.detach(), env._base_pull_best_open
    )
    env._base_pull_prev_unlocked = unlocked.detach()

    if use_best_open_progress:
        previous_best_open = env._base_pull_best_open
        env._base_pull_best_open = torch.maximum(previous_best_open, door_open.detach())
        door_delta = env._base_pull_best_open - previous_best_open
    else:
        door_delta = door_open - env._base_pull_prev_open
    env._base_pull_prev_open = door_open.detach()
    progress_scale = max(float(door_progress_scale), 1.0e-6)
    door_progress_reward = torch.clamp(torch.relu(door_delta) / progress_scale, 0.0, 1.0)
    if use_best_open_progress:
        door_backtrack_cost = torch.clamp(
            torch.relu(env._base_pull_best_open - door_open) / progress_scale, 0.0, 1.0
        )
    else:
        door_backtrack_cost = torch.clamp(torch.relu(-door_delta) / progress_scale, 0.0, 1.0)
    open_range = max(abs(float(pull_end_joint_pos) - float(pull_start_joint_pos)), 1.0e-6)
    door_angle_reward = torch.clamp(door_open / open_range, 0.0, 1.0)

    # Rotate the unlock-time base-to-handle stance with the CCW door motion.
    stance_at_unlock = env._base_pull_anchor_base_xy - env._base_pull_anchor_handle_xy
    rotation = door_open - env._base_pull_anchor_open
    cos_a = torch.cos(rotation)
    sin_a = torch.sin(rotation)
    rotated_stance = torch.stack(
        (
            cos_a * stance_at_unlock[:, 0] - sin_a * stance_at_unlock[:, 1],
            sin_a * stance_at_unlock[:, 0] + cos_a * stance_at_unlock[:, 1],
        ),
        dim=-1,
    )
    target_base_xy = handle_xy + rotated_stance
    target_error = target_base_xy - base_xy
    target_distance = torch.linalg.norm(target_error, dim=-1)
    stance_reward = torch.exp(-torch.square(target_distance / max(float(stance_std), 1.0e-6)))
    # Positive door_open is counter-clockwise around world z.  Reward the
    # base's velocity projected onto the corresponding circular tangent.
    hinge_offset = torch.tensor(
        (hinge_offset_d[0], hinge_offset_d[1], 0.0),
        dtype=door.data.root_pos_w.dtype,
        device=door.data.root_pos_w.device,
    ).view(1, 3).expand(n, 3)
    hinge_xy = (door.data.root_pos_w + quat_rotate(door.data.root_quat_w, hinge_offset))[:, :2]
    target_radius = target_base_xy - hinge_xy
    tangent_direction = _unit_xy(torch.stack((-target_radius[:, 1], target_radius[:, 0]), dim=-1))
    tangent_velocity_reward = torch.clamp(
        torch.sum(base_vel_xy * tangent_direction, dim=-1) / max(float(tangent_velocity_scale), 1.0e-6),
        0.0,
        1.0,
    )
    planar_speed = torch.linalg.norm(base_vel_xy, dim=-1)
    overspeed_cost = torch.clamp(
        torch.square(
            torch.relu(planar_speed - float(max_planar_speed))
            / max(float(overspeed_margin), 1.0e-6)
        ),
        0.0,
        1.0,
    )

    to_handle = handle_xy - base_xy
    target_yaw = torch.atan2(to_handle[:, 1], to_handle[:, 0])
    face_handle_reward = torch.exp(-float(yaw_gain) * _wrap_to_pi(yaw - target_yaw) ** 2)
    current_reach = torch.linalg.norm(to_handle, dim=-1)
    target_reach = torch.linalg.norm(stance_at_unlock, dim=-1)
    reach_error = torch.clamp(torch.abs(current_reach - target_reach) - float(reach_deadzone), min=0.0)
    reach_cost = torch.square(reach_error / max(float(reach_scale), 1.0e-6))

    reward = pull_mask.float() * (
        float(door_progress_weight) * door_progress_reward
        - float(door_progress_weight) * float(backtrack_penalty) * door_backtrack_cost
        + float(door_angle_weight) * door_angle_reward
        + float(stance_weight) * stance_reward
        + float(tangent_velocity_weight) * tangent_velocity_reward
        + float(face_handle_weight) * face_handle_reward
        - float(reach_penalty_weight) * reach_cost
        - float(overspeed_penalty_weight) * overspeed_cost
    )
    _base_log(env, "pull_follow", reward)
    _base_log(env, "pull_follow_mask", pull_mask.float())
    _base_log(env, "pull_follow_joint_pos", joint_pos)
    _base_log(env, "pull_follow_door_open", door_open)
    _base_log(env, "pull_follow_door_progress", door_progress_reward)
    _base_log(env, "pull_follow_door_backtrack", door_backtrack_cost)
    _base_log(env, "pull_follow_stance_error", target_distance)
    _base_log(env, "pull_follow_tangent_velocity", tangent_velocity_reward)
    _base_log(env, "pull_follow_planar_speed", planar_speed)
    _base_log(env, "pull_follow_overspeed_cost", overspeed_cost)
    _base_log(env, "pull_follow_reach_cost", reach_cost)
    return reward


def base_pull_traverse_reward(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    door_joint_cfg: SceneEntityCfg = SceneEntityCfg("door", joint_names=["door_joint"]),
    activation_joint_pos: float = -0.8,
    door_closed_pos: float = 0.0,
    panel_rotation_sign: float = -1.0,
    closed_panel_axis_d: tuple[float, float] = (1.0, 0.0),
    hinge_offset_d: tuple[float, float] = (0.0, 0.0),
    door_length: float = 1.2,
    radius: float = 3.0,
    safe_radius: float = 1.4,
    z1_hold_steps: int = 3,
    z1_transition_bonus: float = 25.0,
    z2_transition_bonus: float = 60.0,
    direct_z3_to_z2_penalty: float = 80.0,
    doorway_center_xy: tuple[float, float] = (0.0, 0.0),
    doorway_forward_axis: tuple[float, float] = (0.0, -1.0),
    pass_distance: float = 0.9,
    z1_waypoint_clearance: float = 0.40,
    z1_progress_scale: float = 15.0,
    z2_progress_scale: float = 20.0,
    progress_step_clip: float = 0.03,
    velocity_scale: float = 0.4,
    velocity_reward_weight: float = 0.5,
    stall_speed_threshold: float = 0.05,
    stall_progress_threshold: float = 0.002,
    stall_delay_steps: int = 25,
    stall_penalty: float = 0.15,
) -> torch.Tensor:
    """Strict Z3 -> Z1 -> Z2 state machine with dense waypoint guidance.

    State 0 waits for an active Z3 sample, state 1 accepts only a consecutive
    Z3-to-safe-Z1 transition, state 2 accepts only a direct safe-Z1-to-safe-Z2
    transition, and state 3 is complete. Entering Z2 from state 1 is treated as
    a shortcut and penalized once per episode. Geometry is frozen when the
    unlocked pull door first reaches ``activation_joint_pos``. During state 1,
    the base is guided around the frozen panel endpoint to an outward waypoint;
    during state 2 it is guided to the final doorway pass point. Both shaping
    terms reward distance reduction rather than proximity, so standing still
    cannot accumulate progress reward.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    door: Articulation = env.scene[door_joint_cfg.name]
    base_xy = robot.data.root_pos_w[:, :2]
    jids = door_joint_cfg.joint_ids
    joint_pos = (
        door.data.joint_pos[:, jids[0]]
        if len(jids) == 1
        else door.data.joint_pos[:, jids].mean(dim=-1)
    )
    n = joint_pos.shape[0]
    device = joint_pos.device

    live_zones = compute_door_zone_masks(
        env,
        base_xy,
        door_joint_cfg=door_joint_cfg,
        door_closed_pos=door_closed_pos,
        panel_rotation_sign=panel_rotation_sign,
        closed_panel_axis_d=closed_panel_axis_d,
        hinge_offset_d=hinge_offset_d,
        door_length=door_length,
        radius=radius,
    )

    if (not hasattr(env, "_pull_traverse_state")) or (env._pull_traverse_state.shape[0] != n):
        env._pull_traverse_state = torch.zeros(n, dtype=torch.int32, device=device)
        env._pull_traverse_started = torch.zeros(n, dtype=torch.bool, device=device)
        env._pull_traverse_cheat_given = torch.zeros(n, dtype=torch.bool, device=device)
        env._pull_traverse_z1_counter = torch.zeros(n, dtype=torch.int32, device=device)
        env._pull_traverse_prev_z1 = torch.zeros(n, dtype=torch.bool, device=device)
        env._pull_traverse_prev_z2 = torch.zeros(n, dtype=torch.bool, device=device)
        env._pull_traverse_prev_z3 = torch.zeros(n, dtype=torch.bool, device=device)
        env._pull_traverse_prev_target_distance = torch.zeros(n, dtype=base_xy.dtype, device=device)
        env._pull_traverse_guidance_prev_state = torch.zeros(n, dtype=torch.int32, device=device)
        env._pull_traverse_active_steps = torch.zeros(n, dtype=torch.int32, device=device)
        env._pull_traverse_frozen_hinge_xy = live_zones["hinge_xy"].detach().clone()
        env._pull_traverse_frozen_d0 = live_zones["d0"].detach().clone()
        env._pull_traverse_frozen_d = live_zones["d"].detach().clone()

    reset_mask = _base_reset_mask(env, "pull_traverse", n)
    env._pull_traverse_state = torch.where(
        reset_mask, torch.zeros_like(env._pull_traverse_state), env._pull_traverse_state
    )
    env._pull_traverse_started = torch.where(
        reset_mask, torch.zeros_like(env._pull_traverse_started), env._pull_traverse_started
    )
    env._pull_traverse_cheat_given = torch.where(
        reset_mask, torch.zeros_like(env._pull_traverse_cheat_given), env._pull_traverse_cheat_given
    )
    env._pull_traverse_z1_counter = torch.where(
        reset_mask, torch.zeros_like(env._pull_traverse_z1_counter), env._pull_traverse_z1_counter
    )
    env._pull_traverse_prev_z1 = torch.where(
        reset_mask, torch.zeros_like(env._pull_traverse_prev_z1), env._pull_traverse_prev_z1
    )
    env._pull_traverse_prev_z2 = torch.where(
        reset_mask, torch.zeros_like(env._pull_traverse_prev_z2), env._pull_traverse_prev_z2
    )
    env._pull_traverse_prev_z3 = torch.where(
        reset_mask, torch.zeros_like(env._pull_traverse_prev_z3), env._pull_traverse_prev_z3
    )
    env._pull_traverse_prev_target_distance = torch.where(
        reset_mask,
        torch.zeros_like(env._pull_traverse_prev_target_distance),
        env._pull_traverse_prev_target_distance,
    )
    env._pull_traverse_guidance_prev_state = torch.where(
        reset_mask,
        torch.zeros_like(env._pull_traverse_guidance_prev_state),
        env._pull_traverse_guidance_prev_state,
    )
    env._pull_traverse_active_steps = torch.where(
        reset_mask, torch.zeros_like(env._pull_traverse_active_steps), env._pull_traverse_active_steps
    )

    unlocked = _physical_door_unlocked(env)
    reached_activation = unlocked & (joint_pos <= float(activation_joint_pos))
    newly_active = reached_activation & (~env._pull_traverse_started)
    env._pull_traverse_frozen_hinge_xy = torch.where(
        newly_active.unsqueeze(-1), live_zones["hinge_xy"].detach(), env._pull_traverse_frozen_hinge_xy
    )
    env._pull_traverse_frozen_d0 = torch.where(
        newly_active.unsqueeze(-1), live_zones["d0"].detach(), env._pull_traverse_frozen_d0
    )
    env._pull_traverse_frozen_d = torch.where(
        newly_active.unsqueeze(-1), live_zones["d"].detach(), env._pull_traverse_frozen_d
    )
    env._pull_traverse_started = env._pull_traverse_started | reached_activation
    # Once the door has reached the activation angle, keep the frozen traverse
    # state machine active through small door-angle rebound. Physical relocking
    # still disables it, and episode reset clears the latch above.
    active = unlocked & env._pull_traverse_started

    zones = compute_door_zone_masks(
        env,
        base_xy,
        door_joint_cfg=door_joint_cfg,
        door_closed_pos=door_closed_pos,
        panel_rotation_sign=panel_rotation_sign,
        closed_panel_axis_d=closed_panel_axis_d,
        hinge_offset_d=hinge_offset_d,
        door_length=door_length,
        radius=radius,
        hinge_xy_override=env._pull_traverse_frozen_hinge_xy,
        d0_override=env._pull_traverse_frozen_d0,
        d_override=env._pull_traverse_frozen_d,
    )
    z1 = zones["z1"]
    z2 = zones["z2"]
    z3 = zones["z3"]
    safe_z1 = z1 & (zones["radial_distance"] >= float(safe_radius))
    # Z2 follows the shared geometric definition exactly, including the
    # H/d0/d triangle and without a minimum hinge-distance requirement.
    target_z2 = z2

    state = env._pull_traverse_state
    see_z3 = active & (state == 0) & z3
    state = torch.where(see_z3, torch.ones_like(state), state)
    guidance_state = state.clone()

    # State 1 target: continue radially beyond the frozen panel endpoint.  For
    # the pull geometry this lies on the Z1 side of the endpoint chord and
    # gives enough clearance to round the panel instead of cutting through it.
    frozen_d_unit = _unit_xy(env._pull_traverse_frozen_d)
    z1_waypoint = (
        env._pull_traverse_frozen_hinge_xy
        + env._pull_traverse_frozen_d
        + float(z1_waypoint_clearance) * frozen_d_unit
    )
    doorway_center, doorway_forward, _, _, _, _ = _doorway_geometry(
        env, base_xy, doorway_center_xy, doorway_forward_axis, door_joint_cfg
    )
    z2_waypoint = doorway_center + float(pass_distance) * doorway_forward
    target_xy = torch.where((guidance_state == 1).unsqueeze(-1), z1_waypoint, z2_waypoint)
    target_delta = target_xy - base_xy
    target_distance = torch.linalg.norm(target_delta, dim=-1)
    same_guidance_phase = (
        active
        & ((guidance_state == 1) | (guidance_state == 2))
        & (env._pull_traverse_guidance_prev_state == guidance_state)
    )
    distance_progress = torch.where(
        same_guidance_phase,
        env._pull_traverse_prev_target_distance - target_distance,
        torch.zeros_like(target_distance),
    )
    clipped_progress = torch.clamp(
        distance_progress,
        min=-float(progress_step_clip),
        max=float(progress_step_clip),
    )
    progress_scale = torch.where(
        guidance_state == 1,
        torch.full_like(clipped_progress, float(z1_progress_scale)),
        torch.full_like(clipped_progress, float(z2_progress_scale)),
    )
    dense_progress_reward = same_guidance_phase.float() * progress_scale * clipped_progress

    target_direction = _unit_xy(target_delta)
    target_velocity = torch.sum(robot.data.root_lin_vel_w[:, :2] * target_direction, dim=-1)
    velocity_reward = (
        same_guidance_phase.float()
        * float(velocity_reward_weight)
        * torch.clamp(
            target_velocity / max(float(velocity_scale), 1.0e-6),
            min=-1.0,
            max=1.0,
        )
    )
    env._pull_traverse_active_steps = torch.where(
        same_guidance_phase,
        env._pull_traverse_active_steps + 1,
        torch.zeros_like(env._pull_traverse_active_steps),
    )
    planar_speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
    stalled = (
        same_guidance_phase
        & (env._pull_traverse_active_steps > int(max(0, stall_delay_steps)))
        & (planar_speed < float(stall_speed_threshold))
        & (torch.abs(distance_progress) < float(stall_progress_threshold))
    )
    stall_cost = float(stall_penalty) * stalled.float()

    waiting_z1 = active & (state == 1)
    begin_z1 = waiting_z1 & safe_z1 & env._pull_traverse_prev_z3
    continue_z1 = waiting_z1 & safe_z1 & (env._pull_traverse_z1_counter > 0)
    env._pull_traverse_z1_counter = torch.where(
        begin_z1 | continue_z1,
        env._pull_traverse_z1_counter + 1,
        torch.zeros_like(env._pull_traverse_z1_counter),
    )
    entered_z1 = waiting_z1 & (
        env._pull_traverse_z1_counter >= int(max(1, z1_hold_steps))
    )
    state = torch.where(entered_z1, torch.full_like(state, 2), state)

    illegal_z2 = active & (state == 1) & z2 & (~env._pull_traverse_cheat_given)
    env._pull_traverse_cheat_given = env._pull_traverse_cheat_given | illegal_z2

    waiting_z2 = active & (state == 2)
    entered_z2 = waiting_z2 & target_z2 & env._pull_traverse_prev_z1
    state = torch.where(entered_z2, torch.full_like(state, 3), state)

    # Returning to Z3 before completing Z2 invalidates the previous Z1 pass.
    regressed = active & (state == 2) & z3
    state = torch.where(regressed, torch.ones_like(state), state)
    env._pull_traverse_z1_counter = torch.where(
        regressed, torch.zeros_like(env._pull_traverse_z1_counter), env._pull_traverse_z1_counter
    )
    env._pull_traverse_state = state

    reward = (
        dense_progress_reward
        + velocity_reward
        - stall_cost
        + float(z1_transition_bonus) * entered_z1.float()
        + float(z2_transition_bonus) * entered_z2.float()
        - float(direct_z3_to_z2_penalty) * illegal_z2.float()
    )

    env._pull_traverse_prev_target_distance = target_distance.detach()
    env._pull_traverse_guidance_prev_state = guidance_state.detach()
    env._pull_traverse_prev_z1 = active & z1
    env._pull_traverse_prev_z2 = active & z2
    env._pull_traverse_prev_z3 = active & z3

    _base_log(env, "pull_traverse", reward)
    _base_log(env, "pull_traverse_reached_activation", reached_activation.float())
    _base_log(env, "pull_traverse_active", active.float())
    _base_log(env, "pull_traverse_state", state.float())
    _base_log(env, "pull_traverse_z1", z1.float())
    _base_log(env, "pull_traverse_z2", z2.float())
    _base_log(env, "pull_traverse_z3", z3.float())
    _base_log(env, "pull_traverse_entered_z1", entered_z1.float())
    _base_log(env, "pull_traverse_entered_z2", entered_z2.float())
    _base_log(env, "pull_traverse_illegal_z2", illegal_z2.float())
    _base_log(env, "pull_traverse_radial_distance", zones["radial_distance"])
    _base_log(env, "pull_traverse_target_distance", target_distance)
    _base_log(env, "pull_traverse_distance_progress", distance_progress)
    _base_log(env, "pull_traverse_dense_progress", dense_progress_reward)
    _base_log(env, "pull_traverse_target_velocity", target_velocity)
    _base_log(env, "pull_traverse_velocity_reward", velocity_reward)
    _base_log(env, "pull_traverse_stalled", stalled.float())
    return reward


def base_traverse_reward(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    door_joint_cfg: SceneEntityCfg = SceneEntityCfg("door", joint_names=["door_joint"]),
    doorway_center_xy: tuple[float, float] = (0.0, 0.0),
    doorway_forward_axis: tuple[float, float] = (1.0, 0.0),
    hook_contact_sensor_name: str | None = "hook_contact",
    stage3_start_angle: float = 0.10,
    stage4_start_angle: float = 0.70,
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
    k_lat: float = 6.0,
    k_yaw: float = 2.0,
    progress_vel_scale: float = 0.5,
    release_contact_threshold: float = 0.2,
    release_near_doorway_distance: float = 0.35,
    keep_opening_reward: float = 8.0,
) -> torch.Tensor:
    _, base_xy, base_vel_xy, yaw = _base_pose_vel(env, robot_cfg)
    door_open = _door_angle(env, door_joint_cfg, door_closed_pos, door_open_sign)
    _, _, traverse_mask = _base_stage_masks(door_open, stage3_start_angle, stage4_start_angle)
    center, forward, to_center_u, signed_forward, lateral_error, doorway_yaw = _doorway_geometry(
        env, base_xy, doorway_center_xy, doorway_forward_axis, door_joint_cfg
    )

    progress_axis = torch.where((signed_forward > 0.0).unsqueeze(-1), forward, to_center_u)
    pass_reward = torch.clamp(
        torch.sum(base_vel_xy * progress_axis, dim=-1) / float(max(progress_vel_scale, 1e-6)),
        0.0,
        1.0,
    )
    center_reward = torch.exp(-float(k_lat) * lateral_error ** 2)
    yaw_align_reward = torch.exp(-float(k_yaw) * _wrap_to_pi(yaw - doorway_yaw) ** 2)

    distance_to_doorway = torch.linalg.norm(center - base_xy, dim=-1)
    handle_contact = _contact_indicator_from_sensor(env, hook_contact_sensor_name, release_contact_threshold) > 0.0
    release_gate = traverse_mask & (distance_to_doorway < float(release_near_doorway_distance))
    release_handle_reward = release_gate.float() * (~handle_contact).float()

    reward = traverse_mask.float() * (
        float(keep_opening_reward)
        + 8.0 * pass_reward
        + 2.0 * center_reward
        + yaw_align_reward
        + 2.0 * release_handle_reward
    )
    _base_log(env, "traverse", reward)
    _base_log(env, "traverse_mask", traverse_mask.float())
    _base_log(env, "traverse_pass", pass_reward)
    _base_log(env, "traverse_release", release_handle_reward)
    return reward


def base_safety_reward(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    door_joint_cfg: SceneEntityCfg = SceneEntityCfg("door", joint_names=["door_joint"]),
    body_door_sensor_name: str | None = "body_door_contact",
    leg_door_sensor_name: str | None = "leg_door_contact",
    body_frame_sensor_name: str | None = "body_door_frame_contact",
    leg_frame_sensor_name: str | None = "leg_door_frame_contact",
    force_ref: float = 50.0,
    default_height: float = 0.43,
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
    stage3_start_angle: float = 0.10,
    stage4_start_angle: float = 0.70,
    early_body_door_weight: float = 1.0,
    early_leg_door_weight: float = 0.5,
    early_body_frame_weight: float = 1.0,
    early_leg_frame_weight: float = 0.5,
    late_body_door_weight: float = 0.5,
    late_leg_door_weight: float = 0.2,
    late_body_frame_weight: float = 0.5,
    late_leg_frame_weight: float = 0.2,
    cmd_rate_weight: float = 0.05,
    height_pitch_weight: float = 2.0,
) -> torch.Tensor:
    cmd = _base_cmd(env)
    requested_cmd = _requested_base_cmd(env)
    n = cmd.shape[0]
    if (not hasattr(env, "_base_reward_prev_cmd")) or (env._base_reward_prev_cmd.shape != cmd.shape):
        env._base_reward_prev_cmd = cmd.detach().clone()
    reset_mask = _base_reset_mask(env, "base_reward_cmd", n)
    env._base_reward_prev_cmd = torch.where(reset_mask.unsqueeze(-1), cmd.detach(), env._base_reward_prev_cmd)
    cmd_rate = torch.sum((cmd - env._base_reward_prev_cmd) ** 2, dim=-1)
    env._base_reward_prev_cmd = cmd.detach()

    height_pitch_reg = (requested_cmd[:, 3] - float(default_height)) ** 2 + requested_cmd[:, 4] ** 2

    body_door_force = torch.clamp(
        _contact_force_norm_from_sensor(env, body_door_sensor_name) / max(float(force_ref), 1.0e-6),
        0.0,
        1.0,
    )
    leg_door_force = torch.clamp(
        _contact_force_norm_from_sensor(env, leg_door_sensor_name) / max(float(force_ref), 1.0e-6),
        0.0,
        1.0,
    )
    body_frame_force = torch.clamp(
        _contact_force_norm_from_sensor(env, body_frame_sensor_name) / max(float(force_ref), 1.0e-6),
        0.0,
        1.0,
    )
    leg_frame_force = torch.clamp(
        _contact_force_norm_from_sensor(env, leg_frame_sensor_name) / max(float(force_ref), 1.0e-6),
        0.0,
        1.0,
    )

    door_open = _door_angle(env, door_joint_cfg, door_closed_pos, door_open_sign)
    late_mask = door_open >= float(stage3_start_angle)
    w_body_door = torch.where(
        late_mask,
        torch.full_like(body_door_force, float(late_body_door_weight)),
        torch.full_like(body_door_force, float(early_body_door_weight)),
    )
    w_leg_door = torch.where(
        late_mask,
        torch.full_like(leg_door_force, float(late_leg_door_weight)),
        torch.full_like(leg_door_force, float(early_leg_door_weight)),
    )
    w_body_frame = torch.where(
        late_mask,
        torch.full_like(body_frame_force, float(late_body_frame_weight)),
        torch.full_like(body_frame_force, float(early_body_frame_weight)),
    )
    w_leg_frame = torch.where(
        late_mask,
        torch.full_like(leg_frame_force, float(late_leg_frame_weight)),
        torch.full_like(leg_frame_force, float(early_leg_frame_weight)),
    )

    reward = (
        -w_body_door * body_door_force
        -w_leg_door * leg_door_force
        -w_body_frame * body_frame_force
        -w_leg_frame * leg_frame_force
        -float(cmd_rate_weight) * cmd_rate
        -float(height_pitch_weight) * height_pitch_reg
    )
    _base_log(env, "safety", reward)
    _base_log(env, "safety_body_door_force_term", body_door_force)
    _base_log(env, "safety_leg_door_force_term", leg_door_force)
    _base_log(env, "safety_body_frame_force_term", body_frame_force)
    _base_log(env, "safety_leg_frame_force_term", leg_frame_force)
    _base_log(env, "safety_late_weight_mask", late_mask.float())
    _base_log(env, "safety_cmd_rate", cmd_rate)
    _base_log(env, "safety_height_pitch_reg", height_pitch_reg)
    return reward


def base_traverse_success(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    door_joint_cfg: SceneEntityCfg = SceneEntityCfg("door", joint_names=["door_joint"]),
    doorway_center_xy: tuple[float, float] = (0.0, 0.0),
    doorway_forward_axis: tuple[float, float] = (1.0, 0.0),
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
    required_door_angle: float = 1.0,
    pass_distance: float = 0.5,
    num_steps: int = 3,
    exclude_pull_traverse_cheat: bool = False,
    require_pull_traverse_complete: bool = False,
) -> torch.Tensor:
    _, base_xy, _, _ = _base_pose_vel(env, robot_cfg)
    door_open = _door_angle(env, door_joint_cfg, door_closed_pos, door_open_sign)
    _, _, _, signed_forward, _, _ = _doorway_geometry(
        env, base_xy, doorway_center_xy, doorway_forward_axis, door_joint_cfg
    )
    opened_once = _base_door_opened_once(
        env, door_open, required_door_angle, "base_traverse_success_door_opened_once"
    )
    good = opened_once & (signed_forward > float(pass_distance))
    if exclude_pull_traverse_cheat:
        cheat_given = getattr(env, "_pull_traverse_cheat_given", None)
        if cheat_given is not None and cheat_given.shape[0] == good.shape[0]:
            good = good & (~cheat_given.to(dtype=torch.bool))
    if require_pull_traverse_complete:
        traverse_state = getattr(env, "_pull_traverse_state", None)
        if traverse_state is None or traverse_state.shape[0] != good.shape[0]:
            good = torch.zeros_like(good)
        else:
            good = good & (traverse_state == 3)

    n = good.shape[0]
    if (not hasattr(env, "_base_traverse_success_counter")) or (env._base_traverse_success_counter.shape[0] != n):
        env._base_traverse_success_counter = torch.zeros(n, dtype=torch.int32, device=env.device)
    reset_mask = _base_reset_mask(env, "base_traverse_success", n)
    env._base_traverse_success_counter = torch.where(
        reset_mask,
        torch.zeros_like(env._base_traverse_success_counter),
        env._base_traverse_success_counter,
    )
    env._base_traverse_success_counter = torch.where(
        good,
        env._base_traverse_success_counter + 1,
        torch.zeros_like(env._base_traverse_success_counter),
    )
    done = env._base_traverse_success_counter >= int(max(1, num_steps))
    _base_log(env, "success_done", done.float())
    _base_log(env, "success_signed_forward", signed_forward)
    _base_log(env, "success_door_opened_once", opened_once.float())
    if exclude_pull_traverse_cheat:
        _base_log(env, "success_no_pull_traverse_cheat", good.float())
    if require_pull_traverse_complete:
        _base_log(env, "success_pull_traverse_complete", good.float())
    return done


def base_traverse_success_bonus(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    door_joint_cfg: SceneEntityCfg = SceneEntityCfg("door", joint_names=["door_joint"]),
    doorway_center_xy: tuple[float, float] = (0.0, 0.0),
    doorway_forward_axis: tuple[float, float] = (1.0, 0.0),
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
    required_door_angle: float = 1.0,
    pass_distance: float = 0.5,
    num_steps: int = 3,
    bonus: float = 80.0,
    exclude_pull_traverse_cheat: bool = False,
    require_pull_traverse_complete: bool = False,
) -> torch.Tensor:
    """Give a one-shot bonus when traverse-success conditions hold long enough."""
    _, base_xy, _, _ = _base_pose_vel(env, robot_cfg)
    door_open = _door_angle(env, door_joint_cfg, door_closed_pos, door_open_sign)
    _, _, _, signed_forward, _, _ = _doorway_geometry(
        env, base_xy, doorway_center_xy, doorway_forward_axis, door_joint_cfg
    )
    opened_once = _base_door_opened_once(
        env, door_open, required_door_angle, "base_traverse_bonus_door_opened_once"
    )
    good = opened_once & (signed_forward > float(pass_distance))
    if exclude_pull_traverse_cheat:
        cheat_given = getattr(env, "_pull_traverse_cheat_given", None)
        if cheat_given is not None and cheat_given.shape[0] == good.shape[0]:
            good = good & (~cheat_given.to(dtype=torch.bool))
    if require_pull_traverse_complete:
        traverse_state = getattr(env, "_pull_traverse_state", None)
        if traverse_state is None or traverse_state.shape[0] != good.shape[0]:
            good = torch.zeros_like(good)
        else:
            good = good & (traverse_state == 3)

    n = good.shape[0]
    if (not hasattr(env, "_base_traverse_bonus_counter")) or (env._base_traverse_bonus_counter.shape[0] != n):
        env._base_traverse_bonus_counter = torch.zeros(n, dtype=torch.int32, device=env.device)
        env._base_traverse_bonus_given = torch.zeros(n, dtype=torch.bool, device=env.device)
    reset_mask = _base_reset_mask(env, "base_traverse_success_bonus", n)
    env._base_traverse_bonus_counter = torch.where(
        reset_mask,
        torch.zeros_like(env._base_traverse_bonus_counter),
        env._base_traverse_bonus_counter,
    )
    env._base_traverse_bonus_given = torch.where(
        reset_mask,
        torch.zeros_like(env._base_traverse_bonus_given),
        env._base_traverse_bonus_given,
    )
    env._base_traverse_bonus_counter = torch.where(
        good,
        env._base_traverse_bonus_counter + 1,
        torch.zeros_like(env._base_traverse_bonus_counter),
    )
    success_now = env._base_traverse_bonus_counter >= int(max(1, num_steps))
    give = success_now & (~env._base_traverse_bonus_given)
    env._base_traverse_bonus_given = env._base_traverse_bonus_given | give
    reward = give.float() * float(bonus)
    _base_log(env, "traverse_success_bonus", reward)
    _base_log(env, "traverse_success_bonus_door_opened_once", opened_once.float())
    return reward


def bad_arm_pose_penalty(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    gripper_cfg: SceneEntityCfg,
    door_joint_cfg: SceneEntityCfg,
    doorway_center_xy: tuple[float, float] = (0.0, 0.0),
    doorway_forward_axis: tuple[float, float] = (1.0, 0.0),
    pass_distance: float = 0.8,
    height_margin: float = 0.0,
    bonus: float = -20.0,
) -> torch.Tensor:
    """Apply a one-shot penalty if the gripper hangs below the base after traversal."""
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_asset: Articulation = env.scene[gripper_cfg.name]
    _, _, _, signed_forward, _, _ = _doorway_geometry(
        env,
        robot.data.root_pos_w[:, :2],
        doorway_center_xy,
        doorway_forward_axis,
        door_joint_cfg,
    )
    gripper_z = gripper_asset.data.body_pos_w[:, gripper_cfg.body_ids[0], 2]
    base_z = robot.data.root_pos_w[:, 2]
    bad = (signed_forward > float(pass_distance)) & (
        gripper_z < base_z - float(height_margin)
    )

    n = bad.shape[0]
    if (not hasattr(env, "_bad_arm_pose_penalty_given")) or (
        env._bad_arm_pose_penalty_given.shape[0] != n
    ):
        env._bad_arm_pose_penalty_given = torch.zeros(n, dtype=torch.bool, device=env.device)
    reset_mask = _base_reset_mask(env, "bad_arm_pose_penalty", n)
    env._bad_arm_pose_penalty_given = torch.where(
        reset_mask,
        torch.zeros_like(env._bad_arm_pose_penalty_given),
        env._bad_arm_pose_penalty_given,
    )
    give = bad & (~env._bad_arm_pose_penalty_given)
    env._bad_arm_pose_penalty_given = env._bad_arm_pose_penalty_given | give
    reward = give.float() * float(bonus)
    _base_log(env, "bad_arm_pose_penalty", reward)
    return reward


def base_task_failure(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    door_joint_cfg: SceneEntityCfg = SceneEntityCfg("door", joint_names=["door_joint"]),
    doorway_center_xy: tuple[float, float] = (0.0, 0.0),
    doorway_forward_axis: tuple[float, float] = (1.0, 0.0),
    base_door_sensor_name: str | None = None,
    thigh_calf_door_sensor_name: str | None = None,
    arm_body_leg_sensor_name: str | None = None,
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
    tilt_limit_deg: float = 25.0,
    collision_force_threshold: float = 30.0,
    door_close_after_angle: float = 0.10,
    door_closed_failure_angle: float = 0.03,
    stuck_stage_angle: float = 0.70,
    stuck_progress_speed: float = 0.02,
    stuck_steps: int = 180,
    num_bad_steps: int = 2,
) -> torch.Tensor:
    robot, base_xy, base_vel_xy, _ = _base_pose_vel(env, robot_cfg)
    door_open = _door_angle(env, door_joint_cfg, door_closed_pos, door_open_sign)
    _, forward, to_center_u, signed_forward, _, _ = _doorway_geometry(
        env, base_xy, doorway_center_xy, doorway_forward_axis, door_joint_cfg
    )

    base_door_collision = _contact_indicator_from_sensor(env, base_door_sensor_name, collision_force_threshold) > 0.0
    thigh_calf_door_collision = (
        _contact_indicator_from_sensor(env, thigh_calf_door_sensor_name, collision_force_threshold) > 0.0
    )
    arm_body_leg_collision = _contact_indicator_from_sensor(env, arm_body_leg_sensor_name, collision_force_threshold) > 0.0
    tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
    fallen = tilt > math.radians(float(tilt_limit_deg))

    n = door_open.shape[0]
    if (not hasattr(env, "_base_failure_max_door_open")) or (env._base_failure_max_door_open.shape[0] != n):
        env._base_failure_max_door_open = door_open.detach().clone()
    reset_mask = _base_reset_mask(env, "base_task_failure", n)
    env._base_failure_max_door_open = torch.where(
        reset_mask,
        door_open.detach(),
        torch.maximum(env._base_failure_max_door_open, door_open.detach()),
    )
    door_bumped_closed = (
        (env._base_failure_max_door_open > float(door_close_after_angle))
        & (door_open < float(door_closed_failure_angle))
    )

    progress_axis = torch.where((signed_forward > 0.0).unsqueeze(-1), forward, to_center_u)
    progress_speed = torch.sum(base_vel_xy * progress_axis, dim=-1)
    stuck_active = door_open > float(stuck_stage_angle)
    stuck_now = stuck_active & (progress_speed < float(stuck_progress_speed))
    if (not hasattr(env, "_base_stuck_counter")) or (env._base_stuck_counter.shape[0] != n):
        env._base_stuck_counter = torch.zeros(n, dtype=torch.int32, device=env.device)
    env._base_stuck_counter = torch.where(
        reset_mask | (~stuck_now),
        torch.zeros_like(env._base_stuck_counter),
        env._base_stuck_counter + 1,
    )
    stuck = env._base_stuck_counter >= int(max(1, stuck_steps))

    severe = base_door_collision | thigh_calf_door_collision | arm_body_leg_collision | fallen | door_bumped_closed | stuck
    if (not hasattr(env, "_base_failure_counter")) or (env._base_failure_counter.shape[0] != n):
        env._base_failure_counter = torch.zeros(n, dtype=torch.int32, device=env.device)
    env._base_failure_counter = torch.where(
        reset_mask,
        torch.zeros_like(env._base_failure_counter),
        env._base_failure_counter,
    )
    env._base_failure_counter = torch.where(
        severe,
        env._base_failure_counter + 1,
        torch.zeros_like(env._base_failure_counter),
    )
    done = env._base_failure_counter >= int(max(1, num_bad_steps))
    _base_log(env, "failure_done", done.float())
    _base_log(env, "failure_fallen", fallen.float())
    _base_log(env, "failure_door_bumped_closed", door_bumped_closed.float())
    _base_log(env, "failure_stuck", stuck.float())
    return done

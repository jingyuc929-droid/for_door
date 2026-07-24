"""Reward terms for hierarchical pick-and-place tasks."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse

from .observations import ee_position_in_base, object_position_in_base, place_position_in_base


def _exp_distance(distance: torch.Tensor, std: float) -> torch.Tensor:
    return torch.exp(-torch.square(distance) / max(float(std) ** 2, 1.0e-12))


def _get_object(env, object_cfg: SceneEntityCfg):
    try:
        return env.scene[object_cfg.name]
    except Exception:
        return None


def _zeros(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


def _phase_mask(env, phases: int | tuple[int, ...] | list[int] | None, ref: torch.Tensor) -> torch.Tensor:
    if phases is None:
        return torch.ones_like(ref, dtype=torch.bool)
    phase = getattr(env, "pick_place_phase", None)
    if phase is None:
        return torch.zeros_like(ref, dtype=torch.bool)
    if isinstance(phases, int):
        phases = (phases,)
    mask = torch.zeros_like(ref, dtype=torch.bool)
    for phase_id in phases:
        mask |= phase.to(device=ref.device) == int(phase_id)
    return mask


def _masked(value: torch.Tensor, env, phases: int | tuple[int, ...] | list[int] | None) -> torch.Tensor:
    return value * _phase_mask(env, phases, value).to(dtype=value.dtype)


def _ee_object_distance(
    env,
    ee_asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor | None:
    obj = _get_object(env, object_cfg)
    if obj is None:
        return None
    ee_b = ee_position_in_base(env, ee_asset_cfg)
    obj_b = object_position_in_base(env, object_cfg)
    return torch.linalg.norm(ee_b - obj_b, dim=-1)


def _object_place_distance(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor | None:
    obj = _get_object(env, object_cfg)
    if obj is None:
        return None
    place_b = place_position_in_base(env)
    obj_b = object_position_in_base(env, object_cfg)
    return torch.linalg.norm(obj_b - place_b, dim=-1)


def _ee_linear_velocity_w(env, ee_asset_cfg: SceneEntityCfg) -> torch.Tensor:
    ee_asset: Articulation = env.scene[ee_asset_cfg.name]
    if len(ee_asset_cfg.body_ids) < 1:
        raise ValueError("ee_asset_cfg must resolve at least one body.")
    return ee_asset.data.body_lin_vel_w[:, ee_asset_cfg.body_ids].mean(dim=1)


def _object_in_gripper(
    env,
    ee_asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    distance_threshold: float = 0.08,
    velocity_threshold: float = 0.35,
) -> torch.Tensor:
    obj = _get_object(env, object_cfg)
    if obj is None:
        return _zeros(env).bool()
    dist = _ee_object_distance(env, ee_asset_cfg, object_cfg)
    ee_vel_w = _ee_linear_velocity_w(env, ee_asset_cfg)
    rel_speed = torch.linalg.norm(ee_vel_w - obj.data.root_lin_vel_w, dim=-1)
    return (dist < float(distance_threshold)) & (rel_speed < float(velocity_threshold))


def _gripper_open_bool(
    env,
    gripper_asset_cfg: SceneEntityCfg | None,
    open_threshold: float = 0.06,
) -> torch.Tensor | None:
    if gripper_asset_cfg is None:
        return None
    robot: Articulation = env.scene[gripper_asset_cfg.name]
    if len(gripper_asset_cfg.joint_ids) < 1:
        raise ValueError("gripper_asset_cfg must resolve at least one joint.")
    joint_pos = robot.data.joint_pos[:, gripper_asset_cfg.joint_ids]
    return torch.mean(joint_pos, dim=-1) > float(open_threshold)


def _place_success_bool(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    distance_threshold: float = 0.12,
    velocity_threshold: float = 0.25,
    require_object_still: bool = True,
) -> torch.Tensor:
    obj: RigidObject | None = _get_object(env, object_cfg)
    if obj is None:
        return _zeros(env).bool()
    place_pos_w = getattr(env, "pick_place_place_pos_w", None)
    if place_pos_w is None:
        return _zeros(env).bool()
    dist = torch.linalg.norm(obj.data.root_pos_w - place_pos_w, dim=-1)
    placed = dist < float(distance_threshold)
    if bool(require_object_still):
        speed = torch.linalg.norm(obj.data.root_lin_vel_w, dim=-1)
        placed &= speed < float(velocity_threshold)
    return placed


def object_fallen_penalty(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    min_height: float = 0.12,
    phases: tuple[int, ...] | None = (2, 3, 4, 5),
) -> torch.Tensor:
    obj = _get_object(env, object_cfg)
    if obj is None:
        return _zeros(env)
    fallen = (obj.data.root_pos_w[:, 2] < float(min_height)).to(dtype=torch.float32)
    return _masked(fallen, env, phases)


def object_below_pick_penalty(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    margin: float = 0.03,
    scale: float = 0.08,
    phases: tuple[int, ...] | None = (1, 2, 3),
) -> torch.Tensor:
    obj = _get_object(env, object_cfg)
    pick_pos_w = getattr(env, "pick_place_pick_pos_w", None)
    if obj is None or pick_pos_w is None:
        return _zeros(env)
    drop = pick_pos_w[:, 2] - obj.data.root_pos_w[:, 2] - float(margin)
    penalty = torch.clamp(drop / max(float(scale), 1.0e-6), min=0.0, max=1.0)
    return _masked(penalty, env, phases)


def base_to_pick_stance_exp(
    env,
    std: float = 0.5,
    stance_offset: tuple[float, float, float] = (-0.45, 0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    phases: tuple[int, ...] | None = (0, 1),
) -> torch.Tensor:
    pick_pos_w = getattr(env, "pick_place_pick_pos_w", None)
    if pick_pos_w is None:
        return torch.zeros(env.num_envs, device=env.device)
    robot: Articulation = env.scene[asset_cfg.name]
    offset = torch.tensor(stance_offset, device=env.device, dtype=pick_pos_w.dtype).view(1, 3)
    target = pick_pos_w + offset
    dist = torch.linalg.norm(robot.data.root_pos_w[:, :2] - target[:, :2], dim=-1)
    reward = _exp_distance(dist, std)
    return reward * _phase_mask(env, phases, reward).to(dtype=reward.dtype)


def ee_to_object_exp(
    env,
    ee_asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    std: float = 0.18,
    phases: tuple[int, ...] | None = (0, 1),
) -> torch.Tensor:
    obj = _get_object(env, object_cfg)
    if obj is None:
        return torch.zeros(env.num_envs, device=env.device)
    ee_b = ee_position_in_base(env, ee_asset_cfg)
    obj_b = object_position_in_base(env, object_cfg)
    dist = torch.linalg.norm(ee_b - obj_b, dim=-1)
    reward = _exp_distance(dist, std)
    return reward * _phase_mask(env, phases, reward).to(dtype=reward.dtype)


def ee_to_object_shaped(
    env,
    ee_asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    k_fast: float = 2.0,
    k_slow: float = 0.5,
    fast_gain: float = 25.0,
    slow_gain: float = 1.0,
    phases: tuple[int, ...] | None = (0, 1),
) -> torch.Tensor:
    dist = _ee_object_distance(env, ee_asset_cfg, object_cfg)
    if dist is None:
        return _zeros(env)
    reward = float(k_fast) * torch.exp(-float(fast_gain) * torch.square(dist))
    reward += float(k_slow) * torch.exp(-float(slow_gain) * torch.square(dist))
    return _masked(reward, env, phases)


def ee_to_target_shaped(
    env,
    ee_asset_cfg: SceneEntityCfg,
    k_fast: float = 2.0,
    k_slow: float = 0.5,
    fast_gain: float = 25.0,
    slow_gain: float = 1.0,
    phases: tuple[int, ...] | None = (0, 1),
) -> torch.Tensor:
    pick_pos_w = getattr(env, "pick_place_pick_pos_w", None)
    if pick_pos_w is None:
        return _zeros(env)
    ee_b = ee_position_in_base(env, ee_asset_cfg)
    robot: Articulation = env.scene["robot"]
    target_b = quat_apply_inverse(robot.data.root_quat_w, pick_pos_w - robot.data.root_pos_w)
    dist = torch.linalg.norm(ee_b - target_b, dim=-1)
    reward = float(k_fast) * torch.exp(-float(fast_gain) * torch.square(dist))
    reward += float(k_slow) * torch.exp(-float(slow_gain) * torch.square(dist))
    return _masked(reward, env, phases)


def ee_to_place_shaped(
    env,
    ee_asset_cfg: SceneEntityCfg,
    k_fast: float = 2.0,
    k_slow: float = 0.5,
    fast_gain: float = 25.0,
    slow_gain: float = 1.0,
    phases: tuple[int, ...] | None = (1, 2),
) -> torch.Tensor:
    place_pos_w = getattr(env, "pick_place_place_pos_w", None)
    if place_pos_w is None:
        return _zeros(env)
    ee_b = ee_position_in_base(env, ee_asset_cfg)
    robot: Articulation = env.scene["robot"]
    target_b = quat_apply_inverse(robot.data.root_quat_w, place_pos_w - robot.data.root_pos_w)
    dist = torch.linalg.norm(ee_b - target_b, dim=-1)
    reward = float(k_fast) * torch.exp(-float(fast_gain) * torch.square(dist))
    reward += float(k_slow) * torch.exp(-float(slow_gain) * torch.square(dist))
    return _masked(reward, env, phases)


def ee_object_contact(
    env,
    ee_asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    distance_threshold: float = 0.07,
    phases: tuple[int, ...] | None = (0, 1),
) -> torch.Tensor:
    dist = _ee_object_distance(env, ee_asset_cfg, object_cfg)
    if dist is None:
        return _zeros(env)
    reward = (dist < float(distance_threshold)).to(dtype=torch.float32)
    return _masked(reward, env, phases)


def grasp_success(
    env,
    ee_asset_cfg: SceneEntityCfg,
    gripper_asset_cfg: SceneEntityCfg | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    distance_threshold: float = 0.08,
    velocity_threshold: float = 0.35,
    lift_height: float = 0.04,
    gripper_closed_threshold: float = 0.035,
    one_shot: bool = False,
    phases: tuple[int, ...] | None = (1, 2),
) -> torch.Tensor:
    obj = _get_object(env, object_cfg)
    if obj is None:
        return torch.zeros(env.num_envs, device=env.device)
    in_gripper = _object_in_gripper(env, ee_asset_cfg, object_cfg, distance_threshold, velocity_threshold)
    pick_pos_w = getattr(env, "pick_place_pick_pos_w", None)
    if pick_pos_w is None:
        lift_ok = torch.ones_like(in_gripper)
    else:
        lift_ok = obj.data.root_pos_w[:, 2] > pick_pos_w[:, 2] + float(lift_height)
    gripper_closed = torch.ones_like(in_gripper)
    if gripper_asset_cfg is not None:
        robot: Articulation = env.scene[gripper_asset_cfg.name]
        joint_pos = robot.data.joint_pos[:, gripper_asset_cfg.joint_ids]
        gripper_closed = torch.mean(torch.abs(joint_pos), dim=-1) < float(gripper_closed_threshold)
    success = in_gripper & lift_ok & gripper_closed
    reward = success.to(dtype=torch.float32)
    if one_shot:
        paid = getattr(env, "pick_place_grasp_success_paid", None)
        if paid is None or paid.shape != (env.num_envs,):
            paid = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
            setattr(env, "pick_place_grasp_success_paid", paid)
        phase = getattr(env, "pick_place_phase", None)
        if phase is not None:
            paid[phase.to(device=env.device) <= 0] = False
        reward = reward * (~paid).to(dtype=reward.dtype)
        paid[success] = True
    return reward * _phase_mask(env, phases, reward).to(dtype=reward.dtype)


def grasp_hold_success(
    env,
    ee_asset_cfg: SceneEntityCfg,
    gripper_asset_cfg: SceneEntityCfg | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    distance_threshold: float = 0.08,
    velocity_threshold: float = 0.35,
    lift_height: float = 0.04,
    gripper_closed_threshold: float = 0.035,
    require_lift: bool = True,
    phases: tuple[int, ...] | None = (3,),
) -> torch.Tensor:
    obj = _get_object(env, object_cfg)
    if obj is None:
        return _zeros(env)
    held = _object_in_gripper(env, ee_asset_cfg, object_cfg, distance_threshold, velocity_threshold)
    if gripper_asset_cfg is not None:
        robot: Articulation = env.scene[gripper_asset_cfg.name]
        joint_pos = robot.data.joint_pos[:, gripper_asset_cfg.joint_ids]
        held &= torch.mean(torch.abs(joint_pos), dim=-1) < float(gripper_closed_threshold)
    if bool(require_lift):
        pick_pos_w = getattr(env, "pick_place_pick_pos_w", None)
        if pick_pos_w is None:
            held &= torch.ones_like(held)
        else:
            held &= obj.data.root_pos_w[:, 2] > pick_pos_w[:, 2] + float(lift_height)
    reward = held.to(dtype=torch.float32)
    return _masked(reward, env, phases)


def grasp_hold_failure_penalty(
    env,
    ee_asset_cfg: SceneEntityCfg,
    gripper_asset_cfg: SceneEntityCfg | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    distance_threshold: float = 0.08,
    velocity_threshold: float = 0.35,
    lift_height: float = 0.04,
    gripper_closed_threshold: float = 0.035,
    require_lift: bool = True,
    phases: tuple[int, ...] | None = (2, 3),
) -> torch.Tensor:
    obj = _get_object(env, object_cfg)
    if obj is None:
        return _zeros(env)
    held = _object_in_gripper(env, ee_asset_cfg, object_cfg, distance_threshold, velocity_threshold)
    if gripper_asset_cfg is not None:
        robot: Articulation = env.scene[gripper_asset_cfg.name]
        joint_pos = robot.data.joint_pos[:, gripper_asset_cfg.joint_ids]
        held &= torch.mean(torch.abs(joint_pos), dim=-1) < float(gripper_closed_threshold)
    if bool(require_lift):
        pick_pos_w = getattr(env, "pick_place_pick_pos_w", None)
        if pick_pos_w is not None:
            held &= obj.data.root_pos_w[:, 2] > pick_pos_w[:, 2] + float(lift_height)
    penalty = (~held).to(dtype=torch.float32)
    return _masked(penalty, env, phases)


def grasping_success_shaped(
    env,
    ee_asset_cfg: SceneEntityCfg,
    gripper_asset_cfg: SceneEntityCfg | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    distance_threshold: float = 0.08,
    velocity_threshold: float = 0.35,
    lift_height: float = 0.04,
    gripper_closed_threshold: float = 0.035,
    held_reward: float = 1.0,
    lifted_reward: float = 1.0,
    phases: tuple[int, ...] | None = (1, 2),
) -> torch.Tensor:
    obj = _get_object(env, object_cfg)
    if obj is None:
        return _zeros(env)
    in_gripper = _object_in_gripper(env, ee_asset_cfg, object_cfg, distance_threshold, velocity_threshold)
    pick_pos_w = getattr(env, "pick_place_pick_pos_w", None)
    if pick_pos_w is None:
        lift_ok = torch.ones_like(in_gripper)
    else:
        lift_ok = obj.data.root_pos_w[:, 2] > pick_pos_w[:, 2] + float(lift_height)
    gripper_closed = torch.ones_like(in_gripper)
    if gripper_asset_cfg is not None:
        robot: Articulation = env.scene[gripper_asset_cfg.name]
        joint_pos = robot.data.joint_pos[:, gripper_asset_cfg.joint_ids]
        gripper_closed = torch.mean(torch.abs(joint_pos), dim=-1) < float(gripper_closed_threshold)
    held = in_gripper & gripper_closed
    reward = float(held_reward) * held.to(dtype=torch.float32)
    reward += float(lifted_reward) * (held & lift_ok).to(dtype=torch.float32)
    return _masked(reward, env, phases)


def object_lift_progress(
    env,
    ee_asset_cfg: SceneEntityCfg | None = None,
    gripper_asset_cfg: SceneEntityCfg | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    lift_height: float = 0.04,
    distance_threshold: float = 0.08,
    velocity_threshold: float = 0.35,
    gripper_closed_threshold: float = 0.035,
    require_grasp: bool = False,
    phases: tuple[int, ...] | None = (1,),
) -> torch.Tensor:
    obj = _get_object(env, object_cfg)
    pick_pos_w = getattr(env, "pick_place_pick_pos_w", None)
    if obj is None or pick_pos_w is None:
        return _zeros(env)
    lift = obj.data.root_pos_w[:, 2] - pick_pos_w[:, 2]
    reward = torch.clamp(lift / max(float(lift_height), 1.0e-6), min=0.0, max=1.0)
    if require_grasp:
        if ee_asset_cfg is None:
            return _zeros(env)
        grasped = _object_in_gripper(env, ee_asset_cfg, object_cfg, distance_threshold, velocity_threshold)
        if gripper_asset_cfg is not None:
            robot: Articulation = env.scene[gripper_asset_cfg.name]
            joint_pos = robot.data.joint_pos[:, gripper_asset_cfg.joint_ids]
            closed = torch.mean(torch.abs(joint_pos), dim=-1) < float(gripper_closed_threshold)
            grasped &= closed
        reward *= grasped.to(dtype=reward.dtype)
    return _masked(reward, env, phases)


def gripper_close_near_object(
    env,
    ee_asset_cfg: SceneEntityCfg,
    gripper_asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    distance_threshold: float = 0.08,
    closed_threshold: float = 0.035,
    phases: tuple[int, ...] | None = (1,),
) -> torch.Tensor:
    dist = _ee_object_distance(env, ee_asset_cfg, object_cfg)
    if dist is None:
        return _zeros(env)
    robot: Articulation = env.scene[gripper_asset_cfg.name]
    gripper_pos = robot.data.joint_pos[:, gripper_asset_cfg.joint_ids].mean(dim=-1)
    near = dist < float(distance_threshold)
    closed = gripper_pos < float(closed_threshold)
    reward = (near & closed).to(dtype=torch.float32)
    return _masked(reward, env, phases)


def base_heading_to_place(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    phases: tuple[int, ...] | None = (2, 3),
) -> torch.Tensor:
    place_pos_w = getattr(env, "pick_place_place_pos_w", None)
    if place_pos_w is None:
        return _zeros(env)
    robot: Articulation = env.scene[asset_cfg.name]
    direction = place_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    direction_norm = torch.linalg.norm(direction, dim=-1, keepdim=True).clamp_min(1.0e-6)
    direction = direction / direction_norm
    forward_w = quat_apply(
        robot.data.root_quat_w,
        torch.tensor((1.0, 0.0, 0.0), device=env.device, dtype=torch.float32).repeat(env.num_envs, 1),
    )
    reward = torch.sum(forward_w[:, :2] * direction, dim=-1).clamp(min=-1.0, max=1.0)
    return _masked(reward, env, phases)


def object_to_place_exp(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    std: float = 0.25,
    phases: tuple[int, ...] | None = (2, 3),
) -> torch.Tensor:
    obj = _get_object(env, object_cfg)
    if obj is None:
        return torch.zeros(env.num_envs, device=env.device)
    place_b = place_position_in_base(env)
    obj_b = object_position_in_base(env, object_cfg)
    dist = torch.linalg.norm(obj_b - place_b, dim=-1)
    reward = _exp_distance(dist, std)
    return reward * _phase_mask(env, phases, reward).to(dtype=reward.dtype)


def object_to_place_shaped(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    k_wide: float = 2.0,
    k_narrow: float = 2.0,
    wide_gain: float = 5.0,
    narrow_gain: float = 25.0,
    phases: tuple[int, ...] | None = (2, 3),
) -> torch.Tensor:
    dist = _object_place_distance(env, object_cfg)
    if dist is None:
        return _zeros(env)
    reward = float(k_wide) * torch.exp(-float(wide_gain) * torch.square(dist))
    reward += float(k_narrow) * torch.exp(-float(narrow_gain) * torch.square(dist))
    return _masked(reward, env, phases)


def object_to_place_progress(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    negative_scale: float = 1.0,
    phases: tuple[int, ...] | None = (2,),
) -> torch.Tensor:
    dist = _object_place_distance(env, object_cfg)
    if dist is None:
        return _zeros(env)

    prev = getattr(env, "pick_place_prev_object_place_dist", None)
    if prev is None or prev.shape != (env.num_envs,):
        prev = torch.full((env.num_envs,), float("nan"), device=env.device, dtype=torch.float32)
        setattr(env, "pick_place_prev_object_place_dist", prev)

    mask = _phase_mask(env, phases, dist)
    progress = torch.where(torch.isfinite(prev), prev - dist, torch.zeros_like(dist))
    progress = torch.clamp(progress, min=-0.25, max=0.25)
    progress = torch.where(progress < 0.0, progress * float(negative_scale), progress)
    prev[mask] = dist[mask]
    prev[~mask] = float("nan")
    return progress * mask.to(dtype=progress.dtype)


def base_to_place_exp(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_distance: float = 0.3,
    gain: float = 0.5,
    phases: tuple[int, ...] | None = (2, 3),
) -> torch.Tensor:
    place_pos_w = getattr(env, "pick_place_place_pos_w", None)
    if place_pos_w is None:
        return _zeros(env)
    robot: Articulation = env.scene[asset_cfg.name]
    dist = torch.linalg.norm(robot.data.root_pos_w[:, :2] - place_pos_w[:, :2], dim=-1)
    reward = torch.exp(-float(gain) * torch.clamp(dist, min=float(min_distance)))
    return _masked(reward, env, phases)


def place_success(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    distance_threshold: float = 0.12,
    velocity_threshold: float = 0.25,
    require_object_still: bool = True,
    phases: tuple[int, ...] | None = (3, 4, 5),
) -> torch.Tensor:
    obj: RigidObject | None = _get_object(env, object_cfg)
    if obj is None:
        return torch.zeros(env.num_envs, device=env.device)
    place_pos_w = getattr(env, "pick_place_place_pos_w", None)
    if place_pos_w is None:
        return torch.zeros(env.num_envs, device=env.device)
    dist = torch.linalg.norm(obj.data.root_pos_w - place_pos_w, dim=-1)
    placed = dist < float(distance_threshold)
    if bool(require_object_still):
        speed = torch.linalg.norm(obj.data.root_lin_vel_w, dim=-1)
        placed &= speed < float(velocity_threshold)
    reward = placed.to(dtype=torch.float32)
    return reward * _phase_mask(env, phases, reward).to(dtype=reward.dtype)


def gripper_release(
    env,
    ee_asset_cfg: SceneEntityCfg | None = None,
    gripper_asset_cfg: SceneEntityCfg | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    open_threshold: float = 0.06,
    place_distance_threshold: float = 0.12,
    object_velocity_threshold: float = 0.25,
    require_object_still_for_place: bool = True,
    require_place_height: bool = False,
    place_height_threshold: float = 0.05,
    release_distance_threshold: float = 0.12,
    phases: tuple[int, ...] | None = (4, 5),
) -> torch.Tensor:
    placed = _place_success_bool(
        env,
        object_cfg,
        place_distance_threshold,
        object_velocity_threshold,
        require_object_still=require_object_still_for_place,
    )
    if bool(require_place_height):
        obj = _get_object(env, object_cfg)
        place_pos_w = getattr(env, "pick_place_place_pos_w", None)
        if obj is None or place_pos_w is None:
            return _zeros(env)
        height_ok = torch.abs(obj.data.root_pos_w[:, 2] - place_pos_w[:, 2]) < float(place_height_threshold)
        placed &= height_ok
    gripper_open = _gripper_open_bool(env, gripper_asset_cfg, open_threshold)
    if gripper_open is None:
        if ee_asset_cfg is None:
            return _zeros(env)
        dist = _ee_object_distance(env, ee_asset_cfg, object_cfg)
        if dist is None:
            return _zeros(env)
        gripper_open = dist > float(release_distance_threshold)
    reward = (placed & gripper_open).to(dtype=torch.float32)
    return _masked(reward, env, phases)


def object_on_place_height_success(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    gripper_asset_cfg: SceneEntityCfg | None = None,
    open_threshold: float = 0.06,
    place_xy_threshold: float = 0.25,
    height_std: float = 0.04,
    require_gripper_open: bool = False,
    phases: tuple[int, ...] | None = (4, 5),
) -> torch.Tensor:
    obj = _get_object(env, object_cfg)
    place_pos_w = getattr(env, "pick_place_place_pos_w", None)
    if obj is None or place_pos_w is None:
        return _zeros(env)
    xy_dist = torch.linalg.norm(obj.data.root_pos_w[:, :2] - place_pos_w[:, :2], dim=-1)
    height_error = torch.abs(obj.data.root_pos_w[:, 2] - place_pos_w[:, 2])
    reward = torch.exp(-torch.square(height_error) / max(float(height_std) ** 2, 1.0e-12))
    reward *= (xy_dist < float(place_xy_threshold)).to(dtype=reward.dtype)
    if bool(require_gripper_open):
        gripper_open = _gripper_open_bool(env, gripper_asset_cfg, open_threshold)
        if gripper_open is None:
            return _zeros(env)
        reward *= gripper_open.to(dtype=reward.dtype)
    return _masked(reward, env, phases)


def object_below_place_penalty(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    place_xy_threshold: float = 0.3,
    margin: float = 0.02,
    scale: float = 0.08,
    phases: tuple[int, ...] | None = (3, 4, 5, 6),
) -> torch.Tensor:
    obj = _get_object(env, object_cfg)
    place_pos_w = getattr(env, "pick_place_place_pos_w", None)
    if obj is None or place_pos_w is None:
        return _zeros(env)
    xy_dist = torch.linalg.norm(obj.data.root_pos_w[:, :2] - place_pos_w[:, :2], dim=-1)
    below = place_pos_w[:, 2] - obj.data.root_pos_w[:, 2] - float(margin)
    penalty = torch.clamp(below / max(float(scale), 1.0e-6), min=0.0, max=1.0)
    penalty *= (xy_dist < float(place_xy_threshold)).to(dtype=penalty.dtype)
    return _masked(penalty, env, phases)


def gripper_hold_after_place_penalty(
    env,
    gripper_asset_cfg: SceneEntityCfg | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    open_threshold: float = 0.06,
    place_distance_threshold: float = 0.25,
    object_velocity_threshold: float = 0.35,
    require_object_still_for_place: bool = False,
    phases: tuple[int, ...] | None = (4,),
) -> torch.Tensor:
    placed = _place_success_bool(
        env,
        object_cfg,
        place_distance_threshold,
        object_velocity_threshold,
        require_object_still=require_object_still_for_place,
    )
    gripper_open = _gripper_open_bool(env, gripper_asset_cfg, open_threshold)
    if gripper_open is None:
        return _zeros(env)
    reward = (placed & (~gripper_open)).to(dtype=torch.float32)
    return _masked(reward, env, phases)


def gripper_hold_near_place_penalty(
    env,
    gripper_asset_cfg: SceneEntityCfg | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    open_threshold: float = 0.06,
    place_xy_threshold: float = 0.3,
    max_height_error: float = 0.25,
    phases: tuple[int, ...] | None = (3, 4, 5, 6),
) -> torch.Tensor:
    obj = _get_object(env, object_cfg)
    place_pos_w = getattr(env, "pick_place_place_pos_w", None)
    gripper_open = _gripper_open_bool(env, gripper_asset_cfg, open_threshold)
    if obj is None or place_pos_w is None or gripper_open is None:
        return _zeros(env)
    xy_dist = torch.linalg.norm(obj.data.root_pos_w[:, :2] - place_pos_w[:, :2], dim=-1)
    height_error = torch.abs(obj.data.root_pos_w[:, 2] - place_pos_w[:, 2])
    near_place = (xy_dist < float(place_xy_threshold)) & (height_error < float(max_height_error))
    reward = (near_place & (~gripper_open)).to(dtype=torch.float32)
    return _masked(reward, env, phases)


def object_hover_over_place_penalty(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    place_distance_threshold: float = 0.25,
    hover_margin: float = 0.06,
    hover_scale: float = 0.12,
    phases: tuple[int, ...] | None = (4, 5),
) -> torch.Tensor:
    obj = _get_object(env, object_cfg)
    place_pos_w = getattr(env, "pick_place_place_pos_w", None)
    if obj is None or place_pos_w is None:
        return _zeros(env)
    dist = torch.linalg.norm(obj.data.root_pos_w - place_pos_w, dim=-1)
    hover = torch.clamp((obj.data.root_pos_w[:, 2] - place_pos_w[:, 2] - float(hover_margin)) / max(float(hover_scale), 1.0e-6), min=0.0, max=1.0)
    reward = hover * (dist < float(place_distance_threshold)).to(dtype=hover.dtype)
    return _masked(reward, env, phases)


def post_place_still_success(
    env,
    ee_asset_cfg: SceneEntityCfg | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    gripper_asset_cfg: SceneEntityCfg | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    open_threshold: float = 0.06,
    distance_threshold: float = 0.12,
    object_velocity_threshold: float = 0.25,
    require_place_height: bool = False,
    place_height_threshold: float = 0.05,
    require_base_still: bool = True,
    require_ee_clear: bool = False,
    ee_object_min_distance: float = 0.12,
    base_velocity_threshold: float = 0.08,
    yaw_velocity_threshold: float = 0.25,
    phases: tuple[int, ...] | None = (5, 6),
) -> torch.Tensor:
    placed = _place_success_bool(env, object_cfg, distance_threshold, object_velocity_threshold)
    if bool(require_place_height):
        obj = _get_object(env, object_cfg)
        place_pos_w = getattr(env, "pick_place_place_pos_w", None)
        if obj is None or place_pos_w is None:
            return _zeros(env)
        height_ok = torch.abs(obj.data.root_pos_w[:, 2] - place_pos_w[:, 2]) < float(place_height_threshold)
        placed &= height_ok
    gripper_open = _gripper_open_bool(env, gripper_asset_cfg, open_threshold)
    if gripper_open is None:
        gripper_open = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    robot: Articulation = env.scene[asset_cfg.name]
    base_planar_speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
    base_yaw_speed = torch.abs(robot.data.root_ang_vel_b[:, 2])
    still = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    if bool(require_base_still):
        still = (base_planar_speed < float(base_velocity_threshold)) & (
            base_yaw_speed < float(yaw_velocity_threshold)
        )
    ee_clear = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    if bool(require_ee_clear):
        if ee_asset_cfg is None:
            return _zeros(env)
        dist = _ee_object_distance(env, ee_asset_cfg, object_cfg)
        if dist is None:
            return _zeros(env)
        ee_clear = dist > float(ee_object_min_distance)
    reward = (placed & gripper_open & still & ee_clear).to(dtype=torch.float32)
    return _masked(reward, env, phases)


def base_retreat(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    phases: tuple[int, ...] | None = (5,),
) -> torch.Tensor:
    place_pos_w = getattr(env, "pick_place_place_pos_w", None)
    if place_pos_w is None:
        return _zeros(env)
    robot: Articulation = env.scene[asset_cfg.name]
    dist_sq = torch.sum(torch.square(robot.data.root_pos_w[:, :2] - place_pos_w[:, :2]), dim=-1)
    return _masked(torch.clamp(dist_sq, max=1.0), env, phases)


def ee_retreat(
    env,
    ee_asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    phases: tuple[int, ...] | None = (5,),
) -> torch.Tensor:
    dist = _ee_object_distance(env, ee_asset_cfg, object_cfg)
    if dist is None:
        return _zeros(env)
    reward = 1.0 - torch.exp(-5.0 * torch.clamp(dist, max=1.0))
    return _masked(reward, env, phases)


def complete_success(
    env,
    ee_asset_cfg: SceneEntityCfg,
    gripper_asset_cfg: SceneEntityCfg | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    open_threshold: float = 0.06,
    place_distance_threshold: float = 0.25,
    object_velocity_threshold: float = 0.35,
    require_object_still_for_place: bool = True,
    release_distance_threshold: float = 0.12,
    retreat_distance_threshold: float = 0.25,
    retreat_bonus: float = 2.0,
    require_still: bool = False,
    base_velocity_threshold: float = 0.08,
    yaw_velocity_threshold: float = 0.25,
    phases: tuple[int, ...] | None = (5,),
) -> torch.Tensor:
    placed = _place_success_bool(
        env,
        object_cfg,
        place_distance_threshold,
        object_velocity_threshold,
        require_object_still=require_object_still_for_place,
    )
    dist = _ee_object_distance(env, ee_asset_cfg, object_cfg)
    if dist is None:
        return _zeros(env)
    gripper_open = _gripper_open_bool(env, gripper_asset_cfg, open_threshold)
    if gripper_open is None:
        gripper_open = dist > float(release_distance_threshold)
    release_success = placed & gripper_open
    if bool(require_still):
        robot: Articulation = env.scene[asset_cfg.name]
        base_planar_speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
        base_yaw_speed = torch.abs(robot.data.root_ang_vel_b[:, 2])
        release_success &= (base_planar_speed < float(base_velocity_threshold)) & (
            base_yaw_speed < float(yaw_velocity_threshold)
        )
    retreat_success = placed & (dist > float(retreat_distance_threshold))
    reward = release_success.to(dtype=torch.float32)
    if float(retreat_bonus) != 0.0:
        reward += float(retreat_bonus) * retreat_success.to(dtype=torch.float32)
    return _masked(reward, env, phases)


def virtual_complete_success(
    env,
    phases: tuple[int, ...] | None = (2,),
) -> torch.Tensor:
    success = getattr(env, "pick_place_virtual_success", None)
    if success is None:
        return _zeros(env)
    reward = success.to(device=env.device, dtype=torch.float32)
    return _masked(reward, env, phases)


def phase2_hold_success(
    env,
    phases: tuple[int, ...] | None = (2,),
) -> torch.Tensor:
    condition = getattr(env, "pick_place_phase2_success_condition", None)
    if condition is None:
        return _zeros(env)
    reward = condition.to(device=env.device, dtype=torch.float32)
    return _masked(reward, env, phases)


def grasp_stable_progress(
    env,
    stable_time_s: float = 0.08,
    phases: tuple[int, ...] | None = (1,),
) -> torch.Tensor:
    counter = getattr(env, "pick_place_grasp_stable_counter", None)
    if counter is None or counter.shape != (env.num_envs,):
        return _zeros(env)
    required_steps = max(1, int(float(stable_time_s) / float(env.step_dt)))
    reward = torch.clamp(counter.to(dtype=torch.float32) / float(required_steps), min=0.0, max=1.0)
    return _masked(reward, env, phases)


def phase_progress(
    env,
    max_phase: int = 5,
    one_shot: bool = False,
    normalize: bool = True,
    phases: tuple[int, ...] | None = None,
) -> torch.Tensor:
    phase = getattr(env, "pick_place_phase", None)
    if phase is None:
        return _zeros(env)
    phase = phase.to(device=env.device, dtype=torch.long).clamp(min=0, max=int(max_phase))
    if not one_shot:
        reward = phase.to(dtype=torch.float32)
        if normalize:
            reward = reward / max(float(max_phase), 1.0)
        return _masked(reward, env, phases)

    paid = getattr(env, "pick_place_phase_progress_paid", None)
    if paid is None or paid.shape != (env.num_envs,):
        paid = torch.zeros((env.num_envs,), device=env.device, dtype=torch.long)
        setattr(env, "pick_place_phase_progress_paid", paid)
    mask = _phase_mask(env, phases, phase)
    delta = torch.clamp(phase - paid, min=0)
    reward = delta.to(dtype=torch.float32)
    if normalize:
        reward = reward / max(float(max_phase), 1.0)
    reward = reward * mask.to(dtype=reward.dtype)
    update_mask = (delta > 0) & mask
    paid[update_mask] = phase[update_mask]
    return _masked(reward, env, phases)


def phase_transition_bonus(
    env,
    bonuses: tuple[float, ...] | list[float] = (40.0, 80.0, 100.0, 100.0, 120.0),
    phases: tuple[int, ...] | list[int] | None = None,
) -> torch.Tensor:
    phase = getattr(env, "pick_place_phase", None)
    if phase is None:
        return _zeros(env)
    phase = phase.to(device=env.device, dtype=torch.long).clamp(min=0, max=len(bonuses))

    paid = getattr(env, "pick_place_phase_transition_bonus_paid", None)
    if paid is None or paid.shape != (env.num_envs,):
        paid = torch.zeros((env.num_envs,), device=env.device, dtype=torch.long)
        setattr(env, "pick_place_phase_transition_bonus_paid", paid)

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    phase_filter = None if phases is None else {int(phase_id) for phase_id in phases}
    for phase_id, bonus in enumerate(bonuses, start=1):
        if phase_filter is not None and phase_id not in phase_filter:
            continue
        newly_reached = (phase >= phase_id) & (paid < phase_id)
        reward[newly_reached] += float(bonus)

    reached = phase > paid
    paid[reached] = phase[reached]
    return reward


def ee_to_place_progress(
    env,
    ee_asset_cfg: SceneEntityCfg,
    phases: tuple[int, ...] | None = (1, 2),
) -> torch.Tensor:
    place_pos_w = getattr(env, "pick_place_place_pos_w", None)
    if place_pos_w is None:
        return _zeros(env)
    ee_b = ee_position_in_base(env, ee_asset_cfg)
    place_b = place_position_in_base(env)
    dist = torch.linalg.norm(ee_b - place_b, dim=-1)

    prev = getattr(env, "pick_place_prev_ee_place_dist", None)
    if prev is None or prev.shape != (env.num_envs,):
        prev = torch.full((env.num_envs,), float("nan"), device=env.device, dtype=torch.float32)
        setattr(env, "pick_place_prev_ee_place_dist", prev)

    mask = _phase_mask(env, phases, dist)
    progress = torch.where(torch.isfinite(prev), prev - dist, torch.zeros_like(dist))
    progress = torch.clamp(progress, min=-0.25, max=0.25)
    prev[mask] = dist[mask]
    prev[~mask] = float("nan")
    return progress * mask.to(dtype=progress.dtype)


def ee_to_pick_progress(
    env,
    ee_asset_cfg: SceneEntityCfg,
    phases: tuple[int, ...] | None = (0,),
) -> torch.Tensor:
    pick_pos_w = getattr(env, "pick_place_pick_pos_w", None)
    if pick_pos_w is None:
        return _zeros(env)
    ee_b = ee_position_in_base(env, ee_asset_cfg)
    robot: Articulation = env.scene["robot"]
    pick_b = quat_apply_inverse(robot.data.root_quat_w, pick_pos_w - robot.data.root_pos_w)
    dist = torch.linalg.norm(ee_b - pick_b, dim=-1)

    prev = getattr(env, "pick_place_prev_ee_pick_dist", None)
    if prev is None or prev.shape != (env.num_envs,):
        prev = torch.full((env.num_envs,), float("nan"), device=env.device, dtype=torch.float32)
        setattr(env, "pick_place_prev_ee_pick_dist", prev)

    mask = _phase_mask(env, phases, dist)
    progress = torch.where(torch.isfinite(prev), prev - dist, torch.zeros_like(dist))
    progress = torch.clamp(progress, min=-0.25, max=0.25)
    prev[mask] = dist[mask]
    prev[~mask] = float("nan")
    return progress * mask.to(dtype=progress.dtype)


def pick_reached_success(
    env,
    ee_asset_cfg: SceneEntityCfg,
    distance_threshold: float = 0.14,
    phases: tuple[int, ...] | None = (0,),
) -> torch.Tensor:
    pick_pos_w = getattr(env, "pick_place_pick_pos_w", None)
    if pick_pos_w is None:
        return _zeros(env)
    ee_b = ee_position_in_base(env, ee_asset_cfg)
    robot: Articulation = env.scene["robot"]
    pick_b = quat_apply_inverse(robot.data.root_quat_w, pick_pos_w - robot.data.root_pos_w)
    dist = torch.linalg.norm(ee_b - pick_b, dim=-1)
    reward = (dist < float(distance_threshold)).to(dtype=torch.float32)
    return _masked(reward, env, phases)


def high_action_rate_l2(env) -> torch.Tensor:
    current = getattr(env, "high_level_previous_action", None)
    action_term = getattr(getattr(env, "action_manager", None), "_terms", {}).get("high_level", None)
    if current is None or action_term is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.sum(torch.square(action_term.raw_actions - current), dim=-1)


def high_action_acc_l2(env) -> torch.Tensor:
    prev = getattr(env, "high_level_previous_action", None)
    prev_prev = getattr(env, "high_level_previous_previous_action", None)
    action_term = getattr(getattr(env, "action_manager", None), "_terms", {}).get("high_level", None)
    if prev is None or prev_prev is None or action_term is None:
        return _zeros(env)
    return torch.sum(torch.square(action_term.raw_actions - 2.0 * prev + prev_prev), dim=-1)


def arm_action_rate_l2(env, phases: tuple[int, ...] | None = None) -> torch.Tensor:
    prev = getattr(env, "high_level_previous_action", None)
    action_term = getattr(getattr(env, "action_manager", None), "_terms", {}).get("high_level", None)
    if prev is None or action_term is None:
        return _zeros(env)
    reward = torch.sum(torch.square(action_term.raw_actions[:, 5:] - prev[:, 5:]), dim=-1)
    return _masked(reward, env, phases)


def base_orientation_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    phases: tuple[int, ...] | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    reward = torch.square(roll) + torch.square(pitch)
    return _masked(reward, env, phases)


def base_height_l2(
    env,
    target_height: float = 0.42,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    phases: tuple[int, ...] | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_pos_w[:, 2] - float(target_height))
    return _masked(reward, env, phases)


def arm_joint_vel_l2(
    env,
    asset_cfg: SceneEntityCfg,
    phases: tuple[int, ...] | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=-1)
    return _masked(reward, env, phases)


def arm_joint_torque_l2(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=-1)


def arm_nominal_pose_l2(
    env,
    asset_cfg: SceneEntityCfg,
    phases: tuple[int, ...] | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(
        torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=-1,
    )
    return _masked(reward, env, phases)


def arm_joint_limit_margin(
    env,
    asset_cfg: SceneEntityCfg,
    margin_ratio: float = 0.05,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, :]
    lower = limits[..., 0]
    upper = limits[..., 1]
    margin = (upper - lower).clamp_min(1.0e-6) * float(margin_ratio)
    lower_violation = torch.clamp(lower + margin - joint_pos, min=0.0) / margin
    upper_violation = torch.clamp(joint_pos - (upper - margin), min=0.0) / margin
    return torch.sum(torch.square(lower_violation) + torch.square(upper_violation), dim=-1)


def base_vertical_vel_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_w[:, 2])


def base_roll_pitch_ang_vel_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=-1)


def base_planar_velocity_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    yaw_weight: float = 0.25,
    phases: tuple[int, ...] | None = (5,),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    lin_xy = torch.sum(torch.square(asset.data.root_lin_vel_w[:, :2]), dim=-1)
    yaw = torch.square(asset.data.root_ang_vel_b[:, 2])
    return _masked(lin_xy + float(yaw_weight) * yaw, env, phases)


def velocity_command_rate_l2(env) -> torch.Tensor:
    current = getattr(env, "high_level_base_command", None)
    prev = getattr(env, "high_level_previous_base_command", None)
    if current is None or prev is None:
        return _zeros(env)
    return torch.sum(torch.square(current[:, :3] - prev[:, :3]), dim=-1)


def velocity_command_acc_l2(env) -> torch.Tensor:
    current = getattr(env, "high_level_base_command", None)
    prev = getattr(env, "high_level_previous_base_command", None)
    prev_prev = getattr(env, "high_level_previous_previous_base_command", None)
    if current is None or prev is None or prev_prev is None:
        return _zeros(env)
    return torch.sum(torch.square(current[:, :3] - 2.0 * prev[:, :3] + prev_prev[:, :3]), dim=-1)


def velocity_command_norm_l2(
    env,
    phases: tuple[int, ...] | None = (5,),
) -> torch.Tensor:
    current = getattr(env, "high_level_base_command", None)
    if current is None:
        return _zeros(env)
    reward = torch.sum(torch.square(current[:, :3]), dim=-1)
    return _masked(reward, env, phases)


def base_height_pitch_command_l2(
    env,
    target_height: float = 0.42,
    target_pitch: float = 0.0,
    height_weight: float = 1.0,
    pitch_weight: float = 1.0,
    phases: tuple[int, ...] | None = (6,),
) -> torch.Tensor:
    current = getattr(env, "high_level_base_command", None)
    if current is None:
        return _zeros(env)
    height_err = torch.square(current[:, 3] - float(target_height))
    pitch_err = torch.square(current[:, 4] - float(target_pitch))
    reward = float(height_weight) * height_err + float(pitch_weight) * pitch_err
    return _masked(reward, env, phases)


def body_command_norm_l2(env) -> torch.Tensor:
    current = getattr(env, "high_level_base_command", None)
    if current is None:
        return _zeros(env)
    return torch.sum(torch.square(current[:, 3:5]), dim=-1)


def body_command_rate_l2(env) -> torch.Tensor:
    current = getattr(env, "high_level_base_command", None)
    prev = getattr(env, "high_level_previous_base_command", None)
    if current is None or prev is None:
        return _zeros(env)
    return torch.sum(torch.square(current[:, 3:5] - prev[:, 3:5]), dim=-1)


def body_command_acc_l2(env) -> torch.Tensor:
    current = getattr(env, "high_level_base_command", None)
    prev = getattr(env, "high_level_previous_base_command", None)
    prev_prev = getattr(env, "high_level_previous_previous_base_command", None)
    if current is None or prev is None or prev_prev is None:
        return _zeros(env)
    return torch.sum(torch.square(current[:, 3:5] - 2.0 * prev[:, 3:5] + prev_prev[:, 3:5]), dim=-1)


def excessive_pitch_l2(env, threshold: float = 0.25) -> torch.Tensor:
    current = getattr(env, "high_level_base_command", None)
    if current is None:
        return _zeros(env)
    excess = torch.clamp(torch.abs(current[:, 4]) - float(threshold), min=0.0)
    return torch.square(excess)


def excessive_height_change_l2(env, nominal_height: float = 0.35, threshold: float = 0.1) -> torch.Tensor:
    current = getattr(env, "high_level_base_command", None)
    if current is None:
        return _zeros(env)
    excess = torch.clamp(torch.abs(current[:, 3] - float(nominal_height)) - float(threshold), min=0.0)
    return torch.square(excess)


def support_clearance_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    support_names: tuple[str, ...] = ("pick_support", "place_support"),
    min_distance: float = 0.45,
    phases: tuple[int, ...] | None = (3,),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    base_xy = asset.data.root_pos_w[:, :2]
    for support_name in support_names:
        try:
            support = env.scene[support_name]
        except Exception:
            continue
        dist = torch.linalg.norm(base_xy - support.data.root_pos_w[:, :2], dim=-1)
        violation = torch.clamp(float(min_distance) - dist, min=0.0) / max(float(min_distance), 1.0e-6)
        penalty += torch.square(violation)
    return _masked(penalty, env, phases)


def object_stability_l2(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    obj: RigidObject | None = _get_object(env, object_cfg)
    if obj is None:
        return torch.zeros(env.num_envs, device=env.device)
    roll, pitch, _ = euler_xyz_from_quat(obj.data.root_quat_w)
    return torch.square(roll) + torch.square(pitch)


def object_tilt_near_place_l2(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    distance_threshold: float = 0.25,
) -> torch.Tensor:
    obj: RigidObject | None = _get_object(env, object_cfg)
    if obj is None:
        return _zeros(env)
    dist = _object_place_distance(env, object_cfg)
    near = dist < float(distance_threshold)
    tilt = object_stability_l2(env, object_cfg)
    return tilt * near.to(dtype=tilt.dtype)


def ee_object_relative_velocity_l2(
    env,
    ee_asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    obj: RigidObject | None = _get_object(env, object_cfg)
    if obj is None:
        return torch.zeros(env.num_envs, device=env.device)
    ee_vel_w = _ee_linear_velocity_w(env, ee_asset_cfg)
    robot: Articulation = env.scene[robot_asset_cfg.name]
    rel_vel_b = quat_apply_inverse(robot.data.root_quat_w, ee_vel_w - obj.data.root_lin_vel_w)
    return torch.sum(torch.square(rel_vel_b), dim=-1)

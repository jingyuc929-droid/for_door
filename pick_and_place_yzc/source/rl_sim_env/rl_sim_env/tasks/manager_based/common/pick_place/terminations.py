"""Termination terms for hierarchical pick-and-place tasks."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply

from .observations import ee_position_in_base, place_position_in_base


def object_fallen(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    min_height: float = 0.12,
    phases: tuple[int, ...] | list[int] | None = None,
) -> torch.Tensor:
    """Terminate an environment when the manipulated object has fallen below the support height."""
    try:
        obj: RigidObject = env.scene[object_cfg.name]
    except Exception:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    fallen = obj.data.root_pos_w[:, 2] < float(min_height)
    if phases is None:
        return fallen
    phase = getattr(env, "pick_place_phase", None)
    if phase is None:
        return fallen
    phase = phase.to(device=env.device, dtype=torch.long)
    mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for phase_id in phases:
        mask |= phase == int(phase_id)
    return fallen & mask


def _ensure_bool_attr(env, name: str) -> torch.Tensor:
    value = getattr(env, name, None)
    if value is None or value.shape != (env.num_envs,):
        value = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        setattr(env, name, value)
    return value


def _ensure_long_attr(env, name: str) -> torch.Tensor:
    value = getattr(env, name, None)
    if value is None or value.shape != (env.num_envs,):
        value = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        setattr(env, name, value)
    return value


def virtual_place_success_condition(
    env,
    ee_asset_cfg: SceneEntityCfg,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    phase_id: int = 2,
    ee_place_threshold: float = 0.12,
    base_place_threshold: float = 0.45,
    heading_threshold: float = 0.5,
    roll_limit: float = 0.4,
    pitch_limit: float = 0.5,
) -> torch.Tensor:
    phase = getattr(env, "pick_place_phase", None)
    if phase is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    place_pos_w = getattr(env, "pick_place_place_pos_w", None)
    if place_pos_w is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    robot: Articulation = env.scene[robot_asset_cfg.name]
    ee_b = ee_position_in_base(env, ee_asset_cfg, robot_asset_cfg)
    place_b = place_position_in_base(env)
    ee_ok = torch.linalg.norm(ee_b - place_b, dim=-1) < float(ee_place_threshold)

    base_dist = torch.linalg.norm(robot.data.root_pos_w[:, :2] - place_pos_w[:, :2], dim=-1)
    base_ok = base_dist < float(base_place_threshold)

    direction = place_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    direction_norm = torch.linalg.norm(direction, dim=-1, keepdim=True).clamp_min(1.0e-6)
    direction = direction / direction_norm
    forward_w = quat_apply(
        robot.data.root_quat_w,
        torch.tensor((1.0, 0.0, 0.0), device=env.device, dtype=torch.float32).repeat(env.num_envs, 1),
    )
    heading_cos = torch.sum(forward_w[:, :2] * direction, dim=-1).clamp(min=-1.0, max=1.0)
    heading_error = torch.arccos(heading_cos)
    heading_ok = heading_error < float(heading_threshold)

    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    posture_ok = (torch.abs(roll) < float(roll_limit)) & (torch.abs(pitch) < float(pitch_limit))
    phase_ok = phase.to(device=env.device, dtype=torch.long) == int(phase_id)
    return phase_ok & ee_ok & base_ok & heading_ok & posture_ok


def virtual_place_success_hold(
    env,
    ee_asset_cfg: SceneEntityCfg,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    phase_id: int = 2,
    ee_place_threshold: float = 0.12,
    base_place_threshold: float = 0.45,
    heading_threshold: float = 0.5,
    roll_limit: float = 0.4,
    pitch_limit: float = 0.5,
    hold_time_s: float = 0.3,
) -> torch.Tensor:
    counter = _ensure_long_attr(env, "pick_place_phase2_hold_counter")
    phase2_steps = _ensure_long_attr(env, "pick_place_phase2_step_counter")
    condition_buf = _ensure_bool_attr(env, "pick_place_phase2_success_condition")
    success_buf = _ensure_bool_attr(env, "pick_place_virtual_success")

    phase = getattr(env, "pick_place_phase", None)
    if phase is None:
        phase2_steps.zero_()
        counter.zero_()
        condition_buf.zero_()
        success_buf.zero_()
        return success_buf

    phase_ok = phase.to(device=env.device, dtype=torch.long) == int(phase_id)
    condition = virtual_place_success_condition(
        env,
        ee_asset_cfg=ee_asset_cfg,
        robot_asset_cfg=robot_asset_cfg,
        phase_id=phase_id,
        ee_place_threshold=ee_place_threshold,
        base_place_threshold=base_place_threshold,
        heading_threshold=heading_threshold,
        roll_limit=roll_limit,
        pitch_limit=pitch_limit,
    )
    phase2_steps[phase_ok] += 1
    phase2_steps[~phase_ok] = 0
    counter[condition] += 1
    counter[~condition] = 0

    required_steps = max(1, int(float(hold_time_s) / float(env.step_dt)))
    success = counter >= required_steps
    condition_buf[:] = condition
    success_buf[:] = success
    return success

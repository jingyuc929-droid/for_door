# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, Sequence

import carb
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.events import _randomize_prop_by_op, _validate_scale_range
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_term_cfg import EventTermCfg
from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_body_collision_enabled(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | list[int] | None,
    body_names: list[str] | tuple[str, ...],
    robot_prim_path: str = "Robot",
    collision_enabled: bool = True,
):
    """Enable/disable collision for selected robot bodies across environments.

    Notes:
    - This applies at USD collider level under `{env_prim_path}/{robot_prim_path}/{body_name}`.
    - Disabling collisions here means the body won't collide with any object (including ground).
    """
    if not body_names:
        return

    if env_ids is None or isinstance(env_ids, slice):
        env_indices = list(range(env.scene.num_envs))
    elif isinstance(env_ids, torch.Tensor):
        env_indices = env_ids.detach().cpu().tolist()
    else:
        env_indices = list(env_ids)

    collision_cfg = sim_utils.CollisionPropertiesCfg(
        collision_enabled=bool(collision_enabled)
    )

    failed_prim_paths: list[str] = []
    for env_id in env_indices:
        env_prim_path = env.scene.env_prim_paths[int(env_id)]
        for body_name in body_names:
            prim_path = f"{env_prim_path}/{robot_prim_path}/{body_name}"
            ok = sim_utils.modify_collision_properties(prim_path, collision_cfg)
            if not ok:
                failed_prim_paths.append(prim_path)

    if failed_prim_paths:
        preview = ", ".join(failed_prim_paths[:5])
        carb.log_warn(
            "[set_body_collision_enabled] Failed to modify collision for some prims. "
            f"Examples: {preview}"
        )


def _ensure_push_force_visualizer(env: "ManagerBasedEnv"):
    """Lazy-init visualization marker for push-force (safe in headless)."""
    if hasattr(env, "event_push_force_visualizer"):
        return
    # If rendering not available, skip.
    try:
        is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    except Exception:
        is_rendering = False
    if not is_rendering:
        env.event_push_force_visualizer = None
        return

    try:
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG as _ARROW_CFG
    except Exception:
        try:
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
            from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG as _ARROW_CFG
        except Exception:
            env.event_push_force_visualizer = None
            return

    cfg: VisualizationMarkersCfg = _ARROW_CFG.replace(prim_path="/Visuals/Events/push_force")
    cfg.markers["arrow"].scale = (0.5, 0.15, 0.15)
    try:
        env.event_push_force_visualizer = VisualizationMarkers(cfg)
        env.event_push_force_visualizer.set_visibility(True)
    except Exception as e:
        carb.log_warn(
            "[push_force] Failed to create VisualizationMarkers; disabling debug_vis. "
            f"Error: {e}"
        )
        env.event_push_force_visualizer = None


def _ensure_push_yaw_torque_visualizer(env: "ManagerBasedEnv"):
    """Lazy-init visualization markers for yaw-torque (safe in headless)."""
    if hasattr(env, "event_push_yaw_torque_visualizer"):
        return
    try:
        is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    except Exception:
        is_rendering = False
    if not is_rendering:
        env.event_push_yaw_torque_visualizer = None
        return

    try:
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG as _ARROW_CFG
    except Exception:
        try:
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
            from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG as _ARROW_CFG
        except Exception:
            env.event_push_yaw_torque_visualizer = None
            return

    cfg: VisualizationMarkersCfg = _ARROW_CFG.replace(prim_path="/Visuals/Events/push_yaw_torque")
    cfg.markers["arrow"].scale = (0.5, 0.22, 0.22)
    try:
        env.event_push_yaw_torque_visualizer = VisualizationMarkers(cfg)
        env.event_push_yaw_torque_visualizer.set_visibility(True)
    except Exception as e:
        carb.log_warn(
            "[push_yaw_torque] Failed to create VisualizationMarkers; disabling debug_vis. "
            f"Error: {e}"
        )
        env.event_push_yaw_torque_visualizer = None


def _ensure_push_force_buffers(env: "ManagerBasedEnv"):
    """Create buffers for external force push event (lazy init)."""
    if hasattr(env, "event_push_force_state"):
        return
    num_envs = env.scene.num_envs
    device = getattr(env, "device", torch.device("cpu"))

    # state: 0 idle, 1 ramp_to_target, 2 hold, 3 ramp_to_zero
    env.event_push_force_state = torch.zeros((num_envs,), device=device, dtype=torch.int64)
    env.event_push_force_curr_xy = torch.zeros((num_envs, 2), device=device, dtype=torch.float32)
    env.event_push_force_buf = torch.zeros((num_envs, 2), device=device, dtype=torch.float32)  # for observations
    env.event_push_force_ramp_start_xy = torch.zeros((num_envs, 2), device=device, dtype=torch.float32)
    env.event_push_force_ramp_target_xy = torch.zeros((num_envs, 2), device=device, dtype=torch.float32)
    env.event_push_force_ramp_elapsed = torch.zeros((num_envs,), device=device, dtype=torch.float32)
    env.event_push_force_ramp_time = torch.zeros((num_envs,), device=device, dtype=torch.float32)
    env.event_push_force_hold_left = torch.zeros((num_envs,), device=device, dtype=torch.float32)
    # per-env pitch offset (rad) sampled at event start; used to tilt force application
    env.event_push_force_pitch_offset_buf = torch.zeros((num_envs,), device=device, dtype=torch.float32)
    # per-env cached terrain slope estimate (sampled/refreshed for terrain_heading)
    env.event_push_force_slope_a_buf = torch.zeros((num_envs,), device=device, dtype=torch.float32)
    env.event_push_force_slope_b_buf = torch.zeros((num_envs,), device=device, dtype=torch.float32)
    # low-frequency slope refresh timer (seconds) for terrain_heading
    env.event_push_force_slope_refresh_left_s = torch.zeros((num_envs,), device=device, dtype=torch.float32)


def _ensure_push_yaw_torque_buffers(env: "ManagerBasedEnv"):
    """Create buffers for external yaw-torque push event (lazy init)."""
    if hasattr(env, "event_push_yaw_torque_state"):
        return
    num_envs = env.scene.num_envs
    device = getattr(env, "device", torch.device("cpu"))

    # state: 0 idle, 1 ramp_to_target, 2 hold, 3 ramp_to_zero
    env.event_push_yaw_torque_state = torch.zeros((num_envs,), device=device, dtype=torch.int64)
    env.event_push_yaw_torque_curr_z = torch.zeros((num_envs,), device=device, dtype=torch.float32)
    env.event_push_yaw_torque_buf = torch.zeros((num_envs, 1), device=device, dtype=torch.float32)  # for observations
    env.event_push_yaw_torque_ramp_start_z = torch.zeros((num_envs,), device=device, dtype=torch.float32)
    env.event_push_yaw_torque_ramp_target_z = torch.zeros((num_envs,), device=device, dtype=torch.float32)
    env.event_push_yaw_torque_ramp_elapsed = torch.zeros((num_envs,), device=device, dtype=torch.float32)
    env.event_push_yaw_torque_ramp_time = torch.zeros((num_envs,), device=device, dtype=torch.float32)
    env.event_push_yaw_torque_hold_left = torch.zeros((num_envs,), device=device, dtype=torch.float32)


def reset_push_force(env: "ManagerBasedEnv", env_ids: torch.Tensor | list[int] | None = None):
    """Reset internal push-force buffers (and stop applying forces)."""
    _ensure_push_force_buffers(env)

    if env_ids is None:
        env_ids = slice(None)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, dtype=torch.long, device=env.device)

    env.event_push_force_state[env_ids] = 0
    env.event_push_force_curr_xy[env_ids] = 0.0
    env.event_push_force_buf[env_ids] = 0.0
    env.event_push_force_ramp_start_xy[env_ids] = 0.0
    env.event_push_force_ramp_target_xy[env_ids] = 0.0
    env.event_push_force_ramp_elapsed[env_ids] = 0.0
    env.event_push_force_ramp_time[env_ids] = 0.0
    env.event_push_force_hold_left[env_ids] = 0.0
    env.event_push_force_pitch_offset_buf[env_ids] = 0.0
    env.event_push_force_slope_a_buf[env_ids] = 0.0
    env.event_push_force_slope_b_buf[env_ids] = 0.0
    env.event_push_force_slope_refresh_left_s[env_ids] = 0.0


def reset_push_yaw_torque(env: "ManagerBasedEnv", env_ids: torch.Tensor | list[int] | None = None):
    """Reset internal yaw-torque buffers (and stop applying torques)."""
    _ensure_push_yaw_torque_buffers(env)

    if env_ids is None:
        env_ids = slice(None)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, dtype=torch.long, device=env.device)

    env.event_push_yaw_torque_state[env_ids] = 0
    env.event_push_yaw_torque_curr_z[env_ids] = 0.0
    env.event_push_yaw_torque_buf[env_ids] = 0.0
    env.event_push_yaw_torque_ramp_start_z[env_ids] = 0.0
    env.event_push_yaw_torque_ramp_target_z[env_ids] = 0.0
    env.event_push_yaw_torque_ramp_elapsed[env_ids] = 0.0
    env.event_push_yaw_torque_ramp_time[env_ids] = 0.0
    env.event_push_yaw_torque_hold_left[env_ids] = 0.0


def start_push_force_xy_base(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    force_xy_range: dict[str, tuple[float, float]],
    duration_range_s: tuple[float, float],
    ramp_time_s: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # force direction frame control
    force_frame: Literal["yaw_horizontal", "terrain_heading"] = "yaw_horizontal",
    pitch_offset_range_rad: tuple[float, float] = (0.0, 0.0),
    # terrain-slope mode settings (uses height scanner hits to estimate local slope)
    terrain_sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    # If slope is too small, avoid unstable yaw (atan2(0,0)) by treating terrain as flat.
    slope_min_grad: float = 0.02,
    # Low-frequency refresh of terrain slope estimates (seconds). Set <=0 to disable (lock per event).
    slope_update_period_s: float = 0.0,
    # debug visualization (optional)
    debug_vis: bool = False,
    debug_vis_height: float = 0.6,
    debug_vis_force_to_length: float = 0.1,
):
    """Start (or change) a 2D force push for a duration, with smooth ramping.

    The push is sampled as a 2D vector (fx, fy) (in the selected application frame) and stored in
    ``env.event_push_force_curr_xy`` / ``env.event_push_force_buf`` for observation/reward purposes.

    Note on buffers:
    - ``env.event_push_force_curr_xy``: the current sampled force components (fx, fy) in the chosen force frame.
    - ``env.event_push_force_buf``: the **applied world-frame force projected into the trunk/body frame** (x/y),
      i.e., ``force_b = quat_apply_inverse(body_quat_w, force_w)`` then keep ``force_b[:, :2]``.

    Force sampling:
    - We sample fx, fy independently from the given ranges.
      Example: ``force_xy_range={"x": (0, 80), "y": (0, 0)}`` gives a forward-only pull.

    Force direction/application frames:
    - ``force_frame="yaw_horizontal"`` (default): interpret (fx, fy) in the **yaw-only horizontal frame**
      and apply it in world frame. If ``pitch_offset_range_rad`` is non-zero, the force is tilted by the sampled
      pitch offset (introducing a Z component).
    - ``force_frame="terrain_heading"``: keep yaw aligned with robot heading, but constrain the force direction
      to the local terrain tangent plane estimated from the height scanner hits (more stable than using base pitch).
    """
    _ensure_push_force_buffers(env)

    # resolve asset/body ids once and cache on env
    if not hasattr(env, "event_push_force_asset_cfg"):
        env.event_push_force_asset_cfg = asset_cfg
        try:
            env.event_push_force_asset_cfg.resolve(env.scene)
        except Exception:
            pass
    asset_cfg = env.event_push_force_asset_cfg

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    device = asset.device
    env_ids = env_ids.to(device=device)

    # store frame config on env for application stage
    env.event_push_force_frame = force_frame
    env.event_push_force_slope_min_grad = float(slope_min_grad)
    env.event_push_force_slope_update_period_s = float(slope_update_period_s)
    # cache terrain sensor cfg (best-effort resolve)
    env.event_push_force_terrain_sensor_cfg = terrain_sensor_cfg
    try:
        env.event_push_force_terrain_sensor_cfg.resolve(env.scene)
    except Exception:
        pass
    off_min, off_max = pitch_offset_range_rad
    pitch_offsets = math_utils.sample_uniform(
        torch.tensor(float(off_min), device=device),
        torch.tensor(float(off_max), device=device),
        (len(env_ids),),
        device=device,
    )
    env.event_push_force_pitch_offset_buf[env_ids] = pitch_offsets.to(
        device=env.event_push_force_pitch_offset_buf.device, dtype=env.event_push_force_pitch_offset_buf.dtype
    )

    # For terrain_heading, sample slope ONCE at event start (then optionally refresh at low frequency).
    if force_frame == "terrain_heading":
        sensor_cfg = terrain_sensor_cfg
        try:
            sensor_cfg.resolve(env.scene)
        except Exception:
            pass
        # Use the same body we apply forces to (first body_id in cfg).
        try:
            asset_cfg.resolve(env.scene)
        except Exception:
            pass
        body_ids = asset_cfg.body_ids if hasattr(asset_cfg, "body_ids") else None
        body_id = int(body_ids[0].item()) if isinstance(body_ids, torch.Tensor) else int(body_ids[0])
        body_pos_w = asset.data.body_pos_w[:, body_id].to(device=device, dtype=torch.float32)
        body_quat_w = asset.data.body_quat_w[:, body_id].to(device=device, dtype=torch.float32)

        a, b = _estimate_local_terrain_plane_gradients_yaw_frame(
            env=env, body_pos_w=body_pos_w, body_quat_w=body_quat_w, sensor_cfg=sensor_cfg
        )
        # guard: if slope almost flat, force direction should not spin; pin to 0 slope
        grad_norm = torch.sqrt(a * a + b * b)
        flat = grad_norm < float(slope_min_grad)
        a = torch.where(flat, torch.zeros_like(a), a)
        b = torch.where(flat, torch.zeros_like(b), b)

        env.event_push_force_slope_a_buf[env_ids] = a[env_ids].to(
            device=env.event_push_force_slope_a_buf.device, dtype=env.event_push_force_slope_a_buf.dtype
        )
        env.event_push_force_slope_b_buf[env_ids] = b[env_ids].to(
            device=env.event_push_force_slope_b_buf.device, dtype=env.event_push_force_slope_b_buf.dtype
        )
        # initialize refresh timers
        period = float(slope_update_period_s)
        if period > 0.0:
            env.event_push_force_slope_refresh_left_s[env_ids] = period
        else:
            env.event_push_force_slope_refresh_left_s[env_ids] = 0.0

    # ----------------------------
    # sample target force (2D)
    # ----------------------------
    fx_min, fx_max = force_xy_range.get("x", (0.0, 0.0))
    fy_min, fy_max = force_xy_range.get("y", (0.0, 0.0))
    fx = math_utils.sample_uniform(
        torch.tensor(float(fx_min), device=device),
        torch.tensor(float(fx_max), device=device),
        (len(env_ids),),
        device=device,
    )
    fy = math_utils.sample_uniform(
        torch.tensor(float(fy_min), device=device),
        torch.tensor(float(fy_max), device=device),
        (len(env_ids),),
        device=device,
    )
    target_xy = torch.stack([fx, fy], dim=-1)

    # sample hold duration (excluding ramps)
    dur_min, dur_max = duration_range_s
    hold_s = math_utils.sample_uniform(
        torch.tensor(dur_min, device=device),
        torch.tensor(dur_max, device=device),
        (len(env_ids),),
        device=device,
    )

    # schedule ramp to new target from current
    env.event_push_force_ramp_start_xy[env_ids] = env.event_push_force_curr_xy[env_ids]
    env.event_push_force_ramp_target_xy[env_ids] = target_xy
    env.event_push_force_ramp_elapsed[env_ids] = 0.0
    env.event_push_force_ramp_time[env_ids] = float(max(ramp_time_s, 0.0))
    env.event_push_force_hold_left[env_ids] = hold_s
    env.event_push_force_state[env_ids] = 1

    # setup debug visualization if requested
    if debug_vis:
        _ensure_push_force_visualizer(env)
        env.event_push_force_debug_vis = True
        env.event_push_force_debug_vis_height = float(debug_vis_height)
        env.event_push_force_debug_vis_force_to_length = float(debug_vis_force_to_length)


def _refresh_cached_terrain_slope(
    env: "ManagerBasedEnv",
    asset: RigidObject | Articulation,
    body_id: int,
    env_ids: torch.Tensor,
    slope_min_grad: float,
):
    """Recompute and update cached terrain slope buffers for selected env_ids."""
    sensor_cfg = getattr(env, "event_push_force_terrain_sensor_cfg", SceneEntityCfg("height_scanner"))
    try:
        sensor_cfg.resolve(env.scene)
    except Exception:
        pass

    device = asset.device
    body_pos_w = asset.data.body_pos_w[:, body_id].to(device=device, dtype=torch.float32)
    body_quat_w = asset.data.body_quat_w[:, body_id].to(device=device, dtype=torch.float32)
    a, b = _estimate_local_terrain_plane_gradients_yaw_frame(
        env=env, body_pos_w=body_pos_w, body_quat_w=body_quat_w, sensor_cfg=sensor_cfg
    )
    grad_norm = torch.sqrt(a * a + b * b)
    flat = grad_norm < float(slope_min_grad)
    a = torch.where(flat, torch.zeros_like(a), a)
    b = torch.where(flat, torch.zeros_like(b), b)

    env.event_push_force_slope_a_buf[env_ids] = a[env_ids].to(
        device=env.event_push_force_slope_a_buf.device, dtype=env.event_push_force_slope_a_buf.dtype
    )
    env.event_push_force_slope_b_buf[env_ids] = b[env_ids].to(
        device=env.event_push_force_slope_b_buf.device, dtype=env.event_push_force_slope_b_buf.dtype
    )


def _estimate_local_terrain_plane_gradients_yaw_frame(
    env: "ManagerBasedEnv",
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    sensor_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate local terrain plane gradients (a, b) in the base-yaw frame from height-scanner hits.

    Returns:
      a, b: plane gradients in base-yaw frame, where z ~= a*x + b*y + c.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    hits_w = sensor.data.ray_hits_w  # (N, M, 3)
    if hits_w is None:
        zeros = torch.zeros((env.scene.num_envs,), device=body_pos_w.device, dtype=torch.float32)
        return zeros, zeros

    # transform hit points into base-yaw frame (origin at body position)
    rel_w = hits_w.to(device=body_pos_w.device, dtype=torch.float32) - body_pos_w.unsqueeze(1)
    yaw_q = math_utils.yaw_quat(body_quat_w).to(device=body_pos_w.device, dtype=torch.float32)
    yaw_q_inv = math_utils.quat_inv(yaw_q)

    N, M, _ = rel_w.shape
    rel_w_flat = rel_w.reshape(N * M, 3)
    yaw_q_inv_flat = yaw_q_inv.unsqueeze(1).expand(N, M, 4).reshape(N * M, 4)
    pts_yaw = math_utils.quat_apply(yaw_q_inv_flat, rel_w_flat).reshape(N, M, 3)

    x = pts_yaw[:, :, 0]
    y = pts_yaw[:, :, 1]
    z = pts_yaw[:, :, 2]
    # Robustify against missing ray hits (NaN/Inf). Replace invalid z per-env with the mean of valid samples.
    # This keeps slope estimation stable on complex terrains (pits/overhangs) where some rays may miss.
    valid = torch.isfinite(z)
    valid_count = valid.sum(dim=1).clamp(min=1)
    z_sum = torch.where(valid, z, torch.zeros_like(z)).sum(dim=1)
    z_fill = (z_sum / valid_count).unsqueeze(1)
    z = torch.where(valid, z, z_fill)

    # cache pseudo-inverse for plane fit: z = a*x + b*y + c
    cache_key = "event_push_force_slope_pinv"
    cache_sig_key = "event_push_force_slope_pinv_sig"
    sig = (int(M),)
    if not hasattr(env, cache_key) or getattr(env, cache_sig_key, None) != sig:
        A0 = torch.stack([x[0], y[0], torch.ones_like(x[0])], dim=1)  # (M, 3)
        pinv = torch.linalg.pinv(A0).to(device=body_pos_w.device, dtype=torch.float32)  # (3, M)
        setattr(env, cache_key, pinv)
        setattr(env, cache_sig_key, sig)

    pinv = getattr(env, cache_key)  # (3, M)
    # beta: (N,3) where beta[:,0]=a, beta[:,1]=b
    beta = z @ pinv.T
    a = beta[:, 0]
    b = beta[:, 1]
    return a, b


def _quat_wxyz_from_x_axis_to_vec(vec_w: torch.Tensor) -> torch.Tensor:
    """Quaternion (wxyz) that rotates +X axis onto `vec_w` (world).

    Uses a numerically stable "from-to" quaternion construction.
    Assumes `vec_w` is shape (N, 3). Returns shape (N, 4) in **wxyz** ordering.
    """
    # sanitize invalid vectors to avoid propagating NaNs into visualization
    vec_w = torch.nan_to_num(vec_w, nan=0.0, posinf=0.0, neginf=0.0)
    v1 = torch.nn.functional.normalize(vec_w, dim=-1, eps=1.0e-8)
    # v0 = +X axis
    v0 = torch.zeros_like(v1)
    v0[:, 0] = 1.0

    dot = torch.sum(v0 * v1, dim=-1)  # (N,)
    cross = torch.cross(v0, v1, dim=-1)  # (N,3)

    # If vectors are nearly opposite, pick a 180-deg rotation around +Y (maps +X -> -X).
    opp = dot < -1.0 + 1.0e-6
    q = torch.zeros((v1.shape[0], 4), device=v1.device, dtype=v1.dtype)
    if opp.any():
        # (w,x,y,z) = (0,0,1,0)
        q[opp, 2] = 1.0

    safe = ~opp
    if safe.any():
        w = 1.0 + dot[safe]
        xyz = cross[safe]
        q_safe = torch.cat([w.unsqueeze(-1), xyz], dim=-1)  # (w,x,y,z)
        q_safe = torch.nn.functional.normalize(q_safe, dim=-1, eps=1.0e-8)
        q[safe] = q_safe

    return q

def start_push_yaw_torque_z_base(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    torque_magnitude_range: tuple[float, float],
    duration_range_s: tuple[float, float],
    ramp_time_s: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # debug visualization (optional)
    debug_vis: bool = False,
    debug_vis_height: float = 0.6,
    debug_vis_radius: float = 0.25,
    debug_vis_num_arrows: int = 8,
    debug_vis_max_envs: int = 64,
    debug_vis_torque_to_length: float = 0.02,
):
    """Start (or change) a yaw torque (about global Z) for a duration, with smooth ramping."""
    _ensure_push_yaw_torque_buffers(env)

    # resolve asset/body ids once and cache on env
    if not hasattr(env, "event_push_yaw_torque_asset_cfg"):
        env.event_push_yaw_torque_asset_cfg = asset_cfg
        try:
            env.event_push_yaw_torque_asset_cfg.resolve(env.scene)
        except Exception:
            pass
    asset_cfg = env.event_push_yaw_torque_asset_cfg

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    device = asset.device
    env_ids = env_ids.to(device=device)

    # sample torque (allow signed range)
    tau_min, tau_max = torque_magnitude_range
    torques_z = math_utils.sample_uniform(
        torch.tensor(tau_min, device=device),
        torch.tensor(tau_max, device=device),
        (len(env_ids),),
        device=device,
    )

    # sample hold duration (excluding ramps)
    dur_min, dur_max = duration_range_s
    hold_s = math_utils.sample_uniform(
        torch.tensor(dur_min, device=device),
        torch.tensor(dur_max, device=device),
        (len(env_ids),),
        device=device,
    )

    # schedule ramp to new target from current
    env.event_push_yaw_torque_ramp_start_z[env_ids] = env.event_push_yaw_torque_curr_z[env_ids]
    env.event_push_yaw_torque_ramp_target_z[env_ids] = torques_z
    env.event_push_yaw_torque_ramp_elapsed[env_ids] = 0.0
    env.event_push_yaw_torque_ramp_time[env_ids] = float(max(ramp_time_s, 0.0))
    env.event_push_yaw_torque_hold_left[env_ids] = hold_s
    env.event_push_yaw_torque_state[env_ids] = 1

    if debug_vis:
        _ensure_push_yaw_torque_visualizer(env)
        env.event_push_yaw_torque_debug_vis = True
        env.event_push_yaw_torque_debug_vis_height = float(debug_vis_height)
        env.event_push_yaw_torque_debug_vis_radius = float(debug_vis_radius)
        env.event_push_yaw_torque_debug_vis_num_arrows = int(max(1, debug_vis_num_arrows))
        env.event_push_yaw_torque_debug_vis_max_envs = int(max(1, debug_vis_max_envs))
        env.event_push_yaw_torque_debug_vis_torque_to_length = float(debug_vis_torque_to_length)


def update_push_wrench_base(env: "ManagerBasedEnv", dt: float):  # noqa: C901
    """Update external push **wrench** (push-force + yaw-torque) ramps and apply to the asset.

    Notes:
    - We update both the 2D force (yaw-aligned horizontal frame) and the yaw torque (about world Z).
    - Call this once per physics substep, *before* `scene.write_data_to_sim()`.
    """
    has_force = hasattr(env, "event_push_force_state")
    has_yaw_torque = hasattr(env, "event_push_yaw_torque_state")
    if not (has_force or has_yaw_torque):
        return

    # pick asset cfg
    asset_cfg: SceneEntityCfg | None = getattr(env, "event_push_force_asset_cfg", None)
    if asset_cfg is None:
        asset_cfg = getattr(env, "event_push_yaw_torque_asset_cfg", None)
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    env.event_push_force_asset_cfg = asset_cfg
    try:
        asset_cfg.resolve(env.scene)
    except Exception:
        pass

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    if not isinstance(asset, Articulation):
        return

    # choose a single trunk body id
    body_ids = getattr(asset_cfg, "body_ids", None)
    if body_ids is None:
        body_ids = torch.tensor([0], dtype=torch.long, device=asset.device)
    elif isinstance(body_ids, slice):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.long, device=asset.device)[body_ids]
        if body_ids.numel() == 0:
            body_ids = torch.tensor([0], dtype=torch.long, device=asset.device)
    elif not isinstance(body_ids, torch.Tensor):
        body_ids = torch.tensor(body_ids, dtype=torch.long, device=asset.device)
    if body_ids.numel() != 1:
        body_ids = body_ids[:1]

    dt_t = torch.tensor(float(dt), device=env.device, dtype=torch.float32)

    # --------------------
    # 2D force in yaw frame
    # --------------------
    if has_force:
        state = env.event_push_force_state
        curr = env.event_push_force_curr_xy
    else:
        _ensure_push_force_buffers(env)
        state = env.event_push_force_state
        curr = env.event_push_force_curr_xy
        state[:] = 0
        curr[:] = 0.0

    ramp_mask = (state == 1) | (state == 3)
    if ramp_mask.any():
        elapsed = env.event_push_force_ramp_elapsed[ramp_mask]
        ramp_t = env.event_push_force_ramp_time[ramp_mask].clamp(min=0.0)
        start_xy = env.event_push_force_ramp_start_xy[ramp_mask]
        target_xy = env.event_push_force_ramp_target_xy[ramp_mask]

        done_instant = ramp_t <= 1e-8
        if done_instant.any():
            start_xy[done_instant] = target_xy[done_instant]
            elapsed[done_instant] = ramp_t[done_instant]

        elapsed = elapsed + dt_t
        s = torch.where(ramp_t > 1e-8, (elapsed / ramp_t).clamp(0.0, 1.0), torch.ones_like(elapsed))
        new_xy = start_xy + (target_xy - start_xy) * s.unsqueeze(-1)
        curr[ramp_mask] = new_xy
        env.event_push_force_ramp_elapsed[ramp_mask] = elapsed

        done = s >= 1.0 - 1e-6
        if done.any():
            ramp_idx = ramp_mask.nonzero(as_tuple=False).squeeze(-1)
            done_idx = ramp_idx[done]
            curr[done_idx] = env.event_push_force_ramp_target_xy[done_idx]
            to_hold = done_idx[state[done_idx] == 1]
            to_idle = done_idx[state[done_idx] == 3]
            if to_hold.numel() > 0:
                state[to_hold] = 2
            if to_idle.numel() > 0:
                state[to_idle] = 0
                curr[to_idle] = 0.0

    hold_mask = state == 2
    if hold_mask.any():
        hold_left = env.event_push_force_hold_left[hold_mask] - dt_t
        env.event_push_force_hold_left[hold_mask] = hold_left
        finished = hold_left <= 0.0
        if finished.any():
            hold_idx = hold_mask.nonzero(as_tuple=False).squeeze(-1)
            done_idx = hold_idx[finished]
            env.event_push_force_ramp_start_xy[done_idx] = curr[done_idx]
            env.event_push_force_ramp_target_xy[done_idx] = 0.0
            env.event_push_force_ramp_elapsed[done_idx] = 0.0
            state[done_idx] = 3

    # --------------------
    # low-frequency refresh of cached terrain slope (terrain_heading only)
    # --------------------
    force_frame = getattr(env, "event_push_force_frame", "yaw_horizontal")
    if force_frame == "terrain_heading":
        active = (state != 0)
        if active.any():
            period_left = env.event_push_force_slope_refresh_left_s
            period_left[active] = (period_left[active] - dt_t).clamp(min=-1.0e6)
            to_refresh = active & (period_left <= 0.0)
            if to_refresh.any():
                env_ids = to_refresh.nonzero(as_tuple=False).squeeze(-1).to(device=asset.device, dtype=torch.long)
                _refresh_cached_terrain_slope(
                    env=env,
                    asset=asset,
                    body_id=int(body_ids[0].item()),
                    env_ids=env_ids,
                    slope_min_grad=float(getattr(env, "event_push_force_slope_min_grad", 0.02)),
                )
                # reset timer; if period is not configured, lock (no further refresh)
                period = float(getattr(env, "event_push_force_slope_update_period_s", 0.0))
                if period > 0.0:
                    period_left[env_ids] = period
                else:
                    period_left[env_ids] = 0.0
            env.event_push_force_slope_refresh_left_s = period_left

    # --------------------
    # yaw torque about global Z
    # --------------------
    if has_yaw_torque:
        t_state = env.event_push_yaw_torque_state
        t_curr = env.event_push_yaw_torque_curr_z

        t_ramp_mask = (t_state == 1) | (t_state == 3)
        if t_ramp_mask.any():
            t_elapsed = env.event_push_yaw_torque_ramp_elapsed[t_ramp_mask]
            t_ramp_t = env.event_push_yaw_torque_ramp_time[t_ramp_mask].clamp(min=0.0)
            t_start = env.event_push_yaw_torque_ramp_start_z[t_ramp_mask]
            t_target = env.event_push_yaw_torque_ramp_target_z[t_ramp_mask]

            t_done_instant = t_ramp_t <= 1e-8
            if t_done_instant.any():
                t_start[t_done_instant] = t_target[t_done_instant]
                t_elapsed[t_done_instant] = t_ramp_t[t_done_instant]

            t_elapsed = t_elapsed + dt_t
            s = torch.where(t_ramp_t > 1e-8, (t_elapsed / t_ramp_t).clamp(0.0, 1.0), torch.ones_like(t_elapsed))
            t_new = t_start + (t_target - t_start) * s
            t_curr[t_ramp_mask] = t_new
            env.event_push_yaw_torque_ramp_elapsed[t_ramp_mask] = t_elapsed

            t_done = s >= 1.0 - 1e-6
            if t_done.any():
                t_ramp_idx = t_ramp_mask.nonzero(as_tuple=False).squeeze(-1)
                t_done_idx = t_ramp_idx[t_done]
                t_curr[t_done_idx] = env.event_push_yaw_torque_ramp_target_z[t_done_idx]
                to_hold = t_done_idx[t_state[t_done_idx] == 1]
                to_idle = t_done_idx[t_state[t_done_idx] == 3]
                if to_hold.numel() > 0:
                    t_state[to_hold] = 2
                if to_idle.numel() > 0:
                    t_state[to_idle] = 0
                    t_curr[to_idle] = 0.0

        t_hold_mask = t_state == 2
        if t_hold_mask.any():
            t_hold_left = env.event_push_yaw_torque_hold_left[t_hold_mask] - dt_t
            env.event_push_yaw_torque_hold_left[t_hold_mask] = t_hold_left
            t_finished = t_hold_left <= 0.0
            if t_finished.any():
                t_hold_idx = t_hold_mask.nonzero(as_tuple=False).squeeze(-1)
                t_done_idx = t_hold_idx[t_finished]
                env.event_push_yaw_torque_ramp_start_z[t_done_idx] = t_curr[t_done_idx]
                env.event_push_yaw_torque_ramp_target_z[t_done_idx] = 0.0
                env.event_push_yaw_torque_ramp_elapsed[t_done_idx] = 0.0
                t_state[t_done_idx] = 3

        env.event_push_yaw_torque_buf[:] = t_curr.unsqueeze(-1)
        torque_z = t_curr.to(device=asset.device, dtype=torch.float32)
    else:
        torque_z = torch.zeros((env.num_envs,), device=asset.device, dtype=torch.float32)

    # Convert force -> world-frame force, then apply in global frame.
    # See `start_push_force_xy_base(..., force_frame=...)` for frame conventions.
    body_id = int(body_ids[0].item())
    body_quat_w = asset.data.body_quat_w[:, body_id]  # (N, 4)
    force_frame = getattr(env, "event_push_force_frame", "yaw_horizontal")
    pitch_offset = getattr(env, "event_push_force_pitch_offset_buf", None)
    if pitch_offset is None:
        pitch_offset = torch.zeros((env.num_envs,), device=asset.device, dtype=torch.float32)
    else:
        pitch_offset = pitch_offset.to(device=asset.device, dtype=torch.float32)

    # local force (2D -> 3D) in the selected "base-aligned" frame
    force_local = torch.zeros((env.num_envs, 3), device=asset.device, dtype=torch.float32)
    force_local[:, 0:2] = curr.to(device=asset.device, dtype=torch.float32)

    # Convert local force -> world force.
    # Depending on the frame, we either:
    # - build a yaw/pitch(/roll) quaternion and rotate `force_local`, or
    # - directly construct a tangent-plane direction from terrain slope (more stable).
    _, pitch_meas, yaw = math_utils.euler_xyz_from_quat(body_quat_w)
    # defaults (ensure all variables are always defined)
    yaw_used = yaw
    pitch_used = pitch_offset
    roll_used = torch.zeros_like(pitch_used)
    force_w = None

    if force_frame == "terrain_heading":
        # Keep yaw aligned with robot heading, but constrain the force direction to the local terrain tangent plane.
        # We build the tangent directions in the yaw frame (x=forward, y=left), then rotate by yaw to world.
        a = getattr(env, "event_push_force_slope_a_buf", torch.zeros_like(yaw_used)).to(device=asset.device, dtype=torch.float32)
        b = getattr(env, "event_push_force_slope_b_buf", torch.zeros_like(yaw_used)).to(device=asset.device, dtype=torch.float32)

        # terrain plane normal in yaw frame for z = a*x + b*y + c is n ~ (-a, -b, 1)
        n = torch.stack([-a, -b, torch.ones_like(a)], dim=-1)
        n = n / torch.linalg.norm(n, dim=-1, keepdim=True).clamp(min=1.0e-8)

        fwd = torch.zeros((env.num_envs, 3), device=asset.device, dtype=torch.float32)
        fwd[:, 0] = 1.0
        left = torch.zeros_like(fwd)
        left[:, 1] = 1.0

        # project basis vectors onto terrain tangent plane
        fwd_t = fwd - torch.sum(fwd * n, dim=-1, keepdim=True) * n
        left_t = left - torch.sum(left * n, dim=-1, keepdim=True) * n
        fwd_t = fwd_t / torch.linalg.norm(fwd_t, dim=-1, keepdim=True).clamp(min=1.0e-8)
        left_t = left_t / torch.linalg.norm(left_t, dim=-1, keepdim=True).clamp(min=1.0e-8)

        # optional extra pitch tilt around the yaw-frame left axis
        if pitch_offset is not None:
            zeros = torch.zeros_like(pitch_offset)
            q_off = math_utils.quat_from_euler_xyz(zeros, pitch_offset, zeros)  # (w,x,y,z)
            fwd_t = math_utils.quat_apply(q_off, fwd_t)
            left_t = math_utils.quat_apply(q_off, left_t)

        # compose force in yaw frame tangent basis and rotate to world via yaw
        fx = force_local[:, 0].unsqueeze(-1)
        fy = force_local[:, 1].unsqueeze(-1)
        force_yaw = fx * fwd_t + fy * left_t
        yaw_q = math_utils.yaw_quat(body_quat_w).to(device=asset.device, dtype=torch.float32)
        force_w = math_utils.quat_apply(yaw_q, force_yaw)

    if force_w is None:
        yaw_pitch_q = math_utils.quat_from_euler_xyz(roll_used, pitch_used, yaw_used)
        force_w = math_utils.quat_apply(yaw_pitch_q, force_local)

    # Update observation/reward buffer `event_push_force_buf`:
    # store the applied world-frame force projected into the full trunk/body frame (x/y).
    if has_force and hasattr(env, "event_push_force_buf"):
        buf = env.event_push_force_buf
        force_b = math_utils.quat_apply_inverse(body_quat_w, force_w)  # (N,3) in trunk/body frame
        force_b_xy = force_b[:, :2].to(device=buf.device, dtype=buf.dtype)
        buf[:, 0] = force_b_xy[:, 0]
        buf[:, 1] = force_b_xy[:, 1]

    forces = force_w.view(env.num_envs, 1, 3)
    torques = torch.zeros_like(forces)
    torques[:, 0, 2] = torque_z
    asset.set_external_force_and_torque(
        forces=forces,
        torques=torques,
        body_ids=body_ids,
        env_ids=None,
        is_global=True,
    )

    # --- optional visualization: arrow at trunk, direction = applied world-frame push force ---
    if (
        getattr(env, "event_push_force_debug_vis", False)
        and getattr(env, "event_push_force_visualizer", None) is not None
    ):
        try:
            pos_w = asset.data.body_pos_w[:, body_id].clone()
            pos_w[:, 2] += float(getattr(env, "event_push_force_debug_vis_height", 0.6))

            # visualize true 3D push direction (including Z component)
            force_w_vis = torch.nan_to_num(force_w, nan=0.0, posinf=0.0, neginf=0.0)
            force_mag = torch.linalg.norm(force_w_vis, dim=1)
            default_scale = env.event_push_force_visualizer.cfg.markers["arrow"].scale
            arrow_scale = torch.tensor(default_scale, device=asset.device, dtype=torch.float32).repeat(env.scene.num_envs, 1)
            force_to_len = float(getattr(env, "event_push_force_debug_vis_force_to_length", 0.1))
            arrow_scale[:, 0] *= force_mag * force_to_len
            arrow_scale[force_mag <= 1.0e-6] = 0.0

            # IsaacLab math/markers use quaternions in (w, x, y, z).
            arrow_quat_w = _quat_wxyz_from_x_axis_to_vec(force_w_vis)
            env.event_push_force_visualizer.visualize(pos_w, arrow_quat_w, arrow_scale)
        except Exception as e:
            # Don't permanently disable debug vis due to transient failures (e.g. stage not ready).
            carb.log_warn(f"[push_force] Visualization failed once; skipping this frame. Error: {e}")

    # --- optional visualization: ring arrows indicating yaw torque ---
    if (
        getattr(env, "event_push_yaw_torque_debug_vis", False)
        and getattr(env, "event_push_yaw_torque_visualizer", None) is not None
    ):
        try:
            max_envs = int(getattr(env, "event_push_yaw_torque_debug_vis_max_envs", 64))
            num_envs_vis = int(min(env.scene.num_envs, max_envs))
            n_arrows = int(getattr(env, "event_push_yaw_torque_debug_vis_num_arrows", 8))
            n_arrows = max(1, n_arrows)

            center_w = asset.data.body_pos_w[:num_envs_vis, body_id].clone()
            center_w[:, 2] += float(getattr(env, "event_push_yaw_torque_debug_vis_height", 0.6))
            radius = float(getattr(env, "event_push_yaw_torque_debug_vis_radius", 0.25))

            tau_z = torque_z[:num_envs_vis].clone()
            tau_abs = torch.abs(tau_z)
            tau_sign = torch.sign(tau_z)
            tau_sign[tau_sign == 0.0] = 0.0

            angles = torch.linspace(0.0, 2.0 * math.pi, steps=n_arrows + 1, device=asset.device, dtype=torch.float32)[:-1]
            cos_a = torch.cos(angles)
            sin_a = torch.sin(angles)
            offsets = torch.stack([radius * cos_a, radius * sin_a, torch.zeros_like(cos_a)], dim=-1)  # (A,3)
            offsets = offsets.unsqueeze(0).repeat(num_envs_vis, 1, 1)  # (E,A,3)
            pos_w = center_w.unsqueeze(1) + offsets  # (E,A,3)

            tan_x = -sin_a
            tan_y = cos_a
            tan_x = tan_x.unsqueeze(0).repeat(num_envs_vis, 1) * tau_sign.unsqueeze(-1)
            tan_y = tan_y.unsqueeze(0).repeat(num_envs_vis, 1) * tau_sign.unsqueeze(-1)
            heading = torch.atan2(tan_y, tan_x)
            zeros = torch.zeros_like(heading)
            quat_w = math_utils.quat_from_euler_xyz(zeros, zeros, heading)

            default_scale = env.event_push_yaw_torque_visualizer.cfg.markers["arrow"].scale
            scale = torch.tensor(default_scale, device=asset.device, dtype=torch.float32).repeat(num_envs_vis, n_arrows, 1)
            to_len = float(getattr(env, "event_push_yaw_torque_debug_vis_torque_to_length", 0.02))
            scale[..., 0] *= tau_abs.unsqueeze(-1) * to_len
            scale[tau_abs <= 1.0e-6] = 0.0

            env.event_push_yaw_torque_visualizer.visualize(
                pos_w.reshape(-1, 3), quat_w.reshape(-1, 4), scale.reshape(-1, 3)
            )
        except Exception as e:
            carb.log_warn(f"[push_yaw_torque] Visualization failed once; disabling. Error: {e}")
            env.event_push_yaw_torque_debug_vis = False


def reset_joint_offset(
    env,
    env_ids,
    randomization_params: tuple[float | Sequence[float], float | Sequence[float]] | None = None,
    operation: Literal["add", "scale", "abs"] = "add",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Event function to randomize joint position offset.
    Args:
        env: The environment instance.
        env_ids: Environment IDs to randomize.
        randomization_params: Scalar or per-joint lower/upper bounds. If None, no randomization is applied.
        operation: Operation to apply for offset randomization. Defaults to 'add'.
        distribution: Distribution type for offset randomization. Defaults to 'uniform'.
    """
    if hasattr(env.action_manager._terms["joint_pos"], "randomize_offset"):
        env.action_manager._terms["joint_pos"].randomize_offset(
            env=env,
            env_ids=env_ids,
            randomization_params=randomization_params,
            operation=operation,
            distribution=distribution,
        )


def apply_external_force_torque_3d(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_x_range: tuple[float, float] = (-0.0, 0.0),
    force_y_range: tuple[float, float] = (-0.0, 0.0),
    force_z_range: tuple[float, float] = (-0.0, 0.0),
    torque_x_range: tuple[float, float] = (-0.0, 0.0),
    torque_y_range: tuple[float, float] = (-0.0, 0.0),
    torque_z_range: tuple[float, float] = (-0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply a 3-D external force disturbance per (env, body).

    Sampling ranges:
      force_x ~ Uniform(force_x_range)
      force_y ~ Uniform(force_y_range)
      force_z ~ Uniform(force_z_range)
      torque_x ~ Uniform(torque_x_range)
      torque_y ~ Uniform(torque_y_range)
      torque_z ~ Uniform(torque_z_range)

    Values are written into the asset external force/torque buffer and applied when
    `asset.write_data_to_sim()` is called.
    """
    # extract asset
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve number of bodies
    body_ids = getattr(asset_cfg, "body_ids", None)
    if isinstance(body_ids, torch.Tensor):
        num_bodies = int(body_ids.numel())
    elif isinstance(body_ids, list):
        num_bodies = len(body_ids)
    elif isinstance(body_ids, slice):
        num_bodies = int(torch.arange(asset.num_bodies, device=asset.device)[body_ids].numel())
    else:
        num_bodies = int(asset.num_bodies)

    # sampling size: (num_envs_selected, num_bodies, 3)
    size_forces = (len(env_ids), num_bodies, 3)
    size_torques = (len(env_ids), num_bodies, 3)

    # prepare per-axis bounds on correct device
    forces_lower = torch.tensor([force_x_range[0], force_y_range[0], force_z_range[0]], device=asset.device)
    forces_upper = torch.tensor([force_x_range[1], force_y_range[1], force_z_range[1]], device=asset.device)
    torques_lower = torch.tensor([torque_x_range[0], torque_y_range[0], torque_z_range[0]], device=asset.device)
    torques_upper = torch.tensor([torque_x_range[1], torque_y_range[1], torque_z_range[1]], device=asset.device)

    # sample forces/torques
    forces = math_utils.sample_uniform(forces_lower, forces_upper, size_forces, asset.device)
    torques = math_utils.sample_uniform(torques_lower, torques_upper, size_torques, asset.device)

    # write into asset buffer (applied when asset.write_data_to_sim() is called)
    asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)

    # record applied wrench for observations/debug (generic, world axes)
    # buffer convention: (Fx, Fy, Fz, Tx, Ty, Tz), aggregated over selected bodies
    if not hasattr(env, "event_apply_forces_torques_buf"):
        env.event_apply_forces_torques_buf = torch.zeros(
            (env.scene.num_envs, 6),
            device=asset.device,
            dtype=torch.float32,
            requires_grad=False,
        )
    else:
        env.event_apply_forces_torques_buf = env.event_apply_forces_torques_buf.to(device=asset.device)

    forces_net = forces.sum(dim=1)  # (E, 3)
    torques_net = torques.sum(dim=1)  # (E, 3)
    env.event_apply_forces_torques_buf[env_ids] = torch.cat([forces_net, torques_net], dim=-1)


def _ensure_ee_external_force_buffers(env: "ManagerBasedEnv"):
    """Lazy-init ramp state buffers for EE external force (force-only, 3D, world frame).

    Mirrors ``_ensure_push_force_buffers`` but tracks only a 3D force (no torque channel,
    since ``force_control`` applies force only). State machine: 0 idle, 1 ramping, 2 hold.
    """
    if hasattr(env, "ee_force_ramp_state"):
        return
    num_envs = env.scene.num_envs
    device = getattr(env, "device", torch.device("cpu"))

    env.ee_force_ramp_state = torch.zeros((num_envs,), device=device, dtype=torch.int64)
    env.ee_force_curr = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
    env.ee_force_ramp_start = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
    env.ee_force_ramp_target = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
    env.ee_force_ramp_elapsed = torch.zeros((num_envs,), device=device, dtype=torch.float32)
    env.ee_force_ramp_time = torch.zeros((num_envs,), device=device, dtype=torch.float32)
    env.ee_force_zero_torques = torch.zeros(
        (num_envs, 1, 3), device=device, dtype=torch.float32
    )
    # cached resolved asset cfg (EE body, e.g. link6) for the per-step update
    env.ee_force_asset_cfg = None
    env.ee_force_asset = None


def start_ee_external_force_3d(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    force_x_range: tuple[float, float] = (-0.0, 0.0),
    force_y_range: tuple[float, float] = (-0.0, 0.0),
    force_z_range: tuple[float, float] = (-0.0, 0.0),
    torque_x_range: tuple[float, float] = (-0.0, 0.0),
    torque_y_range: tuple[float, float] = (-0.0, 0.0),
    torque_z_range: tuple[float, float] = (-0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ramp_duration_s: float | None = 1.0,
):
    """Interval-mode event: sample a new EE force target (world frame) and arm a linear ramp.

    Only records ramp state (mirrors ``start_push_force_xy_base``). The per-physics-step
    ``update_ee_external_force`` is the sole writer to the asset / wrench buf.

    Param names match ``apply_external_force_torque_3d`` so the curriculum (patches
    ``force_*_range``) and config_dict need zero changes. ``torque_*_range`` are accepted
    for signature compatibility but ignored (force_control applies force only).

    ``ramp_duration_s=None`` -> legacy step-and-hold via ``apply_external_force_torque_3d``.
    """
    # legacy fallback: no ramp -> behave like the original step-and-hold applier
    if ramp_duration_s is None:
        apply_external_force_torque_3d(
            env,
            env_ids,
            force_x_range,
            force_y_range,
            force_z_range,
            torque_x_range,
            torque_y_range,
            torque_z_range,
            asset_cfg,
        )
        return

    _ensure_ee_external_force_buffers(env)

    # cache resolved asset cfg for the per-step update
    if env.ee_force_asset_cfg is None:
        env.ee_force_asset_cfg = asset_cfg
    asset_cfg = env.ee_force_asset_cfg
    try:
        asset_cfg.resolve(env.scene)
    except Exception:
        pass
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    env.ee_force_asset = asset
    device = asset.device

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, dtype=torch.long, device=device)
    else:
        env_ids = env_ids.to(device=device)

    # sample new target force (world frame, per-axis uniform)
    lower = torch.tensor([force_x_range[0], force_y_range[0], force_z_range[0]], device=device)
    upper = torch.tensor([force_x_range[1], force_y_range[1], force_z_range[1]], device=device)
    new_target = math_utils.sample_uniform(lower, upper, (len(env_ids), 3), device)

    # arm ramp: start = current applied force (continuity); target = new sample
    env.ee_force_ramp_start[env_ids] = env.ee_force_curr[env_ids]
    env.ee_force_ramp_target[env_ids] = new_target
    env.ee_force_ramp_elapsed[env_ids] = 0.0
    env.ee_force_ramp_time[env_ids] = float(max(ramp_duration_s, 0.0))
    env.ee_force_ramp_state[env_ids] = 1


def update_ee_external_force(env: "ManagerBasedEnv", dt: float):
    """Advance EE external-force ramp and apply to asset (world frame, ``is_global=True``).

    Call once per physics substep, BEFORE ``scene.write_data_to_sim()``
    (mirrors ``update_push_wrench_base``). No-op if EE ramp buffers were never created
    (configs without ``ee_external_force``).
    """
    if not hasattr(env, "ee_force_ramp_state"):
        return
    asset_cfg = getattr(env, "ee_force_asset_cfg", None)
    if asset_cfg is None:
        return
    asset: RigidObject | Articulation | None = getattr(
        env, "ee_force_asset", None
    )
    if asset is None:
        # Compatibility fallback for an already-created env after a live code reload.
        try:
            asset_cfg.resolve(env.scene)
            asset = env.scene[asset_cfg.name]
            env.ee_force_asset = asset
        except (AttributeError, KeyError, RuntimeError):
            return
    if not isinstance(asset, Articulation):
        return

    state = env.ee_force_ramp_state
    curr = env.ee_force_curr

    # Dense, branchless state update.  The old ``if ramp_mask.any()`` and
    # ``if done.any()`` converted CUDA scalars to Python bools on every physics
    # substep (up to 8 device synchronizations per policy step with decimation=4).
    # torch.where keeps idle/hold rows bitwise unchanged while updating ramp rows.
    ramp_mask = state == 1
    elapsed = env.ee_force_ramp_elapsed + float(dt)
    ramp_t = env.ee_force_ramp_time.clamp(min=1e-8)
    s = (elapsed / ramp_t).clamp(0.0, 1.0)
    interpolated = env.ee_force_ramp_start + (
        env.ee_force_ramp_target - env.ee_force_ramp_start
    ) * s.unsqueeze(-1)
    curr.copy_(torch.where(ramp_mask.unsqueeze(-1), interpolated, curr))
    env.ee_force_ramp_elapsed.copy_(
        torch.where(ramp_mask, elapsed, env.ee_force_ramp_elapsed)
    )
    done = ramp_mask & (s >= 1.0 - 1e-6)
    curr.copy_(
        torch.where(done.unsqueeze(-1), env.ee_force_ramp_target, curr)
    )
    state.masked_fill_(done, 2)  # hold until next interval trigger re-arms

    # apply current force to asset for ALL envs (idle envs -> curr=0 -> zero force).
    # is_global=True keeps the articulation's wrench frame in world (same as
    # update_push_wrench_base), so EE and push never fight over _use_global_wrench_frame.
    body_ids = getattr(asset_cfg, "body_ids", None)
    forces = curr.unsqueeze(1).contiguous()  # (E, 1, 3)
    torques = getattr(env, "ee_force_zero_torques", None)
    if torques is None or torques.shape != forces.shape:
        torques = torch.zeros_like(forces)
        env.ee_force_zero_torques = torques
    asset.set_external_force_and_torque(
        forces=forces,
        torques=torques,
        body_ids=body_ids,
        env_ids=None,
        is_global=True,
    )

    # write wrench buf (sole writer): force in world frame; torque channel kept zero
    if hasattr(env, "event_apply_forces_torques_buf"):
        buf = env.event_apply_forces_torques_buf
        buf[:, :3] = curr
        buf[:, 3:] = 0.0


def reset_ee_external_force(env: "ManagerBasedEnv", env_ids: torch.Tensor | list[int] | None = None):
    """Zero EE external-force ramp state and applied wrench for reset envs.

    Mirrors ``reset_push_force``.
    """
    if not hasattr(env, "ee_force_ramp_state"):
        return
    if env_ids is None:
        env_ids = slice(None)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, dtype=torch.long, device=env.device)

    env.ee_force_ramp_state[env_ids] = 0
    env.ee_force_curr[env_ids] = 0.0
    env.ee_force_ramp_start[env_ids] = 0.0
    env.ee_force_ramp_target[env_ids] = 0.0
    env.ee_force_ramp_elapsed[env_ids] = 0.0
    env.ee_force_ramp_time[env_ids] = 0.0
    # zero wrench buf so first obs/reward after reset sees zero force
    if hasattr(env, "event_apply_forces_torques_buf"):
        env.event_apply_forces_torques_buf[env_ids] = 0.0


def _external_force_channel(
    env: "ManagerBasedEnv",
    channel_name: str,
    asset_cfg: SceneEntityCfg,
    reference_body_cfg: SceneEntityCfg,
    force_frame: Literal["world", "base"],
) -> dict:
    """Create or retrieve one independently sampled external-force channel."""
    if force_frame not in ("world", "base"):
        raise ValueError(f"Unsupported external-force frame: {force_frame!r}.")
    if not hasattr(env, "external_force_channels"):
        env.external_force_channels = {}
    if channel_name in env.external_force_channels:
        channel = env.external_force_channels[channel_name]
        if channel["force_frame"] != force_frame:
            raise ValueError(
                f"External-force channel {channel_name!r} was already configured in "
                f"{channel['force_frame']!r}, not {force_frame!r}."
            )
        return channel

    asset_cfg.resolve(env.scene)
    reference_body_cfg.resolve(env.scene)

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    reference_asset: RigidObject | Articulation = env.scene[reference_body_cfg.name]

    def _single_body_id(cfg: SceneEntityCfg, cfg_asset, label: str) -> int:
        body_ids = cfg.body_ids
        if isinstance(body_ids, slice):
            body_ids = torch.arange(cfg_asset.num_bodies, device=cfg_asset.device)[body_ids]
        elif not isinstance(body_ids, torch.Tensor):
            body_ids = torch.tensor(body_ids, dtype=torch.long, device=cfg_asset.device)
        if len(body_ids) != 1:
            raise ValueError(f"An external-force channel requires exactly one {label}.")
        return int(body_ids[0].item())

    num_envs = env.scene.num_envs
    device = asset.device
    channel = {
        "asset_cfg": asset_cfg,
        "asset": asset,
        "reference_body_cfg": reference_body_cfg,
        "reference_asset": reference_asset,
        "body_id": _single_body_id(asset_cfg, asset, "target body"),
        "reference_body_id": _single_body_id(
            reference_body_cfg, reference_asset, "reference body"
        ),
        "force_frame": force_frame,
        "state": torch.zeros(num_envs, device=device, dtype=torch.int64),
        "current": torch.zeros((num_envs, 3), device=device, dtype=torch.float32),
        "ramp_start": torch.zeros((num_envs, 3), device=device, dtype=torch.float32),
        "ramp_target": torch.zeros((num_envs, 3), device=device, dtype=torch.float32),
        "ramp_elapsed": torch.zeros(num_envs, device=device, dtype=torch.float32),
        "ramp_time": torch.zeros(num_envs, device=device, dtype=torch.float32),
    }
    env.external_force_channels[channel_name] = channel
    # A newly added channel changes the target-body/reference grouping.
    env._external_force_write_plan = None
    return channel


def start_external_force_channel_3d(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    channel_name: str,
    force_x_range: tuple[float, float],
    force_y_range: tuple[float, float],
    force_z_range: tuple[float, float],
    ramp_duration_range_s: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    reference_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base_link"),
    force_frame: Literal["world", "base"] = "world",
):
    """Sample a target for one named 3-D force channel and start a linear ramp.

    Each channel owns independent current/target/ramp buffers. Forces can be
    expressed in world axes or in the current orientation of a reference body.
    """
    channel = _external_force_channel(
        env,
        channel_name=channel_name,
        asset_cfg=asset_cfg,
        reference_body_cfg=reference_body_cfg,
        force_frame=force_frame,
    )
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    device = asset.device
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, dtype=torch.long, device=device)
    else:
        env_ids = env_ids.to(device=device)

    lower = torch.tensor(
        [force_x_range[0], force_y_range[0], force_z_range[0]],
        device=device,
        dtype=torch.float32,
    )
    upper = torch.tensor(
        [force_x_range[1], force_y_range[1], force_z_range[1]],
        device=device,
        dtype=torch.float32,
    )
    if any(low > high for low, high in (force_x_range, force_y_range, force_z_range)):
        raise ValueError(
            "Every force range must satisfy low <= high, got "
            f"x={force_x_range}, y={force_y_range}, z={force_z_range}."
        )
    target = math_utils.sample_uniform(lower, upper, (len(env_ids), 3), device)

    ramp_low, ramp_high = ramp_duration_range_s
    if ramp_low < 0.0 or ramp_high < ramp_low:
        raise ValueError(
            "ramp_duration_range_s must satisfy 0 <= low <= high, got "
            f"{ramp_duration_range_s}."
        )
    ramp_time = math_utils.sample_uniform(
        torch.tensor(float(ramp_low), device=device),
        torch.tensor(float(ramp_high), device=device),
        (len(env_ids),),
        device,
    )

    channel["ramp_start"][env_ids] = channel["current"][env_ids]
    channel["ramp_target"][env_ids] = target
    channel["ramp_elapsed"][env_ids] = 0.0
    channel["ramp_time"][env_ids] = ramp_time
    channel["state"][env_ids] = 1


def _build_external_force_write_plan(env: "ManagerBasedEnv", channels: dict) -> list[dict]:
    """Build reusable public-API write buffers for the current channel layout."""
    cached_plan = getattr(env, "_external_force_write_plan", None)
    if cached_plan is not None:
        return cached_plan

    asset_groups: dict[str, dict] = {}
    for channel in channels.values():
        asset_cfg: SceneEntityCfg = channel["asset_cfg"]
        asset_group = asset_groups.setdefault(
            asset_cfg.name,
            {
                "asset": channel["asset"],
                "channels": [],
            },
        )
        asset_group["channels"].append(channel)

    write_plan = []
    for asset_group in asset_groups.values():
        asset: RigidObject | Articulation = asset_group["asset"]
        grouped_channels = asset_group["channels"]
        body_ids = sorted({channel["body_id"] for channel in grouped_channels})
        body_slot = {body_id: index for index, body_id in enumerate(body_ids)}

        world_entries = []
        reference_groups: dict[tuple[str, int], dict] = {}
        for channel in grouped_channels:
            entry = {
                "channel": channel,
                "body_slot": body_slot[channel["body_id"]],
            }
            if channel["force_frame"] == "world":
                world_entries.append(entry)
                continue

            reference_cfg: SceneEntityCfg = channel["reference_body_cfg"]
            reference_key = (
                reference_cfg.name,
                channel["reference_body_id"],
            )
            reference_group = reference_groups.setdefault(
                reference_key,
                {
                    "reference_asset": channel["reference_asset"],
                    "reference_body_id": channel["reference_body_id"],
                    "entries": [],
                },
            )
            reference_group["entries"].append(entry)

        num_envs = env.scene.num_envs
        force_buffer = torch.empty(
            (num_envs, len(body_ids), 3),
            device=asset.device,
            dtype=torch.float32,
        )
        torque_buffer = torch.zeros_like(force_buffer)

        direct_reference_group = None
        if not world_entries and len(reference_groups) == 1:
            candidate = next(iter(reference_groups.values()))
            candidate_slots = [entry["body_slot"] for entry in candidate["entries"]]
            if (
                len(candidate_slots) == len(body_ids)
                and candidate_slots == list(range(len(body_ids)))
            ):
                direct_reference_group = candidate

        for reference_group in reference_groups.values():
            reference_group["local_force_buffer"] = torch.empty(
                (num_envs, len(reference_group["entries"]), 3),
                device=asset.device,
                dtype=torch.float32,
            )

        write_plan.append(
            {
                "asset": asset,
                "body_ids": torch.tensor(
                    body_ids,
                    device=asset.device,
                    dtype=torch.long,
                ),
                "force_buffer": force_buffer,
                "torque_buffer": torque_buffer,
                "world_entries": world_entries,
                "reference_groups": tuple(reference_groups.values()),
                "direct_reference_group": direct_reference_group,
            }
        )

    env._external_force_write_plan = write_plan
    return write_plan


def _quat_apply_force_batch(
    quat_w: torch.Tensor,
    forces: torch.Tensor,
) -> torch.Tensor:
    """Rotate ``(env, channel, 3)`` forces with one quaternion per environment."""
    quat_xyz = quat_w[:, None, 1:]
    cross = torch.linalg.cross(quat_xyz, forces, dim=-1) * 2.0
    return (
        forces
        + quat_w[:, None, :1] * cross
        + torch.linalg.cross(quat_xyz, cross, dim=-1)
    )


def update_external_force_channels(env: "ManagerBasedEnv", dt: float):
    """Advance all named force ramps and write their summed world-frame forces."""
    channels = getattr(env, "external_force_channels", None)
    if not channels:
        return

    for channel in channels.values():
        state = channel["state"]
        current = channel["current"]
        ramp_mask = state == 1
        elapsed = channel["ramp_elapsed"] + float(dt)
        ramp_time = channel["ramp_time"]
        # A zero ramp duration reaches one on the first positive-dt update.
        fraction = (elapsed / ramp_time.clamp(min=1.0e-8)).clamp(0.0, 1.0)
        interpolated = channel["ramp_start"] + (
            channel["ramp_target"] - channel["ramp_start"]
        ) * fraction.unsqueeze(-1)
        current.copy_(torch.where(ramp_mask.unsqueeze(-1), interpolated, current))
        channel["ramp_elapsed"].copy_(
            torch.where(ramp_mask, elapsed, channel["ramp_elapsed"])
        )
        done = ramp_mask & (fraction >= 1.0 - 1.0e-6)
        current.copy_(torch.where(done.unsqueeze(-1), channel["ramp_target"], current))
        state.masked_fill_(done, 2)

    for asset_write in _build_external_force_write_plan(env, channels):
        asset = asset_write["asset"]
        forces = asset_write["force_buffer"]
        direct_reference_group = asset_write["direct_reference_group"]

        if direct_reference_group is None:
            forces.zero_()
            for entry in asset_write["world_entries"]:
                forces[:, entry["body_slot"]].add_(entry["channel"]["current"])

        for reference_group in asset_write["reference_groups"]:
            local_forces = reference_group["local_force_buffer"]
            for index, entry in enumerate(reference_group["entries"]):
                local_forces[:, index].copy_(entry["channel"]["current"])
            reference_asset = reference_group["reference_asset"]
            reference_quat_w = reference_asset.data.body_quat_w[
                :, reference_group["reference_body_id"]
            ]
            force_w = _quat_apply_force_batch(reference_quat_w, local_forces)

            if reference_group is direct_reference_group:
                forces.copy_(force_w)
            else:
                for index, entry in enumerate(reference_group["entries"]):
                    forces[:, entry["body_slot"]].add_(force_w[:, index])

        asset.set_external_force_and_torque(
            forces=forces,
            torques=asset_write["torque_buffer"],
            body_ids=asset_write["body_ids"],
            env_ids=None,
            is_global=True,
        )


def reset_external_force_channels(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | list[int] | None = None,
):
    """Clear every named external-force channel for the selected environments."""
    channels = getattr(env, "external_force_channels", None)
    if not channels:
        return
    if env_ids is None:
        env_ids = slice(None)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, dtype=torch.long, device=env.device)

    for channel in channels.values():
        channel["state"][env_ids] = 0
        channel["current"][env_ids] = 0.0
        channel["ramp_start"][env_ids] = 0.0
        channel["ramp_target"][env_ids] = 0.0
        channel["ramp_elapsed"][env_ids] = 0.0
        channel["ramp_time"][env_ids] = 0.0


def push_by_setting_velocity_obs_xy(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset by setting the root velocity to a random value within the given ranges.

    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
    It samples the root velocity from the given ranges and sets the velocity into the physics simulation.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # velocities
    vel_w = asset.data.root_vel_w[env_ids]
    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    vel_add = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    env.event_push_vel_buf[env_ids] = vel_add[:, :2]
    vel_w += vel_add
    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


class randomize_actuator_gains_plus(ManagerTermBase):
    """Randomize the actuator gains in an articulation by adding, scaling, or setting random values.

    This function allows randomizing the actuator stiffness and damping gains.

    The function samples random values from the given distribution parameters and applies the operation to the joint properties.
    It then sets the values into the actuator models. If the distribution parameters are not provided for a particular property,
    the function does not modify the property.

    .. tip::
        For implicit actuators, this function uses CPU tensors to assign the actuator gains into the simulation.
        In such cases, it is recommended to use this function only during the initialization of the environment.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            TypeError: If `params` is not a tuple of two numbers.
            ValueError: If the operation is not supported.
            ValueError: If the lower bound is negative or zero when not allowed.
            ValueError: If the upper bound is less than the lower bound.
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        # check for valid operation
        if cfg.params["operation"] == "scale":
            if "stiffness_distribution_params" in cfg.params:
                _validate_scale_range(
                    cfg.params["stiffness_distribution_params"], "stiffness_distribution_params", allow_zero=False
                )
            if "damping_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["damping_distribution_params"], "damping_distribution_params")
            if "kt_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["kt_distribution_params"], "kt_distribution_params")
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_actuator_gains' does not support operation:"
                f" '{cfg.params['operation']}'."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        stiffness_distribution_params: tuple[float, float] | None = None,
        damping_distribution_params: tuple[float, float] | None = None,
        kt_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        # Resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)

        def randomize(data: torch.Tensor, params: tuple[float, float]) -> torch.Tensor:
            return _randomize_prop_by_op(
                data, params, dim_0_ids=None, dim_1_ids=actuator_indices, operation=operation, distribution=distribution
            )

        # Loop through actuators and randomize gains
        for actuator in self.asset.actuators.values():
            if isinstance(self.asset_cfg.joint_ids, slice):
                # we take all the joints of the actuator
                actuator_indices = slice(None)
                if isinstance(actuator.joint_indices, slice):
                    global_indices = slice(None)
                else:
                    global_indices = torch.tensor(actuator.joint_indices, device=self.asset.device)
            elif isinstance(actuator.joint_indices, slice):
                # we take the joints defined in the asset config
                global_indices = actuator_indices = torch.tensor(self.asset_cfg.joint_ids, device=self.asset.device)
            else:
                # we take the intersection of the actuator joints and the asset config joints
                actuator_joint_indices = torch.tensor(actuator.joint_indices, device=self.asset.device)
                asset_joint_ids = torch.tensor(self.asset_cfg.joint_ids, device=self.asset.device)
                # the indices of the joints in the actuator that have to be randomized
                actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
                if len(actuator_indices) == 0:
                    continue
                # maps actuator indices that have to be randomized to global joint indices
                global_indices = actuator_joint_indices[actuator_indices]
            # Randomize kt
            kt = actuator.stiffness[env_ids].clone()
            if kt_distribution_params is not None:
                min_val, max_val = kt_distribution_params
                kt = math_utils.sample_uniform(min_val, max_val, kt.shape, device=kt.device)
            else:
                kt = torch.ones_like(kt)
            # Randomize stiffness
            if stiffness_distribution_params is not None:
                stiffness = actuator.stiffness[env_ids].clone()
                stiffness[:, actuator_indices] = self.asset.data.default_joint_stiffness[env_ids][
                    :, global_indices
                ].clone()
                randomize(stiffness, stiffness_distribution_params)
                stiffness[:, actuator_indices] *= kt[:, actuator_indices]
                actuator.stiffness[env_ids] = stiffness
                if isinstance(actuator, ImplicitActuator):
                    self.asset.write_joint_stiffness_to_sim(
                        stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids
                    )
            # Randomize damping
            if damping_distribution_params is not None:
                damping = actuator.damping[env_ids].clone()
                damping[:, actuator_indices] = self.asset.data.default_joint_damping[env_ids][:, global_indices].clone()
                randomize(damping, damping_distribution_params)
                damping[:, actuator_indices] *= kt[:, actuator_indices]
                actuator.damping[env_ids] = damping
                if isinstance(actuator, ImplicitActuator):
                    self.asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)

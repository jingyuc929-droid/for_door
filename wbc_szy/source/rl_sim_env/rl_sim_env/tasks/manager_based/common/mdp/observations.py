# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer, RayCaster, ContactSensor
from isaaclab.sensors.ray_caster import RayCasterCameraCfg
from isaaclab.utils.math import quat_apply_yaw, quat_inv, quat_rotate_inverse, yaw_quat, quat_mul, quat_from_euler_xyz, quat_apply, quat_apply_inverse
from isaaclab.utils.math import transform_points
# from torch_cluster import fps, radius
from rl_sim_env.tasks.manager_based.common.utils.grid import extract_3x3_patches, grid_pattern
from rl_sim_env.tasks.manager_based.common.utils.projected_com_frame import (
    compute_projected_com_yaw_frame,
    world_to_projected_frame,
)
from torch.func import vmap

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


"""
Root state.
"""


def base_lin_xy_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, :2]


def base_ang_yaw_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b[:, 2].unsqueeze(-1)


def base_height_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
                foot_sensor_cfg: SceneEntityCfg = None, contact_sensor_cfg: SceneEntityCfg = None) -> torch.Tensor:
    """Base height calculated from foot positions and contact states.
    
    Calculates the base height as the mean of foot positions relative to the base frame
    for feet that are in contact with the ground.
    """
    # 获取足端传感器（FrameTransformer）
    foot_sensor: FrameTransformer = env.scene.sensors[foot_sensor_cfg.name]
    # 获取足端相对于机身的位置，shape: (num_envs, num_feet, 3)
    foot_pos_b = foot_sensor.data.target_pos_source
    
    # 获取接触传感器
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    # 计算接触状态，shape: (num_envs, num_feet)
    contacts = contact_sensor.data.net_forces_w_history[:, :, contact_sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 0.5
    
    # 提取z方向位置，shape: (num_envs, num_feet)
    foot_z = foot_pos_b[:, :, 2]
    num_contacts = contacts.sum(dim=1, keepdim=True)
    
    # 如果有接触，使用接触足端的均值；否则使用所有足端的均值
    mean_z = torch.where(
        num_contacts > 0,
        (foot_z * contacts.float()).sum(dim=1, keepdim=True) / num_contacts,
        foot_z.mean(dim=1, keepdim=True)
    )
    return mean_z.abs()


def foot_positions(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Foot positions from the base link frame."""
    sensor: FrameTransformer = env.scene.sensors[sensor_cfg.name]
    return sensor.data.target_pos_source.flatten(start_dim=1)

def foot_clearance(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    foot_sensor_cfgs: dict[str, SceneEntityCfg],
    foot_tf_cfg: SceneEntityCfg,
    offset: float = 0.0,
    center_index: int | None = None,
) -> torch.Tensor:
    """估计各足端距离地面的高度（逐腿返回）。

    单射线 foot scanner 直接使用中心射线；旧的 3×3 scanner 仍只取中间射线。
    将射线击中的地面高度与足端高度做差，得到正向的“足端-地面”距离。
    """
    num_envs = env.scene.num_envs
    num_feet = len(foot_sensor_cfgs)

    if num_feet == 0:
        return torch.zeros((num_envs, 0), device=env.device, dtype=torch.float32)

    asset = env.scene[asset_cfg.name]
    foot_tf: FrameTransformer = env.scene.sensors[foot_tf_cfg.name]

    origin_pos = asset.data.root_pos_w.to(env.device).unsqueeze(1)
    origin_quat = asset.data.root_quat_w.to(env.device)

    foot_pos_world = foot_tf.data.target_pos_w[:, :num_feet, :].to(env.device)
    foot_pos_yaw = transform_points(
        foot_pos_world - origin_pos,
        pos=None,
        quat=quat_inv(yaw_quat(origin_quat)),
    )

    clearances = []
    grid = None
    for idx, sensor_cfg in enumerate(foot_sensor_cfgs.values()):
        lidar_sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        num_rays = int(lidar_sensor.data.ray_hits_w.shape[1])
        if num_rays == 1:
            ray_starts = foot_pos_yaw[:, idx, :].unsqueeze(1)
            ray_starts[..., 2] = 20.0
            lidar_sensor.ray_starts = ray_starts
            active_center_index = 0
        else:
            if grid is None:
                size = 0.1
                grid = grid_pattern(length=size, width=size, resolution=size / 2.0, device=env.device)
                grid = grid.repeat(num_envs, 1, 1)
                grid[..., 2] = 20.0
            active_center_index = grid.shape[1] // 2 if center_index is None else int(center_index)
            if not (0 <= active_center_index < num_rays):
                raise ValueError(
                    f"center_index={active_center_index} 超出射线数量范围 (0, {num_rays - 1})。"
                )
            lidar_sensor.ray_starts = foot_pos_yaw[:, idx, :].unsqueeze(1) + grid

        ray_z = lidar_sensor.data.ray_hits_w[..., 2]

        foot_z = foot_pos_world[:, idx, 2]
        ground_z = ray_z[:, active_center_index]

        clearances.append((foot_z - ground_z - offset).unsqueeze(1))

    clearance = torch.cat(clearances, dim=1)
    clearance = torch.nan_to_num(clearance, nan=1.0, posinf=1.0, neginf=1.0)
    clearance = torch.clip(clearance, -1.0, 1.0)
    env.foot_clearance_buf = clearance.detach()

    return clearance

def depth_images(
    env: ManagerBasedEnv,
    sensor_cfg: RayCasterCameraCfg,
    min_depth: float = 0.0,
    max_depth: float = 2.0,
) -> torch.Tensor:
    """返回前向深度相机的 depth-to-image-plane 数据。

    先对原始 depth 按 ``[min_depth, max_depth]`` 做裁剪，然后按该区间做归一化，最后
    展平为形状为 ``(env_num, height*width)`` 的张量。
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    outputs = getattr(sensor.data, "output", None)
    if outputs is None:
        raise RuntimeError(f"传感器 `{sensor_cfg.name}` 未提供 output 数据。")

    depth = outputs.get("distance_to_image_plane")
    if depth is None:
        raise RuntimeError(f"传感器 `{sensor_cfg.name}` 未提供 distance_to_image_plane 数据。")

    if max_depth <= min_depth:
        raise ValueError(f"depth_images: 期望 max_depth ({max_depth}) > min_depth ({min_depth})。")

    depth = depth.to(env.device)
    depth = torch.nan_to_num(depth, nan=0.0, posinf=max_depth, neginf=min_depth)

    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    elif depth.ndim != 3:
        raise RuntimeError(
            f"`{sensor_cfg.name}` 的 depth_to_image_plane 数据维度 {depth.shape} 非预期，需要 (env_num, H, W)。"
        )

    # 先按给定区间裁剪，再归一化到 [0, 1]
    depth = depth.clamp(min=min_depth, max=max_depth)
    depth = (depth - min_depth) / (max_depth - min_depth) - 0.5

    # 展平为 (env_num, height*width)，按行优先（先展平 width，再展平 height）
    # 搞清楚depth原始先按照weight是从左到右还是从右到左，height是从上到下还是从下到上，待检查
    return depth.contiguous().view(depth.shape[0], -1)

def push_vel(env: ManagerBasedEnv) -> torch.Tensor:
    if not hasattr(env, "event_push_vel_buf"):
        num_envs = env.scene.num_envs
        device = getattr(env, "device", torch.device("cpu"))
        env.event_push_vel_buf = torch.zeros((num_envs, 2), device=device, dtype=torch.float32, requires_grad=False)
    # print("event_push_vel_buf", env.event_push_vel_buf.shape)
    return env.event_push_vel_buf


def push_force(env: ManagerBasedEnv) -> torch.Tensor:
    """当前施加的外力（2D：fx, fy）。

    约定：该 2D 力来自事件缓冲 ``env.event_push_force_buf``，用于特权观测/奖励偏置。
    当前约定（倾斜地形）：
    - fx：外力在躯干前向方向（yaw+pitch，忽略 roll）上的投影标量
    - fy：暂保持为 yaw 对齐水平系的 y 分量
    """
    if not hasattr(env, "event_push_force_buf"):
        num_envs = env.scene.num_envs
        device = getattr(env, "device", torch.device("cpu"))
        env.event_push_force_buf = torch.zeros((num_envs, 2), device=device, dtype=torch.float32, requires_grad=False)
    return env.event_push_force_buf


def push_yaw_torque(env: ManagerBasedEnv) -> torch.Tensor:
    """当前施加的外部 yaw 扭矩（1D：τz）。

    约定：该 τz 是绕世界 Z 轴（重力轴）的扭矩，用于特权观测/奖励偏置。
    """
    if not hasattr(env, "event_push_yaw_torque_buf"):
        num_envs = env.scene.num_envs
        device = getattr(env, "device", torch.device("cpu"))
        env.event_push_yaw_torque_buf = torch.zeros(
            (num_envs, 1), device=device, dtype=torch.float32, requires_grad=False
        )
    return env.event_push_yaw_torque_buf


def _get_static_obs_cache(env: ManagerBasedEnv) -> dict:
    cache = getattr(env, "_static_observation_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_static_observation_cache", cache)
    return cache


def random_com_obs(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    cache_static: bool = False,
) -> torch.Tensor:
    cache_key = ("random_com", asset_cfg.name, repr(asset_cfg.body_ids))
    cache = _get_static_obs_cache(env) if cache_static else None
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    asset: RigidObject = env.scene[asset_cfg.name]
    # 先选所需 body/xyz 再 clone，避免每步复制所有 body 的完整 7D COM 数据。
    coms = (
        asset.root_physx_view.get_coms()[:, asset_cfg.body_ids, :3]
        .clone()
        .squeeze(1)
    )
    # print("coms", coms.shape)
    coms = coms.to(env.device)
    if cache is not None:
        cache[cache_key] = coms
    return coms


def random_mass_obs(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    cache_static: bool = False,
) -> torch.Tensor:
    cache_key = ("random_mass", asset_cfg.name, repr(asset_cfg.body_ids))
    cache = _get_static_obs_cache(env) if cache_static else None
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    asset: RigidObject = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses()
    mass_obs = masses[:, asset_cfg.body_ids] - asset.data.default_mass[:, asset_cfg.body_ids]
    # print("mass_obs", mass_obs.shape)
    mass_obs = mass_obs.to(env.device)
    if cache is not None:
        cache[cache_key] = mass_obs
    return mass_obs


def random_material_obs(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    cache_static: bool = False,
) -> torch.Tensor:
    cache_key = ("random_material", asset_cfg.name, repr(asset_cfg.body_ids))
    cache = _get_static_obs_cache(env) if cache_static else None
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    asset: RigidObject = env.scene[asset_cfg.name]
    material_obs = asset.root_physx_view.get_material_properties()[:, asset_cfg.body_ids, :]

    material_obs = material_obs.reshape(material_obs.shape[0], -1)
    material_obs = material_obs.to(env.device)
    if cache is not None:
        cache[cache_key] = material_obs
    return material_obs


def joint_friction_obs(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    cache_static: bool = False,
) -> torch.Tensor:
    """关节摩擦系数（每关节独立，由 randomize_joint_parameters 写入 asset.data.joint_friction_coeff）。"""
    cache_key = ("joint_friction", asset_cfg.name, repr(asset_cfg.joint_ids))
    cache = _get_static_obs_cache(env) if cache_static else None
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    asset: Articulation = env.scene[asset_cfg.name]
    friction = asset.data.joint_friction_coeff[:, asset_cfg.joint_ids]
    friction = friction.to(env.device)
    if cache is not None:
        cache[cache_key] = friction
    return friction


def randomize_actuator_gains_obs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, kp_default: float, kd_default: float) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # Subtraction already allocates the observation tensor; cloning first only
    # adds two full-size device copies.
    kp = asset.actuators["base_legs"].stiffness - kp_default
    kd = asset.actuators["base_legs"].damping - kd_default
    actuator_gains = torch.cat([kp, kd], dim=1)
    # print("actuator_gains", actuator_gains.shape)
    return actuator_gains.to(env.device)


def randomize_actuator_lag_obs(env: ManagerBasedEnv) -> torch.Tensor:
    lag = env.scene.articulations["robot"].actuators["base_legs"].positions_delay_buffer.time_lags
    lag = lag.to(device=env.device, dtype=torch.float32)
    # print("lag", lag.shape)
    return lag.unsqueeze(1)


def generated_commands_scale(env: ManagerBasedRLEnv, command_name: str, scale: tuple[float, ...]) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    command = env.command_manager.get_command(command_name)
    cache = getattr(env, "_command_scale_tensor_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_command_scale_tensor_cache", cache)
    cache_key = (tuple(scale), command.dtype, command.device)
    scale_tensor = cache.get(cache_key)
    if scale_tensor is None:
        scale_tensor = torch.tensor(
            scale, dtype=command.dtype, device=command.device
        )
        cache[cache_key] = scale_tensor
    return command * scale_tensor


def pitch_command(env: ManagerBasedRLEnv, command_name: str = "pitch_command") -> torch.Tensor:
    """The pitch command (pitch angle) from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)


def command_9d(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """从 CommandManager 读取一个 9D 命令（用于末端目标点等）。"""
    return env.command_manager.get_command(command_name)


def _get_command_term_for_obs(env: ManagerBasedRLEnv, command_name: str):
    cmd_mgr = getattr(env, "command_manager", None)
    if cmd_mgr is None:
        return None
    if hasattr(cmd_mgr, "get_term"):
        try:
            return cmd_mgr.get_term(command_name)
        except Exception:
            return None
    return getattr(cmd_mgr, "_terms", {}).get(command_name)


def _quat_tuple_matches(lhs, rhs) -> bool:
    try:
        lhs_f = tuple(float(v) for v in lhs)
        rhs_f = tuple(float(v) for v in rhs)
    except Exception:
        return False
    return len(lhs_f) == len(rhs_f) == 4 and all(
        abs(a - b) <= 1.0e-9 for a, b in zip(lhs_f, rhs_f)
    )


def ee_target_points_error_9d(
    env: ManagerBasedRLEnv,
    command_name: str,
    ee_asset_cfg: SceneEntityCfg,
    trunk_asset_cfg: SceneEntityCfg,
    terrain_sensor_cfg: SceneEntityCfg,
    offset_distance: float,
    ee_tool_quat_offset: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    command_active_threshold: float = 1.0e-6,
    read_cached_ground_z_only: bool = False,
) -> torch.Tensor:
    """9D end-effector point tracking error in the projected COM yaw frame.

    Returns ``current_ee_points_9d - command_9d`` for the main point, x-offset
    point, and y-offset point. This mirrors the frame and TCP convention used by
    the ee tracking rewards, but exposes the signed error as a privileged
    observation for the critic.
    """
    robot: Articulation = env.scene[ee_asset_cfg.name]

    if len(ee_asset_cfg.body_ids) != 1:
        raise ValueError(
            "ee_asset_cfg.body_names must contain exactly one end-effector body, "
            f"got {len(ee_asset_cfg.body_ids)}."
        )
    if len(trunk_asset_cfg.body_ids) != 1:
        raise ValueError(
            "trunk_asset_cfg.body_names must contain exactly one trunk body, "
            f"got {len(trunk_asset_cfg.body_ids)}."
        )

    ee_body_id = ee_asset_cfg.body_ids[0]
    trunk_body_id = trunk_asset_cfg.body_ids[0]
    cmd = env.command_manager.get_command(command_name)
    cmd_active = torch.linalg.norm(cmd, dim=1) > command_active_threshold

    cmd_term = _get_command_term_for_obs(env, command_name)
    if cmd_term is not None and hasattr(
        cmd_term, "enable_ee_target_points_error_cache"
    ):
        cfg = getattr(cmd_term, "cfg", None)
        try:
            offset_matches = abs(
                float(offset_distance) - float(getattr(cmd_term, "offset_distance"))
            ) <= 1.0e-9
        except Exception:
            offset_matches = False
        cache_matches = (
            getattr(cmd_term, "ee_body_id", None) == ee_body_id
            and getattr(cmd_term, "trunk_body_id", None) == trunk_body_id
            and bool(getattr(cmd_term, "use_projected_origin", True))
            and getattr(cfg, "terrain_sensor_name", terrain_sensor_cfg.name)
            == terrain_sensor_cfg.name
            and offset_matches
            and _quat_tuple_matches(
                getattr(cfg, "ee_tool_quat_offset", (1.0, 0.0, 0.0, 0.0)),
                ee_tool_quat_offset,
            )
        )
        if cache_matches and cmd_term.enable_ee_target_points_error_cache():
            cached_error = getattr(cmd_term, "ee_target_points_error_raw", None)
            cached_step = getattr(cmd_term, "ee_target_points_error_step", -1)
            current_step = int(getattr(env, "common_step_counter", -2))
            if (
                isinstance(cached_error, torch.Tensor)
                and cached_error.shape == (env.num_envs, 9)
                and cached_step == current_step
            ):
                return torch.where(
                    cmd_active.unsqueeze(1),
                    cached_error,
                    torch.zeros_like(cached_error),
                )

    terrain_sensor: RayCaster = env.scene.sensors[terrain_sensor_cfg.name]

    frame = compute_projected_com_yaw_frame(
        asset=robot,
        trunk_body_id=trunk_body_id,
        terrain_height_sensor=terrain_sensor,
        lpf_state_key=command_name,
        read_cached_ground_z_only=read_cached_ground_z_only,
    )

    ee_pos_w = robot.data.body_pos_w[:, ee_body_id]
    ee_quat_w = robot.data.body_quat_w[:, ee_body_id]

    ee_pos_p = world_to_projected_frame(ee_pos_w, frame)
    ee_quat_p = quat_mul(quat_inv(frame.yaw_quat_w), ee_quat_w)

    tool_offset = torch.tensor(
        ee_tool_quat_offset,
        dtype=torch.float32,
        device=env.device,
    ).repeat(env.num_envs, 1)
    ee_quat_p = quat_mul(ee_quat_p, tool_offset)

    local_x = torch.zeros((env.num_envs, 3), device=env.device)
    local_y = torch.zeros((env.num_envs, 3), device=env.device)
    local_x[:, 0] = float(offset_distance)
    local_y[:, 1] = float(offset_distance)

    ee_x_p = ee_pos_p + quat_apply(ee_quat_p, local_x)
    ee_y_p = ee_pos_p + quat_apply(ee_quat_p, local_y)

    ee_points = torch.cat((ee_pos_p, ee_x_p, ee_y_p), dim=-1)
    error = ee_points - cmd
    return torch.where(cmd_active.unsqueeze(1), error, torch.zeros_like(error))


def ee_target_points_error_delta_9d(
    env: ManagerBasedRLEnv,
    command_name: str,
    ee_asset_cfg: SceneEntityCfg,
    trunk_asset_cfg: SceneEntityCfg,
    terrain_sensor_cfg: SceneEntityCfg,
    offset_distance: float,
    ee_tool_quat_offset: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    command_active_threshold: float = 1.0e-6,
    read_cached_ground_z_only: bool = False,
) -> torch.Tensor:
    """Per-step change of the 9D end-effector point tracking error.

    This returns ``error_t - error_{t-1}`` using the same 9D signed error as
    :func:`ee_target_points_error_9d`. Newly reset environments emit zeros and
    refresh their previous-error cache with the post-reset value.
    """
    error = ee_target_points_error_9d(
        env=env,
        command_name=command_name,
        ee_asset_cfg=ee_asset_cfg,
        trunk_asset_cfg=trunk_asset_cfg,
        terrain_sensor_cfg=terrain_sensor_cfg,
        offset_distance=offset_distance,
        ee_tool_quat_offset=ee_tool_quat_offset,
        command_active_threshold=command_active_threshold,
        read_cached_ground_z_only=read_cached_ground_z_only,
    )

    current_step = int(getattr(env, "common_step_counter", -1))
    prev_attr = f"_ee_target_points_error_delta_prev__{command_name}"
    prev_step_attr = f"_ee_target_points_error_delta_prev_step__{command_name}"
    delta_attr = f"_ee_target_points_error_delta_cached__{command_name}"
    delta_step_attr = f"_ee_target_points_error_delta_cached_step__{command_name}"

    reset_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    episode_length_buf = getattr(env, "episode_length_buf", None)
    if isinstance(episode_length_buf, torch.Tensor) and episode_length_buf.shape[0] == env.num_envs:
        reset_mask |= episode_length_buf.to(device=env.device) <= 0
    reset_buf = getattr(env, "reset_buf", None)
    if isinstance(reset_buf, torch.Tensor) and reset_buf.shape[0] == env.num_envs:
        reset_mask |= reset_buf.to(device=env.device, dtype=torch.bool)

    cached_delta = getattr(env, delta_attr, None)
    cached_step = getattr(env, delta_step_attr, None)
    if (
        isinstance(cached_delta, torch.Tensor)
        and cached_delta.shape == error.shape
        and cached_step == current_step
    ):
        if reset_mask.any():
            cached_delta = cached_delta.clone()
            cached_delta[reset_mask] = 0.0
            prev_error = getattr(env, prev_attr, None)
            if isinstance(prev_error, torch.Tensor) and prev_error.shape == error.shape:
                prev_error = prev_error.clone()
                prev_error[reset_mask] = error.detach()[reset_mask]
            else:
                prev_error = error.detach().clone()
            setattr(env, prev_attr, prev_error)
            setattr(env, prev_step_attr, current_step)
            setattr(env, delta_attr, cached_delta)
        return cached_delta

    prev_error = getattr(env, prev_attr, None)
    prev_step = getattr(env, prev_step_attr, None)
    if (
        isinstance(prev_error, torch.Tensor)
        and prev_error.shape == error.shape
        and prev_step == current_step - 1
    ):
        delta = error - prev_error
    else:
        delta = torch.zeros_like(error)

    if reset_mask.any():
        delta = delta.clone()
        delta[reset_mask] = 0.0

    setattr(env, prev_attr, error.detach().clone())
    setattr(env, prev_step_attr, current_step)
    setattr(env, delta_attr, delta)
    setattr(env, delta_step_attr, current_step)
    return delta


def pose_2d_command_with_remaining_time(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    command = env.command_manager.get_command(command_name)
    xy_error = command[:, :2]
    yaw_error = command[:, 3]
    if not hasattr(env, "remaining_episode_time"):
        remaining_time = torch.full((env.num_envs, ), env.max_episode_length_s, dtype=torch.float32, device=env.device)
    else:
        remaining_time = env.remaining_episode_time
    pose_2d_command = torch.cat([xy_error, yaw_error.unsqueeze(-1), remaining_time.unsqueeze(-1)], dim=1)
    return pose_2d_command


def height_scan_fix(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.4) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # 原始的射线击中 Z 值
    ray_z = sensor.data.ray_hits_w[..., 2]
    # 把 nan 全部替换成 0.0
    # ray_z = torch.where(torch.isnan(ray_z), 0.0, ray_z)
    # ray_z.nan_to_num_(0.0, 2.0, -2.0)
    height_b = sensor.data.pos_w[:, 2].unsqueeze(1)- offset - ray_z
    height_b.nan_to_num_(1.0, 1.0, 1.0)
    height_b=torch.clip(height_b,-1.0,1.0)

    return height_b


def get_foot_contact_flags(env: ManagerBasedEnv, contact_sensor_cfg: SceneEntityCfg = None) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    foot_contact_flags = contact_sensor.data.net_forces_w_history[:, :, contact_sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 0.5
    return foot_contact_flags.float().to(env.device)


def get_bodies_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # 获取指令link的质量
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.root_physx_view.get_masses()[:, asset_cfg.body_ids].to(env.device)


def get_friction_coeff(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    # 获取材料属性，返回形状为 (num_envs, num_shapes, 3)，其中索引0是static_friction
    material_props = asset.root_physx_view.get_material_properties()[:, asset_cfg.body_ids, 0]
    return material_props.to(env.device)


def get_com_pos_xy_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """计算指定link的加权质心位置xy（相对于机身body frame）

    计算公式: sum(mass_i * pos_i) / sum(mass_i)，取前两维
    其中pos_i是各link相对于root的body frame坐标
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # 获取指定link的质量，shape: (num_envs, num_bodies)
    masses = asset.root_physx_view.get_masses()[:, asset_cfg.body_ids].to(env.device)

    # 获取指定link的质心位置（世界坐标），shape: (num_envs, num_bodies, 3)
    com_pos_w = asset.root_physx_view.get_coms()[:, asset_cfg.body_ids, :3].to(env.device)

    # 转换到root的body frame
    # com_pos_w - root_pos_w 得到相对于root的世界坐标偏移
    # 然后通过quat_rotate_inverse转到body frame
    num_envs = com_pos_w.shape[0]
    num_bodies = com_pos_w.shape[1]
    
    # 计算相对位置（世界坐标）
    relative_pos_w = com_pos_w - asset.data.root_pos_w.unsqueeze(1)  # (num_envs, num_bodies, 3)
    
    # 将quat和vec都展平到相同的batch维度
    root_quat_expanded = asset.data.root_quat_w.unsqueeze(1).expand(num_envs, num_bodies, 4)  # (num_envs, num_bodies, 4)
    root_quat_flat = root_quat_expanded.reshape(num_envs * num_bodies, 4)  # (num_envs*num_bodies, 4)
    relative_pos_flat = relative_pos_w.reshape(num_envs * num_bodies, 3)  # (num_envs*num_bodies, 3)
    
    # 旋转到body frame
    com_pos_b_flat = quat_apply_inverse(root_quat_flat, relative_pos_flat)  # (num_envs*num_bodies, 3)
    com_pos_b = com_pos_b_flat.reshape(num_envs, num_bodies, 3).to(env.device)  # (num_envs, num_bodies, 3)

    # 计算加权质心: sum(mass * pos) / sum(mass)
    # masses: (num_envs, num_bodies) -> (num_envs, num_bodies, 1)
    total_mass = masses.sum(dim=1, keepdim=True)  # (num_envs, 1)
    weighted_com = (masses.unsqueeze(-1) * com_pos_b).sum(dim=1) / total_mass  # (num_envs, 3)

    # 取前两维(x, y)
    return weighted_com[:, :2]


def external_forces_torques_applied(env: ManagerBasedEnv) -> torch.Tensor:
    if not hasattr(env, "event_apply_forces_torques_buf"):
        num_envs = env.scene.num_envs
        device = getattr(env, "device", torch.device("cpu"))
        env.event_apply_forces_torques_buf = torch.zeros((num_envs, 6), device=device, dtype=torch.float32, requires_grad=False)
    env.event_apply_forces_torques_buf = env.event_apply_forces_torques_buf.to(env.device)
    # print("event_apply_forces_torques_buf", env.event_apply_forces_torques_buf.shape)
    return env.event_apply_forces_torques_buf


def ee_external_forces_applied(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """末端外力（projected COM yaw frame 3D）。

    wrench buf 存的是 inertial world 系力（由 ``update_ee_external_force`` 写入，
    ``is_global=True`` 施加）。这里把它旋到躯干 yaw frame —— 与 ``ee_target_points``
    命令、``force_compliance`` 的 delta_p 同系。用 ``yaw_quat(root_quat_w)``，与
    ``compute_projected_com_yaw_frame`` 的 yaw_quat_w 一致（仅依赖躯干 yaw，不需地面高度）。
    若事件未启用，则返回全零。
    """
    wrench = external_forces_torques_applied(env)  # (E, 6) world frame
    f_w = wrench[:, :3]
    robot: Articulation = env.scene[asset_cfg.name]
    yaw_q = yaw_quat(robot.data.root_quat_w)
    # world -> projected COM yaw frame
    return quat_apply(quat_inv(yaw_q), f_w)


def get_joint_kp(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.actuators["base_legs"].stiffness.clone().to(env.device)


def get_joint_kd(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.actuators["base_legs"].damping.clone().to(env.device)


def motor_offset(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns the motor offset (random offset) generated by reset_joint_offset event.
    
    This retrieves the _random_offset from the action manager's joint_pos term,
    which is set by the reset_joint_offset event during environment reset.
    """
    if hasattr(env.action_manager._terms["joint_pos"], "_random_offset"):
        return env.action_manager._terms["joint_pos"]._random_offset.clone().to(env.device)
    else:
        # 如果没有_random_offset属性，返回零张量
        asset: Articulation = env.scene[asset_cfg.name]
        num_joints = len(asset_cfg.joint_ids) if isinstance(asset_cfg.joint_ids, list) else asset.num_bodies
        return torch.zeros((env.num_envs, num_joints), device=asset.device).to(env.device)


def foot_scan(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, foot_sensor_cfgs: dict[str, SceneEntityCfg], foot_tf_cfg: SceneEntityCfg, offset: float = 0.0) -> torch.Tensor:
    data = []
    asset = env.scene[asset_cfg.name]
    size = 0.1
    grid = grid_pattern(length=size, width=size, resolution= size / 2.0, device=env.device).repeat(asset.data.root_pos_w.shape[0], 1, 1)  # 生成9乘3的网格点
    grid[..., 2] = 20.0
    origin_pos = asset.data.root_pos_w.to(env.device).unsqueeze(1)   # (B, 1, 3).
    origin_quat = asset.data.root_quat_w.to(env.device)  # (B, 4).
    foot_tf: FrameTransformer = env.scene.sensors[foot_tf_cfg.name]
    foot_pos_yaw = transform_points(
        foot_tf.data.target_pos_w[..., :3] - origin_pos,
        pos=None,
        quat=quat_inv(yaw_quat(origin_quat))
    )
    # 旋转grid
    # grid_b = quat_apply(yaw_quat(origin_quat), grid)
    # print("foot_pos_yaw", foot_pos_yaw[:, 0, :].shape)
    # print("grid", grid.shape)
    ray_starts_pos = {"fl_foot_scanner": foot_pos_yaw[:, 0, :].unsqueeze(1) + grid,
                      "fr_foot_scanner": foot_pos_yaw[:, 1, :].unsqueeze(1) + grid,
                      "rl_foot_scanner": foot_pos_yaw[:, 2, :].unsqueeze(1) + grid,
                      "rr_foot_scanner": foot_pos_yaw[:, 3, :].unsqueeze(1) + grid}
    foot_pos_w = {
        "fl_foot_scanner": foot_tf.data.target_pos_w[:, 0, :],
        "fr_foot_scanner": foot_tf.data.target_pos_w[:, 1, :],
        "rl_foot_scanner": foot_tf.data.target_pos_w[:, 2, :],
        "rr_foot_scanner": foot_tf.data.target_pos_w[:, 3, :]
    }

    if not hasattr(env, "foot_scan_buf"):
        num_envs = env.scene.num_envs
        device = getattr(env, "device", torch.device("cpu"))
        env.foot_scan_buf = torch.zeros((num_envs, 36), device=device, dtype=torch.float32, requires_grad=False)
    for lidar_sensor_cfg in foot_sensor_cfgs.values():
        lidar_sensor: RayCaster = env.scene.sensors[lidar_sensor_cfg.name]
        lidar_sensor.ray_starts = ray_starts_pos[lidar_sensor_cfg.name]
        sensor_z = foot_pos_w[lidar_sensor_cfg.name][:, 2].unsqueeze(1)
        ray_z = lidar_sensor.data.ray_hits_w[..., 2]
        ray_z.nan_to_num_(0, -2.0, -2.0)
        data.append(ray_z - sensor_z - offset)
    ray_z = torch.cat(data, dim=1).to(env.device)
    env.foot_scan_buf = ray_z
    # print("ray_z", ray_z)
    return ray_z

def foot_scan_without_transform(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, foot_sensor_cfgs: dict[str, SceneEntityCfg], foot_tf_cfg: SceneEntityCfg, offset: float = 0.0) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    foot_tf: FrameTransformer = env.scene.sensors[foot_tf_cfg.name]
    foot_pos_w = {
        "fl_foot_scanner": foot_tf.data.target_pos_w[:, 0, :],
        "fr_foot_scanner": foot_tf.data.target_pos_w[:, 1, :],
        "rl_foot_scanner": foot_tf.data.target_pos_w[:, 2, :],
        "rr_foot_scanner": foot_tf.data.target_pos_w[:, 3, :]
    }
    if not hasattr(env, "foot_scan_buf"):
        num_envs = env.scene.num_envs
        device = getattr(env, "device", torch.device("cpu"))
        env.foot_scan_buf = torch.zeros((num_envs, 36), device=device, dtype=torch.float32, requires_grad=False)
    data = []
    for lidar_sensor_cfg in foot_sensor_cfgs.values():
        lidar_sensor: RayCaster = env.scene.sensors[lidar_sensor_cfg.name]
        sensor_z = foot_pos_w[lidar_sensor_cfg.name][:, 2].unsqueeze(1)
        ray_z = lidar_sensor.data.ray_hits_w[..., 2]
        ray_z.nan_to_num_(0.0, -2.0, -2.0)
        data.append(ray_z - sensor_z - offset)
    env.foot_scan_buf = torch.cat(data, dim=1).to(env.device)
    return env.foot_scan_buf


# phase observation
def phase_obs(env: ManagerBasedEnv, period: float) -> torch.Tensor:
    """Phase observation."""
    # ROBOT_FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    offset = 0.5
    # 注意：在 ObservationManager 初始化 (_prepare_terms) 阶段调用本函数时，
    # 自定义 RL 环境（如 LocomotionRLEnv）里的 episode_length_buf 可能尚未创建，
    # 这会导致 AttributeError。这里做一个安全保护：
    device = getattr(env, "device", torch.device("cuda:0"))
    if hasattr(env, "episode_length_buf"):
        # 正常训练/采样阶段，使用真实的 episode_length_buf
        phase_fl = (env.episode_length_buf.to(device=device, dtype=torch.float32) * env.step_dt) % period / period
    else:
        # 仅在初始化推断 shape 时走到这里：用全零占位，shape 正确即可
        phase_fl = torch.zeros(env.scene.num_envs, device=device, dtype=torch.float32)

    phase_rr = phase_fl
    phase_fr = (phase_fl + offset) % 1
    phase_rl = phase_fr
    if not hasattr(env, "leg_phase"):
        env.leg_phase = torch.zeros((env.scene.num_envs, 4), device=device, dtype=torch.float32, requires_grad=False)
    env.leg_phase = torch.cat([phase_fl.unsqueeze(1), phase_fr.unsqueeze(1), phase_rl.unsqueeze(1), phase_rr.unsqueeze(1)], dim=-1)
    phase_sin_cos = torch.stack([2 * torch.pi * torch.sin(phase_fl), 2 * torch.pi * torch.cos(phase_fl)], dim=-1)
    return phase_sin_cos


def sample_in_obb_out_sphere(points: torch.Tensor,
                             half_extents: torch.Tensor,
                             sphere_radius: float,
                             B: int,
                             ) -> torch.Tensor:
    """
    在每个环境的 OBB 内且球外的点上，随机采样 B 个；若某环境无有效点，则该环境全部返回(0,0,0)。
    """
    N, M, _ = points.shape
    # print("points", points.shape)
    # 1. 生成有效掩码
    inside_obb = (points.abs() <= half_extents[None, None, :]).all(dim=-1)  # (N, M)
    dist2 = (points ** 2).sum(dim=-1)                                 # (N, M)
    outside_sphere = dist2 > sphere_radius**2                                  # (N, M)
    valid_mask = inside_obb & outside_sphere                               # (N, M), bool

    # 2. 给有效点分配随机键 key，无效点设为大值 2
    rnd = torch.rand((N, M), device=points.device, dtype=points.dtype)
    rnd = torch.where(valid_mask, rnd, torch.full_like(rnd, 2.0))

    # 3. 取最小 B 个 key 的索引 —— 等价于不放回随机抽样
    #    如果某行有效点 < B，则后面的 key 都是 2，会被选中但随后被置零
    _, idx = rnd.topk(k=B, dim=1, largest=False)      # idx: (N, B)

    # 4. 根据 idx 拿出点
    #    gather 需要 idx 在最后一维上扩成三通道
    gathered = points.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))  # (N, B, 3)

    # 5. 把那些“本就无效”的位置置零（补零）
    valid_sel = valid_mask.gather(1, idx)          # (N, B), bool
    sampled = gathered * valid_sel.unsqueeze(-1)  # (N, B, 3)

    return sampled


def lidar_voxel_obs(env: ManagerBasedEnv,
                    half_size: tuple[float, float, float],
                    sphere_radius: float,
                    voxel_max_num: int,
                    fps_ratio: float,
                    ball_query_r: float,
                    ball_query_k: int,
                    rpy_range: tuple[float, float, float],
                    xyz_range: tuple[float, float, float],
                    asset_cfg: SceneEntityCfg,
                    sensor_cfg: dict[str, SceneEntityCfg]) -> torch.Tensor:
    lidar_front = env.scene.sensors[sensor_cfg["front_lidar"].name]
    lidar_back = env.scene.sensors[sensor_cfg["back_lidar"].name]

    # —— 初始化随机扰动，只做一次 —— #
    if not hasattr(env, "lidar_random_sample_buf"):
        num_envs = env.scene.num_envs
        device = getattr(env, "device", torch.device("cpu"))
        rpy_max = torch.tensor(rpy_range, device=device).view(1, 3)
        xyz_max = torch.tensor(xyz_range, device=device).view(1, 3)

        def rand_buf(max_val):
            return (torch.rand(num_envs, 3, device=device) * 2 - 1) * max_val

        random_rpy_front = rand_buf(rpy_max)
        random_xyz_front = rand_buf(xyz_max)
        random_rpy_back = rand_buf(rpy_max)
        random_xyz_back = rand_buf(xyz_max)

        # 生成四元数
        roll, pitch, yaw = random_rpy_front.unbind(dim=1)
        random_quat_front = quat_from_euler_xyz(roll, pitch, yaw)
        roll, pitch, yaw = random_rpy_back.unbind(dim=1)
        random_quat_back = quat_from_euler_xyz(roll, pitch, yaw)

        buf = {
            "random_quat_front": random_quat_front,
            "random_xyz_front": random_xyz_front,
            "random_quat_back": random_quat_back,
            "random_xyz_back": random_xyz_back,
        }
        env.lidar_random_sample_buf = buf
    random_buf = env.lidar_random_sample_buf
    # 1. OBB 半尺寸
    half = torch.tensor(half_size, device=env.device)

    # 2. 机器人位姿
    asset = env.scene[asset_cfg.name]
    origin_pos = asset.data.root_pos_w.to(env.device)
    origin_quat = asset.data.root_quat_w.to(env.device)

    # 3. 读取并合并前后 LiDAR 点云，转到 yaw 局部坐标系
    front = transform_points(
        lidar_front.data.ray_hits_w[..., :3] - origin_pos.unsqueeze(1) + random_buf["random_xyz_front"].unsqueeze(1),
        pos=None,
        quat=quat_mul(random_buf["random_quat_front"], quat_inv(yaw_quat(origin_quat)))
    )
    back = transform_points(
        lidar_back.data.ray_hits_w[..., :3] - origin_pos.unsqueeze(1) + random_buf["random_xyz_back"].unsqueeze(1),
        pos=None,
        quat=quat_mul(random_buf["random_quat_back"], quat_inv(yaw_quat(origin_quat)))
    )
    ray_pos_yaw = torch.cat([front, back], dim=1)
    ray_pos_yaw.nan_to_num_(0.0, 0.0, 0.0)

    env_num = env.scene.num_envs

    # 4. 体素下采样（OBB 内 & 球外）
    filt_pts_yaw = sample_in_obb_out_sphere(
        ray_pos_yaw, half, sphere_radius,
        voxel_max_num
    )

    # 5. FPS
    pts_flat = filt_pts_yaw.reshape(-1, 3)
    batch_fps = torch.arange(env_num, device=env.device).unsqueeze(1) \
        .expand(-1, voxel_max_num).reshape(-1)
    idx_flat = fps(pts_flat, batch=batch_fps,
                   ratio=fps_ratio, random_start=True)
    idx_flat = idx_flat.reshape(env_num, -1)
    filt_pts_yaw_fps = pts_flat[idx_flat]

    # 6. Radius Search
    x_flat = filt_pts_yaw.reshape(-1, 3)
    y_flat = filt_pts_yaw_fps.reshape(-1, 3)
    batch_x = torch.arange(env_num, device=env.device).unsqueeze(1) \
        .expand(-1, voxel_max_num).reshape(-1)
    batch_y = torch.arange(env_num, device=env.device).unsqueeze(1) \
        .expand(-1, int(voxel_max_num * fps_ratio)).reshape(-1)
    assign = radius(x_flat, y_flat, r=ball_query_r,
                    batch_x=batch_x, batch_y=batch_y,
                    max_num_neighbors=ball_query_k * 2)
    qy, px = assign[0], assign[1]

    # 7. 排序 & 计数 & 前缀和
    order = torch.argsort(qy)
    qy_s, px_s = qy[order], px[order]
    tot = env_num * int(voxel_max_num * fps_ratio)
    counts = torch.bincount(qy_s, minlength=tot)         # 每个中心候选数
    ptr = torch.cat([
        torch.zeros(1, device=env.device, dtype=torch.long),
        counts.cumsum(0)
    ], dim=0)                                           # 前缀和

    # 8. 分支式“不放回优先，候选不足才放回”采样
    K = ball_query_k - 1  # 除中心点外还需多少邻居
    # 构造 weight 矩阵 (tot, max_count)
    max_count = int(counts.max().item())
    idx_range = torch.arange(max_count, device=env.device).unsqueeze(0)
    counts_expand = counts.unsqueeze(1).expand(-1, max_count)
    weight_mat = (idx_range < counts_expand).to(torch.float32)

    many_mask = counts >= K
    few_mask = ~many_mask

    # offsets 用于存放每行采样结果
    offsets = torch.empty((tot, K), dtype=torch.long, device=env.device)

    # 不放回采样
    if many_mask.any():
        many_idx = many_mask.nonzero(as_tuple=False).squeeze(1)
        w_many = weight_mat[many_idx]
        off_many = torch.multinomial(w_many, num_samples=K, replacement=False)
        offsets[many_idx] = off_many

    # 放回采样
    if few_mask.any():
        few_idx = few_mask.nonzero(as_tuple=False).squeeze(1)
        w_few = weight_mat[few_idx]
        off_few = torch.multinomial(w_few, num_samples=K, replacement=True)
        offsets[few_idx] = off_few

    # 9. 全局索引 & gather
    sample_idx = ptr[:-1, None] + offsets   # (tot, K)
    sampled_px = px_s[sample_idx]           # (tot, K)

    # 10. 拼回中心点 & reshape
    grouped = x_flat[sampled_px] \
        .reshape(env_num, int(voxel_max_num * fps_ratio), K, 3)
    # 把中心点放在第一位
    centers = filt_pts_yaw_fps.reshape(env_num, int(voxel_max_num * fps_ratio), 1, 3)
    grouped = torch.cat([centers, grouped], dim=2).reshape(env_num, -1, 3)

    # 11. 转回世界 & 机器人局部坐标系
    pts_world = transform_points(
        grouped, pos=None, quat=yaw_quat(origin_quat))
    pts_local = transform_points(
        pts_world, pos=None, quat=quat_inv(origin_quat))

    return pts_local


def lidar_grid_obs(env: ManagerBasedEnv,
                   obb_half_size: tuple[float, float, float],
                   sphere_radius: float,
                   downsample_num: int,
                   random_rpy_range: tuple[float, float, float],
                   random_xyz_range: tuple[float, float, float],
                   asset_cfg: SceneEntityCfg,
                   sensor_cfg: dict[str, SceneEntityCfg]) -> torch.Tensor:
    lidar_front = env.scene.sensors[sensor_cfg["front_lidar"].name]
    lidar_back = env.scene.sensors[sensor_cfg["back_lidar"].name]

    # —— 初始化随机扰动，只做一次 —— #
    if not hasattr(env, "lidar_random_sample_buf"):
        num_envs = env.scene.num_envs
        device = getattr(env, "device", torch.device("cpu"))
        rpy_max = torch.tensor(random_rpy_range, device=device).view(1, 3)
        xyz_max = torch.tensor(random_xyz_range, device=device).view(1, 3)

        def rand_buf(max_val):
            return (torch.rand(num_envs, 3, device=device) * 2 - 1) * max_val

        random_rpy_front = rand_buf(rpy_max)
        random_xyz_front = rand_buf(xyz_max)
        random_rpy_back = rand_buf(rpy_max)
        random_xyz_back = rand_buf(xyz_max)

        # 生成四元数
        roll, pitch, yaw = random_rpy_front.unbind(dim=1)
        random_quat_front = quat_from_euler_xyz(roll, pitch, yaw)
        roll, pitch, yaw = random_rpy_back.unbind(dim=1)
        random_quat_back = quat_from_euler_xyz(roll, pitch, yaw)

        buf = {
            "random_quat_front": random_quat_front,
            "random_xyz_front": random_xyz_front,
            "random_quat_back": random_quat_back,
            "random_xyz_back": random_xyz_back,
        }
        env.lidar_random_sample_buf = buf
    random_buf = env.lidar_random_sample_buf
    # 1. OBB 半尺寸
    half = torch.tensor(obb_half_size, device=env.device)

    # 2. 机器人位姿
    asset = env.scene[asset_cfg.name]
    origin_pos = asset.data.root_pos_w.to(env.device)
    origin_quat = asset.data.root_quat_w.to(env.device)

    # 3. 读取并合并前后 LiDAR 点云，转到 yaw 角坐标系
    front = transform_points(
        lidar_front.data.ray_hits_w[..., :3] - origin_pos.unsqueeze(1) + random_buf["random_xyz_front"].unsqueeze(1),
        pos=None,
        quat=quat_mul(random_buf["random_quat_front"], quat_inv(yaw_quat(origin_quat)))
    )
    back = transform_points(
        lidar_back.data.ray_hits_w[..., :3] - origin_pos.unsqueeze(1) + random_buf["random_xyz_back"].unsqueeze(1),
        pos=None,
        quat=quat_mul(random_buf["random_quat_back"], quat_inv(yaw_quat(origin_quat)))
    )
    ray_pos_yaw = torch.cat([front, back], dim=1)
    ray_pos_yaw.nan_to_num_(0.0, 0.0, 0.0)

    # 4. 体素下采样（OBB 内 & 球外）
    filt_pts_yaw = sample_in_obb_out_sphere(
        ray_pos_yaw, half, sphere_radius,
        downsample_num
    )

    return filt_pts_yaw


def make_grid_xyz_nodes(points, length, width, resolution, device, fill_value=0.0, xy_mask=True):
    """
    points: Tensor (N, M, 3)
    返回: Tensor (N, n_pts, 3)，其中 n_pts = round(L/Δ)+1 乘以 round(W/Δ)+1
          每个 (x,y) 是网格节点坐标，每个 z 是落在该节点周围
          (“以节点为中心、边长=resolution 的方形区域”) 的所有点的最大 z，
          如果该区域内没有点，则 (x,y)=(0,0)，z=fill_value。
    """
    N, M, _ = points.shape

    # 1. 计算节点数
    n_x = int(round(length / resolution)) + 1  # e.g. round(3.0/0.05)=60 → +1=61
    n_y = int(round(width / resolution)) + 1  # round(1.2/0.05)=24 → +1=25
    n_pts = n_x * n_y

    # 2. 生成节点坐标 (n_pts, 2)
    xs = torch.linspace(-length / 2, length / 2, steps=n_x, device=device)
    ys = torch.linspace(-width / 2, width / 2, steps=n_y, device=device)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    grid_nodes = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

    # 3. 点 → 最近节点索引
    x, y, z = points.unbind(-1)  # (N, M)
    i_x = ((x + length / 2 + resolution / 2) / resolution)\
        .floor().long().clamp(0, n_x - 1)  # (N, M)
    i_y = ((y + width / 2 + resolution / 2) / resolution)\
        .floor().long().clamp(0, n_y - 1)  # (N, M)
    idx = i_x * n_y + i_y                  # (N, M)

    # 4. scatter_reduce 求 max-z
    heights = torch.full((N, n_pts), float("-inf"), device=device)
    heights = heights.scatter_reduce(
        dim=1,
        index=idx,
        src=z,
        reduce="amax",
        include_self=True
    )  # (N, n_pts)

    # 5. 记住哪些是有效格子
    valid_mask = heights != float("-inf")  # (N, n_pts)

    # 6. 把无点用 fill_value 填充
    heights = torch.where(
        valid_mask,
        heights,
        torch.full_like(heights, fill_value)
    )  # (N, n_pts)

    # 7. 对 grid_nodes 中对应无效行的坐标也设为 0
    #    先把 grid_nodes 扩成 (N, n_pts, 2)
    nodes_exp = grid_nodes.unsqueeze(0).expand(N, -1, -1).clone()  # (N, n_pts, 2)
    #    再把无效位置置零
    if xy_mask:
        nodes_exp[~valid_mask] = 0.0

    # 8. 拼成 (N, n_pts, 3)
    z_exp = heights.unsqueeze(-1)            # (N, n_pts, 1)
    grid_xyz = torch.cat([nodes_exp, z_exp], dim=2)

    return grid_xyz


def zero_points_in_random_boxes(points: torch.Tensor,
                                Q: int,
                                rx: float, ry: float, rz: float,
                                qx: float, qy: float, qz: float,
                                fill_value: float = 0.0,
                                xy_mask: bool = True) -> torch.Tensor:
    """
    Args:
        points: Tensor of shape (N, M, 3)
        Q:      每帧随机生成的长方体数量
        rx,ry,rz: 盒子在 x,y,z 方向的最大半尺寸
        qx,qy,qz: roll, pitch, yaw 的最大角度范围（弧度）
    Returns:
        masked: Tensor of shape (N, M, 3)，落入任意盒子的点被置为 0
    """
    N, M, _ = points.shape
    device = points.device
    pts_origin = points.clone()

    # 1. 随机选中心点索引
    idx = torch.randint(0, M, (N, Q), device=device)      # (N, Q)
    centers = points[torch.arange(N, device=device).unsqueeze(1), idx]  # (N, Q, 3)

    # 2. 随机生成半尺寸
    half_sizes = torch.stack([
        torch.rand(N, Q, device=device) * rx,
        torch.rand(N, Q, device=device) * ry,
        torch.rand(N, Q, device=device) * rz,
    ], dim=-1)  # (N, Q, 3)

    # 3. 随机生成 r,p,y
    roll = (torch.rand(N, Q, device=device) * 2 - 1) * qx
    pitch = (torch.rand(N, Q, device=device) * 2 - 1) * qy
    yaw = (torch.rand(N, Q, device=device) * 2 - 1) * qz

    # 3.1 构造 ZYX 顺序的旋转矩阵 R (N,Q,3,3)
    def euler_to_R(r, p, y):
        cr, sr = torch.cos(r), torch.sin(r)
        cp, sp = torch.cos(p), torch.sin(p)
        cy, sy = torch.cos(y), torch.sin(y)
        R = torch.zeros(N, Q, 3, 3, device=device)
        R[..., 0, 0] = cy * cp
        R[..., 0, 1] = cy * sp * sr - sy * cr
        R[..., 0, 2] = cy * sp * cr + sy * sr
        R[..., 1, 0] = sy * cp
        R[..., 1, 1] = sy * sp * sr + cy * cr
        R[..., 1, 2] = sy * sp * cr - cy * sr
        R[..., 2, 0] = -sp
        R[..., 2, 1] = cp * sr
        R[..., 2, 2] = cp * cr
        return R

    R = euler_to_R(roll, pitch, yaw)  # (N, Q, 3, 3)

    # 4. 计算点到盒子中心的相对向量，并旋转到盒子局部坐标系
    pts = points.unsqueeze(1)   # (N,1,M,3)
    ctr = centers.unsqueeze(2)  # (N,Q,1,3)
    delta = pts - ctr             # (N,Q,M,3)

    # ←— 关键修改 —→
    # 直接用批量矩阵乘：对最后两个维度（3,3）与（3）做乘法
    delta_local = torch.matmul(delta, R.transpose(-1, -2))  # (N, Q, M, 3)

    # 5. 判断哪些点落在盒子内
    mask_per_box = (delta_local.abs() <= half_sizes.unsqueeze(2)).all(dim=-1)  # (N,Q,M)

    # 6. 合并盒子：只要落入任意一个，就置为 True
    inside_any = mask_per_box.any(dim=1)  # (N, M)

    # 7. 构造输出，把这些点置 0
    if xy_mask:
        pts_origin[inside_any] = fill_value
    else:
        pts_origin[inside_any][..., 2] = fill_value

    return pts_origin


def sample_points(
    full_obs: torch.Tensor,
    num_samples: int,
    threshold: float = 0.001
) -> torch.Tensor:
    """
    对输入 (B, N, C) 的点云 full_obs 做采样：
    1) 先把 NaN/Inf 全置 0；
    2) sq_sum < threshold 的点视为无效；
    3) 对每个 batch：
       • 如果有效点 ≥ num_samples：无放回随机选 num_samples 个；
       • 如果 0 < 有效点 < num_samples：先无放回选完所有有效点（随机顺序），再有放回补齐到 num_samples；
       • 如果有效点 = 0：最后输出全 0。
    返回形状 (B, num_samples, C)。
    """
    B, N, C = full_obs.shape      # N 是原始点的个数
    device = full_obs.device

    # 1. NaN/Inf → 0
    full_obs = full_obs.nan_to_num(0.0, 0.0, 0.0)

    # 2. 有效点掩码
    valid_mask = full_obs.pow(2).sum(dim=-1) >= threshold  # (B, N)
    n_valid = valid_mask.sum(dim=1)                       # (B,)

    # 3. “有放回”采样索引
    weights = valid_mask.float()                          # (B, N)
    weights_adj = torch.where(
        n_valid.unsqueeze(1) > 0,
        weights,
        torch.ones_like(weights)
    )
    rep_idx = torch.multinomial(weights_adj, num_samples, replacement=True)  # (B, num_samples)

    # 4. “无放回”选全体有效点的随机顺序索引
    rand_vals = torch.rand((B, N), device=device)
    rand_vals = torch.where(valid_mask, rand_vals, torch.full_like(rand_vals, -1.0))
    sorted_idx = rand_vals.argsort(dim=1, descending=True)  # (B, N)

    # 5. 合成最终索引 out_idx
    col_idx = torch.arange(num_samples, device=device).unsqueeze(0)      # (1, num_samples)
    fill_mask = col_idx < n_valid.unsqueeze(1)                           # (B, num_samples)
    # 对前 n_valid 个位置用 sorted_idx（无放回），后面位置用 rep_idx（补齐）
    out_idx = torch.where(
        fill_mask,
        sorted_idx[:, :num_samples],
        rep_idx
    )  # (B, num_samples)

    # 6. Gather 出最终采样点
    sampled = full_obs.gather(
        dim=1,
        index=out_idx.unsqueeze(-1).expand(-1, -1, C)
    )  # (B, num_samples, C)

    # 7. 完全无有效点环境置零
    no_valid = (n_valid == 0)
    if no_valid.any():
        sampled[no_valid] = 0.0

    return sampled


def dynamic_lidar_obs(env: ManagerBasedEnv,
                      asset_cfg: SceneEntityCfg,
                      gt_sensor_cfg: SceneEntityCfg,
                      front_lidar_sensor_cfg: SceneEntityCfg,
                      back_lidar_sensor_cfg: SceneEntityCfg,
                      low_obs_num: int,
                      mid_obs_num: int,
                      high_obs_num: int,
                      random_rpy_range: tuple[float, float, float],
                      random_xyz_range: tuple[float, float, float],
                      point_gauss_noise_low: tuple[float, float, float],
                      point_gauss_noise_high: tuple[float, float, float],
                      dist_beta: float,
                      box_num: int,
                      box_size: tuple[float, float, float],
                      box_rpy: tuple[float, float, float]
                      ) -> torch.Tensor:
    if not hasattr(env, "dynamic_lidar_random_sample_buf"):
        num_envs = env.scene.num_envs
        device = getattr(env, "device", torch.device("cpu"))
        rpy_max = torch.tensor(random_rpy_range, device=device).view(1, 3)
        xyz_max = torch.tensor(random_xyz_range, device=device).view(1, 3)

        def rand_buf(max_val):
            return (torch.rand(num_envs, 3, device=device) * 2 - 1) * max_val

        random_rpy_front = rand_buf(rpy_max)
        random_xyz_front = rand_buf(xyz_max)
        random_rpy_back = rand_buf(rpy_max)
        random_xyz_back = rand_buf(xyz_max)

        # 生成四元数
        roll, pitch, yaw = random_rpy_front.unbind(dim=1)
        random_quat_front = quat_from_euler_xyz(roll, pitch, yaw)
        roll, pitch, yaw = random_rpy_back.unbind(dim=1)
        random_quat_back = quat_from_euler_xyz(roll, pitch, yaw)

        buf = {
            "random_quat_front": random_quat_front,
            "random_xyz_front": random_xyz_front,
            "random_quat_back": random_quat_back,
            "random_xyz_back": random_xyz_back,
        }
        env.dynamic_lidar_random_sample_buf = buf
    random_buf = env.dynamic_lidar_random_sample_buf
    # print("random_buf", random_buf)

    asset: RigidObject = env.scene[asset_cfg.name]
    gt_sensor: RayCaster = env.scene.sensors[gt_sensor_cfg.name]
    front_lidar_sensor: RayCaster = env.scene.sensors[front_lidar_sensor_cfg.name]
    back_lidar_sensor: RayCaster = env.scene.sensors[back_lidar_sensor_cfg.name]
    root_pos_w = asset.data.root_pos_w.to(env.device)
    root_quat_w = asset.data.root_quat_w.to(env.device)
    # print("gt_sensor.data.ray_hits_w", gt_sensor.data.ray_hits_w.shape)
    gt_sensor_data = gt_sensor.data.ray_hits_w
    gt_front_pos_local = transform_points(gt_sensor_data[:, -525:, :] - root_pos_w.unsqueeze(1), pos=None, quat=quat_inv(root_quat_w))
    gt_back_pos_local = transform_points(gt_sensor_data[:, :525, :] - root_pos_w.unsqueeze(1), pos=None, quat=quat_inv(root_quat_w))
    front_ray_starts_local = torch.tensor(front_lidar_sensor.cfg.offset.pos, device=env.device).view(1, 1, 3)
    front_raw_dirs = gt_front_pos_local - front_ray_starts_local
    front_ray_directions = torch.nn.functional.normalize(front_raw_dirs, dim=-1)
    front_lidar_sensor.ray_directions = front_ray_directions
    back_ray_starts_local = torch.tensor(back_lidar_sensor.cfg.offset.pos, device=env.device).view(1, 1, 3)
    back_raw_dirs = gt_back_pos_local - back_ray_starts_local
    back_ray_directions = torch.nn.functional.normalize(back_raw_dirs, dim=-1)
    back_lidar_sensor.ray_directions = back_ray_directions

    front_yaw = transform_points(
        front_lidar_sensor.data.ray_hits_w[..., :3] - root_pos_w.unsqueeze(1) + random_buf["random_xyz_front"].unsqueeze(1),
        pos=None,
        quat=quat_mul(random_buf["random_quat_front"], quat_inv(yaw_quat(root_quat_w)))
    )
    back_yaw = transform_points(
        back_lidar_sensor.data.ray_hits_w[..., :3] - root_pos_w.unsqueeze(1) + random_buf["random_xyz_back"].unsqueeze(1),
        pos=None,
        quat=quat_mul(random_buf["random_quat_back"], quat_inv(yaw_quat(root_quat_w)))
    )

    # front_yaw = front_yaw + front_yaw_noise
    # back_yaw = back_yaw + back_yaw_noise

    data = torch.cat([front_yaw, back_yaw], dim=1)
    grid_xyz = make_grid_xyz_nodes(data, length=3.0, width=1.2, resolution=0.05, device=env.device)
    front_grid_xyz = grid_xyz[:, -525:, :]
    back_grid_xyz = grid_xyz[:, :525, :]

    front_grid_xyz = zero_points_in_random_boxes(front_grid_xyz, box_num, box_size[0], box_size[1], box_size[2], box_rpy[0], box_rpy[1], box_rpy[2])
    back_grid_xyz = zero_points_in_random_boxes(back_grid_xyz, box_num, box_size[0], box_size[1], box_size[2], box_rpy[0], box_rpy[1], box_rpy[2])

    front_low_pts = front_grid_xyz[:, :175, :]
    front_mid_pts = front_grid_xyz[:, 175:305, :]
    front_high_pts = front_grid_xyz[:, 305:, :]

    back_low_pts = back_grid_xyz[:, 305:, :]
    back_mid_pts = back_grid_xyz[:, 175:305, :]
    back_high_pts = back_grid_xyz[:, :175, :]

    front_sampled_low_pts = sample_points(front_low_pts, low_obs_num // 2)
    front_sampled_mid_pts = sample_points(front_mid_pts, mid_obs_num // 2)
    front_sampled_high_pts = sample_points(front_high_pts, high_obs_num // 2)

    back_sampled_low_pts = sample_points(back_low_pts, low_obs_num // 2)
    back_sampled_mid_pts = sample_points(back_mid_pts, mid_obs_num // 2)
    back_sampled_high_pts = sample_points(back_high_pts, high_obs_num // 2)

    sampled_pts = torch.cat([front_sampled_low_pts, front_sampled_mid_pts, front_sampled_high_pts, back_sampled_low_pts, back_sampled_mid_pts, back_sampled_high_pts], dim=1)

    dist = torch.norm(sampled_pts, dim=-1)
    noise_low = torch.tensor(point_gauss_noise_low, device=env.device).view(1, 1, 3)  # (1,1,3)
    noise_high = torch.tensor(point_gauss_noise_high, device=env.device).view(1, 1, 3)  # (1,1,3)

    weight = torch.exp(-dist_beta * dist).unsqueeze(-1)       # (N,1)
    noise = noise_low + (noise_high - noise_low) * weight    # 自动广播，得到 (N,3)

    sampled_pts_noise = torch.randn_like(sampled_pts) * noise

    final_obs = sampled_pts + sampled_pts_noise
    final_obs[..., 2] = -final_obs[..., 2] - 0.5
    final_obs[..., 2] = final_obs[..., 2].clamp(-2.0, 2.0)
    final_obs[..., :3] = final_obs[..., :3] * 5.0

    # print("final_obs", final_obs.shape)
    B, N, C = final_obs.shape
    perm = torch.argsort(torch.rand(B, N, device=final_obs.device), dim=1)     # (B, N)
    final_obs_shuffled = final_obs.gather(1, perm.unsqueeze(-1).expand(-1, -1, C))
    return final_obs_shuffled


def filter_points_by_obb(
    points: torch.Tensor,                 # (N, M, 3)
    center: torch.Tensor,                 # (N, 3) 或 (3,)
    R_local_to_world: torch.Tensor,       # (N, 3, 3) 或 (3, 3)
    half_sizes: torch.Tensor,             # (N, 3) 或 (3,)
    remove: str = "outside",              # "inside" | "outside"
    fill_value: float | torch.Tensor = 0.0,
    return_mask: bool = False,
    eps: float = 1e-6,
):
    """
    基于 OBB 过滤点：被剔除的点用 fill_value 填充。

    - remove="outside": 剔除盒外（保留盒内）
    - remove="inside":  剔除盒内（保留盒外）
    - fill_value: 可以是标量，或形状可广播到 (N,M,3) 的张量，例如 (3,), (N,3), (N,M,3)
    """
    if points.ndim != 3 or points.size(-1) != 3:
        raise ValueError("points 需为 (N, M, 3)")

    N, M, _ = points.shape
    device, dtype = points.device, points.dtype

    center = center.to(device=device, dtype=dtype)
    half_sizes = half_sizes.to(device=device, dtype=dtype)
    R = R_local_to_world.to(device=device, dtype=dtype)

    # 广播到批维 N
    if center.ndim == 1:      # (3,) -> (N,3)
        center = center.expand(N, -1)
    if half_sizes.ndim == 1:  # (3,) -> (N,3)
        half_sizes = half_sizes.expand(N, -1)
    if R.ndim == 2:           # (3,3) -> (N,3,3)
        R = R.expand(N, -1, -1)

    # 世界 -> 局部
    p_local = torch.matmul(points - center[:, None, :], R.transpose(1, 2))  # (N,M,3)

    # 盒内判定
    inside = (p_local.abs() <= (half_sizes[:, None, :] + eps)).all(dim=-1)  # (N,M)

    if remove == "inside":
        keep = ~inside
    elif remove == "outside":
        keep = inside
    else:
        raise ValueError("remove 只能是 'inside' 或 'outside'")

    # 准备 fill 张量（支持任意可广播形状）
    if torch.is_tensor(fill_value):
        fill = fill_value.to(device=device, dtype=dtype)
    else:
        fill = torch.as_tensor(fill_value, device=device, dtype=dtype)
    # 通过与 zeros_like 相加完成广播到 (N,M,3)
    fill = torch.zeros_like(points) + fill

    # 用 where 选择：保留的点用原值，剔除的点用 fill
    out = torch.where(keep[..., None], points, fill)

    return (out, keep) if return_mask else out


def partial_heightmap_obs(env: ManagerBasedEnv,
                          asset_cfg: SceneEntityCfg,
                          gt_sensor_cfg: SceneEntityCfg,
                          front_lidar_sensor_cfg: SceneEntityCfg,
                          back_lidar_sensor_cfg: SceneEntityCfg,
                          random_rpy_range: tuple[float, float, float],
                          random_xyz_range: tuple[float, float, float],
                          box_num: int,
                          box_size: tuple[float, float, float],
                          box_rpy: tuple[float, float, float],
                          z_noise: float = 0.0
                          ) -> torch.Tensor:
    if not hasattr(env, "dynamic_lidar_random_sample_buf"):
        num_envs = env.scene.num_envs
        device = getattr(env, "device", torch.device("cpu"))
        rpy_max = torch.tensor(random_rpy_range, device=device).view(1, 3)
        xyz_max = torch.tensor(random_xyz_range, device=device).view(1, 3)

        def rand_buf(max_val):
            return (torch.rand(num_envs, 3, device=device) * 2 - 1) * max_val

        random_rpy_front = rand_buf(rpy_max)
        random_xyz_front = rand_buf(xyz_max)
        random_rpy_back = rand_buf(rpy_max)
        random_xyz_back = rand_buf(xyz_max)

        # 生成四元数
        roll, pitch, yaw = random_rpy_front.unbind(dim=1)
        random_quat_front = quat_from_euler_xyz(roll, pitch, yaw)
        roll, pitch, yaw = random_rpy_back.unbind(dim=1)
        random_quat_back = quat_from_euler_xyz(roll, pitch, yaw)

        buf = {
            "random_quat_front": random_quat_front,
            "random_xyz_front": random_xyz_front,
            "random_quat_back": random_quat_back,
            "random_xyz_back": random_xyz_back,
        }
        env.dynamic_lidar_random_sample_buf = buf
    random_buf = env.dynamic_lidar_random_sample_buf
    # print("random_buf", random_buf)

    asset: RigidObject = env.scene[asset_cfg.name]
    gt_sensor: RayCaster = env.scene.sensors[gt_sensor_cfg.name]
    front_lidar_sensor: RayCaster = env.scene.sensors[front_lidar_sensor_cfg.name]
    back_lidar_sensor: RayCaster = env.scene.sensors[back_lidar_sensor_cfg.name]
    root_pos_w = asset.data.root_pos_w.to(env.device)
    root_quat_w = asset.data.root_quat_w.to(env.device)
    # print("gt_sensor.data.ray_hits_w", gt_sensor.data.ray_hits_w.shape)
    gt_sensor_data = gt_sensor.data.ray_hits_w
    gt_pos_local = transform_points(gt_sensor_data - root_pos_w.unsqueeze(1), pos=None, quat=quat_inv(root_quat_w))
    front_ray_starts_local = torch.tensor(front_lidar_sensor.cfg.offset.pos, device=env.device).view(1, 1, 3)
    front_raw_dirs = gt_pos_local - front_ray_starts_local
    front_ray_directions = torch.nn.functional.normalize(front_raw_dirs, dim=-1)
    front_lidar_sensor.ray_directions = front_ray_directions
    back_ray_starts_local = torch.tensor(back_lidar_sensor.cfg.offset.pos, device=env.device).view(1, 1, 3)
    back_raw_dirs = gt_pos_local - back_ray_starts_local
    back_ray_directions = torch.nn.functional.normalize(back_raw_dirs, dim=-1)
    back_lidar_sensor.ray_directions = back_ray_directions

    local_front_ray_hits_w = transform_points(front_lidar_sensor.data.ray_hits_w - root_pos_w.unsqueeze(1), pos=None, quat=quat_inv(root_quat_w))
    local_back_ray_hits_w = transform_points(back_lidar_sensor.data.ray_hits_w - root_pos_w.unsqueeze(1), pos=None, quat=quat_inv(root_quat_w))

    center_front = torch.tensor([-1.55, 0.0, 0.0])
    center_back = torch.tensor([1.55, 0.0, 0.0])
    Rot = torch.eye(3)  # 轴对齐盒：无旋转
    half = torch.tensor([2.0, 5.0, 5.0], device=env.device)
    local_front_ray_hits_w = filter_points_by_obb(local_front_ray_hits_w, center_front, Rot, half, remove="inside", return_mask=False)
    local_back_ray_hits_w = filter_points_by_obb(local_back_ray_hits_w, center_back, Rot, half, remove="inside", return_mask=False)

    world_front_ray_hits_w = transform_points(local_front_ray_hits_w, pos=None, quat=root_quat_w)
    world_back_ray_hits_w = transform_points(local_back_ray_hits_w, pos=None, quat=root_quat_w)

    front_yaw = transform_points(
        world_front_ray_hits_w + random_buf["random_xyz_front"].unsqueeze(1),
        pos=None,
        quat=quat_mul(random_buf["random_quat_front"], quat_inv(yaw_quat(root_quat_w)))
    )
    back_yaw = transform_points(
        world_back_ray_hits_w + random_buf["random_xyz_back"].unsqueeze(1),
        pos=None,
        quat=quat_mul(random_buf["random_quat_back"], quat_inv(yaw_quat(root_quat_w)))
    )

    center = torch.tensor([0.0, 0.0, 0.0], device=env.device)
    half = torch.tensor([0.1, 0.1, 0.1], device=env.device)
    data = torch.cat([front_yaw, back_yaw], dim=1)
    data = filter_points_by_obb(data, center, Rot, half, remove="inside", fill_value=-100, return_mask=False)

    grid_xyz = make_grid_xyz_nodes(data, length=3.0, width=1.2, resolution=0.05, device=env.device, fill_value=-100, xy_mask=False)
    grid_xyz = zero_points_in_random_boxes(grid_xyz, box_num, box_size[0], box_size[1], box_size[2], box_rpy[0], box_rpy[1], box_rpy[2], fill_value=-100, xy_mask=True)

    mask = grid_xyz[..., 2] > -50

    gaussian_noise = torch.randn_like(grid_xyz[..., 2]) * z_noise
    grid = - grid_xyz[..., 2] - 0.5 + gaussian_noise
    grid = grid.clamp(-2.0, 2.0)
    grid = grid * 5.0
    grid[~mask] = -100.0
    # print("grid", grid.shape)
    return grid


def feet_contact(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 0.1
    # print("contacts:", contacts)
    return contacts.float()

"""机械臂/操作相关奖励项（可迁移模块）。

说明：
- 该文件与 locomotion 通用奖励分离，便于迁移到其它任务/机器人。
- `mdp.track_ee_target_points_exp` 仍保持可用：由 `common/mdp/__init__.py` 统一导出。
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

from rl_sim_env.tasks.manager_based.common.utils.projected_com_frame import (
    compute_projected_com_yaw_frame,
    compute_trunk_com_yaw_frame,
    world_to_projected_frame,
)

from .force_compliance import (
    get_force_compliance_delta_p,
    scale_delta_p_to_box,
)


def _get_command_term(env, command_name: str):
    """Best-effort: fetch the command term object (not just the tensor) from
    CommandManager.

    We use this to read command cfg flags (e.g. `use_projected_origin`) so that
    rewards compute errors in the *same* reference frame as the command is
    defined.
    """
    cmd_mgr = getattr(env, "command_manager", None)
    if cmd_mgr is None:
        return None

    # Preferred public API (exists in this codebase; see
    # common/mdp/curriculums.py)
    if hasattr(cmd_mgr, "get_term"):
        try:
            return cmd_mgr.get_term(command_name)
        except Exception:
            pass

    # Fallback to common internal containers (dict-like)
    for attr in ("_terms", "_command_terms", "terms"):
        container = getattr(cmd_mgr, attr, None)
        if container is None:
            continue
        try:
            candidate = container[command_name]
        except Exception:
            candidate = None
        if candidate is not None:
            return candidate

    return None


def _resolve_use_projected_origin(
    env, command_name: str, default: bool = True
) -> bool:
    """Resolve whether a command is defined in projected-COM origin or
    trunk-COM origin.
    """
    cmd_term = _get_command_term(env, command_name)
    if cmd_term is None:
        return bool(default)

    # Prefer cfg if present (config is the source of truth)
    cfg = getattr(cmd_term, "cfg", None)
    if cfg is not None and hasattr(cfg, "use_projected_origin"):
        try:
            return bool(getattr(cfg, "use_projected_origin"))
        except Exception:
            pass

    # Fallback: some terms expose the flag directly
    if hasattr(cmd_term, "use_projected_origin"):
        try:
            return bool(getattr(cmd_term, "use_projected_origin"))
        except Exception:
            pass

    return bool(default)


def _get_ee_target_reference_frame(
    env,
    command_name: str,
    robot: Articulation,
    trunk_body_id: int,
    terrain_sensor_cfg: SceneEntityCfg,
):
    """Get the yaw frame used by ee_target_points rewards, aligned with the
    command definition.
    """
    use_projected_origin = _resolve_use_projected_origin(
        env, command_name, default=True
    )
    if use_projected_origin:
        terrain_sensor: RayCaster = env.scene.sensors[terrain_sensor_cfg.name]
        return compute_projected_com_yaw_frame(
            asset=robot,
            trunk_body_id=trunk_body_id,
            terrain_height_sensor=terrain_sensor,
            lpf_state_key=command_name,
            read_cached_ground_z_only=True,
        )
    else:
        return compute_trunk_com_yaw_frame(
            asset=robot,
            trunk_body_id=trunk_body_id,
        )


def track_ee_target_points_exp(
    env,
    command_name: str,
    ee_asset_cfg: SceneEntityCfg,
    trunk_asset_cfg: SceneEntityCfg,
    terrain_sensor_cfg: SceneEntityCfg,
    offset_distance: float,
    # 向后兼容：旧配置只提供 std（主目标点与偏移点共用）
    std: float = 0.2,
    # 新配置：分别指定主目标点与偏移目标点的核宽度（更可控）
    main_std: float | None = None,
    offset_std: float | None = None,
    main_weight: float = 1.0,
    offset_weight: float = 0.5,
    command_active_threshold: float = 1.0e-6,
) -> torch.Tensor:
    """末端跟随目标点（9D: main/x_offset/y_offset）的指数核奖励。

    参考系：
    - 原点：躯干质心在地面上的投影点
    - 朝向：仅 yaw 对齐（忽略 pitch/roll）

    命令 `command_name` 输出 9D，表达在该参考系下：
    - [0:3] main target
    - [3:6] x-offset target
    - [6:9] y-offset target
    """
    robot: Articulation = env.scene[ee_asset_cfg.name]

    if len(ee_asset_cfg.body_ids) != 1:
        raise ValueError(
            "ee_asset_cfg.body_names 需包含 1 个末端刚体，"
            f"实际为 {len(ee_asset_cfg.body_ids)}"
        )
    if len(trunk_asset_cfg.body_ids) != 1:
        raise ValueError(
            "trunk_asset_cfg.body_names 需包含 1 个躯干刚体，"
            f"实际为 {len(trunk_asset_cfg.body_ids)}"
        )

    ee_body_id = ee_asset_cfg.body_ids[0]
    trunk_body_id = trunk_asset_cfg.body_ids[0]

    frame = _get_ee_target_reference_frame(
        env, command_name, robot, trunk_body_id, terrain_sensor_cfg
    )

    cmd = env.command_manager.get_command(command_name)  # (B, 9)
    cmd_main = cmd[:, 0:3]
    cmd_x = cmd[:, 3:6]
    cmd_y = cmd[:, 6:9]

    cmd_active = torch.linalg.norm(cmd, dim=1) > command_active_threshold

    ee_pos_w = robot.data.body_pos_w[:, ee_body_id]
    ee_quat_w = robot.data.body_quat_w[:, ee_body_id]

    ee_pos_p = world_to_projected_frame(ee_pos_w, frame)

    yaw_q_inv = quat_inv(frame.yaw_quat_w)
    ee_quat_p = quat_mul(yaw_q_inv, ee_quat_w)

    local_x = torch.zeros((env.num_envs, 3), device=env.device)
    local_y = torch.zeros((env.num_envs, 3), device=env.device)
    local_x[:, 0] = offset_distance
    local_y[:, 1] = offset_distance

    ee_x_p = ee_pos_p + quat_apply(ee_quat_p, local_x)
    ee_y_p = ee_pos_p + quat_apply(ee_quat_p, local_y)

    main_err = torch.sum(torch.square(ee_pos_p - cmd_main), dim=1)
    x_err = torch.sum(torch.square(ee_x_p - cmd_x), dim=1)
    y_err = torch.sum(torch.square(ee_y_p - cmd_y), dim=1)

    # 分开 std：主目标点与偏移点分别做指数核，再在指数里加权相加
    # 这样可以独立调节“主点更严格/偏移点更宽松”等行为。
    main_sigma = float(std if main_std is None else main_std)
    offset_sigma = float(std if offset_std is None else offset_std)
    main_sigma2 = max(main_sigma * main_sigma, 1.0e-12)
    offset_sigma2 = max(offset_sigma * offset_sigma, 1.0e-12)

    weighted_exp = (main_weight * main_err) / main_sigma2 + (
        offset_weight * (x_err + y_err)
    ) / offset_sigma2
    reward = torch.exp(-weighted_exp)

    return torch.where(cmd_active, reward, torch.zeros_like(reward))


def track_ee_target_main_exp(
    env,
    command_name: str,
    ee_asset_cfg: SceneEntityCfg,
    trunk_asset_cfg: SceneEntityCfg,
    terrain_sensor_cfg: SceneEntityCfg,
    std: float = 0.2,
    command_active_threshold: float = 1.0e-6,
) -> torch.Tensor:
    """末端 main 点跟随的指数核奖励（仅位置）。

    参考系与命令一致：Projected COM yaw frame。
    - cmd_main = cmd[:, 0:3]
    - reward = exp(-||ee_pos_p - cmd_main||^2 / std^2)
    """
    robot: Articulation = env.scene[ee_asset_cfg.name]

    if len(ee_asset_cfg.body_ids) != 1:
        raise ValueError(
            "ee_asset_cfg.body_names 需包含 1 个末端刚体，"
            f"实际为 {len(ee_asset_cfg.body_ids)}"
        )
    if len(trunk_asset_cfg.body_ids) != 1:
        raise ValueError(
            "trunk_asset_cfg.body_names 需包含 1 个躯干刚体，"
            f"实际为 {len(trunk_asset_cfg.body_ids)}"
        )

    ee_body_id = ee_asset_cfg.body_ids[0]
    trunk_body_id = trunk_asset_cfg.body_ids[0]
    frame = _get_ee_target_reference_frame(
        env, command_name, robot, trunk_body_id, terrain_sensor_cfg
    )

    cmd = env.command_manager.get_command(command_name)  # (B, 9)
    cmd_active = torch.linalg.norm(cmd, dim=1) > command_active_threshold
    cmd_main = cmd[:, 0:3]

    ee_pos_w = robot.data.body_pos_w[:, ee_body_id]
    ee_pos_p = world_to_projected_frame(ee_pos_w, frame)

    sigma2 = max(float(std) * float(std), 1.0e-12)
    err = torch.sum(torch.square(ee_pos_p - cmd_main), dim=1)
    reward = torch.exp(-err / sigma2)
    return torch.where(cmd_active, reward, torch.zeros_like(reward))


def track_ee_target_x_offset_exp(
    env,
    command_name: str,
    ee_asset_cfg: SceneEntityCfg,
    trunk_asset_cfg: SceneEntityCfg,
    terrain_sensor_cfg: SceneEntityCfg,
    offset_distance: float,
    std: float = 0.2,
    command_active_threshold: float = 1.0e-6,
) -> torch.Tensor:
    """末端 x-offset 点跟随的指数核奖励（main+x 方向点）。

    - cmd_x = cmd[:, 3:6]
    - ee_x_p = ee_pos_p + quat_apply(ee_quat_p, [offset_distance, 0, 0])
    """
    robot: Articulation = env.scene[ee_asset_cfg.name]

    if len(ee_asset_cfg.body_ids) != 1:
        raise ValueError(
            "ee_asset_cfg.body_names 需包含 1 个末端刚体，"
            f"实际为 {len(ee_asset_cfg.body_ids)}"
        )
    if len(trunk_asset_cfg.body_ids) != 1:
        raise ValueError(
            "trunk_asset_cfg.body_names 需包含 1 个躯干刚体，"
            f"实际为 {len(trunk_asset_cfg.body_ids)}"
        )

    ee_body_id = ee_asset_cfg.body_ids[0]
    trunk_body_id = trunk_asset_cfg.body_ids[0]
    frame = _get_ee_target_reference_frame(
        env, command_name, robot, trunk_body_id, terrain_sensor_cfg
    )

    cmd = env.command_manager.get_command(command_name)  # (B, 9)
    cmd_active = torch.linalg.norm(cmd, dim=1) > command_active_threshold
    cmd_x = cmd[:, 3:6]

    ee_pos_w = robot.data.body_pos_w[:, ee_body_id]
    ee_quat_w = robot.data.body_quat_w[:, ee_body_id]
    ee_pos_p = world_to_projected_frame(ee_pos_w, frame)

    yaw_q_inv = quat_inv(frame.yaw_quat_w)
    ee_quat_p = quat_mul(yaw_q_inv, ee_quat_w)

    local_x = torch.zeros((env.num_envs, 3), device=env.device)
    local_x[:, 0] = float(offset_distance)
    ee_x_p = ee_pos_p + quat_apply(ee_quat_p, local_x)

    sigma2 = max(float(std) * float(std), 1.0e-12)
    err = torch.sum(torch.square(ee_x_p - cmd_x), dim=1)
    reward = torch.exp(-err / sigma2)
    return torch.where(cmd_active, reward, torch.zeros_like(reward))


def track_ee_target_y_offset_exp(
    env,
    command_name: str,
    ee_asset_cfg: SceneEntityCfg,
    trunk_asset_cfg: SceneEntityCfg,
    terrain_sensor_cfg: SceneEntityCfg,
    offset_distance: float,
    std: float = 0.2,
    command_active_threshold: float = 1.0e-6,
) -> torch.Tensor:
    """末端 y-offset 点跟随的指数核奖励（main+y 方向点）。"""
    robot: Articulation = env.scene[ee_asset_cfg.name]

    if len(ee_asset_cfg.body_ids) != 1:
        raise ValueError(
            "ee_asset_cfg.body_names 需包含 1 个末端刚体，"
            f"实际为 {len(ee_asset_cfg.body_ids)}"
        )
    if len(trunk_asset_cfg.body_ids) != 1:
        raise ValueError(
            "trunk_asset_cfg.body_names 需包含 1 个躯干刚体，"
            f"实际为 {len(trunk_asset_cfg.body_ids)}"
        )

    ee_body_id = ee_asset_cfg.body_ids[0]
    trunk_body_id = trunk_asset_cfg.body_ids[0]
    frame = _get_ee_target_reference_frame(
        env, command_name, robot, trunk_body_id, terrain_sensor_cfg
    )

    cmd = env.command_manager.get_command(command_name)  # (B, 9)
    cmd_active = torch.linalg.norm(cmd, dim=1) > command_active_threshold
    cmd_y = cmd[:, 6:9]

    ee_pos_w = robot.data.body_pos_w[:, ee_body_id]
    ee_quat_w = robot.data.body_quat_w[:, ee_body_id]
    ee_pos_p = world_to_projected_frame(ee_pos_w, frame)

    yaw_q_inv = quat_inv(frame.yaw_quat_w)
    ee_quat_p = quat_mul(yaw_q_inv, ee_quat_w)

    local_y = torch.zeros((env.num_envs, 3), device=env.device)
    local_y[:, 1] = float(offset_distance)
    ee_y_p = ee_pos_p + quat_apply(ee_quat_p, local_y)

    sigma2 = max(float(std) * float(std), 1.0e-12)
    err = torch.sum(torch.square(ee_y_p - cmd_y), dim=1)
    reward = torch.exp(-err / sigma2)
    return torch.where(cmd_active, reward, torch.zeros_like(reward))


def track_ee_target_main_exp_force_compliance(
    env,
    command_name: str,
    ee_asset_cfg: SceneEntityCfg,
    trunk_asset_cfg: SceneEntityCfg,
    terrain_sensor_cfg: SceneEntityCfg,
    std: float = 0.2,
    command_active_threshold: float = 1.0e-6,
) -> torch.Tensor:
    """末端 main 点跟随奖励（仅位置），并对目标点做“外力顺应偏移”。

    - 参考系：Projected COM yaw frame（与命令一致）
    - 偏移：cmd_main' = cmd_main + delta_p(force)
    """
    robot: Articulation = env.scene[ee_asset_cfg.name]

    if len(ee_asset_cfg.body_ids) != 1:
        raise ValueError(
            "ee_asset_cfg.body_names 需包含 1 个末端刚体，"
            f"实际为 {len(ee_asset_cfg.body_ids)}"
        )
    if len(trunk_asset_cfg.body_ids) != 1:
        raise ValueError(
            "trunk_asset_cfg.body_names 需包含 1 个躯干刚体，"
            f"实际为 {len(trunk_asset_cfg.body_ids)}"
        )

    ee_body_id = ee_asset_cfg.body_ids[0]
    trunk_body_id = trunk_asset_cfg.body_ids[0]

    frame = _get_ee_target_reference_frame(
        env, command_name, robot, trunk_body_id, terrain_sensor_cfg
    )

    cmd = env.command_manager.get_command(command_name)  # (B, 9)
    cmd_active = torch.linalg.norm(cmd, dim=1) > command_active_threshold
    cmd_main = cmd[:, 0:3]

    # 外力偏移管理器参数：统一从任务 config_summary.command 读取
    cfg_root = getattr(getattr(env, "cfg", None), "config_summary", None)
    cfg_cmd = (
        getattr(cfg_root, "command", None) if cfg_root is not None else None
    )
    if cfg_cmd is None:
        enable = True
        scale = (0.0, 0.0, 0.0)
        deadzone = 0.0
        f_clip = None
        d_clip = None
        clamp_ws = False
        r_max = None
        z_range = None
    else:
        enable = bool(getattr(cfg_cmd, "ee_force_compliance_enable", True))
        scale = getattr(cfg_cmd, "ee_force_compliance_scale", (0.0, 0.0, 0.0))
        deadzone = getattr(cfg_cmd, "ee_force_compliance_force_deadzone", 0.0)
        f_clip = getattr(cfg_cmd, "ee_force_compliance_force_clip", None)
        d_clip = getattr(cfg_cmd, "ee_force_compliance_delta_clip", None)
        clamp_ws = bool(
            getattr(
                cfg_cmd,
                "ee_force_compliance_workspace_clamp_enable",
                True,
            )
        )
        r_max = float(getattr(cfg_cmd, "ee_target_pos_r_range", (0.0, 0.0))[1])
        z_range = getattr(cfg_cmd, "ee_target_pos_z_range", (0.0, 0.0))
    if not enable:
        scale = (0.0, 0.0, 0.0)

    delta_p = get_force_compliance_delta_p(
        env,
        frame,
        force_to_pos_scale=scale,
        force_deadzone=deadzone,
        force_clip=f_clip,
        delta_clip=d_clip,
        cache_key="ee_target_points",
    )
    # 避免把目标推到明显不可达区域：将偏移按比例缩小，保证 main 点仍落在工作空间盒子内。
    #
    # 注意（重要）：
    # - 这里 clamp 的对象是 **main 点**（cmd_main），而不是 x/y offset 点。
    # - 因此不应再用 offset_distance 去额外收缩 r/z 的允许范围，否则会导致：
    #   1) 大量合法 command（例如 r 接近 r_max，或 z 覆盖完整 pos_z_range）下，delta_p 被缩放到 0；
    #   2) 外力顺应在训练中几乎“失效”，表现为你看到的效果不好/不随外力调整。
    # - 如果未来确实需要对 offset 点做保守保护，应在 command 采样范围层面收缩，
    #   或者为 offset 点单独实现约束，而不是在这里把 main 点的可行域缩到非常小。
    if clamp_ws and (r_max is not None) and (z_range is not None):
        r_lim = max(r_max, 0.0)
        z_min = float(z_range[0])
        z_max = float(z_range[1])
        if z_max > z_min + 1.0e-6:
            low = torch.tensor(
                [-r_lim, -r_lim, z_min],
                device=env.device,
                dtype=torch.float32,
            )
            high = torch.tensor(
                [r_lim, r_lim, z_max],
                device=env.device,
                dtype=torch.float32,
            )
            delta_p = scale_delta_p_to_box(cmd_main, delta_p, low, high)
    cmd_main = cmd_main + delta_p

    ee_pos_w = robot.data.body_pos_w[:, ee_body_id]
    ee_pos_p = world_to_projected_frame(ee_pos_w, frame)

    sigma2 = max(float(std) * float(std), 1.0e-12)
    err = torch.sum(torch.square(ee_pos_p - cmd_main), dim=1)
    reward = torch.exp(-err / sigma2)
    return torch.where(cmd_active, reward, torch.zeros_like(reward))


def track_ee_target_x_offset_exp_force_compliance(
    env,
    command_name: str,
    ee_asset_cfg: SceneEntityCfg,
    trunk_asset_cfg: SceneEntityCfg,
    terrain_sensor_cfg: SceneEntityCfg,
    offset_distance: float,
    std: float = 0.2,
    command_active_threshold: float = 1.0e-6,
) -> torch.Tensor:
    """末端 x-offset 点跟随奖励，并对目标点做“外力顺应偏移”。"""
    robot: Articulation = env.scene[ee_asset_cfg.name]

    if len(ee_asset_cfg.body_ids) != 1:
        raise ValueError(
            "ee_asset_cfg.body_names 需包含 1 个末端刚体，"
            f"实际为 {len(ee_asset_cfg.body_ids)}"
        )
    if len(trunk_asset_cfg.body_ids) != 1:
        raise ValueError(
            "trunk_asset_cfg.body_names 需包含 1 个躯干刚体，"
            f"实际为 {len(trunk_asset_cfg.body_ids)}"
        )

    ee_body_id = ee_asset_cfg.body_ids[0]
    trunk_body_id = trunk_asset_cfg.body_ids[0]

    frame = _get_ee_target_reference_frame(
        env, command_name, robot, trunk_body_id, terrain_sensor_cfg
    )

    cmd = env.command_manager.get_command(command_name)  # (B, 9)
    cmd_active = torch.linalg.norm(cmd, dim=1) > command_active_threshold
    cmd_main = cmd[:, 0:3]
    cmd_x = cmd[:, 3:6]

    cfg_root = getattr(getattr(env, "cfg", None), "config_summary", None)
    cfg_cmd = (
        getattr(cfg_root, "command", None) if cfg_root is not None else None
    )
    if cfg_cmd is None:
        enable = True
        scale = (0.0, 0.0, 0.0)
        deadzone = 0.0
        f_clip = None
        d_clip = None
        clamp_ws = False
        r_max = None
        z_range = None
    else:
        enable = bool(getattr(cfg_cmd, "ee_force_compliance_enable", True))
        scale = getattr(cfg_cmd, "ee_force_compliance_scale", (0.0, 0.0, 0.0))
        deadzone = getattr(cfg_cmd, "ee_force_compliance_force_deadzone", 0.0)
        f_clip = getattr(cfg_cmd, "ee_force_compliance_force_clip", None)
        d_clip = getattr(cfg_cmd, "ee_force_compliance_delta_clip", None)
        clamp_ws = bool(
            getattr(
                cfg_cmd,
                "ee_force_compliance_workspace_clamp_enable",
                True,
            )
        )
        r_max = float(getattr(cfg_cmd, "ee_target_pos_r_range", (0.0, 0.0))[1])
        z_range = getattr(cfg_cmd, "ee_target_pos_z_range", (0.0, 0.0))
    if not enable:
        scale = (0.0, 0.0, 0.0)

    delta_p = get_force_compliance_delta_p(
        env,
        frame,
        force_to_pos_scale=scale,
        force_deadzone=deadzone,
        force_clip=f_clip,
        delta_clip=d_clip,
        cache_key="ee_target_points",
    )
    if clamp_ws and (r_max is not None) and (z_range is not None):
        r_lim = max(r_max, 0.0)
        z_min = float(z_range[0])
        z_max = float(z_range[1])
        if z_max > z_min + 1.0e-6:
            low = torch.tensor(
                [-r_lim, -r_lim, z_min],
                device=env.device,
                dtype=torch.float32,
            )
            high = torch.tensor(
                [r_lim, r_lim, z_max],
                device=env.device,
                dtype=torch.float32,
            )
            delta_p = scale_delta_p_to_box(cmd_main, delta_p, low, high)
    cmd_x = cmd_x + delta_p

    ee_pos_w = robot.data.body_pos_w[:, ee_body_id]
    ee_quat_w = robot.data.body_quat_w[:, ee_body_id]
    ee_pos_p = world_to_projected_frame(ee_pos_w, frame)

    yaw_q_inv = quat_inv(frame.yaw_quat_w)
    ee_quat_p = quat_mul(yaw_q_inv, ee_quat_w)

    local_x = torch.zeros((env.num_envs, 3), device=env.device)
    local_x[:, 0] = float(offset_distance)
    ee_x_p = ee_pos_p + quat_apply(ee_quat_p, local_x)

    sigma2 = max(float(std) * float(std), 1.0e-12)
    err = torch.sum(torch.square(ee_x_p - cmd_x), dim=1)
    reward = torch.exp(-err / sigma2)
    return torch.where(cmd_active, reward, torch.zeros_like(reward))


def track_ee_target_y_offset_exp_force_compliance(
    env,
    command_name: str,
    ee_asset_cfg: SceneEntityCfg,
    trunk_asset_cfg: SceneEntityCfg,
    terrain_sensor_cfg: SceneEntityCfg,
    offset_distance: float,
    std: float = 0.2,
    command_active_threshold: float = 1.0e-6,
) -> torch.Tensor:
    """末端 y-offset 点跟随奖励，并对目标点做“外力顺应偏移”。"""
    robot: Articulation = env.scene[ee_asset_cfg.name]

    if len(ee_asset_cfg.body_ids) != 1:
        raise ValueError(
            "ee_asset_cfg.body_names 需包含 1 个末端刚体，"
            f"实际为 {len(ee_asset_cfg.body_ids)}"
        )
    if len(trunk_asset_cfg.body_ids) != 1:
        raise ValueError(
            "trunk_asset_cfg.body_names 需包含 1 个躯干刚体，"
            f"实际为 {len(trunk_asset_cfg.body_ids)}"
        )

    ee_body_id = ee_asset_cfg.body_ids[0]
    trunk_body_id = trunk_asset_cfg.body_ids[0]

    frame = _get_ee_target_reference_frame(
        env, command_name, robot, trunk_body_id, terrain_sensor_cfg
    )

    cmd = env.command_manager.get_command(command_name)  # (B, 9)
    cmd_active = torch.linalg.norm(cmd, dim=1) > command_active_threshold
    cmd_main = cmd[:, 0:3]
    cmd_y = cmd[:, 6:9]

    cfg_root = getattr(getattr(env, "cfg", None), "config_summary", None)
    cfg_cmd = (
        getattr(cfg_root, "command", None) if cfg_root is not None else None
    )
    if cfg_cmd is None:
        enable = True
        scale = (0.0, 0.0, 0.0)
        deadzone = 0.0
        f_clip = None
        d_clip = None
        clamp_ws = False
        r_max = None
        z_range = None
    else:
        enable = bool(getattr(cfg_cmd, "ee_force_compliance_enable", True))
        scale = getattr(cfg_cmd, "ee_force_compliance_scale", (0.0, 0.0, 0.0))
        deadzone = getattr(cfg_cmd, "ee_force_compliance_force_deadzone", 0.0)
        f_clip = getattr(cfg_cmd, "ee_force_compliance_force_clip", None)
        d_clip = getattr(cfg_cmd, "ee_force_compliance_delta_clip", None)
        clamp_ws = bool(
            getattr(
                cfg_cmd,
                "ee_force_compliance_workspace_clamp_enable",
                True,
            )
        )
        r_max = float(getattr(cfg_cmd, "ee_target_pos_r_range", (0.0, 0.0))[1])
        z_range = getattr(cfg_cmd, "ee_target_pos_z_range", (0.0, 0.0))
    if not enable:
        scale = (0.0, 0.0, 0.0)

    delta_p = get_force_compliance_delta_p(
        env,
        frame,
        force_to_pos_scale=scale,
        force_deadzone=deadzone,
        force_clip=f_clip,
        delta_clip=d_clip,
        cache_key="ee_target_points",
    )
    if clamp_ws and (r_max is not None) and (z_range is not None):
        r_lim = max(r_max, 0.0)
        z_min = float(z_range[0])
        z_max = float(z_range[1])
        if z_max > z_min + 1.0e-6:
            low = torch.tensor(
                [-r_lim, -r_lim, z_min],
                device=env.device,
                dtype=torch.float32,
            )
            high = torch.tensor(
                [r_lim, r_lim, z_max],
                device=env.device,
                dtype=torch.float32,
            )
            delta_p = scale_delta_p_to_box(cmd_main, delta_p, low, high)
    cmd_y = cmd_y + delta_p

    ee_pos_w = robot.data.body_pos_w[:, ee_body_id]
    ee_quat_w = robot.data.body_quat_w[:, ee_body_id]
    ee_pos_p = world_to_projected_frame(ee_pos_w, frame)

    yaw_q_inv = quat_inv(frame.yaw_quat_w)
    ee_quat_p = quat_mul(yaw_q_inv, ee_quat_w)

    local_y = torch.zeros((env.num_envs, 3), device=env.device)
    local_y[:, 1] = float(offset_distance)
    ee_y_p = ee_pos_p + quat_apply(ee_quat_p, local_y)

    sigma2 = max(float(std) * float(std), 1.0e-12)
    err = torch.sum(torch.square(ee_y_p - cmd_y), dim=1)
    reward = torch.exp(-err / sigma2)
    return torch.where(cmd_active, reward, torch.zeros_like(reward))


def track_pitch_with_ee_target_height_exp(
    env,
    command_name: str,
    # 当末端指令 main 点 z 低于/高于阈值时，分别鼓励的机身 pitch 目标（弧度）
    # 注意：pitch 的正负号与机器人坐标系/关节定义相关；如果发现方向相反，交换两者即可。
    pitch_at_z_low: float,
    pitch_at_z_high: float,
    z_low: float | None = None,
    z_high: float | None = None,
    # 中性区间（"0点 / deadband"）配置：更推荐使用上下限形式
    # - 若同时提供 neutral_z_low & neutral_z_high 且 high > low：
    #   使用平滑过渡函数：区间内 pitch 引导很小，区间外平滑增强
    # - 否则可使用 neutral_zone_width/neutral_z（兼容旧配置）
    neutral_z_low: float | None = None,
    neutral_z_high: float | None = None,
    # 兼容旧配置：中心点+宽度
    neutral_z: float | None = None,
    neutral_zone_width: float = 0.0,
    neutral_pitch: float = 0.0,
    # 奖励核参数
    std: float = 0.3,
    command_active_threshold: float = 1.0e-6,
    # 可选：对奖励做直立缩放（与 track_pitch_exp 一致），避免倒地时奖励误导
    upright_scale_max: float | None = 0.7,
) -> torch.Tensor:
    """根据末端目标高度（z）引导机身 pitch（低头/抬头）配合的指数核奖励。

    思路：
    - 命令 `command_name` 为 9D（Projected COM yaw frame），取 main 点 z = cmd[:, 2]
    - 将 z 映射为 pitch_target，采用平滑过渡函数：
      * 区间内接近 neutral_pitch（但不硬钉死）
      * 区间外平滑地恢复到完整的线性映射 [pitch_at_z_low, pitch_at_z_high]
    - 若 z_low/z_high 为 None，则尝试从 command 生成器 cfg 的采样范围读取（cfg.ranges.pos_z）
    - 从 projected_gravity_b 反解当前 pitch（yaw 不敏感）
    - reward = exp(-(pitch_target - pitch)^2 / std^2)
    """
    def _resolve_command_z_range(
        _env, _command_name: str, _fallback_z: torch.Tensor
    ) -> tuple[float, float]:
        """优先从 command term cfg 解析 (z_low, z_high)，否则用当前命令 z 的 min/max 兜底。"""
        cmd_mgr = getattr(_env, "command_manager", None)
        cmd_term = None
        if cmd_mgr is not None:
            for attr in ("_terms", "_command_terms", "terms"):
                container = getattr(cmd_mgr, attr, None)
                if container is None:
                    continue
                try:
                    candidate = container[_command_name]
                except Exception:
                    candidate = None
                if candidate is not None:
                    cmd_term = candidate
                    break

        cfg = getattr(cmd_term, "cfg", None)
        ranges = getattr(cfg, "ranges", None) if cfg is not None else None
        pos_z = getattr(ranges, "pos_z", None) if ranges is not None else None
        if pos_z is not None:
            try:
                low = float(pos_z[0])
                high = float(pos_z[1])
                if high >= low:
                    return low, high
                return high, low
            except Exception:
                pass

        # fallback: use current command z range
        low = float(torch.min(_fallback_z).item())
        high = float(torch.max(_fallback_z).item())
        if high >= low:
            return low, high
        return high, low

    def _smooth_deadband_weight(
        z: torch.Tensor, z_low_edge: float, z_high_edge: float
    ) -> torch.Tensor:
        """计算平滑的区间外权重函数（连续且可微）。

        特性：
        - 在 [z_low_edge, z_high_edge] 内权重接近 0（引导很小）
        - 区间外权重平滑地增长到 1（恢复完整映射）
        - 过渡宽度根据区间大小自动确定

        实现：使用 smoothstep-like 函数基于归一化距离
        """
        # 自动计算过渡宽度：取区间宽度的 15% 作为过渡带
        deadband_width = z_high_edge - z_low_edge
        transition_width = max(deadband_width * 0.15, 0.02)  # 至少 2cm

        # 计算到区间的归一化距离（负数表示在区间内）
        # d < 0: 区间内, d = 0: 边界, d > 0: 区间外
        dist_from_low = (z_low_edge - z) / transition_width
        dist_from_high = (z - z_high_edge) / transition_width

        # 取两个方向的最大值（哪边出界更多就按哪边算）
        normalized_dist = torch.maximum(dist_from_low, dist_from_high)

        # 使用平滑 sigmoid: w(d) = 1 / (1 + exp(-4*d))
        # d << 0 (区间内深处) -> w ≈ 0
        # d = 0 (边界) -> w = 0.5
        # d >> 0 (区间外远处) -> w ≈ 1
        weight = torch.sigmoid(4.0 * normalized_dist)

        return weight

    cmd = env.command_manager.get_command(command_name)  # (B, 9)
    cmd_active = torch.linalg.norm(cmd, dim=1) > command_active_threshold
    cmd_z = cmd[:, 2]

    if z_low is None or z_high is None:
        cfg_low, cfg_high = _resolve_command_z_range(env, command_name, cmd_z)
        z_low = cfg_low if z_low is None else z_low
        z_high = cfg_high if z_high is None else z_high

    z_low_f = float(z_low)
    z_high_f = float(z_high)
    pitch_low = float(pitch_at_z_low)
    pitch_high = float(pitch_at_z_high)

    # 全区间线性映射（基础映射函数）
    denom_full = max(z_high_f - z_low_f, 1.0e-6)
    t_full = ((cmd_z - z_low_f) / denom_full).clamp_(0.0, 1.0)
    pitch_linear = pitch_low + t_full * (pitch_high - pitch_low)

    pitch_target: torch.Tensor
    # 1) 优先使用上下限形式的中性区间（平滑过渡）
    use_bounds_deadband = (
        neutral_z_low is not None
        and neutral_z_high is not None
        and float(neutral_z_high) > float(neutral_z_low)
    )
    if use_bounds_deadband:
        neutral_pitch_f = float(neutral_pitch)
        # clamp edges into [z_low, z_high] to avoid invalid ranges
        z_low_edge = max(z_low_f, float(neutral_z_low))
        z_high_edge = min(z_high_f, float(neutral_z_high))

        if z_high_edge > z_low_edge:
            # 计算平滑权重：区间内 ≈ 0，区间外 ≈ 1
            weight = _smooth_deadband_weight(cmd_z, z_low_edge, z_high_edge)
            # 混合：neutral_pitch (区间内) <-> pitch_linear (区间外)
            pitch_target = neutral_pitch_f + (
                pitch_linear - neutral_pitch_f
            ) * weight
        else:
            pitch_target = pitch_linear

    # 2) 兼容旧配置：中心点 + 宽度（同样使用平滑过渡）
    elif float(neutral_zone_width) > 0.0:
        neutral_pitch_f = float(neutral_pitch)
        z0 = (
            float(neutral_z)
            if neutral_z is not None
            else 0.5 * (z_low_f + z_high_f)
        )
        half_w = 0.5 * float(neutral_zone_width)
        z_low_edge = max(z_low_f, z0 - half_w)
        z_high_edge = min(z_high_f, z0 + half_w)

        if z_high_edge > z_low_edge:
            weight = _smooth_deadband_weight(cmd_z, z_low_edge, z_high_edge)
            pitch_target = neutral_pitch_f + (
                pitch_linear - neutral_pitch_f
            ) * weight
        else:
            pitch_target = pitch_linear

    # 3) 无 deadband：简单线性映射
    else:
        pitch_target = pitch_linear

    # 当前 pitch（rad）：由 projected_gravity_b 反解，yaw 不敏感
    g_b = env.scene["robot"].data.projected_gravity_b
    pitch = torch.atan2(g_b[:, 0], -g_b[:, 2])

    pitch_error = torch.square(pitch_target - pitch)
    reward = torch.exp(-pitch_error / (float(std) ** 2))

    if upright_scale_max is not None and float(upright_scale_max) > 0.0:
        z_upright = -g_b[:, 2]
        reward = reward * (
            torch.clamp(z_upright, 0.0, float(upright_scale_max))
            / float(upright_scale_max)
        )

    return torch.where(cmd_active, reward, torch.zeros_like(reward))


def track_pitch_with_ee_target_height_exp_force_compliance(
    env,
    command_name: str,
    pitch_at_z_low: float,
    pitch_at_z_high: float,
    z_low: float | None = None,
    z_high: float | None = None,
    neutral_z_low: float | None = None,
    neutral_z_high: float | None = None,
    neutral_z: float | None = None,
    neutral_zone_width: float = 0.0,
    neutral_pitch: float = 0.0,
    std: float = 0.3,
    command_active_threshold: float = 1.0e-6,
    upright_scale_max: float | None = 0.7,
) -> torch.Tensor:
    """外力偏移版本：用偏移管理器修正 main 点 z，再执行原 z->pitch 跟踪逻辑。"""

    def _resolve_command_z_range(
        _env, _command_name: str, _fallback_z: torch.Tensor
    ) -> tuple[float, float]:
        cmd_mgr = getattr(_env, "command_manager", None)
        cmd_term = None
        if cmd_mgr is not None:
            for attr in ("_terms", "_command_terms", "terms"):
                container = getattr(cmd_mgr, attr, None)
                if container is None:
                    continue
                try:
                    candidate = container[_command_name]
                except Exception:
                    candidate = None
                if candidate is not None:
                    cmd_term = candidate
                    break

        cfg = getattr(cmd_term, "cfg", None)
        ranges = getattr(cfg, "ranges", None) if cfg is not None else None
        pos_z = getattr(ranges, "pos_z", None) if ranges is not None else None
        if pos_z is not None:
            try:
                low = float(pos_z[0])
                high = float(pos_z[1])
                if high >= low:
                    return low, high
                return high, low
            except Exception:
                pass

        low = float(torch.min(_fallback_z).item())
        high = float(torch.max(_fallback_z).item())
        if high >= low:
            return low, high
        return high, low

    def _smooth_deadband_weight(
        z: torch.Tensor, z_low_edge: float, z_high_edge: float
    ) -> torch.Tensor:
        deadband_width = z_high_edge - z_low_edge
        transition_width = max(deadband_width * 0.15, 0.02)
        dist_from_low = (z_low_edge - z) / transition_width
        dist_from_high = (z - z_high_edge) / transition_width
        normalized_dist = torch.maximum(dist_from_low, dist_from_high)
        return torch.sigmoid(4.0 * normalized_dist)

    # 1) command z
    cmd = env.command_manager.get_command(command_name)  # (B, 9)
    cmd_active = torch.linalg.norm(cmd, dim=1) > command_active_threshold
    cmd_main = cmd[:, 0:3]
    cmd_z = cmd[:, 2]

    # 2) apply force-compliance z offset (same manager as EE tracking rewards)
    cfg_root = getattr(getattr(env, "cfg", None), "config_summary", None)
    cfg_cmd = (
        getattr(cfg_root, "command", None) if cfg_root is not None else None
    )
    if cfg_cmd is None:
        enable = True
        scale = (0.0, 0.0, 0.0)
        deadzone = 0.0
        f_clip = None
        d_clip = None
        clamp_ws = False
        r_max = None
        z_range = None
    else:
        enable = bool(getattr(cfg_cmd, "ee_force_compliance_enable", True))
        scale = getattr(cfg_cmd, "ee_force_compliance_scale", (0.0, 0.0, 0.0))
        deadzone = getattr(cfg_cmd, "ee_force_compliance_force_deadzone", 0.0)
        f_clip = getattr(cfg_cmd, "ee_force_compliance_force_clip", None)
        d_clip = getattr(cfg_cmd, "ee_force_compliance_delta_clip", None)
        clamp_ws = bool(
            getattr(
                cfg_cmd,
                "ee_force_compliance_workspace_clamp_enable",
                True,
            )
        )
        r_max = float(getattr(cfg_cmd, "ee_target_pos_r_range", (0.0, 0.0))[1])
        z_range = getattr(cfg_cmd, "ee_target_pos_z_range", (0.0, 0.0))
    if not enable:
        scale = (0.0, 0.0, 0.0)

    # We need the projected-yaw frame; reuse robot/trunk from scene
    robot = env.scene["robot"]
    trunk_body_id = robot.find_bodies("base_link")[0][0]
    terrain_sensor = env.scene.sensors.get("height_scanner", None)
    frame = compute_projected_com_yaw_frame(
        asset=robot,
        trunk_body_id=trunk_body_id,
        terrain_height_sensor=terrain_sensor,
        lpf_state_key=command_name,
        read_cached_ground_z_only=True,
    )
    delta_p = get_force_compliance_delta_p(
        env,
        frame,
        force_to_pos_scale=scale,
        force_deadzone=deadzone,
        force_clip=f_clip,
        delta_clip=d_clip,
        cache_key="ee_target_points",
    )
    if clamp_ws and (r_max is not None) and (z_range is not None):
        r_lim = max(r_max, 0.0)
        z_min = float(z_range[0])
        z_max = float(z_range[1])
        if z_max > z_min + 1.0e-6:
            low = torch.tensor(
                [-r_lim, -r_lim, z_min],
                device=env.device,
                dtype=torch.float32,
            )
            high = torch.tensor(
                [r_lim, r_lim, z_max],
                device=env.device,
                dtype=torch.float32,
            )
            delta_p = scale_delta_p_to_box(cmd_main, delta_p, low, high)

    cmd_z = cmd_z + delta_p[:, 2]

    # 3) resolve z range for mapping
    if z_low is None or z_high is None:
        cfg_low, cfg_high = _resolve_command_z_range(env, command_name, cmd_z)
        z_low = cfg_low if z_low is None else z_low
        z_high = cfg_high if z_high is None else z_high

    z_low_f = float(z_low)
    z_high_f = float(z_high)
    pitch_low = float(pitch_at_z_low)
    pitch_high = float(pitch_at_z_high)

    denom_full = max(z_high_f - z_low_f, 1.0e-6)
    t_full = ((cmd_z - z_low_f) / denom_full).clamp_(0.0, 1.0)
    pitch_linear = pitch_low + t_full * (pitch_high - pitch_low)

    # 4) deadband mapping (same as original)
    if (
        neutral_z_low is not None
        and neutral_z_high is not None
        and float(neutral_z_high) > float(neutral_z_low)
    ):
        neutral_pitch_f = float(neutral_pitch)
        z_low_edge = max(z_low_f, float(neutral_z_low))
        z_high_edge = min(z_high_f, float(neutral_z_high))
        if z_high_edge > z_low_edge:
            w = _smooth_deadband_weight(cmd_z, z_low_edge, z_high_edge)
            pitch_target = neutral_pitch_f + (
                pitch_linear - neutral_pitch_f
            ) * w
        else:
            pitch_target = pitch_linear
    elif float(neutral_zone_width) > 0.0:
        neutral_pitch_f = float(neutral_pitch)
        z0 = (
            float(neutral_z)
            if neutral_z is not None
            else 0.5 * (z_low_f + z_high_f)
        )
        half_w = 0.5 * float(neutral_zone_width)
        z_low_edge = max(z_low_f, z0 - half_w)
        z_high_edge = min(z_high_f, z0 + half_w)
        if z_high_edge > z_low_edge:
            w = _smooth_deadband_weight(cmd_z, z_low_edge, z_high_edge)
            pitch_target = neutral_pitch_f + (
                pitch_linear - neutral_pitch_f
            ) * w
        else:
            pitch_target = pitch_linear
    else:
        pitch_target = pitch_linear

    # current pitch from projected gravity
    g_b = env.scene["robot"].data.projected_gravity_b
    pitch = torch.atan2(g_b[:, 0], -g_b[:, 2])

    pitch_error = torch.square(pitch_target - pitch)
    reward = torch.exp(-pitch_error / (float(std) ** 2))

    if upright_scale_max is not None and float(upright_scale_max) > 0.0:
        z_upright = -g_b[:, 2]
        reward = reward * (
            torch.clamp(z_upright, 0.0, float(upright_scale_max))
            / float(upright_scale_max)
        )

    return torch.where(cmd_active, reward, torch.zeros_like(reward))

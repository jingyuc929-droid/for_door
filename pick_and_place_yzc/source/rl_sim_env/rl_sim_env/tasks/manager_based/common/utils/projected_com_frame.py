"""投影参考系工具：以躯干质心在地面上的投影点作为原点，yaw 对齐作为朝向。

设计目标：
- **原点**：躯干(指定刚体)质心在世界系的 (x, y) 投影到地面高度 z_ground 上的点。
- **朝向**：仅使用机器人躯干的 yaw（忽略 pitch/roll），得到一个水平
  参考系，方便把 locomotion 与 manipulation 解耦。

该参考系将用于：
- 机械臂末端目标 command 的采样（命令在该参考系下表达）
- 末端跟随 reward 的误差计算（末端状态变换到该参考系下）

注意：
- 地面高度通过 `RayCaster` 的 `ray_hits_w` 估计，采用“与 query_xy 最近
  的 hit 点”的 z 作为地面高度，避免依赖网格顺序。
- 不做任何硬编码：传入的 sensor / body / joint 名称均来自配置。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.assets import Articulation
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import yaw_quat, quat_apply_inverse


@dataclass(frozen=True)
class ProjectedComYawFrame:
    """以躯干 COM 地面投影为原点、yaw 对齐的参考系。"""

    origin_w: torch.Tensor  # (B, 3)
    yaw_quat_w: torch.Tensor  # (B, 4) in wxyz


def _batched_nearest_ground_z_from_raycast(
    ray_hits_w: torch.Tensor, query_xy_w: torch.Tensor
) -> torch.Tensor:
    """从 RayCaster 命中点中，为每个环境选择距离 query_xy_w 最近的 hit 的 z。

    Args:
        ray_hits_w: (B, M, 3) 世界坐标命中点。
        query_xy_w: (B, 2) 查询点 (x, y)。

    Returns:
        ground_z: (B,) 每个环境估计的地面高度。
    """
    # (B, M, 2)
    hits_xy = ray_hits_w[..., :2]
    # (B, 1, 2)
    q_xy = query_xy_w.unsqueeze(1)
    # (B, M)
    d2 = torch.sum(torch.square(hits_xy - q_xy), dim=-1)
    # (B,)
    idx = torch.argmin(d2, dim=1)
    # (B,)
    env_idx = torch.arange(ray_hits_w.shape[0], device=ray_hits_w.device)
    ground_z = ray_hits_w[env_idx, idx, 2]
    return ground_z


def get_body_com_pos_w(asset: Articulation, body_id: int) -> torch.Tensor:
    """获取指定刚体的质心世界坐标 (B, 3)。

    注意：在部分 IsaacLab/PhysX 版本中，`root_physx_view.get_coms()` 的坐标系
    可能与 `asset.data.body_pos_w` 的 world 坐标不一致（例如缺少 env origin
    偏移），从而导致可视化点“堆在地图中心”。

    这里做一个鲁棒融合：
    - 优先使用 physx view 的 COM（更符合“质心”定义）
    - 若其与 `body_pos_w` 差异过大（通常意味着坐标系不一致），则回退到
      `body_pos_w`（在 world frame 下稳定可靠）
    """
    device = asset.device

    # world body origin position (reliable world frame)
    body_pos_w = asset.data.body_pos_w[:, body_id].to(device)

    # physx-view COM candidate
    coms = asset.root_physx_view.get_coms()
    com_pos_w = coms[:, body_id, :3].to(device)

    # detect mismatch (typically missing env origin offset)
    delta_xy = torch.linalg.norm((com_pos_w - body_pos_w)[:, :2], dim=1)
    use_body_pos = delta_xy > 0.5
    return torch.where(use_body_pos.unsqueeze(1), body_pos_w, com_pos_w)


def compute_projected_com_yaw_frame(
    asset: Articulation,
    trunk_body_id: int,
    terrain_height_sensor: RayCaster,
    ground_z_lpf_alpha: float = 1.0,
    lpf_state_key: str | None = None,
    update_lpf_state: bool = True,
    read_cached_ground_z_only: bool = False,
) -> ProjectedComYawFrame:
    """计算“躯干 COM 地面投影 + yaw 对齐”的参考系。

    Args:
        asset: 机器人 articulation（通常是 env.scene["robot"]）。
        trunk_body_id: 躯干刚体 id（例如 base_link 对应的 id）。
        terrain_height_sensor: RayCaster（通常是 height_scanner）。

    Returns:
        ProjectedComYawFrame: origin_w 与 yaw_quat_w。
    """
    # 1) 躯干 COM 的世界坐标
    trunk_com_w = get_body_com_pos_w(asset, trunk_body_id)  # (B, 3)

    # 2) 地面高度：从高度扫描命中点中找与 (x, y) 最近的 z
    hits_w = terrain_height_sensor.data.ray_hits_w[..., :3]  # (B, M, 3)
    ground_z = _batched_nearest_ground_z_from_raycast(
        hits_w, trunk_com_w[:, :2]
    )  # (B,)

    # 3) 地面高度滤波状态（按 command_name/主体 id 隔离）
    state_suffix = lpf_state_key if lpf_state_key is not None else str(trunk_body_id)
    state_name = f"_projected_com_ground_z_lpf_{state_suffix}"
    prev_ground_z = getattr(asset, state_name, None)
    has_valid_prev = (
        isinstance(prev_ground_z, torch.Tensor)
        and prev_ground_z.shape == ground_z.shape
        and prev_ground_z.device == ground_z.device
    )

    if read_cached_ground_z_only:
        if not has_valid_prev:
            raise RuntimeError(
                "Projected COM frame requires cached filtered ground_z, "
                f"but state '{state_name}' is missing or invalid. "
                "Please ensure command-side frame update runs before reward evaluation."
            )
        ground_z = prev_ground_z
    else:
        # 可选：对 ground_z 做一阶低通滤波，降低台阶边缘的目标点跳变
        alpha = float(ground_z_lpf_alpha)
        alpha = max(0.0, min(1.0, alpha))
        if alpha < 1.0 and has_valid_prev:
            ground_z = alpha * ground_z + (1.0 - alpha) * prev_ground_z
        # 命令侧作为唯一写入者：无论是否启用 LPF 都回写当前 ground_z
        if update_lpf_state:
            setattr(asset, state_name, ground_z.detach())

    # 4) 原点 = (com_x, com_y, ground_z)
    origin_w = torch.stack(
        [trunk_com_w[:, 0], trunk_com_w[:, 1], ground_z], dim=-1
    )  # (B, 3)

    # 5) yaw-only quaternion from root orientation
    yaw_q_w = yaw_quat(asset.data.root_quat_w)

    return ProjectedComYawFrame(origin_w=origin_w, yaw_quat_w=yaw_q_w)


def compute_trunk_com_yaw_frame(
    asset: Articulation,
    trunk_body_id: int,
) -> ProjectedComYawFrame:
    """计算"躯干 COM + yaw 对齐"的参考系（不投影到地面）。

    与 compute_projected_com_yaw_frame 的区别：
    - 该函数的原点是躯干质心的实际位置（包括 z 坐标）
    - 不需要地面高度传感器

    Args:
        asset: 机器人 articulation（通常是 env.scene["robot"]）。
        trunk_body_id: 躯干刚体 id（例如 base_link 对应的 id）。

    Returns:
        ProjectedComYawFrame: origin_w 与 yaw_quat_w（复用同一数据结构）。
    """
    # 1) 躯干 COM 的世界坐标（包括实际 z 高度）
    trunk_com_w = get_body_com_pos_w(asset, trunk_body_id)  # (B, 3)

    # 2) yaw-only quaternion from root orientation
    yaw_q_w = yaw_quat(asset.data.root_quat_w)

    return ProjectedComYawFrame(origin_w=trunk_com_w, yaw_quat_w=yaw_q_w)


def world_to_projected_frame(
    pos_w: torch.Tensor, frame: ProjectedComYawFrame
) -> torch.Tensor:
    """把世界坐标点变换到投影参考系（yaw 对齐、原点在地面投影）。"""
    return quat_apply_inverse(frame.yaw_quat_w, pos_w - frame.origin_w)

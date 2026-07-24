# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        return False  # we have infinite terrain because it is a plane
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")


def illegal_contact_new(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    illegal_contact = torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )
    # 只在指定的 terrain 类型上生效（例如 env.terrain_types == 5）
    terrain_mask = env.terrain_types != 8
    return torch.logical_and(illegal_contact, terrain_mask)


def bad_orientation_new(
    env: ManagerBasedRLEnv, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return (torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle) * (env.terrain_types >= 7).float()


def any_feet_below_height(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("frame_transform"),
    threshold: float = -0.2,
) -> torch.Tensor:
    """当任意足端在世界系的 z 轴高度低于给定阈值时终止（仅在指定地形类型生效）。

    参数:
        sensor_cfg: 足端位姿由 `FrameTransformer` 提供，通常为 `SceneEntityCfg("frame_transform")`。
        threshold: 判定阈值（米），默认 -0.2。
    返回:
        shape=(num_envs,) 的 bool 张量。
    """
    terrain_type_valid = [3, 7, 8, 9]
    foot_tf = env.scene.sensors[sensor_cfg.name]
    # 形状: (num_envs, num_feet)
    feet_z = foot_tf.data.target_pos_w[:, :, 2]
    feet_cond = (feet_z < threshold).any(dim=1)
    # 仅当 env.terrain_types 属于白名单时生效
    terrain_types = env.terrain_types.to(device=feet_z.device)
    valid_types = torch.tensor(terrain_type_valid, device=terrain_types.device, dtype=terrain_types.dtype)
    valid_mask = (env.terrain_types.unsqueeze(-1) == valid_types.unsqueeze(0)).any(dim=1)
    return torch.logical_and(feet_cond, valid_mask)


def body_stillness_within_window(
    env: ManagerBasedRLEnv,
    threshold: float,
    window_seconds: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """当机身在过去 window_seconds 内的位移变化小于阈值时终止。

    判定逻辑：以“最近一次位移超过阈值的时刻”为参考点，若之后累计的位移始终小于阈值并持续时间达到 window_seconds，则视为静止终止。

    参数:
        threshold: 位移阈值（米）。
        window_seconds: 时间窗口长度（秒），默认 10s。
        asset_cfg: 机器人实体，默认 `robot`。
    返回:
        shape=(num_envs,) 的 bool 张量。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    curr_pos = asset.data.root_pos_w  # (num_envs, 3)

    # 需要的步数
    required_steps = max(1, int(window_seconds / float(env.step_dt)))

    # 懒初始化：参考位置与计数器（逐环境）
    if not hasattr(env, "_stillness_ref_pos_w"):
        env._stillness_ref_pos_w = curr_pos.clone()
    if not hasattr(env, "_stillness_counter"):
        env._stillness_counter = torch.zeros(curr_pos.shape[0], device=curr_pos.device, dtype=torch.long)

    # 计算与参考位置的位移（仅考虑 XY 平面）
    disp = torch.norm((curr_pos - env._stillness_ref_pos_w)[:, :2], dim=1)  # (num_envs,)

    # 位移超过阈值的环境：重置参考点与计数器
    moved_mask = disp > threshold
    if moved_mask.any():
        env._stillness_ref_pos_w[moved_mask] = curr_pos[moved_mask]
        env._stillness_counter[moved_mask] = 0

    # 位移未超过阈值的环境：计数 +1
    stay_mask = ~moved_mask
    if stay_mask.any():
        env._stillness_counter[stay_mask] += 1

    terrain_type_valid = [3, 7, 8, 9]
    # 仅当 env.terrain_types 属于白名单时生效
    terrain_types = env.terrain_types.to(device=env.device)
    valid_types = torch.tensor(terrain_type_valid, device=terrain_types.device, dtype=terrain_types.dtype)
    valid_mask = (terrain_types.unsqueeze(-1) == valid_types.unsqueeze(0)).any(dim=1)
    # 达到窗口步数则判定终止
    return torch.logical_and(env._stillness_counter >= required_steps, valid_mask)

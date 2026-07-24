# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_command")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_pos(
    env: ManagerBasedRLEnv, env_ids: Sequence[int]
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    # asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_command")
    # compute the distance the robot walked
    distance = torch.norm(command[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance < 0.3
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance > 0.3
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def lin_vel_x_command_threshold(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    command = env.command_manager.get_term("base_command")
    max_episode_length = env.max_episode_length

    lin_x_level = command.cfg.lin_x_level
    max_lin_x_level = command.cfg.max_lin_x_level
    if (env.common_step_counter > ((lin_x_level + 1) * max_episode_length * 8)) and (lin_x_level < max_lin_x_level):
        if lin_x_level < max_lin_x_level:
            lin_x_level += 1.0
            command.cfg.lin_x_level = lin_x_level
        for key, range_cfg in command.cfg.ranges.items():
            step0 = (range_cfg.max_curriculum_lin_x[0] - range_cfg.start_curriculum_lin_x[0]) / max_lin_x_level
            step1 = (range_cfg.max_curriculum_lin_x[1] - range_cfg.start_curriculum_lin_x[1]) / max_lin_x_level
            new_min = range_cfg.start_curriculum_lin_x[0] + step0 * lin_x_level
            new_max = range_cfg.start_curriculum_lin_x[1] + step1 * lin_x_level
            range_cfg.lin_vel_x = (new_min, new_max)

    return torch.tensor(lin_x_level, device=env.device)


def ang_vel_z_command_threshold(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    command = env.command_manager.get_term("base_command")
    max_episode_length = env.max_episode_length

    ang_z_level = command.cfg.ang_z_level
    max_ang_z_level = command.cfg.max_ang_z_level
    if (env.common_step_counter > ((ang_z_level + 1) * max_episode_length * 8)) and (ang_z_level < max_ang_z_level):
        if ang_z_level < max_ang_z_level:
            ang_z_level += 1.0
            command.cfg.ang_z_level = ang_z_level
        for key, range_cfg in command.cfg.ranges.items():
            step0 = (range_cfg.max_curriculum_ang_z[0] - range_cfg.start_curriculum_ang_z[0]) / max_ang_z_level
            step1 = (range_cfg.max_curriculum_ang_z[1] - range_cfg.start_curriculum_ang_z[1]) / max_ang_z_level
            new_min = range_cfg.start_curriculum_ang_z[0] + step0 * ang_z_level
            new_max = range_cfg.start_curriculum_ang_z[1] + step1 * ang_z_level
            range_cfg.ang_vel_z = (new_min, new_max)

    return torch.tensor(ang_z_level, device=env.device)


def exponentially_anneal_reward_weight(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, weight_max: float, w: float):
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    print(env.common_step_counter)
    if env.common_step_counter % 100 == 0:
        if abs(term_cfg.weight) < weight_max:
            term_cfg.weight = term_cfg.weight / w
            env.reward_manager.set_term_cfg(term_name, term_cfg)
    return torch.tensor(term_cfg.weight, device=env.device)


def ee_external_force_threshold(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """末端外力课程：按训练步数逐级增大 ee_external_force 的施力范围（世界坐标系）。

    设计目标（仿照速度课程）：
    - 通过离散 level（0..max_level）控制难度；
    - 每达到一定步数阈值就提升 1 级；
    - 将 force/torque 的 range 从 start 线性插值到 max；
    - 仅当任务配置显式提供课程参数时启用，否则 no-op，避免影响其它工程。
    """
    # ----------------------------
    # 1) 读取课程配置（缺省则不启用）
    # ----------------------------
    cfg_root = getattr(getattr(env, "cfg", None), "config_summary", None)
    cfg_event = getattr(cfg_root, "event", None)
    if cfg_event is None:
        return torch.tensor(0.0, device=env.device)

    # enable flag (optional)
    if hasattr(cfg_event, "ee_force_curriculum_enable") and (not cfg_event.ee_force_curriculum_enable):
        return torch.tensor(0.0, device=env.device)

    required = [
        "ee_force_curriculum_max_level",
        "ee_force_curriculum_episode_mult",
        "ee_force_x_range_start",
        "ee_force_x_range_max",
        "ee_force_y_range_start",
        "ee_force_y_range_max",
        "ee_force_z_range_start",
        "ee_force_z_range_max",
        "ee_torque_x_range_start",
        "ee_torque_x_range_max",
        "ee_torque_y_range_start",
        "ee_torque_y_range_max",
        "ee_torque_z_range_start",
        "ee_torque_z_range_max",
    ]
    if not all(hasattr(cfg_event, k) for k in required):
        # not configured for this task -> no-op
        return torch.tensor(0.0, device=env.device)

    max_level = float(cfg_event.ee_force_curriculum_max_level)
    if max_level <= 0.0:
        return torch.tensor(0.0, device=env.device)

    # threshold cadence: emulate velocity curriculum using episode multiples (avoid hardcoding step counts)
    episode_mult = float(cfg_event.ee_force_curriculum_episode_mult)
    if episode_mult <= 0.0:
        return torch.tensor(0.0, device=env.device)

    # ----------------------------
    # 2) 取/存当前 level（存 env 上，避免侵入底层 EventTermCfg 结构）
    # ----------------------------
    if not hasattr(env, "ee_force_curriculum_level"):
        env.ee_force_curriculum_level = 0.0

    level = float(env.ee_force_curriculum_level)
    if level < 0.0:
        level = 0.0
    if level > max_level:
        level = max_level

    # ----------------------------
    # 3) 按步数阈值推进 level
    # ----------------------------
    # 对齐速度课程：基于 env.common_step_counter
    step_counter = int(getattr(env, "common_step_counter", 0))
    max_episode_length = int(getattr(env, "max_episode_length", 0))
    steps_per_level = int(max_episode_length * episode_mult)
    if steps_per_level <= 0:
        return torch.tensor(level, device=env.device)

    # next threshold: (level + 1) * steps_per_level
    if (step_counter > int((level + 1.0) * steps_per_level)) and (level < max_level):
        level = min(level + 1.0, max_level)
        env.ee_force_curriculum_level = level

    # ----------------------------
    # 4) 更新 ee_external_force 的 range（start -> max 线性插值）
    # ----------------------------
    # event term cfg lives under env.cfg.events
    ev_cfg = getattr(getattr(env, "cfg", None), "events", None)
    ee_term = getattr(ev_cfg, "ee_external_force", None)
    if ee_term is None:
        return torch.tensor(level, device=env.device)

    # interpolate helper
    def _lerp_range(r0: tuple[float, float], r1: tuple[float, float], a: float) -> tuple[float, float]:
        return (r0[0] + (r1[0] - r0[0]) * a, r0[1] + (r1[1] - r0[1]) * a)

    alpha = float(level / max_level) if max_level > 0.0 else 0.0

    ee_term.params["force_x_range"] = _lerp_range(cfg_event.ee_force_x_range_start, cfg_event.ee_force_x_range_max, alpha)
    ee_term.params["force_y_range"] = _lerp_range(cfg_event.ee_force_y_range_start, cfg_event.ee_force_y_range_max, alpha)
    ee_term.params["force_z_range"] = _lerp_range(cfg_event.ee_force_z_range_start, cfg_event.ee_force_z_range_max, alpha)
    ee_term.params["torque_x_range"] = _lerp_range(cfg_event.ee_torque_x_range_start, cfg_event.ee_torque_x_range_max, alpha)
    ee_term.params["torque_y_range"] = _lerp_range(cfg_event.ee_torque_y_range_start, cfg_event.ee_torque_y_range_max, alpha)
    ee_term.params["torque_z_range"] = _lerp_range(cfg_event.ee_torque_z_range_start, cfg_event.ee_torque_z_range_max, alpha)

    return torch.tensor(level, device=env.device)

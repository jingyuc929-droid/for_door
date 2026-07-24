# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import math
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


def _apply_velocity_command_range(command, *, axis: str) -> None:
    """Materialize one velocity curriculum level into every terrain command range."""
    if axis == "lin_x":
        level = float(command.cfg.lin_x_level)
        max_level = float(command.cfg.max_lin_x_level)
        start_attr = "start_curriculum_lin_x"
        max_attr = "max_curriculum_lin_x"
        output_attr = "lin_vel_x"
    elif axis == "ang_z":
        level = float(command.cfg.ang_z_level)
        max_level = float(command.cfg.max_ang_z_level)
        start_attr = "start_curriculum_ang_z"
        max_attr = "max_curriculum_ang_z"
        output_attr = "ang_vel_z"
    else:
        raise ValueError(f"Unsupported velocity curriculum axis: {axis}")

    if max_level <= 0.0:
        return
    level = min(max(level, 0.0), max_level)
    for range_cfg in command.cfg.ranges.values():
        start_range = getattr(range_cfg, start_attr)
        max_range = getattr(range_cfg, max_attr)
        setattr(
            range_cfg,
            output_attr,
            (
                start_range[0]
                + ((max_range[0] - start_range[0]) / max_level) * level,
                start_range[1]
                + ((max_range[1] - start_range[1]) / max_level) * level,
            ),
        )


def apply_velocity_command_curriculum_ranges(env: ManagerBasedRLEnv) -> None:
    """Apply already-restored lin-x/ang-z levels without advancing either curriculum."""
    command = env.command_manager.get_term("base_command")
    _apply_velocity_command_range(command, axis="lin_x")
    _apply_velocity_command_range(command, axis="ang_z")


def derive_velocity_curriculum_level(
    env: ManagerBasedRLEnv, *, max_level: float, step_counter: int
) -> float:
    """Derive the level implied by the strict ``step > threshold`` runtime rule."""
    command = env.command_manager.get_term("base_command")
    period = float(env.max_episode_length) * float(
        command.cfg.vel_curriculum_episode_mult
    )
    if max_level <= 0.0 or period <= 0.0:
        return 0.0
    # Number of positive integer thresholds k*period strictly below step_counter.
    level = max(0, math.ceil(float(step_counter) / period) - 1)
    return min(float(max_level), float(level))


def lin_vel_x_command_threshold(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    command = env.command_manager.get_term("base_command")
    max_episode_length = env.max_episode_length

    lin_x_level = command.cfg.lin_x_level
    max_lin_x_level = command.cfg.max_lin_x_level
    if (env.common_step_counter > ((lin_x_level + 1) * max_episode_length * command.cfg.vel_curriculum_episode_mult)) and (lin_x_level < max_lin_x_level):
        lin_x_level = min(lin_x_level + 1.0, max_lin_x_level)
        command.cfg.lin_x_level = lin_x_level
    # 每次都根据当前 level 重新插值命令范围：resume 写回 level 后可立即物化。
    _apply_velocity_command_range(command, axis="lin_x")

    return torch.tensor(lin_x_level, device=env.device)


def ang_vel_z_command_threshold(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    command = env.command_manager.get_term("base_command")
    max_episode_length = env.max_episode_length

    ang_z_level = command.cfg.ang_z_level
    max_ang_z_level = command.cfg.max_ang_z_level
    if (env.common_step_counter > ((ang_z_level + 1) * max_episode_length * command.cfg.vel_curriculum_episode_mult)) and (ang_z_level < max_ang_z_level):
        ang_z_level = min(ang_z_level + 1.0, max_ang_z_level)
        command.cfg.ang_z_level = ang_z_level
    _apply_velocity_command_range(command, axis="ang_z")

    return torch.tensor(ang_z_level, device=env.device)


def exponentially_anneal_reward_weight(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, weight_max: float, w: float):
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    print(env.common_step_counter)
    if env.common_step_counter % 100 == 0:
        if abs(term_cfg.weight) < weight_max:
            term_cfg.weight = term_cfg.weight / w
            env.reward_manager.set_term_cfg(term_name, term_cfg)
    return torch.tensor(term_cfg.weight, device=env.device)


_EE_FORCE_CURRICULUM_FIELDS = (
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
)


def _get_ee_force_curriculum_cfg(env: ManagerBasedRLEnv):
    cfg_root = getattr(getattr(env, "cfg", None), "config_summary", None)
    cfg_event = getattr(cfg_root, "event", None)
    if cfg_event is None:
        return None
    if hasattr(cfg_event, "ee_force_curriculum_enable") and (
        not cfg_event.ee_force_curriculum_enable
    ):
        return None
    if not all(hasattr(cfg_event, key) for key in _EE_FORCE_CURRICULUM_FIELDS):
        return None
    return cfg_event


def get_ee_force_curriculum_velocity_offset_steps(
    env: ManagerBasedRLEnv, *, force_episode_mult: float
) -> int:
    """Return the exact velocity-delay offset used by the EE force curriculum."""
    max_episode_length = int(getattr(env, "max_episode_length", 0))
    try:
        base_cmd = env.command_manager.get_term("base_command")
        vel_max_level = float(
            getattr(base_cmd.cfg, "max_lin_x_level", 0.0)
        )
        vel_episode_mult = float(
            getattr(
                base_cmd.cfg,
                "vel_curriculum_episode_mult",
                force_episode_mult,
            )
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return 0
    return int(
        (vel_max_level + 1.0) * vel_episode_mult * max_episode_length
    )


def derive_ee_external_force_curriculum_level(
    env: ManagerBasedRLEnv, *, step_counter: int
) -> float:
    """Derive the force level implied by the same thresholds as uninterrupted training."""
    cfg_event = _get_ee_force_curriculum_cfg(env)
    if cfg_event is None:
        return 0.0

    max_level = float(cfg_event.ee_force_curriculum_max_level)
    episode_mult = float(cfg_event.ee_force_curriculum_episode_mult)
    max_episode_length = int(getattr(env, "max_episode_length", 0))
    steps_per_level = int(max_episode_length * episode_mult)
    if max_level <= 0.0 or episode_mult <= 0.0 or steps_per_level <= 0:
        return 0.0
    vel_offset_steps = get_ee_force_curriculum_velocity_offset_steps(
        env, force_episode_mult=episode_mult
    )
    elapsed = int(step_counter) - vel_offset_steps
    # Runtime uses ``step > offset + k * period``.  ceil(elapsed/period)-1
    # counts exactly how many such strict thresholds have been crossed.
    level = max(0, math.ceil(float(elapsed) / steps_per_level) - 1)
    return min(max_level, float(level))


def apply_ee_external_force_curriculum_level(
    env: ManagerBasedRLEnv, level: float
) -> float:
    """Materialize a force level into EventTerm ranges and command force amplitude."""
    cfg_event = _get_ee_force_curriculum_cfg(env)
    if cfg_event is None:
        return 0.0

    max_level = float(cfg_event.ee_force_curriculum_max_level)
    if max_level <= 0.0:
        return 0.0
    level = min(max(float(level), 0.0), max_level)
    env.ee_force_curriculum_level = level

    # ManagerBase deep-copies its cfg during construction.  Updating only
    # ``env.cfg.events`` changes exported config/logs but not the EventManager term
    # that actually supplies kwargs to interval events.  Materialize into both.
    event_terms = []
    ev_cfg = getattr(getattr(env, "cfg", None), "events", None)
    source_term = getattr(ev_cfg, "ee_external_force", None)
    if source_term is not None:
        event_terms.append(source_term)
    event_manager = getattr(env, "event_manager", None)
    if event_manager is not None and hasattr(event_manager, "get_term_cfg"):
        try:
            active_term = event_manager.get_term_cfg("ee_external_force")
        except (KeyError, ValueError):
            active_term = None
        if active_term is not None and all(
            active_term is not term for term in event_terms
        ):
            event_terms.append(active_term)
    if not event_terms:
        return level

    def _lerp_range(r0: tuple[float, float], r1: tuple[float, float], a: float) -> tuple[float, float]:
        return (r0[0] + (r1[0] - r0[0]) * a, r0[1] + (r1[1] - r0[1]) * a)

    alpha = float(level / max_level)

    ranges = {
        "force_x_range": _lerp_range(cfg_event.ee_force_x_range_start, cfg_event.ee_force_x_range_max, alpha),
        "force_y_range": _lerp_range(cfg_event.ee_force_y_range_start, cfg_event.ee_force_y_range_max, alpha),
        "force_z_range": _lerp_range(cfg_event.ee_force_z_range_start, cfg_event.ee_force_z_range_max, alpha),
        "torque_x_range": _lerp_range(cfg_event.ee_torque_x_range_start, cfg_event.ee_torque_x_range_max, alpha),
        "torque_y_range": _lerp_range(cfg_event.ee_torque_y_range_start, cfg_event.ee_torque_y_range_max, alpha),
        "torque_z_range": _lerp_range(cfg_event.ee_torque_z_range_start, cfg_event.ee_torque_z_range_max, alpha),
    }
    for event_term in event_terms:
        event_term.params.update(ranges)

    # 暴露当前等级的 force max 给命令 term（目标力幅值跟外力课程同步）
    # current_max = alpha * |ee_force_x_range_max|（三轴对称，取 x 轴 max 绝对值）
    _force_max_abs = max(abs(cfg_event.ee_force_x_range_max[1]), abs(cfg_event.ee_force_x_range_max[0]))
    env.ee_force_curriculum_current_max = float(alpha * _force_max_abs)

    return level


def ee_external_force_threshold(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """逐级增大 EE 外力范围，并与 target-force 命令幅值保持同步。"""
    cfg_event = _get_ee_force_curriculum_cfg(env)
    if cfg_event is None:
        return torch.tensor(0.0, device=env.device)

    max_level = float(cfg_event.ee_force_curriculum_max_level)
    episode_mult = float(cfg_event.ee_force_curriculum_episode_mult)
    max_episode_length = int(getattr(env, "max_episode_length", 0))
    steps_per_level = int(max_episode_length * episode_mult)
    if max_level <= 0.0 or episode_mult <= 0.0 or steps_per_level <= 0:
        return torch.tensor(0.0, device=env.device)

    level = min(
        max(float(getattr(env, "ee_force_curriculum_level", 0.0)), 0.0),
        max_level,
    )
    step_counter = int(getattr(env, "common_step_counter", 0))
    vel_offset_steps = get_ee_force_curriculum_velocity_offset_steps(
        env, force_episode_mult=episode_mult
    )
    next_threshold = vel_offset_steps + int(
        (level + 1.0) * steps_per_level
    )
    if step_counter > next_threshold and level < max_level:
        level = min(level + 1.0, max_level)

    level = apply_ee_external_force_curriculum_level(env, level)

    return torch.tensor(level, device=env.device)

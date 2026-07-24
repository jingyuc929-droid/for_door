# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

import rl_sim_env.tasks.manager_based.common.terrain.trimesh.mesh_terrains as mesh_terrains
from isaaclab.utils import configclass

from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg
"""
Different trimesh terrain configurations.
"""


@configclass
class MeshObjTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a plane mesh terrain."""

    function = mesh_terrains.obj_terrain

    obj_path: str = MISSING

    spawn_origin_path: str = MISSING


@configclass
class CachedTerrainGenCfg(MeshObjTerrainCfg):
    """Configuration for a plane mesh terrain."""

    function = mesh_terrains.cached_terrain_gen

    height: float = MISSING

    levels: float = MISSING

    include_overhang: bool = MISSING

    task_descriptor: str = MISSING


@configclass
class TerrainGenCfg(MeshObjTerrainCfg):
    """Configuration for a plane mesh terrain."""

    function = mesh_terrains.terrain_gen

    height: float = MISSING

    levels: float = MISSING

    include_overhang: bool = MISSING

    terrain_styles: list = MISSING

    yaml_path: str = (MISSING,)

    spawn_origin_path: str = MISSING

    python_script: str = MISSING


@configclass
class MeshStonesEverywhereTerrainCfg(SubTerrainBaseCfg):
    """
    A terrain with stones everywhere
    """

    function = mesh_terrains.stones_everywhere_terrain

    # stone gap width
    w_gap: tuple[float, float] = MISSING

    # grid square stone size (width)
    w_stone: tuple[float, float] = MISSING

    # the maximum shift, both x and y shift is uniformly sample from [-s_max, s_max]
    s_max: tuple[float, float] = MISSING

    # the maximum height, the height is uniformly sample from [-hmax, h_max], default height is 1.0 m
    h_max: tuple[float, float] = MISSING

    # holes depth
    holes_depth: float = MISSING

    # the platform width
    platform_width: float = MISSING


@configclass
class MeshSquareDaisTerrainCfg(SubTerrainBaseCfg):
    """
    A terrain with a square dais
    """

    function = mesh_terrains.square_dais_terrain

    # the platform width
    platform_width: float = MISSING

    # the dais height range
    dais_height: tuple[float, float] = MISSING

    # the dais width range
    dais_width: tuple[float, float] = MISSING

    # the dais length range
    dais_length: tuple[float, float] = MISSING


@configclass
class MeshSquareHoleTerrainCfg(SubTerrainBaseCfg):
    """
    A square ring terrain suspended above the ground with a central hole.
    """

    function = mesh_terrains.square_hole_terrain

    # the platform width (size of central hole in x/y)
    platform_width: float = MISSING

    # fixed dais (ring) height
    dais_height: float = MISSING

    # the outer dais width range (y dir)
    dais_width: tuple[float, float] = MISSING

    # the outer dais length range (x dir)
    dais_length: tuple[float, float] = MISSING

    # vertical distance from ground (z=0) to the ring's lower surface: (low, high)
    # difficulty in [0,1] interpolates from high -> low (harder -> lower)
    hole_height: tuple[float, float] = MISSING


@configclass
class MeshBalanceBeamsTerrainCfg(SubTerrainBaseCfg):
    """
    A terrain with balance-beams
    """

    # balance beams terrain function
    function = mesh_terrains.balance_beams_terrain

    # the platform width
    platform_width: float = MISSING

    # the height offset
    h_offset: tuple[float, float] = MISSING

    # stone width
    w_stone: tuple[float, float] = MISSING

    # the gap between two beams
    mid_gap: float = MISSING

    # the longitudinal gap between stones on the same beam (min, max)
    # default keeps previous behavior (no extra gap)
    x_gap: tuple[float, float] = (0.0, 0.0)


@configclass
class MeshSteppingBeamsTerrainCfg(SubTerrainBaseCfg):
    """
    A terrain with stepping-beams
    """

    # stepping beams terrain function
    function = mesh_terrains.stepping_beams_terrain

    # the platform width
    platform_width: float = MISSING

    # the height offset
    h_offset: tuple[float, float] = MISSING

    # stone width
    w_stone: tuple[float, float] = MISSING

    # length of the stepping beams
    l_stone: tuple[float, float] = MISSING

    #  the gap between two beams
    gap: tuple[float, float] = MISSING

    # the yaw angle of the stepping beams
    yaw: tuple[float, float] = MISSING


@configclass
class MeshDiversityBoxTerrainCfg(SubTerrainBaseCfg):
    """
    A terrain with boxes for anymal parkour
    """

    function = mesh_terrains.box_terrain

    # the box width range
    box_width_range: tuple[float, float] = MISSING
    # the box length range
    box_length_range: tuple[float, float] = MISSING
    # the box height range
    box_height_range: tuple[float, float] = MISSING

    # the gap between two boxes
    box_gap_range: tuple[float, float] = None  # type: ignore

    # flag for climbing up (box is set at the origin ) or climb down (box is set near the origin)
    up_or_down: str = None  # type: ignore


@configclass
class MeshPassageTerrainCfg(SubTerrainBaseCfg):
    """
    A terrain with passage
    """

    function = mesh_terrains.passage_terrain

    # the passage width (y dir)
    passage_width: float | tuple[float, float] = MISSING

    # the passage height
    passage_height: float | tuple[float, float] = MISSING

    # the passage length (x dir)
    passage_length: float | tuple[float, float] = MISSING


@configclass
class MeshStructuredTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a structured terrain."""

    function = mesh_terrains.structured_terrain
    terrain_type: Literal["stairs", "inverted_stairs", "obstacles", "walls"] = MISSING


@configclass
class MeshParkourTerrainCfg(SubTerrainBaseCfg):
    """
    A parkour-style terrain with platforms and stepping stones in a pit.

    该配置参考 legged_gym 中的 parkour_terrain，高度场版本被这里的 mesh 版近似为多个 box。
    """

    function = mesh_terrains.parkour_terrain

    # 起始/结束平台长度与高度（沿前进方向 x）
    platform_len: float = MISSING
    platform_height: float = 0.0

    # 中间石块数量
    num_stones: int = 8

    # 相邻石块在 x 方向、y 方向、以及高度偏移的随机范围
    x_range: tuple[float, float] = (1.8, 1.9)
    y_range: tuple[float, float] = (0.0, 0.1)
    z_range: tuple[float, float] = (-0.2, 0.2)

    # 石块长度与宽度（单块尺寸）
    stone_len: float = 1.0
    stone_width: float = 0.6

    # 场地边缘 pad 尺寸（宽度和高度）
    pad_width: float = 0.1
    pad_height: float = 0.5

    # 中间石块的高度幅度 & 最后一块石头的高度幅度和长度
    incline_height: float = 0.1
    last_incline_height: float = 0.6
    last_stone_len: float = 1.6

    # 坑深度范围（difficulty 在 [0,1] 时线性插值）
    pit_depth: tuple[float, float] = (0.5, 1.0)

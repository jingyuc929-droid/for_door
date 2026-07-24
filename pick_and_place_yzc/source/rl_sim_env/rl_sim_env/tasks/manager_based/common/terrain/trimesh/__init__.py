# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
## from https://github.com/UW-Lab/UWLab.git
"""
This sub-module provides methods to create different terrains using the ``trimesh`` library.

In contrast to the height-field representation, the trimesh representation does not
create arbitrarily small triangles. Instead, the terrain is represented as a single
tri-mesh primitive. Thus, this representation is more computationally and memory
efficient than the height-field representation, but it is not as flexible.
"""

from .mesh_terrains_cfg import (
    CachedTerrainGenCfg,
    MeshBalanceBeamsTerrainCfg,
    MeshDiversityBoxTerrainCfg,
    MeshObjTerrainCfg,
    MeshPassageTerrainCfg,
    MeshSteppingBeamsTerrainCfg,
    MeshStonesEverywhereTerrainCfg,
    MeshStructuredTerrainCfg,
    TerrainGenCfg,
    MeshSquareDaisTerrainCfg,
    MeshParkourTerrainCfg,
    MeshSquareHoleTerrainCfg,
)

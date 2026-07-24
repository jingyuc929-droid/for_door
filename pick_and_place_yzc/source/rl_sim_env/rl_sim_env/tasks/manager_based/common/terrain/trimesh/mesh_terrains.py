# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate different terrains using the ``trimesh`` library."""

from __future__ import annotations

import io
import numpy as np
import os
import random
import subprocess
import torch
import trimesh
import yaml
from scipy.spatial.transform import Rotation as R
from typing import TYPE_CHECKING

import requests

"""Extension metadata dictionary parsed from the extension.toml file."""
UWLAB_CLOUD_ASSETS_DIR = "https://uwlab-assets.s3.us-west-004.backblazeb2.com"

from isaaclab.terrains.trimesh.mesh_terrains import inverted_pyramid_stairs_terrain, pyramid_stairs_terrain
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshInvertedPyramidStairsTerrainCfg, MeshPyramidStairsTerrainCfg
from isaaclab.terrains.trimesh.utils import make_border, make_plane

if TYPE_CHECKING:
    from . import mesh_terrains_cfg


def obj_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshObjTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray, np.ndarray] | tuple[list[trimesh.Trimesh], np.ndarray]:
    mesh: trimesh.Trimesh = trimesh.load(cfg.obj_path)  # type: ignore
    mesh: trimesh.Trimesh = trimesh.load(cfg.obj_path)  # type: ignore
    xy_scale = cfg.size / (mesh.bounds[1] - mesh.bounds[0])[:2]
    # set the height scale to the average between length and width scale to preserve as much original shap as possible
    height_scale = (xy_scale[0] + xy_scale[1]) / 2
    xyz_scale = np.array([*xy_scale, height_scale])
    mesh.apply_scale(xyz_scale)
    translation = -mesh.bounds[0]
    mesh.apply_translation(translation)

    extend = mesh.bounds[1] - mesh.bounds[0]
    origin = (*((extend[:2]) / 2), mesh.bounds[1][2] / 2)

    if isinstance(cfg.spawn_origin_path, str):
        spawning_option = np.load(cfg.spawn_origin_path, allow_pickle=True)
        spawning_option *= xyz_scale
        spawning_option += translation
        # insert the center of the terrain as the first indices
        # the rest of the indices represents the spawning locations
        return [mesh], np.insert(spawning_option, 0, origin, axis=0)
    else:
        return [mesh], np.array(origin)


def terrain_gen(
    difficulty: float, cfg: mesh_terrains_cfg.TerrainGenCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray, np.ndarray] | tuple[list[trimesh.Trimesh], np.ndarray]:
    terrain_yaml = {
        "terrain": {
            "shape": [int(cfg.size[0] / 2), int(cfg.size[1] / 2)],
            "height": cfg.height,
            "levels": cfg.levels,
            "include_overhang": cfg.include_overhang,
            "all_terrain_styles": ["stair", "ramp", "box", "platform", "random_box", "perlin", "wall"],
            "terrain_styles": cfg.terrain_styles,
        }
    }
    terrain_style = "_".join(cfg.terrain_styles)
    os.makedirs(os.path.dirname(cfg.yaml_path), exist_ok=True)
    yaml_file_path = cfg.yaml_path.replace(".yaml", f"_{terrain_style}.yaml")
    with open(yaml_file_path, "w") as file:
        yaml.dump(terrain_yaml, file, default_flow_style=False)

    mesh_origin_dir = os.path.dirname(cfg.obj_path)
    mesh_dir = os.path.dirname(mesh_origin_dir)
    # Prepare the command and arguments for the subprocess
    command = [
        "python",
        cfg.python_script,
        "--input_path",
        yaml_file_path,
        "--enable_sdf",
        "--mesh_dir",
        mesh_dir,
        "--mesh_name",
        f"{terrain_style}",
    ]

    # Invoke the subprocess and run the other script
    try:
        result = subprocess.run(command, check=True, capture_output=True)
        print("Subprocess completed successfully!")
        print("Output:", result.stdout.decode())
        print("Errors:", result.stderr.decode())
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with error: {e}")
        print(f"Subprocess output: {e.output.decode()}")
        print(f"Subprocess stderr: {e.stderr.decode()}")

    return obj_terrain(difficulty, cfg)


def cached_terrain_gen(
    difficulty: float, cfg: mesh_terrains_cfg.CachedTerrainGenCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray, np.ndarray] | tuple[list[trimesh.Trimesh], np.ndarray]:
    terrain_type = cfg.task_descriptor
    level = cfg.levels
    height = cfg.height
    overhang = "overhang_yes" if cfg.include_overhang else "overhang_no"
    mesh_id = "mesh_0"
    root_path = f"{UWLAB_CLOUD_ASSETS_DIR}/dataset/terrains/dataset/generated_terrain/{terrain_type}/shape_8/height_{height}/level_{level}/{overhang}/{mesh_id}"

    terrain_mesh_path = os.path.join(root_path, "mesh_terrain.obj")
    spawnfile_path = os.path.join(root_path, "spawnable_locations.npy")

    mesh: trimesh.Trimesh = load_mesh(terrain_mesh_path)
    xy_scale = cfg.size / (mesh.bounds[1] - mesh.bounds[0])[:2]
    # set the height scale to the average between length and width scale to preserve as much original shap as possible
    height_scale = (xy_scale[0] + xy_scale[1]) / 2
    xyz_scale = np.array([*xy_scale, height_scale])
    mesh.apply_scale(xyz_scale)
    translation = -mesh.bounds[0]
    mesh.apply_translation(translation)

    extend = mesh.bounds[1] - mesh.bounds[0]
    origin = (*((extend[:2]) / 2), mesh.bounds[1][2] / 2)

    if isinstance(spawnfile_path, str):
        spawning_option = load_numpy(spawnfile_path)
        spawning_option *= xyz_scale
        spawning_option += translation
        # insert the center of the terrain as the first indices
        # the rest of the indices represents the spawning locations
        return [mesh], np.insert(spawning_option, 0, origin, axis=0)
    else:
        return [mesh], np.array(origin)


def load_mesh(terrain_mesh_path: str) -> trimesh.Trimesh:
    """Load a mesh from a URL or a local file."""
    if terrain_mesh_path.startswith("http"):
        # Load from URL
        response = requests.get(terrain_mesh_path)
        if response.status_code == 200:
            mesh = trimesh.load(io.BytesIO(response.content), file_type="obj")
            return mesh  # type: ignore
            return mesh  # type: ignore
        else:
            raise Exception(f"Failed to load mesh from {terrain_mesh_path}")
    else:
        # Load from local path
        return trimesh.load(terrain_mesh_path)  # type: ignore

        return trimesh.load(terrain_mesh_path)  # type: ignore


def load_numpy(spawnfile_path: str) -> np.ndarray:
    """Load a NumPy array from a URL or a local file."""
    if spawnfile_path.startswith("http"):
        # Load from URL
        response = requests.get(spawnfile_path)
        if response.status_code == 200:
            data = np.load(io.BytesIO(response.content), allow_pickle=True)
            return data
        else:
            raise Exception(f"Failed to load NumPy file from {spawnfile_path}")
    else:
        # Load from local path
        return np.load(spawnfile_path, allow_pickle=True)


def stones_everywhere_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshStonesEverywhereTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    # check to ensure square terrain
    assert cfg.size[0] == cfg.size[1], "The terrain should be square"

    # resolve the terrain configuration based on the difficulty
    gap_width = cfg.w_gap[0] + difficulty * (cfg.w_gap[1] - cfg.w_gap[0])
    stone_width = cfg.w_stone[0] - difficulty * (cfg.w_stone[0] - cfg.w_stone[1])
    s_max = cfg.s_max[0] + difficulty * (cfg.s_max[1] - cfg.s_max[0])
    h_max = cfg.h_max[0] + difficulty * (cfg.h_max[1] - cfg.h_max[0])

    # initialize list of meshes
    meshes_list = list()

    # compute the number of stones in x and y directions
    num_stones_axis = int(cfg.size[0] / (gap_width + stone_width))

    # constants
    ground_height = 1.0  # 底层平地高度为 0（不下沉）
    stone_height = cfg.holes_depth  # 石头的基础高度
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 添加底层平地 - 在所有地形之下提供一个基础平面（降低到-1米）
    ground_plane = make_plane(cfg.size, height=-ground_height, center_zero=False)
    meshes_list.append(ground_plane)

    # generate the border
    border_width = cfg.size[0] - num_stones_axis * (gap_width + stone_width)
    if border_width > 0:
        border_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -ground_height / 2)
        border_inner_size = (cfg.size[0] - border_width, cfg.size[1] - border_width)
        # create border meshes - 边框从底层平地延伸到石头高度
        make_borders = make_border(cfg.size, border_inner_size, ground_height, border_center)
        meshes_list += make_borders
    # create a template grid of the stone height
    grid_dim = [stone_width, stone_width, stone_height]
    grid_position = [0.5 * (stone_width + gap_width), 0.5 * (stone_width + gap_width), -stone_height / 2]
    template_box = trimesh.creation.box(grid_dim, trimesh.transformations.translation_matrix(grid_position))
    # extract vertices and faces
    template_vertices = template_box.vertices  # (8, 3)
    template_faces = template_box.faces

    # repeat the template box vertices to space the terrain(num_boxes_axis**2, 8, 3)
    vertices = torch.tensor(template_vertices, device=device).repeat(num_stones_axis**2, 1, 1)
    # create a meshgrid to offset the vertices
    x = torch.arange(0, num_stones_axis, device=device)
    y = torch.arange(0, num_stones_axis, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    xx = xx.flatten().view(-1, 1)
    yy = yy.flatten().view(-1, 1)
    xx_yy = torch.cat((xx, yy), dim=1)
    # offset the vertices
    offsets = (
        (stone_width + gap_width) * xx_yy
        + border_width / 2
        + (2 * torch.rand(*xx_yy.shape, device=xx_yy.device) - 1) * s_max
    )
    vertices[:, :, :2] += offsets.unsqueeze(1)

    # add noise on height
    num_boxes = len(vertices)
    h_noise = torch.zeros((num_boxes, 3), device=device)
    h_noise[:, 2].uniform_(-h_max, h_max)
    # reshape noise to match the vertices (num_boxes, 4, 3)
    # only top vertices are affected
    vertices_noise = torch.zeros((num_boxes, 4, 3), device=device)
    vertices_noise += h_noise.unsqueeze(1)
    # add height only to the top vertices of the box
    vertices[vertices[:, :, 2] == 0] += vertices_noise.view(-1, 3)
    # move to numpy
    vertices = vertices.reshape(-1, 3).cpu().numpy()

    # create faces for boxes(num_boxes, 12, 3), each box has 6 faces, each face has 2 triangles
    faces = torch.tensor(template_faces, device=device).repeat(num_boxes, 1, 1)
    face_offsets = torch.arange(0, num_boxes, device=device).unsqueeze(1).repeat(1, 12) * 8
    faces += face_offsets.unsqueeze(2)
    faces = faces.view(-1, 3).cpu().numpy()

    # convert to trimesh
    grid_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    meshes_list.append(grid_mesh)

    # add a platform in the center of the terrain that is accessible from all sides
    # 平台从底层平地(-ground_height)延伸到石头最高点(h_max)
    platform_height = ground_height
    dim = (cfg.platform_width, cfg.platform_width, platform_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -ground_height + platform_height / 2)
    box_platform = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(box_platform)

    # specify the origin of the terrain - 机器人生成在平台顶部，高度为h_max
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], h_max])

    return meshes_list, origin


def square_dais_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshSquareDaisTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    # check to ensure square terrain
    assert cfg.size[0] == cfg.size[1], "The terrain should be square"

    # 根据难度计算参数
    dais_height = cfg.dais_height[0] + difficulty * (cfg.dais_height[1] - cfg.dais_height[0])
    dais_width = cfg.dais_width[0] + difficulty * (cfg.dais_width[1] - cfg.dais_width[0])
    dais_length = cfg.dais_length[0] + difficulty * (cfg.dais_length[1] - cfg.dais_length[0])

    meshes_list = list()

    # 添加底层平地 - 上表面高度为0
    ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground_plane)

    # 检查是否需要生成环
    if dais_length < cfg.platform_width or dais_width < cfg.platform_width:
        print(f"Warning: dais_length ({dais_length}) or dais_width ({dais_width}) is smaller than platform_width ({cfg.platform_width}). Skipping dais ring generation.")
    else:
        # 生成方形环
        ring_thickness_x = (dais_length - cfg.platform_width) / 2
        ring_thickness_y = (dais_width - cfg.platform_width) / 2

        # 计算中心位置
        center_x = 0.5 * cfg.size[0]
        center_y = 0.5 * cfg.size[1]

        # 创建4个长方体组成环（在平地上叠加，高度为dais_height）
        # 1. 上边（y方向正向）
        top_dim = (dais_length, ring_thickness_y, dais_height)
        top_pos = (center_x, center_y + cfg.platform_width / 2 + ring_thickness_y / 2, dais_height / 2)
        top_box = trimesh.creation.box(top_dim, trimesh.transformations.translation_matrix(top_pos))
        meshes_list.append(top_box)

        # 2. 下边（y方向负向）
        bottom_dim = (dais_length, ring_thickness_y, dais_height)
        bottom_pos = (center_x, center_y - cfg.platform_width / 2 - ring_thickness_y / 2, dais_height / 2)
        bottom_box = trimesh.creation.box(bottom_dim, trimesh.transformations.translation_matrix(bottom_pos))
        meshes_list.append(bottom_box)

        # 3. 左边（x方向负向）
        left_dim = (ring_thickness_x, cfg.platform_width, dais_height)
        left_pos = (center_x - cfg.platform_width / 2 - ring_thickness_x / 2, center_y, dais_height / 2)
        left_box = trimesh.creation.box(left_dim, trimesh.transformations.translation_matrix(left_pos))
        meshes_list.append(left_box)

        # 4. 右边（x方向正向）
        right_dim = (ring_thickness_x, cfg.platform_width, dais_height)
        right_pos = (center_x + cfg.platform_width / 2 + ring_thickness_x / 2, center_y, dais_height / 2)
        right_box = trimesh.creation.box(right_dim, trimesh.transformations.translation_matrix(right_pos))
        meshes_list.append(right_box)

    # 指定机器人生成位置（中心方台的中心，高度为0）
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])

    return meshes_list, origin


def square_hole_terrain(
    difficulty: float, cfg: "mesh_terrains_cfg.MeshSquareHoleTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """方形“空中环”地形：中心留出 square hole，环整体抬升到空中。

    - x/y 平面上的几何与 ``square_dais_terrain`` 一致；
    - ``dais_height`` 为固定高度；
    - ``hole_height`` 为环下平面到地面 (z=0) 的距离，随 difficulty 变小；
    - 机器人初始位置保持在地面中心 (与 ``square_dais_terrain`` 一致)。
    """
    # 保持与 square_dais_terrain 一致：要求正方形地形
    assert cfg.size[0] == cfg.size[1], "The terrain should be square"

    # 固定台阶高度
    dais_height = cfg.dais_height

    # 根据难度计算环在 x/y 方向的范围
    dais_width = cfg.dais_width[0] + difficulty * (cfg.dais_width[1] - cfg.dais_width[0])
    dais_length = cfg.dais_length[0] + difficulty * (cfg.dais_length[1] - cfg.dais_length[0])

    meshes_list: list[trimesh.Trimesh] = []

    # 地面平板：保持在 z = 0，不动
    ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground_plane)

    # 根据难度从配置中插值得到 hole_height，难度越大，hole_height 越小
    # 约定 hole_height = (low, high)，在 difficulty 从 0->1 时从 high 线性插值到 low
    hole_low, hole_high = cfg.hole_height
    hole_height = hole_high - difficulty * (hole_high - hole_low)

    # 环的下平面相对于地面的高度
    z_offset = hole_height

    # 检查是否需要生成环（与 square_dais_terrain 相同的条件）
    if dais_length < cfg.platform_width or dais_width < cfg.platform_width:
        print(
            f"Warning: dais_length ({dais_length}) or dais_width ({dais_width}) is smaller than "
            f"platform_width ({cfg.platform_width}). Skipping square hole ring generation."
        )
    else:
        # 生成方形环
        ring_thickness_x = (dais_length - cfg.platform_width) / 2
        ring_thickness_y = (dais_width - cfg.platform_width) / 2

        # 计算中心位置
        center_x = 0.5 * cfg.size[0]
        center_y = 0.5 * cfg.size[1]

        # 1. 上边（y 方向正向）
        top_dim = (dais_length, ring_thickness_y, dais_height)
        top_pos = (
            center_x,
            center_y + cfg.platform_width / 2 + ring_thickness_y / 2,
            z_offset + dais_height / 2,
        )
        top_box = trimesh.creation.box(top_dim, trimesh.transformations.translation_matrix(top_pos))
        meshes_list.append(top_box)

        # 2. 下边（y 方向负向）
        bottom_dim = (dais_length, ring_thickness_y, dais_height)
        bottom_pos = (
            center_x,
            center_y - cfg.platform_width / 2 - ring_thickness_y / 2,
            z_offset + dais_height / 2,
        )
        bottom_box = trimesh.creation.box(
            bottom_dim, trimesh.transformations.translation_matrix(bottom_pos)
        )
        meshes_list.append(bottom_box)

        # 3. 左边（x 方向负向）
        left_dim = (ring_thickness_x, cfg.platform_width, dais_height)
        left_pos = (
            center_x - cfg.platform_width / 2 - ring_thickness_x / 2,
            center_y,
            z_offset + dais_height / 2,
        )
        left_box = trimesh.creation.box(left_dim, trimesh.transformations.translation_matrix(left_pos))
        meshes_list.append(left_box)

        # 4. 右边（x 方向正向）
        right_dim = (ring_thickness_x, cfg.platform_width, dais_height)
        right_pos = (
            center_x + cfg.platform_width / 2 + ring_thickness_x / 2,
            center_y,
            z_offset + dais_height / 2,
        )
        right_box = trimesh.creation.box(
            right_dim, trimesh.transformations.translation_matrix(right_pos)
        )
        meshes_list.append(right_box)

    # 机器人初始位置：保持与 square_dais_terrain 一致，生成在地面中心，高度 0
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])

    return meshes_list, origin


def balance_beams_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshBalanceBeamsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    # check to ensure square terrain
    assert cfg.size[0] == cfg.size[1], "The terrain should be square"

    stone_width = cfg.w_stone[0] - difficulty * (cfg.w_stone[0] - cfg.w_stone[1])
    h_offset = cfg.h_offset[0] + difficulty * (cfg.h_offset[1] - cfg.h_offset[0])
    mid_gap = cfg.mid_gap

    meshes_list = list()
    num_stones = int(((cfg.size[0] - 0.25 - cfg.platform_width) / 2 - 1) / stone_width * 2)

    terrain_height = 1
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 添加底层平地 - 在所有地形之下提供一个基础平面
    ground_plane = make_plane(cfg.size, height=-terrain_height, center_zero=False)
    meshes_list.append(ground_plane)

    border_width = (cfg.size[1] - cfg.platform_width) / 2 - 1 - num_stones * stone_width
    if border_width > 0:
        border_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
        border_inner_size = (cfg.size[0] - border_width - 2, cfg.size[1] - border_width - 2)
        # create border meshes
        make_borders = make_border(cfg.size, border_inner_size, terrain_height, border_center)
        meshes_list += make_borders

    grid_dim = [stone_width, stone_width, terrain_height]
    grid_position = [0.5 * stone_width, 0.5 * stone_width, -0.5 * terrain_height]
    template_box = trimesh.creation.box(grid_dim, trimesh.transformations.translation_matrix(grid_position))
    # extract vertices and faces
    template_vertices = template_box.vertices  # (8, 3)
    template_faces = template_box.faces

    # 前进方向的beams
    # repeat the template box vertices to space the terrain(num_stones, 8, 3)
    vertices = torch.tensor(template_vertices, device=device).repeat(num_stones, 1, 1)
    index = torch.arange(0, num_stones, device=device)
    vertices[:, :, 0] += cfg.size[0] / 2 + cfg.platform_width / 2
    vertices[:, :, 0] += (index * stone_width * 0.75).unsqueeze(-1)
    # 交替行在 x 方向错半个石块宽度（使前后交替的步进间距为 0.5×stone_width）
    # vertices[(index % 2) == 1, :, 0] += 0.5 * stone_width
    vertices[(index % 2) == 0, :, 1] += cfg.size[1] / 2 - mid_gap / 2
    vertices[(index % 2) == 1, :, 1] += cfg.size[1] / 2 + mid_gap / 2

    num_boxes = len(vertices)
    h_noise = torch.zeros((num_boxes, 3), device=device)
    h_noise[:, 2].uniform_(-h_offset, h_offset)
    # reshape noise to match the vertices (num_boxes, 4, 3)
    # only top vertices are affected
    vertices_noise = torch.zeros((num_boxes, 4, 3), device=device)
    vertices_noise += h_noise.unsqueeze(1)
    # add height only to the top vertices of the box
    vertices[vertices[:, :, 2] == 0] += vertices_noise.view(-1, 3)
    # move to numpy
    vertices_forward = vertices.reshape(-1, 3).cpu().numpy()

    # create faces for boxes(num_boxes, 12, 3), each box has 6 faces, each face has 2 triangles
    faces = torch.tensor(template_faces, device=device).repeat(num_boxes, 1, 1)
    face_offsets = torch.arange(0, num_boxes, device=device).unsqueeze(1).repeat(1, 12) * 8
    faces += face_offsets.unsqueeze(2)
    faces_forward = faces.view(-1, 3).cpu().numpy()

    # convert to trimesh
    grid_mesh_forward = trimesh.Trimesh(vertices=vertices_forward, faces=faces_forward)
    meshes_list.append(grid_mesh_forward)

    # 后退方向的beams (独立随机参数)
    # 重新计算后退方向的mid_gap (独立随机)
    mid_gap_backward = cfg.mid_gap
    vertices_backward = torch.tensor(template_vertices, device=device).repeat(num_stones, 1, 1)
    index_backward = torch.arange(0, num_stones, device=device)
    # 后退方向在platform的左侧
    vertices_backward[:, :, 0] += cfg.size[0] / 2 - cfg.platform_width / 2
    vertices_backward[:, :, 0] -= ((index_backward + 1) * stone_width * 0.75).unsqueeze(-1)
    # 交替行在 x 方向错半个石块宽度（与右侧对称，向后再移半块）
    # vertices_backward[(index_backward % 2) == 1, :, 0] -= 0.5 * stone_width
    vertices_backward[(index_backward % 2) == 0, :, 1] += cfg.size[1] / 2 - mid_gap_backward / 2
    vertices_backward[(index_backward % 2) == 1, :, 1] += cfg.size[1] / 2 + mid_gap_backward / 2

    # 独立的高度噪声
    h_noise_backward = torch.zeros((num_stones, 3), device=device)
    h_noise_backward[:, 2].uniform_(-h_offset, h_offset)
    vertices_noise_backward = torch.zeros((num_stones, 4, 3), device=device)
    vertices_noise_backward += h_noise_backward.unsqueeze(1)
    vertices_backward[vertices_backward[:, :, 2] == 0] += vertices_noise_backward.view(-1, 3)
    vertices_backward = vertices_backward.reshape(-1, 3).cpu().numpy()

    # create faces for backward boxes
    faces_backward = torch.tensor(template_faces, device=device).repeat(num_stones, 1, 1)
    face_offsets_backward = torch.arange(0, num_stones, device=device).unsqueeze(1).repeat(1, 12) * 8
    faces_backward += face_offsets_backward.unsqueeze(2)
    faces_backward = faces_backward.view(-1, 3).cpu().numpy()

    # convert to trimesh
    grid_mesh_backward = trimesh.Trimesh(vertices=vertices_backward, faces=faces_backward)
    meshes_list.append(grid_mesh_backward)

    # add a platform in the center of the terrain that is accessible from all sides
    dim = (cfg.platform_width, cfg.size[1], terrain_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    box_platform = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(box_platform)

    # specify the origin of the terrain
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0])

    return meshes_list, origin


def stepping_beams_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshSteppingBeamsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    stone_width = cfg.w_stone[0] - difficulty * (cfg.w_stone[0] - cfg.w_stone[1])
    h_offset = cfg.h_offset[0] + difficulty * (cfg.h_offset[1] - cfg.h_offset[0])
    gap_width = cfg.gap[0] + difficulty * (cfg.gap[1] - cfg.gap[0])
    yaw = cfg.yaw[0] + difficulty * (cfg.yaw[1] - cfg.yaw[0])
    assert cfg.yaw[0] < cfg.yaw[1], "The yaw range should be in ascending order(0 means no yaw)"
    low_stone_l = cfg.l_stone[0]
    high_stone_l = cfg.l_stone[1]

    meshes_list = list()
    num_stones = int(((cfg.size[0] - cfg.platform_width) / 2) / (gap_width + stone_width))

    terrain_height = 1

    # 添加底层平地 - 在所有地形之下提供一个基础平面
    ground_plane = make_plane(cfg.size, height=-terrain_height, center_zero=False)
    meshes_list.append(ground_plane)

    border_width = (cfg.size[1] - cfg.platform_width) / 2 - num_stones * (stone_width + gap_width)

    if border_width > 0:
        border_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
        border_inner_size = (cfg.size[0] - border_width - 2, cfg.size[1] - border_width - 2)
        # create border meshes
        make_borders = make_border(cfg.size, border_inner_size, terrain_height, border_center)
        meshes_list += make_borders

    # 前进方向的stepping beams
    # calculate the center of all the stones
    # add random noise to the center
    # add noise to the height
    # create the stones
    for i in range(num_stones):
        transform = np.eye(4)
        grid_dim = [
            stone_width,
            low_stone_l + random.uniform(0, high_stone_l - low_stone_l),
            terrain_height + random.uniform(-h_offset, h_offset),
        ]
        center = [
            cfg.size[0] / 2
            + cfg.platform_width / 2
            + (i + 1) * gap_width
            + (i + 0.5) * stone_width
            + random.uniform(-0.25, 0.25) * gap_width,
            cfg.size[1] / 2 + random.uniform(-0.1, 0.1) * grid_dim[1],
            -terrain_height / 2,
        ]
        transform[0:3, -1] = np.asarray(center)
        # create rotation matrix
        transform[0:3, 0:3] = R.from_euler("z", random.uniform(-yaw, yaw), degrees=True).as_matrix()
        meshes_list.append(trimesh.creation.box(grid_dim, transform))

    # 后退方向的stepping beams (独立随机参数)
    for i in range(num_stones):
        transform = np.eye(4)
        grid_dim = [
            stone_width,
            low_stone_l + random.uniform(0, high_stone_l - low_stone_l),
            terrain_height + random.uniform(-h_offset, h_offset),
        ]
        center = [
            cfg.size[0] / 2
            - cfg.platform_width / 2
            - (i + 1) * gap_width
            - (i + 0.5) * stone_width
            + random.uniform(-0.25, 0.25) * gap_width,
            cfg.size[1] / 2 + random.uniform(-0.1, 0.1) * grid_dim[1],
            -terrain_height / 2,
        ]
        transform[0:3, -1] = np.asarray(center)
        # create rotation matrix (独立随机)
        transform[0:3, 0:3] = R.from_euler("z", random.uniform(-yaw, yaw), degrees=True).as_matrix()
        meshes_list.append(trimesh.creation.box(grid_dim, transform))

    # add a platform in the center of the terrain that is accessible from all sides
    dim = (cfg.platform_width, cfg.size[1], terrain_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    box_platform = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(box_platform)

    # specify the origin of the terrain
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0])

    return meshes_list, origin


def box_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshDiversityBoxTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    #
    box_width = cfg.box_width_range[1] - difficulty * (cfg.box_width_range[1] - cfg.box_width_range[0])
    box_length = cfg.box_length_range[1] - difficulty * (cfg.box_length_range[1] - cfg.box_length_range[0])
    box_height = cfg.box_height_range[0] + difficulty * (cfg.box_height_range[1] - cfg.box_height_range[0])
    meshes_list = []
    terrain_height = 1.0
    middle_height = 0.0
    # check if box_gap_range is a tuple
    if isinstance(cfg.box_gap_range, tuple):
        # Task of jumping over neighboring boxes
        gap_width = cfg.box_gap_range[0] + difficulty * (cfg.box_gap_range[1] - cfg.box_gap_range[0])
        # generate the box at the origin
        box_dim = (box_width, box_length, box_height + terrain_height)
        pos = (cfg.size[0] / 2, cfg.size[1] / 2, -terrain_height / 2 + box_height / 2)
        box = trimesh.creation.box(box_dim, trimesh.transformations.translation_matrix(pos))
        meshes_list.append(box)
        # generate the neighboring boxes
        box_dim = (box_width, box_length, box_height + terrain_height)
        offset_x = box_width / 2 + box_width / 2 + gap_width
        pos = (cfg.size[0] / 2 + offset_x, cfg.size[1] / 2, -terrain_height / 2 + box_height / 2)
        box = trimesh.creation.box(box_dim, trimesh.transformations.translation_matrix(pos))
        meshes_list.append(box)
        middle_height = box_height
    elif cfg.box_gap_range is None:
        # Task for climbing up or down boxes
        if cfg.up_or_down == "up":
            # for climbing up
            box_dim = (box_width, box_length, box_height + terrain_height)
            offset_x = box_width
            pos = (cfg.size[0] / 2 + offset_x, cfg.size[1] / 2, -terrain_height / 2 + box_height / 2)
            box = trimesh.creation.box(box_dim, trimesh.transformations.translation_matrix(pos))
            meshes_list.append(box)
            middle_height = 0.0
        elif cfg.up_or_down == "down":
            # for climbing down
            box_dim = (box_width, box_length, box_height + terrain_height)
            pos = (cfg.size[0] / 2, cfg.size[1] / 2, -terrain_height / 2 + box_height / 2)
            box = trimesh.creation.box(box_dim, trimesh.transformations.translation_matrix(pos))
            meshes_list.append(box)
            middle_height = box_height
        else:
            raise ValueError("up_or_down should be either 'up' or 'down'")
    else:
        raise ValueError("box_gap_range should be a tuple or None")

    # generate the ground
    pos = (cfg.size[0] / 2, cfg.size[1] / 2, -terrain_height / 2)
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    ground = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground)

    # specify the origin of the terrain
    origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, middle_height])

    return meshes_list, origin


def passage_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshPassageTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    if isinstance(cfg.passage_width, tuple):
        width = cfg.passage_width[1] - difficulty * (cfg.passage_width[1] - cfg.passage_width[0])
    elif isinstance(cfg.passage_width, float):
        width = cfg.passage_width
    else:
        raise ValueError("passage_width should be a tuple or a float")
    if isinstance(cfg.passage_length, tuple):
        length = cfg.passage_length[0] + difficulty * (cfg.passage_length[1] - cfg.passage_length[0])
    elif isinstance(cfg.passage_length, float):
        length = cfg.passage_length
    else:
        raise ValueError("passage_length should be a tuple or a float")
    if isinstance(cfg.passage_height, tuple):
        height = cfg.passage_height[1] - difficulty * (cfg.passage_height[1] - cfg.passage_height[0])
    elif isinstance(cfg.passage_height, float):
        height = cfg.passage_height
    else:
        raise ValueError("passage_height should be a tuple or a float")
    # generate the passage
    meshes_list = []
    terrain_height = 1.0
    offset_x = 1.0
    # four legs of the passage
    dim = (0.05 + np.random.uniform(0.0, 0.1), 0.05 + np.random.uniform(0.0, 0.1), terrain_height + height)
    pos1 = (offset_x + cfg.size[0] / 2 - length / 2, cfg.size[1] / 2 - width / 2, -terrain_height / 2 + height / 2)
    box1 = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos1))
    meshes_list.append(box1)
    pos2 = (offset_x + cfg.size[0] / 2 - length / 2, cfg.size[1] / 2 + width / 2, -terrain_height / 2 + height / 2)
    box2 = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos2))
    meshes_list.append(box2)
    pos3 = (offset_x + cfg.size[0] / 2 + length / 2, cfg.size[1] / 2 - width / 2, -terrain_height / 2 + height / 2)
    box3 = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos3))
    meshes_list.append(box3)
    pos4 = (offset_x + cfg.size[0] / 2 + length / 2, cfg.size[1] / 2 + width / 2, -terrain_height / 2 + height / 2)
    box4 = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos4))
    meshes_list.append(box4)
    # top of the passage
    dim = (length + dim[0], width + dim[1], 0.05 + np.random.uniform(0, 0.1))
    pos = (offset_x + cfg.size[0] / 2, cfg.size[1] / 2, dim[2] / 2 + height)
    top = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(top)
    # ground
    pos = (cfg.size[0] / 2, cfg.size[1] / 2, -terrain_height / 2)
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    ground = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground)

    # specify the origin of the terrain
    origin = np.array([cfg.size[0] / 2 - 1.0, cfg.size[1] / 2, 0.0])

    return meshes_list, origin


def structured_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshStructuredTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    mesh_list = []
    terrain = cfg.terrain_type
    # generate the terrain
    if terrain == "obstacles":
        origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0])
        for i in range(12):
            if i < 8:
                length = random.uniform(0.2, 2.0)
                width = random.uniform(0.2, 2.0)
                height = random.uniform(0.08, 0.25)
            else:
                length = random.uniform(0.2, 1.0)
                width = random.uniform(0.2, 1.0)
                height = 3.0
            center = (
                cfg.size[0] / 2 + random.uniform(1, cfg.size[0] / 2) * (-1) ** (random.randint(1, 2)),
                cfg.size[1] / 2 + random.uniform(1, cfg.size[0] / 2) * (-1) ** (random.randint(1, 2)),
                height / 2,
            )
            transform = np.eye(4)
            transform[0:3, -1] = np.asarray(center)
            # create the box
            dims = (length, width, height)
            mesh = trimesh.creation.box(dims, transform=transform)
            mesh_list.append(mesh)
        # add walls
        if random.uniform(0, 1) > 0.1:
            center_pts = [(0, 0, 0), (cfg.size[0], 0, 0), (0, cfg.size[1], 0), (cfg.size[0], cfg.size[1], 0)]
            for i, center in enumerate(center_pts):
                if random.uniform(0, 1) > 0.5:
                    continue
                length = cfg.size[0] * random.uniform(0.2, 0.4)
                width = cfg.size[1] * random.uniform(0.2, 0.4)
                height = 6.0
                transform = np.eye(4)
                c = (center[0] + (-1) ** i * length / 2, center[1] + (-1) ** (i // 2) * width / 2, center[2])
                transform[0:3, -1] = np.asarray(c)
                # create the box
                dims = (length, width, height)
                mesh = trimesh.creation.box(dims, transform=transform)
                mesh_list.append(mesh)
        # add plane
        ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
        mesh_list.append(ground_plane)

    elif terrain == "stairs":
        step_width = random.uniform(0.2, 0.5)
        _mesh_list, origin = pyramid_stairs_terrain(
            difficulty,
            MeshPyramidStairsTerrainCfg(
                size=cfg.size,
                border_width=1.0,
                step_height_range=(0.08, 0.20),
                step_width=step_width,
                platform_width=2.0,
            ),
        )
        mesh_list += _mesh_list
        # add walls
        if random.uniform(0, 1) > 0.05:
            center_pts = [(0, 0, 0), (cfg.size[0], 0, 0), (0, cfg.size[1], 0), (cfg.size[0], cfg.size[1], 0)]
            for i, center in enumerate(center_pts):
                if random.uniform(0, 1) > 0.75:
                    continue
                length = cfg.size[0] * random.uniform(0.3, 0.4)
                width = cfg.size[1] * random.uniform(0.3, 0.4)
                height = 6.0
                transform = np.eye(4)
                c = (center[0] + (-1) ** i * length / 2, center[1] + (-1) ** (i // 2) * width / 2, center[2])
                transform[0:3, -1] = np.asarray(c)
                # create the box
                dims = (length, width, height)
                mesh = trimesh.creation.box(dims, transform=transform)
                mesh_list.append(mesh)
    elif terrain == "inverted_stairs":
        step_width = random.uniform(0.2, 0.5)
        # inverted prymaid
        _mesh_list, origin = inverted_pyramid_stairs_terrain(
            difficulty,
            MeshInvertedPyramidStairsTerrainCfg(
                size=cfg.size,
                border_width=1.0,
                step_height_range=(0.08, 0.20),
                step_width=step_width,
                platform_width=2.0,
            ),
        )
        mesh_list += _mesh_list
        # add walls
        if random.uniform(0, 1) > 0.05:
            center_pts = [(0, 0, 0), (cfg.size[0], 0, 0), (0, cfg.size[1], 0), (cfg.size[0], cfg.size[1], 0)]
            for i, center in enumerate(center_pts):
                if random.uniform(0, 1) > 0.75:
                    continue
                length = cfg.size[0] * random.uniform(0.3, 0.4)
                width = cfg.size[1] * random.uniform(0.3, 0.4)
                height = 6.0
                transform = np.eye(4)
                c = (center[0] + (-1) ** i * length / 2, center[1] + (-1) ** (i // 2) * width / 2, center[2])
                transform[0:3, -1] = np.asarray(c)
                # create the box
                dims = (length, width, height)
                mesh = trimesh.creation.box(dims, transform=transform)
                mesh_list.append(mesh)
    elif terrain == "walls":
        origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0])
        # add walls
        center_pts = [(0, 0, 0), (cfg.size[0], 0, 0), (0, cfg.size[1], 0), (cfg.size[0], cfg.size[1], 0)]
        for i, center in enumerate(center_pts):
            if random.uniform(0, 1) > 0.75:
                continue
            length = cfg.size[0] * random.uniform(0.3, 0.4)
            width = cfg.size[1] * random.uniform(0.3, 0.4)
            height = 6.0
            transform = np.eye(4)
            c = (center[0] + (-1) ** i * length / 2, center[1] + (-1) ** (i // 2) * width / 2, center[2])
            transform[0:3, -1] = np.asarray(c)
            # create the box
            dims = (length, width, height)
            mesh = trimesh.creation.box(dims, transform=transform)
            mesh_list.append(mesh)
        # add plane
        ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
        mesh_list.append(ground_plane)
    else:
        raise ValueError(f"terrain_type {terrain} is not supported")
    # update the origin in a free space
    return mesh_list, origin


def parkour_terrain(
    difficulty: float, cfg: "mesh_terrains_cfg.MeshParkourTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Parkour-style terrain inspired by legged_gym's height-field implementation.

    这里使用一系列 box mesh 近似出：
    - 起始与结束平台；
    - 坑底平地；
    - 若干左右交替分布的“石块”台阶（高度随 difficulty 与随机数变化）；
    - 场地四周的安全踏板（pad）。
    """
    meshes_list: list[trimesh.Trimesh] = []

    # 1. 坑底平地：difficulty 越大，坑越深
    pit_min, pit_max = cfg.pit_depth
    pit_depth = pit_min + difficulty * (pit_max - pit_min)
    ground_plane = make_plane(cfg.size, height=-pit_depth, center_zero=False)
    meshes_list.append(ground_plane)

    size_x, size_y = cfg.size

    # 2. 起始平台（位于 x 方向起点附近，覆盖整个 y 宽度）
    platform_len = cfg.platform_len
    platform_height = cfg.platform_height

    start_dim = (platform_len, size_y, pit_depth + platform_height)
    start_center = (platform_len / 2.0, size_y / 2.0, platform_height - pit_depth / 2.0)
    start_platform = trimesh.creation.box(start_dim, trimesh.transformations.translation_matrix(start_center))
    meshes_list.append(start_platform)

    # 3. 生成中间的“石块”台阶
    num_stones = cfg.num_stones
    x_min, x_max = cfg.x_range
    y_min, y_max = cfg.y_range
    z_min, z_max = cfg.z_range

    stone_len = cfg.stone_len
    stone_width = cfg.stone_width

    # difficulty 只通过坑深度体现，这里保留原配置的高度幅度
    incline_height = cfg.incline_height
    last_incline_height = cfg.last_incline_height
    last_stone_len = cfg.last_stone_len

    current_x = platform_len
    mid_y = size_y / 2.0

    # 基础高度偏移
    base_z = np.random.uniform(z_min, z_max)

    # 左右交替
    left_right_flag = np.random.randint(0, 2)

    for i in range(num_stones):
        # x 方向步进距离
        step_x = np.random.uniform(x_min, x_max)
        current_x += step_x

        # 决定当前石块在 y 方向偏左还是偏右
        pos_neg = 1.0 if left_right_flag == 1 else -1.0
        delta_y = np.random.uniform(y_min, y_max)
        center_y = mid_y + pos_neg * delta_y

        # 最后一块石头使用独立的长度和高度幅度
        if i == num_stones - 1:
            this_len = last_stone_len
            local_height = base_z + pos_neg * last_incline_height
        else:
            this_len = stone_len
            local_height = base_z + pos_neg * incline_height

        # 石块从坑底(-pit_depth)抬升到 local_height + platform_height
        top_height = platform_height + local_height
        stone_height = pit_depth + top_height
        center_z = top_height - pit_depth / 2.0

        stone_dim = (this_len, stone_width, stone_height)
        stone_center = (current_x, center_y, center_z)
        stone_box = trimesh.creation.box(stone_dim, trimesh.transformations.translation_matrix(stone_center))
        meshes_list.append(stone_box)

        # 交替左右
        left_right_flag = 1 - left_right_flag

    # 4. 结束平台：与起始平台高度一致，放在最后一块石头之后
    final_step = 2.0 * np.random.uniform(x_min, x_max)
    final_center_x = current_x + final_step
    # 保证结束平台不会超出整体 size 太多（按需要裁剪）
    final_center_x = min(final_center_x, size_x - platform_len / 2.0)

    end_dim = (platform_len, size_y, pit_depth + platform_height)
    end_center = (final_center_x, size_y / 2.0, platform_height - pit_depth / 2.0)
    end_platform = trimesh.creation.box(end_dim, trimesh.transformations.translation_matrix(end_center))
    meshes_list.append(end_platform)

    # 5. 场地四周的 pad 边缘（高度略高于 0）
    pad_width = cfg.pad_width
    pad_height = cfg.pad_height
    if pad_width > 0.0 and pad_height > 0.0:
        pad_z_top = pad_height
        pad_total_height = pit_depth + pad_z_top
        pad_center_z = pad_z_top - pit_depth / 2.0

        # 下边缘（y 接近 0）
        bottom_dim = (size_x, pad_width, pad_total_height)
        bottom_center = (size_x / 2.0, pad_width / 2.0, pad_center_z)
        meshes_list.append(trimesh.creation.box(bottom_dim, trimesh.transformations.translation_matrix(bottom_center)))

        # 上边缘（y 接近 size_y）
        top_dim = (size_x, pad_width, pad_total_height)
        top_center = (size_x / 2.0, size_y - pad_width / 2.0, pad_center_z)
        meshes_list.append(trimesh.creation.box(top_dim, trimesh.transformations.translation_matrix(top_center)))

        # 左边缘（x 接近 0）
        left_dim = (pad_width, size_y, pad_total_height)
        left_center = (pad_width / 2.0, size_y / 2.0, pad_center_z)
        meshes_list.append(trimesh.creation.box(left_dim, trimesh.transformations.translation_matrix(left_center)))

        # 右边缘（x 接近 size_x）
        right_dim = (pad_width, size_y, pad_total_height)
        right_center = (size_x - pad_width / 2.0, size_y / 2.0, pad_center_z)
        meshes_list.append(trimesh.creation.box(right_dim, trimesh.transformations.translation_matrix(right_center)))

    # 起点生成在起始平台中间
    origin = np.array([platform_len / 2.0, size_y / 2.0, platform_height])

    return meshes_list, origin

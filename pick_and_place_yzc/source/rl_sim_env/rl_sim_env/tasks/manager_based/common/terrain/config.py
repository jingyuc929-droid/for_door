# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import numpy as np
import scipy.interpolate as interpolate
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.height_field import hf_terrains_cfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from rl_sim_env.tasks.manager_based.common.terrain.trimesh import MeshBalanceBeamsTerrainCfg, MeshSteppingBeamsTerrainCfg, MeshStonesEverywhereTerrainCfg
from rl_sim_env.tasks.manager_based.common.terrain.trimesh import MeshDiversityBoxTerrainCfg, MeshPassageTerrainCfg, MeshStructuredTerrainCfg
from rl_sim_env.tasks.manager_based.common.terrain.trimesh import MeshSquareDaisTerrainCfg, MeshSquareHoleTerrainCfg
from rl_sim_env.tasks.manager_based.common.terrain.trimesh import MeshParkourTerrainCfg
from rl_sim_env.tasks.manager_based.common.terrain.terrain_cfg import MultiPrimTerrainImporterCfg


@height_field_to_mesh
def random_uniform_terrain_difficulty(difficulty: float, cfg: hf_terrains_cfg.HfRandomUniformTerrainCfg) -> np.ndarray:
    """Generate a terrain whose噪声幅度随 difficulty 变化，且始终保持上下起伏，不会直接成凸台。

    逻辑：
      - difficulty=0 → 完全平坦，所有高度都为 0。
      - 0<difficulty≤1 → 在 [-max_h, +max_h] 区间里做均匀随机采样，其中 max_h = cfg.noise_range[1] * difficulty。
        这样即便 difficulty 很小，也会有正负两个方向的起伏。
      - 如果计算后离散化得到的 height_min == height_max == 0（即幅度太小），则强制让 height_min=-1，height_max=+1，
        保证至少产生一点正负波动，而不是全部为 0。
    """

    # 1. 校验 downsampled_scale
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale
    elif cfg.downsampled_scale < cfg.horizontal_scale:
        raise ValueError(
            f"Downsampled scale must be ≥ horizontal scale: {cfg.downsampled_scale} < {cfg.horizontal_scale}."
        )

    # 2. 强制把 difficulty 截断在 [0,1]
    difficulty = float(np.clip(difficulty, 0.0, 1.0))

    # 3. 当 difficulty = 0 时，直接返回全 0（完全平坦）
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    if difficulty <= 0.0:
        return np.zeros((width_pixels, length_pixels), dtype=np.int16)

    # 4. 计算“连续高度上限” max_h = cfg.noise_range[1] * difficulty
    #    只使用 noise_range[1] 作为最大正向高度；负向高度对称
    real_max = cfg.noise_range[1]
    max_h_continuous = real_max * difficulty
    min_h_continuous = -max_h_continuous  # 对称取反

    # 5. 离散化到“高度索引”空间
    #    vertical_scale = 几米 对应 1 个离散层
    height_min = int(np.floor(min_h_continuous / cfg.vertical_scale))
    height_max = int(np.ceil(max_h_continuous / cfg.vertical_scale))
    # 如果两者计算后反过来了，就交换
    if height_min > height_max:
        height_min, height_max = height_max, height_min

    # 6. 计算“离散步长”——高度索引之间的差距
    height_step = int(np.round(cfg.noise_step / cfg.vertical_scale))
    if height_step < 1:
        raise ValueError(f"noise_step ({cfg.noise_step}) must be at least vertical_scale ({cfg.vertical_scale}).")

    # 7. 如果离散化后幅度太小（height_min == height_max == 0），强制扩展为 [-1, +1]
    #    这样就不会全部采到 0，起码会有 -1, +1 两个可能
    if height_min == 0 and height_max == 0:
        height_min = -1
        height_max = +1

    # 8. 计算横向下采样与全分辨率网格大小
    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)

    # 9. 构造离散索引区间 [height_min, height_max]，步长为 height_step
    #    例如 height_min=-3, height_max=3, height_step=2 → [-3, -1, 1, 3]
    height_range = np.arange(height_min, height_max + 1, height_step, dtype=np.int32)
    if height_range.size == 0:
        # 极端保护：如果运算出现意外，比如 height_step 非常大，就至少保留 [-1, 1]
        height_range = np.array([-1, 1], dtype=np.int32)

    # 10. 在下采样网格上做均匀随机采样（产生离散高度索引）
    #     形状是 (width_downsampled, length_downsampled)
    height_field_downsampled = np.random.choice(height_range, size=(width_downsampled, length_downsampled))

    # 11. 插值：把下采样网格扩展到全分辨率
    x_ds = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y_ds = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)
    interp_func = interpolate.RectBivariateSpline(x_ds, y_ds, height_field_downsampled)

    x_full = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_full = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_full = interp_func(x_full, y_full)

    # 12. 四舍五入并转为 int16
    z_int = np.rint(z_full).astype(np.int16)
    return z_int


@configclass
class HfRandomUniformTerrainDifficultyCfg(terrain_gen.HfTerrainBaseCfg):
    """Configuration for a random uniform height field terrain."""

    function = random_uniform_terrain_difficulty

    noise_range: tuple[float, float] = MISSING
    """The minimum and maximum height noise (i.e. along z) of the terrain (in m)."""
    noise_step: float = MISSING
    """The minimum height (in m) change between two points."""
    downsampled_scale: float | None = None
    """The distance between two randomly sampled points on the terrain. Defaults to None,
    in which case the :obj:`horizontal scale` is used.

    The heights are sampled at this resolution and interpolation is performed for intermediate points.
    This must be larger than or equal to the :obj:`horizontal scale`.
    """


PLANE_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=1.0,
        ),
    },
)

# ROUGH_TERRAINS_CFG2d4 = terrain_gen.TerrainGeneratorCfg(
#     size=(10.0, 10.0),
#     border_width=60.0,
#     num_rows=10,
#     num_cols=20,
#     horizontal_scale=0.1,
#     vertical_scale=0.005,
#     slope_threshold=0.75,
#     use_cache=False,
#     curriculum=True,
#     sub_terrains={
#         "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
#             proportion=0.001,
#             step_height_range=(0.05, 0.23),
#             step_width=0.26,
#             platform_width=3.0,
#             border_width=1.0,
#             holes=False,
#         ),  # 测试过，无法正常平地运动
#         "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
#             proportion=0.001,
#             step_height_range=(0.05, 0.23),
#             step_width=0.26,
#             platform_width=3.0,
#             border_width=1.0,
#             holes=False,
#         ),
#         "boxes": terrain_gen.MeshRandomGridTerrainCfg(
#             proportion=0.001, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
#         ),  # 在测试
#         "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
#             proportion=0.001, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.3
#         ),
#         "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
#             proportion=0.1, slope_range=(0.0, 0.65), platform_width=2.0, border_width=0.3
#         ),  # 测试结果正常
#         "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
#             proportion=0.2, slope_range=(0.0, 0.65), platform_width=2.0, border_width=0.3
#         ),
#         "plane_run": terrain_gen.MeshPlaneTerrainCfg(
#             proportion=0.696,
#         ),  # 测试结果正常
#     },
# )

ROUGH_TERRAINS_CFG2d4 = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=60.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),  # 测试过，无法正常平地运动
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.02, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),  # 在测试
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.03, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.3
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.65), platform_width=2.0, border_width=0.3
        ),  # 测试结果正常
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.65), platform_width=2.0, border_width=0.3
        ),
        "plane_run": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.15,
        ),  # 测试结果正常
    },
)

# ----------------------------
# Locomotion (2d4) rough-only: 仅保留 random_rough，多 tile，便于 env 分散重置
# ----------------------------
ROUGH_ONLY_TERRAINS_CFG2d4 = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=60.0,
    # 多块 tile：避免所有 env 都落在同一块 rough 上
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    # rough-only 不做 curriculum：直接在多 tile 上分布
    curriculum=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.3
        ),
    },
)


ROUGH_TERRAINS_CFGex1 = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=60.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.26),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),  # 测试过，无法正常平地运动
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.26),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.02, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),  # 在测试
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.03, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.3
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.65), platform_width=2.0, border_width=0.3
        ),  # 测试结果正常
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.65), platform_width=2.0, border_width=0.3
        ),
        "plane_run": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.15,
        ),  # 测试结果正常
    },
)


ROUGH_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=60.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),  # 测试过，无法正常平地运动
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.23),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.05, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),  # 在测试
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.05, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.3
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.3
        ),  # 测试结果正常
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.3
        ),
        "plane_run": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.3,
        ),  # 测试结果正常
    },
)

MARG_ROUGH_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=60.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.23),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),  # 测试过，正常平地运动
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.23),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "square_dais": MeshSquareDaisTerrainCfg(
            proportion=0.1, platform_width=4.0, dais_height=(0.1, 0.8), dais_width=(7, 9), dais_length=(7, 9),
        ),  # 在测试
        "gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.1, gap_width_range=(0.05, 0.8), platform_width=4.0,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.3
        ),  # 测试结果正常
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.3
        ),
        "plane_run": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.1,
        ),  # 测试结果正常
        "stones_everywhere": MeshStonesEverywhereTerrainCfg(
            proportion=0.1,
            w_gap=(0.05, 0.15),
            w_stone=(0.45, 0.25),
            s_max=(0.10, 0.23),
            h_max=(0.00, 0.1),
            platform_width=2.0,
            holes_depth=0.05,
        ),
        "hf_stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.1,
            platform_width=1.0,
            stone_height_max=0.05,
            stone_width_range=(0.4, 0.4),
            stone_distance_range=(0.05, 0.05),
            horizontal_scale=0.1,
        ),
        # 'balance_beams': MeshBalanceBeamsTerrainCfg(
        #     proportion=0.1, platform_width=2.0, h_offset=(0.04, 0.1), w_stone=(0.45, 0.2), mid_gap=0.5, x_gap=(0.05, 0.3),
        # ),
        "stepping_beams": MeshSteppingBeamsTerrainCfg(
            proportion=0.1,
            platform_width=2.0,
            h_offset=(0.04, 0.1),
            w_stone=(0.3, 0.1),
            l_stone=(7.0, 7.0),
            gap=(0.05, 0.3),
            yaw=(0.0, 10.0),
        ),
    },
)

PIE_ROUGH_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=60.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.23),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),  # 测试过，正常平地运动
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.23),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "square_dais": MeshSquareDaisTerrainCfg(
            proportion=0.2, platform_width=4.0, dais_height=(0.1, 0.8), dais_width=(7, 11), dais_length=(7, 11),
        ),  # 在测试
        "gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.2, gap_width_range=(0.05, 1.0), platform_width=4.0,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.3
        ),  # 测试结果正常
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.3
        ),
        "plane_run": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.2,
        ),  # 测试结果正常
        # 'stones_everywhere': MeshStonesEverywhereTerrainCfg(
        #     proportion=0.1, w_gap=(0.05, 0.15), w_stone=(0.45, 0.25), s_max=(0.10, 0.23), h_max=(0.00, 0.1), platform_width=2.0, holes_depth=0.05,
        # ),
        # 'parkour': MeshParkourTerrainCfg(
        #     proportion=0.1, platform_len=1.5, platform_height=0.0, num_stones=8, x_range=(1.8, 1.9), y_range=(0.2, 0.4), z_range=(-0.05, 0.05), stone_len=1.0, stone_width=0.6, pad_width=0.1, pad_height=0.5, incline_height=0.1, last_incline_height=0.6, last_stone_len=1.6, pit_depth=(0.5, 1.0),
        # ),
        # "stepping_beams": MeshSteppingBeamsTerrainCfg(
        #     proportion=0.1,
        #     platform_width=2.0,
        #     h_offset=(0.04, 0.1),
        #     w_stone=(0.3, 0.1),
        #     l_stone=(7.0, 7.0),
        #     gap=(0.10, 0.4),
        #     yaw=(0.0, 5.0),
        # ),
        # "square_hole": MeshSquareHoleTerrainCfg(
        #     proportion=0.1,
        #     platform_width=4.0,
        #     dais_height=1.5,
        #     dais_width=(7, 11),
        #     dais_length=(7, 11),
        #     hole_height=(0.25, 0.6),
        # ),
    },
)

ROUGH_TERRAINS_CFG_HARD = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=60.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.1, grid_width=0.45, grid_height_range=(0.025, 0.1), platform_width=2.0
        # ),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        # ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "mesh_gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.15,
            gap_width_range=(0.05, 0.9),
            platform_width=2.5,
        ),
        "mesh_pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.15,
            pit_depth_range=(0.05, 0.9),
            platform_width=2.5,
            double_pit=True,
        ),
        "mesh_box": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.1,
            box_height_range=(0.05, 0.9),
            platform_width=2.5,
            double_box=True,
        ),
    },
    # sub_terrains={
    #     "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
    #         proportion=0.15,
    #         step_height_range=(0.05, 0.23),
    #         step_width=0.26,
    #         platform_width=3.0,
    #         border_width=1.0,
    #         holes=False,
    #     ),
    #     "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
    #         proportion=0.2,
    #         step_height_range=(0.05, 0.23),
    #         step_width=0.26,
    #         platform_width=3.0,
    #         border_width=1.0,
    #         holes=False,
    #     ),
    #     "boxes": terrain_gen.MeshRandomGridTerrainCfg(
    #         proportion=0.15, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
    #     ),
    #     "random_rough": HfRandomUniformTerrainDifficultyCfg(
    #         proportion=0.15, noise_range=(0.01, 0.06), noise_step=0.02, border_width=0.25
    #     ),
    #     "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
    #         proportion=0.1, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.25
    #     ),
    #     "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
    #         proportion=0.1, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.25
    #     ),
    #     "plane_run": terrain_gen.MeshPlaneTerrainCfg(
    #         proportion=0.15,
    #     ),
    #     # "plane_yaw": terrain_gen.MeshPlaneTerrainCfg(
    #     #     proportion=0.05,
    #     # ),
    #     # "plane_stand": terrain_gen.MeshPlaneTerrainCfg(
    #     #     proportion=0.05,
    #     # ),
    # },
)

mesh_plane = terrain_gen.MeshPlaneTerrainCfg(
    proportion=0.05,
)

# 2. 金字塔阶梯
mesh_pyramid = terrain_gen.MeshPyramidStairsTerrainCfg(
    proportion=0.05,
    step_height_range=(0.1, 0.3),
    step_width=0.5,
    platform_width=1.0,
    border_width=0.2,
    holes=False,
)

# 3. 反向金字塔阶梯
mesh_pyramid_inv = terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
    proportion=0.05,
    step_height_range=(0.1, 0.3),
    step_width=0.5,
    platform_width=1.0,
    border_width=0.2,
    holes=False,
)

# 4. 随机网格
mesh_random_grid = terrain_gen.MeshRandomGridTerrainCfg(
    proportion=0.05,
    grid_width=0.4,
    grid_height_range=(0.05, 0.2),
    platform_width=1.0,
    holes=False,
)

# 5. 铁轨地形
mesh_rails = terrain_gen.MeshRailsTerrainCfg(
    proportion=0.05,
    rail_thickness_range=(0.05, 0.1),
    rail_height_range=(0.1, 0.2),
    platform_width=1.0,
)

# 6. 坑洞地形
mesh_pit = terrain_gen.MeshPitTerrainCfg(
    proportion=0.05,
    pit_depth_range=(0.2, 0.5),
    platform_width=1.0,
    double_pit=False,
)

# 7. 盒状凸台
mesh_box = terrain_gen.MeshBoxTerrainCfg(
    proportion=0.05,
    box_height_range=(0.1, 0.3),
    platform_width=1.0,
    double_box=False,
)

# 8. 间隙地形
mesh_gap = terrain_gen.MeshGapTerrainCfg(
    proportion=0.05,
    gap_width_range=(0.2, 0.5),
    platform_width=1.0,
)

# 9. 悬浮环地形
mesh_floating_ring = terrain_gen.MeshFloatingRingTerrainCfg(
    proportion=0.05,
    ring_width_range=(0.2, 0.4),
    ring_height_range=(0.1, 0.2),
    ring_thickness=0.05,
    platform_width=1.0,
)

# 10. 星形地形
mesh_star = terrain_gen.MeshStarTerrainCfg(
    proportion=0.05,
    num_bars=6,
    bar_width_range=(0.05, 0.1),
    bar_height_range=(0.1, 0.2),
    platform_width=1.0,
)

# 11. 重复金字塔（圆锥）地形
mesh_repeat_pyramids = terrain_gen.MeshRepeatedPyramidsTerrainCfg(
    proportion=0.05,
    object_params_start=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
        num_objects=5,
        height=0.2,
        radius=0.1,
        max_yx_angle=15.0,
        degrees=True,
    ),
    object_params_end=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
        num_objects=10,
        height=0.4,
        radius=0.2,
        max_yx_angle=30.0,
        degrees=True,
    ),
    max_height_noise=0.05,
    platform_width=1.0,
)

# 12. 重复盒子地形
mesh_repeat_boxes = terrain_gen.MeshRepeatedBoxesTerrainCfg(
    proportion=0.05,
    object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        num_objects=5,
        height=0.2,
        size=(0.2, 0.3),
        max_yx_angle=10.0,
        degrees=True,
    ),
    object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        num_objects=10,
        height=0.4,
        size=(0.4, 0.5),
        max_yx_angle=20.0,
        degrees=True,
    ),
    max_height_noise=0.05,
    platform_width=1.0,
)

# 13. 重复圆柱地形
mesh_repeat_cylinders = terrain_gen.MeshRepeatedCylindersTerrainCfg(
    proportion=0.05,
    object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
        num_objects=5,
        height=0.2,
        radius=0.1,
        max_yx_angle=10.0,
        degrees=True,
    ),
    object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
        num_objects=10,
        height=0.4,
        radius=0.2,
        max_yx_angle=20.0,
        degrees=True,
    ),
    max_height_noise=0.05,
    platform_width=1.0,
)
# 1. 随机均匀噪声
hf_random_uniform = terrain_gen.HfRandomUniformTerrainCfg(
    proportion=0.05,
    noise_range=(0.01, 0.05),
    noise_step=0.02,
    downsampled_scale=None,   # 若不下采样可留 None
    border_width=0.2,         # ≥ horizontal_scale
    horizontal_scale=0.1,
    vertical_scale=0.005,
)

# 2. 金字塔斜坡
hf_pyramid_slope = terrain_gen.HfPyramidSlopedTerrainCfg(
    proportion=0.05,
    slope_range=(0.1, 0.4),
    platform_width=2.0,
    inverted=False,
    border_width=0.2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
)

# 3. 反向金字塔斜坡
hf_pyramid_slope_inv = terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
    proportion=0.05,
    slope_range=(0.1, 0.4),
    platform_width=2.0,
    # inverted=True 已由子类设定
    border_width=0.2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
)

# 4. 金字塔阶梯
hf_pyramid_stairs = terrain_gen.HfPyramidStairsTerrainCfg(
    proportion=0.05,
    step_height_range=(0.05, 0.25),
    step_width=0.3,
    platform_width=1.5,
    inverted=False,
    border_width=0.2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
)

# 5. 反向金字塔阶梯
hf_pyramid_stairs_inv = terrain_gen.HfInvertedPyramidStairsTerrainCfg(
    proportion=0.05,
    step_height_range=(0.05, 0.25),
    step_width=0.3,
    platform_width=1.5,
    # inverted=True 已由子类设定
    border_width=0.2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
)

# 6. 离散障碍物
hf_discrete_obstacles = terrain_gen.HfDiscreteObstaclesTerrainCfg(
    proportion=0.05,
    obstacle_height_mode="choice",
    obstacle_width_range=(0.1, 0.3),
    obstacle_height_range=(0.05, 0.2),
    num_obstacles=8,
    platform_width=1.0,
    border_width=0.2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
)

# 7. 波浪地形
hf_wave = terrain_gen.HfWaveTerrainCfg(
    proportion=0.05,
    amplitude_range=(0.02, 0.1),
    num_waves=2,
    border_width=0.2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
)

# 8. 跳石地形
hf_stepping_stones = terrain_gen.HfSteppingStonesTerrainCfg(
    proportion=0.05,
    stone_height_max=0.15,
    stone_width_range=(0.1, 1.5),
    stone_distance_range=(0.3, 0.6),
    holes_depth=-1.0,
    platform_width=1.0,
    border_width=0.2,
    horizontal_scale=0.1,
    vertical_scale=0.5,
)

VIT_ROUGH_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=60.0,
    num_rows=10,
    num_cols=40,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "plane": mesh_plane,
        # "pyramid_stairs": mesh_pyramid,
        # "pyramid_stairs_inv": mesh_pyramid_inv,
        # "boxes": mesh_random_grid,
        "rails": mesh_rails,
        "pit": mesh_pit,
        "box": mesh_box,
        "gap": mesh_gap,
        "floating_ring": mesh_floating_ring,
        "star": mesh_star,
        "repeat_pyramids": mesh_repeat_pyramids,
        "repeat_boxes": mesh_repeat_boxes,
        "repeat_cylinders": mesh_repeat_cylinders,
        "random_uniform": hf_random_uniform,
        "pyramid_slope": hf_pyramid_slope,
        "pyramid_slope_inv": hf_pyramid_slope_inv,
        "pyramid_stairs": hf_pyramid_stairs,
        "pyramid_stairs_inv": hf_pyramid_stairs_inv,
        "discrete_obs": hf_discrete_obstacles,
        "wave": hf_wave,
        "stepping_stones": hf_stepping_stones,
    },
)

"""Rough terrains configuration."""

ROUGH_TERRAINS_CFG_ORIGINAL = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=True,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.30),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.30),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.025, 0.1), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.6), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.6), platform_width=2.0, border_width=0.25
        ),
    },
)
"""Rough terrains configuration."""

AMP_VAE_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
    max_init_terrain_level=1,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.25, 0.25),
    ),
    debug_vis=False,
)


AMP_VAE_VIT_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
    max_init_terrain_level=1,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.25, 0.25),
    ),
    debug_vis=False,
)

# PARKOUR_TERRAIN_CFG = TerrainImporterCfg(
#     prim_path="/World/ground",
#     terrain_type="generator",
#     terrain_generator=ROUGH_TERRAINS_CFG,
#     max_init_terrain_level=1,
#     collision_group=-1,
#     physics_material=sim_utils.RigidBodyMaterialCfg(
#         friction_combine_mode="multiply",
#         restitution_combine_mode="multiply",
#         static_friction=1.0,
#         dynamic_friction=1.0,
#     ),
#     visual_material=sim_utils.MdlFileCfg(
#         mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
#         project_uvw=True,
#         texture_scale=(0.25, 0.25),
#     ),
#     debug_vis=False,
# )

AMP_VAE_PERCEPTION_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
    max_init_terrain_level=1,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.25, 0.25),
    ),
    debug_vis=False,
)

LOCOMOTION_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
    max_init_terrain_level=1,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.25, 0.25),
    ),
    debug_vis=False,
)

LOCOMOTION_TERRAIN_CFG2d4 = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG2d4,
    max_init_terrain_level=1,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.25, 0.25),
    ),
    debug_vis=False,
)

# grq20_v2d4 rough-only（多 tile）：任务侧直接引用该配置即可
LOCOMOTION_ROUGH_ONLY_TERRAIN_CFG2d4 = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=ROUGH_ONLY_TERRAINS_CFG2d4,
    # 不限制初始 level：允许 env 分布到整个 tile 网格
    max_init_terrain_level=None,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=(
            f"{ISAACLAB_NUCLEUS_DIR}/Materials/"
            "TilesMarbleSpiderWhiteBrickBondHoned/"
            "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
        ),
        project_uvw=True,
        texture_scale=(0.25, 0.25),
    ),
    debug_vis=False,
)

LOCOMOTION_TERRAIN_CFGex1 = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFGex1,
    max_init_terrain_level=1,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.25, 0.25),
    ),
    debug_vis=False,
)

MARG_LOCOMOTION_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=MARG_ROUGH_TERRAINS_CFG,
    max_init_terrain_level=1,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.25, 0.25),
    ),
    debug_vis=False,
)

PIE_ROUGH_TERRAIN_CFG = MultiPrimTerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=PIE_ROUGH_TERRAINS_CFG,
    max_init_terrain_level=1,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        dynamic_friction=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.25, 0.25),
    ),
    debug_vis=False,
    # 仅对 square_hole 子地形做“按高度拆分”，其余子地形仍然全部挂在 /World/ground
    obstacle_prim_path="/World/obstacles",
    obstacle_sub_terrain_names=("square_hole",),
    ground_split_height=0.1,
    # 悬空部分使用玻璃 MDL 材质（实体 + 透光），只影响外观，不改物理和高度图逻辑
    obstacle_visual_material=None,
    # obstacle_visual_material=sim_utils.GlassMdlCfg(
    #     glass_color=(0.8, 0.95, 1.0),   # 微蓝色玻璃
    #     frosting_roughness=0.5,         # 越大越“毛玻璃”，越小越透明
    #     thin_walled=False,
    # ),
)

# PIE_ROUGH_TERRAIN_CFG = TerrainImporterCfg(
#     prim_path="/World/ground",
#     terrain_type="generator",
#     terrain_generator=PIE_ROUGH_TERRAINS_CFG,
#     max_init_terrain_level=1,
#     collision_group=-1,
#     physics_material=sim_utils.RigidBodyMaterialCfg(
#         friction_combine_mode="multiply",
#         restitution_combine_mode="multiply",

#         dynamic_friction=1.0,
#     ),
#     visual_material=sim_utils.MdlFileCfg(
#         mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
#         project_uvw=True,
#         texture_scale=(0.25, 0.25),
#     ),
#     debug_vis=False,
# )



LOCOMOTION_PLANE_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=PLANE_TERRAINS_CFG,
    max_init_terrain_level=1,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.25, 0.25),
    ),
    debug_vis=False,
)
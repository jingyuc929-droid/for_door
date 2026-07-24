from __future__ import annotations

from typing import Tuple

from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaaclab.utils import configclass

from .terrain import MultiPrimTerrainImporter


@configclass
class MultiPrimTerrainImporterCfg(TerrainImporterCfg):
    """扩展版地形导入配置，支持将部分子地形的高处几何拆分到单独的 prim。

    - 仍然复用 IsaacLab 的 ``TerrainGenerator``，保持地形统计/难度/flat patches 等逻辑不变。
    - 通过在导入阶段，按子地形名称与高度阈值，把网格拆成：
        - 低处（近地面）的三角形 → 挂在 ``base_prim_path``（通常为 ``/World/ground``）
        - 高处（悬空部分）的三角形 → 挂在 ``obstacle_prim_path``（例如 ``/World/obstacles``）
    """

    # 使用自定义的导入类
    class_type: type = MultiPrimTerrainImporter

    # 基础地形 prim（原 TerrainImporterCfg.prim_path 仍然保留，用于兼容）
    base_prim_path: str | None = None
    """基础地形（用于高度图等）的 prim 根路径，默认等于 ``prim_path``。"""

    obstacle_prim_path: str = "/World/obstacles"
    """悬空障碍物的 prim 根路径。"""

    obstacle_sub_terrain_names: Tuple[str, ...] = ("square_hole",)
    """需要做“悬空部分拆分”的子地形名称集合（对应 TerrainGeneratorCfg.sub_terrains 的 key）。"""

    ground_split_height: float = 0.1
    """判定“近地面”三角形的 z 高度阈值（单位：米）。

    对应子地形中，所有三角形若所有顶点的 z 坐标都小于该阈值，则认为属于地面；否则属于“悬空部分”。
    """

    obstacle_visual_material: object | None = None
    """悬空障碍物使用的可视化材质。

    - 若为 None，则复用基础地形的 ``visual_material``（不改变外观）。
    - 若设置为 ``sim_utils.PreviewSurfaceCfg(opacity<1.0)`` 等，则可以让悬空部分半透明，方便调试。
    """

    def __post_init__(self):
        # 如果没有显式指定 base_prim_path，则默认和 prim_path 一致
        if self.base_prim_path is None:
            self.base_prim_path = self.prim_path

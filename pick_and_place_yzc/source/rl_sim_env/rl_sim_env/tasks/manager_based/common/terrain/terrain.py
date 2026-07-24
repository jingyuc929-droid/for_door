from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.terrains.terrain_importer import TerrainImporter
from isaaclab.terrains.utils import create_prim_from_mesh
import trimesh

if TYPE_CHECKING:
    from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
    from .terrain_cfg import MultiPrimTerrainImporterCfg


class MultiPrimTerrainImporter(TerrainImporter):
    """自定义地形导入器：将指定子地形的高处网格拆分到单独的 prim。

    设计约束：
    - 不改 IsaacLab 的 ``TerrainGenerator`` / ``TerrainImporter`` 源码；
    - 保持原有 curriculum / env_origins / flat_patches 等逻辑；
    - 仅在“网格导入到 USD”这一步，根据子地形名称 + 高度阈值做几何拆分。
    """

    def __init__(self, cfg: "MultiPrimTerrainImporterCfg"):
        # 先让基类检查配置合法性，但不调用其 __init__ 导入网格逻辑
        cfg.validate()

        # 保存配置与仿真设备
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore

        # 初始化与 TerrainImporter 保持一致的属性
        self.terrain_prim_paths: list[str] = []
        self.terrain_origins = None
        self.env_origins = None
        self._terrain_flat_patches: dict = {}

        # 仅对 "generator" 类型做特殊处理，其余类型直接走父类逻辑
        if self.cfg.terrain_type != "generator":
            # 回退到标准 TerrainImporter 行为
            super().__init__(cfg)  # type: ignore[arg-type]
            return

        if self.cfg.terrain_generator is None:
            raise ValueError("Input terrain type is 'generator' but no value provided for 'terrain_generator'.")

        # 1) 调用原 TerrainGenerator，生成完整地形（含所有子地形和 border）
        terrain_gen_cfg: "TerrainGeneratorCfg" = self.cfg.terrain_generator
        terrain_generator = terrain_gen_cfg.class_type(cfg=terrain_gen_cfg, device=self.device)

        # 注意：TerrainGenerator 在内部对合并后的 terrain_mesh 做了一次“整体居中”的平移变换，
        # 但是并没有对 terrain_meshes 逐块网格应用这一步。
        # env_origins/terrain_origins 是基于“居中后”的坐标系计算的。
        # 因此这里需要对每个 tile mesh 应用相同的平移，使 mesh 与 origins 对齐。
        transform = np.eye(4)
        transform[:2, -1] = (
            -terrain_gen_cfg.size[0] * terrain_gen_cfg.num_rows * 0.5,
            -terrain_gen_cfg.size[1] * terrain_gen_cfg.num_cols * 0.5,
        )
        for tile_mesh in terrain_generator.terrain_meshes:
            tile_mesh.apply_transform(transform)

        # 记录 flat patches，保持与原 TerrainImporter 行为一致
        self._terrain_flat_patches = terrain_generator.flat_patches

        # 2) 基于 TerrainGenerator 的 cfg，重建“每个子地形网格对应的 sub_terrain 名称”顺序
        sub_terrain_order = self._build_sub_terrain_order(terrain_gen_cfg)

        num_tiles = terrain_gen_cfg.num_rows * terrain_gen_cfg.num_cols
        if len(terrain_generator.terrain_meshes) < num_tiles:
            raise RuntimeError(
                f"TerrainGenerator produced fewer meshes ({len(terrain_generator.terrain_meshes)}) "
                f"than expected tiles ({num_tiles})."
            )

        # 3) 遍历所有 tile，将它们重新组合成：
        #    - 一个完整的 ground mesh（等价于原始 TerrainImporter 的 terrain_mesh）
        #    - 一个只包含“悬空部分”的 obstacle mesh（仅对指定子地形）
        ground_meshes: list[trimesh.Trimesh] = []
        obstacle_meshes: list[trimesh.Trimesh] = []

        for tile_idx in range(num_tiles):
            mesh = terrain_generator.terrain_meshes[tile_idx]
            sub_name = sub_terrain_order[tile_idx]
            is_obstacle_sub = sub_name in self.cfg.obstacle_sub_terrain_names

            if is_obstacle_sub:
                # 对于指定的悬空地形：按高度拆分
                g_mesh, o_mesh = self._split_mesh_by_height(mesh, self.cfg.ground_split_height)
                if g_mesh is not None:
                    ground_meshes.append(g_mesh)
                if o_mesh is not None:
                    obstacle_meshes.append(o_mesh)
            else:
                # 其它地形：整体当作 ground，行为与原始 TerrainImporter 完全一致
                ground_meshes.append(mesh)

        # 最后一个 mesh 是 border，将其整体加入 ground
        if len(terrain_generator.terrain_meshes) > num_tiles:
            border_mesh = terrain_generator.terrain_meshes[-1]
            ground_meshes.append(border_mesh)

        # 4) 将 ground / obstacle 分别合并成单一 mesh，并导入到各自 prim
        if len(ground_meshes) > 0:
            ground_mesh_combined = trimesh.util.concatenate(ground_meshes)
            self._import_mesh_to_base(name="terrain", mesh=ground_mesh_combined)

        if len(obstacle_meshes) > 0:
            obstacle_mesh_combined = trimesh.util.concatenate(obstacle_meshes)
            self._import_mesh_to_obstacle(name="terrain_obstacles", mesh=obstacle_mesh_combined)

        # 5) 复用基类的 env_origins 计算逻辑
        #    注意：这里直接调用父类的实例方法实现，避免重复代码
        TerrainImporter.configure_env_origins(self, terrain_generator.terrain_origins)

        # 6) 根据配置开启/关闭 debug 可视化
        self.set_debug_vis(self.cfg.debug_vis)

    # --------------------------------------------------------------------- #
    # 内部辅助：根据 TerrainGeneratorCfg 还原 tile -> sub_terrain name 顺序
    # --------------------------------------------------------------------- #
    def _build_sub_terrain_order(self, cfg: "TerrainGeneratorCfg") -> List[str]:
        """复制 terrain_generator._generate_*_terrains 中的“子地形选择逻辑”，
        但只返回每个 tile 对应的 sub_terrain 名称顺序，不实际生成网格。
        """
        sub_cfgs = list(cfg.sub_terrains.values())
        sub_names = list(cfg.sub_terrains.keys())
        proportions = np.array([sub_cfg.proportion for sub_cfg in sub_cfgs], dtype=float)
        if np.any(proportions < 0.0):
            raise ValueError("Sub-terrain 'proportion' must be non-negative.")
        if np.all(proportions == 0.0):
            raise ValueError("At least one sub-terrain 'proportion' must be positive.")
        proportions /= np.sum(proportions)

        order: List[str] = []

        if cfg.curriculum:
            # 对应 TerrainGenerator._generate_curriculum_terrains 的逻辑：
            #   - 列方向根据 proportion 决定使用哪种 sub_terrain
            cumsum = np.cumsum(proportions)
            num_cols = cfg.num_cols
            num_rows = cfg.num_rows

            sub_indices: list[int] = []
            for index in range(num_cols):
                threshold = index / num_cols + 0.001
                sub_index = int(np.min(np.where(threshold < cumsum)[0]))
                sub_indices.append(sub_index)

            for sub_col in range(num_cols):
                for sub_row in range(num_rows):
                    sub_idx = sub_indices[sub_col]
                    order.append(sub_names[sub_idx])
        else:
            # 对应 TerrainGenerator._generate_random_terrains 的逻辑：
            #   - 使用与 TerrainGenerator 相同的随机种子，按相同次序 sampling sub_terrain
            if cfg.seed is not None:
                seed = cfg.seed
            else:
                # 与 TerrainGenerator 中的实现保持一致
                seed = np.random.get_state()[1][0]
            rng = np.random.default_rng(seed)

            num_rows = cfg.num_rows
            num_cols = cfg.num_cols
            for index in range(num_rows * num_cols):
                # 注意：这里与 TerrainGenerator._generate_random_terrains 使用相同的循环顺序
                _ = np.unravel_index(index, (num_rows, num_cols))
                sub_index = int(rng.choice(len(proportions), p=proportions))
                order.append(sub_names[sub_index])

        return order

    # --------------------------------------------------------------------- #
    # 内部辅助：网格拆分与导入
    # --------------------------------------------------------------------- #
    def _split_mesh_by_height(
        self, mesh: "trimesh.Trimesh", ground_split_height: float
    ) -> tuple["trimesh.Trimesh | None", "trimesh.Trimesh | None"]:
        """将单个 tile 网格按高度拆成 ground / obstacle 两部分，但不直接导入。

        返回值:
            ground_mesh: 仅包含“近地面”三角形的 mesh（若没有则为 None）
            obstacle_mesh: 仅包含“高于阈值”的三角形 mesh（若没有则为 None）
        """
        # faces -> (F, 3, 3) 顶点坐标
        tri_vertices = mesh.vertices[mesh.faces]
        z_coords = tri_vertices[..., 2]
        max_z = z_coords.max(axis=1)

        # ground：所有顶点 z 都在 ground_split_height 以下
        ground_mask = max_z <= ground_split_height
        obstacle_mask = ~ground_mask

        ground_mesh: "trimesh.Trimesh | None" = None
        obstacle_mesh: "trimesh.Trimesh | None" = None

        if np.any(ground_mask):
            ground_faces_idx = np.nonzero(ground_mask)[0]
            gm = mesh.submesh([ground_faces_idx], append=True)
            if isinstance(gm, list):
                gm = trimesh.util.concatenate(gm)
            ground_mesh = gm

        if np.any(obstacle_mask):
            obstacle_faces_idx = np.nonzero(obstacle_mask)[0]
            om = mesh.submesh([obstacle_faces_idx], append=True)
            if isinstance(om, list):
                om = trimesh.util.concatenate(om)
            obstacle_mesh = om

        return ground_mesh, obstacle_mesh

    def _import_mesh_to_base(self, name: str, mesh: "trimesh.Trimesh"):
        """将 mesh 导入到基础地形 prim（base_prim_path）下。"""
        prim_root = self.cfg.base_prim_path or self.cfg.prim_path
        prim_path = prim_root + f"/{name}"

        if prim_path in self.terrain_prim_paths:
            raise ValueError(
                f"A terrain with the name '{name}' already exists. "
                f"Existing terrains: {', '.join(self.terrain_names)}."
            )

        self.terrain_prim_paths.append(prim_path)
        create_prim_from_mesh(
            prim_path,
            mesh,
            visual_material=self.cfg.visual_material,
            physics_material=self.cfg.physics_material,
        )

    def _import_mesh_to_obstacle(self, name: str, mesh: "trimesh.Trimesh"):
        """将 mesh 导入到障碍物 prim（obstacle_prim_path）下。"""
        prim_root = self.cfg.obstacle_prim_path
        prim_path = prim_root + f"/{name}"

        if prim_path in self.terrain_prim_paths:
            raise ValueError(
                f"A terrain with the name '{name}' already exists. "
                f"Existing terrains: {', '.join(self.terrain_names)}."
            )

        self.terrain_prim_paths.append(prim_path)
        # 如果为障碍物单独指定了材质（例如半透明），优先使用该材质；否则复用基础地形的材质
        visual_material = self.cfg.obstacle_visual_material or self.cfg.visual_material
        create_prim_from_mesh(
            prim_path,
            mesh,
            visual_material=visual_material,
            physics_material=self.cfg.physics_material,
        )

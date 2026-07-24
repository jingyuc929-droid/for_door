import isaaclab.sim as sim_utils
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


GREEN_SPHERE_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.015,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        )
    }
)
"""Configuration for the green sphere marker."""
RED_SPHERE_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.015,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        )
    }
)
"""Configuration for the red sphere marker."""

BLUE_SPHERE_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.015,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        )
    }
)
"""Configuration for the blue sphere marker."""
YELLOW_SPHERE_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.015,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
        )
    }
)
"""Configuration for the yellow sphere marker."""

WHITE_SPHERE_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.03,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
        )
    }
)


GREEN_CUBOID_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "cuboid": sim_utils.CuboidCfg(
            size=(0.035, 0.035, 0.035),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), opacity=0.5),
        )
    }
)
"""Configuration for the green sphere marker."""
RED_CUBOID_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "cuboid": sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.5),
        )
    }
)
"""Configuration for the red sphere marker."""

BLUE_CUBOID_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "cuboid": sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), opacity=0.5),
        )
    }
)
"""Configuration for the blue sphere marker."""
YELLOW_CUBOID_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "cuboid": sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), opacity=0.5),
        )
    }
)

# 单独的坐标轴可视化配置（每个轴一个独立的VisualizationMarkers实例）
# 单向 30cm：cuboid 以中心为原点，可视化时把位置沿 ee 局部轴偏移 0.15m，
# 即从 ee 中心单向延伸到 +0.30m，端点正好落在对应偏移球上。
# X轴=红色（对应红色x偏移球）
X_AXIS_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "x_axis": sim_utils.CuboidCfg(
            size=(0.3, 0.015, 0.015),  # 长度0.3m，粗细1.5cm
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # 红色
                opacity=0.9,
            ),
        ),
    }
)

# Y轴=蓝色（对应蓝色y偏移球）
Y_AXIS_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "y_axis": sim_utils.CuboidCfg(
            size=(0.015, 0.3, 0.015),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 1.0),  # 蓝色
                opacity=0.9,
            ),
        ),
    }
)

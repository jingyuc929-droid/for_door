from math import fabs
from pickle import TRUE

from numpy import False_
from isaaclab.sensors import RayCasterCfg, patterns
import isaaclab.sim as sim_utils
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from rl_sim_env.tasks.manager_based.common.sensor.lidar_pattern import LidarDynamicPatternCfg
##
# Sensors.
##

RAY_CASTER_MARKER_CFG_RED = VisualizationMarkersCfg(
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
)

RAY_CASTER_MARKER_CFG_GREEN = VisualizationMarkersCfg(
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    },
)

RAY_CASTER_MARKER_CFG_BLUE = VisualizationMarkersCfg(
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    },
)

RAY_CASTER_MARKER_CFG_YELLOW = VisualizationMarkersCfg(
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
        ),
    },
)
RAY_CASTER_MARKER_CFG_PURPLE = VisualizationMarkersCfg(
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
        ),
    },
)
RAY_CASTER_MARKER_CFG_ORANGE = VisualizationMarkersCfg(
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
        ),
    },
)


CRITIC_HEIGHT_SCANNER_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
)

PERCEPTION_HEIGHT_SCANNER_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    attach_yaw_only=True,
    pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[3.0, 1.2], ordering="yx"),
    debug_vis=False,
    max_distance=30.0,
    mesh_prim_paths=["/World/ground"],
    visualizer_cfg=RAY_CASTER_MARKER_CFG_RED.replace(prim_path="/Visuals/RayCaster"),
)

VOXEL_SCANNER_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    attach_yaw_only=False,
    pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
)

FOOT_SCANNER_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.1, 0.1]),
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
    visualizer_cfg=RAY_CASTER_MARKER_CFG_GREEN.replace(prim_path="/Visuals/RayCaster"),
)

E1R_FRONT_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(0.43018, 0.0, 0.02301), rot=(0.9659258, 0, 0.258819, 0)),
    attach_yaw_only=False,
    pattern_cfg=patterns.LidarPatternCfg(
        channels=64, vertical_fov_range=(-45.0, 45.0), horizontal_fov_range=(-60.0, 60.0), horizontal_res=2.0
    ),
    debug_vis=False,
    max_distance=2.0,
    mesh_prim_paths=["/World/ground"],
    visualizer_cfg=RAY_CASTER_MARKER_CFG_YELLOW.replace(prim_path="/Visuals/RayCaster"),
)

E1R_BACK_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(-0.44386, 0.0, -0.01934), rot=(0, -0.258819, 0, 0.9659258)),
    attach_yaw_only=False,
    pattern_cfg=patterns.LidarPatternCfg(
        channels=64, vertical_fov_range=(-45.0, 45.0), horizontal_fov_range=(-60.0, 60.0), horizontal_res=2.0
    ),
    debug_vis=False,
    max_distance=2.0,
    mesh_prim_paths=["/World/ground"],
    visualizer_cfg=RAY_CASTER_MARKER_CFG_BLUE.replace(prim_path="/Visuals/RayCaster"),

)

MID360_UP_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(0.3398, 0.0, 0.125)),
    attach_yaw_only=False,
    pattern_cfg=patterns.LidarPatternCfg(
        channels=32, vertical_fov_range=(-7.0, 52.0), horizontal_fov_range=(-180, 180.0), horizontal_res=1.5
    ),
    debug_vis=False,
    max_distance=5.0,
    mesh_prim_paths=["/World/ground"],
    visualizer_cfg=RAY_CASTER_MARKER_CFG_PURPLE.replace(prim_path="/Visuals/RayCaster"),
)

AIRY_FRONT_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(0.45, 0.0, 0.0), rot=(0.7071068, 0, 0.7071068, 0)),
    attach_yaw_only=False,
    pattern_cfg=patterns.LidarPatternCfg(
        channels=64, vertical_fov_range=(0, 90), horizontal_fov_range=(-180, 180), horizontal_res=3.0
    ),
    debug_vis=False,
    max_distance=2.2,
    mesh_prim_paths=["/World/ground"],
    visualizer_cfg=RAY_CASTER_MARKER_CFG_YELLOW.replace(prim_path="/Visuals/RayCaster"),
)

AIRY_BACK_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(-0.45, 0.0, 0.0), rot=(0, -0.7071068, 0, 0.7071068)),
    attach_yaw_only=False,
    pattern_cfg=patterns.LidarPatternCfg(
        channels=64, vertical_fov_range=(0, 90), horizontal_fov_range=(-180, 180), horizontal_res=3.0
    ),
    debug_vis=False,
    max_distance=2.2,
    mesh_prim_paths=["/World/ground"],
    visualizer_cfg=RAY_CASTER_MARKER_CFG_BLUE.replace(prim_path="/Visuals/RayCaster"),
)


FRONT_DYNAMIC_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(0.45, 0.0, 0.0)),
    attach_yaw_only=False,
    pattern_cfg=LidarDynamicPatternCfg(ray_num=1525),
    debug_vis=False,
    max_distance=30.0,
    mesh_prim_paths=["/World/ground"],
    visualizer_cfg=RAY_CASTER_MARKER_CFG_YELLOW.replace(prim_path="/Visuals/RayCaster"),
)

BACK_DYNAMIC_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(-0.45, 0.0, 0.0)),
    attach_yaw_only=False,
    pattern_cfg=LidarDynamicPatternCfg(ray_num=1525),
    debug_vis=False,
    max_distance=30.0,
    mesh_prim_paths=["/World/ground"],
    visualizer_cfg=RAY_CASTER_MARKER_CFG_BLUE.replace(prim_path="/Visuals/RayCaster"),
)


EXPERT_HEIGHT_SCANNER_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=0.0625, size=[2.9375, 0.9375], ordering="yx"),
    debug_vis=False,
    max_distance=30.0,
    mesh_prim_paths=["/World/ground"],
    visualizer_cfg=RAY_CASTER_MARKER_CFG_RED.replace(prim_path="/Visuals/RayCaster"),
)

BLIND_HEIGHT_SCANNER_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    debug_vis=False,
    max_distance=30.0,
    mesh_prim_paths=["/World/ground"],
    visualizer_cfg=RAY_CASTER_MARKER_CFG_RED.replace(prim_path="/Visuals/RayCaster"),
)

MARG_HEIGHT_SCANNER_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[3.0, 1.5], ordering="yx"),
    debug_vis=False,
    max_distance=30.0,
    mesh_prim_paths=["/World/ground"],
    visualizer_cfg=RAY_CASTER_MARKER_CFG_RED.replace(prim_path="/Visuals/RayCaster"),
)

PIE_HEIGHT_SCANNER_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2.0, 1.2], ordering="yx"),
    debug_vis=False,
    max_distance=30.0,
    mesh_prim_paths=["/World/ground"],
    visualizer_cfg=RAY_CASTER_MARKER_CFG_RED.replace(prim_path="/Visuals/RayCaster"),
)

BLIND_FL_FOOT_SCANNER_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/FL_foot",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.1, 0.1]),
    debug_vis=False,
    max_distance=30.0,
    mesh_prim_paths=["/World/ground"],
    visualizer_cfg=RAY_CASTER_MARKER_CFG_GREEN.replace(prim_path="/Visuals/RayCaster/FL_foot"),
)

BLIND_FR_FOOT_SCANNER_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/FR_foot",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.1, 0.1]),
    debug_vis=False,
    max_distance=30.0,
    mesh_prim_paths=["/World/ground"],
    visualizer_cfg=RAY_CASTER_MARKER_CFG_GREEN.replace(prim_path="/Visuals/RayCaster/FR_foot"),
)

BLIND_RL_FOOT_SCANNER_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/RL_foot",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.1, 0.1]),
    debug_vis=False,
    max_distance=30.0,
    mesh_prim_paths=["/World/ground"],
    visualizer_cfg=RAY_CASTER_MARKER_CFG_GREEN.replace(prim_path="/Visuals/RayCaster/RL_foot"),
)

BLIND_RR_FOOT_SCANNER_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/RR_foot",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.1, 0.1]),
    debug_vis=False,
    max_distance=30.0,
    mesh_prim_paths=["/World/ground"],
    visualizer_cfg=RAY_CASTER_MARKER_CFG_GREEN.replace(prim_path="/Visuals/RayCaster/RR_foot"),
)
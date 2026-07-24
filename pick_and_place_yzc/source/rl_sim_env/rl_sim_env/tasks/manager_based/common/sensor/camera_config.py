import isaaclab.sim as sim_utils
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.sensors import PinholeCameraCfg
from isaaclab.sensors.ray_caster import RayCasterCameraCfg
from isaaclab.sensors.ray_caster import patterns

##
# Camera.
##

CAMERA_MARKER_CFG_RED = VisualizationMarkersCfg(
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
)

CAMERA_MARKER_CFG_GREEN = VisualizationMarkersCfg(
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    },
)

CAMERA_MARKER_CFG_BLUE = VisualizationMarkersCfg(
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    },
)

CAMERA_MARKER_CFG_YELLOW = VisualizationMarkersCfg(
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
        ),
    },
)
CAMERA_MARKER_CFG_PURPLE = VisualizationMarkersCfg(
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
        ),
    },
)
CAMERA_MARKER_CFG_ORANGE = VisualizationMarkersCfg(
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
        ),
    },
)

# 新建一个深度相机
# 参数含义与 FOV（视场）关系说明：
# - 单位：focal_length 与 aperture（水平/垂直孔径）以厘米(cm)为单位
# - 针孔模型 FOV 公式：
#     FOV_h = 2 * atan(horizontal_aperture / (2 * focal_length))
#     FOV_v = 2 * atan(vertical_aperture   / (2 * focal_length))
# - 本配置代入数值：f = 24.0 cm, a_h = 45.55 cm, a_v = 26.6 cm
#     FOV_h ≈ 2 * atan(45.55 / 48) ≈ 87.1°
#     FOV_v ≈ 2 * atan(26.6  / 48) ≈ 57.8°
# - 说明：
#     * 焦距越大，视场越窄；焦距越小，视场越宽
#     * width/height 影响角分辨率(每像素角度)，不改变总 FOV
#     * 主点偏移（若设置）不改变 FOV 数值
front_depth_camera_cfg = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/camera_front/depth_camera",
    offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.00), convention="world"),
    spawn=PinholeCameraCfg(
        focal_length=24.0,
        horizontal_aperture=45.55,
        vertical_aperture=26.6,
        focus_distance=400.0,
        f_stop=0.0,
    ),
    data_types=["depth"],
    width=1280,
    height=720,
    update_latest_camera_pose=False,
    semantic_filter="*",
    colorize_semantic_segmentation=False,
    colorize_instance_id_segmentation=False,
    colorize_instance_segmentation=False,
    semantic_segmentation_mapping={},
    history_length=2,
)

camera_width = 64   # 每隔10个像素取一个像素
camera_height = 64   # 每隔10个像素取一个像素
scale = 1.0

# 加载一个基于针孔模型的深度相机
# 这里的参数待确认正确性
# 参数含义与 FOV（视场）关系说明（针孔模型，单位：cm）：
# - focal_length：焦距；越大视场越窄
# - horizontal_aperture / vertical_aperture：成像面宽/高
# - width / height：像素宽/高；影响角分辨率，不改变总 FOV
# FOV 计算公式：
#   FOV_h = 2 * atan(horizontal_aperture / (2 * focal_length))
#   FOV_v = 2 * atan(vertical_aperture   / (2 * focal_length))
# 代入当前数值：f = 24.0 cm, a_h = 45.55 cm, a_v = 26.6 cm
#   FOV_h ≈ 2 * atan(45.55 / 48) ≈ 87.1°
#   FOV_v ≈ 2 * atan(26.6  / 48) ≈ 57.8°
# （可选）对角视场：a_d = sqrt(a_h^2 + a_v^2)，FOV_d = 2 * atan(a_d / (2f)) ≈ 95.5°
# rot=(w, x, y, z)=(0.924, -0.383, 0.0, 0)<-->(roll, pitch, yaw)=(-45°, 0, 0)
# - ``"opengl"`` - forward axis: ``-Z`` - up axis: ``+Y`` - Offset is applied in the OpenGL (Usd.Camera) convention.
# - ``"ros"``    - forward axis: ``+Z`` - up axis: ``-Y`` - Offset is applied in the ROS convention.
# - ``"world"``  - forward axis: ``+X`` - up axis: ``+Z`` - Offset is applied in the World Frame convention.
# (0,45,0) -->(0.924, 0.0, 0.383, 0)
# (0,30,0) -->(0.966, 0.0, 0.259, 0)
# (0,20,0) -->(0.985, 0.0, 0.174, 0)
# (0,15,0) -->(0.991, 0.0, 0.131, 0)

front_depth_camera_cfg_from_ray_caster = RayCasterCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/camera_front",
    mesh_prim_paths=["/World/ground"],
    max_distance=10,
    offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.00),rot=(0.991, 0.0, 0.131, 0), convention="world"),
    depth_clipping_behavior="max",
    pattern_cfg=patterns.PinholeCameraPatternCfg(
        focal_length=24.0*scale,  # cm
        horizontal_aperture=45.55*scale,  # cm
        vertical_aperture=26.6*scale,  # cm
        width=int(camera_width),
        height=int(camera_height),
    ),
    attach_yaw_only=False,
    debug_vis=False,
    history_length=1,
    visualizer_cfg=CAMERA_MARKER_CFG_GREEN.replace(prim_path="/Visuals/RayCaster"),
)
"""机械臂/操作相关 command 配置工厂函数。

这里的函数尽量保持与 common/command/config.py 的风格一致，便于迁移。
"""

from __future__ import annotations

import isaaclab.sim as sim_utils

from .projected_com_target_points_command_cfg import (
    ProjectedComTargetPointsCommandCfg,
)

from rl_debug.marker import (
    BLUE_SPHERE_MARKER_CFG,
    GREEN_SPHERE_MARKER_CFG,
    RED_SPHERE_MARKER_CFG,
    YELLOW_SPHERE_MARKER_CFG,
)


def create_projected_com_target_points_command_cfg(
    asset_name: str = "robot",
    trunk_body_name: str = "base_link",
    terrain_sensor_name: str = "height_scanner",
    ee_body_name: str | None = None,
    pos_r_range: tuple[float, float] = (0.3, 0.8),
    pos_theta_range: tuple[float, float] = (-1.57, 1.57),
    pos_z_range: tuple[float, float] = (0.1, 0.5),
    roll_range: tuple[float, float] = (-0.2, 0.2),
    pitch_range: tuple[float, float] = (-0.2, 0.2),
    yaw_range: tuple[float, float] = (-0.2, 0.2),
    offset_distance: float = 0.1,
    resampling_time_range: tuple[float, float] = (4.0, 4.0),
    ramp_time_s: float = 0.0,
    debug_vis: bool = False,
    # debug markers
    target_marker_radius: float = 0.03,
    ee_pos_marker_radius: float = 0.02,
    # 坐标系原点选择
    use_projected_origin: bool = True,
    # 地面高度低通滤波系数（仅 projected-origin 模式生效）
    ground_z_lpf_alpha: float = 1.0,
) -> ProjectedComTargetPointsCommandCfg:
    """创建目标点命令配置（main/x/y 三点）。

    Args:
        use_projected_origin: 坐标系原点选择
            - True（默认）: 使用躯干质心在地面的投影作为原点（projected COM）
            - False: 使用躯干质心的实际位置作为原点（trunk COM，不投影到地面）
        ground_z_lpf_alpha: 地面高度低通滤波系数（1.0 表示不滤波）
    """
    from .projected_com_target_points_command import (
        ProjectedComTargetPointsCommand,
    )

    cfg = ProjectedComTargetPointsCommandCfg(
        class_type=ProjectedComTargetPointsCommand,
        asset_name=asset_name,
        trunk_body_name=trunk_body_name,
        terrain_sensor_name=terrain_sensor_name,
        ee_body_name=ee_body_name,
        offset_distance=offset_distance,
        resampling_time_range=resampling_time_range,
        ramp_time_s=ramp_time_s,
        debug_vis=debug_vis,
        use_projected_origin=use_projected_origin,
        ground_z_lpf_alpha=ground_z_lpf_alpha,
    )
    cfg.ranges.pos_r = pos_r_range
    cfg.ranges.pos_theta = pos_theta_range
    cfg.ranges.pos_z = pos_z_range
    cfg.ranges.roll = roll_range
    cfg.ranges.pitch = pitch_range
    cfg.ranges.yaw = yaw_range

    # Debug marker sizes (avoid mutating global marker singletons)
    cfg.target_main_visualizer_cfg = GREEN_SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/ee_target_main",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=float(target_marker_radius),
                visual_material=GREEN_SPHERE_MARKER_CFG.markers[
                    "sphere"
                ].visual_material,
            )
        },
    )
    cfg.target_x_offset_visualizer_cfg = RED_SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/ee_target_x",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=float(target_marker_radius),
                visual_material=RED_SPHERE_MARKER_CFG.markers[
                    "sphere"
                ].visual_material,
            )
        },
    )
    cfg.target_y_offset_visualizer_cfg = BLUE_SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/ee_target_y",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=float(target_marker_radius),
                visual_material=BLUE_SPHERE_MARKER_CFG.markers[
                    "sphere"
                ].visual_material,
            )
        },
    )
    cfg.ee_pos_visualizer_cfg = YELLOW_SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/ee_current_pos",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=float(ee_pos_marker_radius),
                visual_material=YELLOW_SPHERE_MARKER_CFG.markers[
                    "sphere"
                ].visual_material,
            )
        },
    )
    return cfg

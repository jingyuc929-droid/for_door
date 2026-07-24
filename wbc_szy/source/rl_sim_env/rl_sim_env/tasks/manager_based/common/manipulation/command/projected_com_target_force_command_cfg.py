"""机械臂/操作相关：COM 投影参考系目标力命令配置（6D）。

该模块只服务于 force_control 任务：命令为 main 位置 + target_force。
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers.command_manager import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from rl_debug.marker import (
    GREEN_SPHERE_MARKER_CFG,
    YELLOW_SPHERE_MARKER_CFG,
)


@configclass
class ProjectedComTargetForceCommandCfg(CommandTermCfg):
    """目标力命令配置（COM 投影参考系，6D：main 位置 + target_force）。"""

    class_type: type = MISSING  # 使用处注入，避免循环依赖

    asset_name: str = MISSING
    trunk_body_name: str = "base_link"
    terrain_sensor_name: str = "height_scanner"

    # 末端刚体名（用于计算 tracking metrics；不影响命令生成本身）
    ee_body_name: str | None = None

    # 坐标系原点选择：
    # - True（默认）: 使用躯干质心在地面的投影作为原点（projected COM）
    # - False: 使用躯干质心的实际位置作为原点（trunk COM，不投影到地面）
    use_projected_origin: bool = True

    # 地面高度低通滤波系数（仅在 use_projected_origin=True 时生效）
    ground_z_lpf_alpha: float = 1.0

    @configclass
    class Ranges:
        # main 目标位置（柱坐标，projected COM yaw frame）
        pos_r: tuple[float, float] = (0.3, 0.8)
        pos_theta: tuple[float, float] = (-1.57, 1.57)
        pos_z: tuple[float, float] = (0.1, 0.5)

    ranges: Ranges = Ranges()

    # 位置 main 过渡插值时间（秒）。
    ramp_time_s: float = 0.0

    # target_force 独立 interval（秒，per-env 随机重采样新目标力）+ 线性 ramp（秒）
    # 力幅值跟外力课程同步：env.ee_force_curriculum_current_max
    force_interval_range_s: tuple[float, float] = (6.0, 8.0)
    force_ramp_time_s: float = 1.0

    target_main_visualizer_cfg: VisualizationMarkersCfg = (
        GREEN_SPHERE_MARKER_CFG.replace(
            prim_path="/Visuals/Command/ee_target_main"
        )
    )

    # 末端实际位置可视化（用于与目标点重合情况对比）
    ee_pos_visualizer_cfg: VisualizationMarkersCfg = (
        YELLOW_SPHERE_MARKER_CFG.replace(
            prim_path="/Visuals/Command/ee_current_pos"
        )
    )

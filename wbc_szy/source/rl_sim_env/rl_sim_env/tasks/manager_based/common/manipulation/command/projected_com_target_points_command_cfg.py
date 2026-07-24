"""机械臂/操作相关：COM 投影参考系目标点命令配置。

该模块专门服务于 whole-body manipulation，可独立迁移到其它任务。
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers.command_manager import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from rl_debug.marker import (
    BLUE_SPHERE_MARKER_CFG,
    GREEN_SPHERE_MARKER_CFG,
    RED_SPHERE_MARKER_CFG,
    YELLOW_SPHERE_MARKER_CFG,
)


@configclass
class ProjectedComTargetPointsCommandCfg(CommandTermCfg):
    """目标点命令配置（COM 投影参考系，9D：main/x/y）。"""

    class_type: type = MISSING  # 使用处注入，避免循环依赖

    asset_name: str = MISSING
    trunk_body_name: str = "base_link"
    terrain_sensor_name: str = "height_scanner"

    # 末端刚体名（用于计算 tracking metrics；不影响命令生成本身）
    ee_body_name: str | None = None

    # 是否将末端位置和姿态跟踪指标记为当前 episode 的直接 MAE。
    # - False（默认）: 保持历史行为，仅记录按 command duration
    #   归一化的位置误差与姿态测地角误差累加值。
    # - True: 将 main 位置与 TCP 姿态测地角误差记为逐环境在线均值，
    #   并记录各自的有效样本数。
    use_episode_mae_metrics: bool = False

    # 末端工具系相对 ee_body_name/link6 局部系的旋转 (w, x, y, z)。
    # reward 侧读取该字段，使 piper link6 系可对齐到实际 TCP 工具系。
    ee_tool_quat_offset: tuple[float, float, float, float] = (
        1.0,
        0.0,
        0.0,
        0.0,
    )

    # 坐标系原点选择：
    # - True（默认）: 使用躯干质心在地面的投影作为原点（projected COM）
    # - False: 使用躯干质心的实际位置作为原点（trunk COM，不投影到地面）
    use_projected_origin: bool = True

    # 地面高度低通滤波系数（仅在 use_projected_origin=True 时生效）
    # y_t = alpha * x_t + (1 - alpha) * y_{t-1}
    # - 1.0: 不滤波（默认）
    # - 越小越平滑，但延迟越大（建议 0.1 ~ 0.5）
    ground_z_lpf_alpha: float = 1.0

    @configclass
    class Ranges:
        pos_r: tuple[float, float] = (0.3, 0.8)
        pos_theta: tuple[float, float] = (-1.57, 1.57)
        pos_z: tuple[float, float] = (0.1, 0.5)
        roll: tuple[float, float] = (-0.2, 0.2)
        pitch: tuple[float, float] = (-0.2, 0.2)
        yaw: tuple[float, float] = (-0.2, 0.2)

    ranges: Ranges = Ranges()

    offset_distance: float = 0.1

    # 目标点过渡插值时间（秒）。
    # - 0.0: 保持旧行为：每次 resample 直接“闪现”到新目标。
    # - >0: 在 ramp_time_s 内从当前 command 线性插值到新目标（9D 同步插值）。
    ramp_time_s: float = 0.0

    target_main_visualizer_cfg: VisualizationMarkersCfg = (
        GREEN_SPHERE_MARKER_CFG.replace(
            prim_path="/Visuals/Command/ee_target_main"
        )
    )
    target_x_offset_visualizer_cfg: VisualizationMarkersCfg = (
        RED_SPHERE_MARKER_CFG.replace(
            prim_path="/Visuals/Command/ee_target_x"
        )
    )
    target_y_offset_visualizer_cfg: VisualizationMarkersCfg = (
        BLUE_SPHERE_MARKER_CFG.replace(
            prim_path="/Visuals/Command/ee_target_y"
        )
    )

    # 末端实际位置可视化（用于与目标点重合情况对比）
    ee_pos_visualizer_cfg: VisualizationMarkersCfg = (
        YELLOW_SPHERE_MARKER_CFG.replace(
            prim_path="/Visuals/Command/ee_current_pos"
        )
    )

"""文本可视化工具：用于在Isaac Sim视口中显示实时信息（如速度、命令等）。"""

import carb
import numpy as np
import torch
from omni.isaac.debug_draw import _debug_draw
from isaaclab.assets import Articulation


class TextVisualizer:
    """在Isaac Sim视口中显示实时文本信息的可视化工具。

    使用Isaac Sim的DebugDrawAPI在屏幕上绘制文本，用于显示速度、命令等实时信息。
    """

    def __init__(self):
        """初始化文本可视化器。"""
        self._debug_draw = _debug_draw.acquire_debug_draw_interface()
        self._lines = []

    def clear(self):
        """清除所有绘制的文本。"""
        self._debug_draw.clear_lines()

    def draw_text_2d(
        self,
        text: str,
        position: tuple[int, int] = (50, 50),
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        size: int = 20,
    ):
        """在2D屏幕位置绘制文本。

        Args:
            text: 要显示的文本内容
            position: 屏幕位置(x, y)，左上角为(0,0)
            color: RGB颜色，范围0-1
            size: 字体大小
        """
        # 将颜色转换为0-255范围
        color_uint8 = (
            int(color[0] * 255),
            int(color[1] * 255),
            int(color[2] * 255),
        )
        self._debug_draw.draw_text_2d(
            text,
            position,
            color_uint8,
            size,
        )

    def draw_text_background(
        self,
        text: str,
        position: tuple[int, int] = (50, 50),
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        size: int = 20,
        bg_color: tuple[float, float, float] = (0.0, 0.0, 0.5),
        padding: int = 5,
    ):
        """绘制带背景的文本，提高可读性。

        Args:
            text: 要显示的文本内容
            position: 屏幕位置(x, y)
            color: 文本RGB颜色
            size: 字体大小
            bg_color: 背景RGB颜色
            padding: 背景边距
        """
        # 先绘制背景（使用半透明矩形）
        bg_color_uint8 = (
            int(bg_color[0] * 255),
            int(bg_color[1] * 255),
            int(bg_color[2] * 255),
        )
        # 估算文本宽度（每个字符约8像素）
        text_width = len(text) * 8 + 2 * padding
        text_height = size + 2 * padding

        # 绘制背景矩形
        self._debug_draw.draw_rectangle(
            (position[0] - padding, position[1] - padding),
            (text_width, text_height),
            bg_color_uint8,
            filled=True,
        )

        # 绘制文本
        self.draw_text_2d(text, position, color, size)

    def draw_robot_velocity(
        self,
        robot: Articulation,
        env_idx: int = 0,
        position: tuple[int, int] = (50, 50),
    ):
        """显示机器人的实际速度和命令速度。

        Args:
            robot: 机器人Articulation对象
            env_idx: 要显示的环境索引
            position: 屏幕显示位置
        """
        if not robot.is_initialized:
            return

        # 获取实际速度（世界坐标系）
        base_lin_vel_w = robot.data.root_lin_vel_w[env_idx]
        base_ang_vel_w = robot.data.root_ang_vel_w[env_idx]

        # 获取命令速度（如果可用）
        cmd_vel = None
        cmd_mgr = getattr(robot, "_env", None)
        if cmd_mgr is not None:
            command_manager = getattr(cmd_mgr, "command_manager", None)
            if command_manager is not None:
                base_cmd = command_manager.get_command("base_command")
                if base_cmd is not None and len(base_cmd) > env_idx:
                    cmd_vel = base_cmd[env_idx]

        # 准备显示文本
        lines = [
            f"=== Robot Velocity (Env {env_idx}) ===",
            f"Linear Vel X: {base_lin_vel_w[0]:.2f} m/s",
            f"Linear Vel Y: {base_lin_vel_w[1]:.2f} m/s",
            f"Linear Vel Z: {base_lin_vel_w[2]:.2f} m/s",
            f"Angular Vel X: {base_ang_vel_w[0]:.2f} rad/s",
            f"Angular Vel Y: {base_ang_vel_w[1]:.2f} rad/s",
            f"Angular Vel Z: {base_ang_vel_w[2]:.2f} rad/s",
        ]

        # 添加命令速度
        if cmd_vel is not None:
            lines.extend([
                "",
                f"=== Command Velocity ===",
                f"Cmd Lin X: {cmd_vel[0]:.2f} m/s",
                f"Cmd Lin Y: {cmd_vel[1]:.2f} m/s",
                f"Cmd Ang Z: {cmd_vel[2]:.2f} rad/s",
            ])

        # 绘制文本（每行间隔20像素）
        y_offset = position[1]
        for line in lines:
            self.draw_text_2d(line, (position[0], y_offset), size=16)
            y_offset += 20


# 全局单例
_global_visualizer: TextVisualizer | None = None


def get_text_visualizer() -> TextVisualizer:
    """获取全局文本可视化器单例。"""
    global _global_visualizer
    if _global_visualizer is None:
        _global_visualizer = TextVisualizer()
    return _global_visualizer


def draw_velocity_info(
    robot: Articulation,
    env_idx: int = 0,
    position: tuple[int, int] = (50, 50),
):
    """便捷函数：绘制机器人速度信息。

    Args:
        robot: 机器人Articulation对象
        env_idx: 要显示的环境索引
        position: 屏幕显示位置
    """
    visualizer = get_text_visualizer()
    visualizer.draw_robot_velocity(robot, env_idx, position)

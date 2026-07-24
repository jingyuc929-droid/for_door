"""机械臂/操作相关：COM 投影参考系目标点命令生成器（9D）。"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from rl_sim_env.tasks.manager_based.common.utils.projected_com_frame import (
    compute_projected_com_yaw_frame,
    compute_trunk_com_yaw_frame,
    world_to_projected_frame,
)

from .projected_com_target_points_command_cfg import (
    ProjectedComTargetPointsCommandCfg,
)


class ProjectedComTargetPointsCommand(CommandTerm):
    """在 COM 投影参考系下采样目标点三元组（main/x/y）。"""

    cfg: ProjectedComTargetPointsCommandCfg

    def __init__(
        self, cfg: ProjectedComTargetPointsCommandCfg, env: ManagerBasedEnv
    ):
        # NOTE:
        # 父类 CommandTerm.__init__ 会调用 set_debug_vis() -> _set_debug_vis_impl()。
        # 因此所有在 _set_debug_vis_impl() 内会访问的成员必须在 super().__init__()
        # 之前初始化，否则会在构造阶段触发 AttributeError。
        self._debug_markers_ready = False

        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.trunk_body_id = self.robot.find_bodies(
            cfg.trunk_body_name
        )[0][0]

        # 坐标系原点模式
        self.use_projected_origin = getattr(cfg, "use_projected_origin", True)
        self.ground_z_lpf_alpha = float(
            getattr(cfg, "ground_z_lpf_alpha", 1.0)
        )
        self._lpf_state_key = str(getattr(cfg, "name", "ee_target_points"))

        # terrain sensor 仅在使用投影原点时需要
        self.terrain_sensor: RayCaster | None = None
        if self.use_projected_origin:
            if cfg.terrain_sensor_name not in env.scene.sensors:
                raise KeyError(
                    "找不到 terrain sensor: "
                    f"{cfg.terrain_sensor_name}，请确认 scene 里已挂载。"
                )
            self.terrain_sensor = env.scene.sensors[cfg.terrain_sensor_name]

        self.offset_distance = float(cfg.offset_distance)
        self.ranges = cfg.ranges

        self.command_buf = torch.zeros((self.num_envs, 9), device=self.device)
        # smooth ramp buffers (start/target in projected frame)
        self._cmd_start = torch.zeros((self.num_envs, 9), device=self.device)
        self._cmd_target = torch.zeros((self.num_envs, 9), device=self.device)
        self._ramp_elapsed_s = torch.zeros(
            (self.num_envs,), device=self.device
        )

        # ee body id (optional): used for tracking metrics
        self.ee_body_id: int | None = None
        if cfg.ee_body_name is not None:
            self.ee_body_id = self.robot.find_bodies(cfg.ee_body_name)[0][0]

        # metrics (logged as Metrics/ee_target_points/...)
        self.metrics["error_ee_main_pos"] = torch.zeros(
            self.num_envs, device=self.device
        )

        # 预初始化一次 projected ground_z 缓存，确保 reward 侧严格只读模式可用。
        if self.use_projected_origin:
            _ = self._get_reference_frame()

    @property
    def command(self) -> torch.Tensor:
        return self.command_buf

    def _get_reference_frame(self):
        """根据配置获取当前参考坐标系。"""
        if self.use_projected_origin:
            return compute_projected_com_yaw_frame(
                asset=self.robot,
                trunk_body_id=self.trunk_body_id,
                terrain_height_sensor=self.terrain_sensor,
                ground_z_lpf_alpha=self.ground_z_lpf_alpha,
                lpf_state_key=self._lpf_state_key,
            )
        else:
            return compute_trunk_com_yaw_frame(
                asset=self.robot,
                trunk_body_id=self.trunk_body_id,
            )

    def _resample_command(self, env_ids: Sequence[int]):
        num = len(env_ids)
        r = torch.empty((num,), device=self.device).uniform_(
            *self.ranges.pos_r
        )
        theta = torch.empty((num,), device=self.device).uniform_(
            *self.ranges.pos_theta
        )
        z = torch.empty((num,), device=self.device).uniform_(
            *self.ranges.pos_z
        )

        main = torch.zeros((num, 3), device=self.device)
        main[:, 0] = r * torch.cos(theta)
        main[:, 1] = r * torch.sin(theta)
        main[:, 2] = z

        roll = torch.empty((num,), device=self.device).uniform_(
            *self.ranges.roll
        )
        pitch = torch.empty((num,), device=self.device).uniform_(
            *self.ranges.pitch
        )
        yaw = torch.empty((num,), device=self.device).uniform_(
            *self.ranges.yaw
        )
        rot_q = quat_from_euler_xyz(roll, pitch, yaw)

        local_x = torch.zeros((num, 3), device=self.device)
        local_y = torch.zeros((num, 3), device=self.device)
        local_x[:, 0] = self.offset_distance
        local_y[:, 1] = self.offset_distance

        x_off = main + quat_apply(rot_q, local_x)
        y_off = main + quat_apply(rot_q, local_y)

        env_ids_t = torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long
        )
        new_target = torch.zeros((num, 9), device=self.device)
        new_target[:, 0:3] = main
        new_target[:, 3:6] = x_off
        new_target[:, 6:9] = y_off

        ramp_time_s = float(getattr(self.cfg, "ramp_time_s", 0.0))
        if ramp_time_s <= 0.0:
            # keep legacy behavior: jump immediately
            self.command_buf[env_ids_t, :] = new_target
            self._cmd_target[env_ids_t, :] = new_target
            self._cmd_start[env_ids_t, :] = new_target
            self._ramp_elapsed_s[env_ids_t] = ramp_time_s
            return

        # start from current command and ramp to target
        self._cmd_start[env_ids_t, :] = self.command_buf[env_ids_t, :]
        self._cmd_target[env_ids_t, :] = new_target
        self._ramp_elapsed_s[env_ids_t] = 0.0

    def _update_command(self):
        # projected-origin 模式下每步刷新一次 ground_z 缓存，
        # 供 reward 侧严格只读（read_cached_ground_z_only=True）使用。
        if self.use_projected_origin:
            _ = self._get_reference_frame()

        ramp_time_s = float(getattr(self.cfg, "ramp_time_s", 0.0))
        if ramp_time_s <= 0.0:
            # make sure the command follows the latest target
            self.command_buf[:, :] = self._cmd_target
            return

        dt = float(self._env.step_dt)
        self._ramp_elapsed_s = torch.minimum(
            self._ramp_elapsed_s + dt,
            torch.full_like(self._ramp_elapsed_s, ramp_time_s),
        )
        frac = (
            (self._ramp_elapsed_s / ramp_time_s)
            .clamp_(0.0, 1.0)
            .unsqueeze(-1)
        )
        self.command_buf[:, :] = self._cmd_start + frac * (
            self._cmd_target - self._cmd_start
        )

    def _update_metrics(self):
        """更新末端跟随误差指标（用于训练日志上传）。"""
        if self.ee_body_id is None:
            return

        # normalize per command duration (match base_command metrics style)
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        frame = self._get_reference_frame()

        # command points in reference frame
        cmd_main = self.command_buf[:, 0:3]

        # ee main point in reference frame
        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_id]
        ee_pos_p = world_to_projected_frame(ee_pos_w, frame)

        err = torch.linalg.norm(cmd_main - ee_pos_p, dim=1)
        self.metrics["error_ee_main_pos"] += err / max_command_step

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis and not self._debug_markers_ready:
            self._vis_main = VisualizationMarkers(
                self.cfg.target_main_visualizer_cfg
            )
            self._vis_x = VisualizationMarkers(
                self.cfg.target_x_offset_visualizer_cfg
            )
            self._vis_y = VisualizationMarkers(
                self.cfg.target_y_offset_visualizer_cfg
            )
            self._vis_ee = VisualizationMarkers(
                self.cfg.ee_pos_visualizer_cfg
            )
            self._debug_markers_ready = True

        if self._debug_markers_ready:
            self._vis_main.set_visibility(debug_vis)
            self._vis_x.set_visibility(debug_vis)
            self._vis_y.set_visibility(debug_vis)
            self._vis_ee.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        if not self._debug_markers_ready or not self.robot.is_initialized:
            return

        frame = self._get_reference_frame()

        cmd = self.command_buf
        main_p = cmd[:, 0:3]
        x_p = cmd[:, 3:6]
        y_p = cmd[:, 6:9]

        main_w = frame.origin_w + quat_apply(frame.yaw_quat_w, main_p)
        x_w = frame.origin_w + quat_apply(frame.yaw_quat_w, x_p)
        y_w = frame.origin_w + quat_apply(frame.yaw_quat_w, y_p)

        q = torch.zeros((self.num_envs, 4), device=self.device)
        q[:, 0] = 1.0

        self._vis_main.visualize(main_w, q)
        self._vis_x.visualize(x_w, q)
        self._vis_y.visualize(y_w, q)

        # ee current position in world frame (if configured)
        if self.ee_body_id is not None:
            ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_id]
            self._vis_ee.visualize(ee_pos_w, q)

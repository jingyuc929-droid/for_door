"""机械臂/操作相关：COM 投影参考系目标点命令生成器（9D）。"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul

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


def _decode_target_rotation_6d(
    command: torch.Tensor, eps: float = 1.0e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode a rotation matrix from ``main/x/y`` target points.

    The two point offsets form the continuous 6D rotation representation.
    Invalid rows are returned as finite zero matrices and marked invalid so
    callers can skip them without contaminating running statistics.
    """
    if command.shape[-1] != 9:
        raise ValueError(
            "Expected a 9D main/x/y target-points command, "
            f"got shape {tuple(command.shape)}."
        )

    main = command[..., 0:3]
    x_raw = command[..., 3:6] - main
    y_raw = command[..., 6:9] - main
    finite = torch.isfinite(x_raw).all(dim=-1) & torch.isfinite(y_raw).all(
        dim=-1
    )

    # Sanitize before normalization: masking a NaN after division is too late
    # for some downstream operations even when that row is ultimately skipped.
    x_raw = torch.where(torch.isfinite(x_raw), x_raw, torch.zeros_like(x_raw))
    y_raw = torch.where(torch.isfinite(y_raw), y_raw, torch.zeros_like(y_raw))

    x_norm = torch.linalg.vector_norm(x_raw, dim=-1)
    x_axis = x_raw / x_norm.clamp_min(eps).unsqueeze(-1)

    # Gram--Schmidt: remove y's component along x, then normalize it.
    y_orthogonal = y_raw - (
        torch.sum(y_raw * x_axis, dim=-1, keepdim=True) * x_axis
    )
    y_norm = torch.linalg.vector_norm(y_orthogonal, dim=-1)
    y_axis = y_orthogonal / y_norm.clamp_min(eps).unsqueeze(-1)
    z_axis = torch.linalg.cross(x_axis, y_axis, dim=-1)

    valid = (
        finite
        & torch.isfinite(x_norm)
        & torch.isfinite(y_norm)
        & (x_norm > eps)
        & (y_norm > eps)
    )
    rotation = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    rotation = torch.where(
        valid[..., None, None], rotation, torch.zeros_like(rotation)
    )
    return rotation, valid


def _rotation_matrix_from_quaternion_wxyz(
    quaternion: torch.Tensor, eps: float = 1.0e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert scalar-first quaternions to finite rotation matrices."""
    if quaternion.shape[-1] != 4:
        raise ValueError(
            "Expected scalar-first quaternions with shape (..., 4), "
            f"got shape {tuple(quaternion.shape)}."
        )

    finite = torch.isfinite(quaternion).all(dim=-1)
    quaternion = torch.where(
        torch.isfinite(quaternion), quaternion, torch.zeros_like(quaternion)
    )
    quat_norm = torch.linalg.vector_norm(quaternion, dim=-1)
    valid = finite & torch.isfinite(quat_norm) & (quat_norm > eps)
    q = quaternion / quat_norm.clamp_min(eps).unsqueeze(-1)
    w, x, y, z = q.unbind(dim=-1)

    two = 2.0
    rotation = torch.stack(
        (
            1.0 - two * (y * y + z * z),
            two * (x * y - w * z),
            two * (x * z + w * y),
            two * (x * y + w * z),
            1.0 - two * (x * x + z * z),
            two * (y * z - w * x),
            two * (x * z - w * y),
            two * (y * z + w * x),
            1.0 - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)
    rotation = torch.where(
        valid[..., None, None], rotation, torch.zeros_like(rotation)
    )
    return rotation, valid


def _so3_geodesic_angle(
    target_rotation: torch.Tensor, current_rotation: torch.Tensor
) -> torch.Tensor:
    """Return the shortest SO(3) angle (radians) between two rotations."""
    relative_trace = torch.sum(target_rotation * current_rotation, dim=(-2, -1))
    cosine = ((relative_trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cosine)


def _update_running_mean(
    mean: torch.Tensor,
    sample_count: torch.Tensor,
    sample: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    """Update an in-place per-environment mean for the selected samples."""
    valid = valid & torch.isfinite(sample)
    valid_float = valid.to(dtype=sample_count.dtype)
    new_count = sample_count + valid_float
    safe_sample = torch.where(valid, sample, mean)
    mean.add_(
        (safe_sample - mean)
        * valid_float
        / new_count.clamp_min(1.0)
    )
    sample_count.copy_(new_count)


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
        self._use_episode_mae_metrics = bool(
            getattr(cfg, "use_episode_mae_metrics", False)
        )

        self.command_buf = torch.zeros((self.num_envs, 9), device=self.device)
        self.ee_target_points_error_raw = torch.zeros(
            (self.num_envs, 9), device=self.device
        )
        self.ee_target_points_error_step = -1
        self._cache_ee_target_points_error = False
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

        self._tool_quat_offset = torch.tensor(
            tuple(float(v) for v in cfg.ee_tool_quat_offset),
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.num_envs, 1)
        self._local_x = torch.zeros((self.num_envs, 3), device=self.device)
        self._local_y = torch.zeros((self.num_envs, 3), device=self.device)
        self._local_x[:, 0] = self.offset_distance
        self._local_y[:, 1] = self.offset_distance
        self._last_reference_frame = None
        self._last_reference_frame_step = -1
        # Opt-in episode metrics and the critic error cache run back-to-back in
        # CommandTerm.compute().  For the no-LPF path they observe the exact
        # same robot state and reference-frame object, so retain those two pose
        # tensors for that one command-manager step.  The step/frame guards in
        # _update_ee_target_points_error_cache prevent cross-step reuse.
        self._episode_metric_ee_pose_step = -1
        self._episode_metric_ee_pose_frame = None
        self._episode_metric_ee_pos_p: torch.Tensor | None = None
        self._episode_metric_ee_tcp_quat_p: torch.Tensor | None = None

        # metrics (logged as Metrics/ee_target_points/...)
        self.metrics["error_ee_main_pos"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_ee_orientation"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self._hidden_log_metrics = frozenset()
        if self._use_episode_mae_metrics:
            self.metrics["samples_ee_main_pos"] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics["samples_ee_orientation"] = torch.zeros(
                self.num_envs, device=self.device
            )
            # Sample counts validate the online means internally, but do not
            # carry task-performance information in the training dashboard.
            self._hidden_log_metrics = frozenset(
                {"samples_ee_main_pos", "samples_ee_orientation"}
            )

        # 预初始化一次 projected ground_z 缓存，确保 reward 侧严格只读模式可用。
        if self.use_projected_origin:
            _ = self._get_reference_frame()

    @property
    def command(self) -> torch.Tensor:
        return self.command_buf

    def enable_ee_target_points_error_cache(self) -> bool:
        """Enable per-step 9D EE error cache for critic observations."""
        self._cache_ee_target_points_error = self.ee_body_id is not None
        return self._cache_ee_target_points_error

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

    def _get_reusable_reference_frame(self):
        """Reuse the metrics frame within the same step when it is exactly safe."""
        if float(self.ground_z_lpf_alpha) < 1.0:
            return None
        step = int(getattr(self._env, "common_step_counter", -1))
        if self._last_reference_frame_step != step:
            return None
        return self._last_reference_frame

    def _update_ee_target_points_error_cache(self, frame):
        if not self._cache_ee_target_points_error or self.ee_body_id is None:
            return

        can_reuse_episode_metric_pose = False
        if (
            self._use_episode_mae_metrics
            and float(self.ground_z_lpf_alpha) >= 1.0
        ):
            current_step = int(
                getattr(self._env, "common_step_counter", -1)
            )
            can_reuse_episode_metric_pose = (
                self._episode_metric_ee_pose_step == current_step
                and self._episode_metric_ee_pose_frame is frame
                and self._episode_metric_ee_pos_p is not None
                and self._episode_metric_ee_tcp_quat_p is not None
            )
        if can_reuse_episode_metric_pose:
            ee_pos_p = self._episode_metric_ee_pos_p
            ee_quat_p = self._episode_metric_ee_tcp_quat_p
        else:
            ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_id]
            ee_quat_w = self.robot.data.body_quat_w[:, self.ee_body_id]

            ee_pos_p = world_to_projected_frame(ee_pos_w, frame)
            ee_quat_p = quat_mul(quat_inv(frame.yaw_quat_w), ee_quat_w)
            ee_quat_p = quat_mul(ee_quat_p, self._tool_quat_offset)

        ee_x_p = ee_pos_p + quat_apply(ee_quat_p, self._local_x)
        ee_y_p = ee_pos_p + quat_apply(ee_quat_p, self._local_y)

        self.ee_target_points_error_raw[:, 0:3] = ee_pos_p
        self.ee_target_points_error_raw[:, 3:6] = ee_x_p
        self.ee_target_points_error_raw[:, 6:9] = ee_y_p
        self.ee_target_points_error_raw -= self.command_buf
        self.ee_target_points_error_step = int(
            getattr(self._env, "common_step_counter", -1)
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
        frame = None
        if self.use_projected_origin:
            frame = self._get_reusable_reference_frame()
            if frame is None:
                frame = self._get_reference_frame()

        ramp_time_s = float(getattr(self.cfg, "ramp_time_s", 0.0))
        if ramp_time_s <= 0.0:
            # make sure the command follows the latest target
            self.command_buf[:, :] = self._cmd_target
        else:
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

        if self._cache_ee_target_points_error:
            if frame is None:
                frame = self._get_reference_frame()
            self._update_ee_target_points_error_cache(frame)

    def _update_metrics(self):
        """更新末端跟随误差指标（用于训练日志上传）。"""
        if self.ee_body_id is None:
            return

        frame = self._get_reference_frame()
        if float(self.ground_z_lpf_alpha) >= 1.0:
            self._last_reference_frame = frame
            self._last_reference_frame_step = int(
                getattr(self._env, "common_step_counter", -1)
            )

        # command points in reference frame
        cmd_main = self.command_buf[:, 0:3]

        # ee main point in reference frame
        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_id]
        ee_pos_p = world_to_projected_frame(ee_pos_w, frame)

        err = torch.linalg.norm(cmd_main - ee_pos_p, dim=1)

        # Decode the effective 9D target (including any current ramp fraction)
        # and compare it with the TCP orientation in the same reference frame.
        target_rotation, target_valid = _decode_target_rotation_6d(
            self.command_buf
        )
        ee_quat_w = self.robot.data.body_quat_w[:, self.ee_body_id]
        ee_quat_p = quat_mul(quat_inv(frame.yaw_quat_w), ee_quat_w)
        ee_tcp_quat_p = quat_mul(ee_quat_p, self._tool_quat_offset)
        current_rotation, current_valid = (
            _rotation_matrix_from_quaternion_wxyz(ee_tcp_quat_p)
        )
        orientation_error = _so3_geodesic_angle(
            target_rotation, current_rotation
        )
        orientation_valid = (
            target_valid
            & current_valid
            & torch.isfinite(orientation_error)
        )

        if not self._use_episode_mae_metrics:
            # Normalize per command duration, matching base-command metrics.
            max_command_time = self.cfg.resampling_time_range[1]
            max_command_step = max_command_time / self._env.step_dt
            self.metrics["error_ee_main_pos"] += err / max_command_step
            self.metrics["error_ee_orientation"] += torch.where(
                orientation_valid,
                orientation_error,
                torch.zeros_like(orientation_error),
            ) / max_command_step
            return

        # CommandManager also computes once immediately after an environment is
        # reset.  That frame has not executed the new command yet (and a ramped
        # command may still contain the previous target), so it is not part of
        # the new episode's tracking MAE.
        has_started_episode = self._env.episode_length_buf > 0
        position_valid = (
            has_started_episode
            & torch.isfinite(cmd_main).all(dim=-1)
            & torch.isfinite(ee_pos_p).all(dim=-1)
            & torch.isfinite(err)
        )
        _update_running_mean(
            self.metrics["error_ee_main_pos"],
            self.metrics["samples_ee_main_pos"],
            err,
            position_valid,
        )

        if (
            self._cache_ee_target_points_error
            and float(self.ground_z_lpf_alpha) >= 1.0
        ):
            self._episode_metric_ee_pose_step = int(
                getattr(self._env, "common_step_counter", -1)
            )
            self._episode_metric_ee_pose_frame = frame
            self._episode_metric_ee_pos_p = ee_pos_p
            self._episode_metric_ee_tcp_quat_p = ee_tcp_quat_p
        _update_running_mean(
            self.metrics["error_ee_orientation"],
            self.metrics["samples_ee_orientation"],
            orientation_error,
            has_started_episode & orientation_valid,
        )

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

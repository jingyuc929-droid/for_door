"""机械臂/操作相关：COM 投影参考系目标力命令生成器（6D）。

命令 6D = [main(3), target_force(3)]，均在 projected COM yaw frame。
- main（目标 ee 位置）：柱坐标采样，随基类 resampling_time_range + ramp_time_s 重采样。
- target_force（目标力）：独立 per-env interval（force_interval_range_s）+ 线性 ramp
  （force_ramp_time_s），幅值跟外力课程同步（env.ee_force_curriculum_current_max）。
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from rl_sim_env.tasks.manager_based.common.utils.projected_com_frame import (
    compute_projected_com_yaw_frame,
    compute_trunk_com_yaw_frame,
    world_to_projected_frame,
)

from .projected_com_target_force_command_cfg import (
    ProjectedComTargetForceCommandCfg,
)


class ProjectedComTargetForceCommand(CommandTerm):
    """在 COM 投影参考系下采样目标位置(main) + 目标力(target_force)。"""

    cfg: ProjectedComTargetForceCommandCfg

    def __init__(
        self, cfg: ProjectedComTargetForceCommandCfg, env: ManagerBasedEnv
    ):
        # NOTE: 父类 CommandTerm.__init__ 会调用 set_debug_vis() -> _set_debug_vis_impl()。
        # 因此所有在 _set_debug_vis_impl() 内会访问的成员必须在 super().__init__() 之前初始化。
        self._debug_markers_ready = False

        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.trunk_body_id = self.robot.find_bodies(
            cfg.trunk_body_name
        )[0][0]

        self.use_projected_origin = getattr(cfg, "use_projected_origin", True)
        self.ground_z_lpf_alpha = float(
            getattr(cfg, "ground_z_lpf_alpha", 1.0)
        )
        self._lpf_state_key = str(getattr(cfg, "name", "ee_target_points"))

        self.terrain_sensor: RayCaster | None = None
        if self.use_projected_origin:
            if cfg.terrain_sensor_name not in env.scene.sensors:
                raise KeyError(
                    "找不到 terrain sensor: "
                    f"{cfg.terrain_sensor_name}，请确认 scene 里已挂载。"
                )
            self.terrain_sensor = env.scene.sensors[cfg.terrain_sensor_name]

        self.ranges = cfg.ranges

        # 力独立 interval / ramp 参数
        self._force_interval_range_s = tuple(
            getattr(cfg, "force_interval_range_s", (6.0, 8.0))
        )
        self._force_ramp_time_s = float(
            getattr(cfg, "force_ramp_time_s", 1.0)
        )

        # command_buf 6D: [main(3), target_force(3)]
        self.command_buf = torch.zeros((self.num_envs, 6), device=self.device)
        # 位置 main ramp（随基类 resampling_time_range + ramp_time_s）
        self._main_start = torch.zeros((self.num_envs, 3), device=self.device)
        self._main_target = torch.zeros((self.num_envs, 3), device=self.device)
        self._main_ramp_elapsed_s = torch.zeros(
            (self.num_envs,), device=self.device
        )
        # target_force 独立 interval / ramp 状态机（per-env）
        self._force_next_resample_step = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.long
        )
        self._force_ramp_start = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self._force_ramp_target = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self._force_ramp_elapsed = torch.zeros(
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
        """基类 resampling_time_range 触发：只重采样位置 main。

        target_force 由独立 interval（_update_command/_update_force_command）维护，
        这里同时重置 reset envs 的力状态（next_resample_step=0 -> 下一步立即重采样）。
        """
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

        new_main = torch.zeros((num, 3), device=self.device)
        new_main[:, 0] = r * torch.cos(theta)
        new_main[:, 1] = r * torch.sin(theta)
        new_main[:, 2] = z

        env_ids_t = torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long
        )

        ramp_time_s = float(getattr(self.cfg, "ramp_time_s", 0.0))
        if ramp_time_s <= 0.0:
            # keep legacy behavior: jump immediately
            self.command_buf[env_ids_t, 0:3] = new_main
            self._main_target[env_ids_t] = new_main
            self._main_start[env_ids_t] = new_main
            self._main_ramp_elapsed_s[env_ids_t] = ramp_time_s
        else:
            # start from current main and ramp to target
            self._main_start[env_ids_t] = self.command_buf[env_ids_t, 0:3]
            self._main_target[env_ids_t] = new_main
            self._main_ramp_elapsed_s[env_ids_t] = 0.0

        # 重置 reset envs 的力状态：清零当前力，下次 _update_force_command 立即重采样
        self.command_buf[env_ids_t, 3:6] = 0.0
        self._force_ramp_start[env_ids_t] = 0.0
        self._force_ramp_target[env_ids_t] = 0.0
        self._force_ramp_elapsed[env_ids_t] = 0.0
        self._force_next_resample_step[env_ids_t] = 0

    def _update_command(self):
        # projected-origin 模式下每步刷新一次 ground_z 缓存，
        # 供 reward 侧严格只读（read_cached_ground_z_only=True）使用。
        # CommandTerm 的固定顺序是 _update_metrics -> resample -> _update_command。
        # 有 EE metric 且 alpha=1 时，metrics 已在同一仿真状态下完成了完全相同的
        # frame/cache 更新，第二次 E x rays 的 nearest-ground/get_coms 计算可省掉。
        # alpha<1 时保留历史上的双 LPF 更新语义；没有 EE metric 时也必须补算。
        if self.use_projected_origin and (
            self.ee_body_id is None or self.ground_z_lpf_alpha < 1.0
        ):
            _ = self._get_reference_frame()

        # 1) 位置 main ramp（前 3 维）
        ramp_time_s = float(getattr(self.cfg, "ramp_time_s", 0.0))
        if ramp_time_s <= 0.0:
            self.command_buf[:, 0:3] = self._main_target
        else:
            dt = float(self._env.step_dt)
            self._main_ramp_elapsed_s = torch.minimum(
                self._main_ramp_elapsed_s + dt,
                torch.full_like(self._main_ramp_elapsed_s, ramp_time_s),
            )
            frac = (
                (self._main_ramp_elapsed_s / ramp_time_s)
                .clamp_(0.0, 1.0)
                .unsqueeze(-1)
            )
            self.command_buf[:, 0:3] = self._main_start + frac * (
                self._main_target - self._main_start
            )

        # 2) 目标力独立 interval + ramp（后 3 维）
        self._update_force_command()

    def _update_force_command(self):
        """推进 target_force 的 per-env interval 重采样 + 1s 线性 ramp。

        力幅值跟外力课程同步：env.ee_force_curriculum_current_max（由外力课程写到 env 上）。
        未启用外力课程时 current_force_max=0，target_force 保持 0。
        """
        env = self._env
        step = int(getattr(env, "common_step_counter", 0))
        step_dt = float(env.step_dt)

        # 到期 env 重采样力目标
        due = step >= self._force_next_resample_step
        if due.any():
            due_idx = due.nonzero(as_tuple=False).squeeze(-1)
            n_due = int(due_idx.shape[0])
            cur_max = float(getattr(env, "ee_force_curriculum_current_max", 0.0))
            if cur_max > 0.0:
                new_force = torch.empty(
                    (n_due, 3), device=self.device
                ).uniform_(-cur_max, cur_max)
            else:
                new_force = torch.zeros((n_due, 3), device=self.device)
            # start = 当前 target_force（连续）
            self._force_ramp_start[due_idx] = self.command_buf[due_idx, 3:6]
            self._force_ramp_target[due_idx] = new_force
            self._force_ramp_elapsed[due_idx] = 0.0
            # 下次重采样 step（interval_range_s -> 步数，per-env 独立随机）
            interval_s = torch.empty(
                (n_due,), device=self.device
            ).uniform_(*self._force_interval_range_s)
            steps = (interval_s / step_dt).round().long().clamp(min=1)
            self._force_next_resample_step[due_idx] = step + steps

        # 推进 ramp lerp
        if self._force_ramp_time_s <= 0.0:
            self.command_buf[:, 3:6] = self._force_ramp_target
        else:
            dt = step_dt
            elapsed = self._force_ramp_elapsed + dt
            alpha = (
                (elapsed / self._force_ramp_time_s)
                .clamp_(0.0, 1.0)
                .unsqueeze(-1)
            )
            self.command_buf[:, 3:6] = self._force_ramp_start + alpha * (
                self._force_ramp_target - self._force_ramp_start
            )
            self._force_ramp_elapsed = elapsed

    def _update_metrics(self):
        """更新末端位置跟随误差指标（用于训练日志）。"""
        if self.ee_body_id is None:
            return

        # normalize per command duration (match base_command metrics style)
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        frame = self._get_reference_frame()

        # command main point in reference frame
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
            self._vis_ee = VisualizationMarkers(
                self.cfg.ee_pos_visualizer_cfg
            )
            self._debug_markers_ready = True

        if self._debug_markers_ready:
            self._vis_main.set_visibility(debug_vis)
            self._vis_ee.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        if not self._debug_markers_ready or not self.robot.is_initialized:
            return

        frame = self._get_reference_frame()

        cmd = self.command_buf
        main_p = cmd[:, 0:3]

        main_w = frame.origin_w + quat_apply(frame.yaw_quat_w, main_p)

        q = torch.zeros((self.num_envs, 4), device=self.device)
        q[:, 0] = 1.0

        self._vis_main.visualize(main_w, q)

        # ee current position in world frame (if configured)
        if self.ee_body_id is not None:
            ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_id]
            self._vis_ee.visualize(ee_pos_w, q)

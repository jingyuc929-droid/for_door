# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from turtle import back
from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster, FrameTransformer
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, yaw_quat
import torch.nn.functional as F
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.1
    return reward

def stuck(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), command_name: str = "base_command"):
    """Penalize stuck"""
    # Penalize stuck
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = (torch.abs(asset.data.root_lin_vel_b[:, 0]) < 0.1) * (torch.abs(env.command_manager.get_command(command_name)[:, 0]) > 0.1)
    return reward


def clearance(env: ManagerBasedRLEnv,target_clearance_height: float = 0.1, foot_asset_cfg: SceneEntityCfg = SceneEntityCfg(".*_foot")):
    """Reward clearance"""
    clearance = env.foot_clearance_buf
    foot_asset: RigidObject = env.scene[foot_asset_cfg.name]
    foot_lateral_vel = torch.norm(
        foot_asset.data.body_lin_vel_w[:, foot_asset_cfg.body_ids, :2],  # 取每个足端在 x、y 方向的速度
        dim=-1                                             # 对最后一维做范数：sqrt(vx^2 + vy^2)
    ) 
    reward = torch.sum(foot_lateral_vel*torch.square(clearance - target_clearance_height), dim=-1)
    return reward


def obstacle_avoidance_penalty(
    env: ManagerBasedRLEnv,
    max_avoidance_angle: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """根据机身的 yaw 朝向判断是否超出避障角度范围。

    逻辑：
    1. 从机身四元数中提取 yaw 角。
    2. 将机体系前向向量 [1, 0, 0] 通过该 yaw 旋转到世界系，得到 forward。
    3. 计算 heading = atan2(forward_y, forward_x)。
    4. 与 max_avoidance_angle 比较，返回 cheat = |heading| > max_avoidance_angle。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    quat_w = asset.data.root_quat_w  # (num_envs, 4) in (x, y, z, w)

    # 使用 quat_apply 将机体系前向向量 [1, 0, 0] 旋转到世界系
    forward_b = torch.tensor(
        [1.0, 0.0, 0.0],
        device=env.device,
        dtype=quat_w.dtype,
    ).expand(quat_w.shape[0], -1)  # (num_envs, 3)
    forward_w = quat_apply(quat_w, forward_b)  # (num_envs, 3)
    forward = forward_w[:, :2]  # 只取平面分量 (x, y)

    # 计算世界系下的 heading，并与阈值比较
    heading = torch.atan2(forward[:, 1], forward[:, 0])  # (num_envs,)
    cheat = torch.abs(heading) > max_avoidance_angle
    return cheat.float()


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    threshold: float,
    front_foot_sensor_cfg: SceneEntityCfg,
    back_foot_sensor_cfg: SceneEntityCfg,
    command_name: str = "base_command",
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    orientation_flag = env.command_manager.get_command(command_name)[:, 3]  # [num_envs]
    
    front_contact_sensor: ContactSensor = env.scene.sensors[front_foot_sensor_cfg.name]
    
    # 计算前足的奖励
    front_air_time = front_contact_sensor.data.current_air_time[:, front_foot_sensor_cfg.body_ids]
    front_contact_time = front_contact_sensor.data.current_contact_time[:, front_foot_sensor_cfg.body_ids]
    front_in_contact = front_contact_time > 0.0
    front_in_mode_time = torch.where(front_in_contact, front_contact_time, front_air_time)
    front_single_stance = torch.sum(front_in_contact.int(), dim=1) == 1
    front_reward = torch.min(torch.where(front_single_stance.unsqueeze(-1), front_in_mode_time, 0.0), dim=1)[0]
    front_reward = torch.clamp(front_reward, max=threshold)
    
    back_contact_sensor: ContactSensor = env.scene.sensors[back_foot_sensor_cfg.name]
    # 计算后足的奖励
    back_air_time = back_contact_sensor.data.current_air_time[:, back_foot_sensor_cfg.body_ids]
    back_contact_time = back_contact_sensor.data.current_contact_time[:, back_foot_sensor_cfg.body_ids]
    back_in_contact = back_contact_time > 0.0
    back_in_mode_time = torch.where(back_in_contact, back_contact_time, back_air_time)
    back_single_stance = torch.sum(back_in_contact.int(), dim=1) == 1
    back_reward = torch.min(torch.where(back_single_stance.unsqueeze(-1), back_in_mode_time, 0.0), dim=1)[0]
    back_reward = torch.clamp(back_reward, max=threshold)
    
    # 根据 orientation_flag 选择对应的奖励
    # orientation_flag: 1=前腿倒立，后腿触地时间奖励, -1=后腿倒立, 前腿触地时间奖励, 0=正常行走
    front_mask = (orientation_flag == -1).float()  # [num_envs]
    back_mask = (orientation_flag == 1).float()  # [num_envs]
    
    reward = front_mask * front_reward + back_mask * back_reward
    
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.1
    
    # 正常行走时不给奖励
    reward = reward * (orientation_flag != 0).float()
    
    return reward


def flat_orientation_new_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    # return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)*(env.terrain_types==6)


def track_pitch_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "pitch_command",
    upright_scale_max: float | None = 0.7,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # debug visualization (optional, safe in headless)
    debug_vis: bool = False,
    debug_vis_height: float = 0.6,
    debug_vis_target_scale: tuple[float, float, float] = (0.5, 0.15, 0.15),
    debug_vis_current_scale: tuple[float, float, float] = (0.5, 0.15, 0.15),
    debug_vis_env_id: int = 0,
) -> torch.Tensor:
    """Reward tracking of the commanded base pitch angle using an exponential kernel.

    Notes:
    - Pitch is computed from the projected gravity vector in base frame (yaw-invariant).
    - The command is expected to be a 1D tensor (shape: (num_envs, 1) or (num_envs,)).
    - Optionally scales reward by an upright factor derived from projected gravity.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # commanded pitch (rad)
    cmd = env.command_manager.get_command(command_name)
    if cmd.ndim > 1:
        cmd = cmd.squeeze(-1)

    # current pitch (rad) from projected gravity in base frame (yaw-invariant)
    # projected_gravity_b is the gravity direction expressed in base frame.
    # For a pure pitch rotation p around +Y, projected_gravity_b ≈ [sin(p), 0, -cos(p)].
    # Therefore pitch can be recovered as atan2(gx, -gz), which does not depend on yaw.
    g_b = env.scene["robot"].data.projected_gravity_b
    pitch = torch.atan2(g_b[:, 0], -g_b[:, 2])

    # optional: visualize target vs current base orientation in world frame
    if debug_vis:
        _debug_vis_track_pitch_target(
            env=env,
            asset=asset,
            target_pitch=cmd,
            height=float(debug_vis_height),
            target_scale=debug_vis_target_scale,
            current_scale=debug_vis_current_scale,
            env_id=int(debug_vis_env_id),
        )

    pitch_error = torch.square(cmd - pitch)
    reward = torch.exp(-pitch_error / (std**2))

    if upright_scale_max is not None and float(upright_scale_max) > 0.0:
        z = -env.scene["robot"].data.projected_gravity_b[:, 2]
        reward *= torch.clamp(z, 0.0, float(upright_scale_max)) / float(upright_scale_max)

    return reward


def _debug_vis_track_pitch_target(
    env: "ManagerBasedRLEnv",
    asset: RigidObject,
    target_pitch: torch.Tensor,
    height: float,
    target_scale: tuple[float, float, float],
    current_scale: tuple[float, float, float],
    env_id: int,
):
    """Visualize target vs current base orientation for a single env id (best-effort, headless-safe).

    - target (green): yaw = current base yaw, pitch = target_pitch, roll = 0
    - current (blue): current base root_quat_w
    """
    # If rendering not available, skip.
    try:
        is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    except Exception:
        is_rendering = False
    if not is_rendering:
        return

    # lazy-init markers once
    if not hasattr(env, "track_pitch_target_visualizer") or not hasattr(env, "track_pitch_current_visualizer"):
        try:
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
        except Exception:
            return

        tgt_cfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Reward/track_pitch_target")
        cur_cfg = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Reward/track_pitch_current")
        # set default marker scales (can still be overridden per-call via visualize(scale))
        try:
            tgt_cfg.markers["arrow"].scale = target_scale
            cur_cfg.markers["arrow"].scale = current_scale
        except Exception:
            pass

        try:
            env.track_pitch_target_visualizer = VisualizationMarkers(tgt_cfg)
            env.track_pitch_current_visualizer = VisualizationMarkers(cur_cfg)
        except Exception:
            return

    # guard env_id
    if env_id < 0 or env_id >= int(env.num_envs):
        return

    import isaaclab.utils.math as math_utils
    import torch as _torch

    # marker position above base
    base_pos_w = asset.data.root_pos_w.clone()
    base_pos_w[:, 2] += float(height)
    pos = base_pos_w[env_id : env_id + 1]

    # current orientation (world)
    cur_quat_w = asset.data.root_quat_w[env_id : env_id + 1]

    # target orientation: keep current yaw, apply target pitch (world euler XYZ)
    yaw_q = math_utils.yaw_quat(cur_quat_w)
    _, _, yaw = euler_xyz_from_quat(yaw_q)
    zeros = _torch.zeros_like(yaw)
    tgt_pitch = target_pitch[env_id : env_id + 1].to(dtype=cur_quat_w.dtype)
    tgt_quat_w = math_utils.quat_from_euler_xyz(zeros, tgt_pitch, yaw)

    # scales (repeat for batch=1)
    tgt_scale = _torch.tensor(target_scale, device=pos.device, dtype=pos.dtype).unsqueeze(0)
    cur_scale = _torch.tensor(current_scale, device=pos.device, dtype=pos.dtype).unsqueeze(0)

    env.track_pitch_target_visualizer.visualize(pos, tgt_quat_w, tgt_scale)
    env.track_pitch_current_visualizer.visualize(pos, cur_quat_w, cur_scale)


def track_base_height_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward tracking of base height commands using exponential kernel."""
    base_height_obs = env.obs_tensor_dict['gt_base_height_obs'].squeeze(-1) / env.cfg.config_summary.observation.obs_term_dict['ground_truth_obs']['base_height_b_gt']['scale']
    base_height_error = torch.square(env.command_manager.get_command(command_name)[:, 3] - base_height_obs)
    # print("base_command: ", env.command_manager.get_command(command_name)[:, 3])
    reward = torch.exp(-base_height_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def base_height_from_transform_l2(env: ManagerBasedRLEnv, foot_tf_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, base_height_target: float) -> torch.Tensor:
    """Reward tracking of base height commands using exponential kernel."""
    sensor: FrameTransformer = env.scene.sensors[foot_tf_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    feet_pos_b_z = sensor.data.target_pos_source[:, :, 2].abs()  # (num_envs, num_feet, 3) in base frame

    # 若当前环境四足都未接触地面，则该项奖励为 0，且避免除以 0 导致的 NaN/Inf
    contacts_sum = contacts.sum(dim=1)  # (num_envs,)
    has_contact = contacts_sum > 0

    # 初始化 base_height 与 reward，默认全 0；只在有接触的环境上计算
    base_height = torch.zeros_like(contacts_sum, dtype=feet_pos_b_z.dtype)
    reward = torch.zeros_like(base_height)
    epsilon = 1e-6

    if has_contact.any():
        contacts_f = contacts.float()
        base_height_valid = (feet_pos_b_z[has_contact] * contacts_f[has_contact]).sum(dim=1) / (contacts_sum[has_contact] + epsilon)
        base_height[has_contact] = base_height_valid

        reward_valid = torch.square(base_height_valid - base_height_target)
        gravity_scale = torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        reward[has_contact] = reward_valid * gravity_scale[has_contact]

    return reward


def first_contact_foot_forces_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """在足端从未接触到接触的第一个时刻，惩罚**超过阈值部分**的接触力大小。

    具体逻辑：
    - 使用 ``ContactSensor.compute_first_contact`` 得到“第一次接触”的布尔掩码；
    - 从 ``net_forces_w_history`` 中取当前时刻（最近一帧）的接触力；
    - 对应位置做逐足端的力范数与 first_contact 掩码相乘，再在足端维度上求和。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # [num_envs, num_feet]，True 表示该步从“未接触”变为“接触”
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

    # 取当前时刻（历史缓冲区最后一帧）的接触力，shape: [num_envs, num_feet, 3]
    forces_curr = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, :]
    force_mag = torch.norm(forces_curr, dim=-1)  # [num_envs, num_feet]

    # 只惩罚超过阈值的那一部分：max(|F|-threshold, 0)
    excess_force = torch.clamp(force_mag - threshold, min=0.0)

    # 只在 first_contact 的足端上惩罚力大小，使用平方项
    reward = torch.sum(torch.square(excess_force) * first_contact, dim=1)
    return reward

def track_base_height_exp_partial(env: ManagerBasedRLEnv, std: float, height_target: float,
                foot_sensor_cfg: SceneEntityCfg = None, contact_sensor_cfg: SceneEntityCfg = None) -> torch.Tensor:
    """Reward tracking of base height commands using exponential kernel."""
    """Base height calculated from foot positions and contact states.
    
    Calculates the base height as the mean of foot positions relative to the base frame
    for feet that are in contact with the ground.
    """
    # 获取足端传感器（FrameTransformer）
    foot_sensor: FrameTransformer = env.scene.sensors[foot_sensor_cfg.name]
    # 获取足端相对于机身的位置，shape: (num_envs, num_feet, 3)
    foot_pos_b = foot_sensor.data.target_pos_source
    
    # 获取接触传感器
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    # 计算接触状态，shape: (num_envs, num_feet)
    contacts = contact_sensor.data.net_forces_w_history[:, :, contact_sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 0.5
    
    # 提取z方向位置，shape: (num_envs, num_feet)
    foot_z = foot_pos_b[:, :, 2]
    num_contacts = contacts.sum(dim=1, keepdim=True)
    
    # 如果有接触，使用接触足端的均值；否则使用所有足端的均值
    mean_z = torch.where(
        num_contacts > 0,
        (foot_z * contacts.float()).sum(dim=1, keepdim=True) / num_contacts,
        foot_z.mean(dim=1, keepdim=True)
    )
    base_height_obs = mean_z.abs().squeeze(-1)
    base_height_error = torch.square(height_target- base_height_obs)
    reward = torch.exp(-base_height_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def base_height_new_l2(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reward tracking of base height commands using exponential kernel."""
    base_height_obs = env.obs_tensor_dict['gt_base_height_obs'].squeeze(-1) / env.cfg.config_summary.observation.obs_term_dict['ground_truth_obs']['base_height_b_gt']['scale']
    # reward = torch.abs(env.command_manager.get_command(command_name)[:, 3] - base_height_obs)
    reward = torch.square(env.command_manager.get_command(command_name)[:, 3] - base_height_obs)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def body_stillness_within_window_reward(
    env: ManagerBasedRLEnv,
    threshold: float,
    window_seconds: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """当机身在过去 window_seconds 内的位移变化小于阈值时惩罚。

    判定逻辑：以“最近一次位移超过阈值的时刻”为参考点，若之后累计的位移始终小于阈值并持续时间达到 window_seconds，则视为静止惩罚。

    参数:
        threshold: 位移阈值（米）。
        window_seconds: 时间窗口长度（秒），默认 10s。
        asset_cfg: 机器人实体，默认 `robot`。
    返回:
        shape=(num_envs,) 的奖励值。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    curr_pos = asset.data.root_pos_w  # (num_envs, 3)

    # 需要的步数
    required_steps = max(1, int(window_seconds / float(env.step_dt)))

    # 懒初始化：参考位置与计数器（逐环境）
    if not hasattr(env, "_still_ref_pos_w"):
        env._still_ref_pos_w = curr_pos.clone()
    if not hasattr(env, "_still_counter"):
        env._still_counter = torch.zeros(curr_pos.shape[0], device=curr_pos.device, dtype=torch.long)

    # 计算与参考位置的位移（仅考虑 XY 平面）
    disp = torch.norm((curr_pos - env._still_ref_pos_w)[:, :2], dim=1)  # (num_envs,)

    # 位移超过阈值的环境：重置参考点与计数器
    moved_mask = disp > threshold
    if moved_mask.any():
        env._still_ref_pos_w[moved_mask] = curr_pos[moved_mask]
        env._still_counter[moved_mask] = 0

    # 位移未超过阈值的环境：计数 +1
    stay_mask = ~moved_mask
    if stay_mask.any():
        env._still_counter[stay_mask] += 1

    # 达到窗口步数则惩罚
    return (env._still_counter >= required_steps).float()
    

def lateral_foot_x_separation_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    foot_tf_cfg: SceneEntityCfg,
    threshold: float = 0.2,
) -> torch.Tensor:
    """
    惩罚左右脚在机身 x 方向上的间距差值过大。

    假设 `foot_asset_cfg.body_ids` 含 4 个脚，顺序为 [fl, fr, rl, rr]，
    分别计算：
        front_diff = |x_fl - x_fr|
        rear_diff  = |x_rl - x_rr|
    若 diff 大于速度相关阈值 `|vx| * threshold`，则按超出部分线性惩罚：
        penalty_pair = relu(diff - |vx| * threshold)
    最终奖励为前后两对脚惩罚之和。
    """
    # 使用 frame_transform 直接获取足端在机身坐标系下的位置
    # 这里假设 scene 中存在名为 "frame_transform" 的 FrameTransformer 传感器，
    # 且其 target_frames 顺序与 ROBOT_FOOT_NAMES 一致（[fl, fr, rl, rr]）。
    sensor: FrameTransformer = env.scene.sensors[foot_tf_cfg.name]
    feet_pos_b = sensor.data.target_pos_source  # (num_envs, num_feet, 3) in base frame

    # 需要 4 个足端
    if feet_pos_b.shape[1] != 4:
        return torch.zeros(env.scene.num_envs, device=env.device, dtype=torch.float32)

    # 机身坐标系下各足端的 x 坐标
    x_coords = feet_pos_b[..., 0]  # (num_envs, 4)

    # 顺序来自ROBOT_FOOT_NAMES = [ROBOT_FL_FOOT_LINK, ROBOT_FR_FOOT_LINK, ROBOT_RL_FOOT_LINK, ROBOT_RR_FOOT_LINK]
    front_diff = torch.abs(x_coords[:, 0] - x_coords[:, 1])  # (num_envs,)
    rear_diff = torch.abs(x_coords[:, 2] - x_coords[:, 3])   # (num_envs,)

    # 速度相关阈值：|vx| * threshold
    sep_threshold = torch.abs(env.command_manager.get_command(command_name)[:, 0]) * threshold  # (num_envs,)

    # 对超出阈值的部分施加线性惩罚
    reward = F.relu(front_diff - sep_threshold) + F.relu(rear_diff - sep_threshold)
    return reward


def foot_pos_x_forward(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, foot_tf_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, front_pos_x=0.15, rear_pos_x=-0.35) -> torch.Tensor:
    """奖励足端前进时靠前放置。"""
    sensor: FrameTransformer = env.scene.sensors[foot_tf_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    feet_pos_b = sensor.data.target_pos_source  # (num_envs, num_feet, 3) in base frame
    x_coords = feet_pos_b[..., 0]  # (num_envs, num_feet)
    vel_x_cmd = env.command_manager.get_command(command_name)[:, 0]  # (num_envs,)
    vel_x = vel_x_cmd.unsqueeze(1)  # (num_envs, 1)，便于在足端维度上广播
    # 前后两对足端分别相对于期望前/后位置的超前距离，逐足端计算
    front_diff = (x_coords[:, :2] - front_pos_x - vel_x * 0.2).clamp(min=0)  # (num_envs, 2)
    rear_diff = (x_coords[:, 2:] - rear_pos_x - vel_x * 0.2).clamp(min=0)   # (num_envs, 2)
    diff_all = torch.cat([front_diff, rear_diff], dim=1)  # (num_envs, 4)，与 first_contact 对齐
    reward = torch.sum(diff_all * first_contact, dim=1)
    reward *= (asset.data.projected_gravity_b[:, 2].abs() < 0.9).float()
    reward *= (vel_x_cmd > 0.1)
    return reward


def lateral_foot_z_separation_penalty(
    env: ManagerBasedRLEnv,
    foot_tf_cfg: SceneEntityCfg,
    threshold: float = 0.2,
) -> torch.Tensor:
    """
    惩罚左右脚在机身 z 方向上的间距差值过大。

    假设 `foot_asset_cfg.body_ids` 含 4 个脚，顺序为 [fl, fr, rl, rr]，
    分别计算：
        front_diff = |z_fl - z_fr|
        rear_diff  = |z_rl - z_rr|
    若 diff 大于阈值 `threshold`，则按超出部分线性惩罚：
        penalty_pair = relu(diff - threshold)
    最终奖励为前后两对脚惩罚之和。
    """
    # 使用 frame_transform 直接获取足端在机身坐标系下的位置
    # 这里假设 scene 中存在名为 "frame_transform" 的 FrameTransformer 传感器，
    # 且其 target_frames 顺序与 ROBOT_FOOT_NAMES 一致（[fl, fr, rl, rr]）。
    sensor: FrameTransformer = env.scene.sensors[foot_tf_cfg.name]
    feet_pos_b = sensor.data.target_pos_source  # (num_envs, num_feet, 3) in base frame

    # 需要 4 个足端
    if feet_pos_b.shape[1] != 4:
        return torch.zeros(env.scene.num_envs, device=env.device, dtype=torch.float32)

    # 机身坐标系下各足端的 x 坐标
    z_coords = feet_pos_b[..., 2]  # (num_envs, 4)

    # 顺序来自ROBOT_FOOT_NAMES = [ROBOT_FL_FOOT_LINK, ROBOT_FR_FOOT_LINK, ROBOT_RL_FOOT_LINK, ROBOT_RR_FOOT_LINK]
    front_diff = torch.abs(z_coords[:, 0] - z_coords[:, 1])  # (num_envs,)
    rear_diff = torch.abs(z_coords[:, 2] - z_coords[:, 3])   # (num_envs,)

    # 对超出阈值的部分施加线性惩罚
    reward = F.relu(front_diff - threshold) + F.relu(rear_diff - threshold)
    return reward


def hip_joint_penalty(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg,) -> torch.Tensor:
    """惩罚hip关节角度，并且根据机身的y或yaw速度指令大小来调整惩罚强度。
    
    当机身的y或yaw速度指令越大时，惩罚越小。这鼓励机器人在高速时允许更大的hip关节角度变化。
    
    Args:
        env: 环境实例
        command_name: 命令名称
        asset_cfg: 机器人资产配置，包含hip关节
    
    Returns:
        torch.Tensor: 惩罚奖励值
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 使用传入的关节配置获取hip关节
    hip_joint_ids = asset_cfg.joint_ids
    if not hip_joint_ids:
        return torch.zeros(env.num_envs, device=env.device)
    
    # 计算hip关节角度与默认角度的偏差
    hip_joint_pos = asset.data.joint_pos[:, hip_joint_ids]
    default_hip_angle = asset.data.default_joint_pos[:, hip_joint_ids]
    hip_angle_error = torch.sum(torch.abs(hip_joint_pos - default_hip_angle), dim=1)
    
    # 获取命令中的y和yaw速度分量
    # command = env.command_manager.get_command(command_name)
    # vel_y = torch.abs(command[:, 1])  # y方向速度
    # vel_yaw = torch.abs(command[:, 2])  # yaw角速度
    
    # # 计算最终惩罚（速度越大，惩罚越小）
    # reward = hip_angle_error * torch.logical_and(vel_y < 0.2, vel_yaw < 0.2).float()
    
    # 应用重力投影缩放（与机器人姿态相关）
    reward = hip_angle_error
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.1
    push_vel_gt = env.obs_tensor_dict['privileged_obs'][:, :2]
    reward *= torch.norm(push_vel_gt, dim=1) < 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    # reward *=((env.terrain_types == 6) | (env.terrain_types == 1) | (env.terrain_types == 5))

    
    return reward

def hip_joint_penalty_lateral_velocity(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg,) -> torch.Tensor:
    """惩罚hip关节角度，并且根据机身的y或yaw速度指令大小来调整惩罚强度。
    
    当机身的y或yaw速度指令越大时，惩罚越小。这鼓励机器人在高速时允许更大的hip关节角度变化。
    
    Args:
        env: 环境实例
        command_name: 命令名称
        asset_cfg: 机器人资产配置，包含hip关节
    
    Returns:
        torch.Tensor: 惩罚奖励值
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 使用传入的关节配置获取hip关节
    hip_joint_ids = asset_cfg.joint_ids
    if not hip_joint_ids:
        return torch.zeros(env.num_envs, device=env.device)
    
    # 计算hip关节角度与默认角度的偏差
    hip_joint_pos = asset.data.joint_pos[:, hip_joint_ids]
    default_hip_angle = asset.data.default_joint_pos[:, hip_joint_ids]
    hip_angle_error = torch.sum(torch.abs(hip_joint_pos - default_hip_angle), dim=1)
    
    # 获取命令中的y和yaw速度分量
    # command = env.command_manager.get_command(command_name)
    # vel_y = torch.abs(command[:, 1])  # y方向速度
    # vel_yaw = torch.abs(command[:, 2])  # yaw角速度
    
    # # 计算最终惩罚（速度越大，惩罚越小）
    # reward = hip_angle_error * torch.logical_and(vel_y < 0.2, vel_yaw < 0.2).float()
    
    # 应用重力投影缩放（与机器人姿态相关）
    reward = hip_angle_error
    reward *= torch.abs(env.command_manager.get_command(command_name)[:, 1]) > 0.1
    push_vel_gt = env.obs_tensor_dict['privileged_obs'][:, :2]
    reward *= torch.norm(push_vel_gt, dim=1) < 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    # reward *=((env.terrain_types == 6) | (env.terrain_types == 1) | (env.terrain_types == 5))

    
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Yaw-only aligned XY linear-velocity tracking (exponential kernel).

    This compares the commanded (vx, vy) against the yaw-aligned horizontal velocity components (roll/pitch ignored).
    Kept for backwards-compatibility and optional use, even though current configs prefer base-frame tracking.
    """
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def _clamp_target_xy_to_command_ranges(env, command_name: str, target_xy: torch.Tensor) -> torch.Tensor:
    """Clamp a target (vx, vy) to the configured command ranges.

    Supports both:
    - terrain-aware uniform velocity commands (per-env ranges via cfg.command_ids / cfg.ranges)
    - non-terrain uniform velocity commands (global ranges via cfg.ranges)
    """
    cmd_mgr = getattr(env, "command_manager", None)
    cmd_term = None
    if cmd_mgr is not None:
        # common internal containers (dict-like)
        for attr in ("_terms", "_command_terms", "terms"):
            container = getattr(cmd_mgr, attr, None)
            if container is None:
                continue
            try:
                candidate = container[command_name]
            except Exception:
                candidate = None
            if candidate is not None:
                cmd_term = candidate
                break
    cfg = getattr(cmd_term, "cfg", None)
    if cfg is None:
        return target_xy

    # Terrain command: cfg.ranges is a dict[str, Ranges], cfg.command_ids maps terrain-key -> env-id list
    if isinstance(getattr(cfg, "ranges", None), dict) and isinstance(getattr(cfg, "command_ids", None), dict):
        device = target_xy.device
        dtype = target_xy.dtype
        num_envs = target_xy.shape[0]
        lin_x_low = torch.full((num_envs,), -float("inf"), device=device, dtype=dtype)
        lin_x_high = torch.full((num_envs,), float("inf"), device=device, dtype=dtype)
        lin_y_low = torch.full((num_envs,), -float("inf"), device=device, dtype=dtype)
        lin_y_high = torch.full((num_envs,), float("inf"), device=device, dtype=dtype)
        for key, id_list in cfg.command_ids.items():
            if key not in cfg.ranges:
                continue
            if not id_list:
                continue
            ids_t = torch.tensor(id_list, device=device, dtype=torch.long)
            r = cfg.ranges[key]
            lin_x_low[ids_t] = float(r.lin_vel_x[0])
            lin_x_high[ids_t] = float(r.lin_vel_x[1])
            lin_y_low[ids_t] = float(r.lin_vel_y[0])
            lin_y_high[ids_t] = float(r.lin_vel_y[1])
        return torch.stack(
            (
                torch.clamp(target_xy[:, 0], min=lin_x_low, max=lin_x_high),
                torch.clamp(target_xy[:, 1], min=lin_y_low, max=lin_y_high),
            ),
            dim=-1,
        )

    # Non-terrain uniform command: cfg.ranges is an object with lin_vel_x/lin_vel_y fields
    r = getattr(cfg, "ranges", None)
    if r is None or not hasattr(r, "lin_vel_x") or not hasattr(r, "lin_vel_y"):
        return target_xy
    return torch.stack(
        (
            torch.clamp(target_xy[:, 0], min=float(r.lin_vel_x[0]), max=float(r.lin_vel_x[1])),
            torch.clamp(target_xy[:, 1], min=float(r.lin_vel_y[0]), max=float(r.lin_vel_y[1])),
        ),
        dim=-1,
    )


def track_lin_vel_xy_base_frame_exp_force_bias(
    env,
    std: float,
    command_name: str,
    force_to_vel_scale: float | tuple[float, float] = 0.0,
    force_mode: str = "components",
    force_clip: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Base-frame version of force-biased linear-velocity tracking (exponential kernel).

    This compares:
      - measured velocity: `asset.data.root_lin_vel_b[:, :2]` (full base frame, includes pitch/roll)
      - command: `env.command_manager.get_command(command_name)[:, :2]` (base frame)

    Force bias comes from `env.event_push_force_buf` which is expected to store the **applied world force**
    projected into the trunk/body frame (x/y).

    This is often easier to optimize early in training when trunk pitch/roll are randomized.

    Args:
        force_to_vel_scale: Scale from force to velocity bias. Can be a scalar or a tuple (scale_x, scale_y).
        force_mode:
            - "components": use per-axis force components (current behavior):
                bias_xy = force_xy * scale
            - "norm": use planar force magnitude (L2 norm) as a scalar "total force" (z excluded),
              while preserving each axis sign from force_xy:
                bias_xy = sign(force_xy) * ||force_xy|| * scale
    """
    asset = env.scene[asset_cfg.name]

    vel_xy = asset.data.root_lin_vel_b[:, :2]
    cmd_xy = env.command_manager.get_command(command_name)[:, :2]

    force_xy = getattr(env, "event_push_force_buf", None)
    if force_xy is None:
        force_xy = torch.zeros_like(cmd_xy)
    else:
        force_xy = force_xy.to(device=cmd_xy.device, dtype=cmd_xy.dtype)

    if force_clip is not None:
        force_xy = torch.clamp(force_xy, min=-float(force_clip), max=float(force_clip))

    force_mode_l = str(force_mode).lower()
    if force_mode_l == "components":
        # Per-axis force components.
        if isinstance(force_to_vel_scale, tuple):
            scale = torch.tensor(force_to_vel_scale, device=cmd_xy.device, dtype=cmd_xy.dtype)
            bias_xy = force_xy * scale.unsqueeze(0)
        else:
            bias_xy = force_xy * float(force_to_vel_scale)
    elif force_mode_l == "norm":
        # Use planar force magnitude as "total force" (z excluded), preserve x/y direction (sign).
        force_mag = torch.linalg.norm(force_xy, dim=-1, keepdim=True)  # (N, 1)
        force_sign = torch.sign(force_xy)  # (N, 2)
        if isinstance(force_to_vel_scale, tuple):
            scale = torch.tensor(force_to_vel_scale, device=cmd_xy.device, dtype=cmd_xy.dtype)  # (2,)
            bias_xy = force_sign * force_mag * scale.unsqueeze(0)  # (N, 2)
        else:
            bias_xy = force_sign * force_mag * float(force_to_vel_scale)  # (N, 2)
    else:
        raise ValueError(f"Unsupported force_mode: {force_mode!r}. Expected 'components' or 'norm'.")

    target_xy = cmd_xy + bias_xy
    target_xy = _clamp_target_xy_to_command_ranges(env=env, command_name=command_name, target_xy=target_xy)

    lin_vel_error = torch.sum(torch.square(target_xy - vel_xy), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp_torque_bias(
    env,
    std: float,
    command_name: str,
    torque_to_ang_vel_scale: float = 0.0,
    torque_clip: float | None = None,
    upright_scale_max: float | None = 0.7,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """在 yaw 角速度指令基础上，顺应外部 yaw 扭矩引导，跟随“偏置后的 yaw 角速度”(指数核)。

    目标角速度：
        ω_target_z = ω_cmd_z + torque_to_ang_vel_scale * clamp(τ_z)

    其中 τ_z 来自 env.event_push_yaw_torque_buf（形状 (N, 1) 或 (N,)）。
    """
    asset = env.scene[asset_cfg.name]

    # current yaw angular velocity (base frame)
    omega_z = asset.data.root_ang_vel_b[:, 2]

    # commanded yaw angular velocity
    cmd_z = env.command_manager.get_command(command_name)[:, 2]

    # external yaw torque (about world Z), from event buffer
    torque_z = getattr(env, "event_push_yaw_torque_buf", None)
    if torque_z is None:
        torque_z = torch.zeros_like(cmd_z)
    else:
        torque_z = torque_z.to(device=cmd_z.device, dtype=cmd_z.dtype)
        if torque_z.ndim > 1:
            torque_z = torque_z.squeeze(-1)

    if torque_clip is not None:
        torque_z = torch.clamp(torque_z, min=-float(torque_clip), max=float(torque_clip))

    target_z = cmd_z + torque_z * float(torque_to_ang_vel_scale)
    ang_vel_error = torch.square(target_z - omega_z)
    reward = torch.exp(-ang_vel_error / std**2)
    if upright_scale_max is not None and float(upright_scale_max) > 0.0:
        z = -env.scene["robot"].data.projected_gravity_b[:, 2]
        reward *= torch.clamp(z, 0.0, float(upright_scale_max)) / float(upright_scale_max)
    return reward


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def foot_pos_y_penalty(
    env: ManagerBasedRLEnv, 
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    target_y_offset: float = 0.0,
) -> torch.Tensor:
    """Penalize foot positions in the y direction relative to robot body frame.
    
    This function penalizes the deviation of foot positions from the target y-offset
    in the robot's body coordinate system.
    
    Args:
        env: The learning environment
        command_name: Name of the command
        sensor_cfg: Sensor configuration containing foot body IDs
        target_y_offset: Target y-offset for feet in body frame (default: 0.0)
    
    Returns:
        torch.Tensor: Penalty reward value
    """
    ##只有y和yaw方向速度小于0.1时才惩罚
    # extract the used quantities (to enable type-hinting)
    sensor: FrameTransformer = env.scene.sensors[sensor_cfg.name]
    # compute out of limits constraints
    feet_y_pos = sensor.data.target_pos_source[:, :, 1].flatten(start_dim=1)
    # 计算足端y轴偏移量与目标值的差距
    y_offset_error = torch.abs(torch.abs(feet_y_pos) - target_y_offset)  # [num_envs, num_feet]
    
    # 计算惩罚（对所有足端求和）
    reward = torch.sum(y_offset_error, dim=1)  # [num_envs]
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, 1:3], dim=1) < 0.1
    push_vel_gt = env.obs_tensor_dict['privileged_obs'][:, :2]
    reward *= torch.norm(push_vel_gt, dim=1) < 0.1
    # 根据机器人姿态调整惩罚强度（直立时惩罚更强）
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    
    return reward

def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_exp_wmp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # clip越接近于0，则允许超过命令速度的惩罚越小，越允许超过目标速度；否则，超过速度部分惩罚越大，越不允许超过目标速度
    clip = 0.2
    lin_vel = asset.data.root_lin_vel_b[:, :2].clone()
    vel_xy_command = env.command_manager.get_command(command_name)[:, :2]
    lin_vel_upper_bound = torch.where(vel_xy_command < 0, 1e5, vel_xy_command + clip)
    lin_vel_lower_bound = torch.where(vel_xy_command > 0, -1e5, vel_xy_command - clip)
    clip_lin_vel = torch.clip(lin_vel, lin_vel_lower_bound, lin_vel_upper_bound)
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(vel_xy_command - clip_lin_vel),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    target_clearance_height: float = 0.1,
    foot_asset_cfg: SceneEntityCfg = SceneEntityCfg(".*_foot"),
    std: float = 0.25,
    tanh_mult: float = 5.0,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground.

    参考常见的 Unitree/legged_gym 风格实现：使用足端相对地面的离地高度与目标高度之间的
    二次误差，并根据足端在平面内的线速度，用 ``tanh`` 做一个速度相关的权重，最后通过
    ``exp(-x / std)`` 把误差转成 (0, 1] 区间的奖励。

    参数：
    - target_clearance_height: 期望的离地高度。
    - foot_asset_cfg: 足端刚体的配置。
    - std: 指数核的尺度参数，对误差的敏感度。
    - tanh_mult: 乘在足端平面速度上的系数，控制 ``tanh`` 的饱和速度。

    注意：
    - 这里使用的是在观测中已经计算好的 ``env.foot_clearance_buf``，即足端相对地面的离地高度，
      而不是绝对的世界坐标 ``z``。
    - 如果调用方不传入 ``std`` 和 ``tanh_mult``，则使用默认值，行为与之前版本保持一致。
    """
    # 足端相对地面的离地高度（在 observations 中已经算好并存到了 env.foot_clearance_buf）
    # 形状: (num_envs, num_feet)
    clearance = env.foot_clearance_buf

    # 与期望离地高度之间的误差（二次项）
    foot_z_target_error = torch.square(clearance - target_clearance_height)

    # 足端在平面内 (x, y) 的线速度，用于构造摆动期相关的权重
    foot_asset: RigidObject = env.scene[foot_asset_cfg.name]
    foot_velocity = torch.norm(
        foot_asset.data.body_lin_vel_w[:, foot_asset_cfg.body_ids, :2],
        dim=-1,
    )  # 形状: (num_envs, num_feet)
    foot_velocity_tanh = torch.tanh(tanh_mult * foot_velocity)

    # 逐足端的误差 * 速度权重，然后在足端维度上求和
    weighted_error = foot_z_target_error * foot_velocity_tanh  # (num_envs, num_feet)
    summed_error = torch.sum(weighted_error, dim=1)  # (num_envs,)

    # 指数核，把误差映射到 (0, 1] 区间；误差越小说明高度越接近目标，奖励越接近 1
    reward = torch.exp(-summed_error / std)
    return reward


# 正奖励
class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        # 同步奖励：对角足的air和stand时间相同，如果相同，reward为0，否则最大0.02,相差越大，reward越小,exp->相同1,
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        # 左右，前后异步,鼓励swing和stand时间一样，如果相同reward为0,否则最大0.02，相差越大，reward越小
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """
    # 同步奖励，
    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)

def contact_reward(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # 参考：https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_env.py
    # 传入的足端传感器顺序必须是ROBOT_FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    is_stance = env.leg_phase < 0.65
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    reward = (contacts == is_stance).float().sum(dim=-1)
    return reward


def feet_slide_base_frame(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), body_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_asset = env.scene[body_asset_cfg.name]

    # 足端在世界系下的线速度 [B, num_feet, 3]
    foot_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :3]

    # 机身在世界系下的姿态 [B, 4]
    root_quat_w = body_asset.data.root_quat_w
    # 展开到每个足端 [B, num_feet, 4]
    root_quat_w_expanded = root_quat_w.unsqueeze(1).expand(-1, foot_vel_w.shape[1], -1)

    # 将足端速度从世界系变换到机身系（机身 b 坐标系）
    body_vel_b = quat_apply_inverse(
        root_quat_w_expanded.reshape(-1, 4),
        foot_vel_w.reshape(-1, 3),
    ).reshape_as(foot_vel_w)

    # 只保留机身系下与机身 xy 平面平行的分量（即 b 系的 x、y 方向）
    body_vel_xy_b = body_vel_b[:, :, :2]

    # 对机身 xy 方向上的速度范数进行惩罚（仅在当前有接触的足端上生效）
    reward = torch.sum(body_vel_xy_b.norm(dim=-1) * contacts, dim=1)
    return reward


def track_position_xy_yaw_frame(
    env, command_name: str
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    pos_xy_err = command[:, :2]
    remaining_time = env.remaining_episode_time
    reward_active = remaining_time < 1.0

    reward = 1.0 - 0.5 * torch.norm(pos_xy_err, dim=1)
    return reward * reward_active


def track_heading(
    env, command_name: str
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    heading_err = command[:, 3].unsqueeze(1)
    remaining_time = env.remaining_episode_time
    reward_active = remaining_time < 1.0
    reward = 1.0 - 0.5 * torch.norm(heading_err, dim=1)
    return reward * reward_active


# def track_lin_vel_xy_yaw_frame_exp(
#     env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
#     # extract the used quantities (to enable type-hinting)
#     asset = env.scene[asset_cfg.name]
#     vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
#     lin_vel_error = torch.sum(
#         torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
#     )
#     return torch.exp(-lin_vel_error / std**2)


# def track_ang_vel_z_world_exp(
#     env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
#     # extract the used quantities (to enable type-hinting)
#     asset = env.scene[asset_cfg.name]
#     ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
#     return torch.exp(-ang_vel_error / std**2)


def reached_joint_deviation_l2(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.norm(angle, dim=1)
    cmd = env.command_manager.get_command(command_name)
    pos_reached = torch.norm(cmd[:, :2], dim=1) < 0.25
    ang_reached = torch.norm(cmd[:, 3].unsqueeze(1), dim=1) < 0.5
    reward *= (pos_reached * ang_reached).float()
    return reward


def dont_wait_unreached(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    vel_xy_b = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    slow = vel_xy_b < 0.2
    cmd = env.command_manager.get_command(command_name)
    pos_reached = torch.norm(cmd[:, :2], dim=1) < 0.25
    ang_reached = torch.norm(cmd[:, 3].unsqueeze(1), dim=1) < 0.5
    reached = pos_reached & ang_reached
    return (slow & ~reached).float()


def move_in_direction(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), eps: float = 1e-6
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    v = asset.data.root_lin_vel_b[:, :2]                  # [B, 2]
    e = env.command_manager.get_command(command_name)[:, :2]  # [B, 2]

    # 单位化（避免 0 向量除法，加 eps）
    v_hat = F.normalize(v, p=2, dim=1, eps=eps)           # [B, 2]
    e_hat = F.normalize(e, p=2, dim=1, eps=eps)           # [B, 2]

    # 余弦相似度 = 单位向量点积
    cos = (v_hat * e_hat).sum(dim=1)                      # [B], ∈ [-1, 1]

    # 若速度或误差几乎为 0，置 0 以避免噪声
    nz = (v.norm(dim=1) > 1e-3) & (e.norm(dim=1) > 1e-3)
    cos = torch.where(nz, cos, torch.zeros_like(cos))

    return cos


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint power."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )


def joint_power_distribution(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint power distribution."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.var(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )


def joint_torque_distribution(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torque distribution."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.var(
        torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )


def amp_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward AMP."""
    # extract the used quantities (to enable type-hinting)
    if not hasattr(env, "amp_out"):
        return torch.zeros(env.scene.num_envs, device=env.device, dtype=torch.float32, requires_grad=False)
    reward = torch.clamp(1 - (1 / 4) * torch.square(env.amp_out - 1), min=0)
    return reward.squeeze()


def handstand_feet_height_exp(
    env: ManagerBasedRLEnv,
    std: float,
    front_target_height: float,
    back_target_height: float,
    front_foot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    back_foot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_command",
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    orientation_flag = env.command_manager.get_command(command_name)[:, 3]  # [num_envs]
    
    # 获取两组足端高度
    front_foot_asset: RigidObject = env.scene[front_foot_asset_cfg.name]
    back_foot_asset : RigidObject = env.scene[back_foot_asset_cfg.name]
    
    front_feet_height = front_foot_asset.data.body_pos_w[:, front_foot_asset_cfg.body_ids, 2]  # [num_envs, 2, 1]
    back_feet_height = back_foot_asset.data.body_pos_w[:, back_foot_asset_cfg.body_ids, 2]  # [num_envs, 2, 1]
    
    # 计算高度误差
    back_error = torch.sum(torch.square(back_feet_height - back_target_height), dim=1)  # [num_envs]
    front_error = torch.sum(torch.square(front_feet_height - front_target_height), dim=1)  # [num_envs]
    
    # 根据 orientation_flag 选择对应的误差
    # orientation_flag: 1=前腿倒立, -1=后腿倒立, 0=正常行走
    front_mask = (orientation_flag == 1).float()  # [num_envs]
    back_mask = (orientation_flag == -1).float()  # [num_envs]
    
    feet_height_error = front_mask * front_error + back_mask * back_error
    reward = torch.exp(-feet_height_error / std**2)
    
    # 正常行走时不给奖励
    reward = reward * (orientation_flag != 0).float()
    
    return reward


def handstand_feet_on_air(
    env: ManagerBasedRLEnv,
    front_foot_sensor_cfg: SceneEntityCfg,
    back_foot_sensor_cfg: SceneEntityCfg,
    command_name: str = "base_command",
) -> torch.Tensor:
    """检查倒立时的足端是否离地
    
    根据 orientation_flag 选择检查哪组足端：
    - orientation_flag == 1（前腿倒立）：检查前足是否离地
    - orientation_flag == -1（后腿倒立）：检查后足是否离地
    - orientation_flag == 0（正常行走）：不给奖励
    """
    # extract the used quantities (to enable type-hinting)
    orientation_flag = env.command_manager.get_command(command_name)[:, 3]  # [num_envs]
    
    front_contact_sensor: ContactSensor = env.scene.sensors[front_foot_sensor_cfg.name]
    back_contact_sensor: ContactSensor = env.scene.sensors[back_foot_sensor_cfg.name]
    
    # 获取前足和后足的离地状态
    front_first_air = front_contact_sensor.compute_first_air(env.step_dt)[:, front_foot_sensor_cfg.body_ids]
    back_first_air = back_contact_sensor.compute_first_air(env.step_dt)[:, back_foot_sensor_cfg.body_ids]
    
    # 检查所有足端是否都离地
    front_all_air = torch.all(front_first_air, dim=1).float()  # [num_envs]
    back_all_air = torch.all(back_first_air, dim=1).float()    # [num_envs]
    
    # 根据 orientation_flag 选择对应的奖励
    # orientation_flag: 1=前腿倒立, -1=后腿倒立, 0=正常行走
    front_mask = (orientation_flag == 1).float()   # [num_envs]
    back_mask = (orientation_flag == -1).float()  # [num_envs]
    
    reward = front_mask * front_all_air + back_mask * back_all_air
    
    # 正常行走时不给奖励（已经通过mask实现）
    return reward * (orientation_flag != 0).float()  # [num_envs]


def handstand_base_height_w_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    front_target_height: float = 0.6,
    back_target_height: float = 0.6,
    normal_target_height: float = 0.5,
    command_name: str = "base_command",
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    orientation_flag = env.command_manager.get_command(command_name)[:, 3]  # [num_envs]
    front_mask = (orientation_flag == 1).float()   # [num_envs]
    back_mask = (orientation_flag == -1).float()  # [num_envs]
    normal_mask = (orientation_flag == 0).float()  # [num_envs]
    target_height = front_mask * front_target_height + back_mask * back_target_height + normal_mask * normal_target_height
    return torch.square(asset.data.root_pos_w[:, 2] - target_height).float()


def handstand_feet_air_time(
    env: ManagerBasedRLEnv,
    front_foot_sensor_cfg: SceneEntityCfg,
    back_foot_sensor_cfg: SceneEntityCfg,
    threshold: float,
    command_name: str = "base_command",
) -> torch.Tensor:
    """奖励倒立时的足端滞空时间
    
    根据 orientation_flag 选择奖励哪组足端：
    - orientation_flag == 1（前腿倒立）：奖励前足的滞空时间
    - orientation_flag == -1（后腿倒立）：奖励后足的滞空时间
    - orientation_flag == 0（正常行走）：不给奖励
    """
    # extract the used quantities (to enable type-hinting)
    orientation_flag = env.command_manager.get_command(command_name)[:, 3]  # [num_envs]
    
    front_contact_sensor: ContactSensor = env.scene.sensors[front_foot_sensor_cfg.name]
    back_contact_sensor: ContactSensor = env.scene.sensors[back_foot_sensor_cfg.name]
    
    # 获取前足和后足的接触和滞空信息
    front_first_contact = front_contact_sensor.compute_first_contact(env.step_dt)[:, front_foot_sensor_cfg.body_ids]
    front_last_air_time = front_contact_sensor.data.last_air_time[:, front_foot_sensor_cfg.body_ids]
    
    back_first_contact = back_contact_sensor.compute_first_contact(env.step_dt)[:, back_foot_sensor_cfg.body_ids]
    back_last_air_time = back_contact_sensor.data.last_air_time[:, back_foot_sensor_cfg.body_ids]
    
    # 计算滞空时间奖励
    front_air_reward = torch.sum((front_last_air_time - threshold) * front_first_contact, dim=1)  # [num_envs]
    back_air_reward = torch.sum((back_last_air_time - threshold) * back_first_contact, dim=1)    # [num_envs]
    
    # 根据 orientation_flag 选择对应的奖励
    # orientation_flag: 1=前腿倒立, -1=后腿倒立, 0=正常行走
    front_mask = (orientation_flag == 1).float()   # [num_envs]
    back_mask = (orientation_flag == -1).float()  # [num_envs]
    
    reward = front_mask * front_air_reward + back_mask * back_air_reward
    
    # 正常行走时不给奖励（已经通过mask实现）
    return reward * (orientation_flag != 0).float()  # [num_envs]


def handstand_orientation_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), command_name: str = "base_command"
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # Define the target gravity direction for different orientations in the base frame
    orientation_flag = env.command_manager.get_command(command_name)[:, 3]  # [num_envs]
    
    # 初始化 target_gravity
    target_gravity = torch.zeros(env.scene.num_envs, 3, device=env.device)
    
    # orientation_flag == 1: 前腿倒立，目标重力方向为 [-1, 0, 0]
    front_handstand_mask = (orientation_flag == 1)
    target_gravity[front_handstand_mask, 0] = -1.0
    
    # orientation_flag == -1: 后腿倒立，目标重力方向为 [1, 0, 0]
    back_handstand_mask = (orientation_flag == -1)
    target_gravity[back_handstand_mask, 0] = 1.0
    
    # orientation_flag == 0: 正常姿态，目标重力方向为 [0, 0, -1]
    normal_mask = (orientation_flag == 0)
    target_gravity[normal_mask, 2] = -1.0
    
    # Penalize deviation of the projected gravity vector from the target
    return torch.sum(torch.square(asset.data.projected_gravity_b - target_gravity), dim=1)

def handstand_hip_default_joint_pos_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), command_name: str = "base_command") -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    orientation_flag = env.command_manager.get_command(command_name)[:, 3]  # [num_envs]
    if asset_cfg.joint_ids is None:
        return torch.zeros(env.scene.num_envs, device=env.device, dtype=torch.float32, requires_grad=False)
    hip_default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    hip_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(hip_joint_pos - hip_default_joint_pos), dim=1) * (orientation_flag != 0).float()


def handstand_thigh_default_joint_pos_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), command_name: str = "base_command") -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    orientation_flag = env.command_manager.get_command(command_name)[:, 3]  # [num_envs]
    if asset_cfg.joint_ids is None:
        return torch.zeros(env.scene.num_envs, device=env.device, dtype=torch.float32, requires_grad=False)
    thigh_default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    thigh_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(thigh_joint_pos - thigh_default_joint_pos), dim=1) * (orientation_flag != 0).float()

def handstand_calf_default_joint_pos_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), command_name: str = "base_command") -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    orientation_flag = env.command_manager.get_command(command_name)[:, 3]  # [num_envs]
    if asset_cfg.joint_ids is None:
        return torch.zeros(env.scene.num_envs, device=env.device, dtype=torch.float32, requires_grad=False)
    calf_default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    calf_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(calf_joint_pos - calf_default_joint_pos), dim=1) * (orientation_flag != 0).float()

def action_smoothness_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward action smoothness."""
    # extract the used quantities (to enable type-hinting)
    actions = env.actions_history.buffer
    diff = torch.square(actions[:, 0, :] - 2 * actions[:, 1, :] + actions[:, 2, :])
    return torch.sum(diff, dim=1)


def stand_joint_deviation_l1(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(angle), dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) < 0.1
    return reward

def joint_deviation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(angle), dim=1)


def joint_deviation_l1(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    # print(asset_cfg.joint_ids)
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(angle), dim=1)
    return reward


def stand_feet_xy_deviation_l1(
    env: ManagerBasedRLEnv, command_name: str, default_feet_xy_pos: Sequence[float], sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    sensor: FrameTransformer = env.scene.sensors[sensor_cfg.name]
    # compute out of limits constraints
    feet_xy_pos = sensor.data.target_pos_source[:, :, :2].flatten(start_dim=1)
    default_feet_xy_pos = torch.tensor(default_feet_xy_pos, device=env.device)
    reward = torch.sum(torch.abs(feet_xy_pos - default_feet_xy_pos), dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) < 0.1
    return reward


def base_acc_mix_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize base acceleration."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    lin_acc = asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :]
    ang_acc = asset.data.body_ang_acc_w[:, asset_cfg.body_ids, :]
    return torch.square(lin_acc).sum(dim=(1, 2)) + 0.02 * torch.square(ang_acc).sum(dim=(1, 2))


def feet_lin_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet linear acceleration."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    lin_acc = asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :]
    return torch.norm(lin_acc, dim=-1).sum(dim=1)


def contact_forces_l2(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    F = contact_sensor.data.net_forces_w_history          # [B, T, S, 3]
    F = F[:, :, sensor_cfg.body_ids, :]                   # [B, T, F, 3]  选出脚/触点
    mag = torch.linalg.norm(F, dim=-1)                    # [B, T, F]     每时刻每足的 |F|
    peak = mag.amax(dim=1)                                # [B, F]        历史窗口内最大
    excess = (peak - threshold).clamp_min(0.0)            # [B, F]        只惩罚超限部分
    return torch.square(excess).sum(dim=1)                              # [B]


def base_height_l2_fix(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        # 检查sensor数据是否包含inf或nan
        ray_hits = sensor.data.ray_hits_w[..., 2]
        ray_hits = torch.where(torch.isinf(ray_hits), 0.0, ray_hits)
        ray_hits = torch.where(torch.isnan(ray_hits), 0.0, ray_hits)
        adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def diagonal_contact(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Reward diagonal contact."""

    if not hasattr(env, "time_diagonal_contact_buf"):
        num_envs = env.scene.num_envs
        device = getattr(env, "device", torch.device("cpu"))
        env.time_diagonal_contact_buf = torch.zeros(num_envs, device=device, dtype=torch.float32, requires_grad=False)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_filter = contact_sensor.data.current_contact_time > 0.0
    diagonal_contact1 = contact_filter[:, 0] & contact_filter[:, 3]
    diagonal_contact2 = contact_filter[:, 1] & contact_filter[:, 2]
    num_diagonal_contacts = diagonal_contact1.float() + diagonal_contact2.float()
    single_diagonal_contact = (num_diagonal_contacts == 1)
    env.time_diagonal_contact_buf = torch.where(
        single_diagonal_contact, torch.zeros_like(single_diagonal_contact, dtype=torch.float32, device=env.device, requires_grad=False), env.time_diagonal_contact_buf + env.step_dt
    )

    reward = torch.where(
        torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) < 0.1,
        torch.ones_like(single_diagonal_contact, dtype=torch.float32, device=env.device, requires_grad=False),
        (env.time_diagonal_contact_buf < threshold).float(),
    )
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Penalize feet stumbling over history:
      只要在任意历史帧、任意一只脚满足 ‖(Fx,Fy)‖ > 4·|Fz|，
      就返回 1.0，否则 0.0，shape=(num_envs,1).
    """
    # 1) 取出历史接触力，shape == (E, T, F, 3)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[
        :, :, sensor_cfg.body_ids, :
    ]  # (E, T, F, 3)

    # 2) 水平力 vs 法向力
    horiz = torch.norm(contact_forces[..., :2], dim=-1)  # (E, T, F)
    vert = contact_forces[..., 2].abs()                # (E, T, F)

    # 3) 绊倒条件掩码
    err = horiz - 4 * vert                              # (E, T, F)

    mask = err > 0.01

    # 4) 跨历史帧和脚维 any → (E,)
    stumble_flag = mask.any(dim=(1, 2))                   # (E,) bool

    return stumble_flag.float()

def feet_center(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, 
                height_threshold: float = 0.2) -> torch.Tensor:
    """
    Feet center reward: 惩罚脚踩在边缘的情况。
    
    通过检查脚周围9个点的高度分布来判断是否在边缘：
    - Type 1 (中心点, id=5): 脚的接触位置
    - Type 2 (近点, id=2,4,6,8): 距离脚d1=5cm半径内的4个点  
    - Type 3 (远点, id=1,3,7,9): 距离脚d2=√50cm半径内的4个点
    
    奖励公式: c_t * (n_t^2 + 2 * n_t^3)
    其中 n_t^i 是第i类型中高度 h < threshold 的点数
    
    Args:
        env: RL环境
        sensor_cfg: 接触传感器配置
        height_threshold: 高度阈值，默认-0.2m
    
    Returns:
        reward: (num_envs,) 奖励值
    """
    num_envs = env.scene.num_envs
    device = getattr(env, "device", torch.device("cpu"))
    
    # 确保 foot_scan_buf 已初始化
    if not hasattr(env, "foot_scan_buf"):
        env.foot_scan_buf = torch.zeros(
            (num_envs, 36),
            device=device,
            dtype=torch.float32,
            requires_grad=False
        )
    
    # Reshape: (num_envs, 36) -> (num_envs, 4_feet, 9_points)
    # 9个点的布局（3x3网格）：
    # 3(idx=6)  6(idx=7)  9(idx=8)
    # 2(idx=3)  5(idx=4)  8(idx=5)
    # 1(idx=0)  4(idx=1)  7(idx=2)
    foot_heights = env.foot_scan_buf.view(num_envs, 4, 9)
    
    # 定义点的类型索引
    # type1_idx = [4]              # Type 1: 中心点 (id=5)
    type1_idx = [4]     # Type 1: 远点 (id=1,3,7,9)
    type2_idx = [1, 3, 5, 7]     # Type 2: 近点 (id=2,4,6,8)
    type3_idx = [0, 2, 6, 8]     # Type 3: 远点 (id=1,3,7,9)
    
    # 获取接触传感器数据，判断哪些脚在接触地面
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 0.1
    contact_mask = contacts.float()  # (num_envs, 4)
    
    # 计算每种类型中低于阈值的点数
    # n_t^1: Type 1中 h < threshold 的点数
    n_t1 = (foot_heights[:, :, type1_idx].abs() > height_threshold).sum(dim=-1).float()  # (num_envs, 4)

    # n_t^2: Type 2中 h < threshold 的点数
    n_t2 = (foot_heights[:, :, type2_idx].abs() > height_threshold).sum(dim=-1).float()  # (num_envs, 4)
    
    # n_t^3: Type 3中 h < threshold 的点数
    n_t3 = (foot_heights[:, :, type3_idx].abs() > height_threshold).sum(dim=-1).float()  # (num_envs, 4)
    
    # 根据论文公式计算奖励: c_t * (n_t^2 + 2 * n_t^3)
    # 只对接触地面的脚计算
    # reward_per_foot = (n_t1 + n_t2 + 2.0 * n_t3) * contact_mask  # (num_envs, 4)
    reward_per_foot = (n_t2 + 2.0 * n_t3) * contact_mask  # (num_envs, 4)
    
    # 对所有脚求和得到每个环境的总奖励
    reward = reward_per_foot.sum(dim=1)  # (num_envs,)
    
    return reward


# def dont_wait(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
#     """Reward not waiting for command."""
#     return torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) < 0.1

def step_classifier_by_ratio(
    h: torch.Tensor,
    *,
    ratio_th: float = 15.0,   # 只用这个阈值做最终判定
    eps: float = 1e-6,       # 数值稳定项
    noise_floor: float = 1e-3  # 对分母的噪声地板，避免 emed 过小导致 rho 假大；按你的单位调整
) -> torch.Tensor:
    """
    仅用最大跳变比 rho 判定台阶:
      is_step = (rho > ratio_th)

    参数:
      h: (B, P, 9)，每个 patch 为 3×3 网格按行主序展开
    返回:
      is_step: (B, P)  bool
    """
    assert h.shape[-1] == 9, "最后一维必须为9（3×3）"
    B, P, _ = h.shape
    device = h.device

    # 若为半精度，差分与统计时转到 float32 更稳
    work_dtype = torch.float32 if h.dtype in (torch.float16, torch.bfloat16) else h.dtype
    z = h.to(work_dtype).view(B, P, 3, 3)

    # 4-邻接边差分（共12条边）
    dh = (z[..., :, 1:] - z[..., :, :-1]).abs().reshape(B, P, -1)  # 3*2=6
    dv = (z[..., 1:, :] - z[..., :-1, :]).abs().reshape(B, P, -1)  # 2*3=6
    edges = torch.cat([dh, dv], dim=-1)                             # (B,P,12)

    # 计算 rho = emax / emed（带下限钳位）
    emax = edges.max(dim=-1).values                                  # (B,P)
    emed = edges.median(dim=-1).values                               # (B,P)
    denom = torch.clamp(emed, min=max(eps, noise_floor))             # (B,P)
    rho = emax / denom

    # 仅用 ratio_th 判定
    is_step = rho > ratio_th
    return is_step, rho


def feet_edge_contact(env: ManagerBasedRLEnv, contact_sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet edge contact."""
    num_envs = env.scene.num_envs
    device = getattr(env, "device", torch.device("cpu"))

    if not hasattr(env, "foot_scan_buf"):
        env.foot_scan_buf = torch.zeros(
            (num_envs, 36),
            device=device,
            dtype=torch.float32,
            requires_grad=False
        )

    foot_scan_buf = env.foot_scan_buf.view(num_envs, 4, 9)

    is_step, rho = step_classifier_by_ratio(foot_scan_buf)
    is_step = is_step.float()
    # print("rho:", rho[..., 0])
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, contact_sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 0.1                  # (num_envs, 4) bool

    contact_mask = contacts.float()                    # (num_envs, 4)
    flag_per_env = (is_step * contact_mask).any(dim=1).float()

    return flag_per_env.squeeze()

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse

from ruamel.yaml import YAML

yaml = YAML()
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--debug_lidar_obs", action="store_true", default=False, help="Run in debug mode.")
parser.add_argument("--debug_gt_hmap", action="store_true", default=False, help="Run in debug mode.")
parser.add_argument("--debug_gt_foot", action="store_true", default=False, help="Run in debug mode.")
# append AMP-VAE-VIT cli arguments
cli_args.add_parkour_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import numpy as np
import rl_sim_env.tasks
import torch
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rl_algorithms.rsl_rl.runners import ParkourOnPolicyRunner
from rl_algorithms.rsl_rl_wrapper import (
    ParkourOnPolicyRunnerCfg,
    ParkourVecEnvWrapper,
    export_parkour_policy_as_onnx,
    load_onnx_model,
    onnx_run_inference,
    verify_onnx_model,
)

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from rl_debug.marker import (
    GREEN_SPHERE_MARKER_CFG,
    RED_SPHERE_MARKER_CFG,
    BLUE_SPHERE_MARKER_CFG,
    YELLOW_SPHERE_MARKER_CFG,
    GREEN_CUBOID_MARKER_CFG,
    RED_CUBOID_MARKER_CFG,
    BLUE_CUBOID_MARKER_CFG,
    YELLOW_CUBOID_MARKER_CFG,
)
from rl_debug.text_visualizer import get_text_visualizer
from isaaclab.utils.math import quat_apply, quat_apply_yaw, quat_inv, quat_rotate_inverse, yaw_quat, transform_points
from rl_sim_env.tasks.manager_based.common.utils.grid import get_3x3_grid, grid_pattern

# PLACEHOLDER: Extension template (do not remove this comment)

import math
import carb
from carb.input import KeyboardEventType, KeyboardInput
from omni.appwindow import get_default_app_window

# 全局状态
pos = [0.0, 0.0, 2.0]           # x, y, z
euler = [0.0, 0.0, 0.0]         # roll, pitch, yaw (rad)
step_pos = 0.15                  # 每次平移步长
step_ang = math.radians(5.0)    # 每次旋转 5°
print_velocity = True           # 是否在终端打印速度信息（默认开启）
print_interval = 30             # 打印间隔（帧数，默认约0.5秒）
frame_count = 0                 # 帧计数器
velocity_env_idx = 0            # 显示哪个环境的速度信息

# 初始化输入
input_interface = carb.input.acquire_input_interface()
app_window = get_default_app_window()


def euler_to_quat(roll, pitch, yaw):
    cr = math.cos(roll / 2)
    sr = math.sin(roll / 2)
    cp = math.cos(pitch / 2)
    sp = math.sin(pitch / 2)
    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y, z]


def on_key(event, *args, **kwargs):
    global pos, euler, print_velocity, velocity_env_idx
    if event.type != KeyboardEventType.KEY_PRESS:
        return

    key = event.input
    # 退出
    if key == KeyboardInput.ESCAPE:
        simulation_app.close()

    # 位置控制
    elif key == KeyboardInput.A:
        pos[0] -= step_pos
    elif key == KeyboardInput.S:
        pos[0] += step_pos
    elif key == KeyboardInput.D:
        pos[1] -= step_pos
    elif key == KeyboardInput.F:
        pos[1] += step_pos
    elif key == KeyboardInput.G:
        pos[2] -= step_pos
    elif key == KeyboardInput.H:
        pos[2] += step_pos

    # 欧拉角控制：z/x 控制 roll，c/v 控制 pitch，b/n 控制 yaw
    elif key == KeyboardInput.Z:
        euler[0] -= step_ang
    elif key == KeyboardInput.X:
        euler[0] += step_ang
    elif key == KeyboardInput.C:
        euler[1] -= step_ang
    elif key == KeyboardInput.V:
        euler[1] += step_ang
    elif key == KeyboardInput.B:
        euler[2] -= step_ang
    elif key == KeyboardInput.N:
        euler[2] += step_ang

    # 速度显示控制
    elif key == KeyboardInput.O:
        print_velocity = not print_velocity
        print(f"终端速度打印: {'开启' if print_velocity else '关闭'}")
    elif key == KeyboardInput.LEFT_ARROW:
        velocity_env_idx = max(0, velocity_env_idx - 1)
        print(f"显示环境索引: {velocity_env_idx}")
    elif key == KeyboardInput.RIGHT_ARROW:
        velocity_env_idx = velocity_env_idx + 1
        print(f"显示环境索引: {velocity_env_idx}")

    # 打印调试信息
    print(f"pos = {pos}, euler (deg) = {[math.degrees(a) for a in euler]}")


# 订阅回调
input_interface.subscribe_to_keyboard_events(
    app_window.get_keyboard(),
    on_key
)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: ParkourOnPolicyRunnerCfg = cli_args.parse_parkour_cfg(args_cli.task, args_cli)

    # 全局速度显示控制
show_velocity = True  # 按V键切换速度显示
velocity_env_idx = 0  # 显示哪个环境的速度信息

# specify directory for logging experiments
    log_root_path = os.path.join("logs", "parkour", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("parkour", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    # elif args_cli.checkpoint:
    #     resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = ParkourVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = ParkourOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.physics_dt

    # reset environment

    green_voxel_cfg: VisualizationMarkersCfg = GREEN_SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Voxel")
    red_voxel_cfg: VisualizationMarkersCfg = RED_SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Voxel")
    blue_voxel_cfg: VisualizationMarkersCfg = BLUE_SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Voxel")
    yellow_voxel_cfg: VisualizationMarkersCfg = YELLOW_SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Voxel")
    green_cuboid_cfg: VisualizationMarkersCfg = GREEN_CUBOID_MARKER_CFG.replace(prim_path="/Visuals/Voxel")
    red_cuboid_cfg: VisualizationMarkersCfg = RED_CUBOID_MARKER_CFG.replace(prim_path="/Visuals/Voxel")
    blue_cuboid_cfg: VisualizationMarkersCfg = BLUE_CUBOID_MARKER_CFG.replace(prim_path="/Visuals/Voxel")
    yellow_cuboid_cfg: VisualizationMarkersCfg = YELLOW_CUBOID_MARKER_CFG.replace(prim_path="/Visuals/Voxel")

    zone_voxel_cfg = [green_voxel_cfg, red_voxel_cfg, blue_voxel_cfg, yellow_voxel_cfg, green_cuboid_cfg, red_cuboid_cfg, blue_cuboid_cfg, yellow_cuboid_cfg]
    debug_visualizer = []
    for i in range(3):
        debug_visualizer.append(VisualizationMarkers(zone_voxel_cfg[i]))
    for visualizer in debug_visualizer:
        visualizer.set_visibility(True)

    first_reset = True
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)

            env_ids = torch.arange(env.num_envs, device=env.device)
            if first_reset:
                state = env.unwrapped.scene.get_state()
                print(state)
                first_reset = False
            wxyz = euler_to_quat(*euler)
            pose = [pos[0], pos[1], pos[2], wxyz[0], wxyz[1], wxyz[2], wxyz[3]]
            state["articulation"]["robot"]["root_pose"][:, :7] = torch.tensor(pose, device=env.device).repeat(env.num_envs, 1)
            state["articulation"]["robot"]["root_velocity"][:, :6] = torch.zeros((env.num_envs, 6), device=env.device)
            state["articulation"]["robot"]["joint_position"][:] = torch.zeros((env.num_envs, 12), device=env.device)
            state["articulation"]["robot"]["joint_velocity"][:] = torch.zeros((env.num_envs, 12), device=env.device)
            env.unwrapped.scene.reset_to(state, env_ids)

            # env stepping
            (
                obs_buf,
                rewards,
                dones,
                infos,
                reset_env_ids,
                terminal_amp_states,
                episode_reward,
            ) = env.step(actions)
            pos_env = env.unwrapped.scene["robot"].data.root_pos_w
            quat_env = env.unwrapped.scene["robot"].data.root_quat_w

            if args_cli.debug_lidar_obs:
                grid = grid_pattern(3.0, 1.2, 0.05, env.device).repeat(env.num_envs, 1, 1)
                gt_heightmap_obs = obs_buf["lidar_obs"].reshape(env.num_envs, -1, 1525)[:, 0, :]
                height_obs_rough = pos_env[:, 2].unsqueeze(1) - gt_heightmap_obs.squeeze(0) / 5.0 - 0.5
                grid[:, :, 2] = height_obs_rough
                grid = transform_points(points=grid, pos=None, quat=yaw_quat(quat_env))
                grid[..., :2] += pos_env[:, :2].unsqueeze(1)  # Broadcasting: [1525,2] + [2] -> [1525,2]
                debug_visualizer[0].visualize(
                    translations=grid.reshape(-1, 3),
                )

            if args_cli.debug_gt_hmap:
                grid = grid_pattern(2.9375, 0.9375, 0.0625, env.device).repeat(env.num_envs, 1, 1)
                gt_heightmap_obs = obs_buf["actor_obs"][..., -768:]
                height_obs_rough = pos_env[:, 2].unsqueeze(1) - gt_heightmap_obs.squeeze(0) / 5.0 - 0.5
                grid[..., 2] = height_obs_rough
                grid = transform_points(points=grid, pos=None, quat=yaw_quat(quat_env))
                grid[..., :2] += pos_env[:, :2].unsqueeze(1)  # Broadcasting: [1525,2] + [2] -> [1525,2]
                debug_visualizer[1].visualize(
                    translations=grid.reshape(-1, 3),
                )

            if args_cli.debug_gt_foot:
                grid_foot = grid_pattern(0.1, 0.1, 0.05, env.device).repeat(env.num_envs, 4, 1, 1)

                foot_pos_local = obs_buf["amp_obs"][:, -12:]
                foot_pos_local = foot_pos_local.reshape(env.num_envs, 4, 3)
                foot_pos_world = transform_points(points=foot_pos_local, pos=None, quat=quat_env) + pos_env.unsqueeze(1)
                gt_foot_scan_obs = obs_buf["gt_foot_scan_obs"]
                footheight = foot_pos_world[..., 2:] - gt_foot_scan_obs.reshape(env.num_envs, 4, 9) / 5.0
                grid_foot[..., 2] = footheight.reshape(env.num_envs, 4, 9)
                grid_foot = transform_points(points=grid_foot.reshape(env.num_envs, -1, 3), pos=None, quat=yaw_quat(quat_env))
                grid_foot = grid_foot.reshape(env.num_envs, 4, 9, 3)
                grid_foot[..., :2] += foot_pos_world[..., :2].unsqueeze(2)
                debug_visualizer[2].visualize(
                    translations=grid_foot.reshape(-1, 3),
                )

            # 速度显示（终端打印）
            frame_count += 1
            if print_velocity and frame_count % print_interval == 0:
                robot = env.unwrapped.scene["robot"]
                num_envs = env.num_envs

                # 获取速度数据
                base_lin_vel_w = robot.data.root_lin_vel_w
                base_ang_vel_w = robot.data.root_ang_vel_w

                # 获取命令速度
                cmd_mgr = env.unwrapped.command_manager
                base_cmd = cmd_mgr.get_command("base_command")

                # 确保环境索引有效
                velocity_env_idx = min(velocity_env_idx, num_envs - 1)

                # 打印速度信息
                print(f"\n{'='*40}")
                print(f"环境 {velocity_env_idx}/{num_envs} | 帧 {frame_count}")
                print(f"{'='*40}")
                print(f"实际速度 (m/s):")
                print(f"  X: {base_lin_vel_w[velocity_env_idx, 0]:6.2f}  Y: {base_lin_vel_w[velocity_env_idx, 1]:6.2f}  Z: {base_lin_vel_w[velocity_env_idx, 2]:6.2f}")
                print(f"角速度 (rad/s):")
                print(f"  X: {base_ang_vel_w[velocity_env_idx, 0]:6.2f}  Y: {base_ang_vel_w[velocity_env_idx, 1]:6.2f}  Z: {base_ang_vel_w[velocity_env_idx, 2]:6.2f}")

                if base_cmd is not None:
                    cmd_vel = base_cmd[velocity_env_idx]
                    print(f"命令速度:")
                    print(f"  X: {cmd_vel[0]:6.2f}  Y: {cmd_vel[1]:6.2f}  角Z: {cmd_vel[2]:6.2f}")
                    print(f"跟踪误差:")
                    print(f"  X: {abs(base_lin_vel_w[velocity_env_idx, 0] - cmd_vel[0]):6.2f}  Y: {abs(base_lin_vel_w[velocity_env_idx, 1] - cmd_vel[1]):6.2f}  角Z: {abs(base_ang_vel_w[velocity_env_idx, 2] - cmd_vel[2]):6.2f}")
                print(f"控制: O=开关打印 ←→=切换环境")

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

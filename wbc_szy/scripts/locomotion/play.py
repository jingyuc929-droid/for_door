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
parser.add_argument("--onnx", action="store_true", default=False, help="Run in onnx mode.")
parser.add_argument("--debug_lidar_obs", action="store_true", default=False, help="Run in debug mode.")
parser.add_argument("--debug_gt_hmap", action="store_true", default=False, help="Run in debug mode.")
parser.add_argument("--debug_gt_foot", action="store_true", default=False, help="Run in debug mode.")
parser.add_argument("--debug_rough_map", action="store_true", default=False, help="Run in debug mode.")
parser.add_argument("--debug_fine_map", action="store_true", default=False, help="Run in debug mode.")
parser.add_argument("--debug_decoded_map", action="store_true", default=False, help="Run in debug mode.")
parser.add_argument("--debug_decoded_foot", action="store_true", default=False, help="Run in debug mode.")
# append AMP-VAE-VIT cli arguments
cli_args.add_locomotion_args(parser)
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
import math

import gymnasium as gym
import numpy as np
import rl_sim_env.tasks
import torch
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rl_algorithms.rsl_rl.runners import LocomotionOnPolicyRunner
from rl_algorithms.rsl_rl_wrapper import (
    LocomotionOnPolicyRunnerCfg,
    LocomotionVecEnvWrapper,
    export_locomotion_policy_as_onnx,
    load_onnx_model,
    onnx_run_inference_locomotion,
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
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_apply,
    quat_apply_yaw,
    quat_inv,
    quat_mul,
    quat_rotate_inverse,
    transform_points,
    yaw_quat,
)
from rl_sim_env.tasks.manager_based.common.utils.grid import get_3x3_grid, grid_pattern
from rl_sim_env.tasks.manager_based.common.utils.projected_com_frame import (
    world_to_projected_frame,
)

# PLACEHOLDER: Extension template (do not remove this comment)

import carb
from carb.input import KeyboardEventType, KeyboardInput
from omni.appwindow import get_default_app_window

# -----------------------------
# 速度显示控制（终端打印）
# -----------------------------
print_velocity = True      # 启动后自动打印速度信息
velocity_env_idx = 0       # 显示哪个环境的速度
print_interval = 30        # 打印间隔（帧）
frame_count = 0

input_interface = carb.input.acquire_input_interface()
app_window = get_default_app_window()

# 键盘回调 on_key 在 main() 内定义（需要访问 env.num_envs 做边界检查）


def main():
    """Play with RSL-RL agent."""
    global frame_count, print_velocity, velocity_env_idx
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: LocomotionOnPolicyRunnerCfg = cli_args.parse_locomotion_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "locomotion", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("locomotion", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    # elif args_cli.checkpoint:
    #     resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # Prefer restoring the exact training-time configs saved in the run directory.
    # This avoids state_dict shape mismatches when the code/config has changed since training.
    params_dir = os.path.join(log_dir, "params")
    env_pkl_path = os.path.join(params_dir, "env.pkl")
    agent_pkl_path = os.path.join(params_dir, "agent.pkl")
    if os.path.isfile(env_pkl_path) and os.path.isfile(agent_pkl_path):
        try:
            with open(env_pkl_path, "rb") as f:
                env_cfg = pickle.load(f)
            with open(agent_pkl_path, "rb") as f:
                agent_cfg = pickle.load(f)
            print(f"[INFO] Restored env/agent cfg from: {params_dir}")
        except Exception as e:
            print(f"[WARN] Failed to restore cfg from '{params_dir}': {e}. Falling back to registry configs.")

    # Re-apply critical CLI overrides after restoring configs.
    if args_cli.device is not None:
        if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "device"):
            env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    if args_cli.num_envs is not None and hasattr(env_cfg, "scene"):
        env_cfg.scene.num_envs = args_cli.num_envs
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "use_fabric"):
        env_cfg.sim.use_fabric = not args_cli.disable_fabric

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
    clip_actions = agent_cfg.clip_actions
    clip_obs = None
    if hasattr(env_cfg, "config_summary") and hasattr(env_cfg.config_summary, "env"):
        clip_actions = getattr(env_cfg.config_summary.env, "clip_actions", clip_actions)
        clip_obs = getattr(env_cfg.config_summary.env, "clip_obs", clip_obs)
    env = LocomotionVecEnvWrapper(env, clip_actions=clip_actions, clip_obs=clip_obs)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = LocomotionOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # play/推理不需要 optimizer state，跳过加载以避免 checkpoint 与当前代码
    # optimizer 参数组数量不一致时报错（load_optimizer=False 仅加载模型权重）
    ppo_runner.load(resume_path, load_optimizer=False)

    # obtain the trained policy for inference
    policy, extra_dict = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    train_cfg_dict = agent_cfg.to_dict()
    use_vae = train_cfg_dict["train_cfg_dict"]["use_vae"] if "use_vae" in train_cfg_dict["train_cfg_dict"] else False

    # export policy to onnx/jit
    dir_path, filename = os.path.split(resume_path)
    base_dir, exp_dir = os.path.split(dir_path)
    base_dir = base_dir + os.sep
    name_no_ext, _ = os.path.splitext(filename)
    print("base_dir:", base_dir)
    print("exp_dir:", exp_dir)
    print("name_no_ext:", name_no_ext)
    export_model_dir = os.path.join(os.path.dirname(base_dir), "exported", exp_dir, name_no_ext)
    export_locomotion_policy_as_onnx(
        policy,
        extra_dict["vae"],
        path=export_model_dir,
        filename="policy.onnx",
    )
    verify_onnx_model(os.path.join(export_model_dir, "policy.onnx"), "policy")
    policy_yaml_path = os.path.join(log_dir, "policy.yaml")
    with open(policy_yaml_path, encoding="utf-8") as f:
        policy_yaml = yaml.load(f)
    policy_yaml["load_run"] = agent_cfg.load_run
    policy_yaml["checkpoint"] = agent_cfg.load_checkpoint
    policy_yaml_export_path = os.path.join(export_model_dir, "policy.yaml")
    with open(policy_yaml_export_path, "w", encoding="utf-8") as f:
        yaml.dump(policy_yaml, f)

    dt = env.unwrapped.physics_dt

    # 键盘回调：O/0=开关打印，←→=切换环境，ESC=退出
    # 定义在 main 内以便访问 env.num_envs 做边界检查
    def on_key(event, *args, **kwargs):
        global print_velocity, velocity_env_idx
        if event.type != KeyboardEventType.KEY_PRESS:
            return
        key = event.input
        num_envs = env.num_envs
        if key in (KeyboardInput.O, KeyboardInput.KEY_0):
            print_velocity = not print_velocity
            print(f"[vel] 终端速度打印: {'开启' if print_velocity else '关闭'}")
        elif key == KeyboardInput.LEFT:
            if num_envs <= 1:
                print(f"[vel] 只有 {num_envs} 个环境，无法切换（可用 --num_envs N 跑多环境）")
                return
            velocity_env_idx = (velocity_env_idx - 1) % num_envs
            print(f"[vel] 显示环境: {velocity_env_idx}/{num_envs}")
        elif key == KeyboardInput.RIGHT:
            if num_envs <= 1:
                print(f"[vel] 只有 {num_envs} 个环境，无法切换（可用 --num_envs N 跑多环境）")
                return
            velocity_env_idx = (velocity_env_idx + 1) % num_envs
            print(f"[vel] 显示环境: {velocity_env_idx}/{num_envs}")
        elif key == KeyboardInput.ESCAPE:
            print("[vel] 收到 ESC，退出仿真...")
            simulation_app.close()

    try:
        input_interface.subscribe_to_keyboard_events(app_window.get_keyboard(), on_key)
        print(f"[vel] 键盘就绪: O 或 0=开关打印  ←→=切换环境(共 {env.num_envs} 个)  ESC=退出")
    except Exception as e:
        print(f"[WARN] 键盘事件订阅失败（不影响速度打印）: {e}")

    # reset environment
    obs_dict = env.get_observations()
    obs_dict = obs_dict.to(env.device)
    timestep = 0

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
    for i in range(8):
        debug_visualizer.append(VisualizationMarkers(zone_voxel_cfg[i]))
    for visualizer in debug_visualizer:
        visualizer.set_visibility(True)

    if args_cli.onnx:
        print("[INFO] Running in onnx mode.")
        onnx_session = load_onnx_model(os.path.join(export_model_dir, "policy.onnx"))
        print("[INFO] ONNX model loaded successfully.")

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            if args_cli.onnx:
                actor_obs_onnx = obs_dict["actor_obs"].cpu().numpy().astype(np.float32)
                vae_obs_onnx = None
                if use_vae:
                    vae_obs_onnx = obs_dict['estimator_obs'].cpu().numpy().astype(np.float32)
                actions = torch.from_numpy(onnx_run_inference_locomotion(onnx_session, actor_obs_onnx, vae_obs_onnx)['actions']).to(
                    env.device
                )
            else:
                if use_vae:
                    vae_out = extra_dict["vae"].act_inference(obs_dict['estimator_obs'])
                    obs_dict['estimator_out'] = vae_out.detach()
                actions = policy.act_inference(obs_dict)

            # env stepping
            (
                obs_dict,
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
                grid = grid_pattern(3.0, 1.2, 0.05, env.device).repeat(env.num_envs, 1, 1)
                gt_heightmap_obs = obs_buf["gt_heightmap_obs"]
                height_obs_rough = pos_env[:, 2].unsqueeze(1) - gt_heightmap_obs.squeeze(0) / 5.0 - 0.5
                grid[..., 2] = height_obs_rough
                grid = transform_points(points=grid, pos=None, quat=yaw_quat(quat_env))
                grid[..., :2] += pos_env[:, :2].unsqueeze(1)
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

            if args_cli.debug_rough_map:
                grid = grid_pattern(3.0, 1.2, 0.05, env.device).repeat(env.num_envs, 1, 1)
                gt_heightmap_obs = rough_map
                height_obs_rough = pos_env[:, 2].unsqueeze(1) - gt_heightmap_obs.squeeze(0) / 5.0 - 0.5
                grid[..., 2] = height_obs_rough
                grid = transform_points(points=grid, pos=None, quat=yaw_quat(quat_env))
                grid[..., :2] += pos_env[:, :2].unsqueeze(1)
                debug_visualizer[3].visualize(
                    translations=grid.reshape(-1, 3),
                )

            if args_cli.debug_fine_map:
                grid = grid_pattern(3.0, 1.2, 0.05, env.device).repeat(env.num_envs, 1, 1)
                gt_heightmap_obs = fine_map
                height_obs_rough = pos_env[:, 2].unsqueeze(1) - gt_heightmap_obs.squeeze(0) / 5.0 - 0.5
                grid[..., 2] = height_obs_rough
                grid = transform_points(points=grid, pos=None, quat=yaw_quat(quat_env))
                grid[..., :2] += pos_env[:, :2].unsqueeze(1)
                debug_visualizer[4].visualize(
                    translations=grid.reshape(-1, 3),
                )

            if args_cli.debug_decoded_map:
                grid = grid_pattern(3.0, 1.2, 0.05, env.device).repeat(env.num_envs, 1, 1)
                gt_heightmap_obs = decoded_map
                height_obs_rough = pos_env[:, 2].unsqueeze(1) - gt_heightmap_obs.squeeze(0) / 5.0 - 0.5
                grid[..., 2] = height_obs_rough
                grid = transform_points(points=grid, pos=None, quat=yaw_quat(quat_env))
                grid[..., :2] += pos_env[:, :2].unsqueeze(1)
                debug_visualizer[5].visualize(
                    translations=grid.reshape(-1, 3),
                )

            if args_cli.debug_decoded_foot:
                grid_foot = grid_pattern(0.1, 0.1, 0.05, env.device).repeat(env.num_envs, 4, 1, 1)
                foot_pos_local = obs_buf["amp_obs"][:, -12:]
                foot_pos_local = foot_pos_local.reshape(env.num_envs, 4, 3)
                foot_pos_world = transform_points(points=foot_pos_local, pos=None, quat=quat_env) + pos_env.unsqueeze(1)
                gt_foot_scan_obs = decoded_foot
                footheight = foot_pos_world[..., 2:] - gt_foot_scan_obs.reshape(env.num_envs, 4, 9) / 5.0
                grid_foot[..., 2] = footheight.reshape(env.num_envs, 4, 9)
                grid_foot = transform_points(points=grid_foot.reshape(env.num_envs, -1, 3), pos=None, quat=yaw_quat(quat_env))
                grid_foot = grid_foot.reshape(env.num_envs, 4, 9, 3)
                grid_foot[..., :2] += foot_pos_world[..., :2].unsqueeze(2)
                debug_visualizer[6].visualize(
                    translations=grid_foot.reshape(-1, 3),
                )

            # 速度打印（终端）—— 启动后自动打印，按 O 键开关，←→ 切换环境
            frame_count += 1
            if print_velocity and frame_count % print_interval == 0:
                robot = env.unwrapped.scene["robot"]
                num_envs = env.num_envs
                # 命令速度在【基座系(base)】下表达（commands.py: vel_command_b vs
                # root_lin_vel_b），实际速度也必须取 base 系；否则当机器人 yaw≠0 时，
                # 世界系 X/Y 与命令的“前进/侧移”差一个旋转，命令与实际就对不上。
                base_lin_vel_b = robot.data.root_lin_vel_b
                base_ang_vel_b = robot.data.root_ang_vel_b
                cmd_mgr = env.unwrapped.command_manager
                base_cmd = cmd_mgr.get_command("base_command")

                velocity_env_idx = min(velocity_env_idx, num_envs - 1)

                print(f"\n{'=' * 44}")
                print(f"[vel] 环境 {velocity_env_idx}/{num_envs} | 帧 {frame_count}  (基座系)")
                print(f"{'=' * 44}")
                print(f"实际速度 (m/s):   X={base_lin_vel_b[velocity_env_idx, 0]:6.2f}  Y={base_lin_vel_b[velocity_env_idx, 1]:6.2f}  Z={base_lin_vel_b[velocity_env_idx, 2]:6.2f}")
                print(f"角速度 (rad/s):   X={base_ang_vel_b[velocity_env_idx, 0]:6.2f}  Y={base_ang_vel_b[velocity_env_idx, 1]:6.2f}  Z={base_ang_vel_b[velocity_env_idx, 2]:6.2f}")
                if base_cmd is not None:
                    cv = base_cmd[velocity_env_idx]
                    print(f"命令速度:         X={cv[0]:6.2f}  Y={cv[1]:6.2f}  角Z={cv[2]:6.2f}")
                    print(f"跟踪误差:         X={abs(base_lin_vel_b[velocity_env_idx, 0] - cv[0]):6.2f}  Y={abs(base_lin_vel_b[velocity_env_idx, 1] - cv[1]):6.2f}  角Z={abs(base_ang_vel_b[velocity_env_idx, 2] - cv[2]):6.2f}")

                # 末端执行器：跟踪误差 + 命令高度/实际高度/当前 pitch
                try:
                    ee_term = cmd_mgr.get_term("ee_target_points")
                    if ee_term is not None and hasattr(ee_term, "_last_rot_err_rad"):
                        ang = float(ee_term._last_rot_err_rad[velocity_env_idx].item())
                        pos = float(ee_term._last_main_err_m[velocity_env_idx].item())
                        print(f"末端执行器误差:   位置={pos:5.3f}m  姿态={math.degrees(ang):5.1f}°")
                        # 躯干 pitch：root（base_link）姿态去掉 yaw 后分解，
                        # 得到躯干相对水平面的俯仰。
                        # 注意：这里读的是躯干（root）姿态，不是末端执行器 ee 的姿态。
                        robot_art = ee_term.robot
                        trunk_quat_w = robot_art.data.root_quat_w
                        trunk_quat_no_yaw = quat_mul(quat_inv(yaw_quat(trunk_quat_w)), trunk_quat_w)
                        _, trunk_pitch, _ = euler_xyz_from_quat(trunk_quat_no_yaw)
                        trunk_pitch_deg = float(math.degrees(trunk_pitch[velocity_env_idx].item()))

                        # 命令高度 vs ee 实际高度（均在 projected COM yaw 系下，可直接对比）
                        cmd_h = float(ee_term.command[velocity_env_idx, 2].item())
                        ee_h_str = ""
                        ee_bid = ee_term.ee_body_id
                        if ee_bid is not None:
                            frame = ee_term._get_reference_frame()
                            ee_pos_w = ee_term.robot.data.body_pos_w[:, ee_bid]
                            ee_pos_p = world_to_projected_frame(ee_pos_w, frame)
                            ee_h = float(ee_pos_p[velocity_env_idx, 2].item())
                            ee_h_str = f"  ee高度={ee_h:5.2f}m"
                        print(f"                  躯干 pitch={trunk_pitch_deg:+6.1f}°  命令高度={cmd_h:5.2f}m{ee_h_str}")
                except Exception:
                    pass
                print("[vel] 控制: O=开关打印  ←→=切换环境")

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
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

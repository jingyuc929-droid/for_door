# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse

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
parser.add_argument("--obs_data", action="store_true", default=False, help="Run in debug mode.")
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

import glob
import os
import time

import gymnasium as gym
import numpy as np
import rl_sim_env.tasks
import torch
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import transform_points
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rl_algorithms.amp_utils.motion_loader import AMPLoader
from rl_algorithms.rsl_rl.runners import LocomotionOnPolicyRunner
from rl_algorithms.rsl_rl_wrapper import LocomotionOnPolicyRunnerCfg, LocomotionVecEnvWrapper
from rl_debug.marker import (
    BLUE_SPHERE_MARKER_CFG,
    GREEN_SPHERE_MARKER_CFG,
    RED_SPHERE_MARKER_CFG,
    YELLOW_SPHERE_MARKER_CFG,
)


def main():
    """Play with Parkour agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.scene.num_envs = 1
    # env_cfg.commands.base_velocity.debug_vis = False
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

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    motion_files = env.unwrapped.cfg.config_summary.env.train_cfg_dict['amp']['motion_files']

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = LocomotionVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = LocomotionOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    obs_dict = env.get_observations()
    amp_obs = obs_dict["amp_obs"]
    # wrap around environment for rsl-rl
    frame_dt = env_cfg.decimation * env.unwrapped.physics_dt
    amp_loader_cfg = env.unwrapped.cfg.amp_loader_cfg.to_dict()
    amp_loader = AMPLoader(device=env.device, amp_loader_cfg=amp_loader_cfg, time_between_frames=frame_dt, motion_files=motion_files)

    green_fl_cfg: VisualizationMarkersCfg = GREEN_SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Foot/FL_foot")
    red_fr_cfg: VisualizationMarkersCfg = RED_SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Foot/FR_foot")
    blue_rl_cfg: VisualizationMarkersCfg = BLUE_SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Foot/RL_foot")
    yellow_rr_cfg: VisualizationMarkersCfg = YELLOW_SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Foot/RR_foot")
    green_fl_visualizer = VisualizationMarkers(green_fl_cfg)
    red_fr_visualizer = VisualizationMarkers(red_fr_cfg)
    blue_rl_visualizer = VisualizationMarkers(blue_rl_cfg)
    yellow_rr_visualizer = VisualizationMarkers(yellow_rr_cfg)
    green_fl_visualizer.set_visibility(True)
    red_fr_visualizer.set_visibility(True)
    blue_rl_visualizer.set_visibility(True)
    yellow_rr_visualizer.set_visibility(True)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    dt = env.unwrapped.physics_dt

    # reset environment

    # simulate environment
    first_reset = True
    print(env.unwrapped.scene.articulations["robot"].joint_names)
    print(env.unwrapped.scene.articulations["robot"].body_names)

    t = 0.0
    traj_idx = 0

    while simulation_app.is_running() and traj_idx < amp_loader.traj_length - 1:

        if (t + amp_loader.time_between_frames + frame_dt) >= amp_loader.trajectory_duration[traj_idx]:
            traj_idx += 1
            t = 0
        else:
            t += frame_dt

        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # actions = policy(obs)
            actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)

            env_ids = torch.arange(env.num_envs, device=env.device)
            if first_reset:
                state = env.unwrapped.scene.get_state()
                print(state)
                first_reset = False
            root_pos = amp_loader.get_root_pos_batch(
                amp_loader.get_full_frame_at_time_batch(np.array([traj_idx]), np.array([t]))
            )
            root_orn = amp_loader.get_root_rot_batch(
                amp_loader.get_full_frame_at_time_batch(np.array([traj_idx]), np.array([t]))
            )
            joint_pos = amp_loader.get_joint_pos_batch(
                amp_loader.get_full_frame_at_time_batch(np.array([traj_idx]), np.array([t]))
            )
            foot_pos = amp_loader.get_frame_pos_batch(
                amp_loader.get_full_frame_at_time_batch(np.array([traj_idx]), np.array([t]))
            )
            foot_pos_amp_obs = amp_obs[:, -12:]
            if args_cli.obs_data:
                foot_data = foot_pos
            else:
                foot_data = foot_pos_amp_obs
            fl_foot_pos = transform_points(foot_data[:, 0:3], root_pos, root_orn)
            fr_foot_pos = transform_points(foot_data[:, 3:6], root_pos, root_orn)
            rl_foot_pos = transform_points(foot_data[:, 6:9], root_pos, root_orn)
            rr_foot_pos = transform_points(foot_data[:, 9:12], root_pos, root_orn)

            green_fl_visualizer.visualize(fl_foot_pos)
            red_fr_visualizer.visualize(fr_foot_pos)
            blue_rl_visualizer.visualize(rl_foot_pos)
            yellow_rr_visualizer.visualize(rr_foot_pos)
            state["articulation"]["robot"]["root_pose"][:, :3] = root_pos
            state["articulation"]["robot"]["root_pose"][:, 3:] = root_orn
            state["articulation"]["robot"]["root_velocity"][:, :6] = torch.zeros((env.num_envs, 6), device=env.device)
            state["articulation"]["robot"]["joint_position"][:] = joint_pos
            state["articulation"]["robot"]["joint_velocity"][:] = torch.zeros((env.num_envs, 12), device=env.device)
            # print(state["articulation"]["robot"])
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
            amp_obs = obs_buf["amp_obs"]
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

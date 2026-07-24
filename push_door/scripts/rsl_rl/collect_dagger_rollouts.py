#!/usr/bin/env python3
"""Collect DAgger data: Student drives, Teacher labels the visited states."""

import argparse
import json
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Collect Student-state / Teacher-action DAgger rollouts.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_teacher_cfg_entry_point")
parser.add_argument("--student_checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=None, help="Environment and evaluation seed.")
parser.add_argument("--episodes", type=int, default=300)
parser.add_argument("--chunk_steps", type=int, default=256)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--teacher_mix", type=float, default=0.20, help="Teacher action fraction in executed action.")
parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--disable_staged_reset", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--validate_args_only",
    action="store_true",
    default=False,
    help="Validate CLI arguments and checkpoint paths without launching Isaac Sim.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.episodes <= 0 or args_cli.chunk_steps <= 0 or args_cli.num_envs <= 0:
    parser.error("--episodes, --chunk_steps, and --num_envs must be positive")
if args_cli.checkpoint is None:
    parser.error("the following arguments are required: --checkpoint (Teacher PPO checkpoint)")
if not 0.0 <= args_cli.teacher_mix <= 1.0:
    parser.error("--teacher_mix must be in [0, 1]")
if not os.path.isfile(args_cli.checkpoint):
    parser.error(f"Teacher checkpoint not found: {args_cli.checkpoint}")
if not os.path.isfile(args_cli.student_checkpoint):
    parser.error(f"Student checkpoint not found: {args_cli.student_checkpoint}")
sys.argv = [sys.argv[0]] + hydra_args

import torch
import torch.nn as nn


class StudentGRU(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size, mlp_size):
        super().__init__()
        self.gru = nn.GRU(obs_dim, hidden_size, batch_first=True)
        self.action_head = nn.Sequential(nn.Linear(hidden_size, mlp_size), nn.ELU(), nn.Linear(mlp_size, action_dim))

    def forward(self, obs, hidden=None):
        features, hidden = self.gru(obs, hidden)
        return self.action_head(features), hidden


if args_cli.validate_args_only:
    student = torch.load(args_cli.student_checkpoint, map_location="cpu", weights_only=False)
    model = StudentGRU(
        int(student["obs_dim"]), int(student["action_dim"]), int(student["hidden_size"]), int(student["mlp_size"])
    )
    model.load_state_dict(student["model_state_dict"], strict=True)
    _sensor_settings = student.get("dataset_metadata", {}).get("noise_and_delay", {})
    print("[OK] DAgger CLI arguments and student checkpoint are valid.")
    print(f"task={args_cli.task}")
    print(f"teacher_checkpoint={args_cli.checkpoint}")
    print(f"student_checkpoint={args_cli.student_checkpoint}")
    print(f"student_dims=obs{student['obs_dim']}->action{student['action_dim']}")
    print(f"num_envs={args_cli.num_envs} episodes={args_cli.episodes} seed={args_cli.seed}")
    print(f"teacher_mix={args_cli.teacher_mix} headless={getattr(args_cli, 'headless', None)}")
    print(f"sensor_keys={sorted(_sensor_settings)}")
    sys.exit(0)


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import door_env.tasks  # noqa: F401
from door_env.tasks.manager_based.door_env.distillation import DoorBotDistillationVecEnvWrapper, DoorBotTeacherRunner


class StudentObservationTransform:
    """The same 57-D -> delayed/noisy 55-D transform used during BC training."""

    def __init__(self, initial_clean, settings):
        if initial_clean.shape[-1] != 57:
            raise RuntimeError(f"Expected policy_obs_clean to be 57-D, got {initial_clean.shape[-1]}")
        self.s = settings
        self.max_delay = max(int(settings[k]) for k in ("proprio_delay_steps", "geometry_delay_steps", "panel_delay_steps"))
        self.buffer = [initial_clean.clone() for _ in range(self.max_delay + 1)]
        self.previous_panel = self._clean(initial_clean)[:, -2:].clone()

    @staticmethod
    def _clean(x):
        return torch.cat((x[:, :36], x[:, 38:]), dim=-1)

    @staticmethod
    def _noise(x, width):
        return x if float(width) <= 0 else x + (2 * torch.rand_like(x) - 1) * float(width)

    @staticmethod
    def _rotate(x, angle):
        c, s = torch.cos(angle), torch.sin(angle)
        return torch.stack((c * x[:, 0] - s * x[:, 1], s * x[:, 0] + c * x[:, 1]), dim=-1)

    def reset(self, done, current):
        if torch.any(done):
            for value in self.buffer:
                value[done] = current[done]
            self.previous_panel[done] = self._clean(current)[done, -2:]

    def build(self, current):
        self.buffer.append(current.clone())
        self.buffer.pop(0)
        s = self.s
        proprio = self._clean(self.buffer[-1 - int(s["proprio_delay_steps"])])
        geometry = self._clean(self.buffer[-1 - int(s["geometry_delay_steps"])])
        panel = self._clean(self.buffer[-1 - int(s["panel_delay_steps"])])
        out = proprio.clone()
        out[:, 0:6] = self._noise(out[:, 0:6], s["arm_pos_noise"])
        out[:, 6:12] = self._noise(out[:, 6:12], s["arm_vel_noise"])
        out[:, 36:39] = self._noise(out[:, 36:39], s["imu_ang_vel_noise"])
        out[:, 39:42] = self._noise(out[:, 39:42], s["gravity_noise"])
        out[:, 42:43] = self._noise(out[:, 42:43], s["base_height_noise"])
        out[:, 43:45] = self._noise(geometry[:, 43:45], s["doorway_position_noise"])
        angle = torch.empty(out.shape[0], device=out.device).uniform_(-float(s["direction_noise_deg"]), float(s["direction_noise_deg"])) * torch.pi / 180
        out[:, 45:47] = self._rotate(geometry[:, 45:47], angle)
        out[:, 47:50] = self._noise(geometry[:, 47:50], s["handle_position_noise"])
        out[:, 50:53] = self._noise(geometry[:, 50:53], s["handle_position_noise"])
        panel_angle = torch.empty(out.shape[0], device=out.device).uniform_(-float(s["panel_direction_noise_deg"]), float(s["panel_direction_noise_deg"])) * torch.pi / 180
        panel_value = self._rotate(panel[:, 53:55], panel_angle)
        drop = torch.rand(out.shape[0], device=out.device) < float(s["panel_dropout_prob"])
        panel_value[drop] = self.previous_panel[drop]
        out[:, 53:55] = panel_value
        self.previous_panel = panel_value.clone()
        return out


def _disable_staged_reset(cfg):
    events = getattr(cfg, "events", None)
    for name in ("staged_reset", "reset_staged", "stage_reset"):
        term = getattr(events, name, None) if events is not None else None
        params = getattr(term, "params", None)
        if isinstance(params, dict):
            for key in ("p_grasp_start", "p_unlock_start", "p_opening_start"):
                if key in params:
                    params[key] = 0.0


def _sensor_settings(student_checkpoint):
    settings = dict(student_checkpoint.get("dataset_metadata", {}).get("noise_and_delay", {}))
    names = (
        "proprio_delay_steps", "geometry_delay_steps", "panel_delay_steps", "arm_pos_noise", "arm_vel_noise",
        "imu_ang_vel_noise", "gravity_noise", "base_height_noise", "doorway_position_noise",
        "handle_position_noise", "direction_noise_deg", "panel_direction_noise_deg", "panel_dropout_prob",
    )
    missing = [name for name in names if name not in settings]
    if missing:
        raise RuntimeError(f"Student checkpoint lacks sensor settings: {missing}")
    return settings


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: RslRlBaseRunnerCfg):
    student = torch.load(args_cli.student_checkpoint, map_location="cpu", weights_only=False)
    settings = _sensor_settings(student)
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.disable_staged_reset:
        _disable_staged_reset(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = DoorBotDistillationVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions,
        history_length=agent_cfg.history_length, transition_thresholds=agent_cfg.stage_transition_thresholds,
        collect_distillation_rollout=False)
    base_env = env.unwrapped

    teacher_path = retrieve_file_path(args_cli.checkpoint)
    teacher_runner = DoorBotTeacherRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    teacher_runner.load(teacher_path)
    teacher_policy = teacher_runner.get_inference_policy(device=base_env.device)

    model = StudentGRU(int(student["obs_dim"]), int(student["action_dim"]), int(student["hidden_size"]), int(student["mlp_size"])).to(base_env.device)
    model.load_state_dict(student["model_state_dict"])
    model.eval()
    obs_mean = student["obs_mean"].to(base_env.device)
    obs_std = student["obs_std"].to(base_env.device)
    action_clip = float(student.get("training_args", {}).get("action_clip", 1.0))

    output_dir = os.path.abspath(args_cli.output_dir or os.path.join("logs", "distillation", datetime.now().strftime("%Y-%m-%d_%H-%M-%S_dagger")))
    chunks_dir = os.path.join(output_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    obs = env.get_observations()
    transform = StudentObservationTransform(obs["policy_obs_clean"], settings)
    hidden = None
    episode_ids = torch.arange(env.num_envs, device=base_env.device, dtype=torch.long)
    next_episode_id = env.num_envs
    completed = 0
    chunk_index, chunk = 0, {}

    def append(name, value):
        chunk.setdefault(name, []).append(value.detach().cpu())

    def flush():
        nonlocal chunk_index, chunk
        if chunk:
            payload = {key: torch.stack(values) for key, values in chunk.items()}
            torch.save(payload, os.path.join(chunks_dir, f"rollout_{chunk_index:06d}.pt"))
            chunk_index += 1
            chunk = {}

    with torch.inference_mode():
        while simulation_app.is_running() and completed < args_cli.episodes:
            student_obs = transform.build(obs["policy_obs_clean"])
            normalized = (student_obs - obs_mean) / obs_std
            student_action, hidden = model(normalized.unsqueeze(1), hidden)
            student_action = student_action[:, 0].clamp(-action_clip, action_clip)
            teacher_action = teacher_policy(obs).clamp(-action_clip, action_clip)
            executed = (1.0 - args_cli.teacher_mix) * student_action + args_cli.teacher_mix * teacher_action
            append("student_obs", student_obs)
            append("teacher_action_clipped", teacher_action)
            append("teacher_action_raw", teacher_action)
            append("student_action", student_action)
            append("executed_action", executed)
            append("done", torch.zeros(env.num_envs, dtype=torch.bool, device=base_env.device))
            append("episode_id", episode_ids)
            next_obs, reward, dones, extras = env.step(executed)
            # Correct the done field for the exact transition just saved.
            chunk["done"][-1] = dones.detach().cpu()
            append("reward", reward)
            completed += int(dones.sum().item())
            done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                hidden[:, done_ids, :] = 0.0
                teacher_runner.alg.policy.reset(dones)
                transform.reset(dones, next_obs["policy_obs_clean"])
                for env_id in done_ids.tolist():
                    episode_ids[env_id] = next_episode_id
                    next_episode_id += 1
            obs = next_obs
            if len(chunk.get("student_obs", [])) >= args_cli.chunk_steps:
                flush()
    flush()
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as file:
        json.dump({
            "format_version": 1, "type": "dagger", "teacher_checkpoint": os.path.abspath(teacher_path),
            "student_checkpoint": os.path.abspath(args_cli.student_checkpoint), "task": args_cli.task,
            "seed": int(agent_cfg.seed), "num_envs": env.num_envs, "completed_episodes": completed,
            "teacher_mix": args_cli.teacher_mix, "noise_and_delay": settings,
            "student_observation_dim": int(student["obs_dim"]), "action_dim": int(student["action_dim"]),
        }, file, indent=2)
    env.close()
    print(f"[DONE] DAgger episodes={completed}, teacher_mix={args_cli.teacher_mix}, output={output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()

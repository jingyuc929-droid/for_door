# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Collect DoorBot teacher labels paired with deployable 55-D student observations."""

import argparse
import json
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Collect DoorBot teacher rollouts for student distillation.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_teacher_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--episodes", type=int, default=300)
parser.add_argument("--chunk_steps", type=int, default=256, help="Policy steps per saved tensor chunk.")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--disable_staged_reset", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--reset_xy_range", type=float, default=None)
parser.add_argument("--reset_yaw_range", type=float, default=None)
parser.add_argument("--arm_joint_pos_range", type=float, default=None)
for _name in (
    "door_stiffness_scale", "door_damping_scale", "door_friction_scale",
    "handle_stiffness_scale", "handle_damping_scale", "handle_friction_scale",
    "arm_effort_scale", "arm_damping_scale", "arm_action_scale",
):
    parser.add_argument(f"--{_name}", type=float, default=1.0)
parser.add_argument("--proprio_delay_steps", type=int, default=0)
parser.add_argument("--geometry_delay_steps", type=int, default=2)
parser.add_argument("--panel_delay_steps", type=int, default=4)
parser.add_argument("--arm_pos_noise", type=float, default=0.01)
parser.add_argument("--arm_vel_noise", type=float, default=0.10)
parser.add_argument("--imu_ang_vel_noise", type=float, default=0.03)
parser.add_argument("--gravity_noise", type=float, default=0.015)
parser.add_argument("--base_height_noise", type=float, default=0.01)
parser.add_argument("--doorway_position_noise", type=float, default=0.03)
parser.add_argument("--handle_position_noise", type=float, default=0.015)
parser.add_argument("--direction_noise_deg", type=float, default=3.0)
parser.add_argument("--panel_direction_noise_deg", type=float, default=10.0)
parser.add_argument("--panel_dropout_prob", type=float, default=0.08)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

for name in ("proprio_delay_steps", "geometry_delay_steps", "panel_delay_steps"):
    if getattr(args_cli, name) < 0:
        parser.error(f"--{name} must be non-negative")
if not 0.0 <= args_cli.panel_dropout_prob <= 1.0:
    parser.error("--panel_dropout_prob must be in [0, 1]")
if args_cli.episodes <= 0 or args_cli.chunk_steps <= 0:
    parser.error("--episodes and --chunk_steps must be positive")
if args_cli.reset_xy_range is not None and args_cli.reset_xy_range < 0.0:
    parser.error("--reset_xy_range must be non-negative")
if args_cli.reset_yaw_range is not None and args_cli.reset_yaw_range < 0.0:
    parser.error("--reset_yaw_range must be non-negative")
if args_cli.arm_joint_pos_range is not None and args_cli.arm_joint_pos_range < 0.0:
    parser.error("--arm_joint_pos_range must be non-negative")
for name in (
    "door_stiffness_scale", "door_damping_scale", "door_friction_scale",
    "handle_stiffness_scale", "handle_damping_scale", "handle_friction_scale",
    "arm_effort_scale", "arm_damping_scale", "arm_action_scale",
):
    if getattr(args_cli, name) < 0.0:
        parser.error(f"--{name} must be non-negative")

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaaclab.envs.mdp as mdp_std

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from door_env.tasks.manager_based.door_env.distillation import DoorBotDistillationVecEnvWrapper, DoorBotTeacherRunner

import isaaclab_tasks  # noqa: F401
import door_env.tasks  # noqa: F401


STUDENT_SCHEMA = (
    ("arm_joint_pos_rel", 6),
    ("arm_joint_vel", 6),
    ("last_applied_arm_delta", 6),
    ("arm_q_des_error", 6),
    ("last_high_base_action", 3),
    ("last_arm_action", 6),
    ("high_base_command", 3),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("base_height", 1),
    ("base_to_doorway_center_b_xy", 2),
    ("doorway_forward_axis_b_xy", 2),
    ("ee_to_handle_target_b", 3),
    ("handle_target_position_b", 3),
    ("door_panel_forward_axis_b_xy", 2),
)


def _disable_staged_reset(env_cfg):
    events = getattr(env_cfg, "events", None)
    for name in ("staged_reset", "reset_staged", "stage_reset"):
        term = getattr(events, name, None) if events is not None else None
        params = getattr(term, "params", None)
        if isinstance(params, dict):
            for key in ("p_grasp_start", "p_unlock_start", "p_opening_start"):
                if key in params:
                    params[key] = 0.0


def _configure_reset_ranges(env_cfg, xy_override, yaw_override):
    event = getattr(getattr(env_cfg, "events", None), "reset_robot_root", None)
    params = getattr(event, "params", None)
    if not isinstance(params, dict) or not isinstance(params.get("pose_range"), dict):
        raise RuntimeError("Task cfg has no configurable events.reset_robot_root pose_range.")
    pose_range = params["pose_range"]
    if xy_override is not None:
        value = float(xy_override)
        pose_range["x"] = (-value, value)
        pose_range["y"] = (-value, value)
    if yaw_override is not None:
        value = float(yaw_override)
        pose_range["yaw"] = (-value, value)

    def extent(name):
        low, high = pose_range.get(name, (0.0, 0.0))
        return max(abs(float(low)), abs(float(high)))

    return max(extent("x"), extent("y")), extent("yaw")


def _configure_arm_joint_reset(env_cfg, arm_joint_pos_range):
    value = 0.0 if arm_joint_pos_range is None else float(arm_joint_pos_range)
    if value == 0.0:
        return value
    env_cfg.events.reset_arm_joints_rollout = EventTermCfg(
        func=mdp_std.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["link[1-6]_joint"]),
            "position_range": (-value, value),
            "velocity_range": (0.0, 0.0),
        },
    )
    return value


def _configure_door_dynamics(env_cfg, args):
    actuators = env_cfg.scene.door.actuators
    scales = {
        "door_stiffness_scale": float(args.door_stiffness_scale),
        "door_damping_scale": float(args.door_damping_scale),
        "door_friction_scale": float(args.door_friction_scale),
        "handle_stiffness_scale": float(args.handle_stiffness_scale),
        "handle_damping_scale": float(args.handle_damping_scale),
        "handle_friction_scale": float(args.handle_friction_scale),
    }
    for joint_name in ("door_joint", "handle_joint"):
        prefix = joint_name.removesuffix("_joint")
        actuator = actuators[joint_name]
        actuator.stiffness *= scales[f"{prefix}_stiffness_scale"]
        actuator.damping *= scales[f"{prefix}_damping_scale"]
        friction_scale = scales[f"{prefix}_friction_scale"]
        for field in ("friction", "dynamic_friction", "viscous_friction"):
            value = getattr(actuator, field, None)
            if value is not None:
                setattr(actuator, field, value * friction_scale)
    return scales


def _scale_numeric(value, scale):
    if isinstance(value, tuple):
        return tuple(float(item) * scale for item in value)
    if isinstance(value, list):
        return [float(item) * scale for item in value]
    return float(value) * scale


def _configure_arm_actuator_errors(env_cfg, args):
    cfg = env_cfg.actions.high_level_action
    scales = {
        "arm_effort_scale": float(args.arm_effort_scale),
        "arm_damping_scale": float(args.arm_damping_scale),
        "arm_action_scale": float(args.arm_action_scale),
    }
    cfg.effort_limit = _scale_numeric(cfg.effort_limit, scales["arm_effort_scale"])
    cfg.arm_damping = _scale_numeric(cfg.arm_damping, scales["arm_damping_scale"])
    cfg.arm_action_scale = _scale_numeric(cfg.arm_action_scale, scales["arm_action_scale"])
    return scales


def _adapt_privileged_observations_to_checkpoint(env_cfg, path):
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    mean = checkpoint.get("model_state_dict", {}).get("privileged_normalizer._mean")
    schema = checkpoint.get("privileged_state_schema_and_order")
    names = {str(item[0]) for item in schema} if schema is not None else set()
    if torch.is_tensor(mean):
        dim = int(mean.shape[-1])
        legacy = (schema is not None and "door_open_direction" not in names) or (schema is None and dim in (17, 27))
        if legacy:
            env_cfg.observations.privileged_state.door_open_direction = None
            print(f"[INFO] Legacy {dim}-D checkpoint: disabled door_open_direction.")


class StudentObservationTransform:
    """Build clean/noisy 55-D observations with per-sensor delay buffers."""

    def __init__(self, initial_clean, args):
        if initial_clean.shape[-1] != 57:
            raise RuntimeError(
                f"Expected current policy_obs_clean to be 57-D, got {initial_clean.shape[-1]}. "
                "Update the explicit schema before collecting data."
            )
        self.args = args
        self.max_delay = max(args.proprio_delay_steps, args.geometry_delay_steps, args.panel_delay_steps)
        self.buffer = [initial_clean.clone() for _ in range(self.max_delay + 1)]
        self.previous_panel = self._student_clean(initial_clean)[:, -2:].clone()

    @staticmethod
    def _student_clean(clean57):
        # clean57[36:41] is [vx, vy, wx, wy, wz]. Drop vx/vy for deployment parity.
        return torch.cat((clean57[:, :36], clean57[:, 38:]), dim=-1)

    def reset(self, done, current_clean):
        if not torch.any(done):
            return
        for sample in self.buffer:
            sample[done] = current_clean[done]
        self.previous_panel[done] = self._student_clean(current_clean)[done, -2:]

    def _delayed(self, delay):
        return self.buffer[-1 - delay]

    @staticmethod
    def _add_uniform(x, half_width):
        if half_width <= 0.0:
            return x
        return x + (2.0 * torch.rand_like(x) - 1.0) * float(half_width)

    @staticmethod
    def _rotate_xy(xy, angle_rad):
        c, s = torch.cos(angle_rad), torch.sin(angle_rad)
        x, y = xy[:, 0], xy[:, 1]
        return torch.stack((c * x - s * y, s * x + c * y), dim=-1)

    def build(self, current_clean):
        self.buffer.append(current_clean.clone())
        self.buffer.pop(0)
        a = self.args
        proprio = self._student_clean(self._delayed(a.proprio_delay_steps))
        geometry = self._student_clean(self._delayed(a.geometry_delay_steps))
        panel = self._student_clean(self._delayed(a.panel_delay_steps))
        noisy = proprio.clone()

        noisy[:, 0:6] = self._add_uniform(noisy[:, 0:6], a.arm_pos_noise)
        noisy[:, 6:12] = self._add_uniform(noisy[:, 6:12], a.arm_vel_noise)
        noisy[:, 36:39] = self._add_uniform(noisy[:, 36:39], a.imu_ang_vel_noise)
        noisy[:, 39:42] = self._add_uniform(noisy[:, 39:42], a.gravity_noise)
        noisy[:, 42:43] = self._add_uniform(noisy[:, 42:43], a.base_height_noise)

        noisy[:, 43:45] = self._add_uniform(geometry[:, 43:45], a.doorway_position_noise)
        direction_angle = torch.empty(noisy.shape[0], device=noisy.device).uniform_(
            -a.direction_noise_deg, a.direction_noise_deg
        ) * torch.pi / 180.0
        noisy[:, 45:47] = self._rotate_xy(geometry[:, 45:47], direction_angle)
        noisy[:, 47:50] = self._add_uniform(geometry[:, 47:50], a.handle_position_noise)
        noisy[:, 50:53] = self._add_uniform(geometry[:, 50:53], a.handle_position_noise)

        panel_angle = torch.empty(noisy.shape[0], device=noisy.device).uniform_(
            -a.panel_direction_noise_deg, a.panel_direction_noise_deg
        ) * torch.pi / 180.0
        panel_noisy = self._rotate_xy(panel[:, 53:55], panel_angle)
        dropout = torch.rand(noisy.shape[0], device=noisy.device) < a.panel_dropout_prob
        panel_noisy[dropout] = self.previous_panel[dropout]
        noisy[:, 53:55] = panel_noisy
        self.previous_panel = panel_noisy.clone()
        return self._student_clean(current_clean), noisy, dropout


def _cpu(tensor):
    return tensor.detach().to(device="cpu")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.disable_staged_reset:
        _disable_staged_reset(env_cfg)
    reset_xy_range, reset_yaw_range = _configure_reset_ranges(
        env_cfg, args_cli.reset_xy_range, args_cli.reset_yaw_range
    )
    arm_joint_pos_range = _configure_arm_joint_reset(env_cfg, args_cli.arm_joint_pos_range)
    dynamics_scales = _configure_door_dynamics(env_cfg, args_cli)
    actuator_scales = _configure_arm_actuator_errors(env_cfg, args_cli)
    print(
        f"[INFO] rollout randomization reset_xy=+-{reset_xy_range:.3f}m "
        f"reset_yaw=+-{reset_yaw_range:.3f}rad arm_joint_pos=+-{arm_joint_pos_range:.3f}rad "
        f"dynamics={dynamics_scales} arm_actuator={actuator_scales}"
    )

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else get_checkpoint_path(
        log_root, agent_cfg.load_run, agent_cfg.load_checkpoint
    )
    _adapt_privileged_observations_to_checkpoint(env_cfg, resume_path)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = DoorBotDistillationVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
        history_length=agent_cfg.history_length,
        transition_thresholds=agent_cfg.stage_transition_thresholds,
        collect_distillation_rollout=True,
    )
    base_env = env.unwrapped
    runner = DoorBotTeacherRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading teacher: {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=base_env.device)
    policy_nn = runner.alg.policy

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.abspath(args_cli.output_dir or os.path.join("logs", "distillation", timestamp + "_teacher_rollout"))
    chunks_dir = os.path.join(output_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    obs = env.get_observations()
    transform = StudentObservationTransform(obs["policy_obs_clean"], args_cli)
    episode_ids = torch.arange(env.num_envs, device=base_env.device, dtype=torch.long)
    next_episode_id = env.num_envs
    completed_episodes = 0
    total_transitions = 0
    chunk_index = 0
    chunk = {}

    def append(name, value):
        chunk.setdefault(name, []).append(_cpu(value))

    def flush():
        nonlocal chunk_index, chunk
        if not chunk:
            return
        payload = {name: torch.stack(values, dim=0) for name, values in chunk.items()}
        path = os.path.join(chunks_dir, f"rollout_{chunk_index:06d}.pt")
        torch.save(payload, path)
        print(f"[INFO] Saved {path} ({payload['student_obs'].shape[0]} steps).")
        chunk_index += 1
        chunk = {}

    with torch.inference_mode():
        while simulation_app.is_running() and completed_episodes < args_cli.episodes:
            student_clean, student_obs, panel_dropout = transform.build(obs["policy_obs_clean"])
            actions = policy(obs) if args_cli.deterministic else policy_nn.act(obs)
            next_obs, rewards, dones, extras = env.step(actions)
            dist = extras["distillation"]

            append("student_obs_clean", student_clean)
            append("student_obs", student_obs)
            append("teacher_action_raw", dist["teacher_action_raw"])
            if agent_cfg.clip_actions is None:
                teacher_action_clipped = dist["teacher_action_raw"]
            else:
                teacher_action_clipped = torch.clamp(
                    dist["teacher_action_raw"], -float(agent_cfg.clip_actions), float(agent_cfg.clip_actions)
                )
            append("teacher_action_clipped", teacher_action_clipped)
            append("teacher_action_applied", dist["teacher_action_applied"])
            append("privileged_state", dist["privileged_state"])
            append("decoder_continuous_targets", dist["decoder_continuous_targets"])
            append("decoder_discrete_targets", dist["decoder_discrete_targets"])
            append("stage_id", dist["stage_id"])
            append("transition_flags", dist["transition_flags"])
            append("reward", rewards)
            append("done", dones.to(dtype=torch.bool))
            append("episode_id", episode_ids)
            append("panel_direction_dropout", panel_dropout)
            total_transitions += env.num_envs

            done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            completed_episodes += int(done_ids.numel())
            if done_ids.numel() > 0:
                policy_nn.reset(dones)
                for env_id in done_ids.tolist():
                    episode_ids[env_id] = next_episode_id
                    next_episode_id += 1
                transform.reset(dones.to(dtype=torch.bool), next_obs["policy_obs_clean"])
            obs = next_obs
            if len(chunk["student_obs"]) >= args_cli.chunk_steps:
                flush()

    flush()
    metadata = {
        "format_version": 1,
        "checkpoint": os.path.abspath(resume_path),
        "task": args_cli.task,
        "seed": int(agent_cfg.seed),
        "num_envs": env.num_envs,
        "completed_episodes": completed_episodes,
        "total_transitions": total_transitions,
        "deterministic_teacher": bool(args_cli.deterministic),
        "teacher_action_clip": None if agent_cfg.clip_actions is None else float(agent_cfg.clip_actions),
        "student_observation_dim": sum(dim for _, dim in STUDENT_SCHEMA),
        "student_observation_schema": list(STUDENT_SCHEMA),
        "dropped_from_teacher_clean_obs": ["base_linear_velocity_x", "base_linear_velocity_y"],
        "noise_and_delay": {
            key: getattr(args_cli, key)
            for key in (
                "proprio_delay_steps", "geometry_delay_steps", "panel_delay_steps", "arm_pos_noise",
                "arm_vel_noise", "imu_ang_vel_noise", "gravity_noise", "base_height_noise",
                "doorway_position_noise", "handle_position_noise", "direction_noise_deg",
                "panel_direction_noise_deg", "panel_dropout_prob",
            )
        },
        "environment_randomization": {
            "reset_xy_range_m": reset_xy_range,
            "reset_yaw_range_rad": reset_yaw_range,
            "arm_joint_pos_range_rad": arm_joint_pos_range,
            **dynamics_scales,
            **actuator_scales,
        },
        "chunk_layout": "[policy_step, environment, feature]",
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    env.close()
    print(f"[INFO] Collected {completed_episodes} episodes / {total_transitions} transitions under {output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()

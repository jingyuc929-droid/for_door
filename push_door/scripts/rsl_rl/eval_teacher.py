# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic evaluation for a DoorBot RSL-RL teacher checkpoint."""

import argparse
import csv
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


parser = argparse.ArgumentParser(description="Deterministically evaluate a DoorBot PPO teacher checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record evaluation video.")
parser.add_argument("--video_length", type=int, default=750, help="Length of the recorded video in policy steps.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate, e.g. 64.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_teacher_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--eval_episodes", type=int, default=200, help="Number of complete episodes to evaluate.")
parser.add_argument("--test_name", type=str, default="baseline", help="Name used to group this test's outputs.")
parser.add_argument(
    "--student_checkpoint",
    type=str,
    default=None,
    help="Evaluate an offline-trained GRU student checkpoint instead of the teacher policy.",
)
parser.add_argument(
    "--student_sensor_profile",
    choices=("checkpoint", "clean"),
    default="checkpoint",
    help="Use the checkpoint's rollout sensor settings or disable all student noise/delay/dropout.",
)
for _name, _type, _help in (
    ("proprio_delay_steps", int, "Override student proprioception delay in policy steps."),
    ("geometry_delay_steps", int, "Override student geometry delay in policy steps."),
    ("panel_delay_steps", int, "Override student panel-direction delay in policy steps."),
    ("arm_pos_noise", float, "Override arm-position uniform-noise half width."),
    ("arm_vel_noise", float, "Override arm-velocity uniform-noise half width."),
    ("imu_ang_vel_noise", float, "Override angular-velocity uniform-noise half width."),
    ("gravity_noise", float, "Override projected-gravity uniform-noise half width."),
    ("base_height_noise", float, "Override base-height uniform-noise half width."),
    ("doorway_position_noise", float, "Override doorway-position uniform-noise half width."),
    ("handle_position_noise", float, "Override handle-position uniform-noise half width."),
    ("direction_noise_deg", float, "Override doorway-direction angular noise in degrees."),
    ("panel_direction_noise_deg", float, "Override panel-direction angular noise in degrees."),
    ("panel_dropout_prob", float, "Override panel-direction dropout probability."),
):
    parser.add_argument(f"--{_name}", type=_type, default=None, help=_help)
parser.add_argument(
    "--reset_xy_range",
    type=float,
    default=None,
    help="Override symmetric robot reset x/y range in meters, e.g. 0.10 means [-0.10, 0.10].",
)
parser.add_argument(
    "--reset_yaw_range",
    type=float,
    default=None,
    help="Override symmetric robot reset yaw range in radians.",
)
parser.add_argument(
    "--arm_joint_pos_range",
    type=float,
    default=None,
    help="Add symmetric reset offsets to link1_joint..link6_joint only, in radians.",
)
for _name, _help in (
    ("door_stiffness_scale", "Scale door-joint stiffness."),
    ("door_damping_scale", "Scale door-joint damping."),
    ("door_friction_scale", "Scale all door-joint friction terms."),
    ("handle_stiffness_scale", "Scale handle-joint stiffness."),
    ("handle_damping_scale", "Scale handle-joint damping."),
    ("handle_friction_scale", "Scale all handle-joint friction terms."),
    ("arm_effort_scale", "Scale the six arm joints' effort limits."),
    ("arm_stiffness_scale", "Scale the six arm joints' controller stiffness."),
    ("arm_damping_scale", "Scale the six arm joints' controller damping."),
    ("arm_action_scale", "Scale policy-to-arm target increments."),
):
    parser.add_argument(f"--{_name}", type=float, default=1.0, help=_help)
parser.add_argument(
    "--disable_staged_reset",
    type=_str_to_bool,
    nargs="?",
    const=True,
    default=True,
    help="Disable staged reset starts.",
)
parser.add_argument(
    "--deterministic",
    type=_str_to_bool,
    nargs="?",
    const=True,
    default=True,
    help="Use deterministic inference policy.",
)
parser.add_argument("--door_joint_name", type=str, default="door_joint", help="Door joint used for success detection.")
parser.add_argument("--door_closed_pos", type=float, default=0.0, help="Door closed joint position.")
parser.add_argument(
    "--door_open_sign",
    type=float,
    default=None,
    help="Override the door-open sign. By default it is read from the task cfg.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.task is None:
    parser.error("--task is required, for example: --task Template-Door-Env-v0")
if args_cli.reset_xy_range is not None and args_cli.reset_xy_range < 0.0:
    parser.error("--reset_xy_range must be non-negative")
if args_cli.reset_yaw_range is not None and args_cli.reset_yaw_range < 0.0:
    parser.error("--reset_yaw_range must be non-negative")
if args_cli.arm_joint_pos_range is not None and args_cli.arm_joint_pos_range < 0.0:
    parser.error("--arm_joint_pos_range must be non-negative")
for _name in (
    "door_stiffness_scale", "door_damping_scale", "door_friction_scale",
    "handle_stiffness_scale", "handle_damping_scale", "handle_friction_scale",
    "arm_effort_scale", "arm_stiffness_scale", "arm_damping_scale", "arm_action_scale",
):
    if getattr(args_cli, _name) < 0.0:
        parser.error(f"--{_name} must be non-negative")
for _name in ("proprio_delay_steps", "geometry_delay_steps", "panel_delay_steps"):
    if getattr(args_cli, _name) is not None and getattr(args_cli, _name) < 0:
        parser.error(f"--{_name} must be non-negative")
for _name in (
    "arm_pos_noise", "arm_vel_noise", "imu_ang_vel_noise", "gravity_noise", "base_height_noise",
    "doorway_position_noise", "handle_position_noise", "direction_noise_deg", "panel_direction_noise_deg",
):
    if getattr(args_cli, _name) is not None and getattr(args_cli, _name) < 0.0:
        parser.error(f"--{_name} must be non-negative")
if args_cli.panel_dropout_prob is not None and not 0.0 <= args_cli.panel_dropout_prob <= 1.0:
    parser.error("--panel_dropout_prob must be in [0, 1]")

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import torch.nn as nn
import isaaclab.envs.mdp as mdp_std

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from door_env.tasks.manager_based.door_env.distillation import (
    DoorBotDistillationVecEnvWrapper,
    DoorBotTeacherRunner,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import door_env.tasks  # noqa: F401


class StudentGRU(nn.Module):
    """Architecture saved by train_student.py."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int, mlp_size: int, latent_dim: int = 0):
        super().__init__()
        self.gru = nn.GRU(obs_dim, hidden_size, num_layers=1, batch_first=True)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, mlp_size),
            nn.ELU(),
            nn.Linear(mlp_size, action_dim),
        )
        self.latent_head = nn.Linear(hidden_size, latent_dim) if int(latent_dim) > 0 else None

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor | None = None):
        features, hidden = self.gru(obs, hidden)
        return self.action_head(features), hidden


def _student_sensor_settings(metadata: dict) -> dict:
    settings = metadata.get("noise_and_delay")
    if isinstance(settings, dict):
        return dict(settings)
    for item in metadata.get("datasets", []):
        nested = item.get("metadata", {}) if isinstance(item, dict) else {}
        settings = nested.get("noise_and_delay")
        if isinstance(settings, dict):
            return dict(settings)
    return {}


class OnlineStudentPolicy:
    """Apply the rollout-time 55-D sensor transform and recurrent student online."""

    def __init__(self, checkpoint: dict, initial_clean: torch.Tensor, device: torch.device, eval_args):
        if initial_clean.shape[-1] != 57:
            raise RuntimeError(f"Expected policy_obs_clean to be 57-D, got {initial_clean.shape[-1]}.")
        self.device = device
        self.obs_mean = checkpoint["obs_mean"].to(device)
        self.obs_std = checkpoint["obs_std"].to(device)
        self.model = StudentGRU(
            int(checkpoint["obs_dim"]),
            int(checkpoint["action_dim"]),
            int(checkpoint["hidden_size"]),
            int(checkpoint["mlp_size"]),
            int(checkpoint.get("latent_dim", 0)),
        ).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.hidden = None

        metadata = checkpoint.get("dataset_metadata", {})
        self.sensor = _student_sensor_settings(metadata)
        required = (
            "proprio_delay_steps", "geometry_delay_steps", "panel_delay_steps", "arm_pos_noise",
            "arm_vel_noise", "imu_ang_vel_noise", "gravity_noise", "base_height_noise",
            "doorway_position_noise", "handle_position_noise", "direction_noise_deg",
            "panel_direction_noise_deg", "panel_dropout_prob",
        )
        missing = [name for name in required if name not in self.sensor]
        if missing:
            raise RuntimeError(f"Student checkpoint lacks rollout sensor settings: {missing}")
        if eval_args.student_sensor_profile == "clean":
            for name in required:
                self.sensor[name] = 0 if name.endswith("_steps") else 0.0
        for name in required:
            override = getattr(eval_args, name)
            if override is not None:
                self.sensor[name] = override
        self.max_delay = max(int(self.sensor[name]) for name in required[:3])
        self.buffer = [initial_clean.clone() for _ in range(self.max_delay + 1)]
        self.previous_panel = self._student_clean(initial_clean)[:, -2:].clone()
        training_args = checkpoint.get("training_args", {})
        self.action_clip = float(training_args.get("action_clip", 1.0))

    @staticmethod
    def _student_clean(clean57: torch.Tensor) -> torch.Tensor:
        return torch.cat((clean57[:, :36], clean57[:, 38:]), dim=-1)

    @staticmethod
    def _add_uniform(value: torch.Tensor, half_width: float) -> torch.Tensor:
        if half_width <= 0.0:
            return value
        return value + (2.0 * torch.rand_like(value) - 1.0) * half_width

    @staticmethod
    def _rotate_xy(xy: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        cosine, sine = torch.cos(angle), torch.sin(angle)
        return torch.stack((cosine * xy[:, 0] - sine * xy[:, 1], sine * xy[:, 0] + cosine * xy[:, 1]), dim=-1)

    def _delayed(self, steps: int) -> torch.Tensor:
        return self.buffer[-1 - int(steps)]

    def _build_observation(self, current_clean: torch.Tensor) -> torch.Tensor:
        self.buffer.append(current_clean.clone())
        self.buffer.pop(0)
        s = self.sensor
        proprio = self._student_clean(self._delayed(s["proprio_delay_steps"]))
        geometry = self._student_clean(self._delayed(s["geometry_delay_steps"]))
        panel = self._student_clean(self._delayed(s["panel_delay_steps"]))
        noisy = proprio.clone()
        noisy[:, 0:6] = self._add_uniform(noisy[:, 0:6], float(s["arm_pos_noise"]))
        noisy[:, 6:12] = self._add_uniform(noisy[:, 6:12], float(s["arm_vel_noise"]))
        noisy[:, 36:39] = self._add_uniform(noisy[:, 36:39], float(s["imu_ang_vel_noise"]))
        noisy[:, 39:42] = self._add_uniform(noisy[:, 39:42], float(s["gravity_noise"]))
        noisy[:, 42:43] = self._add_uniform(noisy[:, 42:43], float(s["base_height_noise"]))
        noisy[:, 43:45] = self._add_uniform(geometry[:, 43:45], float(s["doorway_position_noise"]))
        angle = torch.empty(noisy.shape[0], device=self.device).uniform_(-float(s["direction_noise_deg"]), float(s["direction_noise_deg"])) * torch.pi / 180.0
        noisy[:, 45:47] = self._rotate_xy(geometry[:, 45:47], angle)
        noisy[:, 47:50] = self._add_uniform(geometry[:, 47:50], float(s["handle_position_noise"]))
        noisy[:, 50:53] = self._add_uniform(geometry[:, 50:53], float(s["handle_position_noise"]))
        panel_angle = torch.empty(noisy.shape[0], device=self.device).uniform_(-float(s["panel_direction_noise_deg"]), float(s["panel_direction_noise_deg"])) * torch.pi / 180.0
        panel_noisy = self._rotate_xy(panel[:, 53:55], panel_angle)
        dropout = torch.rand(noisy.shape[0], device=self.device) < float(s["panel_dropout_prob"])
        panel_noisy[dropout] = self.previous_panel[dropout]
        noisy[:, 53:55] = panel_noisy
        self.previous_panel = panel_noisy.clone()
        return noisy

    def __call__(self, observations: dict) -> torch.Tensor:
        student_obs = self._build_observation(observations["policy_obs_clean"])
        normalized = (student_obs - self.obs_mean) / self.obs_std
        actions, self.hidden = self.model(normalized.unsqueeze(1), self.hidden)
        return actions[:, 0].clamp(-self.action_clip, self.action_clip)

    def reset(self, done: torch.Tensor, current_clean: torch.Tensor) -> None:
        done = done.to(dtype=torch.bool)
        if not torch.any(done):
            return
        if self.hidden is not None:
            self.hidden[:, done, :] = 0.0
        for sample in self.buffer:
            sample[done] = current_clean[done]
        self.previous_panel[done] = self._student_clean(current_clean)[done, -2:]


def _disable_staged_reset_in_cfg(env_cfg) -> bool:
    changed = False
    candidates = []
    events = getattr(env_cfg, "events", None)
    if events is not None:
        for name in ("staged_reset", "reset_staged", "stage_reset"):
            if hasattr(events, name):
                event_term = getattr(events, name)
                if event_term is None:
                    changed = True
                else:
                    candidates.append(event_term)
    for event_term in candidates:
        params = getattr(event_term, "params", None)
        if isinstance(params, dict):
            for key in ("p_grasp_start", "p_unlock_start", "p_opening_start"):
                if key in params:
                    params[key] = 0.0
                    changed = True
    return changed


def _configure_reset_ranges(env_cfg, xy_override: float | None, yaw_override: float | None) -> tuple[float, float]:
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

    def _symmetric_extent(name: str) -> float:
        limits = pose_range.get(name, (0.0, 0.0))
        return max(abs(float(limits[0])), abs(float(limits[1])))

    xy_range = max(_symmetric_extent("x"), _symmetric_extent("y"))
    yaw_range = _symmetric_extent("yaw")
    return xy_range, yaw_range


def _safe_test_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value.strip())
    return safe or "baseline"


def _configure_arm_joint_reset(env_cfg, arm_joint_pos_range: float | None) -> float:
    """Append an arm-only reset event after the nominal all-joint reset."""
    value = 0.0 if arm_joint_pos_range is None else float(arm_joint_pos_range)
    if value == 0.0:
        return value
    env_cfg.events.reset_arm_joints_test = EventTermCfg(
        func=mdp_std.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["link[1-6]_joint"]),
            "position_range": (-value, value),
            "velocity_range": (0.0, 0.0),
        },
    )
    return value


def _configure_door_dynamics(env_cfg, args) -> dict[str, float]:
    """Apply one or more explicit actuator-property scales before scene creation."""
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
        actuator = actuators[joint_name]
        actuator.stiffness *= scales[f"{joint_name.removesuffix('_joint')}_stiffness_scale"]
        actuator.damping *= scales[f"{joint_name.removesuffix('_joint')}_damping_scale"]
        friction_scale = scales[f"{joint_name.removesuffix('_joint')}_friction_scale"]
        for field in ("friction", "dynamic_friction", "viscous_friction"):
            value = getattr(actuator, field, None)
            if value is not None:
                setattr(actuator, field, value * friction_scale)
    return scales


def _scale_numeric(value, scale: float):
    if isinstance(value, tuple):
        return tuple(float(item) * scale for item in value)
    if isinstance(value, list):
        return [float(item) * scale for item in value]
    return float(value) * scale


def _configure_arm_actuator_errors(env_cfg, args) -> dict[str, float]:
    """Scale the high-level arm controller values that are written to simulation."""
    cfg = env_cfg.actions.high_level_action
    scales = {
        "arm_effort_scale": float(args.arm_effort_scale),
        "arm_stiffness_scale": float(args.arm_stiffness_scale),
        "arm_damping_scale": float(args.arm_damping_scale),
        "arm_action_scale": float(args.arm_action_scale),
    }
    cfg.effort_limit = _scale_numeric(cfg.effort_limit, scales["arm_effort_scale"])
    cfg.arm_stiffness = _scale_numeric(cfg.arm_stiffness, scales["arm_stiffness_scale"])
    cfg.arm_damping = _scale_numeric(cfg.arm_damping, scales["arm_damping_scale"])
    cfg.arm_action_scale = _scale_numeric(cfg.arm_action_scale, scales["arm_action_scale"])
    return scales


def _find_tensor_by_key(obj, key: str):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) == key and torch.is_tensor(v):
                return v
            found = _find_tensor_by_key(v, key)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found = _find_tensor_by_key(item, key)
            if found is not None:
                return found
    return None


def _termination_mask(base_env, extras, name: str, dones: torch.Tensor) -> torch.Tensor:
    for key in (
        name,
        f"Episode_Termination/{name}",
        f"termination/{name}",
        f"terminated/{name}",
        f"truncated/{name}",
    ):
        found = _find_tensor_by_key(extras, key)
        if found is not None and found.numel() == dones.numel():
            return found.to(device=dones.device, dtype=torch.bool).reshape_as(dones)

    term_manager = getattr(base_env, "termination_manager", None)
    if term_manager is not None:
        for attr_name in ("terminated", "truncated", "_term_dones", "_trunc_dones"):
            container = getattr(term_manager, attr_name, None)
            if isinstance(container, dict) and name in container and torch.is_tensor(container[name]):
                value = container[name]
                if value.numel() == dones.numel():
                    return value.to(device=dones.device, dtype=torch.bool).reshape_as(dones)
        for method_name in ("get_term", "get_active_iterable_terms"):
            method = getattr(term_manager, method_name, None)
            if method is not None:
                try:
                    value = method(name)
                    if torch.is_tensor(value) and value.numel() == dones.numel():
                        return value.to(device=dones.device, dtype=torch.bool).reshape_as(dones)
                except Exception:
                    pass

    return torch.zeros_like(dones, dtype=torch.bool)


def _resolve_door_joint_id(door, joint_name: str) -> int:
    joint_names = list(door.data.joint_names)
    if joint_name in joint_names:
        return joint_names.index(joint_name)
    candidates = [i for i, name in enumerate(joint_names) if joint_name.lower() in name.lower()]
    if candidates:
        return candidates[0]
    raise RuntimeError(f"Could not find door joint {joint_name!r}. Available joints: {joint_names}")


def _door_open(door, joint_id: int, closed_pos: float, open_sign: float) -> torch.Tensor:
    pos = door.data.joint_pos[:, joint_id]
    return float(open_sign) * (pos - float(closed_pos))


def _door_open_sign_from_cfg(env_cfg, override: float | None = None) -> float:
    if override is not None:
        return float(override)
    mechanism = getattr(getattr(env_cfg, "events", None), "door_mechanism", None)
    params = getattr(mechanism, "params", None)
    if isinstance(params, dict) and "door_open_sign" in params:
        return float(params["door_open_sign"])
    print("[WARN] door_open_sign was not found in task cfg; falling back to +1.0.")
    return 1.0


def _validate_checkpoint_door_metadata(path: str, task_id: str, door_open_sign: float) -> None:
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    metadata = checkpoint.get("door_task_metadata")
    if metadata is None:
        print("[WARN] Checkpoint has no door_task_metadata; skipping Push/Pull compatibility check.")
        return
    saved_sign = float(metadata.get("door_open_sign", door_open_sign))
    if saved_sign * float(door_open_sign) < 0.0:
        raise RuntimeError(
            f"Checkpoint door mode mismatch: checkpoint={metadata}, task={task_id!r}, "
            f"task door_open_sign={door_open_sign:+.1f}."
        )
    print(f"[INFO] Checkpoint door metadata: {metadata}")


def _adapt_privileged_observations_to_checkpoint(env_cfg, path: str) -> None:
    """Disable the later-added direction term for legacy 17/27-D checkpoints."""
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", {})
    mean = state_dict.get("privileged_normalizer._mean")
    if not torch.is_tensor(mean):
        return
    checkpoint_dim = int(mean.shape[-1])
    schema = checkpoint.get("privileged_state_schema_and_order")
    names = {str(item[0]) for item in schema} if schema is not None else set()
    legacy = (schema is not None and "door_open_direction" not in names) or (
        schema is None and checkpoint_dim in (17, 27)
    )
    if legacy:
        privileged_cfg = env_cfg.observations.privileged_state
        if not hasattr(privileged_cfg, "door_open_direction"):
            raise RuntimeError("Legacy checkpoint detected, but door_open_direction cannot be disabled in this task cfg.")
        privileged_cfg.door_open_direction = None
        print(f"[INFO] Legacy {checkpoint_dim}-D checkpoint: disabled privileged door_open_direction for evaluation.")
    else:
        print(f"[INFO] Checkpoint privileged observation dimension: {checkpoint_dim}")


def _write_results_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "episode_id",
        "env_id",
        "test_name",
        "seed",
        "reset_xy_range_m",
        "reset_yaw_range_rad",
        "arm_joint_pos_range_rad",
        "initial_arm_offset_rms_rad",
        "initial_arm_offset_max_abs_rad",
        "initial_arm_offset_link1",
        "initial_arm_offset_link2",
        "initial_arm_offset_link3",
        "initial_arm_offset_link4",
        "initial_arm_offset_link5",
        "initial_arm_offset_link6",
        "door_stiffness_scale",
        "door_damping_scale",
        "door_friction_scale",
        "handle_stiffness_scale",
        "handle_damping_scale",
        "handle_friction_scale",
        "arm_effort_scale",
        "arm_stiffness_scale",
        "arm_damping_scale",
        "arm_action_scale",
        "success",
        "grasp_reached",
        "unlock_reached",
        "required_door_angle_reached",
        "episode_length",
        "final_door_open",
        "max_door_open",
        "timeout",
        "base_bad_orientation",
        "base_fall",
        "bad_arm_pose",
        "base_out_of_hinge_radius",
        "other_termination",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    test_name = _safe_test_name(args_cli.test_name)
    reset_xy_range, reset_yaw_range = _configure_reset_ranges(
        env_cfg, args_cli.reset_xy_range, args_cli.reset_yaw_range
    )
    arm_joint_pos_range = _configure_arm_joint_reset(env_cfg, args_cli.arm_joint_pos_range)
    dynamics_scales = _configure_door_dynamics(env_cfg, args_cli)
    actuator_scales = _configure_arm_actuator_errors(env_cfg, args_cli)
    door_open_sign = _door_open_sign_from_cfg(env_cfg, args_cli.door_open_sign)
    print(
        f"[INFO] test={test_name} task={args_cli.task} seed={agent_cfg.seed} "
        f"reset_xy=±{reset_xy_range:.3f}m reset_yaw=±{reset_yaw_range:.3f}rad "
        f"arm_joint_pos=±{arm_joint_pos_range:.3f}rad "
        f"dynamics={dynamics_scales} "
        f"arm_actuator={actuator_scales} "
        f"door_open_sign={door_open_sign:+.1f}"
    )

    staged_reset_disabled = False
    if args_cli.disable_staged_reset:
        staged_reset_disabled = _disable_staged_reset_in_cfg(env_cfg)
        if not staged_reset_disabled:
            print("[WARN] Requested --disable_staged_reset=True, but staged reset probability fields were not found.")

    student_path = None
    student_checkpoint = None
    if args_cli.student_checkpoint:
        student_path = retrieve_file_path(args_cli.student_checkpoint)
        student_checkpoint = torch.load(student_path, weights_only=False, map_location="cpu")
        saved_task = student_checkpoint.get("dataset_metadata", {}).get("task")
        if saved_task is not None and saved_task != args_cli.task:
            raise RuntimeError(f"Student was trained for task {saved_task!r}, but evaluation task is {args_cli.task!r}.")

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    elif student_checkpoint is not None:
        resume_path = student_checkpoint.get("dataset_metadata", {}).get("checkpoint")
        if not resume_path or not os.path.isfile(resume_path):
            raise FileNotFoundError(
                "The source teacher checkpoint recorded in the student checkpoint does not exist: "
                f"{resume_path!r}. Pass its current path with --checkpoint."
            )
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    checkpoint_dir = os.path.dirname(student_path or resume_path)
    env_cfg.log_dir = checkpoint_dir
    _validate_checkpoint_door_metadata(resume_path, args_cli.task, door_open_sign)
    _adapt_privileged_observations_to_checkpoint(env_cfg, resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(checkpoint_dir, "videos", "eval_student" if student_path else "eval_teacher"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording evaluation video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if agent_cfg.class_name == "DoorBotTeacherRunner":
        env = DoorBotDistillationVecEnvWrapper(
            env,
            clip_actions=agent_cfg.clip_actions,
            history_length=agent_cfg.history_length,
            transition_thresholds=agent_cfg.stage_transition_thresholds,
            collect_distillation_rollout=getattr(agent_cfg, "collect_distillation_rollout", False),
        )
    else:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped
    door = base_env.scene["door"]
    robot = base_env.scene["robot"]
    arm_joint_ids = robot.find_joints("link[1-6]_joint")[0]
    if len(arm_joint_ids) != 6:
        raise RuntimeError(f"Expected six arm joints, got ids={arm_joint_ids}.")
    door_joint_id = _resolve_door_joint_id(door, args_cli.door_joint_name)
    traverse_cfg = getattr(base_env.cfg.terminations, "base_traverse_success", None)
    if traverse_cfg is None:
        raise RuntimeError("Current task has no base_traverse_success termination.")
    traverse_params = traverse_cfg.params or {}
    required_door_angle = float(traverse_params.get("required_door_angle", 1.0))
    pass_distance = float(traverse_params.get("pass_distance", 1.8))
    required_success_steps = int(traverse_params.get("num_steps", 3))

    obs = env.get_observations()
    if student_checkpoint is not None:
        print(f"[INFO] Loading student checkpoint from: {student_path}")
        policy = OnlineStudentPolicy(student_checkpoint, obs["policy_obs_clean"], base_env.device, args_cli)
        policy_nn = None
    else:
        if agent_cfg.class_name == "DoorBotTeacherRunner":
            runner = DoorBotTeacherRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=base_env.device)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

    num_envs = env.num_envs
    arm_default = robot.data.default_joint_pos[:, arm_joint_ids]
    initial_arm_offsets = (robot.data.joint_pos[:, arm_joint_ids] - arm_default).clone()
    episode_lengths = torch.zeros(num_envs, dtype=torch.int32, device=base_env.device)
    max_door_open = _door_open(door, door_joint_id, args_cli.door_closed_pos, door_open_sign).clone()
    ever_grasped = torch.zeros(num_envs, dtype=torch.bool, device=base_env.device)
    ever_unlocked = torch.zeros_like(ever_grasped)
    ever_required_open = max_door_open >= required_door_angle

    rows = []
    total_episodes = 0
    success_episodes = 0
    timeout_episodes = 0
    grasp_episodes = 0
    unlock_episodes = 0
    required_open_episodes = 0
    termination_counts = {
        "base_bad_orientation": 0,
        "base_fall": 0,
        "bad_arm_pose": 0,
        "base_out_of_hinge_radius": 0,
        "other_termination": 0,
    }
    next_progress_episodes = 20

    with torch.inference_mode():
        while simulation_app.is_running() and total_episodes < int(args_cli.eval_episodes):
            actions = policy(obs)
            door_open_before = _door_open(door, door_joint_id, args_cli.door_closed_pos, door_open_sign)
            max_door_open = torch.maximum(max_door_open, door_open_before)
            grasped = getattr(base_env, "_grasp_success_given", None)
            if torch.is_tensor(grasped):
                ever_grasped |= grasped.to(dtype=torch.bool)
            unlocked = getattr(base_env, "_unlock_success_given", None)
            if torch.is_tensor(unlocked):
                ever_unlocked |= unlocked.to(dtype=torch.bool)
            ever_required_open |= door_open_before >= required_door_angle

            obs, _rew, dones, extras = env.step(actions)

            door_open_after = _door_open(door, door_joint_id, args_cli.door_closed_pos, door_open_sign)
            max_door_open = torch.maximum(max_door_open, door_open_after)
            episode_lengths += 1

            timeout_mask = _termination_mask(base_env, extras, "time_out", dones)
            success_mask = _termination_mask(base_env, extras, "base_traverse_success", dones)
            failure_masks = {
                name: _termination_mask(base_env, extras, name, dones)
                for name in ("base_bad_orientation", "base_fall", "bad_arm_pose", "base_out_of_hinge_radius")
            }

            done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            for env_id_t in done_ids:
                if total_episodes >= int(args_cli.eval_episodes):
                    break
                env_id = int(env_id_t.item())
                final_open = float(torch.maximum(door_open_before[env_id], door_open_after[env_id]).item())
                max_open = float(max_door_open[env_id].item())
                success = bool(success_mask[env_id].item())
                timeout = bool(timeout_mask[env_id].item())
                failures = {name: bool(mask[env_id].item()) for name, mask in failure_masks.items()}
                other_termination = bool(dones[env_id].item()) and not (
                    success or timeout or any(failures.values())
                )
                grasp_reached = bool(ever_grasped[env_id].item())
                unlock_reached = bool(ever_unlocked[env_id].item())
                required_open_reached = bool(ever_required_open[env_id].item()) or success
                arm_offset = initial_arm_offsets[env_id]
                arm_offset_rms = float(torch.sqrt(torch.mean(torch.square(arm_offset))).item())
                arm_offset_max = float(torch.max(torch.abs(arm_offset)).item())

                rows.append(
                    {
                        "episode_id": total_episodes,
                        "env_id": env_id,
                        "test_name": test_name,
                        "seed": int(agent_cfg.seed),
                        "reset_xy_range_m": reset_xy_range,
                        "reset_yaw_range_rad": reset_yaw_range,
                        "arm_joint_pos_range_rad": arm_joint_pos_range,
                        "initial_arm_offset_rms_rad": arm_offset_rms,
                        "initial_arm_offset_max_abs_rad": arm_offset_max,
                        **{
                            f"initial_arm_offset_link{index + 1}": float(arm_offset[index].item())
                            for index in range(6)
                        },
                        **dynamics_scales,
                        **actuator_scales,
                        "success": int(success),
                        "grasp_reached": int(grasp_reached),
                        "unlock_reached": int(unlock_reached),
                        "required_door_angle_reached": int(required_open_reached),
                        "episode_length": int(episode_lengths[env_id].item()),
                        "final_door_open": final_open,
                        "max_door_open": max_open,
                        "timeout": int(timeout),
                        **{name: int(value) for name, value in failures.items()},
                        "other_termination": int(other_termination),
                    }
                )

                total_episodes += 1
                success_episodes += int(success)
                timeout_episodes += int(timeout)
                grasp_episodes += int(grasp_reached)
                unlock_episodes += int(unlock_reached)
                required_open_episodes += int(required_open_reached)
                for name, value in failures.items():
                    termination_counts[name] += int(value)
                termination_counts["other_termination"] += int(other_termination)

                episode_lengths[env_id] = 0
                max_door_open[env_id] = door_open_after[env_id]
                ever_grasped[env_id] = False
                ever_unlocked[env_id] = False
                ever_required_open[env_id] = False

            # ManagerBasedRLEnv auto-resets done environments inside step().
            # Capture the new episode's measured initial offsets before the
            # next policy action can move the arm.
            if done_ids.numel() > 0:
                initial_arm_offsets[done_ids] = (
                    robot.data.joint_pos[done_ids][:, arm_joint_ids]
                    - robot.data.default_joint_pos[done_ids][:, arm_joint_ids]
                )

            if done_ids.numel() > 0:
                if isinstance(policy, OnlineStudentPolicy):
                    policy.reset(dones, obs["policy_obs_clean"])
                else:
                    policy_nn.reset(dones)

            while total_episodes >= next_progress_episodes:
                print(
                    f"Evaluated {total_episodes}/{args_cli.eval_episodes} episodes, "
                    f"success_rate={100.0 * success_episodes / max(1, total_episodes):.2f}%"
                )
                next_progress_episodes += 20

    policy_kind = "student" if student_path else "teacher"
    eval_dir = os.path.join(
        "logs", "eval", test_name, datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S_seed{agent_cfg.seed}_{policy_kind}")
    )
    csv_path = os.path.abspath(os.path.join(eval_dir, f"eval_{policy_kind}_results.csv"))
    _write_results_csv(csv_path, rows)

    total = max(1, len(rows))
    mean_episode_length = sum(row["episode_length"] for row in rows) / total
    mean_final_open = sum(row["final_door_open"] for row in rows) / total
    mean_max_open = sum(row["max_door_open"] for row in rows) / total
    max_open_all = max((row["max_door_open"] for row in rows), default=0.0)
    mean_initial_arm_rms = sum(row["initial_arm_offset_rms_rad"] for row in rows) / total
    max_initial_arm_abs = max((row["initial_arm_offset_max_abs_rad"] for row in rows), default=0.0)

    print(f"\n========== Deterministic {policy_kind.title()} Evaluation ==========")
    print(f"Checkpoint: {student_path or resume_path}")
    if student_path:
        print(f"Source teacher checkpoint: {resume_path}")
        print(f"Student sensor profile: {args_cli.student_sensor_profile}")
        print(f"Student sensor settings: {policy.sensor}")
    print(f"Test name: {test_name}")
    print(f"Seed: {agent_cfg.seed}")
    print(f"Robot reset XY range: ±{reset_xy_range:.3f} m")
    print(f"Robot reset yaw range: ±{reset_yaw_range:.3f} rad")
    print(f"Arm joint reset range: ±{arm_joint_pos_range:.3f} rad")
    print(f"Measured initial arm offset RMS: {mean_initial_arm_rms:.5f} rad")
    print(f"Measured initial arm offset max abs: {max_initial_arm_abs:.5f} rad")
    print(f"Door/handle dynamics scales: {dynamics_scales}")
    print(f"Arm actuator scales: {actuator_scales}")
    print(f"Num envs: {num_envs}")
    print(f"Eval episodes: {len(rows)}")
    print(f"Staged reset disabled: {staged_reset_disabled}")
    print(f"Deterministic policy: {args_cli.deterministic}")
    print(
        f"Success condition: door opened once > {required_door_angle:.3f} rad, "
        f"pass distance > {pass_distance:.3f} m for {required_success_steps} steps"
    )
    print("")
    print(f"Success episodes: {success_episodes}")
    print(f"Success rate: {100.0 * success_episodes / total:.2f} %")
    print("")
    print(f"Grasp reached: {grasp_episodes} ({100.0 * grasp_episodes / total:.2f} %)")
    print(f"Unlock reached: {unlock_episodes} ({100.0 * unlock_episodes / total:.2f} %)")
    print(f"Required door angle reached: {required_open_episodes} ({100.0 * required_open_episodes / total:.2f} %)")
    print("")
    print(f"Timeout episodes: {timeout_episodes}")
    print(f"Bad orientation episodes: {termination_counts['base_bad_orientation']}")
    print(f"Fall episodes: {termination_counts['base_fall']}")
    print(f"Bad arm pose episodes: {termination_counts['bad_arm_pose']}")
    print(f"Out of hinge radius episodes: {termination_counts['base_out_of_hinge_radius']}")
    print(f"Other termination episodes: {termination_counts['other_termination']}")
    print("")
    print(f"Mean episode length: {mean_episode_length:.2f}")
    print(f"Mean final door_open: {mean_final_open:.4f}")
    print(f"Mean max door_open: {mean_max_open:.4f}")
    print(f"Max door_open over all episodes: {max_open_all:.4f}")
    print(f"CSV: {csv_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from collections import deque

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
    "--agent", type=str, default="rsl_rl_teacher_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# --- Debug: force gripper open/close to validate action->target->drive path ---
parser.add_argument(
    "--force_gripper",
    action="store_true",
    default=False,
    help="Override a gripper action for older assets. Ignored for the current fixed-hook robot.",
)
parser.add_argument(
    "--force_gripper_seconds",
    type=float,
    default=1.0,
    help="Duration (seconds) for each forced gripper phase.",
)
parser.add_argument(
    "--force_gripper_pattern",
    type=str,
    default="close_open",
    choices=["close", "open", "close_open"],
    help="Pattern for forced gripper action.",
)
parser.add_argument(
    "--force_gripper_print_every",
    type=int,
    default=10,
    help="Print debug info every N env steps while playing.",
)

parser.add_argument(
    "--print_contact_forces",
    action="store_true",
    default=False,
    help="If set, print left/right contact sensor net force vectors and norms (env0) at the debug print frequency.",
)
parser.add_argument(
    "--left_contact_sensor",
    type=str,
    default="hook_contact",
    help="Scene name of the primary hook/contact sensor (ContactSensor).",
)
parser.add_argument(
    "--right_contact_sensor",
    type=str,
    default="hook_contact",
    help="Optional second contact sensor. Defaults to the hook sensor for fixed-hook assets.",
)

# --- Lightweight rollout stats (reduce print spam) ---
parser.add_argument(
    "--stats_every",
    type=int,
    default=0,
    help="Print compact rollout stats every N env steps (0 disables; disabled by default).",
)
parser.add_argument(
    "--stats_contact_threshold",
    type=float,
    default=0.5,
    help="Threshold (N) on handle-filtered contact force to count as 'any contact'.",
)
parser.add_argument(
    "--stats_min_sep",
    type=float,
    default=0.002,
    help="Minimum separation (in handle frame along grasp axis) to count wrap alignment.",
)
parser.add_argument(
    "--stats_open_width",
    type=float,
    default=0.088,
    help="Open width used to compute closedness for legacy gripper assets. Ignored for the fixed-hook robot.",
)
parser.add_argument(
    "--debug_fingers",
    action="store_true",
    default=False,
    help="Print detailed finger joint/force info for env0 at the stats frequency.",
)
parser.add_argument(
    "--debug_setup",
    action="store_true",
    default=False,
    help="Print actuator/joint mapping once at startup.",
)
parser.add_argument(
    "--record_success_trajectories",
    action="store_true",
    default=False,
    help="Record per-frame episode trajectories and save only successful episodes by default.",
)
parser.add_argument(
    "--success_door_open_threshold",
    type=float,
    default=0.30,
    help="Door-open threshold used to classify a recorded episode as successful.",
)
parser.add_argument(
    "--max_success_trajectories",
    type=int,
    default=5,
    help="Stop play after saving this many successful trajectories.",
)
parser.add_argument(
    "--trajectory_output_dir",
    type=str,
    default="logs/trajectories",
    help="Root directory for saved trajectory recordings.",
)
parser.add_argument(
    "--disable_staged_reset",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Disable staged reset before env creation. Use --no-disable_staged_reset to keep staged reset enabled.",
)
parser.add_argument(
    "--save_failed_trajectories",
    action="store_true",
    default=False,
    help="Also save failed episodes when recording trajectories.",
)
parser.add_argument(
    "--trajectory_format",
    choices=["npz", "csv", "both"],
    default="npz",
    help="Trajectory file format. NPZ is always recommended for waypoint loading.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.record_success_trajectories and "--agent" not in sys.argv:
    args_cli.agent = "rsl_rl_teacher_cfg_entry_point"
    print("[INFO] record_success_trajectories enabled: defaulting --agent to rsl_rl_teacher_cfg_entry_point")
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import csv
import os
import time
from datetime import datetime
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
try:
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    get_published_pretrained_checkpoint = None

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import door_env.tasks  # noqa: F401
from door_env.tasks.manager_based.door_env.distillation import (
    DoorBotDistillationVecEnvWrapper,
    DoorBotTeacherRunner,
    export_doorbot_teacher,
)


def _door_open_sign_from_cfg(env_cfg) -> float:
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


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    door_open_sign = _door_open_sign_from_cfg(env_cfg)
    is_pull_task = "pull" in str(args_cli.task).lower()
    print(f"[INFO] task={args_cli.task} door_open_sign={door_open_sign:+.1f}")

    if args_cli.disable_staged_reset:
        staged_reset = getattr(getattr(env_cfg, "events", None), "staged_reset", None)
        if staged_reset is not None:
            params = staged_reset.params
            for key in ("p_grasp_start", "p_unlock_start", "p_opening_start"):
                if key in params:
                    params[key] = 0.0
            print("Staged reset disabled: True")
            print(f"p_grasp_start={params.get('p_grasp_start', 'N/A')}")
            print(f"p_unlock_start={params.get('p_unlock_start', 'N/A')}")
            if "p_opening_start" in params:
                print(f"p_opening_start={params.get('p_opening_start')}")
        else:
            print("[INFO] Staged reset is already disabled or not configured.")
            print("Staged reset disabled: True")
    else:
        print("Staged reset disabled: False")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        if get_published_pretrained_checkpoint is None:
            raise RuntimeError(
                "--use_pretrained_checkpoint is not supported by this isaaclab_rl installation. "
                "Please pass --checkpoint with a local checkpoint path instead."
            )
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        checkpoint_arg = os.path.expanduser(args_cli.checkpoint)
        if os.path.isabs(checkpoint_arg) or os.path.exists(checkpoint_arg):
            resume_path = retrieve_file_path(checkpoint_arg)
        else:
            candidate_path = os.path.join(log_root_path, agent_cfg.load_run, checkpoint_arg)
            if os.path.exists(candidate_path):
                resume_path = candidate_path
            else:
                resume_path = retrieve_file_path(checkpoint_arg)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

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
    if agent_cfg.class_name == "DoorBotTeacherRunner":
        env = DoorBotDistillationVecEnvWrapper(
            env, clip_actions=agent_cfg.clip_actions, history_length=agent_cfg.history_length,
            transition_thresholds=agent_cfg.stage_transition_thresholds,
            collect_distillation_rollout=False,
        )
    else:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped
        
    # ---------------------------------------------------------------------
    # Debug/Stats helpers: query current fixed-hook robot ids, optional legacy gripper ids, sensors, and targets.
    # ---------------------------------------------------------------------
    robot = None
    door = None
    finger_joint_ids = None
    finger_body_ids = None  # legacy gripper assets only: (left_body_id, right_body_id)
    hand_body_id = None  # current hook grasp center, falling back to hook/link6
    handle_body_id = None
    q_target_field = None
    left_sensor = None
    right_sensor = None
    try:
        robot = base_env.scene["robot"]
        # door articulation (used for handle pose). If your asset name differs, update here.
        try:
            door = base_env.scene["door"]
        except Exception:
            door = None

        # contact sensors (for the current fixed-hook robot both names default to hook_contact)
        try:
            left_sensor = base_env.scene[args_cli.left_contact_sensor]
        except Exception:
            left_sensor = None
        try:
            right_sensor = base_env.scene[args_cli.right_contact_sensor]
        except Exception:
            right_sensor = None
        if (left_sensor is None or right_sensor is None) and (args_cli.debug_setup or args_cli.print_contact_forces):
            print(
                f"[WARN] Contact sensor(s) not found. primary='{args_cli.left_contact_sensor}' -> {left_sensor is not None}, "
                f"secondary='{args_cli.right_contact_sensor}' -> {right_sensor is not None}."
            )

        # joint name -> id mapping
        joint_names = None
        if hasattr(robot, "data") and hasattr(robot.data, "joint_names"):
            joint_names = list(robot.data.joint_names)
        elif hasattr(robot, "joint_names"):
            joint_names = list(robot.joint_names)

        if joint_names is not None:
            finger_joint_ids = []
            # X5 uses a single driven gripper joint. Prefer it explicitly.
            if "gripper_joint" in joint_names:
                finger_joint_ids = [joint_names.index("gripper_joint")]
            else:
                # Fallback for other robots: collect joints whose names suggest finger/gripper motion.
                for i, jn in enumerate(joint_names):
                    name = jn.lower()
                    if ("finger" in name) or ("gripper" in name):
                        finger_joint_ids.append(i)
            if len(finger_joint_ids) == 0:
                finger_joint_ids = None

        # body name -> id mapping (for wrap alignment stats)
        if hasattr(robot, "data") and hasattr(robot.data, "body_names"):
            bnames_r = list(robot.data.body_names)
            # Current fixed-hook asset uses gripper_grasp_center/gripper_hook. Finger bodies are legacy-only.
            left_candidates = ["left_pad", "left_finger", "link7"]
            right_candidates = ["right_pad", "right_finger", "link8"]
            left_bid = next((bnames_r.index(n) for n in left_candidates if n in bnames_r), None)
            right_bid = next((bnames_r.index(n) for n in right_candidates if n in bnames_r), None)
            if left_bid is not None and right_bid is not None:
                finger_body_ids = (left_bid, right_bid)
            hand_candidates = ["gripper_grasp_center", "gripper_hook", "gripper_tcp", "link6"]
            hand_body_id = next((bnames_r.index(n) for n in hand_candidates if n in bnames_r), None)
        if door is not None and hasattr(door, "data") and hasattr(door.data, "body_names"):
            bnames_d = list(door.data.body_names)
            if "handle_1" in bnames_d:
                handle_body_id = bnames_d.index("handle_1")

        # figure out which field stores joint position targets
        if hasattr(robot, "data"):
            if hasattr(robot.data, "joint_pos_target"):
                q_target_field = "joint_pos_target"
            elif hasattr(robot.data, "joint_pos_targets"):
                q_target_field = "joint_pos_targets"

        # print actuator coverage once (only if requested)
        if args_cli.debug_setup and hasattr(robot, "actuators"):
            print("[DEBUG] Actuator groups:", list(robot.actuators.keys()))
            for name, act in robot.actuators.items():
                jids = getattr(act, "joint_ids", None)
                jnames = getattr(act, "joint_names", None)
                jexpr = getattr(act, "joint_names_expr", None)
                print(f"  - {name}: joint_ids={jids} joint_names={jnames} expr={jexpr}")

            print(f"[DEBUG] Legacy gripper joint ids: {finger_joint_ids}")
            print(f"[DEBUG] Hook/hand body id: {hand_body_id}, legacy finger body ids: {finger_body_ids}, handle_body_id: {handle_body_id}")
            print(f"[DEBUG] Target field: {q_target_field}")

    except Exception as e:
        if args_cli.debug_setup:
            print(f"[DEBUG] Could not query robot joint/actuator info: {e}")
    # ---------------------------------------------------------------------
    # Small math helpers for stats (wxyz quaternions)
    # ---------------------------------------------------------------------
    def _quat_conjugate(q):
        return torch.stack((q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]), dim=-1)

    def _quat_mul(q1, q2):
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack((w, x, y, z), dim=-1)

    def _quat_rotate(q, v):
        qv = torch.cat((torch.zeros_like(v[..., :1]), v), dim=-1)
        return _quat_mul(_quat_mul(q, qv), _quat_conjugate(q))[..., 1:]

    def _filtered_force_norm(sensor):
        """Per-env norm of HANDLE-FILTERED force using force_matrix_w (safe-fail to zeros)."""
        fm = getattr(sensor.data, "force_matrix_w", None)
        if fm is None:
            return torch.zeros((env.num_envs,), device=env.unwrapped.device)
        if fm.ndim == 4:
            vec = fm.sum(dim=2)                        # [N,B,3]
            mag = torch.linalg.norm(vec, dim=-1)       # [N,B]
            return mag.max(dim=1).values               # [N]
        if fm.ndim == 3:
            vec = fm.sum(dim=1)                        # [N,3]
            return torch.linalg.norm(vec, dim=-1)      # [N]
        return torch.zeros((env.num_envs,), device=env.unwrapped.device)

    def _safe_unit(v, eps: float = 1e-8):
        return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)


    def _compute_gripper_width(robot, joint_ids):
        """Return per-env derived gripper width.

        - single driven joint (X5): width = 2*q
        - multiple finger joints:   width = sum(q_i)
        """
        if joint_ids is None or len(joint_ids) == 0:
            return None
        if len(joint_ids) == 1:
            q = robot.data.joint_pos[:, joint_ids[0]]
            return 2.0 * q
        return robot.data.joint_pos[:, joint_ids].sum(dim=-1)

    def _compute_tip_gap(robot, finger_body_ids):
        """Approximate fingertip gap from the selected left/right finger bodies."""
        if finger_body_ids is None:
            return None
        lb, rb = finger_body_ids
        pL = robot.data.body_pos_w[:, lb, :]
        pR = robot.data.body_pos_w[:, rb, :]
        return torch.linalg.norm(pL - pR, dim=-1)


    def _best_signed_axis(v_local: torch.Tensor):
        """Return best matching signed basis axis for a 3D vector (single vector, shape [3])."""
        axes = [
            ("+x", torch.tensor([ 1.0,  0.0,  0.0], device=v_local.device, dtype=v_local.dtype)),
            ("-x", torch.tensor([-1.0,  0.0,  0.0], device=v_local.device, dtype=v_local.dtype)),
            ("+y", torch.tensor([ 0.0,  1.0,  0.0], device=v_local.device, dtype=v_local.dtype)),
            ("-y", torch.tensor([ 0.0, -1.0,  0.0], device=v_local.device, dtype=v_local.dtype)),
            ("+z", torch.tensor([ 0.0,  0.0,  1.0], device=v_local.device, dtype=v_local.dtype)),
            ("-z", torch.tensor([ 0.0,  0.0, -1.0], device=v_local.device, dtype=v_local.dtype)),
        ]
        best_name, best_dot = None, -1.0e9
        for name, axis in axes:
            dot = torch.dot(v_local, axis).item()
            if dot > best_dot:
                best_name, best_dot = name, dot
        return best_name, best_dot

    class SuccessTrajectoryRecorder:
        """Per-env episode recorder that persists successful trajectories."""

        def __init__(
            self,
            base_env,
            output_root: str,
            trajectory_format: str,
            success_threshold: float,
            max_success: int,
            save_failed: bool,
        ):
            import numpy as np

            self.np = np
            self.base_env = base_env
            self.num_envs = int(base_env.num_envs)
            self.device = base_env.device
            self.trajectory_format = trajectory_format
            self.success_threshold = float(success_threshold)
            self.max_success = int(max_success)
            self.save_failed = bool(save_failed)
            self.success_count = 0
            self.total_saved = 0
            self.buffers = [[] for _ in range(self.num_envs)]
            self.episode_ids = [0 for _ in range(self.num_envs)]
            self.warned_missing = set()

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.abspath(os.path.join(output_root, f"{stamp}_success_trajs"))
            os.makedirs(self.output_dir, exist_ok=True)
            self.manifest_path = os.path.join(self.output_dir, "manifest.csv")
            self.manifest_fields = [
                "file",
                "env_id",
                "episode_id",
                "success",
                "episode_length",
                "max_door_open",
                "final_door_open",
                "max_grasp_quality",
                "mean_stage1_grasp_quality",
                "mean_stage2_grasp_quality",
                "mean_stage1_closed_no_contact",
                "mean_stage2_closed_no_contact",
                "W1_idx",
                "W2_idx",
                "W3_idx",
                "W4_idx",
                "W5_idx",
                "W6_idx",
            ]
            with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.manifest_fields).writeheader()

            self.robot = self._scene_asset("robot")
            self.door = self._scene_asset("door")
            self.left_sensor = self._scene_asset(args_cli.left_contact_sensor, warn=False)
            self.right_sensor = self._scene_asset(args_cli.right_contact_sensor, warn=False)
            self.robot_joint_ids = self._joint_ids(
                self.robot,
                [f"link{i}_joint" for i in range(1, 7)],
                "Piper arm joints",
            )
            self.gripper_joint_ids = self._joint_ids(self.robot, ["gripper_joint"], "gripper_joint", required=False)
            self.door_joint_id = self._joint_ids(self.door, ["door_joint"], "door_joint", required=False)
            self.handle_joint_id = self._joint_ids(self.door, ["handle_joint"], "handle_joint", required=False)
            self.link6_body_id = self._body_id(self.robot, "link6", "link6")
            self.grasp_center_body_id = self._body_id(
                self.robot,
                "gripper_grasp_center",
                "gripper_grasp_center",
                required=False,
            )
            self.gripper_hook_body_id = self._body_id(self.robot, "gripper_hook", "gripper_hook", required=False)
            self.handle_body_id = self._body_id(self.door, "handle_1", "handle_1")
            self.left_body_id = self._body_id(self.robot, "link7", "left finger link7", required=False)
            self.right_body_id = self._body_id(self.robot, "link8", "right finger link8", required=False)
            self.arm_action_term = self._action_term("arm_action")
            self.gripper_action_term = self._action_term("gripper_action", warn=False)

            print(f"[TRAJ] Recording success trajectories to: {self.output_dir}")
            print(f"[TRAJ] success_door_open_threshold={self.success_threshold:.3f}, max_success={self.max_success}")

        def _warn_once(self, key: str, message: str):
            if key not in self.warned_missing:
                self.warned_missing.add(key)
                print(f"[TRAJ][WARN] {message}")

        def _scene_asset(self, name: str, warn: bool = True):
            try:
                return self.base_env.scene[name]
            except Exception:
                if warn:
                    self._warn_once(f"scene:{name}", f"Scene asset '{name}' not found; related fields use placeholders.")
                return None

        def _joint_ids(self, asset, names: list[str], label: str, required: bool = True):
            if asset is None or not hasattr(asset, "data") or not hasattr(asset.data, "joint_names"):
                if required:
                    self._warn_once(f"joint:{label}", f"Cannot resolve {label}; joint fields use NaN.")
                return []
            available = list(asset.data.joint_names)
            ids = [available.index(name) for name in names if name in available]
            if required and len(ids) != len(names):
                missing = [name for name in names if name not in available]
                self._warn_once(f"joint:{label}", f"Missing {label}: {missing}; available={available}")
            return ids

        def _body_id(self, asset, name: str, label: str, required: bool = True):
            if asset is None or not hasattr(asset, "data") or not hasattr(asset.data, "body_names"):
                if required:
                    self._warn_once(f"body:{label}", f"Cannot resolve {label}; pose fields use NaN.")
                return None
            available = list(asset.data.body_names)
            if name in available:
                return available.index(name)
            if required:
                self._warn_once(f"body:{label}", f"Missing body {name}; available={available}")
            return None

        def _action_term(self, name: str, warn: bool = True):
            action_manager = getattr(self.base_env, "action_manager", None)
            if action_manager is None:
                if warn:
                    self._warn_once("action_manager", "action_manager not found; low-level action fields use NaN.")
                return None
            if hasattr(action_manager, "_terms") and name in action_manager._terms:
                return action_manager._terms[name]
            if hasattr(action_manager, "get_term"):
                try:
                    return action_manager.get_term(name)
                except Exception:
                    pass
            if warn:
                self._warn_once(f"action:{name}", f"Action term '{name}' not found; related fields use NaN.")
            return None

        def _nan_vec(self, dim: int):
            return [float("nan")] * dim

        def _tensor_vec(self, tensor, env_id: int, dim: int | None = None):
            if tensor is None:
                return self._nan_vec(dim or 1)
            try:
                value = tensor[env_id]
                if value.ndim == 0:
                    return [float(value.item())]
                return [float(x) for x in value.detach().cpu().flatten().tolist()]
            except Exception:
                return self._nan_vec(dim or 1)

        def _tensor_scalar(self, tensor, env_id: int, default=float("nan")):
            if tensor is None:
                return default
            try:
                return float(tensor[env_id].detach().cpu().item())
            except Exception:
                return default

        def _bool_scalar(self, tensor, env_id: int, default=False):
            if tensor is None:
                return bool(default)
            try:
                return bool(tensor[env_id].detach().cpu().item())
            except Exception:
                return bool(default)

        def _action_tensor(self, term, attr: str, dim: int):
            if term is None or not hasattr(term, attr):
                return torch.full((self.num_envs, dim), float("nan"), device=self.device)
            try:
                value = getattr(term, attr)
                if value.ndim == 1:
                    value = value.unsqueeze(-1)
                return value[:, :dim]
            except Exception as e:
                self._warn_once(f"action_attr:{attr}", f"Could not read action term attribute '{attr}': {e}")
                return torch.full((self.num_envs, dim), float("nan"), device=self.device)

        def _contact_force(self):
            if self.left_sensor is None:
                return torch.full((self.num_envs,), float("nan"), device=self.device)
            return _filtered_force_norm(self.left_sensor)

        def _hook_quality_terms(self):
            nan = torch.full((self.num_envs,), float("nan"), device=self.device)
            false = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            defaults = {
                "quality": nan,
                "contact_ok": false,
                "closedness": torch.ones((self.num_envs,), device=self.device),
                "tcp_dist": nan,
                "f_left": nan,
                "f_right": nan,
                "f_min": nan,
                "f_max": nan,
                "closed_no_contact": false,
                "single_finger": false,
            }
            if (
                self.robot is None
                or self.door is None
                or self.handle_body_id is None
                or self.grasp_center_body_id is None
            ):
                return defaults
            try:
                p_grasp = self.robot.data.body_pos_w[:, self.grasp_center_body_id, :]
                q_grasp = self.robot.data.body_quat_w[:, self.grasp_center_body_id, :]
                p_handle = self.door.data.body_pos_w[:, self.handle_body_id, :]
                q_handle = self.door.data.body_quat_w[:, self.handle_body_id, :]
                handle_offset = torch.tensor(
                    (-0.08, 0.04, -0.005),
                    dtype=p_handle.dtype,
                    device=self.device,
                ).view(1, 3).repeat(self.num_envs, 1)
                p_target = p_handle + _quat_rotate(q_handle, handle_offset)
                dist = torch.linalg.norm(p_grasp - p_target, dim=-1)

                # Match the current hook reward geometry: hook approach +Y should face handle -Y.
                hook_approach_hand = torch.tensor((0.0, 1.0, 0.0), dtype=p_grasp.dtype, device=self.device).view(1, 3)
                hook_mouth_hand = torch.tensor((1.0, 0.0, 0.0), dtype=p_grasp.dtype, device=self.device).view(1, 3)
                world_down = torch.tensor((0.0, 0.0, -1.0), dtype=p_grasp.dtype, device=self.device).view(1, 3)
                approach_w = _quat_rotate(q_grasp, hook_approach_hand.repeat(self.num_envs, 1))
                mouth_w = _quat_rotate(q_grasp, hook_mouth_hand.repeat(self.num_envs, 1))
                handle_y_w = _quat_rotate(
                    q_handle,
                    torch.tensor((0.0, 1.0, 0.0), dtype=p_grasp.dtype, device=self.device).view(1, 3).repeat(self.num_envs, 1),
                )
                approach_score = torch.clamp(torch.sum(approach_w * (-handle_y_w), dim=-1), min=0.0, max=1.0)
                mouth_down_score = torch.clamp(torch.sum(mouth_w * world_down, dim=-1), min=0.0, max=1.0)
                align = 0.70 * approach_score + 0.30 * mouth_down_score

                force = self._contact_force()
                contact_ok = torch.isfinite(force) & (force > float(args_cli.stats_contact_threshold))
                near_score = torch.exp(-torch.square(dist / 0.14))
                quality = near_score * align * torch.where(contact_ok, torch.ones_like(align), 0.2 * torch.ones_like(align))
                defaults.update(
                    {
                        "quality": quality,
                        "contact_ok": contact_ok,
                        "tcp_dist": dist,
                        "f_left": force,
                        "f_right": force,
                        "f_min": force,
                        "f_max": force,
                        "closed_no_contact": (~contact_ok),
                    }
                )
                return defaults
            except Exception as e:
                self._warn_once("quality_compute", f"Could not compute hook grasp quality: {e}")
                return defaults

        def _quality_terms(self):
            return self._hook_quality_terms()

        def _batch_state(self, actions, rewards, dones, step_index: int):
            nan1 = torch.full((self.num_envs,), float("nan"), device=self.device)
            nan3 = torch.full((self.num_envs, 3), float("nan"), device=self.device)
            nan4 = torch.full((self.num_envs, 4), float("nan"), device=self.device)
            nan6 = torch.full((self.num_envs, 6), float("nan"), device=self.device)

            robot_q = self.robot.data.joint_pos[:, self.robot_joint_ids] if self.robot is not None and len(self.robot_joint_ids) == 6 else nan6
            robot_dq = self.robot.data.joint_vel[:, self.robot_joint_ids] if self.robot is not None and len(self.robot_joint_ids) == 6 else nan6
            gripper_q = (
                self.robot.data.joint_pos[:, self.gripper_joint_ids]
                if self.robot is not None and len(self.gripper_joint_ids) > 0
                else torch.full((self.num_envs, 1), float("nan"), device=self.device)
            )
            gripper_opening = gripper_q[:, 0] if gripper_q.ndim == 2 and gripper_q.shape[1] > 0 else nan1

            link6_pos = self.robot.data.body_pos_w[:, self.link6_body_id, :] if self.robot is not None and self.link6_body_id is not None else nan3
            link6_quat = self.robot.data.body_quat_w[:, self.link6_body_id, :] if self.robot is not None and self.link6_body_id is not None else nan4
            ee_tcp_pos = (
                self.robot.data.body_pos_w[:, self.grasp_center_body_id, :]
                if self.robot is not None and self.grasp_center_body_id is not None
                else nan3
            )
            ee_tcp_quat = (
                self.robot.data.body_quat_w[:, self.grasp_center_body_id, :]
                if self.robot is not None and self.grasp_center_body_id is not None
                else nan4
            )

            handle_pos = self.door.data.body_pos_w[:, self.handle_body_id, :] if self.door is not None and self.handle_body_id is not None else nan3
            handle_quat = self.door.data.body_quat_w[:, self.handle_body_id, :] if self.door is not None and self.handle_body_id is not None else nan4
            handle_offset = torch.tensor((-0.08, 0.04, -0.005), dtype=handle_pos.dtype, device=self.device).view(1, 3).repeat(self.num_envs, 1)
            handle_target = handle_pos + _quat_rotate(handle_quat, handle_offset) if torch.isfinite(handle_pos).any() else nan3

            door_joint_pos = self.door.data.joint_pos[:, self.door_joint_id[0]] if self.door is not None and len(self.door_joint_id) == 1 else nan1
            door_joint_vel = self.door.data.joint_vel[:, self.door_joint_id[0]] if self.door is not None and len(self.door_joint_id) == 1 else nan1
            handle_joint_pos = self.door.data.joint_pos[:, self.handle_joint_id[0]] if self.door is not None and len(self.handle_joint_id) == 1 else nan1
            handle_joint_vel = self.door.data.joint_vel[:, self.handle_joint_id[0]] if self.door is not None and len(self.handle_joint_id) == 1 else nan1
            door_open = torch.clamp(door_open_sign * (door_joint_pos - 0.0), min=0.0)

            if hasattr(self.base_env, "_door_lock_mode"):
                door_lock_mode = self.base_env._door_lock_mode
                physical_unlocked = door_lock_mode == 2
            elif hasattr(self.base_env, "_door_unlocked"):
                door_lock_mode = torch.full((self.num_envs,), -1, device=self.device)
                physical_unlocked = self.base_env._door_unlocked.to(dtype=torch.bool)
            else:
                door_lock_mode = torch.full((self.num_envs,), -1, device=self.device)
                physical_unlocked = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

            grasp_success = (
                self.base_env._grasp_success_given.to(dtype=torch.bool)
                if hasattr(self.base_env, "_grasp_success_given")
                else torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            )
            stage_id = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)
            stage_id = torch.where(grasp_success & (~physical_unlocked), torch.ones_like(stage_id), stage_id)
            stage_id = torch.where(physical_unlocked, torch.full_like(stage_id, 2), stage_id)

            q_des = self._action_tensor(self.arm_action_term, "q_des", 6)
            applied_delta = self._action_tensor(self.arm_action_term, "applied_delta", 6)
            arm_raw = self._action_tensor(self.arm_action_term, "raw_actions", 6)
            gripper_raw = self._action_tensor(self.gripper_action_term, "raw_actions", 1)
            q_des_error = q_des - robot_q if torch.isfinite(q_des).any() and torch.isfinite(robot_q).any() else nan6

            quality = self._quality_terms()
            finger_mid_dist = nan1
            if self.robot is not None and self.door is not None and self.left_body_id is not None and self.right_body_id is not None and self.handle_body_id is not None:
                p_left = self.robot.data.body_pos_w[:, self.left_body_id, :]
                p_right = self.robot.data.body_pos_w[:, self.right_body_id, :]
                finger_mid = 0.5 * (p_left + p_right)
                finger_mid_dist = torch.linalg.norm(finger_mid - handle_target, dim=-1)

            near_ok = quality["tcp_dist"] <= 0.14
            close_ok = quality["closedness"] > 0.35
            left_force = quality["f_left"]
            right_force = quality["f_right"]

            return {
                "step_index": step_index,
                "reward": rewards,
                "done": dones,
                "robot_joint_pos": robot_q,
                "robot_joint_vel": robot_dq,
                "gripper_joint_pos": gripper_q,
                "gripper_opening": gripper_opening,
                "action_raw_or_policy": actions,
                "arm_action": actions[:, :6] if actions.shape[-1] >= 6 else arm_raw,
                "gripper_action": actions[:, 6:7] if actions.shape[-1] >= 7 else gripper_raw,
                "q_des": q_des,
                "q_des_error": q_des_error,
                "applied_delta": applied_delta,
                "ee_tcp_pos_w": ee_tcp_pos,
                "ee_tcp_quat_w": ee_tcp_quat,
                "link6_pos_w": link6_pos,
                "link6_quat_w": link6_quat,
                "handle_pos_w": handle_pos,
                "handle_quat_w": handle_quat,
                "handle_grasp_target_w": handle_target,
                "handle_joint_pos": handle_joint_pos,
                "handle_joint_vel": handle_joint_vel,
                "door_joint_pos": door_joint_pos,
                "door_joint_vel": door_joint_vel,
                "door_open": door_open,
                "grasp_success_given": grasp_success,
                "physical_unlocked": physical_unlocked,
                "door_lock_mode": door_lock_mode,
                "stage_id": stage_id,
                "grasp_quality": quality["quality"],
                "grasp_quality_gate": quality["contact_ok"] & close_ok & near_ok,
                "tcp_to_grasp_dist": quality["tcp_dist"],
                "finger_mid_to_grasp_dist": finger_mid_dist,
                "contact_ok": quality["contact_ok"],
                "close_ok": close_ok,
                "near_ok": near_ok,
                "closed_no_contact": quality["closed_no_contact"],
                "single_finger": quality["single_finger"],
                "left_contact_force": left_force,
                "right_contact_force": right_force,
                "f_min": quality["f_min"],
                "f_max": quality["f_max"],
            }

        def _frame_from_batch(self, batch: dict, env_id: int, episode_step: int):
            frame = {
                "step_index": int(batch["step_index"]),
                "episode_step": int(episode_step),
                "env_id": int(env_id),
                "sim_time": float(batch["step_index"] * self.base_env.step_dt),
            }
            scalar_keys = {
                "reward",
                "done",
                "gripper_opening",
                "handle_joint_pos",
                "handle_joint_vel",
                "door_joint_pos",
                "door_joint_vel",
                "door_open",
                "grasp_success_given",
                "physical_unlocked",
                "door_lock_mode",
                "stage_id",
                "grasp_quality",
                "grasp_quality_gate",
                "tcp_to_grasp_dist",
                "finger_mid_to_grasp_dist",
                "contact_ok",
                "close_ok",
                "near_ok",
                "closed_no_contact",
                "single_finger",
                "left_contact_force",
                "right_contact_force",
                "f_min",
                "f_max",
            }
            for key, value in batch.items():
                if key in ("step_index",):
                    continue
                if key in scalar_keys:
                    if torch.is_tensor(value) and value.dtype == torch.bool:
                        frame[key] = self._bool_scalar(value, env_id)
                    else:
                        frame[key] = self._tensor_scalar(value, env_id)
                else:
                    dim = 1
                    if key in ("robot_joint_pos", "robot_joint_vel", "arm_action", "q_des", "q_des_error", "applied_delta"):
                        dim = 6
                    elif key in ("ee_tcp_pos_w", "link6_pos_w", "handle_pos_w", "handle_grasp_target_w"):
                        dim = 3
                    elif key in ("ee_tcp_quat_w", "link6_quat_w", "handle_quat_w"):
                        dim = 4
                    frame[key] = self._tensor_vec(value, env_id, dim=dim)
            return frame

        def step(self, actions, rewards, dones, step_index: int):
            rewards = rewards if torch.is_tensor(rewards) else torch.as_tensor(rewards, device=self.device)
            dones = dones.to(dtype=torch.bool) if torch.is_tensor(dones) else torch.as_tensor(dones, dtype=torch.bool, device=self.device)
            batch = self._batch_state(actions.detach(), rewards.detach(), dones.detach(), step_index)
            for env_id in range(self.num_envs):
                frame = self._frame_from_batch(batch, env_id, len(self.buffers[env_id]))
                self.buffers[env_id].append(frame)
                if bool(dones[env_id].item()):
                    self._finish_episode(env_id)
            return self.success_count >= self.max_success

        def _episode_arrays(self, episode: list[dict]):
            arrays = {}
            if not episode:
                return arrays
            for key in episode[0].keys():
                values = [frame[key] for frame in episode]
                arrays[key] = self.np.asarray(values)
            return arrays

        def _first_true(self, values):
            idx = self.np.nonzero(self.np.asarray(values, dtype=bool))[0]
            return int(idx[0]) if idx.size > 0 else -1

        def _nearest_idx(self, mask, values, target):
            mask = self.np.asarray(mask, dtype=bool)
            values = self.np.asarray(values, dtype=float)
            valid = self.np.nonzero(mask & self.np.isfinite(values))[0]
            if valid.size == 0:
                return -1
            return int(valid[self.np.argmin(self.np.abs(values[valid] - float(target)))])

        def _argmin_idx(self, mask, values):
            mask = self.np.asarray(mask, dtype=bool)
            values = self.np.asarray(values, dtype=float)
            valid = self.np.nonzero(mask & self.np.isfinite(values))[0]
            if valid.size == 0:
                return -1
            return int(valid[self.np.argmin(values[valid])])

        def _argmax_idx(self, mask, values):
            mask = self.np.asarray(mask, dtype=bool)
            values = self.np.asarray(values, dtype=float)
            valid = self.np.nonzero(mask & self.np.isfinite(values))[0]
            if valid.size == 0:
                return -1
            return int(valid[self.np.argmax(values[valid])])

        def _waypoints(self, arrays: dict):
            stage = arrays.get("stage_id", self.np.zeros((0,), dtype=int))
            quality = arrays.get("grasp_quality", self.np.asarray([]))
            tcp_dist = arrays.get("tcp_to_grasp_dist", self.np.asarray([]))
            grasp = arrays.get("grasp_success_given", self.np.asarray([]))
            unlocked = arrays.get("physical_unlocked", self.np.asarray([]))
            handle = arrays.get("handle_joint_pos", self.np.asarray([]))
            door_open = arrays.get("door_open", self.np.asarray([]))

            w1 = self._argmin_idx(stage == 0, tcp_dist)
            first_grasp = self._first_true(grasp)
            if first_grasp >= 0:
                mask = self.np.zeros_like(grasp, dtype=bool)
                mask[first_grasp : min(len(mask), first_grasp + 11)] = True
                w2 = self._argmax_idx(mask, quality)
            else:
                w2 = -1
            w3 = self._nearest_idx(stage == 1, handle, -0.20)
            w4 = self._first_true(unlocked)
            w5 = self._nearest_idx(stage == 2, door_open, 0.15)
            success_cross = self.np.asarray(door_open, dtype=float) >= self.success_threshold
            w6 = self._first_true(success_cross)
            return {"W1": w1, "W2": w2, "W3": w3, "W4": w4, "W5": w5, "W6": w6}

        def _nanmean(self, values):
            values = self.np.asarray(values, dtype=float)
            if values.size == 0 or not self.np.isfinite(values).any():
                return float("nan")
            return float(self.np.nanmean(values))

        def _metrics(self, arrays: dict):
            door_open = self.np.asarray(arrays.get("door_open", []), dtype=float)
            quality = self.np.asarray(arrays.get("grasp_quality", []), dtype=float)
            stage = self.np.asarray(arrays.get("stage_id", []), dtype=int)
            closed_no_contact = self.np.asarray(arrays.get("closed_no_contact", []), dtype=float)
            max_door = float(self.np.nanmax(door_open)) if door_open.size else float("nan")
            final_door = float(door_open[-1]) if door_open.size else float("nan")
            return {
                "max_door_open": max_door,
                "final_door_open": final_door,
                "max_grasp_quality": float(self.np.nanmax(quality)) if quality.size and self.np.isfinite(quality).any() else float("nan"),
                "mean_stage1_grasp_quality": self._nanmean(quality[stage == 1]) if quality.size and stage.size else float("nan"),
                "mean_stage2_grasp_quality": self._nanmean(quality[stage == 2]) if quality.size and stage.size else float("nan"),
                "mean_stage1_closed_no_contact": self._nanmean(closed_no_contact[stage == 1]) if closed_no_contact.size and stage.size else float("nan"),
                "mean_stage2_closed_no_contact": self._nanmean(closed_no_contact[stage == 2]) if closed_no_contact.size and stage.size else float("nan"),
            }

        def _expanded_csv_row(self, frame: dict):
            row = {}
            for key, value in frame.items():
                arr = self.np.asarray(value)
                if arr.ndim == 0:
                    row[key] = value
                else:
                    flat = arr.flatten()
                    for idx, item in enumerate(flat):
                        row[f"{key}_{idx}"] = item
            return row

        def _save_csv(self, path: str, episode: list[dict]):
            rows = [self._expanded_csv_row(frame) for frame in episode]
            fieldnames = sorted({key for row in rows for key in row.keys()})
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        def _save_episode(self, env_id: int, episode_id: int, episode: list[dict], success: bool, metrics: dict, waypoints: dict):
            prefix = "success" if success else "failed"
            stem = f"{prefix}_env{env_id}_episode{episode_id}_door{metrics['max_door_open']:.3f}"
            arrays = self._episode_arrays(episode)
            arrays["waypoint_indices"] = self.np.asarray(
                [waypoints["W1"], waypoints["W2"], waypoints["W3"], waypoints["W4"], waypoints["W5"], waypoints["W6"]],
                dtype=self.np.int64,
            )
            arrays["waypoint_names"] = self.np.asarray(["W1_pregrasp", "W2_grasp", "W3_press", "W4_unlock", "W5_open_mid", "W6_success"])

            npz_path = os.path.join(self.output_dir, f"{stem}.npz")
            csv_path = os.path.join(self.output_dir, f"{stem}_frames.csv")
            main_file = npz_path
            if self.trajectory_format in ("npz", "both"):
                self.np.savez_compressed(npz_path, **arrays)
            if self.trajectory_format in ("csv", "both"):
                self._save_csv(csv_path, episode)
                if self.trajectory_format == "csv":
                    main_file = csv_path

            row = {
                "file": os.path.basename(main_file),
                "env_id": env_id,
                "episode_id": episode_id,
                "success": success,
                "episode_length": len(episode),
                **metrics,
                "W1_idx": waypoints["W1"],
                "W2_idx": waypoints["W2"],
                "W3_idx": waypoints["W3"],
                "W4_idx": waypoints["W4"],
                "W5_idx": waypoints["W5"],
                "W6_idx": waypoints["W6"],
            }
            with open(self.manifest_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.manifest_fields).writerow(row)

            self.total_saved += 1
            if success:
                self.success_count += 1
                print("Saved success trajectory:")
            else:
                print("Saved failed trajectory:")
            print(f"    {main_file}")
            print(f"    len={len(episode)}")
            print(f"    max_door_open={metrics['max_door_open']:.3f}")
            print(f"    mean_stage1_grasp_quality={metrics['mean_stage1_grasp_quality']:.4f}")
            print(f"    mean_stage2_grasp_quality={metrics['mean_stage2_grasp_quality']:.4f}")
            print(f"    waypoint_indices={waypoints}")

        def _finish_episode(self, env_id: int):
            episode = self.buffers[env_id]
            episode_id = self.episode_ids[env_id]
            self.episode_ids[env_id] += 1
            self.buffers[env_id] = []
            if not episode:
                return
            arrays = self._episode_arrays(episode)
            metrics = self._metrics(arrays)
            success = bool(metrics["max_door_open"] >= self.success_threshold)
            saved = False
            if success or self.save_failed:
                waypoints = self._waypoints(arrays)
                self._save_episode(env_id, episode_id, episode, success, metrics, waypoints)
                saved = True
            print(
                f"Episode finished env={env_id}, success={success}, len={len(episode)}, "
                f"max_door_open={metrics['max_door_open']:.3f}, saved={self.success_count}/{self.max_success}"
            )
            if saved and success:
                pass
    
        # ---------------------------------------------------------------------
    # GUI HUD: Pull uses a compact door-state-only panel. Push retains the
    # detailed manipulation diagnostics below.
    # ---------------------------------------------------------------------
    def _start_pull_door_hud(base_env, env_index: int = 0, update_hz: float = 20.0):
        try:
            import omni.ui as ui
            import omni.kit.app
        except Exception:
            return None, None

        door_asset = base_env.scene["door"]
        joint_names = list(door_asset.data.joint_names)
        handle_jid = joint_names.index("handle_joint")
        door_jid = joint_names.index("door_joint")
        win = ui.Window("Pull Door State", width=430, height=190)
        with win.frame:
            with ui.VStack(spacing=8):
                ui.Label(f"Pull Door env{env_index}")
                lbl_handle = ui.Label("handle_joint: --")
                lbl_hinge = ui.Label("door_joint / hinge: --")
                lbl_unlock = ui.Label("door unlock: --")

        interval = 1.0 / max(float(update_hz), 1.0e-3)
        last_update = 0.0

        def _on_update(_evt):
            nonlocal last_update
            now = time.time()
            if now - last_update < interval:
                return
            last_update = now
            handle_pos = float(door_asset.data.joint_pos[env_index, handle_jid])
            handle_vel = float(door_asset.data.joint_vel[env_index, handle_jid])
            door_pos = float(door_asset.data.joint_pos[env_index, door_jid])
            door_vel = float(door_asset.data.joint_vel[env_index, door_jid])
            signed_open = max(0.0, float(door_open_sign) * door_pos)
            lock_mode_tensor = getattr(base_env, "_door_lock_mode", None)
            unlocked_tensor = getattr(base_env, "_door_unlocked", None)
            lock_mode = int(lock_mode_tensor[env_index]) if lock_mode_tensor is not None else -1
            if unlocked_tensor is not None:
                unlocked = bool(unlocked_tensor[env_index])
            else:
                unlocked = lock_mode == 2
            lbl_handle.text = f"handle_joint: pos={handle_pos:+.4f} rad  vel={handle_vel:+.4f} rad/s"
            lbl_hinge.text = (
                f"door_joint / hinge: pos={door_pos:+.4f} rad  vel={door_vel:+.4f} rad/s  "
                f"open={signed_open:.4f} rad"
            )
            lbl_unlock.text = f"door unlock: {unlocked}  lock_mode={lock_mode}"

        stream = omni.kit.app.get_app().get_update_event_stream()
        return win, stream.create_subscription_to_pop(_on_update)

    def _draw_pull_zones(base_env, draw, env_index: int = 0):
        """Draw live/frozen Pull Z1/Z2/Z3 regions for one environment."""
        if draw is None:
            return
        door_asset = base_env.scene["door"]
        dtype = door_asset.data.root_pos_w.dtype
        device = door_asset.data.root_pos_w.device
        started_buffer = getattr(base_env, "_pull_traverse_started", None)
        started = bool(started_buffer[env_index]) if started_buffer is not None else False
        frozen_ready = all(
            hasattr(base_env, name)
            for name in ("_pull_traverse_frozen_hinge_xy", "_pull_traverse_frozen_d0", "_pull_traverse_frozen_d")
        )
        if started and frozen_ready:
            hinge = base_env._pull_traverse_frozen_hinge_xy[env_index]
            d0 = base_env._pull_traverse_frozen_d0[env_index]
            d = base_env._pull_traverse_frozen_d[env_index]
        else:
            hinge = door_asset.data.root_pos_w[env_index, :2]
            local_x = torch.tensor((1.0, 0.0, 0.0), device=device, dtype=dtype)
            d0_unit = _safe_unit(_quat_rotate(door_asset.data.root_quat_w[env_index : env_index + 1], local_x[None])[0, :2])
            d0 = 1.2 * d0_unit
            joint_names = list(door_asset.data.joint_names)
            door_jid = joint_names.index("door_joint")
            angle = -float(door_asset.data.joint_pos[env_index, door_jid])
            c, s = torch.cos(torch.tensor(angle, device=device, dtype=dtype)), torch.sin(torch.tensor(angle, device=device, dtype=dtype))
            d = torch.stack((c * d0[0] - s * d0[1], s * d0[0] + c * d0[1]))

        radii = torch.arange(0.08, 3.001, 0.08, device=device, dtype=dtype)
        angles = torch.linspace(-torch.pi, torch.pi, 181, device=device, dtype=dtype)[:-1]
        rr, aa = torch.meshgrid(radii, angles, indexing="ij")
        rel = torch.stack((rr * torch.cos(aa), rr * torch.sin(aa)), dim=-1).reshape(-1, 2)
        chord = d - d0
        chord_norm = torch.clamp(torch.linalg.norm(chord), min=1.0e-6)
        hinge_side = chord[0] * (-d0[1]) - chord[1] * (-d0[0])
        point_from_end = rel - d0
        point_side = chord[0] * point_from_end[:, 1] - chord[1] * point_from_end[:, 0]
        chord_inside = point_side * torch.where(hinge_side >= 0, 1.0, -1.0) / chord_norm >= 0
        tri_den = d0[0] * d[1] - d0[1] * d[0]
        safe_den = tri_den if abs(float(tri_den)) > 1.0e-6 else torch.ones_like(tri_den)
        a0 = (rel[:, 0] * d[1] - rel[:, 1] * d[0]) / safe_den
        ad = (d0[0] * rel[:, 1] - d0[1] * rel[:, 0]) / safe_den
        in_triangle = (abs(float(tri_den)) > 1.0e-6) & (a0 >= 0) & (ad >= 0) & ((a0 + ad) <= 1)
        opening_sign = 1.0 if float(tri_den) >= 0.0 else -1.0
        opening_side_d = opening_sign * (d[0] * rel[:, 1] - d[1] * rel[:, 0]) >= 0
        opening_half = opening_sign * (d0[0] * rel[:, 1] - d0[1] * rel[:, 0]) >= 0
        z1 = opening_half & (~chord_inside)
        z3 = opening_half & chord_inside & opening_side_d & (~in_triangle)
        z2 = (~z1) & (~z3)
        world = rel + hinge
        world_cpu = world.detach().cpu()
        z1_cpu = z1.detach().cpu()
        z3_cpu = z3.detach().cpu()
        points = [(float(p[0]), float(p[1]), 0.025) for p in world_cpu]
        colors = [
            (0.20, 0.85, 0.25, 0.35) if bool(z1_cpu[i]) else
            (0.15, 0.55, 1.00, 0.32) if bool(z3_cpu[i]) else
            (1.00, 0.72, 0.12, 0.35)
            for i in range(world_cpu.shape[0])
        ]
        draw.clear_points()
        draw.draw_points(points, colors, [4.0] * len(points))

    def _start_handle_joint_hud(
        base_env,
        joint_name: str = "handle_joint",
        env_index: int = 0,
        update_hz: float = 20.0,
        handle_start_pos: float = 0.0,
        handle_threshold: float = -0.2,
    ):
        """GUI HUD for current fixed-hook Stage-0 reward and grasp gates."""
        try:
            import omni.ui as ui
            import omni.kit.app
        except Exception:
            return None, None

        door = base_env.scene["door"]
        robot = base_env.scene["robot"]

        # Use the exact geometry and threshold configured for the traverse
        # termination so the HUD stays synchronized with task success.
        traverse_cfg = getattr(getattr(base_env.cfg, "terminations", None), "base_traverse_success", None)
        traverse_params = getattr(traverse_cfg, "params", {}) or {}
        doorway_center_xy = traverse_params.get("doorway_center_xy", (0.0, 0.0))
        doorway_forward_axis = traverse_params.get("doorway_forward_axis", (1.0, 0.0))
        traverse_pass_distance = float(traverse_params.get("pass_distance", 0.5))

        # --- resolve handle_joint id ---
        jnames = list(door.data.joint_names)
        if joint_name in jnames:
            handle_jid = jnames.index(joint_name)
            resolved_handle = joint_name
        else:
            cand = [i for i, n in enumerate(jnames) if "handle" in n.lower()]
            handle_jid = cand[0] if cand else 0
            resolved_handle = jnames[handle_jid]

        # --- resolve door_joint id ---
        door_jid = jnames.index("door_joint") if "door_joint" in jnames else None

        win = ui.Window("Door Play HUD", width=720, height=680)
        with win.frame:
            with ui.VStack(spacing=6):
                ui.Label(f"======== Stage 0 grasp evaluation (env{env_index}) ========")
                lbl_stage = ui.Label("stage: --")
                lbl_base_stance = ui.Label("base -> pick stance XY: --")
                lbl_ee_handle = ui.Label("EE -> handle target: --")
                lbl_align = ui.Label("hook alignment: --")
                lbl_height = ui.Label("EE z - target z: --")
                lbl_quality = ui.Label("grasp quality: --")
                lbl_gates = ui.Label("gates [near, align, contact, above, all]: --")
                lbl_contact_force = ui.Label("hook contact force: --")
                ui.Separator()
                ui.Label("======== Arm / unlock diagnosis ========")
                lbl_unlock_sync = ui.Label("unlock transition: --")
                lbl_raw_action = ui.Label("raw arm action: --")
                lbl_applied_delta = ui.Label("applied arm delta: --")
                lbl_q = ui.Label("measured q: --")
                lbl_q_des = ui.Label("q_des: --")
                lbl_q_default_error = ui.Label("q_des - default: --")
                lbl_q_error = ui.Label("q_des - q: --")
                lbl_joint_vel = ui.Label("joint velocity: --")
                lbl_applied_torque = ui.Label("applied torque: --")
                lbl_arm_tracking = ui.Label("tracking summary: --")
                lbl_sync_snapshot = ui.Label("unlock transition snapshot: --")
                ui.Separator()
                lbl_pos = ui.Label("handle pos: --")
                lbl_vel = ui.Label("handle vel: --")
                lbl_prog = ui.Label("handle progress: --")
                lbl_unlock = ui.Label("door_unlocked: --")
                lbl_door = ui.Label("door_joint: --")
                lbl_traverse_pass = ui.Label("traverse pass: --")

        dt = 1.0 / max(1e-3, float(update_hz))
        last_t = 0.0
        denom = float(handle_threshold - handle_start_pos) if abs(handle_threshold - handle_start_pos) > 1e-9 else None
        action_term = base_env.action_manager._terms.get("high_level_action")
        previous_unlocked = False
        previous_q_des = None
        sync_snapshot = "waiting for unlock"

        def _fmt6(value, precision=3):
            """Format an env-0 six-joint tensor compactly for the HUD."""
            vals = value.detach().flatten().tolist()
            return "[" + ", ".join(f"{float(v):+.{precision}f}" for v in vals) + "]"

        def _on_update(_evt):
            nonlocal last_t, previous_unlocked, previous_q_des, sync_snapshot
            now = time.time()
            if now - last_t < dt:
                return
            last_t = now

            # handle joint state
            pos = float(door.data.joint_pos[env_index, handle_jid].item())
            vel = float(door.data.joint_vel[env_index, handle_jid].item())
            lbl_pos.text = f"handle pos: {pos:.6f} rad"
            lbl_vel.text = f"handle vel: {vel:.6f} rad/s"

            # progress
            if denom is None:
                prog = 0.0
            else:
                prog = (pos - float(handle_start_pos)) / denom
                prog = 0.0 if prog < 0.0 else (1.0 if prog > 1.0 else prog)
            lbl_prog.text = f"handle progress: {100.0 * prog:.1f}%   (0→{handle_threshold})"

            # unlocked
            unlocked_flag = None
            if hasattr(base_env, "_door_unlocked"):
                try:
                    unlocked_flag = bool(base_env._door_unlocked[env_index].item())
                except Exception:
                    unlocked_flag = None

            by_handle = (pos < float(handle_threshold)) if float(handle_threshold) < float(handle_start_pos) else (pos > float(handle_threshold))
            if unlocked_flag is None:
                lbl_unlock.text = f"door_unlocked: N/A   | by_handle: {by_handle}"
            else:
                lbl_unlock.text = f"door_unlocked: {unlocked_flag}   | by_handle: {by_handle}"

            # door joint
            if door_jid is None:
                lbl_door.text = "door_joint: N/A"
            else:
                dpos = float(door.data.joint_pos[env_index, door_jid].item())
                dvel = float(door.data.joint_vel[env_index, door_jid].item())
                lbl_door.text = f"door_joint: pos={dpos:.4f} vel={dvel:.4f}"

            # Same signed-forward projection as mdp._doorway_geometry:
            # positive means the base center has passed the doorway center in
            # the configured forward direction; negative means it is before it.
            q_door = door.data.root_quat_w[env_index]
            p_door = door.data.root_pos_w[env_index]
            center_d = torch.tensor(
                (doorway_center_xy[0], doorway_center_xy[1], 0.0),
                device=p_door.device,
                dtype=p_door.dtype,
            )
            forward_d = torch.tensor(
                (doorway_forward_axis[0], doorway_forward_axis[1], 0.0),
                device=p_door.device,
                dtype=p_door.dtype,
            )
            center_w = p_door + _quat_rotate(q_door[None], center_d[None])[0]
            forward_w = _quat_rotate(q_door[None], forward_d[None])[0, :2]
            forward_w = forward_w / torch.clamp(torch.linalg.norm(forward_w), min=1.0e-6)
            signed_forward = torch.dot(robot.data.root_pos_w[env_index, :2] - center_w[:2], forward_w)
            passed = float(signed_forward.item())
            remaining = max(traverse_pass_distance - passed, 0.0)
            lbl_traverse_pass.text = (
                f"traverse pass: passed={passed:+.3f} m  "
                f"target={traverse_pass_distance:.3f} m  remaining={remaining:.3f} m"
            )

            # Current Stage-0 geometry and grasp-quality gates. Constants are
            # intentionally identical to door_env_env_cfg.py.
            try:
                if hand_body_id is None or handle_body_id is None:
                    raise RuntimeError("gripper_grasp_center or handle_1 body was not resolved")
                p_ee = robot.data.body_pos_w[env_index, hand_body_id]
                q_ee = robot.data.body_quat_w[env_index, hand_body_id]
                p_handle = door.data.body_pos_w[env_index, handle_body_id]
                q_handle = door.data.body_quat_w[env_index, handle_body_id]
                offset_h = torch.tensor((-0.08, 0.04, -0.005), device=p_handle.device, dtype=p_handle.dtype)
                target = p_handle + _quat_rotate(q_handle[None], offset_h[None])[0]

                stance = target + torch.tensor((-0.3, 0.3, -1.0), device=target.device, dtype=target.dtype)
                base_dist = torch.linalg.norm(robot.data.root_pos_w[env_index, :2] - stance[:2])
                ee_dist = torch.linalg.norm(p_ee - target)
                z_delta = p_ee[2] - target[2]

                hook_app_w = _safe_unit(_quat_rotate(q_ee[None], torch.tensor((0.0, 1.0, 0.0), device=q_ee.device, dtype=q_ee.dtype)[None]))[0]
                handle_app_w = _safe_unit(_quat_rotate(q_handle[None], torch.tensor((0.0, 1.0, 0.0), device=q_handle.device, dtype=q_handle.dtype)[None]))[0]
                approach_align = torch.clamp(-torch.dot(hook_app_w, handle_app_w), 0.0, 1.0)
                mouth_w = _safe_unit(_quat_rotate(q_ee[None], torch.tensor((1.0, 0.0, 0.0), device=q_ee.device, dtype=q_ee.dtype)[None]))[0]
                mouth_align = torch.clamp(torch.dot(mouth_w, torch.tensor((0.0, 0.0, -1.0), device=q_ee.device, dtype=q_ee.dtype)), 0.0, 1.0)
                align = 0.10 * approach_align + 0.90 * mouth_align

                force = _filtered_force_norm(left_sensor)[env_index] if left_sensor is not None else torch.zeros((), device=p_ee.device)
                # Stage 0 -> 1 grasp-success gate (quality shaping above still
                # intentionally uses its wider 0.10 m distance scale).
                near_ok = ee_dist < 0.05
                align_ok = align >= 0.30
                contact_ok = force > 0.25
                above_ok = z_delta >= 0.0
                all_ok = near_ok & align_ok & contact_ok & above_ok
                near_score = torch.exp(-torch.square(ee_dist / 0.10))
                contact_score = torch.clamp(force / 0.25, 0.0, 1.0)
                above_score = torch.where(
                    z_delta >= 0.0,
                    torch.exp(-torch.square(z_delta / 0.10)),
                    torch.full_like(z_delta, 0.02),
                )
                stable_above_ok = (z_delta >= 0.0) & (z_delta <= 0.03)
                quality = near_score * align * contact_score * above_score

                grasped = bool(getattr(base_env, "_grasp_success_given", torch.zeros(base_env.num_envs, device=p_ee.device, dtype=torch.bool))[env_index].item())
                lock_mode = int(getattr(base_env, "_door_lock_mode", torch.zeros(base_env.num_envs, device=p_ee.device, dtype=torch.long))[env_index].item())
                unlocked = lock_mode == 2
                newly_unlocked = unlocked and not previous_unlocked
                stage = 2 if unlocked else (1 if grasped else 0)
                lbl_stage.text = f"stage: {stage}   grasp_latched={grasped}"

                if action_term is not None and hasattr(action_term, "q_des") and hasattr(action_term, "applied_delta"):
                    q = robot.data.joint_pos[env_index, action_term.joint_ids]
                    dq = robot.data.joint_vel[env_index, action_term.joint_ids]
                    q_des = action_term.q_des[env_index]
                    applied_delta = action_term.applied_delta[env_index]
                    raw_action = action_term.raw_actions[env_index, 5:]
                    q_default = robot.data.default_joint_pos[env_index, action_term.joint_ids]
                    q_default_error = q_des - q_default
                    q_error = q_des - q
                    torque_buffer = getattr(robot.data, "applied_torque", None)
                    if torque_buffer is not None:
                        applied_torque = torque_buffer[env_index, action_term.joint_ids]
                        torque_source = "applied"
                    else:
                        applied_torque = action_term.tau_cmd[env_index]
                        torque_source = "commanded"
                    error_max = float(q_error.abs().max())
                    error_rms = float(torch.sqrt(torch.mean(q_error.square())))
                    delta_max = float(applied_delta.abs().max())
                    lbl_raw_action.text = f"raw arm action:       {_fmt6(raw_action)}"
                    lbl_applied_delta.text = f"applied arm delta:    {_fmt6(applied_delta, 4)} rad"
                    lbl_q.text = f"measured q:           {_fmt6(q)} rad"
                    lbl_q_des.text = f"q_des:                {_fmt6(q_des)} rad"
                    lbl_q_default_error.text = f"q_des - default:      {_fmt6(q_default_error)} rad"
                    lbl_q_error.text = f"q_des - q:            {_fmt6(q_error)} rad"
                    lbl_joint_vel.text = f"joint velocity:       {_fmt6(dq)} rad/s"
                    lbl_applied_torque.text = f"{torque_source} torque:     {_fmt6(applied_torque)} Nm"
                    lbl_arm_tracking.text = (
                        f"tracking summary: qerr max/rms={error_max:.4f}/{error_rms:.4f} rad  "
                        f"delta|max|={delta_max:.4f} rad"
                    )
                    if newly_unlocked:
                        before_error = float((previous_q_des - q).abs().max()) if previous_q_des is not None else float("nan")
                        target_jump = float((q_des - previous_q_des).abs().max()) if previous_q_des is not None else float("nan")
                        sync_snapshot = (
                            f"NEW: qerr before≈{before_error:.4f} -> after={error_max:.4f}, "
                            f"q_des jump|max|={target_jump:.4f}, delta|max|={delta_max:.4f}"
                        )
                    previous_q_des = q_des.detach().clone()
                else:
                    lbl_arm_tracking.text = "tracking summary: N/A"

                sync_enabled = bool(getattr(getattr(action_term, "cfg", None), "sync_arm_target_on_unlock", False))
                lbl_unlock_sync.text = (
                    f"unlock transition: mode={lock_mode} newly_unlocked={newly_unlocked} "
                    f"q_des_sync_enabled={sync_enabled}"
                )
                lbl_sync_snapshot.text = f"unlock transition snapshot: {sync_snapshot}"
                previous_unlocked = unlocked
                lbl_base_stance.text = f"base -> pick stance XY: {float(base_dist):.4f} m"
                lbl_ee_handle.text = f"EE -> handle target: {float(ee_dist):.4f} m"
                lbl_align.text = f"hook alignment: {100.0 * float(align):.1f}%  (approach={100.0 * float(approach_align):.1f}%, mouth={100.0 * float(mouth_align):.1f}%)"
                lbl_height.text = (
                    f"EE z - target z: {float(z_delta):+.4f} m  "
                    f"above_score={float(above_score):.3f} stable_z_gate={bool(stable_above_ok)}"
                )
                lbl_quality.text = f"grasp quality: {100.0 * float(quality):.1f}%"
                lbl_gates.text = (
                    "gates [near, align, contact, above, all]: "
                    f"[{bool(near_ok)}, {bool(align_ok)}, {bool(contact_ok)}, {bool(above_ok)}, {bool(all_ok)}]"
                )
                lbl_contact_force.text = f"hook contact force: {float(force):.4f} N  (>0.25)"
            except Exception as exc:
                lbl_stage.text = "stage: ERR"
                lbl_base_stance.text = "base -> pick stance XY: ERR"
                lbl_ee_handle.text = "EE -> handle target: ERR"
                lbl_align.text = "hook alignment: ERR"
                lbl_height.text = "EE z - target z: ERR"
                lbl_quality.text = "grasp quality: ERR"
                lbl_gates.text = f"gates: ERR ({exc})"
                lbl_contact_force.text = "hook contact force: ERR"

        stream = omni.kit.app.get_app().get_update_event_stream()
        sub = stream.create_subscription_to_pop(_on_update)
        return win, sub


    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    _validate_checkpoint_door_metadata(resume_path, args_cli.task, door_open_sign)
    # load previously trained model
    if agent_cfg.class_name == "DoorBotTeacherRunner":
        runner = DoorBotTeacherRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print("Deterministic policy: True")

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if agent_cfg.class_name == "DoorBotTeacherRunner":
        # The Teacher graph takes 57-D clean obs and 17-D privileged state;
        # the generic exporter incorrectly traces it as one 73-D actor input.
        export_doorbot_teacher(policy_nn, export_model_dir)
    else:
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # forced gripper debug schedule
    force_steps = max(1, int(args_cli.force_gripper_seconds / max(1e-9, dt))) if args_cli.force_gripper else 0

    # reset environment
    obs = env.get_observations()
    trajectory_recorder = None
    if args_cli.record_success_trajectories:
        trajectory_recorder = SuccessTrajectoryRecorder(
            base_env=env.unwrapped,
            output_root=args_cli.trajectory_output_dir,
            trajectory_format=args_cli.trajectory_format,
            success_threshold=args_cli.success_door_open_threshold,
            max_success=args_cli.max_success_trajectories,
            save_failed=args_cli.save_failed_trajectories,
        )
    # ---------------------------------------------------------------------
    # Start HUD (GUI only)
    # ---------------------------------------------------------------------
    _hud_win, _hud_sub = None, None
    try:
        if not args_cli.headless:
            if is_pull_task:
                _hud_win, _hud_sub = _start_pull_door_hud(
                    base_env=env.unwrapped,
                    env_index=0,
                    update_hz=20.0,
                )
            else:
                _hud_win, _hud_sub = _start_handle_joint_hud(
                    base_env=env.unwrapped,
                    joint_name="handle_joint",
                    env_index=0,
                    update_hz=20.0,
                    handle_start_pos=0.0,
                    handle_threshold=-0.3,
                )
    except Exception:
        pass
    timestep = 0
    step_count = 0
    # Event-triggered terminal diagnostics. This stays silent during normal
    # play and prints only a short window around env-0 physical unlock.
    unlock_debug_pre = deque(maxlen=6)
    unlock_debug_post_remaining = 0
    unlock_debug_prev = False
    unlock_debug_term = base_env.action_manager._terms.get("high_level_action")
    unlock_debug_handle_jid = None
    unlock_debug_door_jid = None
    if door is not None:
        unlock_debug_joint_names = list(door.data.joint_names)
        if "handle_joint" in unlock_debug_joint_names:
            unlock_debug_handle_jid = unlock_debug_joint_names.index("handle_joint")
        if "door_joint" in unlock_debug_joint_names:
            unlock_debug_door_jid = unlock_debug_joint_names.index("door_joint")

    def _unlock_debug_vec(tensor, precision=3):
        values = tensor.detach().flatten().cpu().tolist()
        return "[" + ",".join(f"{float(value):+.{precision}f}" for value in values) + "]"

    def _unlock_debug_snapshot(step, reward):
        term = unlock_debug_term
        if robot is None or door is None or term is None:
            return None
        arm_ids = term.joint_ids
        q = robot.data.joint_pos[0, arm_ids]
        dq = robot.data.joint_vel[0, arm_ids]
        q_des = term.q_des[0]
        q_default = robot.data.default_joint_pos[0, arm_ids]
        torque_buffer = getattr(robot.data, "applied_torque", None)
        torque = torque_buffer[0, arm_ids] if torque_buffer is not None else term.tau_cmd[0]
        contact = float(_filtered_force_norm(left_sensor)[0]) if left_sensor is not None else float("nan")
        handle_pos = float(door.data.joint_pos[0, unlock_debug_handle_jid]) if unlock_debug_handle_jid is not None else float("nan")
        handle_vel = float(door.data.joint_vel[0, unlock_debug_handle_jid]) if unlock_debug_handle_jid is not None else float("nan")
        door_pos = float(door.data.joint_pos[0, unlock_debug_door_jid]) if unlock_debug_door_jid is not None else float("nan")
        door_vel = float(door.data.joint_vel[0, unlock_debug_door_jid]) if unlock_debug_door_jid is not None else float("nan")
        lock_mode = int(getattr(base_env, "_door_lock_mode", torch.zeros(1, device=q.device, dtype=torch.long))[0])
        return {
            "step": int(step),
            "lock_mode": lock_mode,
            "reward": float(reward),
            "contact": contact,
            "handle_pos": handle_pos,
            "handle_vel": handle_vel,
            "door_pos": door_pos,
            "door_vel": door_vel,
            "raw": term.raw_actions[0, 5:].detach().clone(),
            "delta": term.applied_delta[0].detach().clone(),
            "q": q.detach().clone(),
            "q_des": q_des.detach().clone(),
            "q_default_err": (q_des - q_default).detach().clone(),
            "q_err": (q_des - q).detach().clone(),
            "dq": dq.detach().clone(),
            "torque": torque.detach().clone(),
        }

    def _print_unlock_debug(snapshot, relative_step):
        q_err = snapshot["q_err"]
        print(
            f"[UNLOCKDBG {relative_step:+03d}] step={snapshot['step']} mode={snapshot['lock_mode']} "
            f"reward={snapshot['reward']:+.4f} contact={snapshot['contact']:.3f}N "
            f"handle={snapshot['handle_pos']:+.4f}/{snapshot['handle_vel']:+.4f}rad/s "
            f"door={snapshot['door_pos']:+.4f}/{snapshot['door_vel']:+.4f}rad/s"
        )
        print(
            f"  raw={_unlock_debug_vec(snapshot['raw'])} "
            f"delta={_unlock_debug_vec(snapshot['delta'], 4)}"
        )
        print(
            f"  q={_unlock_debug_vec(snapshot['q'])} qdes={_unlock_debug_vec(snapshot['q_des'])} "
            f"qdes-default={_unlock_debug_vec(snapshot['q_default_err'])}"
        )
        print(
            f"  qerr={_unlock_debug_vec(q_err)} max/rms="
            f"{float(q_err.abs().max()):.4f}/{float(torch.sqrt(torch.mean(q_err.square()))):.4f} "
            f"dq={_unlock_debug_vec(snapshot['dq'])} tau={_unlock_debug_vec(snapshot['torque'])}"
        )

    stance_draw = None
    if not args_cli.headless:
        try:
            from isaacsim.util.debug_draw import _debug_draw
            stance_draw = _debug_draw.acquire_debug_draw_interface()
        except Exception:
            stance_draw = None
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # Optionally override the last action dim (gripper) for a short window.
            # Convention: +1 -> open, -1 -> close (Binary action will threshold internally).
            stage = "policy"
            a_policy0 = float(actions[0, -1].item()) if actions.numel() > 0 else 0.0
            if args_cli.force_gripper and actions.shape[-1] > 6:
                close_val, open_val = -1.0, 1.0
                if args_cli.force_gripper_pattern == "close":
                    if step_count < force_steps:
                        actions[:, -1] = close_val
                        stage = "forced_close"
                elif args_cli.force_gripper_pattern == "open":
                    if step_count < force_steps:
                        actions[:, -1] = open_val
                        stage = "forced_open"
                else:  # close_open
                    if step_count < force_steps:
                        actions[:, -1] = close_val
                        stage = "forced_close"
                    elif step_count < 2 * force_steps:
                        actions[:, -1] = open_val
                        stage = "forced_open"

            # env stepping
            obs, rewards, dones, _ = env.step(actions)

            # Print only around the first physical-unlock edge. Six preceding
            # and twelve following policy steps expose action/controller/contact ordering.
            try:
                unlock_debug_snapshot = _unlock_debug_snapshot(step_count, rewards[0])
                if unlock_debug_snapshot is not None:
                    unlock_debug_now = unlock_debug_snapshot["lock_mode"] == 2
                    unlock_debug_new = unlock_debug_now and not unlock_debug_prev
                    if unlock_debug_new:
                        print("\n========== UNLOCK DIAGNOSTIC BEGIN (env0) ==========")
                        buffered = list(unlock_debug_pre)
                        for index, buffered_snapshot in enumerate(buffered):
                            _print_unlock_debug(buffered_snapshot, index - len(buffered))
                        _print_unlock_debug(unlock_debug_snapshot, 0)
                        unlock_debug_post_remaining = 12
                        unlock_debug_pre.clear()
                    elif unlock_debug_post_remaining > 0:
                        relative = 13 - unlock_debug_post_remaining
                        _print_unlock_debug(unlock_debug_snapshot, relative)
                        unlock_debug_post_remaining -= 1
                        if unlock_debug_post_remaining == 0:
                            print("========== UNLOCK DIAGNOSTIC END (env0) ==========\n")
                    else:
                        unlock_debug_pre.append(unlock_debug_snapshot)
                    unlock_debug_prev = unlock_debug_now
                    if bool(dones[0]):
                        unlock_debug_prev = False
                        unlock_debug_pre.clear()
                        unlock_debug_post_remaining = 0
            except Exception as exc:
                print(f"[UNLOCKDBG] disabled after diagnostic error: {exc}")
                unlock_debug_term = None

            # Pull displays live zones until activation, then the frozen
            # traverse zones. Push retains its original pick-stance marker.
            if is_pull_task and stance_draw is not None and step_count % 10 == 0:
                _draw_pull_zones(base_env, stance_draw, env_index=0)
            elif stance_draw is not None and door is not None and handle_body_id is not None:
                p_handle = door.data.body_pos_w[0, handle_body_id]
                q_handle = door.data.body_quat_w[0, handle_body_id]
                offset_h = torch.tensor((-0.08, 0.04, -0.005), device=p_handle.device, dtype=p_handle.dtype)
                grasp_target = p_handle + _quat_rotate(q_handle[None], offset_h[None])[0]
                stance_target = grasp_target + torch.tensor((-0.3, 0.3, -1.0), device=p_handle.device, dtype=p_handle.dtype)
                stance_draw.clear_points()
                stance_draw.draw_points(
                    [tuple(float(x) for x in stance_target)],
                    [(0.0, 1.0, 0.0, 1.0)],
                    [20.0],
                )

            if trajectory_recorder is not None:
                if trajectory_recorder.step(actions, rewards, dones, step_count):
                    print(
                        f"[TRAJ] Reached max_success_trajectories={args_cli.max_success_trajectories}; stopping play."
                    )
                    break

            # -----------------------------------------------------------------
            # Optional compact rollout stats (disabled by default; enable with --stats_every N).
            # -----------------------------------------------------------------
            if args_cli.stats_every > 0 and (step_count % args_cli.stats_every == 0):
                try:
                    # gripper close ratio (after any forced override)
                    close_ratio = 0.0
                    if actions.shape[-1] > 6:
                        close_ratio = float((actions[:, -1] < 0.0).float().mean().item())

                    # handle-only contact ratio (filtered)
                    any_contact_ratio = 0.0
                    f_any_mean = 0.0
                    if (left_sensor is not None) and (right_sensor is not None):
                        fL = _filtered_force_norm(left_sensor)
                        fR = _filtered_force_norm(right_sensor)
                        f_any = torch.maximum(fL, fR)
                        any_contact_ratio = float((f_any > args_cli.stats_contact_threshold).float().mean().item())
                        f_any_mean = float(f_any.mean().item())

                    # wrap alignment ratio + fingertip-mid distance mean
                    align_ratio = float("nan")
                    dist_mean = float("nan")
                    closed_mean = float("nan")
                    tip_gap_mean = float("nan")

                    if (robot is not None) and (door is not None) and (finger_body_ids is not None) and (handle_body_id is not None):
                        lb, rb = finger_body_ids
                        pL = robot.data.body_pos_w[:, lb, :]
                        pR = robot.data.body_pos_w[:, rb, :]
                        pH = door.data.body_pos_w[:, handle_body_id, :]
                        qH = door.data.body_quat_w[:, handle_body_id, :]

# fingertip-mid distance to grasp point (handle-frame offset)
                        pTip = 0.5 * (pL + pR)
                        off = torch.tensor([-0.08, 0.04, -0.005], device=pH.device, dtype=pH.dtype).unsqueeze(0).repeat(pH.shape[0], 1)
                        pG = pH + _quat_rotate(qH, off)
                        dist = torch.linalg.norm(pTip - pG, dim=-1)
                        dist_mean = float(dist.mean().item())

                        tip_gap = torch.linalg.norm(pL - pR, dim=-1)
                        tip_gap_mean = float(tip_gap.mean().item())

                        # wrap alignment in handle frame (grasp_axis=2 => z)
                        L_h = _quat_rotate(_quat_conjugate(qH), pL - pH)
                        R_h = _quat_rotate(_quat_conjugate(qH), pR - pH)
                        side = (L_h[:, 2] * R_h[:, 2]) < 0.0
                        sep = torch.abs(L_h[:, 2] - R_h[:, 2]) > float(args_cli.stats_min_sep)
                        align = (side & sep).float()
                        align_ratio = float(align.mean().item())
                    elif (robot is not None) and (door is not None) and (hand_body_id is not None) and (handle_body_id is not None):
                        pG0 = robot.data.body_pos_w[:, hand_body_id, :]
                        pH = door.data.body_pos_w[:, handle_body_id, :]
                        qH = door.data.body_quat_w[:, handle_body_id, :]
                        off = torch.tensor([-0.08, 0.04, -0.005], device=pH.device, dtype=pH.dtype).unsqueeze(0).repeat(pH.shape[0], 1)
                        pTarget = pH + _quat_rotate(qH, off)
                        dist = torch.linalg.norm(pG0 - pTarget, dim=-1)
                        dist_mean = float(dist.mean().item())

                    # closedness mean from finger joints
                    if (robot is not None) and (finger_joint_ids is not None) and (len(finger_joint_ids) >= 1):
                        width = _compute_gripper_width(robot, finger_joint_ids)
                        if width is not None:
                            closed = 1.0 - torch.clamp(width / float(args_cli.stats_open_width), 0.0, 1.0)
                            closed_mean = float(closed.mean().item())

                    print(
                        f"[STATS] step={step_count} stage={stage} "
                        f"close_ratio={close_ratio:.2f} align_ratio={align_ratio:.2f} "
                        f"any_contact_ratio={any_contact_ratio:.2f} f_any_mean={f_any_mean:.2f} "
                        f"dist_mean={dist_mean:.3f} tip_gap_mean={tip_gap_mean:.3f} "
                        f"closed_mean={closed_mean:.2f}"
                    )

                    # optional: detailed env0 finger debug
                    if args_cli.debug_fingers and (robot is not None) and (finger_joint_ids is not None) and (len(finger_joint_ids) >= 1):
                        q0 = robot.data.joint_pos[0, finger_joint_ids]
                        dq0 = robot.data.joint_vel[0, finger_joint_ids]
                        width0 = _compute_gripper_width(robot, finger_joint_ids)[0]
                        if q_target_field is not None:
                            q_t_all = getattr(robot.data, q_target_field)
                            q_t0 = q_t_all[0, finger_joint_ids]
                            print(f"[DBG0] finger q={q0.tolist()} dq={dq0.tolist()} q_target={q_t0.tolist()} width={float(width0.item()):.5f}")
                        else:
                            print(f"[DBG0] finger q={q0.tolist()} dq={dq0.tolist()} width={float(width0.item()):.5f} (no q_target field)")

                        if args_cli.print_contact_forces and (left_sensor is not None) and (right_sensor is not None):
                            fL0 = float(_filtered_force_norm(left_sensor)[0].item())
                            fR0 = float(_filtered_force_norm(right_sensor)[0].item())
                            print(f"[DBG0] filtered_contact |L|={fL0:.3f} |R|={fR0:.3f} (threshold={args_cli.stats_contact_threshold:.2f})")

                except Exception as e:
                    print(f"[STATS] failed: {e}")

            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        step_count += 1

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

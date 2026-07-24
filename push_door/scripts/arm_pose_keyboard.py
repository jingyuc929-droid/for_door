"""Fixed-base keyboard tool for designing the DoorBot arm initial pose.

The quadruped root is fixed and all leg joints are held at their configured
default positions. Only the six Piper arm position targets are editable.
Focus the Isaac Sim viewport before pressing keys.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Edit and save a fixed-base DoorBot arm pose with the keyboard.")
parser.add_argument("--joint-step", type=float, default=0.015, help="Joint target increment per simulation step (rad).")
parser.add_argument("--save-dir", type=str, default=os.path.join(os.path.dirname(__file__), "saved_arm_poses"))
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import omni.appwindow
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from door_env.tasks.manager_based.door_env.door_env_env_cfg import DoorEnvEnvCfg


ARM_NAMES = [f"link{i}_joint" for i in range(1, 7)]
LEG_PATTERN = "(FL|FR|RL|RR)_(hip|thigh|calf)_joint"

TRAIN_ENV_CFG = DoorEnvEnvCfg()
TRAIN_ACTION_CFG = TRAIN_ENV_CFG.actions.high_level_action
ROBOT_CFG = copy.deepcopy(TRAIN_ENV_CFG.scene.robot)
ROBOT_CFG.prim_path = "/World/Robot"
ROBOT_CFG.spawn.articulation_props.fix_root_link = True


@configclass
class ArmPoseSceneCfg(InteractiveSceneCfg):
    num_envs = 1
    env_spacing = 2.0
    replicate_physics = False

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(size=(20.0, 20.0), color=(0.78, 0.80, 0.84)),
    )
    robot = ROBOT_CFG
    light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
    )


def _key_name(key) -> str:
    return str(key).split(".")[-1].upper()


def _joint_index_from_key(key: str) -> int | None:
    """Accept Carb's common number-row and numpad key naming variants."""
    aliases = {
        **{str(i): i - 1 for i in range(1, 7)},
        **{f"KEY_{i}": i - 1 for i in range(1, 7)},
        **{f"NUM_{i}": i - 1 for i in range(1, 7)},
        **{f"NUMPAD_{i}": i - 1 for i in range(1, 7)},
        **{f"KP_{i}": i - 1 for i in range(1, 7)},
    }
    return aliases.get(key)


class Keyboard:
    def __init__(self, save_callback, reset_callback, print_selection_callback):
        self.selected_joint = 0
        self.direction = 0.0
        self.quit_requested = False
        self._save_callback = save_callback
        self._reset_callback = reset_callback
        self._print_selection_callback = print_selection_callback
        self._pressed = set()
        self._save_latched = False
        app_window = omni.appwindow.get_default_app_window()
        self._keyboard = app_window.get_keyboard()
        self._input = carb.input.acquire_input_interface()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)

    def close(self):
        if self._sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None

    def _update_direction(self):
        positive = "UP" in self._pressed or "E" in self._pressed
        negative = "DOWN" in self._pressed or "Q" in self._pressed
        self.direction = float(positive) - float(negative)

    def _on_event(self, event, *args, **kwargs):
        key = _key_name(event.input)
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            print(f"[KEY] PRESS   raw={event.input!s}  normalized={key}")
            self._pressed.add(key)
            selected = _joint_index_from_key(key)
            if selected is not None:
                self.selected_joint = selected
                self._print_selection_callback(selected)
            elif key == "S" and not self._save_latched:
                self._save_latched = True
                self._save_callback()
            elif key == "R":
                self._reset_callback()
            elif key == "ESCAPE":
                self.quit_requested = True
            self._update_direction()
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            print(f"[KEY] RELEASE raw={event.input!s}  normalized={key}")
            self._pressed.discard(key)
            if key == "S":
                self._save_latched = False
            self._update_direction()
        return True


def _ids(names: list[str], requested: list[str]) -> list[int]:
    missing = [name for name in requested if name not in names]
    if missing:
        raise RuntimeError(f"Missing joints {missing}. Available joints: {names}")
    return [names.index(name) for name in requested]


def _pose_dict(pos: torch.Tensor, quat: torch.Tensor) -> dict[str, list[float]]:
    return {
        "position": [float(v) for v in pos.detach().cpu().tolist()],
        "quaternion_wxyz": [float(v) for v in quat.detach().cpu().tolist()],
    }


def main():
    sim_cfg = SimulationCfg(dt=1.0 / 120.0, device=args_cli.device, use_fabric=not args_cli.disable_fabric)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view((2.2, 2.4, 1.7), (0.7, 0.4, 0.8))
    scene = InteractiveScene(ArmPoseSceneCfg())
    sim.reset()
    scene.update(sim_cfg.dt)

    robot: Articulation = scene["robot"]
    joint_names = list(robot.joint_names)
    arm_ids = _ids(joint_names, ARM_NAMES)
    leg_ids, _ = robot.find_joints(LEG_PATTERN, preserve_order=True)
    base_body_id = list(robot.body_names).index("base_link")
    grasp_body_id = list(robot.body_names).index("gripper_grasp_center")

    target = robot.data.default_joint_pos.clone()
    initial_target = target.clone()
    soft_limits = robot.data.soft_joint_pos_limits[0, arm_ids]

    # Match the overrides performed by HighLevelDoorOpenAction during real
    # training/zero-agent environment construction.
    def _arm_param(value) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=robot.device, dtype=torch.float32).flatten()
        if tensor.numel() == 1:
            tensor = tensor.repeat(len(arm_ids))
        return tensor.view(1, -1)

    robot.write_joint_stiffness_to_sim(
        _arm_param(TRAIN_ACTION_CFG.arm_stiffness), joint_ids=arm_ids
    )
    robot.write_joint_damping_to_sim(
        _arm_param(TRAIN_ACTION_CFG.arm_damping), joint_ids=arm_ids
    )
    robot.write_joint_effort_limit_to_sim(
        _arm_param(TRAIN_ACTION_CFG.effort_limit), joint_ids=arm_ids
    )
    robot.write_joint_velocity_limit_to_sim(
        _arm_param(TRAIN_ACTION_CFG.velocity_limit), joint_ids=arm_ids
    )
    robot.write_joint_armature_to_sim(
        _arm_param(TRAIN_ACTION_CFG.armature), joint_ids=arm_ids
    )
    os.makedirs(args_cli.save_dir, exist_ok=True)

    print("\n========== Arm joint limits / initial targets ==========")
    for index, (name, jid) in enumerate(zip(ARM_NAMES, arm_ids)):
        print(
            f"{index + 1}: {name:<11} "
            f"limit=[{float(soft_limits[index, 0]):+.4f}, {float(soft_limits[index, 1]):+.4f}] rad  "
            f"initial={float(target[0, jid]):+.4f} rad"
        )

    def save_pose():
        # Save the measured state so the file describes the pose actually reached.
        q = robot.data.joint_pos[0, arm_ids]
        base_pos = robot.data.body_pos_w[0, base_body_id]
        base_quat = robot.data.body_quat_w[0, base_body_id]
        grasp_pos = robot.data.body_pos_w[0, grasp_body_id]
        grasp_quat = robot.data.body_quat_w[0, grasp_body_id]
        grasp_pos_b, grasp_quat_b = subtract_frame_transforms(base_pos, base_quat, grasp_pos, grasp_quat)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(args_cli.save_dir, f"doorbot_arm_pose_{stamp}.json")
        payload = {
            "timestamp": stamp,
            "usage": "Copy training_init_joint_position into DoorEnvSceneCfg.robot.init_state.joint_pos.",
            "arm_joint_names": ARM_NAMES,
            # The training zero action uses default joint position as q_des.
            # Therefore the command target, not the gravity-deflected measured
            # position, is the value that reproduces this settled pose.
            "training_init_joint_position": [float(v) for v in target[0, arm_ids].detach().cpu().tolist()],
            "settled_measured_joint_position": [float(v) for v in q.detach().cpu().tolist()],
            "q_des_minus_q": [float(v) for v in (target[0, arm_ids] - q).detach().cpu().tolist()],
            "grasp_center_world": _pose_dict(grasp_pos, grasp_quat),
            "grasp_center_base": _pose_dict(grasp_pos_b, grasp_quat_b),
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
            file.write("\n")
        print(f"[SAVE] {path}")
        print(f"       COPY THIS TO TRAINING INIT={payload['training_init_joint_position']}")
        print(f"       settled_measured={payload['settled_measured_joint_position']}")
        print(f"       q_des_minus_q={payload['q_des_minus_q']}")
        print(f"       grasp_center_base={payload['grasp_center_base']}")

    def reset_pose():
        target[:] = initial_target
        print("[RESET] Arm target restored to the configured default pose.")

    def print_selection(index: int):
        jid = arm_ids[index]
        print(
            f"[SELECT] joint={index + 1} name={ARM_NAMES[index]} "
            f"target={float(target[0, jid]):+.4f} rad"
        )

    keyboard = Keyboard(save_pose, reset_pose, print_selection)
    print(
        """
Keyboard controls (focus the viewport)
--------------------------------------
1..6       select arm joint
Up / E     increase selected joint target
Down / Q   decrease selected joint target
S          save current arm and grasp-center pose
R          restore the configured default arm pose
Esc        quit

The robot root is fixed and all leg joints remain at their default targets.
Arm stiffness/damping/effort/velocity/armature match the training action term.
After S, copy training_init_joint_position (not settled_measured_joint_position).
"""
    )

    try:
        adjustment_steps = 0
        while simulation_app.is_running() and not keyboard.quit_requested:
            with torch.inference_mode():
                if keyboard.direction != 0.0:
                    jid = arm_ids[keyboard.selected_joint]
                    before = target[0, jid].clone()
                    requested = before + keyboard.direction * float(args_cli.joint_step)
                    target[0, jid] = torch.clamp(requested, *soft_limits[keyboard.selected_joint])
                    adjustment_steps += 1
                    if adjustment_steps == 1 or adjustment_steps % 12 == 0:
                        actual = robot.data.joint_pos[0, jid]
                        was_clamped = not torch.isclose(requested, target[0, jid])
                        print(
                            f"[ADJUST] {ARM_NAMES[keyboard.selected_joint]} "
                            f"target {float(before):+.4f} -> {float(target[0, jid]):+.4f}, "
                            f"actual={float(actual):+.4f} rad, clamped={bool(was_clamped)}"
                        )
                else:
                    adjustment_steps = 0

                # Hold every leg joint and command only the editable arm targets.
                target[:, leg_ids] = initial_target[:, leg_ids]
                robot.set_joint_position_target(target)
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim_cfg.dt)
    except KeyboardInterrupt:
        pass
    finally:
        keyboard.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

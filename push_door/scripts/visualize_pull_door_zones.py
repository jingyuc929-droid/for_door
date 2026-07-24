# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize chord-based pull-door zones with a zero-action agent.

The hinge H, closed endpoint H+d0, current endpoint H+d, endpoint chord d-d0,
and current door direction partition the displayed disk into Z1/Z2/Z3. The
script fixes the door joint at the requested angle for inspection.
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Visualize Z1/Z2/Z3 for the pull-door task.")
parser.add_argument("--task", type=str, default="Template-Pull-Door-Env-v0", help="Gym task ID.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments; only env0 is drawn.")
parser.add_argument("--door_joint_pos", type=float, default=-0.45, help="Fixed door angle in radians.")
parser.add_argument("--door_length", type=float, default=1.2, help="Hinge-to-panel-end length in metres.")
parser.add_argument("--zone_radius", type=float, default=3.0, help="Displayed zone radius from the hinge in metres.")
parser.add_argument("--radial_step", type=float, default=0.055, help="Spacing between visualization points in metres.")
parser.add_argument("--angular_samples", type=int, default=240, help="Angular samples per ring.")
parser.add_argument("--draw_every", type=int, default=20, help="Redraw interval in simulation steps.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after N steps; 0 keeps running.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Fabric and use USD I/O.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply

import door_env.tasks  # noqa: F401


ZONE_COLORS = {
    "z1": (0.20, 0.85, 0.25, 0.35),
    "z2": (1.00, 0.72, 0.12, 0.35),
    "z3": (0.15, 0.55, 1.00, 0.32),
}


def _make_draw():
    try:
        from isaacsim.util.debug_draw import _debug_draw

        return _debug_draw.acquire_debug_draw_interface()
    except Exception as exc:
        print(f"[WARN] debug_draw unavailable: {exc}")
        return None


def _unit_xy(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.clamp(torch.linalg.norm(vector), min=1.0e-8)


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    result = quat.clone()
    result[..., 1:] = -result[..., 1:]
    return result


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _closed_direction_xy(door, env_id: int = 0) -> torch.Tensor:
    """Closed panel direction: the door-frame local +X axis in world XY."""
    local_positive_x = torch.tensor(
        (1.0, 0.0, 0.0),
        device=door.data.root_pos_w.device,
        dtype=door.data.root_pos_w.dtype,
    ).view(1, 3)
    direction_w = quat_apply(door.data.root_quat_w[env_id : env_id + 1], local_positive_x)[0]
    return _unit_xy(direction_w[:2])


def _current_panel_direction_xy(
    door,
    panel_body_id: int,
    closed_dir: torch.Tensor,
    closed_panel_quat: torch.Tensor,
    env_id: int = 0,
) -> torch.Tensor:
    """Apply the panel's measured orientation change to the closed +X ray."""
    current_quat = door.data.body_quat_w[env_id, panel_body_id, :]
    delta_quat = _quat_mul(current_quat, _quat_conjugate(closed_panel_quat))
    closed_direction_w = torch.cat((closed_dir, torch.zeros_like(closed_dir[:1])))
    direction_w = quat_apply(delta_quat.view(1, 4), closed_direction_w.view(1, 3))[0]
    return _unit_xy(direction_w[:2])


def _cross_xy(a_x: float, a_y: float, b_x: float, b_y: float) -> float:
    return a_x * b_y - a_y * b_x


def _set_door_joint(door, door_joint_id: int, position: float) -> None:
    joint_pos = door.data.joint_pos.clone()
    joint_vel = torch.zeros_like(door.data.joint_vel)
    joint_pos[:, door_joint_id] = float(position)
    door.write_joint_state_to_sim(joint_pos, joint_vel)


def _draw_zones(
    draw,
    hinge_xy: torch.Tensor,
    closed_dir: torch.Tensor,
    current_dir: torch.Tensor,
    door_length: float,
    radius: float,
    radial_step: float,
    angular_samples: int,
    z: float = 0.025,
) -> tuple[float, float]:
    """Draw chord/door half-plane zones and return opening sign and angle."""
    if draw is None:
        return 0.0, 0.0

    cross_closed_current = closed_dir[0] * current_dir[1] - closed_dir[1] * current_dir[0]
    opening_sign = 1.0 if float(cross_closed_current.item()) >= 0.0 else -1.0
    opening_angle = math.atan2(
        abs(float(cross_closed_current.item())),
        float(torch.clamp(torch.dot(closed_dir, current_dir), -1.0, 1.0).item()),
    )

    # d0 and d are both 1.2 m by default and point from H to the panel end.
    d0_x = float(door_length) * float(closed_dir[0].item())
    d0_y = float(door_length) * float(closed_dir[1].item())
    d_x = float(door_length) * float(current_dir[0].item())
    d_y = float(door_length) * float(current_dir[1].item())

    # The chord direction is d-d0. Its hinge-facing half-plane is "inside".
    chord_x = d_x - d0_x
    chord_y = d_y - d0_y
    hinge_chord_side = _cross_xy(chord_x, chord_y, -d0_x, -d0_y)
    triangle_den = _cross_xy(d0_x, d0_y, d_x, d_y)

    points: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float, float]] = []
    sizes: list[float] = []
    ring_count = max(1, int(float(radius) / max(float(radial_step), 1.0e-3)))
    angular_count = max(24, int(angular_samples))

    hinge_x = float(hinge_xy[0].item())
    hinge_y = float(hinge_xy[1].item())
    for ring in range(1, ring_count + 1):
        r = float(radius) * ring / ring_count
        # Use fewer points near the hinge to keep approximately uniform density.
        samples = max(24, int(angular_count * ring / ring_count))
        for index in range(samples):
            world_angle = -math.pi + 2.0 * math.pi * index / samples
            point_x = r * math.cos(world_angle)
            point_y = r * math.sin(world_angle)

            point_chord_side = _cross_xy(
                chord_x,
                chord_y,
                point_x - d0_x,
                point_y - d0_y,
            )
            chord_inside = point_chord_side * hinge_chord_side >= -1.0e-8

            if abs(triangle_den) > 1.0e-8:
                coeff_d0 = _cross_xy(point_x, point_y, d_x, d_y) / triangle_den
                coeff_d = _cross_xy(d0_x, d0_y, point_x, point_y) / triangle_den
                in_triangle = coeff_d0 >= 0.0 and coeff_d >= 0.0 and (coeff_d0 + coeff_d) <= 1.0
            else:
                in_triangle = False

            # Continuing opening side: left of d for CCW, right of d for CW.
            on_opening_side_of_d = opening_sign * _cross_xy(d_x, d_y, point_x, point_y) >= 0.0
            on_opening_halfplane = opening_sign * _cross_xy(d0_x, d0_y, point_x, point_y) >= 0.0

            if not on_opening_halfplane:
                # The entire half-disk opposite the closed door's opening side
                # is Z2, regardless of the chord or current-door tests.
                color = ZONE_COLORS["z2"]
            elif not chord_inside:
                color = ZONE_COLORS["z1"]
            elif in_triangle:
                color = ZONE_COLORS["z2"]
            elif on_opening_side_of_d:
                color = ZONE_COLORS["z3"]
            else:
                # Z2 also contains the former orange chord-inside region.
                color = ZONE_COLORS["z2"]
            points.append((hinge_x + point_x, hinge_y + point_y, z))
            colors.append(color)
            sizes.append(5.0)

    draw.clear_points()
    draw.clear_lines()
    draw.draw_points(points, colors, sizes)

    hinge = (hinge_x, hinge_y, z + 0.015)
    closed_end = (
        hinge_x + d0_x,
        hinge_y + d0_y,
        z + 0.015,
    )
    current_end = (
        hinge_x + d_x,
        hinge_y + d_y,
        z + 0.015,
    )
    chord_norm = max(math.hypot(chord_x, chord_y), 1.0e-8)
    chord_u_x = chord_x / chord_norm
    chord_u_y = chord_y / chord_norm
    chord_mid_x = hinge_x + 0.5 * (d0_x + d_x)
    chord_mid_y = hinge_y + 0.5 * (d0_y + d_y)
    chord_line_start = (
        chord_mid_x - 2.0 * float(radius) * chord_u_x,
        chord_mid_y - 2.0 * float(radius) * chord_u_y,
        z + 0.012,
    )
    chord_line_end = (
        chord_mid_x + 2.0 * float(radius) * chord_u_x,
        chord_mid_y + 2.0 * float(radius) * chord_u_y,
        z + 0.012,
    )
    draw.draw_lines(
        [hinge, hinge, chord_line_start, closed_end],
        [closed_end, current_end, chord_line_end, current_end],
        [
            (0.05, 0.05, 0.05, 1.0),
            (1.0, 1.0, 1.0, 1.0),
            (1.0, 0.0, 1.0, 0.45),
            (1.0, 0.0, 1.0, 1.0),
        ],
        [5.0, 5.0, 2.0, 4.0],
    )
    draw.draw_points(
        [hinge, closed_end, current_end],
        [(1.0, 0.0, 1.0, 1.0), (0.05, 0.05, 0.05, 1.0), (1.0, 1.0, 1.0, 1.0)],
        [14.0, 11.0, 11.0],
    )
    return opening_sign, opening_angle


def _make_legend():
    if args_cli.headless:
        return None
    try:
        import omni.ui as ui

        window = ui.Window("Pull Door Zones", width=430, height=180)
        with window.frame:
            with ui.VStack(spacing=5):
                ui.Label("Z1 = green: opening half-disk, outside endpoint chord")
                ui.Label("Z2 = orange: opposite half-disk + triangle + remainder")
                ui.Label("Z3 = blue: opening half-disk, valid inside region")
                ui.Label("black=d0, white=d, magenta=endpoint chord d-d0")
                status = ui.Label("door joint / opening angle: --")
        return window, status
    except Exception as exc:
        print(f"[WARN] zone legend unavailable: {exc}")
        return None


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # The lock event would force an unpressed handle's door back to zero. It is
    # disabled only in this visualization script, not in the actual task cfg.
    if env_cfg.events is not None and hasattr(env_cfg.events, "door_mechanism"):
        env_cfg.events.door_mechanism = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    base_env = env.unwrapped
    door = base_env.scene["door"]
    joint_names = list(door.data.joint_names)
    body_names = list(door.body_names)
    if "door_joint" not in joint_names:
        raise RuntimeError(f"door_joint not found; available joints: {joint_names}")
    if "door_1" not in body_names:
        raise RuntimeError(f"door_1 not found; available bodies: {body_names}")

    door_joint_id = joint_names.index("door_joint")
    panel_body_id = body_names.index("door_1")
    closed_dir = _closed_direction_xy(door).detach().clone()
    closed_panel_quat = door.data.body_quat_w[0, panel_body_id, :].detach().clone()
    draw = _make_draw()
    legend = _make_legend()
    status_label = legend[1] if legend is not None else None

    print(f"[INFO] task: {args_cli.task}")
    print(f"[INFO] fixed door_joint: {args_cli.door_joint_pos:+.3f} rad")
    print(f"[INFO] closed panel direction XY: {tuple(float(x) for x in closed_dir.tolist())}")
    print(f"[INFO] d0/d length: {args_cli.door_length:.3f} m")
    print("[INFO] zones: Z1=green, Z2=orange, Z3=blue")

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            _set_door_joint(door, door_joint_id, args_cli.door_joint_pos)
            actions = torch.zeros(env.action_space.shape, device=base_env.device)
            env.step(actions)

            if step % max(1, int(args_cli.draw_every)) == 0:
                joint_pos = float(door.data.joint_pos[0, door_joint_id].item())
                current_dir = _current_panel_direction_xy(
                    door,
                    panel_body_id,
                    closed_dir,
                    closed_panel_quat,
                )
                hinge_xy = door.data.root_pos_w[0, :2]
                opening_sign, opening_angle = _draw_zones(
                    draw,
                    hinge_xy,
                    closed_dir,
                    current_dir,
                    args_cli.door_length,
                    args_cli.zone_radius,
                    args_cli.radial_step,
                    args_cli.angular_samples,
                )
                if status_label is not None:
                    geometric_signed_angle = math.atan2(
                        float((closed_dir[0] * current_dir[1] - closed_dir[1] * current_dir[0]).item()),
                        float(torch.clamp(torch.dot(closed_dir, current_dir), -1.0, 1.0).item()),
                    )
                    status_label.text = (
                        f"joint={joint_pos:+.3f}, geometry={geometric_signed_angle:+.3f} rad, "
                        f"opening={opening_angle:.3f}, sign={opening_sign:+.0f}"
                    )
            step += 1
            if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

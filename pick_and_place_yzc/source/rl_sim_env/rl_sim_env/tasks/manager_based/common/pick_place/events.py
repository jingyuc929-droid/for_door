"""Event terms for hierarchical pick-and-place tasks."""

from __future__ import annotations

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp import reset_root_state_uniform
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_yaw


def _env_origins(env) -> torch.Tensor:
    origins = getattr(env.scene, "env_origins", None)
    if origins is None:
        return torch.zeros((env.num_envs, 3), device=env.device)
    return origins.to(env.device)


def _reference_pose_w(
    env,
    env_ids: torch.Tensor,
    reference_asset_cfg: SceneEntityCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if reference_asset_cfg is None:
        return _env_origins(env)[env_ids], torch.tensor((1.0, 0.0, 0.0, 0.0), device=env.device).repeat(
            env_ids.numel(), 1
        )
    try:
        asset: Articulation | RigidObject = env.scene[reference_asset_cfg.name]
    except Exception:
        return _env_origins(env)[env_ids], torch.tensor((1.0, 0.0, 0.0, 0.0), device=env.device).repeat(
            env_ids.numel(), 1
        )
    return asset.data.root_pos_w[env_ids].clone(), asset.data.root_quat_w[env_ids].clone()


def _ground_origins_for(env, env_ids: torch.Tensor) -> torch.Tensor:
    origins = _env_origins(env)[env_ids].clone()
    origins[:, :2] = 0.0
    return origins


def _ensure_pick_place_target_visualizers(
    env,
    pick_radius: float = 0.045,
    place_radius: float = 0.055,
):
    pick_vis = getattr(env, "pick_place_pick_visualizer", None)
    place_vis = getattr(env, "pick_place_place_visualizer", None)
    if pick_vis is not None and place_vis is not None:
        return

    try:
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    except Exception as exc:
        if not getattr(env, "_pick_place_visualizer_warned", False):
            print(f"[WARN] Pick/place target visualization is disabled: failed to import markers ({exc}).")
            env._pick_place_visualizer_warned = True
        env.pick_place_pick_visualizer = None
        env.pick_place_place_visualizer = None
        return

    pick_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/PickPlace/pick_target",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=float(pick_radius),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.55)),
            )
        },
    )
    place_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/PickPlace/place_target",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=float(place_radius),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.62, 0.62)),
            )
        },
    )
    try:
        env.pick_place_pick_visualizer = VisualizationMarkers(pick_cfg)
        env.pick_place_place_visualizer = VisualizationMarkers(place_cfg)
        env.pick_place_pick_visualizer.set_visibility(True)
        env.pick_place_place_visualizer.set_visibility(True)
    except Exception as exc:
        if not getattr(env, "_pick_place_visualizer_warned", False):
            print(f"[WARN] Pick/place target visualization is disabled: failed to create markers ({exc}).")
            env._pick_place_visualizer_warned = True
        env.pick_place_pick_visualizer = None
        env.pick_place_place_visualizer = None


def _visualize_pick_place_targets(env, max_envs: int = 128):
    pick_vis = getattr(env, "pick_place_pick_visualizer", None)
    place_vis = getattr(env, "pick_place_place_visualizer", None)
    pick_pos_w = getattr(env, "pick_place_pick_pos_w", None)
    place_pos_w = getattr(env, "pick_place_place_pos_w", None)
    if pick_vis is None or place_vis is None or pick_pos_w is None or place_pos_w is None:
        return
    count = min(int(max_envs), env.num_envs)
    pick_vis.visualize(translations=pick_pos_w[:count])
    place_vis.visualize(translations=place_pos_w[:count])


def visualize_pick_place_targets(
    env,
    env_ids: torch.Tensor | list[int] | None,
    debug_vis_max_envs: int = 128,
    debug_vis_pick_radius: float = 0.045,
    debug_vis_place_radius: float = 0.055,
):
    """Refresh pick/place target markers for play/debug visualization."""
    pick_pos_w = getattr(env, "pick_place_pick_pos_w", None)
    place_pos_w = getattr(env, "pick_place_place_pos_w", None)
    if pick_pos_w is None or place_pos_w is None:
        return
    _ensure_pick_place_target_visualizers(
        env,
        pick_radius=debug_vis_pick_radius,
        place_radius=debug_vis_place_radius,
    )
    _visualize_pick_place_targets(env, max_envs=debug_vis_max_envs)


def sample_pick_place_targets(
    env,
    env_ids: torch.Tensor | list[int] | None,
    pick_x_range: tuple[float, float] = (0.35, 0.75),
    pick_y_range: tuple[float, float] = (-0.25, 0.25),
    pick_z_range: tuple[float, float] = (0.24, 0.24),
    place_x_range: tuple[float, float] = (0.85, 1.25),
    place_y_range: tuple[float, float] = (-0.25, 0.25),
    place_z_range: tuple[float, float] = (0.24, 0.24),
    min_pick_place_distance: float = 0.0,
    max_resample_attempts: int = 16,
    reference_asset_cfg: SceneEntityCfg | None = SceneEntityCfg("robot"),
    targets_in_reference_yaw_frame: bool = True,
    debug_vis: bool = False,
    debug_vis_max_envs: int = 128,
    debug_vis_pick_radius: float = 0.045,
    debug_vis_place_radius: float = 0.055,
):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, dtype=torch.long, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    def ensure_attr(name: str):
        value = getattr(env, name, None)
        if value is None or value.shape != (env.num_envs, 3):
            value = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
            setattr(env, name, value)
        return value

    pick_pos_w = ensure_attr("pick_place_pick_pos_w")
    place_pos_w = ensure_attr("pick_place_place_pos_w")
    ref_pos_w, ref_quat_w = _reference_pose_w(env, env_ids, reference_asset_cfg)

    cfg_root = getattr(getattr(env, "cfg", None), "config_summary", None)
    cfg_task = getattr(cfg_root, "task", None)
    use_per_env_curriculum = (
        cfg_task is not None
        and getattr(cfg_task, "pick_place_curriculum_enable", False)
        and str(getattr(cfg_task, "pick_place_curriculum_mode", "iteration")).lower() == "success"
    )
    levels = getattr(env, "pick_place_target_curriculum_levels", None)
    use_per_env_curriculum = use_per_env_curriculum and levels is not None
    if use_per_env_curriculum:
        max_level = max(float(getattr(cfg_task, "pick_place_curriculum_max_level", 0.0)), 1.0e-6)
        if levels.shape != (env.num_envs,):
            levels = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
            setattr(env, "pick_place_target_curriculum_levels", levels)
        progress = torch.clamp(levels[env_ids] / max_level, min=0.0, max=1.0)
        progress = progress ** float(getattr(cfg_task, "pick_place_curriculum_growth_power", 1.0))

        def range_tensors(name: str, fallback: tuple[float, float]) -> tuple[torch.Tensor, torch.Tensor]:
            start = getattr(cfg_task, f"{name}_start", fallback)
            final = getattr(cfg_task, f"{name}_final", fallback)
            start_low = float(start[0])
            start_high = float(start[1])
            low = start_low + (float(final[0]) - start_low) * progress
            high = start_high + (float(final[1]) - start_high) * progress
            return low, high

        pick_x_low, pick_x_high = range_tensors("pick_x_range", pick_x_range)
        pick_y_low, pick_y_high = range_tensors("pick_y_range", pick_y_range)
        pick_z_low, pick_z_high = range_tensors("pick_z_range", pick_z_range)
        place_x_low, place_x_high = range_tensors("place_x_range", place_x_range)
        place_y_low, place_y_high = range_tensors("place_y_range", place_y_range)
        place_z_low, place_z_high = range_tensors("place_z_range", place_z_range)
    else:
        num_targets = env_ids.numel()

        def full_range_tensors(value: tuple[float, float]) -> tuple[torch.Tensor, torch.Tensor]:
            low = torch.full((num_targets,), float(value[0]), device=env.device, dtype=torch.float32)
            high = torch.full((num_targets,), float(value[1]), device=env.device, dtype=torch.float32)
            return low, high

        pick_x_low, pick_x_high = full_range_tensors(pick_x_range)
        pick_y_low, pick_y_high = full_range_tensors(pick_y_range)
        pick_z_low, pick_z_high = full_range_tensors(pick_z_range)
        place_x_low, place_x_high = full_range_tensors(place_x_range)
        place_y_low, place_y_high = full_range_tensors(place_y_range)
        place_z_low, place_z_high = full_range_tensors(place_z_range)

    def sample_range(low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        return low + torch.rand_like(low) * (high - low)

    pick = torch.empty((env_ids.numel(), 3), device=env.device)
    place = torch.empty_like(pick)
    pick[:, 0] = sample_range(pick_x_low, pick_x_high)
    pick[:, 1] = sample_range(pick_y_low, pick_y_high)
    pick[:, 2] = sample_range(pick_z_low, pick_z_high)
    place[:, 0] = sample_range(place_x_low, place_x_high)
    place[:, 1] = sample_range(place_y_low, place_y_high)
    place[:, 2] = sample_range(place_z_low, place_z_high)
    if min_pick_place_distance > 0.0:
        min_dist = float(min_pick_place_distance)
        for _ in range(max(1, int(max_resample_attempts))):
            too_close = torch.linalg.norm(place[:, :2] - pick[:, :2], dim=-1) < min_dist
            if not torch.any(too_close):
                break
            place[too_close, 0] = sample_range(place_x_low[too_close], place_x_high[too_close])
            place[too_close, 1] = sample_range(place_y_low[too_close], place_y_high[too_close])
            place[too_close, 2] = sample_range(place_z_low[too_close], place_z_high[too_close])
        too_close = torch.linalg.norm(place[:, :2] - pick[:, :2], dim=-1) < min_dist
        if torch.any(too_close):
            place_x_mid = 0.5 * (place_x_low[too_close] + place_x_high[too_close])
            fallback_x = torch.where(
                pick[too_close, 0] <= place_x_mid,
                place_x_high[too_close],
                place_x_low[too_close],
            )
            place[too_close, 0] = fallback_x
            min_dist_tensor = torch.tensor(min_dist, device=env.device)
            y_delta = torch.sqrt(
                torch.clamp(
                    torch.square(min_dist_tensor) - torch.square(place[too_close, 0] - pick[too_close, 0]),
                    min=0.0,
                )
            )
            fallback_y = torch.where(
                pick[too_close, 1] <= 0.0,
                pick[too_close, 1] + y_delta,
                pick[too_close, 1] - y_delta,
            )
            place[too_close, 1] = torch.minimum(
                torch.maximum(fallback_y, place_y_low[too_close]),
                place_y_high[too_close],
            )

    pick_xy = pick.clone()
    place_xy = place.clone()
    pick_xy[:, 2] = 0.0
    place_xy[:, 2] = 0.0
    if targets_in_reference_yaw_frame:
        pick_xy = quat_apply_yaw(ref_quat_w, pick_xy)
        place_xy = quat_apply_yaw(ref_quat_w, place_xy)
    ground_origins = _ground_origins_for(env, env_ids)
    pick_pos_w[env_ids, :2] = ref_pos_w[:, :2] + pick_xy[:, :2]
    place_pos_w[env_ids, :2] = ref_pos_w[:, :2] + place_xy[:, :2]
    pick_pos_w[env_ids, 2] = ground_origins[:, 2] + pick[:, 2]
    place_pos_w[env_ids, 2] = ground_origins[:, 2] + place[:, 2]

    phase = getattr(env, "pick_place_phase", None)
    if phase is None or phase.shape != (env.num_envs,):
        phase = torch.zeros((env.num_envs,), device=env.device, dtype=torch.long)
        setattr(env, "pick_place_phase", phase)
    phase[env_ids] = 0

    counter = getattr(env, "pick_place_phase2_hold_counter", None)
    if counter is not None:
        counter[env_ids] = 0
    phase2_steps = getattr(env, "pick_place_phase2_step_counter", None)
    if phase2_steps is not None:
        phase2_steps[env_ids] = 0
    grasp_stable_counter = getattr(env, "pick_place_grasp_stable_counter", None)
    if grasp_stable_counter is not None:
        grasp_stable_counter[env_ids] = 0
    success = getattr(env, "pick_place_virtual_success", None)
    if success is not None:
        success[env_ids] = False
    condition = getattr(env, "pick_place_phase2_success_condition", None)
    if condition is not None:
        condition[env_ids] = False
    post_place_counter = getattr(env, "pick_place_post_place_hold_counter", None)
    if post_place_counter is not None:
        post_place_counter[env_ids] = 0
    post_place_condition = getattr(env, "pick_place_post_place_still_condition", None)
    if post_place_condition is not None:
        post_place_condition[env_ids] = False
    prev_dist = getattr(env, "pick_place_prev_ee_place_dist", None)
    if prev_dist is not None:
        prev_dist[env_ids] = float("nan")
    prev_pick_dist = getattr(env, "pick_place_prev_ee_pick_dist", None)
    if prev_pick_dist is not None:
        prev_pick_dist[env_ids] = float("nan")
    prev_object_place_dist = getattr(env, "pick_place_prev_object_place_dist", None)
    if prev_object_place_dist is not None:
        prev_object_place_dist[env_ids] = float("nan")
    grasp_success_paid = getattr(env, "pick_place_grasp_success_paid", None)
    if grasp_success_paid is not None:
        grasp_success_paid[env_ids] = False
    phase_progress_paid = getattr(env, "pick_place_phase_progress_paid", None)
    if phase_progress_paid is not None:
        phase_progress_paid[env_ids] = 0
    phase_transition_paid = getattr(env, "pick_place_phase_transition_bonus_paid", None)
    if phase_transition_paid is not None:
        phase_transition_paid[env_ids] = 0

    if debug_vis:
        _ensure_pick_place_target_visualizers(
            env,
            pick_radius=debug_vis_pick_radius,
            place_radius=debug_vis_place_radius,
        )
        _visualize_pick_place_targets(env, max_envs=debug_vis_max_envs)


def reset_object_to_pick(
    env,
    env_ids: torch.Tensor | list[int] | None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    position_noise: tuple[float, float] = (-0.02, 0.02),
    z_offset: float = 0.03,
):
    try:
        obj: RigidObject = env.scene[object_cfg.name]
    except Exception:
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, dtype=torch.long, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    pick_pos_w = getattr(env, "pick_place_pick_pos_w", None)
    if pick_pos_w is None:
        sample_pick_place_targets(env, env_ids)
        pick_pos_w = getattr(env, "pick_place_pick_pos_w")

    root_state = obj.data.default_root_state[env_ids].clone()
    root_state[:, :3] = pick_pos_w[env_ids]
    root_state[:, :2] += torch.empty((env_ids.numel(), 2), device=env.device).uniform_(*position_noise)
    root_state[:, 2] += float(z_offset)
    root_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=env.device).view(1, 4)
    root_state[:, 7:] = 0.0
    obj.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    obj.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)


def reset_support_cylinders_to_targets(
    env,
    env_ids: torch.Tensor | list[int] | None,
    pick_support_cfg: SceneEntityCfg = SceneEntityCfg("pick_support"),
    place_support_cfg: SceneEntityCfg = SceneEntityCfg("place_support"),
    pick_support_height: float = 0.2,
    place_support_height: float | None = None,
    object_height: float = 0.08,
):
    """Move pick/place support cylinders under the sampled target positions."""
    try:
        pick_support: RigidObject = env.scene[pick_support_cfg.name]
        place_support: RigidObject = env.scene[place_support_cfg.name]
    except Exception:
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, dtype=torch.long, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    pick_pos_w = getattr(env, "pick_place_pick_pos_w", None)
    place_pos_w = getattr(env, "pick_place_place_pos_w", None)
    if pick_pos_w is None or place_pos_w is None:
        sample_pick_place_targets(env, env_ids)
        pick_pos_w = getattr(env, "pick_place_pick_pos_w")
        place_pos_w = getattr(env, "pick_place_place_pos_w")

    if place_support_height is None:
        place_support_height = pick_support_height
    for support, target_pos_w, support_height in (
        (pick_support, pick_pos_w, float(pick_support_height)),
        (place_support, place_pos_w, float(place_support_height)),
    ):
        root_state = support.data.default_root_state[env_ids].clone()
        root_state[:, :2] = target_pos_w[env_ids, :2]
        root_state[:, 2] = target_pos_w[env_ids, 2] - 0.5 * (support_height + float(object_height))
        root_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=env.device).view(1, 4)
        root_state[:, 7:] = 0.0
        support.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        support.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)


def reset_physical_pick_place_scene(
    env,
    env_ids: torch.Tensor | list[int] | None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    pick_support_cfg: SceneEntityCfg = SceneEntityCfg("pick_support"),
    place_support_cfg: SceneEntityCfg = SceneEntityCfg("place_support"),
    pick_x_range: tuple[float, float] = (0.35, 0.75),
    pick_y_range: tuple[float, float] = (-0.25, 0.25),
    pick_z_range: tuple[float, float] = (0.24, 0.24),
    place_x_range: tuple[float, float] = (0.85, 1.25),
    place_y_range: tuple[float, float] = (-0.25, 0.25),
    place_z_range: tuple[float, float] = (0.34, 0.34),
    min_pick_place_distance: float = 0.0,
    reference_asset_cfg: SceneEntityCfg | None = SceneEntityCfg("robot"),
    targets_in_reference_yaw_frame: bool = True,
    pick_support_height: float = 0.2,
    place_support_height: float = 0.3,
    object_height: float = 0.08,
    object_xy_noise: tuple[float, float] = (0.0, 0.0),
    debug_vis: bool = False,
    debug_vis_max_envs: int = 128,
    print_debug_once: bool = True,
):
    """Reset physical object and support cylinders from one coherent target sample."""
    sample_pick_place_targets(
        env,
        env_ids,
        pick_x_range=pick_x_range,
        pick_y_range=pick_y_range,
        pick_z_range=pick_z_range,
        place_x_range=place_x_range,
        place_y_range=place_y_range,
        place_z_range=place_z_range,
        min_pick_place_distance=min_pick_place_distance,
        reference_asset_cfg=reference_asset_cfg,
        targets_in_reference_yaw_frame=targets_in_reference_yaw_frame,
        debug_vis=debug_vis,
        debug_vis_max_envs=debug_vis_max_envs,
    )
    reset_support_cylinders_to_targets(
        env,
        env_ids,
        pick_support_cfg=pick_support_cfg,
        place_support_cfg=place_support_cfg,
        pick_support_height=pick_support_height,
        place_support_height=place_support_height,
        object_height=object_height,
    )
    reset_object_to_pick(
        env,
        env_ids,
        object_cfg=object_cfg,
        position_noise=object_xy_noise,
        z_offset=0.0,
    )
    if print_debug_once and not getattr(env, "_pick_place_physical_reset_printed", False):
        pick_pos_w = getattr(env, "pick_place_pick_pos_w", None)
        place_pos_w = getattr(env, "pick_place_place_pos_w", None)
        try:
            robot = env.scene[reference_asset_cfg.name] if reference_asset_cfg is not None else None
            obj = env.scene[object_cfg.name]
            pick_support = env.scene[pick_support_cfg.name]
            place_support = env.scene[place_support_cfg.name]
            if env_ids is None or isinstance(env_ids, slice):
                first = torch.arange(env.num_envs, device=env.device)[:3]
            elif isinstance(env_ids, torch.Tensor):
                first = env_ids.to(device=env.device, dtype=torch.long).flatten()[:3]
            else:
                first = torch.tensor(env_ids, device=env.device, dtype=torch.long).flatten()[:3]
            print("[pick_place] physical reset called")
            if robot is not None:
                print("[pick_place] robot root:", robot.data.root_pos_w[first].detach().cpu().tolist())
            print("[pick_place] pick target:", pick_pos_w[first].detach().cpu().tolist())
            print("[pick_place] place target:", place_pos_w[first].detach().cpu().tolist())
            print("[pick_place] object:", obj.data.root_pos_w[first].detach().cpu().tolist())
            print("[pick_place] pick support:", pick_support.data.root_pos_w[first].detach().cpu().tolist())
            print("[pick_place] place support:", place_support.data.root_pos_w[first].detach().cpu().tolist())
        except Exception as exc:
            print(f"[pick_place] physical reset debug print failed: {exc}")
        env._pick_place_physical_reset_printed = True


def reset_robot_and_physical_pick_place_scene(
    env,
    env_ids: torch.Tensor | list[int] | None,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    pick_support_cfg: SceneEntityCfg = SceneEntityCfg("pick_support"),
    place_support_cfg: SceneEntityCfg = SceneEntityCfg("place_support"),
    pick_x_range: tuple[float, float] = (0.35, 0.75),
    pick_y_range: tuple[float, float] = (-0.25, 0.25),
    pick_z_range: tuple[float, float] = (0.24, 0.24),
    place_x_range: tuple[float, float] = (0.85, 1.25),
    place_y_range: tuple[float, float] = (-0.25, 0.25),
    place_z_range: tuple[float, float] = (0.34, 0.34),
    min_pick_place_distance: float = 0.0,
    reference_asset_cfg: SceneEntityCfg | None = None,
    targets_in_reference_yaw_frame: bool = True,
    pick_support_height: float = 0.2,
    place_support_height: float = 0.3,
    object_height: float = 0.08,
    object_xy_noise: tuple[float, float] = (0.0, 0.0),
    debug_vis: bool = False,
    debug_vis_max_envs: int = 128,
    print_debug_once: bool = True,
):
    """Reset robot root and immediately place pick/place physical objects relative to that root."""
    reset_root_state_uniform(
        env,
        env_ids,
        pose_range=pose_range,
        velocity_range=velocity_range,
        asset_cfg=asset_cfg,
    )
    if reference_asset_cfg is None:
        reference_asset_cfg = asset_cfg
    reset_physical_pick_place_scene(
        env,
        env_ids,
        object_cfg=object_cfg,
        pick_support_cfg=pick_support_cfg,
        place_support_cfg=place_support_cfg,
        pick_x_range=pick_x_range,
        pick_y_range=pick_y_range,
        pick_z_range=pick_z_range,
        place_x_range=place_x_range,
        place_y_range=place_y_range,
        place_z_range=place_z_range,
        min_pick_place_distance=min_pick_place_distance,
        reference_asset_cfg=reference_asset_cfg,
        targets_in_reference_yaw_frame=targets_in_reference_yaw_frame,
        pick_support_height=pick_support_height,
        place_support_height=place_support_height,
        object_height=object_height,
        object_xy_noise=object_xy_noise,
        debug_vis=debug_vis,
        debug_vis_max_envs=debug_vis_max_envs,
        print_debug_once=print_debug_once,
    )


def update_pick_place_phase(
    env,
    env_ids: torch.Tensor | list[int] | None,
    ee_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="arm_link6"),
    gripper_asset_cfg: SceneEntityCfg | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_object_threshold: float = 0.12,
    grasp_lift_height: float = 0.04,
    grasp_distance_threshold: float = 0.08,
    grasp_velocity_threshold: float = 0.35,
    grasp_object_speed_threshold: float | None = None,
    gripper_closed_threshold: float = 0.035,
    phase1_min_grasp_time_s: float = 0.0,
    phase1_require_stable_grasp: bool = False,
    phase2_min_hold_time_s: float = 0.1,
    phase2_require_grasp_to_advance: bool = False,
    object_place_threshold: float = 0.15,
    object_place_settle_threshold: float | None = None,
    gripper_open_threshold: float = 0.06,
    release_distance_threshold: float = 0.12,
    retreat_distance_threshold: float = 0.25,
    object_velocity_threshold: float = 0.25,
    require_object_still_for_phase4: bool = True,
    require_place_height_for_release: bool = False,
    place_height_threshold: float = 0.05,
    require_base_still_for_phase6: bool = True,
    require_ee_clear_for_phase6: bool = False,
    phase6_ee_object_min_distance: float | None = None,
    post_place_hold_time_s: float = 0.4,
    post_place_base_velocity_threshold: float = 0.08,
    post_place_yaw_velocity_threshold: float = 0.25,
):
    from .observations import ee_position_in_base, object_position_in_base, pick_position_in_base, place_position_in_base

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, dtype=torch.long, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    phase = getattr(env, "pick_place_phase", None)
    if phase is None or phase.shape != (env.num_envs,):
        phase = torch.zeros((env.num_envs,), device=env.device, dtype=torch.long)
        setattr(env, "pick_place_phase", phase)
    phase2_step_counter = getattr(env, "pick_place_phase2_step_counter", None)
    if phase2_step_counter is None or phase2_step_counter.shape != (env.num_envs,):
        phase2_step_counter = torch.zeros((env.num_envs,), device=env.device, dtype=torch.long)
        setattr(env, "pick_place_phase2_step_counter", phase2_step_counter)
    grasp_stable_counter = getattr(env, "pick_place_grasp_stable_counter", None)
    if grasp_stable_counter is None or grasp_stable_counter.shape != (env.num_envs,):
        grasp_stable_counter = torch.zeros((env.num_envs,), device=env.device, dtype=torch.long)
        setattr(env, "pick_place_grasp_stable_counter", grasp_stable_counter)
    post_place_counter = getattr(env, "pick_place_post_place_hold_counter", None)
    if post_place_counter is None or post_place_counter.shape != (env.num_envs,):
        post_place_counter = torch.zeros((env.num_envs,), device=env.device, dtype=torch.long)
        setattr(env, "pick_place_post_place_hold_counter", post_place_counter)
    post_place_condition = getattr(env, "pick_place_post_place_still_condition", None)
    if post_place_condition is None or post_place_condition.shape != (env.num_envs,):
        post_place_condition = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
        setattr(env, "pick_place_post_place_still_condition", post_place_condition)

    try:
        obj = env.scene[object_cfg.name]
        ee_b = ee_position_in_base(env, ee_asset_cfg)
        obj_b = object_position_in_base(env, object_cfg)
        place_b = place_position_in_base(env)
    except Exception:
        obj = None
        ee_b = ee_position_in_base(env, ee_asset_cfg)
        obj_b = pick_position_in_base(env)
        place_b = place_position_in_base(env)

    ee_obj_dist = torch.linalg.norm(ee_b - obj_b, dim=-1)
    obj_place_dist = torch.linalg.norm(obj_b - place_b, dim=-1)
    pick_pos_w = getattr(env, "pick_place_pick_pos_w", None)
    ids = env_ids
    if obj is None:
        ee_place_dist = torch.linalg.norm(ee_b - place_b, dim=-1)
        phase[ids] = torch.where((phase[ids] == 0) & (ee_obj_dist[ids] < ee_object_threshold), 1, phase[ids])
        phase[ids] = torch.where((phase[ids] == 1) & (ee_place_dist[ids] < object_place_threshold), 2, phase[ids])
        return

    obj_speed = torch.linalg.norm(obj.data.root_lin_vel_w, dim=-1)
    ee_asset = env.scene[ee_asset_cfg.name]
    ee_vel_w = ee_asset.data.body_lin_vel_w[:, ee_asset_cfg.body_ids].mean(dim=1)
    obj_ee_rel_speed = torch.linalg.norm(ee_vel_w - obj.data.root_lin_vel_w, dim=-1)
    object_in_gripper = (ee_obj_dist < float(grasp_distance_threshold)) & (
        obj_ee_rel_speed < float(grasp_velocity_threshold)
    )
    if pick_pos_w is None:
        lift_ok = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    else:
        lift_ok = obj.data.root_pos_w[:, 2] > pick_pos_w[:, 2] + float(grasp_lift_height)
    place_reach_threshold = float(object_place_threshold)
    place_settle_threshold = (
        place_reach_threshold if object_place_settle_threshold is None else float(object_place_settle_threshold)
    )
    placed = obj_place_dist < place_settle_threshold
    if bool(require_object_still_for_phase4):
        placed &= obj_speed < float(object_velocity_threshold)
    place_pos_w = getattr(env, "pick_place_place_pos_w", None)
    object_on_place_height = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    if place_pos_w is not None:
        place_height_error = torch.abs(obj.data.root_pos_w[:, 2] - place_pos_w[:, 2])
        object_on_place_height = place_height_error < float(place_height_threshold)
    if gripper_asset_cfg is None:
        gripper_open = ee_obj_dist > float(release_distance_threshold)
        gripper_closed = object_in_gripper
        robot = env.scene["robot"]
    else:
        robot = env.scene[gripper_asset_cfg.name]
        joint_pos = robot.data.joint_pos[:, gripper_asset_cfg.joint_ids]
        gripper_opening = torch.mean(torch.abs(joint_pos), dim=-1)
        gripper_open = gripper_opening > float(gripper_open_threshold)
        gripper_closed = gripper_opening < float(gripper_closed_threshold)
    released = placed & gripper_open
    if bool(require_place_height_for_release):
        released &= object_on_place_height
    retreated = released & (ee_obj_dist > float(retreat_distance_threshold))
    grasped_and_lifted = lift_ok & object_in_gripper & gripper_closed
    if grasp_object_speed_threshold is not None:
        grasped_and_lifted &= obj_speed < float(grasp_object_speed_threshold)
    grasp_stable_counter[grasped_and_lifted] += 1
    grasp_stable_counter[~grasped_and_lifted] = 0
    required_grasp_stable_steps = max(1, int(float(phase1_min_grasp_time_s) / float(env.step_dt)))
    stable_grasped_and_lifted = grasped_and_lifted & (grasp_stable_counter >= required_grasp_stable_steps)
    phase2_condition = getattr(env, "pick_place_phase2_success_condition", None)
    if phase2_condition is not None and phase2_condition.shape == (env.num_envs,):
        phase2_condition[:] = stable_grasped_and_lifted
    base_planar_speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
    base_yaw_speed = torch.abs(robot.data.root_ang_vel_b[:, 2])
    post_place_still = released & (obj_speed < float(object_velocity_threshold))
    if bool(require_base_still_for_phase6):
        post_place_still &= (base_planar_speed < float(post_place_base_velocity_threshold)) & (
            base_yaw_speed < float(post_place_yaw_velocity_threshold)
        )
    if bool(require_ee_clear_for_phase6):
        min_ee_object_dist = (
            float(release_distance_threshold)
            if phase6_ee_object_min_distance is None
            else float(phase6_ee_object_min_distance)
        )
        post_place_still &= ee_obj_dist > min_ee_object_dist

    phase_start = phase.clone()
    required_phase2_steps = max(1, int(float(phase2_min_hold_time_s) / float(env.step_dt)))
    phase2_can_advance = phase2_step_counter >= required_phase2_steps
    phase2_entry_condition = stable_grasped_and_lifted if bool(phase1_require_stable_grasp) else grasped_and_lifted
    phase[ids] = torch.where((phase_start[ids] == 0) & (ee_obj_dist[ids] < ee_object_threshold), 1, phase[ids])
    phase[ids] = torch.where((phase_start[ids] == 1) & phase2_entry_condition[ids], 2, phase[ids])
    phase2_advance_condition = phase2_can_advance & (obj_place_dist < place_reach_threshold)
    if bool(phase2_require_grasp_to_advance):
        phase2_advance_condition &= stable_grasped_and_lifted
    phase[ids] = torch.where(
        (phase_start[ids] == 2) & phase2_advance_condition[ids],
        3,
        phase[ids],
    )
    phase[ids] = torch.where((phase_start[ids] == 3) & placed[ids], 4, phase[ids])
    phase[ids] = torch.where((phase_start[ids] == 4) & released[ids], 5, phase[ids])
    phase[ids] = torch.where((phase_start[ids] == 5) & retreated[ids], 5, phase[ids])
    phase2_step_counter[phase == 2] += 1
    phase2_step_counter[phase != 2] = 0

    phase5_still = (phase == 5) & post_place_still
    phase5_not_still = (phase == 5) & (~post_place_still)
    update_ids = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    update_ids[ids] = True
    post_place_counter[phase5_still & update_ids] += 1
    post_place_counter[(phase5_not_still | (phase != 5)) & update_ids] = 0
    post_place_condition[:] = post_place_still
    required_steps = max(1, int(float(post_place_hold_time_s) / float(env.step_dt)))
    phase[ids] = torch.where((phase[ids] == 5) & (post_place_counter[ids] >= required_steps), 6, phase[ids])


def create_sample_pick_place_targets_event(**kwargs) -> EventTerm:
    return EventTerm(func=sample_pick_place_targets, mode="reset", params=kwargs)


def create_reset_object_to_pick_event(**kwargs) -> EventTerm:
    return EventTerm(func=reset_object_to_pick, mode="reset", params=kwargs)


def create_update_pick_place_phase_event(interval_range_s: tuple[float, float] = (0.02, 0.02), **kwargs) -> EventTerm:
    return EventTerm(
        func=update_pick_place_phase,
        mode="interval",
        interval_range_s=interval_range_s,
        params=kwargs,
    )

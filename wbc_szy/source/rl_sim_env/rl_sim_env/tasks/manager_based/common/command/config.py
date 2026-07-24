from rl_sim_env.tasks.manager_based.common.mdp import (
    UniformVelocityCommandCfg,
    UniformVelocityAndHeightCommandCfg,
    UniformVelocityAndOrientationFlagCommandCfg,
    UniformVelocityCommandTerrainCfg,
    UniformVelocityAndHeightCommandTerrainCfg,
    UniformPitchCommandTerrainCfg,
)


def create_uniform_velocity_and_height_command_terrain_cfg(
    command_ids: dict[str, list[int]],
    ranges: dict[str, UniformVelocityAndHeightCommandTerrainCfg.Ranges],
    lin_x_level: float,
    ang_z_level: float,
    max_lin_x_level: float,
    max_ang_z_level: float,
    heading_control_stiffness: float,
) -> UniformVelocityAndHeightCommandTerrainCfg:
    base_command = UniformVelocityAndHeightCommandTerrainCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        command_ids=command_ids,
        ranges=ranges,
        lin_x_level=lin_x_level,
        ang_z_level=ang_z_level,
        max_lin_x_level=max_lin_x_level,
        max_ang_z_level=max_ang_z_level,
        heading_control_stiffness=heading_control_stiffness,
    )
    return base_command


def create_uniform_velocity_command_terrain_cfg(
    command_ids: dict[str, list[int]],
    ranges: dict[str, UniformVelocityCommandTerrainCfg.Ranges],
    lin_x_level: float,
    ang_z_level: float,
    max_lin_x_level: float,
    max_ang_z_level: float,
    heading_control_stiffness: float,
    vel_curriculum_episode_mult: float = 8.0,
    split_xy_velocity_metrics: bool = False,
) -> UniformVelocityCommandTerrainCfg:
    base_velocity = UniformVelocityCommandTerrainCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        command_ids=command_ids,
        ranges=ranges,
        lin_x_level=lin_x_level,
        ang_z_level=ang_z_level,
        max_lin_x_level=max_lin_x_level,
        max_ang_z_level=max_ang_z_level,
        vel_curriculum_episode_mult=vel_curriculum_episode_mult,
        split_xy_velocity_metrics=split_xy_velocity_metrics,
        heading_control_stiffness=heading_control_stiffness,
    )

    return base_velocity


def create_uniform_pitch_command_terrain_cfg(
    command_ids: dict[str, list[int]],
    pitch_ranges: dict[str, tuple[float, float]],
    resampling_time_range: tuple[float, float],
    ramp_time_s: float,
) -> UniformPitchCommandTerrainCfg:
    """Create terrain-aware pitch command config.

    This is a per-terrain sampling range config.
    """
    return UniformPitchCommandTerrainCfg(
        resampling_time_range=resampling_time_range,
        command_ids=command_ids,
        pitch_ranges=pitch_ranges,
        ramp_time_s=ramp_time_s,
    )


def create_uniform_velocity_command_cfg(
    rel_standing_envs: float,
    rel_heading_envs: float,
    heading_command: bool,
    heading_control_stiffness: float,
    lin_vel_x: tuple[float, float],
    lin_vel_y: tuple[float, float],
    ang_vel_z: tuple[float, float],
    heading: tuple[float, float],
) -> UniformVelocityCommandCfg:
    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=rel_standing_envs,
        rel_heading_envs=rel_heading_envs,
        heading_command=heading_command,
        heading_control_stiffness=heading_control_stiffness,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=lin_vel_x,
            lin_vel_y=lin_vel_y,
            ang_vel_z=ang_vel_z,
            heading=heading,
        ),
    )

    return base_velocity


def create_uniform_velocity_and_height_command_cfg(
    rel_standing_envs: float,
    rel_heading_envs: float,
    heading_command: bool,
    heading_control_stiffness: float,
    lin_vel_x: tuple[float, float],
    lin_vel_y: tuple[float, float],
    ang_vel_z: tuple[float, float],
    heading: tuple[float, float],
    base_height_cmd: tuple[float, float],
    fixed_height_cmd: float,
    probability_of_using_fixed_height_cmd: float,
) -> UniformVelocityAndHeightCommandCfg:
    base_command = UniformVelocityAndHeightCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=rel_standing_envs,
        rel_heading_envs=rel_heading_envs,
        heading_command=heading_command,
        heading_control_stiffness=heading_control_stiffness,
        debug_vis=False,
        ranges=UniformVelocityAndHeightCommandCfg.Ranges(
            lin_vel_x=lin_vel_x,
            lin_vel_y=lin_vel_y,
            ang_vel_z=ang_vel_z,
            heading=heading,
            base_height_cmd=base_height_cmd,
            fixed_height_cmd=fixed_height_cmd,
            probability_of_using_fixed_height_cmd=(
                probability_of_using_fixed_height_cmd
            ),
        ),
    )

    return base_command


def create_uniform_velocity_and_orientation_flag_command_cfg(
    rel_standing_envs: float,
    rel_heading_envs: float,
    heading_command: bool,
    heading_control_stiffness: float,
    lin_vel_x: tuple[float, float],
    lin_vel_y: tuple[float, float],
    ang_vel_z: tuple[float, float],
    heading: tuple[float, float],
    orientation_flags: list[int],
) -> UniformVelocityAndOrientationFlagCommandCfg:
    base_command = UniformVelocityAndOrientationFlagCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=rel_standing_envs,
        rel_heading_envs=rel_heading_envs,
        heading_command=heading_command,
        heading_control_stiffness=heading_control_stiffness,
        debug_vis=False,
        ranges=UniformVelocityAndOrientationFlagCommandCfg.Ranges(
            lin_vel_x=lin_vel_x,
            lin_vel_y=lin_vel_y,
            ang_vel_z=ang_vel_z,
            heading=heading,
            orientation_flags=orientation_flags,
        ),
    )
    return base_command

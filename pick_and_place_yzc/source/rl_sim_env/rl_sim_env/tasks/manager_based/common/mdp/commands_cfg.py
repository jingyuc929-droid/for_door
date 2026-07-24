# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from dataclasses import MISSING
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from .commands import (
    UniformVelocityCommandTerrain,
    UniformVelocityAndHeightCommandTerrain,
    UniformVelocityAndHeightCommand,
    UniformVelocityAndOrientationFlagCommand,
    UniformPitchCommand,
    UniformPitchCommandTerrain,
) 

from isaaclab.envs.mdp.commands import (
    UniformVelocityCommandCfg,
)


@configclass
class UniformVelocityCommandTerrainCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = UniformVelocityCommandTerrain

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    command_ids: dict[str, list[int]] = MISSING
    """The command ids for the different terrain parts."""

    lin_x_level: float = 0.0
    max_lin_x_level: float = 1.0
    ang_z_level: float = 0.0
    max_ang_z_level: float = 1.5

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

        heading_command_prob: float = 0.0
        """Probability of using heading command. Defaults to 0.0."""

        yaw_command_prob: float = 0.0
        """Probability of using yaw command. Defaults to 0.0."""

        standing_command_prob: float = 0.0
        """Probability of using standing command. Defaults to 0.0."""

        heading: tuple[float, float] = MISSING
        """Range for the heading command (in rad). Defaults to (0.0, 0.0).

        This parameter is only used if :attr:`~UniformVelocityCommandCfg.heading_command_prob` is greater than 0.0.
        """

        start_curriculum_lin_x: tuple[float, float] = MISSING
        start_curriculum_ang_z: tuple[float, float] = MISSING
        max_curriculum_lin_x: tuple[float, float] = MISSING
        max_curriculum_ang_z: tuple[float, float] = MISSING

    ranges: dict[str, Ranges] = MISSING
    """Distribution ranges for the velocity commands."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)


@configclass
class UniformVelocityAndHeightCommandTerrainCfg(UniformVelocityCommandTerrainCfg):
    """Configuration for uniform height command generator."""

    class_type: type = UniformVelocityAndHeightCommandTerrain

    @configclass
    class Ranges(UniformVelocityCommandTerrainCfg.Ranges):
        """Uniform distribution ranges for the height commands."""
        base_height_cmd: tuple[float, float] = MISSING
        "Range for the base height command (in m)."

        fixed_height_cmd: float = MISSING
        "Fixed height command (in m)."

        probability_of_using_fixed_height_cmd: float = 1.0
        "Probability of using fixed height command. Defaults to 1.0."

    ranges: dict[str, Ranges] = MISSING
    """Distribution ranges for the velocity and height commands."""


@configclass
class UniformVelocityAndHeightCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the uniform velocity and height command generator."""

    class_type: type = UniformVelocityAndHeightCommand

    @configclass
    class Ranges(UniformVelocityCommandCfg.Ranges):
        """Uniform distribution ranges for the velocity and height commands."""
        base_height_cmd: tuple[float, float] = MISSING
        """Range for the base height command (in m)."""

        fixed_height_cmd: float = MISSING
        """Fixed height command (in m)."""

        probability_of_using_fixed_height_cmd: float = 1.0
        """Probability of using fixed height command. Defaults to 1.0."""
    ranges: Ranges = MISSING
    """Distribution ranges for the velocity and height commands."""


@configclass
class UniformVelocityAndOrientationFlagCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the uniform velocity and height command generator."""

    class_type: type = UniformVelocityAndOrientationFlagCommand

    @configclass
    class Ranges(UniformVelocityCommandCfg.Ranges):
        """Uniform distribution ranges for the velocity and orientation flag commands."""
        orientation_flags: list[int] = MISSING
        """Orientation flag value. Must be one of: -1, 0, 1."""
    ranges: Ranges = MISSING
    """Distribution ranges for the velocity and orientation flag commands."""


@configclass
class UniformPitchCommandCfg(CommandTermCfg):
    """Configuration for the uniform pitch command generator."""

    class_type: type = UniformPitchCommand

    pitch_range: tuple[float, float] = MISSING
    """Range for the pitch command (in rad)."""

    ramp_time_s: float = 0.5
    """Time to linearly ramp from the previous pitch command to the newly sampled one [s]."""


@configclass
class UniformPitchCommandTerrainCfg(CommandTermCfg):
    """Configuration for the uniform pitch command generator with per-terrain ranges.

    This mirrors the mechanism used by `UniformVelocityCommandTerrainCfg`:
    - Use `command_ids` to map each terrain key to the list of env_ids.
    - Use `pitch_ranges` to specify the pitch sampling range per terrain key.
    """

    class_type: type = UniformPitchCommandTerrain

    command_ids: dict[str, list[int]] = MISSING
    """The command ids for the different terrain parts."""

    pitch_ranges: dict[str, tuple[float, float]] = MISSING
    """Pitch sampling ranges per terrain key (in rad)."""

    ramp_time_s: float = 0.5
    """Time to linearly ramp from the previous pitch command to the newly sampled one [s]."""

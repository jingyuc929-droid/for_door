"""Gym registrations for GRQ20 V2D4 PiperL gripper physical pick-and-place."""

import gymnasium as gym


gym.register(
    id="Locomotion-GRQ20-V2D4-PiperL-PickPlace-GripperObject-High-VAE",
    entry_point="rl_sim_env.envs:LocomotionRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_env_cfg:LocomotionPiperLPickPlaceEnvCfg",
        "locomotion_cfg_entry_point": f"{__name__}.config_summary:LocomotionPiperLPickPlaceRunnerCfg",
    },
)


gym.register(
    id="Locomotion-GRQ20-V2D4-PiperL-PickPlace-GripperObject-High-VAE-Play",
    entry_point="rl_sim_env.envs:LocomotionRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_env_cfg:LocomotionPiperLPickPlaceEnvCfg_PLAY",
        "locomotion_cfg_entry_point": f"{__name__}.config_summary:LocomotionPiperLPickPlaceRunnerCfg",
    },
)


gym.register(
    id="Locomotion-GRQ20-V2D4-PiperL-PickPlace-GripperObject-High-VAE-DebugPlay-v0",
    entry_point="rl_sim_env.envs:LocomotionRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_env_cfg:LocomotionPiperLPickPlaceEnvCfg_DEBUG_PLAY",
        "locomotion_cfg_entry_point": f"{__name__}.config_summary:LocomotionPiperLPickPlaceRunnerCfg",
    },
)

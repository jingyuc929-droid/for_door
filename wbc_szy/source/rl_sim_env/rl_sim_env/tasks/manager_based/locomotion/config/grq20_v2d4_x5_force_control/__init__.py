# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Locomotion-GRQ20-V2D4-X5-Force-Control-VAE",
    entry_point="rl_sim_env.envs:LocomotionRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            # 与 grq20_v2d4_default_force_control 风格对齐：统一入口命名
            f"{__name__}.locomotion_env_cfg:LocomotionVaeEnvCfg"
        ),
        "locomotion_cfg_entry_point": (
            f"{__name__}.config_summary:LocomotionPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Locomotion-GRQ20-V2D4-X5-Force-Control-VAE-Play",
    entry_point="rl_sim_env.envs:LocomotionRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.locomotion_env_cfg:LocomotionVaeEnvCfg_PLAY"
        ),
        "locomotion_cfg_entry_point": (
            f"{__name__}.config_summary:LocomotionPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Locomotion-GRQ20-V2D4-X5-Force-Control-ReplayAmpData",
    entry_point="rl_sim_env.envs:LocomotionRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.locomotion_env_cfg:"
            "LocomotionVaeEnvCfg_REPLAY_AMPDATA"
        ),
        "locomotion_cfg_entry_point": (
            f"{__name__}.config_summary:LocomotionPPORunnerCfg"
        ),
    },
)

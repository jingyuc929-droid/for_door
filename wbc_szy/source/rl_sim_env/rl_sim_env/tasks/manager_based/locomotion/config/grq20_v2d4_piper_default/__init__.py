# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Locomotion-GRQ20-V2D4-PIPER-VAE",
    entry_point="rl_sim_env.envs:LocomotionRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.locomotion_env_cfg:LocomotionWholeBodyVaeEnvCfg"
        ),
        "locomotion_cfg_entry_point": (
            f"{__name__}.config_summary:LocomotionWholeBodyPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Locomotion-GRQ20-V2D4-PIPER-VAE-Play",
    entry_point="rl_sim_env.envs:LocomotionRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.locomotion_env_cfg:LocomotionWholeBodyVaeEnvCfg_PLAY"
        ),
        "locomotion_cfg_entry_point": (
            f"{__name__}.config_summary:LocomotionWholeBodyPPORunnerCfg"
        ),
    },
)

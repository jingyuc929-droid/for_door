# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Locomotion-GRQ20-V2D4-PiperL-LocoManip-Lower-VAE",
    entry_point="rl_sim_env.envs:LocomotionRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_env_cfg:LocomotionPiperLLowerBodyEnvCfg",
        "locomotion_cfg_entry_point": f"{__name__}.config_summary:LocomotionPiperLLowerBodyPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-GRQ20-V2D4-PiperL-LocoManip-Lower-VAE-Play",
    entry_point="rl_sim_env.envs:LocomotionRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_env_cfg:LocomotionPiperLLowerBodyEnvCfg_PLAY",
        "locomotion_cfg_entry_point": f"{__name__}.config_summary:LocomotionPiperLLowerBodyPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-GRQ20-V2D4-PiperL-LocoManip-Lower-VAE-Play-OnnxPlay-v0",
    entry_point="rl_sim_env.envs:ParkourRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_env_cfg:LocomotionPiperLLowerBodyEnvCfg_ONNX_PLAY",
        "locomotion_cfg_entry_point": f"{__name__}.config_summary:LocomotionPiperLLowerBodyPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-GRQ20-V2D4-PiperL-LocoManip-Lower-VAE-Play-DebugPlay-v0",
    entry_point="rl_sim_env.envs:ParkourRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_env_cfg:LocomotionPiperLLowerBodyEnvCfg_DEBUG_PLAY",
        "locomotion_cfg_entry_point": f"{__name__}.config_summary:LocomotionPiperLLowerBodyPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-GRQ20-V2D4-PiperL-LocoManip-Lower-ReplayAmpData",
    entry_point="rl_sim_env.envs:LocomotionRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_env_cfg:LocomotionPiperLLowerBodyEnvCfg_REPLAY_AMPDATA",
        "locomotion_cfg_entry_point": f"{__name__}.config_summary:LocomotionPiperLLowerBodyPPORunnerCfg",
    },
)

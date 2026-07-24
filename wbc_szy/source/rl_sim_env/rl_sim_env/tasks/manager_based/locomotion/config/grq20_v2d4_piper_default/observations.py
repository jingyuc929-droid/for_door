"""Default-task-specific Piper latency and privileged observations."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from rl_sim_env.tasks.manager_based.common import mdp
from rl_sim_env.tasks.manager_based.common.sensor.piper_observation_noise import (
    ARM_JOINT_POS_EPISODE_BIAS_ABS,
    ARM_JOINT_POS_STD,
    ARM_JOINT_VEL_EPISODE_BIAS_ABS,
    ARM_JOINT_VEL_STD,
    DeferredVectorBiasNoiseModelCfg,
    IMU_GYRO_EPISODE_BIAS_ABS,
    IMU_GYRO_STD,
    IMU_PROJECTED_GRAVITY_EPISODE_BIAS_ABS,
    IMU_PROJECTED_GRAVITY_STD,
    LEG_JOINT_POS_EPISODE_BIAS_ABS,
    LEG_JOINT_POS_STD,
    LEG_JOINT_VEL_EPISODE_BIAS_ABS,
    LEG_JOINT_VEL_STD,
    PIPER_ARM_JOINT_NAMES,
    PIPER_LEG_JOINT_NAMES,
    SymmetricUniformNoiseCfg,
    TruncatedGaussianNoiseCfg,
)


BODY_PROPRIOCEPTION_TERM = "body_proprioception_nad"
ARM_PROPRIOCEPTION_TERM = "arm_proprioception_nad"
JOINT_VEL_SCALE = 0.05
BASE_ANG_VEL_SCALE = 0.25


def relative_body_mass(
    env,
    asset_cfg: SceneEntityCfg,
    cache_static: bool = False,
) -> torch.Tensor:
    """Return per-body mass ratios relative to the nominal model, minus one."""
    cache = None
    cache_key = ("piper_default_relative_mass", asset_cfg.name, repr(asset_cfg.body_ids))
    if cache_static:
        cache = getattr(env, "_static_observation_cache", None)
        if cache is None:
            cache = {}
            env._static_observation_cache = cache
        if cache_key in cache:
            return cache[cache_key]

    asset: Articulation = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses()[:, asset_cfg.body_ids].to(env.device)
    default_masses = asset.data.default_mass[:, asset_cfg.body_ids].to(env.device)
    relative_mass = masses / default_masses - 1.0
    if cache is not None:
        cache[cache_key] = relative_mass
    return relative_mass


def external_force_channel(env, channel_name: str) -> torch.Tensor:
    """Return the currently applied force in its configured sampling frame."""
    channels = getattr(env, "external_force_channels", None)
    if not channels or channel_name not in channels:
        return torch.zeros((env.num_envs, 3), device=env.device)
    return channels[channel_name]["current"]


def body_proprioception(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return one latency-coupled body stream: IMU plus the 12 leg motors."""
    return torch.cat(
        (
            mdp.base_ang_vel(env) * BASE_ANG_VEL_SCALE,
            mdp.projected_gravity(env),
            mdp.joint_pos_rel(env, asset_cfg),
            mdp.joint_vel_rel(env, asset_cfg) * JOINT_VEL_SCALE,
        ),
        dim=-1,
    )


def arm_proprioception(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return one latency-coupled stream for all six arm motors."""
    return torch.cat(
        (
            mdp.joint_pos_rel(env, asset_cfg),
            mdp.joint_vel_rel(env, asset_cfg) * JOINT_VEL_SCALE,
        ),
        dim=-1,
    )


def observation_lag(env) -> torch.Tensor:
    """Return the sampled body and arm observation delays, in control steps."""
    term_names = (BODY_PROPRIOCEPTION_TERM, ARM_PROPRIOCEPTION_TERM)
    samples = getattr(env, "obs_delay_sample_dict", None)
    if samples is None or any(name not in samples for name in term_names):
        # Observation dimensions are inspected before the delay buffers are built.
        return torch.zeros((env.num_envs, len(term_names)), device=env.device)
    return torch.stack([samples[name] for name in term_names], dim=-1).to(
        device=env.device, dtype=torch.float32
    )


def actuator_lag(env) -> torch.Tensor:
    """Return independently sampled body and arm actuator delays."""
    actuators = env.scene.articulations["robot"].actuators
    body_lag = actuators["base_legs"].positions_delay_buffer.time_lags
    arm_lag = actuators["arm"].positions_delay_buffer.time_lags
    return torch.stack((body_lag, arm_lag), dim=-1).to(
        device=env.device, dtype=torch.float32
    )


def _temporal_sensor_noise(
    std: Sequence[float],
    bias_abs: Sequence[float],
) -> DeferredVectorBiasNoiseModelCfg:
    return DeferredVectorBiasNoiseModelCfg(
        noise_cfg=TruncatedGaussianNoiseCfg(
            mean=0.0,
            std=list(std),
            clip_sigma=3.0,
        ),
        bias_noise_cfg=SymmetricUniformNoiseCfg(
            half_width=list(bias_abs),
            operation="abs",
        ),
        sample_bias_per_component=True,
    )


def create_default_noise_and_delay_obs_cfg(
    leg_joint_names: Sequence[str],
    arm_joint_names: Sequence[str],
) -> dict[str, dict]:
    """Create one delayed body stream and one delayed arm stream."""
    if tuple(leg_joint_names) != PIPER_LEG_JOINT_NAMES:
        raise ValueError("Piper leg-joint order does not match the noise calibration.")
    if tuple(arm_joint_names) != PIPER_ARM_JOINT_NAMES:
        raise ValueError("Piper arm-joint order does not match the noise calibration.")

    body_std = (
        tuple(value * BASE_ANG_VEL_SCALE for value in IMU_GYRO_STD)
        + IMU_PROJECTED_GRAVITY_STD
        + LEG_JOINT_POS_STD
        + tuple(value * JOINT_VEL_SCALE for value in LEG_JOINT_VEL_STD)
    )
    body_bias = (
        tuple(value * BASE_ANG_VEL_SCALE for value in IMU_GYRO_EPISODE_BIAS_ABS)
        + IMU_PROJECTED_GRAVITY_EPISODE_BIAS_ABS
        + LEG_JOINT_POS_EPISODE_BIAS_ABS
        + tuple(value * JOINT_VEL_SCALE for value in LEG_JOINT_VEL_EPISODE_BIAS_ABS)
    )
    arm_std = ARM_JOINT_POS_STD + tuple(
        value * JOINT_VEL_SCALE for value in ARM_JOINT_VEL_STD
    )
    arm_bias = ARM_JOINT_POS_EPISODE_BIAS_ABS + tuple(
        value * JOINT_VEL_SCALE for value in ARM_JOINT_VEL_EPISODE_BIAS_ABS
    )

    return {
        BODY_PROPRIOCEPTION_TERM: {
            "noise": _temporal_sensor_noise(body_std, body_bias),
            "delay": 2,
            "params": {
                "asset_cfg": SceneEntityCfg("robot", joint_names=list(leg_joint_names)),
            },
        },
        ARM_PROPRIOCEPTION_TERM: {
            "noise": _temporal_sensor_noise(arm_std, arm_bias),
            "delay": 2,
            "params": {
                "asset_cfg": SceneEntityCfg("robot", joint_names=list(arm_joint_names)),
            },
        },
    }

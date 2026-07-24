"""Calibrated observation-noise configuration shared by Piper tasks."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg, NoiseModelWithAdditiveBiasCfg
from isaaclab.utils.noise.noise_model import NoiseModelWithAdditiveBias


PIPER_LEG_JOINT_NAMES = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
)
PIPER_ARM_JOINT_NAMES = (
    "link1_joint", "link2_joint", "link3_joint",
    "link4_joint", "link5_joint", "link6_joint",
)

# The 2026-07-17 11-27-48 recording was collected with a policy trained under
# this robust envelope.  Its dynamic residuals are useful lower-bound evidence
# but cannot justify shrinking the training perturbation after the fact.
# Preserve the envelope that produced the improved real-robot behavior.
# Values are raw sensor units, before observation scaling.
IMU_GYRO_STD = (0.03, 0.03, 0.005)
IMU_GYRO_EPISODE_BIAS_ABS = (0.002, 0.002, 0.002)
IMU_PROJECTED_GRAVITY_STD = (0.0005, 0.0005, 0.0005)
IMU_PROJECTED_GRAVITY_EPISODE_BIAS_ABS = (0.0005, 0.0005, 0.0005)
LEG_JOINT_POS_STD = (0.001,) * len(PIPER_LEG_JOINT_NAMES)
ARM_JOINT_POS_STD = (0.01,) * len(PIPER_ARM_JOINT_NAMES)
LEG_JOINT_POS_EPISODE_BIAS_ABS = (0.01,) * len(PIPER_LEG_JOINT_NAMES)
ARM_JOINT_POS_EPISODE_BIAS_ABS = (0.1,) * len(PIPER_ARM_JOINT_NAMES)
LEG_JOINT_VEL_STD = (0.20,) * len(PIPER_LEG_JOINT_NAMES)
ARM_JOINT_VEL_STD = (1.5,) * len(PIPER_ARM_JOINT_NAMES)
LEG_JOINT_VEL_EPISODE_BIAS_ABS = (0.005,) * len(PIPER_LEG_JOINT_NAMES)
ARM_JOINT_VEL_EPISODE_BIAS_ABS = (0.1,) * len(PIPER_ARM_JOINT_NAMES)


def _parameter_on_data_device(cfg, name: str, data: torch.Tensor):
    value = getattr(cfg, name)
    if isinstance(value, Sequence):
        value = torch.as_tensor(value, device=data.device, dtype=data.dtype)
        setattr(cfg, name, value)
    elif isinstance(value, torch.Tensor) and (value.device != data.device or value.dtype != data.dtype):
        value = value.to(device=data.device, dtype=data.dtype)
        setattr(cfg, name, value)
    return value


def truncated_gaussian_noise(data: torch.Tensor, cfg: "TruncatedGaussianNoiseCfg") -> torch.Tensor:
    """Apply a true truncated Gaussian, avoiding clamp point masses."""
    mean = _parameter_on_data_device(cfg, "mean", data)
    std = _parameter_on_data_device(cfg, "std", data)
    cdf_low = 0.5 * (1.0 - math.erf(cfg.clip_sigma / math.sqrt(2.0)))
    uniform = torch.rand_like(data).mul_(1.0 - 2.0 * cdf_low).add_(cdf_low)
    eps = torch.finfo(data.dtype).eps
    standard_noise = math.sqrt(2.0) * torch.erfinv(
        (2.0 * uniform - 1.0).clamp(-1.0 + eps, 1.0 - eps)
    )
    sampled_noise = mean + std * standard_noise
    if cfg.operation == "add":
        return data + sampled_noise
    if cfg.operation == "scale":
        return data * sampled_noise
    if cfg.operation == "abs":
        return sampled_noise
    raise ValueError(f"Unknown operation in noise: {cfg.operation}")


def symmetric_uniform_noise(data: torch.Tensor, cfg: "SymmetricUniformNoiseCfg") -> torch.Tensor:
    """Apply zero-centered uniform noise with scalar or vector half-widths."""
    half_width = _parameter_on_data_device(cfg, "half_width", data)
    sampled_noise = (2.0 * torch.rand_like(data) - 1.0) * half_width
    if cfg.operation == "add":
        return data + sampled_noise
    if cfg.operation == "scale":
        return data * sampled_noise
    if cfg.operation == "abs":
        return sampled_noise
    raise ValueError(f"Unknown operation in noise: {cfg.operation}")


class DeferredVectorBiasNoiseModel(NoiseModelWithAdditiveBias):
    """Delay vector-bias sampling until the observation width is known."""

    def reset(self, env_ids: Sequence[int] | None = None):
        if self._sample_bias_per_component and self._num_components is None:
            return
        super().reset(env_ids)


@configclass
class TruncatedGaussianNoiseCfg(NoiseCfg):
    func = truncated_gaussian_noise
    mean: torch.Tensor | Sequence[float] | float = 0.0
    std: torch.Tensor | Sequence[float] | float = 1.0
    clip_sigma: float = 3.0

    def __post_init__(self):
        if self.clip_sigma <= 0.0:
            raise ValueError(f"clip_sigma must be positive, got {self.clip_sigma}.")
        if isinstance(self.std, (float, int)) and self.std < 0.0:
            raise ValueError("std must be non-negative.")
        if isinstance(self.std, Sequence) and any(value < 0.0 for value in self.std):
            raise ValueError("std must be non-negative.")


@configclass
class SymmetricUniformNoiseCfg(NoiseCfg):
    func = symmetric_uniform_noise
    half_width: torch.Tensor | Sequence[float] | float = 1.0

    def __post_init__(self):
        if isinstance(self.half_width, (float, int)) and self.half_width < 0.0:
            raise ValueError("half_width must be non-negative.")
        if isinstance(self.half_width, Sequence) and any(value < 0.0 for value in self.half_width):
            raise ValueError("half_width must be non-negative.")


@configclass
class DeferredVectorBiasNoiseModelCfg(NoiseModelWithAdditiveBiasCfg):
    class_type = DeferredVectorBiasNoiseModel


def _noise_vector(values: tuple[float, ...]) -> list[float]:
    return list(values)


def _leg_arm_noise_vector(leg_values, arm_values) -> list[float]:
    return _noise_vector(leg_values + arm_values)


def _temporal_sensor_noise(std: list[float], bias_abs: list[float]):
    return DeferredVectorBiasNoiseModelCfg(
        noise_cfg=TruncatedGaussianNoiseCfg(mean=0.0, std=std, clip_sigma=3.0),
        bias_noise_cfg=SymmetricUniformNoiseCfg(half_width=bias_abs, operation="abs"),
        sample_bias_per_component=True,
    )


def create_piper_noise_and_delay_obs_cfg(
    leg_joint_names: Sequence[str], arm_joint_names: Sequence[str]
) -> dict[str, dict]:
    """Create the calibrated NAD terms in their validated joint order."""
    if tuple(leg_joint_names) != PIPER_LEG_JOINT_NAMES:
        raise ValueError("Piper leg-joint order does not match the noise calibration.")
    if tuple(arm_joint_names) != PIPER_ARM_JOINT_NAMES:
        raise ValueError("Piper arm-joint order does not match the noise calibration.")

    whole_body_joint_names = list(leg_joint_names) + list(arm_joint_names)
    return {
        "base_ang_vel_nad": {
            "scale": 0.25,
            "noise": _temporal_sensor_noise(
                _noise_vector(IMU_GYRO_STD), _noise_vector(IMU_GYRO_EPISODE_BIAS_ABS)
            ),
            "delay": 2,
        },
        "projected_gravity_nad": {
            "noise": _temporal_sensor_noise(
                _noise_vector(IMU_PROJECTED_GRAVITY_STD),
                _noise_vector(IMU_PROJECTED_GRAVITY_EPISODE_BIAS_ABS),
            ),
            "delay": 2,
        },
        "joint_pos_rel_nad": {
            "noise": _temporal_sensor_noise(
                _leg_arm_noise_vector(LEG_JOINT_POS_STD, ARM_JOINT_POS_STD),
                _leg_arm_noise_vector(
                    LEG_JOINT_POS_EPISODE_BIAS_ABS, ARM_JOINT_POS_EPISODE_BIAS_ABS
                ),
            ),
            "delay": 1,
            "params": {"asset_cfg": SceneEntityCfg("robot", joint_names=whole_body_joint_names)},
        },
        "joint_vel_rel_nad": {
            "scale": 0.05,
            "noise": _temporal_sensor_noise(
                _leg_arm_noise_vector(LEG_JOINT_VEL_STD, ARM_JOINT_VEL_STD),
                _leg_arm_noise_vector(
                    LEG_JOINT_VEL_EPISODE_BIAS_ABS, ARM_JOINT_VEL_EPISODE_BIAS_ABS
                ),
            ),
            "delay": 2,
            "params": {"asset_cfg": SceneEntityCfg("robot", joint_names=whole_body_joint_names)},
        },
    }

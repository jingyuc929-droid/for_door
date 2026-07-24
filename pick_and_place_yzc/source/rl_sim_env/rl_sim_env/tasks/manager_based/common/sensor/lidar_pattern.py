
import torch
from typing import Callable
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg


def lidar_dynamic_pattern(cfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    ray_directions = torch.zeros(cfg.ray_num, 3, device=device)
    ray_directions[..., 0] = 1.0
    ray_starts = torch.zeros_like(ray_directions, device=device)
    return ray_starts, ray_directions


@configclass
class LidarDynamicPatternCfg(PatternBaseCfg):
    """Configuration for the LiDAR pattern for ray-casting."""

    func: Callable = lidar_dynamic_pattern

    ray_num: int = MISSING
    """Number of rays. """

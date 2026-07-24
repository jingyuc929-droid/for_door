# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

# from .amp_vae_perception_ppo import AMPVAEPerceptionPPO
# from .amp_vae_ppo import AMPVAEPPO
# from .amp_vae_vit_ppo import AMPVAEVITPPO
# from .distillation import Distillation
from .locomotion_ppo import LocomotionPPO   

__all__ = ["LocomotionPPO"]

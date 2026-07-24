# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rl_algorithms.rsl_rl.networks import MLP, EmpiricalNormalization


class ActorCriticEncoder(nn.Module):
    def __init__(
        self,
        module_cfg_dict
    ):
        actor_cfg = module_cfg_dict['actor']
        privileged_encoder_cfg = module_cfg_dict['privileged_encoder']
        heightmap_encoder_cfg = module_cfg_dict['heightmap_encoder']
        critic_cfg = module_cfg_dict['critic']
        noise_std_type = module_cfg_dict['noise_std_type']
        init_noise_std = module_cfg_dict['init_noise_std']
        min_std = module_cfg_dict['min_normalized_std']
        super().__init__()

        # actor
        self.actor = MLP(actor_cfg['num_actor_obs'],
                         actor_cfg['num_actions'],
                         actor_cfg['actor_hidden_dims'],
                         module_cfg_dict['activation'])
        # actor observation normalization
        self.actor_obs_normalization = actor_cfg['actor_obs_normalization']
        if actor_cfg['actor_obs_normalization']:
            self.actor_obs_normalizer = EmpiricalNormalization(actor_cfg['num_actor_obs'])
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        print(f"Actor MLP: {self.actor}")

        # privileged encoder
        self.privileged_encoder = MLP(privileged_encoder_cfg['num_privileged_obs'],
                                      privileged_encoder_cfg['num_privileged_encoder_out'],
                                      privileged_encoder_cfg['privileged_encoder_hidden_dims'],
                                      module_cfg_dict['activation'])
        print(f"Privileged Encoder MLP: {self.privileged_encoder}")
        # heightmap encoder
        self.heightmap_encoder = MLP(heightmap_encoder_cfg['num_heightmap_obs'],
                                     heightmap_encoder_cfg['num_heightmap_encoder_out'],
                                     heightmap_encoder_cfg['heightmap_encoder_hidden_dims'],
                                     module_cfg_dict['activation'])
        print(f"Heightmap Encoder MLP: {self.heightmap_encoder}")

        # critic
        self.critic = MLP(critic_cfg['num_critic_obs'],
                          1,
                          critic_cfg['critic_hidden_dims'],
                          module_cfg_dict['activation'])
        # critic observation normalization
        self.critic_obs_normalization = critic_cfg['critic_obs_normalization']
        if critic_cfg['critic_obs_normalization']:
            self.critic_obs_normalizer = EmpiricalNormalization(critic_cfg['num_critic_obs'])
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(actor_cfg['num_actions']))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(actor_cfg['num_actions'])))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

        self.min_std = min_std

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        # compute mean
        mean = self.actor(obs)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # std = torch.clamp(std, min=self.min_std)
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs):
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        if 'estimator_out' in obs:
            obs = torch.cat((obs['estimator_out'], actor_obs), dim=-1)
        else:
            obs = actor_obs
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        if 'estimator_out' in obs:
            obs = torch.cat((obs['estimator_out'], actor_obs), dim=-1)
        else:
            obs = actor_obs
        return self.actor(obs)

    def evaluate(self, obs):
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        privileged_obs = self.get_privileged_obs(obs)
        heightmap_obs = self.get_heightmap_obs(obs)
        privileged_obs = self.privileged_encoder(privileged_obs)
        heightmap_obs = self.heightmap_encoder(heightmap_obs)
        obs = torch.cat((privileged_obs, heightmap_obs, critic_obs), dim=-1)
        # print(f"critic_obs: {critic_obs}")
        # print(f"heightmap_obs: {heightmap_obs}")
        return self.critic(obs)

    def get_actor_obs(self, obs):
        return obs['actor_obs']

    def get_critic_obs(self, obs):
        return obs['critic_obs']

    def get_privileged_obs(self, obs):
        return obs['privileged_obs']

    def get_heightmap_obs(self, obs):
        return obs['gt_heightmap_obs']

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True  # training resumes

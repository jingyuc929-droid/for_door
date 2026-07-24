# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


class RolloutStorageAMPVAEVIT:
    class Transition:
        def __init__(self):
            self.actor_observations = None
            self.critic_observations = None
            self.privileged_observations = None
            self.amp_observations = None
            self.prop_observations = None
            self.next_observations = None
            self.lidar_observations = None
            self.gt_vel = None
            self.gt_mass = None
            self.gt_footheight = None
            self.gt_heightmap = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        privileged_obs_shape,
        amp_obs_shape,
        prop_obs_shape,
        lidar_obs_shape,
        gt_vel_shape,
        gt_mass_shape,
        gt_footheight_shape,
        gt_heightmap_shape,
        next_obs_shape,
        actions_shape,
        device="cuda:0",
    ):
        # store inputs
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.actor_obs_shape = actor_obs_shape
        self.critic_obs_shape = critic_obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.amp_obs_shape = amp_obs_shape
        self.prop_obs_shape = prop_obs_shape
        self.lidar_obs_shape = lidar_obs_shape
        self.gt_vel_shape = gt_vel_shape
        self.gt_mass_shape = gt_mass_shape
        self.gt_footheight_shape = gt_footheight_shape
        self.gt_heightmap_shape = gt_heightmap_shape
        self.next_obs_shape = next_obs_shape
        self.actions_shape = actions_shape

        # Core
        self.actor_observations = torch.zeros(
            num_transitions_per_env, num_envs, *actor_obs_shape, device=self.device
        )
        self.prop_observations = torch.zeros(num_transitions_per_env, num_envs, *prop_obs_shape, device=self.device)
        self.next_observations = torch.zeros(num_transitions_per_env, num_envs, *next_obs_shape, device=self.device)
        self.critic_observations = torch.zeros(num_transitions_per_env, num_envs, *critic_obs_shape, device=self.device)
        self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
        self.amp_observations = torch.zeros(num_transitions_per_env, num_envs, *amp_obs_shape, device=self.device)
        self.lidar_observations = torch.zeros(num_transitions_per_env, num_envs, *lidar_obs_shape, device=self.device)
        self.gt_vel = torch.zeros(num_transitions_per_env, num_envs, *gt_vel_shape, device=self.device)
        self.gt_mass = torch.zeros(num_transitions_per_env, num_envs, *gt_mass_shape, device=self.device)
        self.gt_footheight = torch.zeros(num_transitions_per_env, num_envs, *gt_footheight_shape, device=self.device)
        self.gt_heightmap = torch.zeros(num_transitions_per_env, num_envs, *gt_heightmap_shape, device=self.device)

        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # counter for the number of transitions stored
        self.step = 0

    def add_transitions(self, transition: Transition):
        # check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.actor_observations[self.step].copy_(transition.actor_observations)
        self.critic_observations[self.step].copy_(transition.critic_observations)
        self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.amp_observations[self.step] = transition.amp_observations
        self.prop_observations[self.step].copy_(transition.prop_observations)
        self.next_observations[self.step].copy_(transition.next_observations)
        self.lidar_observations[self.step].copy_(transition.lidar_observations)
        self.gt_vel[self.step].copy_(transition.gt_vel)
        self.gt_mass[self.step].copy_(transition.gt_mass)
        self.gt_footheight[self.step].copy_(transition.gt_footheight)
        self.gt_heightmap[self.step].copy_(transition.gt_heightmap)

        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # For PPO
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)

        # increment the counter
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            # if we are at the last step, bootstrap the return value
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            self.returns[step] = advantage + self.values[step]

        # Compute the advantages
        self.advantages = self.returns - self.values
        # Normalize the advantages if flag is set
        # This is to prevent double normalization (i.e. if per minibatch normalization is used)
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((
            flat_dones.new_tensor([-1], dtype=torch.int64),
            flat_dones.nonzero(as_tuple=False)[:, 0],
        ))
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Core
        actor_observations = self.actor_observations.flatten(0, 1)
        critic_observations = self.critic_observations.flatten(0, 1)
        privileged_observations = self.privileged_observations.flatten(0, 1)
        amp_observations = self.amp_observations.flatten(0, 1)
        prop_observations = self.prop_observations.flatten(0, 1)
        next_observations = self.next_observations.flatten(0, 1)
        lidar_observations = self.lidar_observations.flatten(0, 1)
        gt_vel = self.gt_vel.flatten(0, 1)
        gt_mass = self.gt_mass.flatten(0, 1)
        gt_footheight = self.gt_footheight.flatten(0, 1)
        gt_heightmap = self.gt_heightmap.flatten(0, 1)

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        # For PPO
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # Create the mini-batch
                # -- Core
                actor_observations_batch = actor_observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                privileged_observations_batch = privileged_observations[batch_idx]
                amp_observations_batch = amp_observations[batch_idx]
                prop_observations_batch = prop_observations[batch_idx]
                next_observations_batch = next_observations[batch_idx]
                lidar_observations_batch = lidar_observations[batch_idx]
                gt_vel_batch = gt_vel[batch_idx]
                gt_mass_batch = gt_mass[batch_idx]
                gt_footheight_batch = gt_footheight[batch_idx]
                gt_heightmap_batch = gt_heightmap[batch_idx]
                actions_batch = actions[batch_idx]

                # -- For PPO
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                # Yield the mini-batch
                yield actor_observations_batch, \
                    critic_observations_batch, \
                    privileged_observations_batch, \
                    amp_observations_batch, \
                    prop_observations_batch, \
                    next_observations_batch, \
                    lidar_observations_batch, \
                    gt_vel_batch, \
                    gt_mass_batch, \
                    gt_footheight_batch, \
                    gt_heightmap_batch, \
                    actions_batch, \
                    target_values_batch, \
                    advantages_batch, \
                    returns_batch, \
                    old_actions_log_prob_batch, \
                    old_mu_batch, \
                    old_sigma_batch, (
                        None,
                        None,
                    ), None

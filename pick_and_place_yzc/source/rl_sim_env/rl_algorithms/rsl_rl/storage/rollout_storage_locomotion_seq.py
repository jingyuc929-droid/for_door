# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from rsl_rl.utils import split_and_pad_trajectories


class RolloutStorageLocomotionSeq:
    class Transition:
        def __init__(self):
            self.actor_observations = None
            self.critic_observations = None
            # self.privileged_observations = None
            self.amp_observations = None
            self.prop_observations = None
            self.next_observations = None
            self.lidar_observations = None
            self.gt_vel = None
            # self.gt_mass = None
            self.gt_footheight = None
            self.gt_heightmap = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            # self.prop_hidden_states = None
            self.heightmap_hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        # privileged_obs_shape,
        amp_obs_shape,
        prop_obs_shape,
        lidar_obs_shape,
        gt_vel_shape,
        # gt_mass_shape,
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
        # self.privileged_obs_shape = privileged_obs_shape
        self.amp_obs_shape = amp_obs_shape
        self.prop_obs_shape = prop_obs_shape
        self.lidar_obs_shape = lidar_obs_shape
        self.gt_vel_shape = gt_vel_shape
        # self.gt_mass_shape = gt_mass_shape
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
        # self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
        self.amp_observations = torch.zeros(num_transitions_per_env, num_envs, *amp_obs_shape, device=self.device)
        self.lidar_observations = torch.zeros(num_transitions_per_env, num_envs, *lidar_obs_shape, device=self.device)
        self.gt_vel = torch.zeros(num_transitions_per_env, num_envs, *gt_vel_shape, device=self.device)
        # self.gt_mass = torch.zeros(num_transitions_per_env, num_envs, *gt_mass_shape, device=self.device)
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

        # For GRU
        self.prop_hidden_states = None
        self.heightmap_hidden_states = None

        # counter for the number of transitions stored
        self.step = 0

    def add_transitions(self, transition: Transition):
        # check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.actor_observations[self.step].copy_(transition.actor_observations)
        self.critic_observations[self.step].copy_(transition.critic_observations)
        # self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.amp_observations[self.step] = transition.amp_observations
        self.prop_observations[self.step].copy_(transition.prop_observations)
        self.next_observations[self.step].copy_(transition.next_observations)
        self.lidar_observations[self.step].copy_(transition.lidar_observations)
        self.gt_vel[self.step].copy_(transition.gt_vel)
        # self.gt_mass[self.step].copy_(transition.gt_mass)
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

        # For GRU

        # self._save_prop_hidden_states(transition.prop_hidden_states)
        self._save_heightmap_hidden_states(transition.heightmap_hidden_states)

        # increment the counter
        self.step += 1

    # def _save_prop_hidden_states(self, hidden_states):
    #     if hidden_states is None:
    #         return

    #     if self.prop_hidden_states is None:
    #         T = self.prop_observations.shape[0]   # 轨迹长度
    #         self.prop_hidden_states = torch.zeros(
    #             T, *hidden_states.shape, device=self.device
    #         )
    #     # print("transition.hidden_states", hidden_states.shape)
    #     # print("self.prop_observations.shape", self.prop_observations.shape)
    #     # 4. 写入当前时间步 self.step
    #     self.prop_hidden_states[self.step].copy_(hidden_states)

    def _save_heightmap_hidden_states(self, hidden_states):
        if hidden_states is None:
            return

        if self.heightmap_hidden_states is None:
            T = self.prop_observations.shape[0]   # 轨迹长度
            self.heightmap_hidden_states = torch.zeros(
                T, *hidden_states.shape, device=self.device
            )
        # print("transition.hidden_states", hidden_states.shape)
        # print("self.prop_observations.shape", self.prop_observations.shape)
        # 4. 写入当前时间步 self.step
        self.heightmap_hidden_states[self.step].copy_(hidden_states)

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

    # for reinfrocement learning with recurrent networks
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        # padded_actor_traj, traj_masks = split_and_pad_trajectories(self.actor_observations, self.dones)
        # padded_critic_traj, _ = split_and_pad_trajectories(self.critic_observations, self.dones)
        # padded_priv_traj, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        # padded_amp_traj, _ = split_and_pad_trajectories(self.amp_observations, self.dones)
        padded_prop_traj, traj_masks = split_and_pad_trajectories(self.prop_observations, self.dones)
        # padded_next_traj, _ = split_and_pad_trajectories(self.next_observations, self.dones)
        padded_lidar_traj, _ = split_and_pad_trajectories(self.lidar_observations, self.dones)
        # padded_gt_vel_traj, _ = split_and_pad_trajectories(self.gt_vel, self.dones)
        # padded_gt_mass_traj, _ = split_and_pad_trajectories(self.gt_mass, self.dones)
        # padded_gt_footheight_traj, _ = split_and_pad_trajectories(self.gt_footheight, self.dones)
        # padded_gt_heightmap_traj, _ = split_and_pad_trajectories(self.gt_heightmap, self.dones)

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = traj_masks[:, first_traj:last_traj]
                # actor_observations_batch = padded_actor_traj[:, first_traj:last_traj]
                # critic_observations_batch = padded_critic_traj[:, first_traj:last_traj]
                # privileged_observations_batch = padded_priv_traj[:, first_traj:last_traj]
                # amp_observations_batch = padded_amp_traj[:, first_traj:last_traj]
                prop_observations_batch = padded_prop_traj[:, first_traj:last_traj]
                # next_observations_batch = padded_next_traj[:, first_traj:last_traj]
                lidar_observations_batch = padded_lidar_traj[:, first_traj:last_traj]

                actor_observations_batch = self.actor_observations[:, start:stop].flatten(0, 1)
                critic_observations_batch = self.critic_observations[:, start:stop].flatten(0, 1)
                # privileged_observations_batch = self.privileged_observations[:, start:stop].flatten(0, 1)
                amp_observations_batch = self.amp_observations[:, start:stop].flatten(0, 1)
                next_observations_batch = self.next_observations[:, start:stop].flatten(0, 1)

                gt_vel_batch = self.gt_vel[:, start:stop].flatten(0, 1)
                # gt_mass_batch = self.gt_mass[:, start:stop].flatten(0, 1)
                gt_footheight_batch = self.gt_footheight[:, start:stop].flatten(0, 1)
                gt_heightmap_batch = self.gt_heightmap[:, start:stop].flatten(0, 1)

                actions_batch = self.actions[:, start:stop].flatten(0, 1)
                old_mu_batch = self.mu[:, start:stop].flatten(0, 1)
                old_sigma_batch = self.sigma[:, start:stop].flatten(0, 1)
                returns_batch = self.returns[:, start:stop].flatten(0, 1)
                advantages_batch = self.advantages[:, start:stop].flatten(0, 1)
                values_batch = self.values[:, start:stop].flatten(0, 1)
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop].flatten(0, 1)

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                # if self.hidden_states is not None:
                # prop_hidden_states_batch = self.prop_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                heightmap_hidden_states_batch = self.heightmap_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                # else:
                # hid_batch = None

                yield actor_observations_batch, \
                    critic_observations_batch, \
                    amp_observations_batch, \
                    prop_observations_batch, \
                    next_observations_batch, \
                    lidar_observations_batch, \
                    gt_vel_batch, \
                    gt_footheight_batch, \
                    gt_heightmap_batch, \
                    actions_batch, \
                    values_batch, \
                    advantages_batch, \
                    returns_batch, \
                    old_actions_log_prob_batch, \
                    old_mu_batch, \
                    old_sigma_batch, \
                    heightmap_hidden_states_batch, \
                    masks_batch,

                first_traj = last_traj

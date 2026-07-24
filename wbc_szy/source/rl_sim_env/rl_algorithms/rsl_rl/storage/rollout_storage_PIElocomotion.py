# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict

from rl_algorithms.rsl_rl.utils import split_and_pad_trajectories


class RolloutStoragePIElocomotion:
    class Transition:
        def __init__(self):
            self.observations = None
            self.actions = None
            self.privileged_actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.gru_out_hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs,
        actions_shape,
        device="cpu",
    ):
        # store inputs
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.actions_shape = actions_shape

        # Core
        self.observations = TensorDict(
            {key: torch.zeros(num_transitions_per_env, *value.shape, device=device) for key, value in obs.items()},
            batch_size=[num_transitions_per_env, num_envs],
            device=self.device,
        )

        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # for distillation
        if training_type == "distillation":
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # for reinforcement learning
        if training_type == "rl":
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # For RNN networks
        self.saved_gru_out_hidden_states = None

        # counter for the number of transitions stored
        self.step = 0

    def add_transitions(self, transition: Transition):
        # check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # for distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        # for reinforcement learning
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

        # For RNN networks
        self._save_gru_out_hidden_states(transition.gru_out_hidden_states)

        # increment the counter
        self.step += 1

    def _save_gru_out_hidden_states(self, gru_out_hidden_states: torch.Tensor):
        '''
        Input: (num_layers, num_envs, hidden_dim)
        '''
        if gru_out_hidden_states is None:
            return
        # initialize if needed
        if self.saved_gru_out_hidden_states is None:
            self.saved_gru_out_hidden_states = torch.zeros(self.observations.shape[0], *gru_out_hidden_states.shape, device=self.device) 

        # copy the states
        self.saved_gru_out_hidden_states[self.step].copy_(gru_out_hidden_states)

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

    # for distillation
    def generator(self):
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            yield self.observations[i], self.actions[i], self.privileged_actions[i], self.dones[i]

    # for reinforcement learning with feedforward networks
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Core
        # 展平为T*N*TensorDict{key:shape}/Tensor
        observations = self.observations.flatten(0, 1)
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
                obs_batch = observations[batch_idx]  # 每个mini_batch的尺寸是mini_batch_size
                actions_batch = actions[batch_idx]

                # -- For PPO
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                # yield the mini-batch
                yield obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    None,
                    None,
                ), None

    # for reinfrocement learning with recurrent networks
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        # 输入T×N×Dict{key:shape}，pad_sequence输出T×M×Dict{key:shape}，trajectory_masks为T×M×1，1表示该轨迹有效，0表示该轨迹无效,trajectory_masks[i,j,0]==1表示第i个时间步，第j个轨迹有效
        # 然后根据trajectory_masks，将padded_obs_trajectories中的无效轨迹去掉，得到T×M×Dict{key:shape}
        # 然后根据trajectory_masks，将self.actions中的无效轨迹去掉，得到T×M×Dict{key:shape}
        # 然后根据trajectory_masks，将self.mu中的无效轨迹去掉，得到T×M×Dict{key:shape}
        # 然后根据trajectory_masks，将self.sigma中的无效轨迹去掉，得到T×M×Dict{key:shape}
        # 然后根据trajectory_masks，将self.returns中的无效轨迹去掉，得到T×M×Dict{key:shape}
        # 然后根据trajectory_masks，将self.advantages中的无效轨迹去掉，得到T×M×Dict{key:shape}
        # 然后根据trajectory_masks，将self.values中的无效轨迹去掉，得到T×M×Dict{key:shape}
        # 然后根据trajectory_masks，将self.actions_log_prob中的无效轨迹去掉，得到T×M×Dict{key:shape}
        ''' Example:
        Input: [[a1,b1],
                [a2,b2],
                [a3,b3],
                [a4,b4],
                [a5,b5],
                [a6,b6]],T=6,N=2
        # 形状 [T=6, N=2]，最后一行按函数会被强制置 1
        dones = [
          [0,0],  # t1
          [0,1],  # t2: env1 结束 → env1 段长 2
          [0,0],  # t3
          [1,0],  # t4: env0 结束 → env0 段长 4
          [0,1],  # t5: env1 结束 → env1 段长 3（t3~t5）
          [1,1],  # t6: 函数内强制 done，env0 段长 2（t5~t6），env1 段长 1（t6）
        ]

        Output:[[a1, a5, b1, b3, b6],   | [[True , True , True , True , True ],
                [a2, a6, b2, b4, 0 ],   |  [True , True , True , True , False],
                [a3, 0 , 0 , b5, 0 ],   |  [True , False, False, True , False],
                [a4, 0 , 0 , 0 , 0 ],   |  [True , False, False, False, False],
                [0 , 0 , 0 , 0 , 0 ],   |  [False, False, False, False, False]
                [0 , 0 , 0 , 0 , 0 ]]   |  [False, False, False, False, False]]
        '''
        proprio_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations['PIE_estimator_net_proprioceptive_obs'], self.dones)
        depth_images_trajectories, _ = split_and_pad_trajectories(self.observations['PIE_estimator_net_depth_images_obs'], self.dones)
        # padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
                first_traj = 0
                for i in range(num_mini_batches):
                    start = i * mini_batch_size
                    stop = (i + 1) * mini_batch_size
    
                    dones = self.dones.squeeze(-1)
                    last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                    last_was_done[1:] = dones[:-1]
                    last_was_done[0] = True  # 和dones中最后一维设置为1相互对应，保证start:stop得到的所有终止切片都被加入
                    trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                    
                    # 这里trajectories_batch_size不一定等于mini_batch_size，导致batch数不一样
                    last_traj = first_traj + trajectories_batch_size
    
                    masks_batch = trajectory_masks[:, first_traj:last_traj]  # 不是随机的，会把start:stop中的所有终止的切片都加入
                    # obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                    obs_batch = self.observations[:, start:stop]
                    proprioceptive_obs_after_pad = proprio_obs_trajectories[:, first_traj:last_traj]
                    depth_images_obs_after_pad = depth_images_trajectories[:, first_traj:last_traj]
                    actions_batch = self.actions[:, start:stop]
                    old_mu_batch = self.mu[:, start:stop]
                    old_sigma_batch = self.sigma[:, start:stop]
                    returns_batch = self.returns[:, start:stop]
                    advantages_batch = self.advantages[:, start:stop]
                    values_batch = self.values[:, start:stop]
                    old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]
    
                    # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                    # then take only time steps after dones (flattens num envs and time dimensions),
                    # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                    last_was_done = last_was_done.permute(1, 0)  # [T, N] -> [N, T]
                    if self.saved_gru_out_hidden_states is None:
                        # 没有 RNN 隐状态被保存（例如使用前馈网络），直接返回 None 占位
                        gru_out_hidden_states_batch = None
                    else:
                        # [T, num_layers, num_envs, hidden_dim] -> [num_valid_trajectory_start, num_layers, hidden_dim]
                        gru_out_hidden_states_batch = self.saved_gru_out_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()  # 为什么需要存储连续
    
                    yield obs_batch, proprioceptive_obs_after_pad, depth_images_obs_after_pad, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, gru_out_hidden_states_batch, masks_batch
    
                    first_traj = last_traj
    
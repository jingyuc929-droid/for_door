# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from math import exp

import torch
import torch.nn as nn
import torch.optim as optim
from rl_algorithms.amp_utils.normalizer import Normalizer
from rl_algorithms.rsl_rl.modules import VAE, ActorCritic, AMPDiscriminator
from rl_algorithms.rsl_rl.storage import ReplayBuffer, RolloutStorageAMPVAEPerception


class AMPVAEPerceptionPPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    actor_critic: ActorCritic
    """The actor critic module."""

    amp_discriminator: AMPDiscriminator
    """The AMP discriminator module."""

    vae: VAE
    """The VAE module."""

    amp_normalizer: Normalizer
    """The AMP normalizer module."""

    def __init__(
        self,
        actor_critic,
        amp_discriminator,
        amp_data,
        amp_normalizer,
        vae,
        amp_replay_buffer_size=100000,
        amp_disc_grad_penalty=5.0,
        vae_beta=1.0,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        learning_rate_vae=1e-3,
        vae_desired_recon_loss=0.1,
        vae_beta_min=1.0e-3,
        vae_beta_max=5.0,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        vae_desired_vel_loss=0.01,
        device="cuda:0",
        normalize_advantage_per_mini_batch=False,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.vae_desired_vel_loss = vae_desired_vel_loss
        self.learning_rate_vae = learning_rate_vae
        self.vae_desired_recon_loss = vae_desired_recon_loss
        self.vae_beta_min = vae_beta_min
        self.vae_beta_max = vae_beta_max
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # AMP Discriminator components
        self.amp_discriminator = amp_discriminator
        self.amp_discriminator.to(self.device)

        # VAE components
        self.vae = vae
        self.vae.to(self.device)

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)

        # Create optimizer
        params = [
            {"params": self.actor_critic.parameters(), "lr": learning_rate, "name": "actor_critic"},
            {
                "params": self.amp_discriminator.trunk.parameters(),
                "lr": learning_rate,
                "weight_decay": 10e-4,
                "name": "amp_trunk",
            },
            {
                "params": self.amp_discriminator.amp_linear.parameters(),
                "lr": learning_rate,
                "weight_decay": 10e-2,
                "name": "amp_head",
            },
            {"params": self.vae.parameters(), "lr": learning_rate_vae, "name": "vae"},
        ]
        self.optimizer = optim.Adam(params)

        # Create rollout storage
        self.storage: RolloutStorageAMPVAEPerception = None  # type: ignore
        self.transition = RolloutStorageAMPVAEPerception.Transition()

        # AMP data
        self.amp_storage = ReplayBuffer(amp_discriminator.input_dim // 2, amp_replay_buffer_size, device)
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # AMP parameters
        self.amp_disc_grad_penalty = amp_disc_grad_penalty

        # VAE parameters
        self.vae_beta = vae_beta

        # Adaboot
        self.p_boot = torch.zeros(200, dtype=torch.float32, device=self.device, requires_grad=False)
        self.p_boot_mean = 0.0
        self.episode_rewards = []

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        actor_obs_shape_vae,
        critic_obs_shape,
        amp_obs_shape,
        vae_obs_shape,
        action_shape,
    ):
        # create rollout storage
        self.storage = RolloutStorageAMPVAEPerception(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            actor_obs_shape_vae,
            critic_obs_shape,
            amp_obs_shape,
            vae_obs_shape,
            action_shape,
            self.device,
        )

    def act(self, actor_obs, critic_obs, amp_obs, vae_obs):
        vae_code, vae_code_vel, vae_code_latent, _, _, _, _, _ = self.vae.cenet_forward(vae_obs)
        obs_full_batch = torch.cat(
            (
                self.p_boot_mean * vae_code_vel + (1 - self.p_boot_mean) * critic_obs[:, 0:3],
                vae_code_latent,
                actor_obs,
            ),
            dim=-1,
        ).detach()
        # compute the actions and values
        self.transition.actions = self.actor_critic.act(obs_full_batch).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.actor_observations = obs_full_batch
        self.transition.critic_observations = critic_obs
        self.transition.amp_observations = amp_obs
        self.transition.vae_observations = vae_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, amp_obs, next_obs, episode_reward):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.next_observations = next_obs

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        self.amp_storage.insert(self.transition.amp_observations, amp_obs)

        # record the transition
        self.storage.add_transitions(self.transition)
        for i, done in enumerate(dones):
            if done:
                self.episode_rewards.append(episode_reward[i].item())
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_bootstrap_probability(self):
        if self.is_multi_gpu:
            # 本地 rewards 列表转张量（形状：[n_local_episodes]）
            rewards_tensor = torch.tensor(self.episode_rewards, dtype=torch.float32, device=self.device)

            # 先收集各卡 episode_count_i
            local_count = torch.tensor([rewards_tensor.numel()], device=self.device, dtype=torch.int32)
            counts = [torch.zeros_like(local_count) for _ in range(self.gpu_world_size)]
            torch.distributed.all_gather(counts, local_count)
            counts = [int(c.item()) for c in counts]

            # 再 all_gather 各卡数据 —— 因为每张卡 episode 数不一，需要先 pad 或者借助 all_gather_object
            rewards_list = [None for _ in range(self.gpu_world_size)]
            torch.distributed.all_gather_object(rewards_list, self.episode_rewards.copy())
            # rewards_list 是一个 Python list，每个元素都是列表，包含对应 rank 的 episode_rewards

            # 把所有 rank 的 episode_rewards 拼成一个大列表
            all_rewards = []
            for r_list in rewards_list:
                all_rewards.extend(r_list)
            if len(all_rewards) < 30:
                return self.p_boot_mean

            # 把全局列表转张量
            all_rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32, device=self.device)
            mean_R = all_rewards_tensor.mean()
            std_R = all_rewards_tensor.std(unbiased=False)
        else:
            if len(self.episode_rewards) < 30:
                return self.p_boot_mean
            rewards_tensor = torch.tensor(self.episode_rewards, dtype=torch.float32).to(self.device)
            mean_R = rewards_tensor.mean()
            std_R = rewards_tensor.std()

        if mean_R.abs() < 1e-6:
            cv = 0.0
        else:
            cv = (std_R / mean_R).item()
        p_boot = 1 - torch.tanh(torch.tensor(cv)).item()  # p_boot ∈ [0, 1]
        p_boot = 1.45 * p_boot
        p_boot = max(0.0, min(p_boot, 1.0))
        return p_boot

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.p_boot[1:] = self.p_boot[:-1].clone()
        if self.p_boot_mean < 0.99:
            self.p_boot[0] = self.compute_bootstrap_probability()
        else:
            self.p_boot[0] = 1.0
        self.p_boot_mean = self.p_boot.mean()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )
        self.episode_rewards.clear()

        return self.p_boot_mean

    def update(self):  # noqa: C901
        # -- For PPO
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0

        # -- For AMP
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0

        # -- For VAE
        mean_vae_vel_loss = 0
        mean_vae_decode_loss = 0
        mean_vae_kl_loss = 0
        mean_vae_loss = 0
        mean_vae_beta = 0

        # generator for mini batches
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )

        # iterate over batches
        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
            (
                actor_obs_batch,
                critic_obs_batch,
                amp_obs_batch,
                vae_obs_batch,
                next_actor_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
            ) = sample

            # original batch size
            original_batch_size = actor_obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.actor_critic.act(actor_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.actor_critic.action_mean[:original_batch_size]
            sigma_batch = self.actor_critic.action_std[:original_batch_size]
            entropy_batch = self.actor_critic.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        if param_group["name"] == "vae":
                            param_group["lr"] = self.learning_rate_vae
                        else:
                            param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # AMP Discriminator loss.
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert

            policy_state_unnorm = torch.clone(policy_state)
            expert_state_unnorm = torch.clone(expert_state)

            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                    expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
            policy_d = self.amp_discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
            expert_d = self.amp_discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
            expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
            policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen_loss = self.amp_discriminator.compute_gradient_penalty(
                expert_state, expert_next_state, lambda_=self.amp_disc_grad_penalty
            )

            # -- beat VAE loss
            vae_vel_target = critic_obs_batch[:, 0:3]
            vae_decode_target = next_actor_obs_batch
            vae_vel_target.requires_grad = False
            vae_decode_target.requires_grad = False
            (
                vae_code,
                vae_code_vel,
                vae_code_latent,
                vae_decoded,
                vae_mean_vel,
                vae_logvar_vel,
                vae_mean_latent,
                vae_logvar_latent,
            ) = self.vae.cenet_forward(vae_obs_batch)
            loss_recon_vel = nn.MSELoss()(vae_code_vel, vae_vel_target)
            loss_recon_decode = nn.MSELoss()(vae_decoded, vae_decode_target)
            loss_recon = loss_recon_vel + loss_recon_decode

            # beta
            k = exp(self.learning_rate * (self.vae_desired_recon_loss - loss_recon_decode.item()))
            self.vae_beta = max(self.vae_beta_min, min(self.vae_beta_max, k * self.vae_beta))

            kl_div = -0.5 * torch.sum(
                1 + vae_logvar_latent - vae_mean_latent.pow(2) - vae_logvar_latent.exp(),
                dim=1,
            )
            kl_loss = self.vae_beta * torch.mean(kl_div)

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + amp_loss
                + grad_pen_loss
                + loss_recon
                + kl_loss
            )

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO
            all_params = [p for pg in self.optimizer.param_groups for p in pg["params"] if p.grad is not None]
            nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
            self.optimizer.step()

            if self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state_unnorm.cpu().numpy())
                self.amp_normalizer.update(expert_state_unnorm.cpu().numpy())

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()

            # Store the AMP losses
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()

            # Store the VAE losses
            mean_vae_vel_loss += loss_recon_vel.item()
            mean_vae_decode_loss += loss_recon_decode.item()
            mean_vae_kl_loss += kl_loss.item()
            mean_vae_beta += self.vae_beta
            mean_vae_loss += loss_recon.item() + kl_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- For AMP
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        # -- For VAE
        mean_vae_loss /= num_updates
        mean_vae_vel_loss /= num_updates
        mean_vae_decode_loss /= num_updates
        mean_vae_kl_loss /= num_updates
        mean_vae_beta /= num_updates

        # -- Clear the storage
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "amp_loss": mean_amp_loss,
            "grad_pen_loss": mean_grad_pen_loss,
            "policy_pred": mean_policy_pred,
            "expert_pred": mean_expert_pred,
            "vae_vel_loss": mean_vae_vel_loss,
            "vae_decode_loss": mean_vae_decode_loss,
            "vae_kl_loss": mean_vae_kl_loss,
            "vae_beta": mean_vae_beta,
        }

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast actor_critic, vae and amp_discriminator parameters to all GPUs."""
        # 把三个模块的 state_dict 打包成一个 dict
        to_sync = {
            "actor_critic": self.actor_critic.state_dict(),
            "vae": self.vae.state_dict(),
            "amp_disc": self.amp_discriminator.state_dict(),
        }
        # 用 object_list 将其广播
        obj_list = [to_sync]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        synced = obj_list[0]
        # 把广播回来的参数加载回各自模块
        self.actor_critic.load_state_dict(synced["actor_critic"])
        self.vae.load_state_dict(synced["vae"])
        self.amp_discriminator.load_state_dict(synced["amp_disc"])

    def reduce_parameters(self):
        """Collect and average gradients from all GPUs for all three modules."""

        # 1) 收集所有要归约的梯度向量
        grads = []
        for module in (self.actor_critic, self.vae, self.amp_discriminator):
            for p in module.parameters():
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
        all_grads = torch.cat(grads)

        # 2) 全局平均
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # 3) 把归约后的梯度写回每个参数
        offset = 0
        for module in (self.actor_critic, self.vae, self.amp_discriminator):
            for p in module.parameters():
                if p.grad is not None:
                    numel = p.numel()
                    p.grad.data.copy_(all_grads[offset : offset + numel].view_as(p.grad.data))
                    offset += numel

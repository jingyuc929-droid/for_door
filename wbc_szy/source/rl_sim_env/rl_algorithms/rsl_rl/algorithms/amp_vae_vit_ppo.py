# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from rl_algorithms.amp_utils.normalizer import Normalizer
from rl_algorithms.rsl_rl.modules import VAEVit, ActorCriticVit, AMPDiscriminator
from rl_algorithms.rsl_rl.storage import ReplayBuffer, RolloutStorageAMPVAEVITSeq

# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.amp import autocast, GradScaler


class AMPVAEVITPPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    actor_critic: ActorCriticVit
    """The actor critic module."""

    amp_discriminator: AMPDiscriminator
    """The AMP discriminator module."""

    amp_normalizer: Normalizer
    """The AMP normalizer module."""

    vae_vit: VAEVit
    """The VAE module."""

    def __init__(
        self,
        actor_critic_vit,
        amp_discriminator,
        amp_data,
        amp_normalizer,
        vae_vit,
        vae_beta=1.0,
        amp_replay_buffer_size=100000,
        amp_disc_grad_penalty=5.0,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        vae_beta_min=1.0e-3,
        vae_beta_max=5.0,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        vae_desired_loss=0.01,
        device="cuda:0",
        normalize_advantage_per_mini_batch=False,
        use_ground_truth=False,
        use_amp=True,
        use_adaboot=False,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        self.use_ground_truth = use_ground_truth
        self.use_amp = use_amp
        self.use_adaboot = use_adaboot
        # if self.use_amp:
        # self.scaler = GradScaler()
        # device-related parameters
        self.device = device
        self.vae_desired_loss = vae_desired_loss
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

        if self.use_amp:
            # AMP Discriminator components
            self.amp_discriminator = amp_discriminator
            self.amp_discriminator.to(self.device)
            # AMP data
            self.amp_storage = ReplayBuffer(amp_discriminator.input_dim // 2, amp_replay_buffer_size, device)
            self.amp_data = amp_data
            self.amp_normalizer = amp_normalizer
            # AMP parameters
            self.amp_disc_grad_penalty = amp_disc_grad_penalty
        else:
            self.amp_discriminator = None
            self.amp_storage = None
            self.amp_data = None
            self.amp_normalizer = None
            self.amp_disc_grad_penalty = None

        # VAE components
        self.vae_vit = vae_vit
        self.vae_vit.to(self.device)

        # PPO components
        self.actor_critic = actor_critic_vit
        self.actor_critic.to(self.device)

        # if self.is_multi_gpu:
        #     self.amp_discriminator = DDP(self.amp_discriminator, device_ids=[self.device], output_device=self.device)
        #     self.actor_critic = DDP(self.actor_critic, device_ids=[self.device], output_device=self.device)
        #     self.vae_vit = DDP(self.vae_vit, device_ids=[self.device], output_device=self.device)

        # Create optimizer
        params = [
            {"params": self.actor_critic.parameters(), "lr": learning_rate, "name": "actor_critic"},
            {"params": self.vae_vit.parameters(), "lr": learning_rate, "name": "vae_vit"},
        ]
        if self.use_amp:
            params += [
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
            ]
        self.optimizer = optim.Adam(params)

        # Create rollout storage
        self.storage: RolloutStorageAMPVAEVITSeq = None  # type: ignore
        self.transition = RolloutStorageAMPVAEVITSeq.Transition()

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
        critic_obs_shape,
        amp_obs_shape,
        prop_obs_shape,
        lidar_obs_shape,
        gt_vel_shape,
        gt_footheight_shape,
        gt_heightmap_shape,
        next_obs_shape,
        action_shape,
        device,
    ):
        # create rollout storage
        self.storage = RolloutStorageAMPVAEVITSeq(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            amp_obs_shape,
            prop_obs_shape,
            lidar_obs_shape,
            gt_vel_shape,
            gt_footheight_shape,
            gt_heightmap_shape,
            next_obs_shape,
            action_shape,
            device,
        )

    def act(self, actor_obs, critic_obs, amp_obs, prop_history_obs, point_history_obs, gt_vel, gt_footheight, gt_heightmap):
        # self.transition.prop_hidden_states = self.vae_vit.get_prop_gru_last_h()
        self.transition.heightmap_hidden_states = self.vae_vit.get_heightmap_gru_last_h()
        # if self.vae_vit.get_gru_last_h() is not None:
        #     print("hidden_states!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", self.transition.hidden_states.shape)
        # print(f"[act] prop_history_obs.shape: {prop_history_obs.shape}")
        # print(f"[act] point_history_obs.shape: {point_history_obs.shape}")
        out_dict = self.vae_vit.cenet_forward(prop_history_obs.unsqueeze(0),
                                              point_history_obs.unsqueeze(0),
                                              heightmap_gt=gt_heightmap.unsqueeze(0),
                                              p_boot_mean=self.p_boot_mean,
                                              use_ground_truth=self.use_ground_truth)
        # print(f"vae_out_dict['code_vel']: {out_dict['code_vel'].shape}")
        # print(f"gt_vel_batch: {gt_vel.shape}")
        # print(gt_mass)
        if self.use_adaboot:
            batch_num, _ = gt_vel.shape
            replace_num = int(batch_num * (1 - self.p_boot_mean))
            if replace_num > 0:
                row_idx = torch.randperm(batch_num)[:replace_num]
                code_vel = out_dict["code_vel"].clone()
                code_vel[row_idx] = gt_vel[row_idx]
            else:
                code_vel = out_dict["code_vel"].squeeze(0)
        else:
            code_vel = out_dict["code_vel"].squeeze(0)
        obs_full_batch = torch.cat(
            (
                code_vel,
                out_dict["code_obs_latent"].squeeze(0),
                out_dict["code_footheight_latent"].squeeze(0),
                out_dict["code_heightmap_latent"].squeeze(0),
                actor_obs,
            ),
            dim=-1,
        )

        # compute the actions and values
        self.transition.actions = self.actor_critic.act(obs_full_batch, print_info=False).detach()
        values = self.actor_critic.evaluate(critic_obs, gt_footheight, gt_heightmap)
        self.transition.values = values.detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.actor_observations = actor_obs.detach()
        self.transition.critic_observations = critic_obs.detach()
        # self.transition.privileged_observations = priv_obs
        self.transition.amp_observations = amp_obs.detach()
        self.transition.prop_observations = prop_history_obs.flatten(start_dim=1).detach()
        self.transition.lidar_observations = point_history_obs.flatten(start_dim=1).detach()
        self.transition.gt_vel = gt_vel.detach()
        # self.transition.gt_mass = gt_mass
        self.transition.gt_footheight = gt_footheight.detach()
        self.transition.gt_heightmap = gt_heightmap.detach()
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
        if self.use_amp:
            self.amp_storage.insert(self.transition.amp_observations, amp_obs)

        # record the transition
        self.storage.add_transitions(self.transition)
        for i, done in enumerate(dones):
            if done:
                self.episode_rewards.append(episode_reward[i].item())
        self.transition.clear()
        self.actor_critic.reset(dones)
        self.vae_vit.reset_state_dones(dones)

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
        # p_boot = 1.1 * p_boot
        p_boot = max(0.0, min(p_boot, 1.0))
        return p_boot

    def compute_returns(self, last_critic_obs, last_gt_footheight, last_gt_heightmap):
        # compute value for the last step
        last_values = self.actor_critic.evaluate(last_critic_obs, last_gt_footheight, last_gt_heightmap)
        self.p_boot[1:] = self.p_boot[:-1].clone()
        if self.p_boot_mean < 0.8:
            self.p_boot[0] = self.compute_bootstrap_probability()
        else:
            self.p_boot[0] = 1.0
        self.p_boot_mean = self.p_boot.mean()
        self.storage.compute_returns(
            last_values.detach(), self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
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
        mean_vae_footheight_loss = 0
        mean_vae_prop_obs_loss = 0
        mean_vae_heightmap_loss = 0
        mean_vae_kl_loss = 0
        mean_vae_loss = 0
        mean_vae_beta = 0
        mean_vae_heightmap_rough_loss = 0
        mean_vae_heightmap_fine_loss = 0

        # -- For VAE Contrastive
        mean_contrastive_loss = 0

        # generator for mini batches
        generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        if self.use_amp:
            amp_policy_generator = self.amp_storage.feed_forward_generator(
                self.num_learning_epochs * self.num_mini_batches,
                self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
            )
            amp_expert_generator = self.amp_data.feed_forward_generator(
                self.num_learning_epochs * self.num_mini_batches,
                self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
            )
            data_generator = zip(generator, amp_policy_generator, amp_expert_generator)
        else:
            amp_policy_generator = None
            amp_expert_generator = None
            data_generator = generator

        # iterate over batches
        for sample_triplet in data_generator:
            if self.use_amp:
                sample, sample_amp_policy, sample_amp_expert = sample_triplet
            else:
                sample = sample_triplet
                sample_amp_policy = None
                sample_amp_expert = None

            (
                actor_obs_batch,
                critic_obs_batch,
                amp_obs_batch,
                prop_obs_batch,
                next_actor_obs_batch,
                lidar_obs_batch,
                gt_vel_batch,
                gt_footheight_batch,
                gt_heightmap_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                heightmap_hidden_states_batch,
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
            # with autocast(device_type='cuda', enabled=self.use_amp, dtype=torch.bfloat16):
            p_boot_mean = self.p_boot.mean().clone().detach().item()
            # print(f"p_boot_mean: {p_boot_mean.shape}")
            # print(f"gt_heightmap_batch: {gt_heightmap_batch.shape}")
            vae_out_dict = self.vae_vit.cenet_forward(prop_obs_batch,
                                                      lidar_obs_batch,
                                                      heightmap_hidden_states_batch.clone().detach(),
                                                      masks_batch, heightmap_gt=gt_heightmap_batch, p_boot_mean=p_boot_mean, use_ground_truth=self.use_ground_truth)

            # print(f"vae_out_dict['code_vel']: {vae_out_dict['code_vel'].shape}")
            # print(f"gt_vel_batch: {gt_vel_batch.shape}")
            # print(f"actor_obs_batch: {actor_obs_batch.shape}")
            # print(f"vae_out_dict['code_mass']: {vae_out_dict['code_mass'].shape}")
            # print(f"gt_mass_batch: {gt_mass_batch.shape}")
            # print(f"vae_out_dict['code_footheight']: {vae_out_dict['code_footheight'].shape}")
            # print(f"gt_footheight_batch: {gt_footheight_batch.shape}")
            # print("vae--------------------------------------------------------------------")
            # print("p_boot_mean", p_boot_mean)
            # print("vae_out_dict['code_vel']", vae_out_dict["code_vel"])
            # print("gt_vel_batch", gt_vel_batch)
            # print("p_boot_mean", p_boot_mean)
            # print("vae_out_dict['code_mass']", vae_out_dict["code_mass"])
            # print("gt_mass_batch", gt_mass_batch)
            # print("vae_out_dict['code_footheight']", vae_out_dict["code_footheight"])
            # print("gt_footheight_batch", gt_footheight_batch)
            # print("vae_out_dict['code_obs_latent']", vae_out_dict["code_obs_latent"])
            # print("vae_out_dict['code_heightmap_latent']", vae_out_dict["code_heightmap_latent"])
            # print("actor_obs_batch", actor_obs_batch)
            if self.use_adaboot:
                batch_num, _ = gt_vel_batch.shape
                replace_num = int(batch_num * (1 - self.p_boot_mean))
                if replace_num > 0:
                    row_idx = torch.randperm(batch_num)[:replace_num]
                    code_vel = vae_out_dict["code_vel"].clone()
                    code_vel[row_idx] = gt_vel_batch[row_idx]
                else:
                    code_vel = vae_out_dict["code_vel"].squeeze(0)
            else:
                code_vel = vae_out_dict["code_vel"]
            obs_full_batch = torch.cat(
                (
                    # p_boot_mean * vae_out_dict["code_vel"] + (1 - p_boot_mean) * gt_vel_batch,
                    code_vel,
                    vae_out_dict["code_obs_latent"],
                    vae_out_dict["code_footheight_latent"],
                    vae_out_dict["code_heightmap_latent"],
                    actor_obs_batch,
                ),
                dim=-1,
            )

            self.actor_critic.act(obs_full_batch, print_info=False)

            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch,
                gt_footheight_batch,
                gt_heightmap_batch,
            )
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.actor_critic.action_mean[:original_batch_size]
            sigma_batch = self.actor_critic.action_std[:original_batch_size]
            entropy_batch = self.actor_critic.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                # with autocast(device_type='cuda', enabled=False):
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
            if self.use_amp:
                # AMP Discriminator loss.
                # with autocast(device_type='cuda', enabled=False):
                policy_state, policy_next_state = sample_amp_policy
                expert_state, expert_next_state = sample_amp_expert

                policy_state_unnorm = torch.clone(policy_state)
                expert_state_unnorm = torch.clone(expert_state)

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
            else:
                amp_loss = 0.0
                grad_pen_loss = 0.0

            # -- beat VAE loss
            # print(f"vae_obs_batch.shape: {vae_obs_batch.shape}")
            # print(f"lidar_obs_batch.shape: {lidar_obs_batch.shape}")

            mse_vae_vel_loss = nn.MSELoss()(vae_out_dict["code_vel"], gt_vel_batch)
            mse_vae_prop_obs_loss = nn.MSELoss()(vae_out_dict["prop_obs_decoded"], next_actor_obs_batch)
            mse_vae_footheight_loss = nn.MSELoss()(vae_out_dict["footheight_decoded"], gt_footheight_batch)
            mse_vae_heightmap_loss = nn.MSELoss()(vae_out_dict["heightmap_decoded"], gt_heightmap_batch)
            if not self.use_ground_truth:
                mse_heightmap_rough_loss = nn.MSELoss()(vae_out_dict["heightmap_rough_decoded"], gt_heightmap_batch)
                mse_heightmap_fine_loss = nn.L1Loss()(vae_out_dict["heightmap_fine_decoded"], gt_heightmap_batch)
                vae_loss = mse_vae_vel_loss + mse_vae_footheight_loss + mse_vae_prop_obs_loss + mse_vae_heightmap_loss + mse_heightmap_rough_loss + mse_heightmap_fine_loss
            else:
                vae_loss = mse_vae_vel_loss + mse_vae_footheight_loss + mse_vae_prop_obs_loss + mse_vae_heightmap_loss

            # with autocast(device_type='cuda', enabled=False):
            k_tensor = torch.exp(self.learning_rate * (self.vae_desired_loss - mse_vae_prop_obs_loss))
            # 如果后面 vae_beta 需要一个 Python float，再调用 .item()
            k = k_tensor.item()
            self.vae_beta = max(self.vae_beta_min,
                                min(self.vae_beta_max, k * self.vae_beta))

            kl_div_obs = -0.5 * torch.sum(
                1 + vae_out_dict["logvar_obs"] - vae_out_dict["mean_obs"].pow(2) - vae_out_dict["logvar_obs"].exp(),
                dim=1,
            )

            kl_loss_obs = self.vae_beta * torch.mean(kl_div_obs)

            # with autocast(device_type='cuda', enabled=False):
            # margin = 1.0
            # lambda_ = 0.5
            # heightmap_critic_encoded = heightmap_encoded.detach()
            # heightmap_vae_decoded = vae_out_dict["code_heightmap_latent"]
            # l_pos = nn.MSELoss()(heightmap_vae_decoded, heightmap_critic_encoded)
            # z_rand = torch.empty_like(heightmap_critic_encoded).to(self.device).uniform_(-1.0, 1.0)  # [B, D]
            # delta = margin - (heightmap_vae_decoded - z_rand)
            # neg_term = torch.relu(delta)
            # l_neg = (neg_term**2).mean()
            # l_h_contrastive = lambda_ * l_pos + (1 - lambda_) * l_neg
            if self.use_amp:
                loss = (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_batch.mean()
                    + amp_loss
                    + grad_pen_loss
                    + vae_loss
                    + kl_loss_obs
                )
            else:
                loss = (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_batch.mean()
                    + vae_loss
                    + kl_loss_obs
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

            # self.optimizer.zero_grad()
            # # 用 scaler 代替直接 backward
            # self.scaler.scale(loss).backward()
            # # unscale 后再 clip
            # self.scaler.unscale_(self.optimizer)
            # if self.is_multi_gpu:
            #     self.reduce_parameters()
            # all_params = [p for pg in self.optimizer.param_groups for p in pg["params"] if p.grad is not None]
            # nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
            # # 用 scaler 代替 step
            # self.scaler.step(self.optimizer)
            # self.scaler.update()

            # torch.cuda.empty_cache()
            # del vae_out_dict
            # del obs_full_batch
            # del policy_state, policy_next_state, expert_state, expert_next_state

            if self.use_amp:
                self.amp_normalizer.update(policy_state_unnorm.cpu().numpy())
                self.amp_normalizer.update(expert_state_unnorm.cpu().numpy())

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()

            # Store the AMP losses
            if self.use_amp:
                mean_amp_loss += amp_loss.item()
                mean_grad_pen_loss += grad_pen_loss.item()
                mean_policy_pred += policy_d.mean().item()
                mean_expert_pred += expert_d.mean().item()

            # Store the VAE losses
            mean_vae_vel_loss += mse_vae_vel_loss.item()
            # mean_vae_mass_loss += mse_vae_mass_loss.item()
            mean_vae_footheight_loss += mse_vae_footheight_loss.item()
            if not self.use_ground_truth:
                mean_vae_heightmap_rough_loss += mse_heightmap_rough_loss.item()
                mean_vae_heightmap_fine_loss += mse_heightmap_fine_loss.item()
            mean_vae_prop_obs_loss += mse_vae_prop_obs_loss.item()
            mean_vae_heightmap_loss += mse_vae_heightmap_loss.item()
            mean_vae_kl_loss += kl_loss_obs.item()
            mean_vae_loss += vae_loss.item()
            mean_vae_beta += self.vae_beta
            # mean_contrastive_loss += l_h_contrastive.item()

        self.vae_vit.reset_state()
        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- For AMP
        if self.use_amp:
            mean_amp_loss /= num_updates
            mean_grad_pen_loss /= num_updates
            mean_policy_pred /= num_updates
            mean_expert_pred /= num_updates
        # -- For VAE
        mean_vae_loss /= num_updates
        mean_vae_vel_loss /= num_updates
        # mean_vae_mass_loss /= num_updates
        mean_vae_footheight_loss /= num_updates
        if not self.use_ground_truth:
            mean_vae_heightmap_rough_loss /= num_updates
            mean_vae_heightmap_fine_loss /= num_updates
        mean_vae_prop_obs_loss /= num_updates
        mean_vae_heightmap_loss /= num_updates
        mean_vae_kl_loss /= num_updates
        mean_vae_beta /= num_updates
        # -- For VAE Contrastive
        # mean_contrastive_loss /= num_updates

        # if self.is_multi_gpu and self.amp_normalizer is not None:
        #     # 调用外面定义的 sync_normalizer 函数
        #     sync_normalizer(self.amp_normalizer, self.device)

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
            # "vae_mass_loss": mean_vae_mass_loss,
            "vae_footheight_loss": mean_vae_footheight_loss,
            "vae_heightmap_rough_loss": mean_vae_heightmap_rough_loss,
            "vae_heightmap_fine_loss": mean_vae_heightmap_fine_loss,
            "vae_obs_prop_loss": mean_vae_prop_obs_loss,
            "vae_heightmap_loss": mean_vae_heightmap_loss,
            "vae_kl_loss": mean_vae_kl_loss,
            "vae_beta": mean_vae_beta,
            # "heightmap_contrastive_loss": mean_contrastive_loss,
            "vae_loss": mean_vae_loss,
        }

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast actor_critic, vae and amp_discriminator parameters to all GPUs."""
        # 把三个模块的 state_dict 打包成一个 dict
        if self.use_amp:
            to_sync = {
                "actor_critic": self.actor_critic.state_dict(),
                "vae_vit": self.vae_vit.state_dict(),
                "amp_disc": self.amp_discriminator.state_dict(),
            }
        else:
            to_sync = {
                "actor_critic": self.actor_critic.state_dict(),
                "vae_vit": self.vae_vit.state_dict(),
            }
        # 用 object_list 将其广播
        obj_list = [to_sync]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        synced = obj_list[0]
        # 把广播回来的参数加载回各自模块
        self.actor_critic.load_state_dict(synced["actor_critic"])
        self.vae_vit.load_state_dict(synced["vae_vit"])
        if self.use_amp:
            self.amp_discriminator.load_state_dict(synced["amp_disc"])

    def reduce_parameters(self):
        """Collect and average gradients from all GPUs for all three modules."""

        # 1) 收集所有要归约的梯度向量
        grads = []
        if self.use_amp:
            modules = (self.actor_critic, self.vae_vit, self.amp_discriminator)
        else:
            modules = (self.actor_critic, self.vae_vit)
        for module in modules:
            for p in module.parameters():
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
        all_grads = torch.cat(grads)

        # 2) 全局平均
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # 3) 把归约后的梯度写回每个参数
        offset = 0
        if self.use_amp:
            modules = (self.actor_critic, self.vae_vit, self.amp_discriminator)
        else:
            modules = (self.actor_critic, self.vae_vit)
        for module in modules:
            for p in module.parameters():
                if p.grad is not None:
                    numel = p.numel()
                    p.grad.data.copy_(all_grads[offset : offset + numel].view_as(p.grad.data))
                    offset += numel

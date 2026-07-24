# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from rl_algorithms.amp_utils.normalizer import Normalizer
from rl_algorithms.rsl_rl.modules import ActorCriticEncoder, AMPDiscriminator
from rl_algorithms.rsl_rl.storage import ReplayBuffer, RolloutStorageLocomotion


class LocomotionPPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    def __init__(
        self,
        module_dict: dict[str, ActorCriticEncoder | AMPDiscriminator | Normalizer],
        train_cfg_dict: dict,
        device="cuda:0",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        self.use_amp = train_cfg_dict['use_amp'] if 'use_amp' in train_cfg_dict else False
        self.use_vae = train_cfg_dict['use_vae'] if 'use_vae' in train_cfg_dict else False
        if self.use_vae:
            self.use_vae_exclusive_optimizer = train_cfg_dict['vae']['use_exclusive_optimizer'] if 'use_exclusive_optimizer' in train_cfg_dict['vae'] else False
        else:
            self.use_vae_exclusive_optimizer = False
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.optimizer_dict = {}
        # PPO components

        self.actor_critic = module_dict['actor_critic']
        self.actor_critic.to(self.device)
        actor_critic_params = [
            {"params": self.actor_critic.parameters(), "lr": train_cfg_dict['ppo_algorithm']['learning_rate'], "name": "actor_critic"},
        ]

        if self.use_amp:
            # AMP Discriminator components
            self.amp_discriminator = module_dict['amp_discriminator']
            self.amp_discriminator.to(self.device)
            # AMP data
            self.amp_storage = ReplayBuffer(self.amp_discriminator.input_dim // 2, train_cfg_dict['amp']['amp_replay_buffer_size'], device)
            self.amp_data = module_dict['amp_data']
            self.amp_normalizer = module_dict['amp_normalizer']
            # AMP parameters
            self.amp_disc_grad_penalty = train_cfg_dict['amp']['amp_disc_grad_penalty']

            amp_params = [
                {
                    "params": self.amp_discriminator.trunk.parameters(),
                    "lr": train_cfg_dict['ppo_algorithm']['learning_rate'],
                    "weight_decay": 10e-4,
                    "name": "amp_trunk",
                },
                {
                    "params": self.amp_discriminator.amp_linear.parameters(),
                    "lr": train_cfg_dict['ppo_algorithm']['learning_rate'],
                    "weight_decay": 10e-2,
                    "name": "amp_head",
                },
            ]
            actor_critic_params += amp_params
        if self.use_vae:
            # VAE components
            self.vae = module_dict['vae']
            self.vae.to(self.device)
            # VAE parameters
            self.vae_beta = train_cfg_dict['vae']['beta']
            self.vae_beta_adaptive = train_cfg_dict['vae']['beta_adaptive']
            self.vae_beta_max_step = train_cfg_dict['vae']['beta_max_step']
            self.vae_beta_max = train_cfg_dict['vae']['beta_max']
            self.vae_beta_step = (self.vae_beta_max - self.vae_beta) / self.vae_beta_max_step
            self.vae_free_bits = train_cfg_dict['vae']['free_bits']
            self.vae_use_adaboot = train_cfg_dict['vae']['use_adaboot']
            if self.vae_use_adaboot:
                self.p_boot = 1.0
                self.p_boot_step = 1.0 / train_cfg_dict['vae']['adaboot_max_step']
            self.use_exclusive_lr = train_cfg_dict['vae']['use_exclusive_lr'] if 'use_exclusive_lr' in train_cfg_dict['vae'] else False
            self.learning_rate_vae = train_cfg_dict['vae']['learning_rate']
            if self.use_vae_exclusive_optimizer:
                self.optimizer_dict['vae'] = optim.Adam(self.vae.parameters(), lr=self.learning_rate_vae)
            else:
                if self.use_exclusive_lr:
                    lr = self.learning_rate_vae
                else:
                    lr = train_cfg_dict['ppo_algorithm']['learning_rate']
                vae_params = [
                    {"params": self.vae.parameters(), "lr": lr, "name": "vae"},
                ]
                actor_critic_params += vae_params

        # PPO optimizer
        self.optimizer_dict['actor_critic'] = optim.Adam(actor_critic_params)

        # Create rollout storage
        self.storage: RolloutStorageLocomotion = None  # type: ignore
        self.transition = RolloutStorageLocomotion.Transition()

        # PPO parameters
        self.clip_param = train_cfg_dict['ppo_algorithm']['clip_param']
        self.num_learning_epochs = train_cfg_dict['ppo_algorithm']['num_learning_epochs']
        self.num_mini_batches = train_cfg_dict['ppo_algorithm']['num_mini_batches']
        self.value_loss_coef = train_cfg_dict['ppo_algorithm']['value_loss_coef']
        self.entropy_coef = train_cfg_dict['ppo_algorithm']['entropy_coef']
        self.gamma = train_cfg_dict['ppo_algorithm']['gamma']
        self.lam = train_cfg_dict['ppo_algorithm']['lam']
        self.max_grad_norm = train_cfg_dict['ppo_algorithm']['max_grad_norm']
        self.use_clipped_value_loss = train_cfg_dict['ppo_algorithm']['use_clipped_value_loss']
        self.desired_kl = train_cfg_dict['ppo_algorithm']['desired_kl']
        self.schedule = train_cfg_dict['ppo_algorithm']['schedule']
        self.learning_rate = train_cfg_dict['ppo_algorithm']['learning_rate']
        self.normalize_advantage_per_mini_batch = train_cfg_dict['ppo_algorithm']['normalize_advantage_per_mini_batch']

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs,
        action_shape,
        device,
    ):
        # create rollout storage
        self.storage = RolloutStorageLocomotion(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            action_shape,
            device,
        )

    def act(self, obs):
        if self.use_vae:
            # The decoder output is only used by the VAE loss in update().
            vae_out = self.vae(obs['estimator_obs'], decode=False)
            vae_code = vae_out['code'].detach()
            if self.vae_use_adaboot:
                if self.p_boot > 0.0:
                    gt_out = torch.cat((obs['gt_vel_obs'], obs['gt_com_obs'], obs['gt_mass_obs'], vae_out['code_latent']), dim=-1).detach()
                    batch_num = gt_out.shape[0]
                    device = gt_out.device
                    k = int(self.p_boot * batch_num)
                    idx = torch.randperm(batch_num, device=device)[:k]
                    vae_code[idx, ...] = gt_out[idx, ...]
            obs['estimator_out'] = vae_code.detach()
        # compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, amp_obs, next_amp_obs, next_actor_obs):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.observations["next_obs"] = next_actor_obs

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
        if self.use_amp and next_amp_obs is not None:
            self.amp_storage.insert(amp_obs, next_amp_obs)

        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, obs):
        # compute value for the last step
        last_values = self.actor_critic.evaluate(obs)
        self.storage.compute_returns(
            last_values.detach(), self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):  # noqa: C901
        # -- For PPO
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0

        # -- For AMP
        if self.use_amp:
            mean_amp_loss = 0
            mean_grad_pen_loss = 0
            mean_policy_pred = 0
            mean_expert_pred = 0

        # -- For VAE
        if self.use_vae:
            mean_vae_loss = 0
            mean_vae_vel_loss = 0
            mean_vae_mass_loss = 0
            mean_vae_com_loss = 0
            mean_vae_decode_loss = 0
            mean_vae_kl_loss = 0
            mean_vae_beta = 0
            mean_kl_raw = 0.0
            mean_kl_per_dim_mean = 0.0
            mean_mu_abs_mean = 0.0
            mean_sigma_mean = 0.0
            mean_kl_under_fb_frac = 0.0

        # generator for mini batches
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
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
            data_generator = generator

        # iterate over batches
        for sample_step in data_generator:
            if self.use_amp:
                sample, sample_amp_policy, sample_amp_expert = sample_step
            else:
                sample = sample_step

            (
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                _,
                _,
            ) = sample

            # original batch size
            original_batch_size = obs_batch["actor_obs"].shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.actor_critic.act(obs_batch)

            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.actor_critic.evaluate(
                obs_batch,
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

                    # all_reduce leaves the same averaged KL on every rank, so
                    # every process makes the same deterministic LR decision.
                    # A second broadcast of that scalar is redundant.
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer_dict['actor_critic'].param_groups:
                        param_group["lr"] = self.learning_rate
                        if self.use_vae:
                            if self.use_exclusive_lr and param_group["name"] == "vae":
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

            loss_dict = {}
            loss_dict['actor_critic'] = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            if self.use_amp:
                # AMP Discriminator loss.
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

                loss_dict['actor_critic'] += amp_loss + grad_pen_loss

            if self.use_vae:
                vae_out = self.vae(obs_batch['estimator_obs'])
                vae_vel_target = obs_batch["gt_vel_obs"].detach()
                vae_mass_target = obs_batch["gt_mass_obs"].detach()
                vae_com_target = obs_batch["gt_com_obs"].detach()
                vae_decode_target = obs_batch["next_obs"].detach()
                loss_recon_vel = nn.MSELoss()(vae_out["code_vel"], vae_vel_target)
                loss_recon_mass = nn.MSELoss()(vae_out["code_mass"], vae_mass_target)
                loss_recon_com = nn.MSELoss()(vae_out["code_com"], vae_com_target)
                loss_recon_decode = nn.MSELoss()(vae_out["decoded"], vae_decode_target)
                loss_recon = loss_recon_vel + loss_recon_mass + loss_recon_com + loss_recon_decode

                mu, lv = vae_out["mean_latent"].float(), vae_out["logvar_latent"].float()
                kl_pd = 0.5 * (mu.pow(2) + lv.exp() - 1.0 - lv)    # [B, D]
                kl_dim = kl_pd.mean(0)
                kl_fb = (kl_dim - self.vae_free_bits).clamp_min(0).sum()

                kl_loss_obs = self.vae_beta * kl_fb

                kl_raw_total = kl_pd.sum(1).mean()
                kl_per_dim_mean = kl_dim.mean()
                mu_abs_mean = mu.abs().mean()
                sigma_mean = (0.5 * lv).exp().mean()
                under_fb_frac = (kl_dim < self.vae_free_bits).float().mean()

                if self.use_vae_exclusive_optimizer:
                    loss_dict['vae'] = loss_recon + kl_loss_obs
                else:
                    loss_dict['actor_critic'] += loss_recon + kl_loss_obs

            # Compute the gradients
            for opt in self.optimizer_dict.values():
                opt.zero_grad(set_to_none=True)

            for key, opt in self.optimizer_dict.items():
                loss_dict[key].backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            for key, opt in self.optimizer_dict.items():
                params = [p for pg in opt.param_groups for p in pg["params"] if p.grad is not None]
                nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                opt.step()

            if self.use_amp:
                # Preserve the two NumPy updates (and therefore their exact
                # running-statistics order), but synchronize/copy from CUDA once.
                policy_batch_size = policy_state_unnorm.shape[0]
                normalizer_batch = torch.cat(
                    (policy_state_unnorm, expert_state_unnorm), dim=0
                ).cpu().numpy()
                self.amp_normalizer.update(
                    normalizer_batch[:policy_batch_size]
                )
                self.amp_normalizer.update(
                    normalizer_batch[policy_batch_size:]
                )

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

            if self.use_vae:
                mean_vae_loss += loss_recon.item()
                mean_vae_vel_loss += loss_recon_vel.item()
                mean_vae_mass_loss += loss_recon_mass.item()
                mean_vae_com_loss += loss_recon_com.item()
                mean_vae_decode_loss += loss_recon_decode.item()
                mean_vae_beta += self.vae_beta
                mean_vae_kl_loss += kl_loss_obs.item()
                mean_kl_raw += kl_raw_total.item()
                mean_kl_per_dim_mean += kl_per_dim_mean.item()
                mean_mu_abs_mean += mu_abs_mean.item()
                mean_sigma_mean += sigma_mean.item()
                mean_kl_under_fb_frac += under_fb_frac.item()

        if self.use_vae:
            if self.vae_beta_adaptive:
                if self.vae_beta < self.vae_beta_max:
                    self.vae_beta += self.vae_beta_step
            if self.vae_use_adaboot:
                if self.p_boot > 0.0:
                    self.p_boot -= self.p_boot_step

        # -- Clear the storage
        self.storage.clear()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        loss_debug_dict = {
            "value_function": self.value_loss_coef * mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": self.entropy_coef * mean_entropy,
        }
        # -- For AMP
        if self.use_amp:
            mean_amp_loss /= num_updates
            mean_grad_pen_loss /= num_updates
            mean_policy_pred /= num_updates
            mean_expert_pred /= num_updates

            loss_debug_dict_amp = {
                "amp_loss": mean_amp_loss,
                "grad_pen_loss": mean_grad_pen_loss,
                "policy_pred": mean_policy_pred,
                "expert_pred": mean_expert_pred,
            }

            loss_debug_dict.update(loss_debug_dict_amp)

        # -- For VAE
        if self.use_vae:
            mean_vae_loss /= num_updates
            mean_vae_vel_loss /= num_updates
            mean_vae_mass_loss /= num_updates
            mean_vae_com_loss /= num_updates
            mean_vae_decode_loss /= num_updates
            mean_vae_beta /= num_updates
            mean_vae_kl_loss /= num_updates
            mean_kl_raw /= num_updates
            mean_kl_per_dim_mean /= num_updates
            mean_mu_abs_mean /= num_updates
            mean_sigma_mean /= num_updates
            mean_kl_under_fb_frac /= num_updates

            loss_debug_dict_vae = {
                "vae_loss": mean_vae_loss,
                "vae_vel_loss": mean_vae_vel_loss,
                "vae_mass_loss": mean_vae_mass_loss,
                "vae_com_loss": mean_vae_com_loss,
                "vae_decode_loss": mean_vae_decode_loss,
                "vae_beta": mean_vae_beta,
                "vae_kl_raw": mean_kl_raw,
                "vae_kl_per_dim_mean": mean_kl_per_dim_mean,
                "vae_mu_abs_mean": mean_mu_abs_mean,
                "vae_sigma_mean": mean_sigma_mean,
                "vae_kl_under_fb_frac": mean_kl_under_fb_frac,
                "vae_kl_loss": mean_vae_kl_loss,
            }

            loss_debug_dict.update(loss_debug_dict_vae)

        return loss_debug_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast actor_critic, vae and amp_discriminator parameters to all GPUs."""
        to_sync = {"actor_critic": self.actor_critic.state_dict()}
        if self.use_amp:
            to_sync.update({"amp_disc": self.amp_discriminator.state_dict()})
        if self.use_vae:
            to_sync.update({"vae": self.vae.state_dict()})

        # 用 object_list 将其广播
        obj_list = [to_sync]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        synced = obj_list[0]
        # 把广播回来的参数加载回各自模块
        self.actor_critic.load_state_dict(synced["actor_critic"])
        if self.use_amp:
            self.amp_discriminator.load_state_dict(synced["amp_disc"])
        if self.use_vae:
            self.vae.load_state_dict(synced["vae"])

    def reduce_parameters(self):
        """Collect and average gradients from all GPUs for all three modules."""

        # 1) 收集所有要归约的梯度向量
        grads = []
        modules = {}
        modules.update({"actor_critic": self.actor_critic})
        if self.use_amp:
            modules.update({"amp_discriminator": self.amp_discriminator})
        if self.use_vae:
            modules.update({"vae": self.vae})
        for module in modules.values():
            for p in module.parameters():
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
        if len(grads) == 0:
            return
        all_grads = torch.cat(grads)

        # 2) 全局平均
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # 3) 把归约后的梯度写回每个参数
        offset = 0
        for module in modules.values():
            for p in module.parameters():
                if p.grad is not None:
                    numel = p.numel()
                    p.grad.data.copy_(all_grads[offset : offset + numel].view_as(p.grad.data))
                    offset += numel

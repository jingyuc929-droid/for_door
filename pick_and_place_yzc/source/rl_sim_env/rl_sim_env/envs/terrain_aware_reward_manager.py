# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""支持地形相关权重的奖励管理器。"""

from __future__ import annotations

import torch
from isaaclab.managers import RewardManager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TerrainAwareRewardManager(RewardManager):
    """扩展的奖励管理器，支持根据地形类型使用不同的权重。
    
    此管理器在标准 RewardManager 的基础上添加了以下功能：
    - 支持为每个奖励项配置按地形类型的权重列表
    - 在计算奖励时，根据每个环境的地形类型应用相应的权重
    
    使用方法：
        在配置文件中，将奖励项的 weight 设置为列表即可：
        'reward_term': {
            'weight': [0, 0, 0, 0, 0, 0, 1.5],  # 7种地形对应7个权重
            'params': {...}
        }
    """

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        """初始化地形感知的奖励管理器。
        
        Args:
            cfg: 奖励配置对象
            env: 环境实例
        """
        super().__init__(cfg, env)
        
        # 存储地形相关的权重 (term_name -> tensor of shape (num_envs,))
        self.terrain_reward_weights: dict[str, torch.Tensor] = {}
        
        # 在初始化后设置地形权重
        self._setup_terrain_weights()
    
    def _setup_terrain_weights(self):
        """扫描所有奖励项，设置地形相关的权重。"""
        if not hasattr(self._env, 'terrain_types'):
            print("[WARNING] 环境没有 terrain_types 属性，无法使用地形相关权重！")
            return
        
        print(f"[DEBUG] 开始设置地形相关权重，共 {len(self._term_names)} 个奖励项")
        
        for term_idx, (term_name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            # 调试：打印所有奖励项的权重信息
            has_terrain_weights = hasattr(term_cfg, 'terrain_weights')
            print(f"[DEBUG] 奖励项 '{term_name}': weight={term_cfg.weight}, has_terrain_weights={has_terrain_weights}")
            
            if has_terrain_weights:
                terrain_weights_list = term_cfg.terrain_weights
                print(f"[DEBUG]   terrain_weights_list = {terrain_weights_list}")
                
                # 为每个环境根据其地形类型创建权重张量
                env_weights = torch.zeros(
                    self._env.num_envs, 
                    device=self._env.device, 
                    dtype=torch.float32
                )
                
                for terrain_idx, weight_value in enumerate(terrain_weights_list):
                    mask = (self._env.terrain_types == terrain_idx)
                    num_envs_this_terrain = mask.sum().item()
                    env_weights[mask] = float(weight_value)
                    print(f"[DEBUG]   地形类型 {terrain_idx}: {num_envs_this_terrain} 个环境, 权重={weight_value}")
                
                self.terrain_reward_weights[term_name] = env_weights
                print(f"[INFO] 奖励项 '{term_name}' 配置地形相关权重: {terrain_weights_list}")
                print(f"[DEBUG]   最终权重统计: min={env_weights.min()}, max={env_weights.max()}, mean={env_weights.mean():.4f}")
        
        print(f"[INFO] 地形相关权重设置完成，共 {len(self.terrain_reward_weights)} 个项使用地形权重")
    
    def compute(self, dt: float) -> torch.Tensor:
        """计算所有奖励项的总和，对于配置了地形权重的项使用环境特定的权重。
        
        此方法重写了父类的 compute 方法，主要改动：
        - 对于配置了 terrain_weights 的项，使用每个环境特定的权重
        - 对于普通项，保持原版行为不变
        
        Args:
            dt: 时间步长（秒）
        
        Returns:
            每个环境的总奖励值，形状为 (num_envs,)
        """
        # 重置奖励缓冲区
        self._reward_buf[:] = 0.0
        
        # 遍历所有奖励项
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            # 判断是否使用地形相关权重
            if name in self.terrain_reward_weights:
                # 使用地形相关权重（每个环境不同）
                # 检查是否所有环境的权重都为0（优化：跳过计算）
                if torch.all(self.terrain_reward_weights[name] == 0.0):
                    self._step_reward[:, term_idx] = 0.0
                    continue
                
                # 计算原始奖励值
                value = term_cfg.func(self._env, **term_cfg.params)
                # 应用地形相关权重和时间步长
                weighted_value = value * self.terrain_reward_weights[name] * dt
                # 更新总奖励
                self._reward_buf += weighted_value
                # 更新累积奖励
                self._episode_sums[name] += weighted_value
                # 更新当前步骤奖励（用于可视化，不含dt）
                self._step_reward[:, term_idx] = value * self.terrain_reward_weights[name]
            else:
                # 使用标准权重（原版逻辑）
                # 检查：如果配置了 terrain_weights 但没有被正确加载，发出警告
                if hasattr(term_cfg, 'terrain_weights') and name not in self.terrain_reward_weights:
                    print(f"[WARNING] 奖励项 '{name}' 有 terrain_weights 属性但未被正确加载到地形权重字典！")
                    print(f"[WARNING]   将使用标量权重 {term_cfg.weight} 作为fallback")
                
                # 如果权重为0，跳过计算（微优化）
                if term_cfg.weight == 0.0:
                    self._step_reward[:, term_idx] = 0.0
                    continue
                
                # 计算带权重的奖励值
                value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
                # 更新总奖励
                self._reward_buf += value
                # 更新累积奖励
                self._episode_sums[name] += value
                # 更新当前步骤奖励
                self._step_reward[:, term_idx] = value / dt
        
        return self._reward_buf


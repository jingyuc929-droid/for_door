# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os

import numpy as np
import torch
from tensordict import TensorDict

from rl_algorithms.rsl_rl.modules import ActorCriticMARGlocomotion, EstimatorNet
from rl_algorithms.rsl_rl.modules import PIEEstimatorNet, ActorCriticPIElocomotion
from rl_algorithms.rsl_rl.modules import VAEBlind, ActorCriticEncoder


def _collect_joint_pd_gains_in_joint_order(robot) -> tuple[list[float], list[float]]:
    """按 `robot.joint_names` 的顺序收集所有 actuator 的 stiffness/damping。

    说明：
    - 过去这里硬编码只取 `base_legs`，导致像 grq20_v2d4_x5 这种“腿+臂”机器人导出时
      `joint_kp/joint_kd` 维度不完整（只有 12）。
    - 这里改为遍历 `robot.actuators`，按关节名对齐合并，确保长度与 joint_names 完全一致。
    """
    joint_names = list(robot.joint_names)
    kp_out: list[float | None] = [None] * len(joint_names)
    kd_out: list[float | None] = [None] * len(joint_names)

    def _indices_to_list(idxs, n: int) -> list[int]:
        """把 actuator 的 joint index 表达统一成 list[int]（兼容 slice/int/list/np/torch）。"""
        if idxs is None:
            return []
        if isinstance(idxs, slice):
            return list(range(n))[idxs]
        if isinstance(idxs, (int,)):
            return [int(idxs)]
        try:
            return [int(i) for i in list(idxs)]
        except Exception:
            return []

    # 遍历所有 actuator，把 stiffness/damping 写回对应 joint 的位置
    for actuator in robot.actuators.values():
        # IsaacLab 常用字段：joint_indices（对应全局 joint 索引）
        joint_ids = getattr(actuator, "joint_indices", None)
        joint_ids_list = _indices_to_list(joint_ids, len(joint_names))

        # 兼容其它实现：joint_ids / joint_names
        if not joint_ids_list:
            joint_ids = getattr(actuator, "joint_ids", None)
            joint_ids_list = _indices_to_list(joint_ids, len(joint_names))
        if not joint_ids_list:
            actuator_joint_names = getattr(actuator, "joint_names", None)
            if actuator_joint_names is not None:
                # 关节数很小，线性查找足够稳健
                joint_ids_list = [joint_names.index(n) for n in list(actuator_joint_names)]
        if not joint_ids_list:
            continue

        kp_vals = actuator.stiffness[0].detach().cpu().numpy().reshape(-1)
        kd_vals = actuator.damping[0].detach().cpu().numpy().reshape(-1)

        if len(kp_vals) != len(joint_ids_list) or len(kd_vals) != len(joint_ids_list):
            raise RuntimeError(
                f"Actuator gains size mismatch: kp={len(kp_vals)}, kd={len(kd_vals)}, joints={len(joint_ids_list)}"
            )
        for i, jid in enumerate(joint_ids_list):
            kp_out[jid] = float(f"{float(kp_vals[i]):.4f}")
            kd_out[jid] = float(f"{float(kd_vals[i]):.4f}")

    # 若仍有未覆盖的关节，直接报错，避免导出一个“维度不一致”的 policy.yaml
    missing = [name for name, v in zip(joint_names, kp_out) if v is None]
    if missing:
        raise RuntimeError(f"Missing actuator gains for joints: {missing}")

    # mypy: 已确保无 None
    return [float(x) for x in kp_out], [float(x) for x in kd_out]


def export_policy_as_jit(actor_critic: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        actor_critic: The actor-critic torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporter(actor_critic, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    """Export policy into a Torch ONNX file.

    Args:
        actor_critic: The actor-critic torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


def export_amp_vae_policy_as_onnx(
    actor: object,
    vae: object,
    path: str,
    filename="policy.onnx",
    verbose=False,
):

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    policy_exporter = OnnxAmpVaeExporter(actor, vae, verbose)
    policy_exporter.export(path, filename)


def export_amp_vae_vit_policy_as_onnx(
    actor: object,
    vae: object,
    path: str,
    filename="policy.onnx",
    verbose=False,
):

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    policy_exporter = OnnxAmpVaeVitExporter(actor, vae, verbose)
    policy_exporter.export(path, filename)


def export_locomotion_policy_as_onnx(
    actor: ActorCriticEncoder,
    vae: VAEBlind,
    path: str,
    filename="policy.onnx",
    verbose=False,
):

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    policy_exporter = OnnxLocomotionExporter(actor, vae, verbose)
    policy_exporter.export(path, filename)


def export_PIElocomotion_policy_as_onnx(
    actor_critic: ActorCriticPIElocomotion,
    PIE_estimator_net: PIEEstimatorNet,
    path: str,
    filename="policy.onnx",
    verbose=False,
):

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    policy_exporter = OnnxPIELocomotionExporter(actor_critic, PIE_estimator_net, verbose)
    policy_exporter.export(path, filename)


def export_MARGlocomotion_policy_as_onnx(
    actor_critic: ActorCriticMARGlocomotion,
    estimator_net: EstimatorNet,
    path: str,
    filename="policy.onnx",
    verbose=False,
):

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    policy_exporter = OnnxMARGlocomotionExporter(actor_critic, estimator_net, verbose)
    policy_exporter.export(path, filename)


def export_amp_vae_perception_policy_as_onnx(
    actor: object,
    vae: object,
    path: str,
    filename="policy.onnx",
    verbose=False,
):

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    policy_exporter = OnnxAmpVaeExporter(actor, vae, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, actor_critic, normalizer=None):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.register_buffer(
                "hidden_state",
                torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size),
            )
            self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.forward = self.forward_lstm
            self.reset = self.reset_memory
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward(self, x):
        return self.actor(self.normalizer(x))

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, actor_critic, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.forward = self.forward_lstm
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward(self, x):
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            actions, h_out, c_out = self(obs, h_in, c_in)
            torch.onnx.export(
                self,
                (obs, h_in, c_in),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            obs = torch.zeros(1, self.actor[0].in_features)
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )


class OnnxAmpVaeVitExporter(torch.nn.Module):
    """
    Exporter of combined VAE encoder and Actor network to a single ONNX file,
    integrating VAE output into Actor input, with dynamic batch support.
    """

    def __init__(self, actor, vae, verbose=False):
        super().__init__()
        self.verbose = verbose
        # Deep copy sub-modules to CPU
        self.actor = copy.deepcopy(actor).cpu().eval()
        self.vae = copy.deepcopy(vae).cpu().eval()

    def forward(self, actor_obs, prop_history, point_history, h_prev):
        # 1) VAE encoding (deterministic)
        code, h_new, hm_fine = self.vae.cenet_forward_export(prop_history, point_history, h_prev)
        # 2) Concatenate VAE code and actor observations
        full_obs = torch.cat((code, actor_obs), dim=-1)
        # 3) Actor network produces actions
        actions = self.actor.act_inference(full_obs)
        return actions, h_new, hm_fine

    def export(self, path, filename):
        """Export the combined model to ONNX at path/filename"""
        self.cpu().eval()
        full_path = os.path.join(path, filename)

        # 1) Infer full input dimension for actor (code + actor_obs)
        actor_in = None
        for module in self.actor.modules():
            if isinstance(module, torch.nn.Linear):
                actor_in = module.in_features
                break
        if actor_in is None:
            raise RuntimeError("Unable to infer actor input dimension")

        # 2) Infer VAE observation dimension from first Linear in encoder
        prop_in_dim = self.vae.prop_encoder[0].in_features
        C, H, W = self.vae.point_history_in_dim, self.vae.hmap_h, self.vae.hmap_w
        point_in_dim = C * H * W
        h_size = self.vae.heightmap_gru.hidden_size
        num_layers = self.vae.heightmap_gru.num_layers

        K = 1  # 你的推理是 K=1
        dummy_prop = torch.zeros(K, prop_in_dim, dtype=torch.float32)
        dummy_point = torch.zeros(K, point_in_dim, dtype=torch.float32)
        dummy_hprev = torch.zeros(num_layers, K, h_size, dtype=torch.float32)

        # code_dim 通过一次前向拿到
        code, _, _ = self.vae.cenet_forward_export(dummy_prop, dummy_point, dummy_hprev)
        code_dim = int(code.shape[-1])

        # 4) Actor obs dimension = actor_in - code_dim
        actor_obs_dim = actor_in - code_dim
        if actor_obs_dim <= 0:
            raise RuntimeError(f"Inferred actor_obs_dim={actor_obs_dim} invalid")
        dummy_actor = torch.zeros(1, actor_obs_dim)

        # if dynamic_batch:
        #     dynamic_axes = {
        #         "actor_obs": {0: "K"},
        #         "prop_t":    {0: "K"},
        #         "point_t":   {0: "K"},
        #         "h_prev":    {1: "K"},
        #         "actions":   {0: "K"},
        #         "h_new":     {1: "K"},
        #         "hm_fine":   {0: "K"},
        #     }
        # else:
        #     dynamic_axes = None  # 完全静态 K=1，通常更快
        # Export to ONNX
        torch.onnx.export(
            self,
            (dummy_actor, dummy_prop, dummy_point, dummy_hprev),
            full_path,
            export_params=True,
            do_constant_folding=True,
            opset_version=19,
            verbose=self.verbose,
            input_names=["actor_obs", "prop_t", "point_t", "h_prev"],
            output_names=["actions", "h_new", "hm_fine"],
            dynamic_axes=None,
        )
        print(f"Saved ONNX combined Actor+VIT model to {full_path}")


class OnnxMARGlocomotionExporter(torch.nn.Module):
    """
    Exporter of MARGlocomotion policy to a single ONNX file.
    """

    def __init__(
        self,
        actor_critic: ActorCriticMARGlocomotion,
        estimator_net: EstimatorNet,
        verbose=False,
    ):
        super().__init__()
        self.verbose = verbose
        # Deep copy sub-modules to CPU
        self.actor_critic = copy.deepcopy(actor_critic).cpu().eval()
        self.estimator_net = copy.deepcopy(estimator_net).cpu().eval()

    def forward(self, estimator_net_obs, actor_obs, gt_heightmap_obs):
        # 1) VAE encoding (deterministic)
        estimator_net_out = self.estimator_net.act_inference(estimator_net_obs)
        # 2) Concatenate VAE code and actor observations
        # full_obs = torch.cat((estimator_net_out, actor_obs, gt_heightmap_obs), dim=-1)
        # 3) Actor network produces actions
        full_obs_dict = {
            "estimator_net_out": estimator_net_out.detach(),
            "actor_obs": actor_obs.detach(),
            "gt_heightmap_obs": gt_heightmap_obs.detach(),
        }
        full_obs_dict = TensorDict(full_obs_dict, batch_size=actor_obs.shape[:-1], device="cpu")
        actions = self.actor_critic.act_inference(full_obs_dict)
        return actions

    def export(self, path, filename):
        """Export the combined model to ONNX at path/filename"""
        self.cpu()
        full_path = os.path.join(path, filename)

        # 1) estimator_ne
        estimator_net_in = None
        for module in self.estimator_net.modules():
            if isinstance(module, torch.nn.Linear):
                estimator_net_in = module.in_features
                break
        if estimator_net_in is None:
            raise RuntimeError("Unable to infer Estimator Net input dimension")

        dummy_estimator_net = torch.zeros(1, estimator_net_in)
        with torch.no_grad():
            estimator_net_out = self.estimator_net.act_inference(dummy_estimator_net)
        estimator_net_out_dim = estimator_net_out.shape[1]

        # 2) heightmap
        heightmap_in = None
        for module in self.actor_critic.heightmap_encoder.modules():
            if isinstance(module, torch.nn.Linear):
                heightmap_in = module.in_features
                break
        if heightmap_in is None:
            raise RuntimeError("Unable to infer Heightmap input dimension")

        dummy_heightmap = torch.zeros(1, heightmap_in)
        with torch.no_grad():
            heightmap_out = self.actor_critic.heightmap_encoder(dummy_heightmap)
        heightmap_out_dim = heightmap_out.shape[1]

        # 3) actor
        actor_in = None
        for module in self.actor_critic.actor.modules():
            if isinstance(module, torch.nn.Linear):
                actor_in = module.in_features
                break
        if actor_in is None:
            raise RuntimeError("Unable to infer actor input dimension")

        dummy_actor = torch.zeros(1, actor_in - estimator_net_out_dim - heightmap_out_dim)
        # Specify dynamic axes for batch dimension
        dynamic_axes = {
            "estimator_net_obs": {0: "batch_size"},
            "actor_obs": {0: "batch_size"},
            "gt_heightmap_obs": {0: "batch_size"},
            "actions": {0: "batch_size"},
        }
        # Export to ONNX
        torch.onnx.export(
            self,
            (dummy_estimator_net, dummy_actor, dummy_heightmap),
            full_path,
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["estimator_net_obs", "actor_obs", "gt_heightmap_obs"],
            output_names=["actions"],
            dynamic_axes=dynamic_axes,
        )
        print(f"Saved ONNX combined Actor+VAE model to {full_path}")


class OnnxLocomotionExporter(torch.nn.Module):
    """
    Exporter of combined VAE encoder and Actor network to a single ONNX file,
    integrating VAE output into Actor input, with dynamic batch support.
    """

    def __init__(self, actor: ActorCriticEncoder, vae: VAEBlind, verbose=False):
        super().__init__()
        self.verbose = verbose
        # Deep copy sub-modules to CPU
        self.actor = copy.deepcopy(actor).cpu().eval()
        self.vae = copy.deepcopy(vae).cpu().eval()

    def forward(self, actor_obs, vae_obs):
        # 1) VAE encoding (deterministic)
        code = self.vae.act_inference(vae_obs)
        # 2) Concatenate VAE code and actor observations
        # full_obs = torch.cat((code, actor_obs), dim=-1)
        # 3) Actor network produces actions
        full_obs_dict = {
            "actor_obs": actor_obs,
            "estimator_out": code,
        }
        full_obs_dict = TensorDict(full_obs_dict, batch_size=actor_obs.shape[:-1])
        actions = self.actor.act_inference(full_obs_dict)
        return actions

    def export(self, path, filename):
        """Export the combined model to ONNX at path/filename"""
        self.cpu()
        full_path = os.path.join(path, filename)

        # 1) Infer full input dimension for actor (code + actor_obs)
        actor_in = None
        for module in self.actor.modules():
            if isinstance(module, torch.nn.Linear):
                actor_in = module.in_features
                break
        if actor_in is None:
            raise RuntimeError("Unable to infer actor input dimension")

        # 2) Infer VAE observation dimension from first Linear in encoder
        vae_in = None
        for module in self.vae.encoder.modules():
            if isinstance(module, torch.nn.Linear):
                vae_in = module.in_features
                break
        if vae_in is None:
            raise RuntimeError("Unable to infer VAE input dimension")

        # 3) Compute code dimension by running dummy through VAE
        dummy_vae = torch.zeros(1, vae_in)
        with torch.no_grad():
            code = self.vae.act_inference(dummy_vae)
        code_dim = code.shape[1]

        # 4) Actor obs dimension = actor_in - code_dim
        actor_obs_dim = actor_in - code_dim
        if actor_obs_dim <= 0:
            raise RuntimeError(f"Inferred actor_obs_dim={actor_obs_dim} invalid")
        dummy_actor = torch.zeros(1, actor_obs_dim)

        # Specify dynamic axes for batch dimension
        dynamic_axes = {
            "actor_obs": {0: "batch_size"},
            "vae_obs": {0: "batch_size"},
            "actions": {0: "batch_size"},
        }
        # Export to ONNX
        torch.onnx.export(
            self,
            (dummy_actor, dummy_vae),
            full_path,
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["actor_obs", "vae_obs"],
            output_names=["actions"],
            dynamic_axes=dynamic_axes,
        )
        print(f"Saved ONNX combined Actor+VAE model to {full_path}")


class OnnxPIELocomotionExporter(torch.nn.Module):
    """
    Exporter of combined PIE estimator net and Actor network to a single ONNX file,
    integrating PIE estimator net output into Actor input, with dynamic batch support.
    """

    def __init__(
        self,
        actor: ActorCriticPIElocomotion,
        PIE_estimator_net: PIEEstimatorNet,
        verbose=False,
    ):
        super().__init__()
        self.verbose = verbose
        # Deep copy sub-modules to CPU
        self.actor = copy.deepcopy(actor).cpu().eval()
        self.PIE_estimator_net = copy.deepcopy(PIE_estimator_net).cpu().eval()

    def forward(
        self,
        actor_obs,
        PIE_estimator_net_proprioceptive_obs,
        PIE_estimator_net_depth_images_obs,
        hidden_states,
    ):
        # 1) PIE estimator net encoding (deterministic)
        gru_encoded_concat, new_hidden_states = self.PIE_estimator_net.act_inference(
            PIE_estimator_net_proprioceptive_obs,
            PIE_estimator_net_depth_images_obs,
            hidden_states,
        )
        # 2) Concatenate PIE estimator net code and actor observations
        # full_obs = torch.cat((gru_encoded_concat, actor_obs), dim=-1)
        # 3) Actor network produces actions
        full_obs_dict = {
            "actor_obs": actor_obs,
            "PIE_estimator_net_encoder_out": gru_encoded_concat,
        }
        full_obs_dict = TensorDict(full_obs_dict, batch_size=actor_obs.shape[:-1])
        actions = self.actor.act_inference(full_obs_dict)
        return actions, new_hidden_states

    def export(self, path, filename):
        """导出组合模型到 ONNX (path/filename)

        输入与输出维度说明:
        - actor_obs: (N, D_actor)                N 为环境/批大小
        - PIE_estimator_net_proprioceptive_obs: (T, N, D_prop)   历史串联后的本体观测向量
        - PIE_estimator_net_depth_images_obs: (T, N, C*H*W)    深度图序列，通道为时间或多相机，注意形状必须是 (num_envs, input_channels_dim*height*width)
        - hidden_states: (num_layers, N, hidden_dim)
        - actions: (N, D_action)
        """
        self.cpu().eval()
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)

        # 1) 推断 actor 的输入维度（= PIE 编码维度 + actor_obs 维度）
        actor_in_dim = None
        for module in self.actor.modules():
            if isinstance(module, torch.nn.Linear):
                actor_in_dim = module.in_features
                break
        if actor_in_dim is None:
            raise RuntimeError("Unable to infer actor input dimension")

        # 2) 推断 PIE 估计器两个输入的维度
        # 2.1 本体观测（proprioceptive）输入维度来自其 MLP 编码器的第一层
        prop_in_dim = None
        for module in self.PIE_estimator_net.proprioceptive_obs_mlp_encoder.modules():
            if isinstance(module, torch.nn.Linear):
                prop_in_dim = module.in_features
                break
        if prop_in_dim is None:
            raise RuntimeError("Unable to infer PIE estimator proprioceptive input dimension")

        # 2.2 深度图输入通道数和空间尺寸来自 PIE_estimator_cfg
        cfg = self.PIE_estimator_net.PIE_estimator_cfg
        depth_in_channels = int(cfg["depth_images_cnn_encoder_input_channels"])
        dummy_H = int(cfg["depth_images_cnn_encoder_input_height"])
        dummy_W = int(cfg["depth_images_cnn_encoder_input_width"])

        # 时间序列长度（与 PIE_estimator_cfg 中的设定保持一致）
        time_steps = int(self.PIE_estimator_net.PIE_estimator_cfg["PIE_estimator_net_input_num_time_series_length"])

        # 3) 通过一次前向计算 PIE 编码输出维度（gru_encoded_concat 的长度）
        # 形状满足:
        #   - PIE_estimator_net_proprioceptive_obs: (T, N, D_prop)
        #   - PIE_estimator_net_depth_images_obs:  (T, N, C*H*W)
        # 这里导出时取 N=1，T=time_steps
        dummy_prop = torch.zeros(time_steps, 1, prop_in_dim, dtype=torch.float32)
        dummy_depth = torch.zeros(time_steps, 1, depth_in_channels * dummy_H * dummy_W, dtype=torch.float32)
        dummy_hidden_states = torch.zeros(
            self.PIE_estimator_net.PIE_estimator_cfg["gru_encoder_num_layers"],
            1,
            self.PIE_estimator_net.gru_encoder.gru_encoder_hidden_dim,
            dtype=torch.float32,
        )
        with torch.no_grad():
            pie_code, _ = self.PIE_estimator_net.act_inference(dummy_prop, dummy_depth, dummy_hidden_states)
        code_dim = int(pie_code.shape[-1])

        # 4) actor 的纯 actor_obs 输入维度 = actor_in_dim - PIE 编码维度
        actor_obs_dim = actor_in_dim - code_dim
        if actor_obs_dim <= 0:
            raise RuntimeError(f"Inferred actor_obs_dim={actor_obs_dim} invalid (actor_in={actor_in_dim}, code_dim={code_dim})")
        dummy_actor_obs = torch.zeros(1, actor_obs_dim, dtype=torch.float32)
        # 5) 设置动态维度：
        #   - actor_obs 和 actions: 第 0 维为 batch_size
        #   - PIE_estimator_net_*: 第 1 维为 batch_size（第 0 维为时间步 T，与配置保持一致）
        dynamic_axes = {
            "actor_obs": {0: "batch_size"},
            "PIE_estimator_net_proprioceptive_obs": {1: "batch_size"},
            "PIE_estimator_net_depth_images_obs": {1: "batch_size"},
            "hidden_states": {1: "batch_size"},
            "actions": {0: "batch_size"},
            "new_hidden_states": {1: "batch_size"},
        }

        # 6) 导出 ONNX
        torch.onnx.export(
            self,
            (dummy_actor_obs, dummy_prop, dummy_depth, dummy_hidden_states),
            full_path,
            export_params=True,
            opset_version=14,
            verbose=self.verbose,
            input_names=[
                "actor_obs",
                "PIE_estimator_net_proprioceptive_obs",
                "PIE_estimator_net_depth_images_obs",
                "hidden_states",
            ],
            output_names=["actions", "new_hidden_states"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
        print(f"Saved ONNX combined Actor+PIE Estimator model to {full_path}")


class OnnxAmpVaeExporter(torch.nn.Module):
    """
    Exporter of combined VAE encoder and Actor network to a single ONNX file,
    integrating VAE output into Actor input, with dynamic batch support.
    """

    def __init__(self, actor, vae, verbose=False):
        super().__init__()
        self.verbose = verbose
        # Deep copy sub-modules to CPU
        self.actor = copy.deepcopy(actor).cpu()
        self.vae = copy.deepcopy(vae).cpu()

    def forward(self, actor_obs, vae_obs):
        # 1) VAE encoding (deterministic)
        code = self.vae.act_inference(vae_obs)
        # 2) Concatenate VAE code and actor observations
        full_obs = torch.cat((code, actor_obs), dim=-1)
        # 3) Actor network produces actions
        actions = self.actor.act_inference(full_obs)
        return actions

    def export(self, path, filename):
        """Export the combined model to ONNX at path/filename"""
        self.cpu()
        full_path = os.path.join(path, filename)

        # 1) Infer full input dimension for actor (code + actor_obs)
        actor_in = None
        for module in self.actor.modules():
            if isinstance(module, torch.nn.Linear):
                actor_in = module.in_features
                break
        if actor_in is None:
            raise RuntimeError("Unable to infer actor input dimension")

        # 2) Infer VAE observation dimension from first Linear in encoder
        vae_in = None
        for module in self.vae.encoder.modules():
            if isinstance(module, torch.nn.Linear):
                vae_in = module.in_features
                break
        if vae_in is None:
            raise RuntimeError("Unable to infer VAE input dimension")

        # 3) Compute code dimension by running dummy through VAE
        dummy_vae = torch.zeros(1, vae_in)
        with torch.no_grad():
            code = self.vae.act_inference(dummy_vae)
        code_dim = code.shape[1]

        # 4) Actor obs dimension = actor_in - code_dim
        actor_obs_dim = actor_in - code_dim
        if actor_obs_dim <= 0:
            raise RuntimeError(f"Inferred actor_obs_dim={actor_obs_dim} invalid")
        dummy_actor = torch.zeros(1, actor_obs_dim)

        # Specify dynamic axes for batch dimension
        dynamic_axes = {
            "actor_obs": {0: "batch_size"},
            "vae_obs": {0: "batch_size"},
            "actions": {0: "batch_size"},
        }
        # Export to ONNX
        torch.onnx.export(
            self,
            (dummy_actor, dummy_vae),
            full_path,
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["actor_obs", "vae_obs"],
            output_names=["actions"],
            dynamic_axes=dynamic_axes,
        )
        print(f"Saved ONNX combined Actor+VAE model to {full_path}")


def export_inference_cfg(env, env_cfg, path, load_run, checkpoint):
    policy_cfg_dict = {}
    policy_cfg_dict["dt"] = env_cfg.decimation * env.unwrapped.physics_dt
    policy_cfg_dict["joint_names"] = env.unwrapped.scene.articulations["robot"].joint_names
    # 1. 直接拿到 numpy 数组
    default_joint_pos = env.unwrapped.scene.articulations["robot"]._data.default_joint_pos[0].cpu().numpy()

    # 2. （可选）保留 4 位小数，再转成 Python 列表
    policy_cfg_dict["default_joint_pos"] = np.round(default_joint_pos, 4).tolist()

    # 如果你要更精确地控制格式，比如总是输出 "-0.0500" 而不是 "-0.05"，也可以这样：
    policy_cfg_dict["default_joint_pos"] = [float(f"{x:.4f}") for x in default_joint_pos]
    policy_cfg_dict["input_names"] = ["actor_obs", "vae_obs"]
    policy_cfg_dict["output_names"] = ["actions"]
    policy_cfg_dict["input_actor_obs_names"] = env.unwrapped.observation_manager._group_obs_term_names["actor_obs"]
    policy_cfg_dict["input_vae_obs_names"] = env.unwrapped.observation_manager._group_obs_term_names["actor_obs"]
    input_actor_obs_scales = {}
    input_vae_obs_scales = {}
    input_obs_size_map = {}
    env_cfg = env.unwrapped.cfg.config_summary.env
    obs_cfg = env.unwrapped.cfg.config_summary.observation
    input_actor_obs_scales["base_ang_vel"] = obs_cfg.scale.base_ang_vel
    input_actor_obs_scales["projected_gravity"] = 1.0
    input_actor_obs_scales["velocity_commands"] = [
        obs_cfg.scale.base_lin_vel,
        obs_cfg.scale.base_lin_vel,
        obs_cfg.scale.base_ang_vel,
    ]
    input_actor_obs_scales["joint_pos"] = obs_cfg.scale.joint_pos
    input_actor_obs_scales["joint_vel"] = obs_cfg.scale.joint_vel
    input_actor_obs_scales["actions"] = 1.0

    input_vae_obs_scales["base_ang_vel"] = obs_cfg.scale.base_ang_vel
    input_vae_obs_scales["projected_gravity"] = 1.0
    input_vae_obs_scales["velocity_commands"] = [
        obs_cfg.scale.base_lin_vel,
        obs_cfg.scale.base_lin_vel,
        obs_cfg.scale.base_ang_vel,
    ]
    input_vae_obs_scales["joint_pos"] = obs_cfg.scale.joint_pos
    input_vae_obs_scales["joint_vel"] = obs_cfg.scale.joint_vel
    input_vae_obs_scales["actions"] = 1.0

    input_obs_size_map["actor_obs"] = env_cfg.num_actor_obs
    input_obs_size_map["vae_obs"] = env_cfg.num_vae_obs

    policy_cfg_dict["input_actor_obs_scales"] = input_actor_obs_scales
    policy_cfg_dict["input_vae_obs_scales"] = input_vae_obs_scales
    policy_cfg_dict["input_obs_size_map"] = input_obs_size_map
    policy_cfg_dict["action_scale"] = env.unwrapped.cfg.config_summary.action.scale
    policy_cfg_dict["clip_actions"] = env.unwrapped.cfg.config_summary.env.clip_actions
    policy_cfg_dict["clip_obs"] = env.unwrapped.cfg.config_summary.env.clip_obs
    actor_obs_history_length = env.unwrapped.observation_manager._group_obs_term_cfgs["actor_obs"][1].history_length
    vae_obs_history_length = env.unwrapped.cfg.config_summary.env.obs_history_length
    policy_cfg_dict["obs_history_length"] = {
        "actor_obs": actor_obs_history_length if actor_obs_history_length > 0 else 1,
        "vae_obs": vae_obs_history_length if vae_obs_history_length > 0 else 1,
    }
    robot = env.unwrapped.scene.articulations["robot"]
    joint_kp, joint_kd = _collect_joint_pd_gains_in_joint_order(robot)
    policy_cfg_dict["joint_kp"] = joint_kp
    policy_cfg_dict["joint_kd"] = joint_kd
    print("joint_names:", policy_cfg_dict["joint_names"])
    print("default_joint_pos:", policy_cfg_dict["default_joint_pos"])
    print("input_names:", policy_cfg_dict["input_names"])
    print("output_names:", policy_cfg_dict["output_names"])
    print("input_actor_obs_names:", policy_cfg_dict["input_actor_obs_names"])
    print("input_vae_obs_names:", policy_cfg_dict["input_vae_obs_names"])
    print("input_actor_obs_scales:", policy_cfg_dict["input_actor_obs_scales"])
    print("input_vae_obs_scales:", policy_cfg_dict["input_vae_obs_scales"])
    print("input_obs_size_map:", policy_cfg_dict["input_obs_size_map"])
    print("action_scale:", policy_cfg_dict["action_scale"])
    print("clip_actions:", policy_cfg_dict["clip_actions"])
    print("clip_obs:", policy_cfg_dict["clip_obs"])
    print("obs_history_length:", policy_cfg_dict["obs_history_length"])
    print("joint_kp:", policy_cfg_dict["joint_kp"])
    print("joint_kd:", policy_cfg_dict["joint_kd"])
    export_inference_cfg_to_yaml(policy_cfg_dict, path, load_run, checkpoint)


def export_inference_cfg_to_yaml(config_dict, path, load_run, checkpoint):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    readme_file_path = os.path.join(path, "policy.yaml")
    content = f'load_run: "{load_run}"\n'
    content += f'checkpoint: "{checkpoint}"\n'
    content += f"dt: {config_dict['dt']}\n"
    # joint_names 多行缩进
    content += "joint_names:\n  [\n"
    for name in config_dict["joint_names"]:
        content += f'    "{name}",\n'
    content += "  ]\n"

    # default_joint_pos 保留 4 位小数
    content += "default_joint_pos: ["
    content += ", ".join(f"{float(v):.4f}" for v in config_dict["default_joint_pos"])
    content += "]\n"

    # input_names 和 output_names
    content += "input_names: ["
    content += ", ".join(f'"{n}"' for n in config_dict["input_names"])
    content += "]\n"

    content += "output_names: ["
    content += ", ".join(f'"{n}"' for n in config_dict["output_names"])
    content += "]\n"

    # input_obs_names_map 多行缩进
    content += "input_obs_names_map:\n  {\n"
    for key, obs_list in (
        ("actor_obs", config_dict["input_actor_obs_names"]),
        ("vae_obs", config_dict["input_vae_obs_names"]),
    ):
        content += f"    {key}: ["
        content += ", ".join(f'"{o}"' for o in obs_list)
        content += "],\n"
    content += "  }\n"

    # input_obs_scales_map 多行缩进，并区分标量／列表
    content += "input_obs_scales_map:\n  {\n"
    for key, scales in (
        ("actor_obs", config_dict["input_actor_obs_scales"]),
        ("vae_obs", config_dict["input_vae_obs_scales"]),
    ):
        content += f"    {key}: {{ "
        parts = []
        for obs, val in scales.items():
            if isinstance(val, list):
                sval = "[" + ", ".join(f"{x}" for x in val) + "]"
            else:
                sval = f"{val}"
            parts.append(f"{obs}: {sval}")
        content += ", ".join(parts)
        content += " },\n"
    content += "  }\n"

    content += "input_obs_size_map:\n  {\n"
    for key, scales in config_dict["input_obs_size_map"].items():
        content += f"    {key}: {scales},\n"
    content += "  }\n"

    # 其余字段
    content += f"action_scale: {config_dict['action_scale']}\n"
    content += f"clip_actions: {config_dict['clip_actions']}\n"
    content += f"clip_obs: {config_dict['clip_obs']}\n"

    # obs_history_length
    content += "obs_history_length: { "
    content += ", ".join(f"{k}: {v}" for k, v in config_dict["obs_history_length"].items())
    content += " }\n"
    content += f"joint_kp: {config_dict['joint_kp']}\n"
    content += f"joint_kd: {config_dict['joint_kd']}\n"

    # 添加固定的速度scales
    content += "velocity_x_forward_scale: 1.0\n"
    content += "velocity_x_backward_scale: 1.0\n"
    content += "velocity_y_scale: 1.0\n"
    content += "velocity_yaw_scale: 1.0\n"

    # 添加固定的速度、加速度和加加速度上限
    content += "max_velocity: [1.0, 0.6, 1.5]\n"
    content += "max_acceleration: [3.0, 3, 6]\n"
    content += "max_jerk: [5, 5, 30]\n"

    # 添加固定的 threshold 配置
    content += "threshold:\n"
    content += "  limit_lower: -0.0\n"
    content += "  limit_upper: 0.0\n"
    content += "  damping: 5.0\n"

    with open(readme_file_path, "w", encoding="utf-8") as f:
        f.write(content)


def export_inference_cfg_amp_vae_vit(env, env_cfg, path, load_run, checkpoint):
    policy_cfg_dict = {}
    policy_cfg_dict["dt"] = env_cfg.decimation * env.unwrapped.physics_dt
    policy_cfg_dict["joint_names"] = env.unwrapped.scene.articulations["robot"].joint_names
    # 1. 直接拿到 numpy 数组
    default_joint_pos = env.unwrapped.scene.articulations["robot"]._data.default_joint_pos[0].cpu().numpy()

    # 2. （可选）保留 4 位小数，再转成 Python 列表
    policy_cfg_dict["default_joint_pos"] = np.round(default_joint_pos, 4).tolist()

    # 如果你要更精确地控制格式，比如总是输出 "-0.0500" 而不是 "-0.05"，也可以这样：
    policy_cfg_dict["default_joint_pos"] = [float(f"{x:.4f}") for x in default_joint_pos]
    policy_cfg_dict["input_names"] = ["actor_obs", "prop_t", "point_t", "h_prev"]
    policy_cfg_dict["output_names"] = ["actions", "h_new", "hm_fine"]
    policy_cfg_dict["input_actor_obs_names"] = env.unwrapped.observation_manager._group_obs_term_names["actor_obs"]
    policy_cfg_dict["input_prop_history_names"] = env.unwrapped.observation_manager._group_obs_term_names["actor_obs"]
    policy_cfg_dict["input_point_history_names"] = ["height_map"]
    policy_cfg_dict["input_h_prev_names"] = ["h_prev"]
    input_actor_obs_scales = {}
    input_prop_history_scales = {}
    input_point_history_scales = {}
    input_h_prev_scales = {}
    input_obs_size_map = {}
    env_cfg = env.unwrapped.cfg.config_summary.env
    obs_cfg = env.unwrapped.cfg.config_summary.observation
    input_actor_obs_scales["base_ang_vel"] = obs_cfg.scale.base_ang_vel
    input_actor_obs_scales["projected_gravity"] = 1.0
    input_actor_obs_scales["velocity_commands"] = [
        obs_cfg.scale.base_lin_vel,
        obs_cfg.scale.base_lin_vel,
        obs_cfg.scale.base_ang_vel,
    ]
    input_actor_obs_scales["joint_pos"] = obs_cfg.scale.joint_pos
    input_actor_obs_scales["joint_vel"] = obs_cfg.scale.joint_vel
    input_actor_obs_scales["actions"] = 1.0

    input_prop_history_scales["base_ang_vel"] = obs_cfg.scale.base_ang_vel
    input_prop_history_scales["projected_gravity"] = 1.0
    input_prop_history_scales["velocity_commands"] = [
        obs_cfg.scale.base_lin_vel,
        obs_cfg.scale.base_lin_vel,
        obs_cfg.scale.base_ang_vel,
    ]
    input_prop_history_scales["joint_pos"] = obs_cfg.scale.joint_pos
    input_prop_history_scales["joint_vel"] = obs_cfg.scale.joint_vel
    input_prop_history_scales["actions"] = 1.0

    input_point_history_scales["height_map"] = 1.0
    input_h_prev_scales["h_prev"] = 1.0

    input_obs_size_map["actor_obs"] = env_cfg.num_actor_obs
    input_obs_size_map["prop_t"] = env_cfg.prop_obs_dim
    input_obs_size_map["point_t"] = env.unwrapped.cfg.config_summary.env.num_heightmap_obs_h * env.unwrapped.cfg.config_summary.env.num_heightmap_obs_w
    input_obs_size_map["h_prev"] = 128

    policy_cfg_dict["input_actor_obs_scales"] = input_actor_obs_scales
    policy_cfg_dict["input_prop_history_scales"] = input_prop_history_scales
    policy_cfg_dict["input_point_history_scales"] = input_point_history_scales
    policy_cfg_dict["input_h_prev_scales"] = input_h_prev_scales
    policy_cfg_dict["input_obs_size_map"] = input_obs_size_map
    policy_cfg_dict["action_scale"] = env.unwrapped.cfg.config_summary.action.scale
    policy_cfg_dict["clip_actions"] = env.unwrapped.cfg.config_summary.env.clip_actions
    policy_cfg_dict["clip_obs"] = env.unwrapped.cfg.config_summary.env.clip_obs
    actor_obs_history_length = env.unwrapped.observation_manager._group_obs_term_cfgs["actor_obs"][1].history_length
    prop_t_history_length = env.unwrapped.cfg.config_summary.env.prop_obs_his
    point_t_history_length = env.unwrapped.cfg.config_summary.env.partial_hmap_obs_his
    h_prev_history_length = 1
    policy_cfg_dict["obs_history_length"] = {
        "actor_obs": actor_obs_history_length if actor_obs_history_length > 0 else 1,
        "prop_t": prop_t_history_length if prop_t_history_length > 0 else 1,
        "point_t": point_t_history_length if point_t_history_length > 0 else 1,
        "h_prev": h_prev_history_length if h_prev_history_length > 0 else 1,
    }
    robot = env.unwrapped.scene.articulations["robot"]
    joint_kp, joint_kd = _collect_joint_pd_gains_in_joint_order(robot)
    policy_cfg_dict["joint_kp"] = joint_kp
    policy_cfg_dict["joint_kd"] = joint_kd
    print("joint_names:", policy_cfg_dict["joint_names"])
    print("default_joint_pos:", policy_cfg_dict["default_joint_pos"])
    print("input_names:", policy_cfg_dict["input_names"])
    print("output_names:", policy_cfg_dict["output_names"])
    print("input_actor_obs_names:", policy_cfg_dict["input_actor_obs_names"])
    print("input_prop_history_names:", policy_cfg_dict["input_prop_history_names"])
    print("input_point_history_names:", policy_cfg_dict["input_point_history_names"])
    print("input_h_prev_names:", policy_cfg_dict["input_h_prev_names"])
    print("input_actor_obs_scales:", policy_cfg_dict["input_actor_obs_scales"])
    print("input_prop_history_scales:", policy_cfg_dict["input_prop_history_scales"])
    print("input_point_history_scales:", policy_cfg_dict["input_point_history_scales"])
    print("input_h_prev_scales:", policy_cfg_dict["input_h_prev_scales"])
    print("input_obs_size_map:", policy_cfg_dict["input_obs_size_map"])
    print("action_scale:", policy_cfg_dict["action_scale"])
    print("clip_actions:", policy_cfg_dict["clip_actions"])
    print("clip_obs:", policy_cfg_dict["clip_obs"])
    print("obs_history_length:", policy_cfg_dict["obs_history_length"])
    print("joint_kp:", policy_cfg_dict["joint_kp"])
    print("joint_kd:", policy_cfg_dict["joint_kd"])
    export_inference_cfg_to_yaml_amp_vae_vit(policy_cfg_dict, path, load_run, checkpoint)


def export_inference_cfg_to_yaml_amp_vae_vit(config_dict, path, load_run, checkpoint):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    readme_file_path = os.path.join(path, "policy.yaml")
    content = f'load_run: "{load_run}"\n'
    content += f'checkpoint: "{checkpoint}"\n'
    content += f"dt: {config_dict['dt']}\n"
    # joint_names 多行缩进
    content += "joint_names:\n  [\n"
    for name in config_dict["joint_names"]:
        content += f'    "{name}",\n'
    content += "  ]\n"

    # default_joint_pos 保留 4 位小数
    content += "default_joint_pos: ["
    content += ", ".join(f"{float(v):.4f}" for v in config_dict["default_joint_pos"])
    content += "]\n"

    # input_names 和 output_names
    content += "input_names: ["
    content += ", ".join(f'"{n}"' for n in config_dict["input_names"])
    content += "]\n"

    content += "output_names: ["
    content += ", ".join(f'"{n}"' for n in config_dict["output_names"])
    content += "]\n"

    # input_obs_names_map 多行缩进
    content += "input_obs_names_map:\n  {\n"
    for key, obs_list in (
        ("actor_obs", config_dict["input_actor_obs_names"]),
        ("prop_t", config_dict["input_prop_history_names"]),
        ("point_t", config_dict["input_point_history_names"]),
        ("h_prev", config_dict["input_h_prev_names"]),
    ):
        content += f"    {key}: ["
        content += ", ".join(f'"{o}"' for o in obs_list)
        content += "],\n"
    content += "  }\n"

    # input_obs_scales_map 多行缩进，并区分标量／列表
    content += "input_obs_scales_map:\n  {\n"
    for key, scales in (
        ("actor_obs", config_dict["input_actor_obs_scales"]),
        ("prop_t", config_dict["input_prop_history_scales"]),
        ("point_t", config_dict["input_point_history_scales"]),
        ("h_prev", config_dict["input_h_prev_scales"]),
    ):
        content += f"    {key}: {{ "
        parts = []
        for obs, val in scales.items():
            if isinstance(val, list):
                sval = "[" + ", ".join(f"{x}" for x in val) + "]"
            else:
                sval = f"{val}"
            parts.append(f"{obs}: {sval}")
        content += ", ".join(parts)
        content += " },\n"
    content += "  }\n"

    content += "input_obs_size_map:\n  {\n"
    for key, scales in config_dict["input_obs_size_map"].items():
        content += f"    {key}: {scales},\n"
    content += "  }\n"

    # 其余字段
    content += f"action_scale: {config_dict['action_scale']}\n"
    content += f"clip_actions: {config_dict['clip_actions']}\n"
    content += f"clip_obs: {config_dict['clip_obs']}\n"

    # obs_history_length
    content += "obs_history_length: { "
    content += ", ".join(f"{k}: {v}" for k, v in config_dict["obs_history_length"].items())
    content += " }\n"
    content += f"joint_kp: {config_dict['joint_kp']}\n"
    content += f"joint_kd: {config_dict['joint_kd']}\n"
    with open(readme_file_path, "w", encoding="utf-8") as f:
        f.write(content)


def export_inference_cfg_locomotion(env, env_cfg, path, load_run, checkpoint, config_path):
    # export config summary
    save_config_path = os.path.join(path, "config_summary.py")

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    configSummary_path = os.path.join(config_path, "config_summary.py")
    with open(configSummary_path, "r", encoding="utf-8") as file:
        content = file.read()
    with open(save_config_path, "w", encoding="utf-8") as file:
        file.write(content)
    policy_cfg_dict = {}
    policy_cfg_dict["dt"] = env_cfg.decimation * env.unwrapped.physics_dt
    policy_cfg_dict["joint_names"] = env.unwrapped.scene.articulations["robot"].joint_names
    # 1. 直接拿到 numpy 数组
    default_joint_pos = env.unwrapped.scene.articulations["robot"]._data.default_joint_pos[0].cpu().numpy()

    # 2. （可选）保留 4 位小数，再转成 Python 列表
    policy_cfg_dict["default_joint_pos"] = np.round(default_joint_pos, 4).tolist()

    # 如果你要更精确地控制格式，比如总是输出 "-0.0500" 而不是 "-0.05"，也可以这样：
    policy_cfg_dict["default_joint_pos"] = [float(f"{x:.4f}") for x in default_joint_pos]

    env_cfg = env.unwrapped.cfg.config_summary.env
    obs_cfg = env.unwrapped.cfg.config_summary.observation

    input_actor_obs_scales = {}
    input_vae_obs_scales = {}
    input_obs_size_map = {}
    policy_cfg_dict["obs_history_length"] = {}
    # 读取是否使用Vae
    policy_cfg_dict["output_names"] = ["actions"]
    policy_cfg_dict["input_actor_obs_names"] = obs_cfg.policy_obs_dict["actor_obs"]["terms"]
    input_obs_size_map["actor_obs"] = env_cfg.num_actor_obs
    if "history_length" in obs_cfg.policy_obs_dict["actor_obs"]:
        actor_obs_history_length = obs_cfg.policy_obs_dict["actor_obs"]["history_length"]
    else:
        actor_obs_history_length = 1
    for actor_obs_sub_term in policy_cfg_dict["input_actor_obs_names"]:
        if "nad" in actor_obs_sub_term:
            input_actor_obs_scales[actor_obs_sub_term] = (
                obs_cfg.obs_term_dict["noise_and_delay_obs"][actor_obs_sub_term]["scale"]
                if "scale" in obs_cfg.obs_term_dict["noise_and_delay_obs"][actor_obs_sub_term]
                else 1.0
            )
        elif "gt" in actor_obs_sub_term:
            input_actor_obs_scales[actor_obs_sub_term] = (
                obs_cfg.obs_term_dict["ground_truth_obs"][actor_obs_sub_term]["scale"]
                if "scale" in obs_cfg.obs_term_dict["ground_truth_obs"][actor_obs_sub_term]
                else 1.0
            )
        else:
            input_actor_obs_scales[actor_obs_sub_term] = None
        if isinstance(input_actor_obs_scales[actor_obs_sub_term], tuple):
            input_actor_obs_scales[actor_obs_sub_term] = list(input_actor_obs_scales[actor_obs_sub_term])

    policy_cfg_dict["input_actor_obs_scales"] = input_actor_obs_scales
    policy_cfg_dict["obs_history_length"]["actor_obs"] = actor_obs_history_length if actor_obs_history_length > 0 else 1
    print("type", type(input_actor_obs_scales["base_commands_gt"]).__name__)

    use_vae = env_cfg.train_cfg_dict["use_vae"]
    policy_cfg_dict["use_vae"] = use_vae
    if use_vae:
        policy_cfg_dict["input_names"] = ["actor_obs", "vae_obs"]
        policy_cfg_dict["input_vae_obs_names"] = obs_cfg.policy_obs_dict["estimator_obs"]["terms"]
        for vae_obs_sub_term in policy_cfg_dict["input_vae_obs_names"]:
            if "nad" in vae_obs_sub_term:
                input_vae_obs_scales[vae_obs_sub_term] = (
                    obs_cfg.obs_term_dict["noise_and_delay_obs"][vae_obs_sub_term]["scale"]
                    if "scale" in obs_cfg.obs_term_dict["noise_and_delay_obs"][vae_obs_sub_term]
                    else 1.0
                )
            elif "gt" in vae_obs_sub_term:
                input_vae_obs_scales[vae_obs_sub_term] = (
                    obs_cfg.obs_term_dict["ground_truth_obs"][vae_obs_sub_term]["scale"]
                    if "scale" in obs_cfg.obs_term_dict["ground_truth_obs"][vae_obs_sub_term]
                    else 1.0
                )
            else:
                input_vae_obs_scales[vae_obs_sub_term] = None
            if isinstance(input_vae_obs_scales[vae_obs_sub_term], tuple):
                input_vae_obs_scales[vae_obs_sub_term] = list(input_vae_obs_scales[vae_obs_sub_term])
        input_obs_size_map["vae_obs"] = env_cfg.num_estimator_step_obs
        policy_cfg_dict["input_vae_obs_scales"] = input_vae_obs_scales
        if "history_length" in obs_cfg.policy_obs_dict["estimator_obs"]:
            vae_obs_history_length = obs_cfg.policy_obs_dict["estimator_obs"]["history_length"]
        else:
            vae_obs_history_length = 5
        policy_cfg_dict["obs_history_length"]["vae_obs"] = vae_obs_history_length if vae_obs_history_length > 0 else 1
    else:
        policy_cfg_dict["input_names"] = ["actor_obs"]

    # 添加调试信息，查看可用的观察组名称
    print(
        "Available observation group names:",
        list(env.unwrapped.observation_manager._group_obs_term_names.keys()),
    )

    policy_cfg_dict["input_obs_size_map"] = input_obs_size_map
    policy_cfg_dict["action_scale"] = env.unwrapped.cfg.config_summary.action.scale
    policy_cfg_dict["clip_actions"] = env.unwrapped.cfg.config_summary.env.clip_actions
    policy_cfg_dict["clip_obs"] = env.unwrapped.cfg.config_summary.env.clip_obs

    robot = env.unwrapped.scene.articulations["robot"]
    joint_kp, joint_kd = _collect_joint_pd_gains_in_joint_order(robot)
    policy_cfg_dict["joint_kp"] = joint_kp
    policy_cfg_dict["joint_kd"] = joint_kd
    hip_torque_limit = env_cfg.hip_tor_limit
    thigh_torque_limit = env_cfg.thigh_tor_limit
    calf_torque_limit = env_cfg.calf_tor_limit
    policy_torque_limit = [hip_torque_limit, thigh_torque_limit, calf_torque_limit] * 4
    policy_cfg_dict["torque_limit"] = [float(f"{x:.4f}") for x in policy_torque_limit]

    print("joint_names:", policy_cfg_dict["joint_names"])
    print("default_joint_pos:", policy_cfg_dict["default_joint_pos"])
    print("input_names:", policy_cfg_dict["input_names"])
    print("output_names:", policy_cfg_dict["output_names"])
    print("input_actor_obs_names:", policy_cfg_dict["input_actor_obs_names"])
    if use_vae:
        print("input_vae_obs_names:", policy_cfg_dict["input_vae_obs_names"])
        print("input_vae_obs_scales:", policy_cfg_dict["input_vae_obs_scales"])
    print("input_actor_obs_scales:", policy_cfg_dict["input_actor_obs_scales"])
    print("input_obs_size_map:", policy_cfg_dict["input_obs_size_map"])
    print("action_scale:", policy_cfg_dict["action_scale"])
    print("clip_actions:", policy_cfg_dict["clip_actions"])
    print("clip_obs:", policy_cfg_dict["clip_obs"])
    print("obs_history_length:", policy_cfg_dict["obs_history_length"])
    print("joint_kp:", policy_cfg_dict["joint_kp"])
    print("joint_kd:", policy_cfg_dict["joint_kd"])
    print("torque_limit:", policy_cfg_dict["torque_limit"])
    export_inference_cfg_to_yaml_locomotion(policy_cfg_dict, path, load_run, checkpoint)


def export_inference_cfg_to_yaml_locomotion(config_dict, path, load_run, checkpoint):
    use_vae = config_dict["use_vae"]
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    readme_file_path = os.path.join(path, "policy.yaml")
    content = f'load_run: "{load_run}"\n'
    content += f'checkpoint: "{checkpoint}"\n'
    content += f"dt: {config_dict['dt']}\n"
    # joint_names 多行缩进
    content += "joint_names:\n  [\n"
    for name in config_dict["joint_names"]:
        content += f'    "{name}",\n'
    content += "  ]\n"

    # default_joint_pos 保留 4 位小数
    content += "default_joint_pos: ["
    content += ", ".join(f"{float(v):.4f}" for v in config_dict["default_joint_pos"])
    content += "]\n"

    # input_names 和 output_names
    content += "input_names: ["
    content += ", ".join(f'"{n}"' for n in config_dict["input_names"])
    content += "]\n"

    content += "output_names: ["
    content += ", ".join(f'"{n}"' for n in config_dict["output_names"])
    content += "]\n"

    # input_obs_names_map 多行缩进
    if use_vae:
        input_obs_names_map = {
            "actor_obs": config_dict["input_actor_obs_names"],
            "vae_obs": config_dict["input_vae_obs_names"],
        }
    else:
        input_obs_names_map = {
            "actor_obs": config_dict["input_actor_obs_names"],
        }
    content += "input_obs_names_map:\n  {\n"
    for key, obs_list in input_obs_names_map.items():
        content += f"    {key}: ["
        content += ", ".join(f'"{o}"' for o in obs_list)
        content += "],\n"
    content += "  }\n"

    # input_obs_scales_map 多行缩进，并区分标量／列表
    if use_vae:
        input_obs_scales_map = {
            "actor_obs": config_dict["input_actor_obs_scales"],
            "vae_obs": config_dict["input_vae_obs_scales"],
        }
    else:
        input_obs_scales_map = {
            "actor_obs": config_dict["input_actor_obs_scales"],
        }
    content += "input_obs_scales_map:\n  {\n"
    for key, scales in input_obs_scales_map.items():
        content += f"    {key}: {{ "
        parts = []
        for obs, val in scales.items():
            if isinstance(val, list):
                sval = "[" + ", ".join(f"{x}" for x in val) + "]"
            else:
                sval = f"{val}"
            parts.append(f"{obs}: {sval}")
        content += ", ".join(parts)
        content += " },\n"
    content += "  }\n"

    content += "input_obs_size_map:\n  {\n"
    for key, scales in config_dict["input_obs_size_map"].items():
        content += f"    {key}: {scales},\n"
    content += "  }\n"

    # 其余字段
    content += f"action_scale: {config_dict['action_scale']}\n"
    content += f"clip_actions: {config_dict['clip_actions']}\n"
    content += f"clip_obs: {config_dict['clip_obs']}\n"

    # obs_history_length
    content += "obs_history_length: { "
    content += ", ".join(f"{k}: {v}" for k, v in config_dict["obs_history_length"].items())
    content += " }\n"
    content += f"joint_kp: {config_dict['joint_kp']}\n"
    content += f"joint_kd: {config_dict['joint_kd']}\n"

    # 添加固定的速度scales
    content += "velocity_x_forward_scale: 1.0\n"
    content += "velocity_x_backward_scale: 0.7\n"
    content += "velocity_y_scale: 0.5\n"
    content += "velocity_yaw_scale: 1.0\n"

    # 添加固定的速度、加速度和加加速度上限
    content += "max_velocity: [1.0, 0.6, 1.5]\n"
    content += "max_acceleration: [1.5, 1.5, 6]\n"
    content += "max_jerk: [5, 5, 30]\n"

    # 添加固定的 threshold 配置
    content += "threshold:\n"
    content += "  limit_lower: -0.0\n"
    content += "  limit_upper: 0.0\n"
    content += "  damping: 5.0\n"

    # 添加固定的max_torques
    # 外挂减速齿轮
    content += f"max_torques: {config_dict['torque_limit']}\n"

    with open(readme_file_path, "w", encoding="utf-8") as f:
        f.write(content)


def export_inference_cfg_PIElocomotion(env, env_cfg, path, load_run, checkpoint, config_path):
    # export config summary
    save_config_path = os.path.join(path, "config_summary.py")

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    configSummary_path = os.path.join(config_path, "config_summary.py")
    with open(configSummary_path, "r", encoding="utf-8") as file:
        content = file.read()
    with open(save_config_path, "w", encoding="utf-8") as file:
        file.write(content)
    policy_cfg_dict = {}
    policy_cfg_dict["dt"] = env_cfg.decimation * env.unwrapped.physics_dt
    policy_cfg_dict["joint_names"] = env.unwrapped.scene.articulations["robot"].joint_names
    # 1. 直接拿到 numpy 数组
    default_joint_pos = env.unwrapped.scene.articulations["robot"]._data.default_joint_pos[0].cpu().numpy()

    # 2. （可选）保留 4 位小数，再转成 Python 列表
    policy_cfg_dict["default_joint_pos"] = np.round(default_joint_pos, 4).tolist()

    # 如果你要更精确地控制格式，比如总是输出 "-0.0500" 而不是 "-0.05"，也可以这样：
    policy_cfg_dict["default_joint_pos"] = [float(f"{x:.4f}") for x in default_joint_pos]

    # 修改了env_cfg
    env_cfg = env.unwrapped.cfg.config_summary.env
    obs_cfg = env.unwrapped.cfg.config_summary.observation

    input_actor_obs_scales = {}
    input_PIE_estimator_net_proprioceptive_obs_scales = {}
    input_PIE_estimator_net_depth_images_obs_scales = {}
    input_hidden_states_scales = {}
    input_obs_size_map = {}
    policy_cfg_dict["obs_history_length"] = {}

    policy_cfg_dict["output_names"] = ["actions", "new_hidden_states"]
    policy_cfg_dict["input_actor_obs_names"] = obs_cfg.policy_obs_dict["actor_obs"]["terms"]
    input_obs_size_map["actor_obs"] = env_cfg.num_actor_obs
    if "history_length" in obs_cfg.policy_obs_dict["actor_obs"]:
        actor_obs_history_length = obs_cfg.policy_obs_dict["actor_obs"]["history_length"]
    else:
        actor_obs_history_length = 1
    for actor_obs_sub_term in policy_cfg_dict["input_actor_obs_names"]:
        if "nad" in actor_obs_sub_term:
            input_actor_obs_scales[actor_obs_sub_term] = (
                obs_cfg.obs_term_dict["noise_and_delay_obs"][actor_obs_sub_term]["scale"]
                if "scale" in obs_cfg.obs_term_dict["noise_and_delay_obs"][actor_obs_sub_term]
                else 1.0
            )
        elif "gt" in actor_obs_sub_term:
            input_actor_obs_scales[actor_obs_sub_term] = (
                obs_cfg.obs_term_dict["ground_truth_obs"][actor_obs_sub_term]["scale"]
                if "scale" in obs_cfg.obs_term_dict["ground_truth_obs"][actor_obs_sub_term]
                else 1.0
            )
        else:
            input_actor_obs_scales[actor_obs_sub_term] = None
        if isinstance(input_actor_obs_scales[actor_obs_sub_term], tuple):
            input_actor_obs_scales[actor_obs_sub_term] = list(input_actor_obs_scales[actor_obs_sub_term])

    policy_cfg_dict["input_actor_obs_scales"] = input_actor_obs_scales
    policy_cfg_dict["obs_history_length"]["actor_obs"] = actor_obs_history_length if actor_obs_history_length > 0 else 1
    print("type", type(input_actor_obs_scales["base_commands_gt"]).__name__)

    # 读取是否使用PIE estimator net
    use_PIE_estimator_net = env_cfg.train_cfg_dict["use_PIE_estimator_net"]
    policy_cfg_dict["use_PIE_estimator_net"] = use_PIE_estimator_net
    if use_PIE_estimator_net:
        policy_cfg_dict["input_names"] = [
            "actor_obs",
            "PIE_estimator_net_proprioceptive_obs",
            "PIE_estimator_net_depth_images_obs",
            "hidden_states",
        ]
        policy_cfg_dict["input_PIE_estimator_net_proprioceptive_obs_names"] = obs_cfg.policy_obs_dict["PIE_estimator_net_proprioceptive_obs"]["terms"]
        policy_cfg_dict["input_PIE_estimator_net_depth_images_obs_names"] = obs_cfg.policy_obs_dict["PIE_estimator_net_depth_images_obs"]["terms"]
        policy_cfg_dict["input_hidden_states_names"] = {"hidden_states"}

        # PIE_estimator_net_proprioceptive_obs
        for PIE_estimator_net_proprioceptive_obs_sub_term in policy_cfg_dict["input_PIE_estimator_net_proprioceptive_obs_names"]:
            if "nad" in PIE_estimator_net_proprioceptive_obs_sub_term:
                input_PIE_estimator_net_proprioceptive_obs_scales[PIE_estimator_net_proprioceptive_obs_sub_term] = (
                    obs_cfg.obs_term_dict["noise_and_delay_obs"][PIE_estimator_net_proprioceptive_obs_sub_term]["scale"]
                    if "scale" in obs_cfg.obs_term_dict["noise_and_delay_obs"][PIE_estimator_net_proprioceptive_obs_sub_term]
                    else 1.0
                )
            elif "gt" in PIE_estimator_net_proprioceptive_obs_sub_term:
                input_PIE_estimator_net_proprioceptive_obs_scales[PIE_estimator_net_proprioceptive_obs_sub_term] = (
                    obs_cfg.obs_term_dict["ground_truth_obs"][PIE_estimator_net_proprioceptive_obs_sub_term]["scale"]
                    if "scale" in obs_cfg.obs_term_dict["ground_truth_obs"][PIE_estimator_net_proprioceptive_obs_sub_term]
                    else 1.0
                )
            else:
                input_PIE_estimator_net_proprioceptive_obs_scales[PIE_estimator_net_proprioceptive_obs_sub_term] = None
            if isinstance(
                input_PIE_estimator_net_proprioceptive_obs_scales[PIE_estimator_net_proprioceptive_obs_sub_term],
                tuple,
            ):
                input_PIE_estimator_net_proprioceptive_obs_scales[PIE_estimator_net_proprioceptive_obs_sub_term] = list(
                    input_PIE_estimator_net_proprioceptive_obs_scales[PIE_estimator_net_proprioceptive_obs_sub_term]
                )
        input_obs_size_map["PIE_estimator_net_proprioceptive_obs"] = env_cfg.num_PIE_estimator_net_proprioceptive_obs_step_obs
        policy_cfg_dict["input_PIE_estimator_net_proprioceptive_obs_scales"] = input_PIE_estimator_net_proprioceptive_obs_scales

        if "history_length" in obs_cfg.policy_obs_dict["PIE_estimator_net_proprioceptive_obs"]:
            PIE_estimator_net_proprioceptive_obs_history_length = obs_cfg.policy_obs_dict["PIE_estimator_net_proprioceptive_obs"]["history_length"]
        else:
            PIE_estimator_net_proprioceptive_obs_history_length = 10
        policy_cfg_dict["obs_history_length"]["PIE_estimator_net_proprioceptive_obs"] = (
            PIE_estimator_net_proprioceptive_obs_history_length if PIE_estimator_net_proprioceptive_obs_history_length > 0 else 1
        )

        # PIE_estimator_net_depth_images_obs
        for PIE_estimator_net_depth_images_obs_sub_term in policy_cfg_dict["input_PIE_estimator_net_depth_images_obs_names"]:
            if "nad" in PIE_estimator_net_depth_images_obs_sub_term:
                input_PIE_estimator_net_depth_images_obs_scales[PIE_estimator_net_depth_images_obs_sub_term] = (
                    obs_cfg.obs_term_dict["noise_and_delay_obs"][PIE_estimator_net_depth_images_obs_sub_term]["scale"]
                    if "scale" in obs_cfg.obs_term_dict["noise_and_delay_obs"][PIE_estimator_net_depth_images_obs_sub_term]
                    else 1.0
                )
            elif "gt" in PIE_estimator_net_depth_images_obs_sub_term:
                input_PIE_estimator_net_depth_images_obs_scales[PIE_estimator_net_depth_images_obs_sub_term] = (
                    obs_cfg.obs_term_dict["ground_truth_obs"][PIE_estimator_net_depth_images_obs_sub_term]["scale"]
                    if "scale" in obs_cfg.obs_term_dict["ground_truth_obs"][PIE_estimator_net_depth_images_obs_sub_term]
                    else 1.0
                )
            else:
                input_PIE_estimator_net_depth_images_obs_scales[PIE_estimator_net_depth_images_obs_sub_term] = None
            if isinstance(
                input_PIE_estimator_net_depth_images_obs_scales[PIE_estimator_net_depth_images_obs_sub_term],
                tuple,
            ):
                input_PIE_estimator_net_depth_images_obs_scales[PIE_estimator_net_depth_images_obs_sub_term] = list(
                    input_PIE_estimator_net_depth_images_obs_scales[PIE_estimator_net_depth_images_obs_sub_term]
                )
        input_obs_size_map["PIE_estimator_net_depth_images_obs"] = (
            env_cfg.PIE_estimator_net_depth_images_cnn_encoder_input_height * env_cfg.PIE_estimator_net_depth_images_cnn_encoder_input_width
        )
        policy_cfg_dict["input_PIE_estimator_net_depth_images_obs_scales"] = input_PIE_estimator_net_depth_images_obs_scales
        if "history_length" in obs_cfg.policy_obs_dict["PIE_estimator_net_depth_images_obs"]:
            PIE_estimator_net_depth_images_obs_history_length = obs_cfg.policy_obs_dict["PIE_estimator_net_depth_images_obs"]["history_length"]
        else:
            PIE_estimator_net_depth_images_obs_history_length = 2
        policy_cfg_dict["obs_history_length"]["PIE_estimator_net_depth_images_obs"] = (
            PIE_estimator_net_depth_images_obs_history_length if PIE_estimator_net_depth_images_obs_history_length > 0 else 1
        )

        # hidden_states
        input_hidden_states_scales["hidden_states"] = 1.0
        policy_cfg_dict["input_hidden_states_scales"] = input_hidden_states_scales
        input_obs_size_map["hidden_states"] = env_cfg.module_cfg_dict["PIE_estimator_net"]["gru_encoder_hidden_dim"]
        policy_cfg_dict["obs_history_length"]["hidden_states"] = 1
    else:
        policy_cfg_dict["input_names"] = ["actor_obs"]

    # 添加调试信息，查看可用的观察组名称
    print(
        "Available observation group names:",
        list(env.unwrapped.observation_manager._group_obs_term_names.keys()),
    )

    policy_cfg_dict["input_obs_size_map"] = input_obs_size_map
    policy_cfg_dict["action_scale"] = env.unwrapped.cfg.config_summary.action.scale
    policy_cfg_dict["clip_actions"] = env.unwrapped.cfg.config_summary.env.clip_actions
    policy_cfg_dict["clip_obs"] = env.unwrapped.cfg.config_summary.env.clip_obs

    robot = env.unwrapped.scene.articulations["robot"]
    joint_kp, joint_kd = _collect_joint_pd_gains_in_joint_order(robot)
    policy_cfg_dict["joint_kp"] = joint_kp
    policy_cfg_dict["joint_kd"] = joint_kd
    hip_torque_limit = env_cfg.hip_tor_limit
    thigh_torque_limit = env_cfg.thigh_tor_limit
    calf_torque_limit = env_cfg.calf_tor_limit
    policy_torque_limit = [hip_torque_limit, thigh_torque_limit, calf_torque_limit] * 4
    policy_cfg_dict["torque_limit"] = [float(f"{x:.4f}") for x in policy_torque_limit]
    print("joint_names:", policy_cfg_dict["joint_names"])
    print("default_joint_pos:", policy_cfg_dict["default_joint_pos"])
    print("input_names:", policy_cfg_dict["input_names"])
    print("output_names:", policy_cfg_dict["output_names"])
    print("input_actor_obs_names:", policy_cfg_dict["input_actor_obs_names"])
    if use_PIE_estimator_net:
        print(
            "input_PIE_estimator_net_proprioceptive_obs_names:",
            policy_cfg_dict["input_PIE_estimator_net_proprioceptive_obs_names"],
        )
        print(
            "input_PIE_estimator_net_proprioceptive_obs_scales:",
            policy_cfg_dict["input_PIE_estimator_net_proprioceptive_obs_scales"],
        )
        print(
            "input_PIE_estimator_net_depth_images_obs_names:",
            policy_cfg_dict["input_PIE_estimator_net_depth_images_obs_names"],
        )
        print(
            "input_PIE_estimator_net_depth_images_obs_scales:",
            policy_cfg_dict["input_PIE_estimator_net_depth_images_obs_scales"],
        )
        print(
            "input_hidden_states_names:",
            policy_cfg_dict["input_hidden_states_names"],
        )
        print(
            "input_hidden_states_scales:",
            policy_cfg_dict["input_hidden_states_scales"],
        )
    print("input_actor_obs_scales:", policy_cfg_dict["input_actor_obs_scales"])
    print("input_obs_size_map:", policy_cfg_dict["input_obs_size_map"])
    print("action_scale:", policy_cfg_dict["action_scale"])
    print("clip_actions:", policy_cfg_dict["clip_actions"])
    print("clip_obs:", policy_cfg_dict["clip_obs"])
    print("obs_history_length:", policy_cfg_dict["obs_history_length"])
    print("joint_kp:", policy_cfg_dict["joint_kp"])
    print("joint_kd:", policy_cfg_dict["joint_kd"])
    print("torque_limit:", policy_cfg_dict["torque_limit"])
    export_inference_cfg_to_yaml_PIElocomotion(policy_cfg_dict, path, load_run, checkpoint)


def export_inference_cfg_to_yaml_PIElocomotion(config_dict, path, load_run, checkpoint):
    use_PIE_estimator_net = config_dict["use_PIE_estimator_net"]
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    readme_file_path = os.path.join(path, "policy.yaml")
    content = f'load_run: "{load_run}"\n'
    content += f'checkpoint: "{checkpoint}"\n'
    content += f"dt: {config_dict['dt']}\n"
    # joint_names 多行缩进
    content += "joint_names:\n  [\n"
    for name in config_dict["joint_names"]:
        content += f'    "{name}",\n'
    content += "  ]\n"

    # default_joint_pos 保留 4 位小数
    content += "default_joint_pos: ["
    content += ", ".join(f"{float(v):.4f}" for v in config_dict["default_joint_pos"])
    content += "]\n"

    # input_names 和 output_names
    content += "input_names: ["
    content += ", ".join(f'"{n}"' for n in config_dict["input_names"])
    content += "]\n"

    content += "output_names: ["
    content += ", ".join(f'"{n}"' for n in config_dict["output_names"])
    content += "]\n"

    # input_obs_names_map 多行缩进
    if use_PIE_estimator_net:
        input_obs_names_map = {
            "actor_obs": config_dict["input_actor_obs_names"],
            "PIE_estimator_net_proprioceptive_obs": config_dict["input_PIE_estimator_net_proprioceptive_obs_names"],
            "PIE_estimator_net_depth_images_obs": config_dict["input_PIE_estimator_net_depth_images_obs_names"],
            "hidden_states": config_dict["input_hidden_states_names"],
        }
    else:
        input_obs_names_map = {
            "actor_obs": config_dict["input_actor_obs_names"],
        }
    content += "input_obs_names_map:\n  {\n"
    for key, obs_list in input_obs_names_map.items():
        content += f"    {key}: ["
        content += ", ".join(f'"{o}"' for o in obs_list)
        content += "],\n"
    content += "  }\n"

    # input_obs_scales_map 多行缩进，并区分标量／列表
    if use_PIE_estimator_net:
        input_obs_scales_map = {
            "actor_obs": config_dict["input_actor_obs_scales"],
            "PIE_estimator_net_proprioceptive_obs": config_dict["input_PIE_estimator_net_proprioceptive_obs_scales"],
            "PIE_estimator_net_depth_images_obs": config_dict["input_PIE_estimator_net_depth_images_obs_scales"],
            "hidden_states": config_dict["input_hidden_states_scales"],
        }
    else:
        input_obs_scales_map = {
            "actor_obs": config_dict["input_actor_obs_scales"],
        }
    content += "input_obs_scales_map:\n  {\n"
    for key, scales in input_obs_scales_map.items():
        content += f"    {key}: {{ "
        parts = []
        for obs, val in scales.items():
            if isinstance(val, list):
                sval = "[" + ", ".join(f"{x}" for x in val) + "]"
            else:
                sval = f"{val}"
            parts.append(f"{obs}: {sval}")
        content += ", ".join(parts)
        content += " },\n"
    content += "  }\n"

    content += "input_obs_size_map:\n  {\n"
    for key, scales in config_dict["input_obs_size_map"].items():
        content += f"    {key}: {scales},\n"
    content += "  }\n"

    # 其余字段
    content += f"action_scale: {config_dict['action_scale']}\n"
    content += f"clip_actions: {config_dict['clip_actions']}\n"
    content += f"clip_obs: {config_dict['clip_obs']}\n"

    # obs_history_length
    content += "obs_history_length: { "
    content += ", ".join(f"{k}: {v}" for k, v in config_dict["obs_history_length"].items())
    content += " }\n"
    content += f"joint_kp: {config_dict['joint_kp']}\n"
    content += f"joint_kd: {config_dict['joint_kd']}\n"

    # 添加固定的速度scales
    content += "velocity_x_forward_scale: 1.0\n"
    content += "velocity_x_backward_scale: 0.7\n"
    content += "velocity_y_scale: 0.5\n"
    content += "velocity_yaw_scale: 1.0\n"

    # 添加固定的速度、加速度和加加速度上限
    content += "max_velocity: [1.0, 0.6, 1.5]\n"
    content += "max_acceleration: [1.5, 1.5, 6]\n"
    content += "max_jerk: [5, 5, 30]\n"

    # 添加固定的 threshold 配置
    content += "threshold:\n"
    content += "  limit_lower: -0.0\n"
    content += "  limit_upper: 0.0\n"
    content += "  damping: 5.0\n"

    # 添加固定的max_torques
    content += f"max_torques: {config_dict['torque_limit']}\n"

    with open(readme_file_path, "w", encoding="utf-8") as f:
        f.write(content)


def export_inference_cfg_MARGlocomotion(env, env_cfg, path, load_run, checkpoint, config_path):
    # export config summary
    save_config_path = os.path.join(path, "config_summary.py")

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    configSummary_path = os.path.join(config_path, "config_summary.py")
    with open(configSummary_path, "r", encoding="utf-8") as file:
        content = file.read()
    with open(save_config_path, "w", encoding="utf-8") as file:
        file.write(content)
    policy_cfg_dict = {}
    policy_cfg_dict["dt"] = env_cfg.decimation * env.unwrapped.physics_dt
    policy_cfg_dict["joint_names"] = env.unwrapped.scene.articulations["robot"].joint_names
    # 1. 直接拿到 numpy 数组
    default_joint_pos = env.unwrapped.scene.articulations["robot"]._data.default_joint_pos[0].cpu().numpy()

    # 2. （可选）保留 4 位小数，再转成 Python 列表
    policy_cfg_dict["default_joint_pos"] = np.round(default_joint_pos, 4).tolist()

    # 如果你要更精确地控制格式，比如总是输出 "-0.0500" 而不是 "-0.05"，也可以这样：
    policy_cfg_dict["default_joint_pos"] = [float(f"{x:.4f}") for x in default_joint_pos]

    env_cfg = env.unwrapped.cfg.config_summary.env
    obs_cfg = env.unwrapped.cfg.config_summary.observation

    input_actor_obs_scales = {}
    input_estimator_net_obs_scales = {}
    input_heightmap_obs_scales = {}
    input_obs_size_map = {}
    policy_cfg_dict["obs_history_length"] = {}

    # 读取estimator_net_obs
    # 读取是否使用Estimator Net
    use_estimator_net = env_cfg.train_cfg_dict["use_estimator_net"]
    policy_cfg_dict["use_estimator_net"] = use_estimator_net
    if use_estimator_net:
        policy_cfg_dict["input_names"] = [
            "estimator_net_obs",
            "actor_obs",
            "gt_heightmap_obs",
        ]
        policy_cfg_dict["input_estimator_net_obs_names"] = obs_cfg.policy_obs_dict["estimator_net_obs"]["terms"]
        input_obs_size_map["estimator_net_obs"] = env_cfg.num_estimator_net_step_obs
        for estimator_net_obs_sub_term in policy_cfg_dict["input_estimator_net_obs_names"]:
            if "nad" in estimator_net_obs_sub_term:
                input_estimator_net_obs_scales[estimator_net_obs_sub_term] = (
                    obs_cfg.obs_term_dict["noise_and_delay_obs"][estimator_net_obs_sub_term]["scale"]
                    if "scale" in obs_cfg.obs_term_dict["noise_and_delay_obs"][estimator_net_obs_sub_term]
                    else 1.0
                )
            elif "gt" in estimator_net_obs_sub_term:
                input_estimator_net_obs_scales[estimator_net_obs_sub_term] = (
                    obs_cfg.obs_term_dict["ground_truth_obs"][estimator_net_obs_sub_term]["scale"]
                    if "scale" in obs_cfg.obs_term_dict["ground_truth_obs"][estimator_net_obs_sub_term]
                    else 1.0
                )
            else:
                input_estimator_net_obs_scales[estimator_net_obs_sub_term] = None
            if isinstance(input_estimator_net_obs_scales[estimator_net_obs_sub_term], tuple):
                input_estimator_net_obs_scales[estimator_net_obs_sub_term] = list(input_estimator_net_obs_scales[estimator_net_obs_sub_term])
        if "history_length" in obs_cfg.policy_obs_dict["estimator_net_obs"]:
            estimator_net_obs_history_length = obs_cfg.policy_obs_dict["estimator_net_obs"]["history_length"]
        else:
            estimator_net_obs_history_length = 5
        policy_cfg_dict["input_estimator_net_obs_scales"] = input_estimator_net_obs_scales
        policy_cfg_dict["obs_history_length"]["estimator_net_obs"] = estimator_net_obs_history_length if estimator_net_obs_history_length > 0 else 1
    else:
        policy_cfg_dict["input_names"] = ["actor_obs", "gt_heightmap_obs"]

    # 读取actor_obs
    policy_cfg_dict["input_actor_obs_names"] = obs_cfg.policy_obs_dict["actor_obs"]["terms"]
    input_obs_size_map["actor_obs"] = env_cfg.num_actor_obs
    for actor_obs_sub_term in policy_cfg_dict["input_actor_obs_names"]:
        if "nad" in actor_obs_sub_term:
            input_actor_obs_scales[actor_obs_sub_term] = (
                obs_cfg.obs_term_dict["noise_and_delay_obs"][actor_obs_sub_term]["scale"]
                if "scale" in obs_cfg.obs_term_dict["noise_and_delay_obs"][actor_obs_sub_term]
                else 1.0
            )
        elif "gt" in actor_obs_sub_term:
            input_actor_obs_scales[actor_obs_sub_term] = (
                obs_cfg.obs_term_dict["ground_truth_obs"][actor_obs_sub_term]["scale"]
                if "scale" in obs_cfg.obs_term_dict["ground_truth_obs"][actor_obs_sub_term]
                else 1.0
            )
        else:
            input_actor_obs_scales[actor_obs_sub_term] = None
        if isinstance(input_actor_obs_scales[actor_obs_sub_term], tuple):
            input_actor_obs_scales[actor_obs_sub_term] = list(input_actor_obs_scales[actor_obs_sub_term])
    if "history_length" in obs_cfg.policy_obs_dict["actor_obs"]:
        actor_obs_history_length = obs_cfg.policy_obs_dict["actor_obs"]["history_length"]
    else:
        actor_obs_history_length = 1
    policy_cfg_dict["input_actor_obs_scales"] = input_actor_obs_scales
    policy_cfg_dict["obs_history_length"]["actor_obs"] = actor_obs_history_length if actor_obs_history_length > 0 else 1

    # 读取heightmap_obs
    policy_cfg_dict["input_heightmap_obs_names"] = obs_cfg.policy_obs_dict["gt_heightmap_obs"]["terms"]
    input_obs_size_map["gt_heightmap_obs"] = env_cfg.num_heightmap_obs
    for heightmap_obs_sub_term in policy_cfg_dict["input_heightmap_obs_names"]:
        if "nad" in heightmap_obs_sub_term:
            input_heightmap_obs_scales[heightmap_obs_sub_term] = (
                obs_cfg.obs_term_dict["noise_and_delay_obs"][heightmap_obs_sub_term]["scale"]
                if "scale" in obs_cfg.obs_term_dict["noise_and_delay_obs"][heightmap_obs_sub_term]
                else 1.0
            )
        elif "gt" in heightmap_obs_sub_term:
            input_heightmap_obs_scales[heightmap_obs_sub_term] = (
                obs_cfg.obs_term_dict["ground_truth_obs"][heightmap_obs_sub_term]["scale"]
                if "scale" in obs_cfg.obs_term_dict["ground_truth_obs"][heightmap_obs_sub_term]
                else 1.0
            )
        else:
            input_heightmap_obs_scales[heightmap_obs_sub_term] = None
        if isinstance(input_heightmap_obs_scales[heightmap_obs_sub_term], tuple):
            input_heightmap_obs_scales[heightmap_obs_sub_term] = list(input_heightmap_obs_scales[heightmap_obs_sub_term])
    if "history_length" in obs_cfg.policy_obs_dict["gt_heightmap_obs"]:
        heightmap_obs_history_length = obs_cfg.policy_obs_dict["gt_heightmap_obs"]["history_length"]
    else:
        heightmap_obs_history_length = 1
    policy_cfg_dict["input_heightmap_obs_scales"] = input_heightmap_obs_scales
    policy_cfg_dict["obs_history_length"]["gt_heightmap_obs"] = heightmap_obs_history_length if heightmap_obs_history_length > 0 else 1

    # 读取actions
    policy_cfg_dict["output_names"] = ["actions"]

    # 添加调试信息，查看可用的观察组名称
    print(
        "Available observation group names:",
        list(env.unwrapped.observation_manager._group_obs_term_names.keys()),
    )

    policy_cfg_dict["input_obs_size_map"] = input_obs_size_map
    policy_cfg_dict["action_scale"] = env.unwrapped.cfg.config_summary.action.scale
    policy_cfg_dict["clip_actions"] = env.unwrapped.cfg.config_summary.env.clip_actions
    policy_cfg_dict["clip_obs"] = env.unwrapped.cfg.config_summary.env.clip_obs

    robot = env.unwrapped.scene.articulations["robot"]
    joint_kp, joint_kd = _collect_joint_pd_gains_in_joint_order(robot)
    policy_cfg_dict["joint_kp"] = joint_kp
    policy_cfg_dict["joint_kd"] = joint_kd
    hip_torque_limit = env_cfg.hip_tor_limit
    thigh_torque_limit = env_cfg.thigh_tor_limit
    calf_torque_limit = env_cfg.calf_tor_limit
    policy_torque_limit = [hip_torque_limit, thigh_torque_limit, calf_torque_limit] * 4
    policy_cfg_dict["torque_limit"] = [float(f"{x:.4f}") for x in policy_torque_limit]
    print("joint_names:", policy_cfg_dict["joint_names"])
    print("default_joint_pos:", policy_cfg_dict["default_joint_pos"])
    print("input_names:", policy_cfg_dict["input_names"])
    print("output_names:", policy_cfg_dict["output_names"])
    print("input_actor_obs_names:", policy_cfg_dict["input_actor_obs_names"])
    if use_estimator_net:
        print(
            "input_estimator_net_obs_names:",
            policy_cfg_dict["input_estimator_net_obs_names"],
        )
        print(
            "input_estimator_net_obs_scales:",
            policy_cfg_dict["input_estimator_net_obs_scales"],
        )
    print("input_heightmap_obs_names:", policy_cfg_dict["input_heightmap_obs_names"])
    print("input_heightmap_obs_scales:", policy_cfg_dict["input_heightmap_obs_scales"])
    print("input_actor_obs_scales:", policy_cfg_dict["input_actor_obs_scales"])
    print("input_obs_size_map:", policy_cfg_dict["input_obs_size_map"])
    print("action_scale:", policy_cfg_dict["action_scale"])
    print("clip_actions:", policy_cfg_dict["clip_actions"])
    print("clip_obs:", policy_cfg_dict["clip_obs"])
    print("obs_history_length:", policy_cfg_dict["obs_history_length"])
    print("joint_kp:", policy_cfg_dict["joint_kp"])
    print("joint_kd:", policy_cfg_dict["joint_kd"])
    print("torque_limit:", policy_cfg_dict["torque_limit"])
    export_inference_cfg_to_yaml_MARGlocomotion(policy_cfg_dict, path, load_run, checkpoint)


def export_inference_cfg_to_yaml_MARGlocomotion(config_dict, path, load_run, checkpoint):
    use_estimator_net = config_dict["use_estimator_net"]
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    readme_file_path = os.path.join(path, "policy.yaml")
    content = f'load_run: "{load_run}"\n'
    content += f'checkpoint: "{checkpoint}"\n'
    content += f"dt: {config_dict['dt']}\n"
    # joint_names 多行缩进
    content += "joint_names:\n  [\n"
    for name in config_dict["joint_names"]:
        content += f'    "{name}",\n'
    content += "  ]\n"

    # default_joint_pos 保留 4 位小数
    content += "default_joint_pos: ["
    content += ", ".join(f"{float(v):.4f}" for v in config_dict["default_joint_pos"])
    content += "]\n"

    # input_names 和 output_names
    content += "input_names: ["
    content += ", ".join(f'"{n}"' for n in config_dict["input_names"])
    content += "]\n"

    content += "output_names: ["
    content += ", ".join(f'"{n}"' for n in config_dict["output_names"])
    content += "]\n"

    # input_obs_names_map 多行缩进
    if use_estimator_net:
        input_obs_names_map = {
            "estimator_net_obs": config_dict["input_estimator_net_obs_names"],
            "actor_obs": config_dict["input_actor_obs_names"],
            "gt_heightmap_obs": config_dict["input_heightmap_obs_names"],
        }
    else:
        input_obs_names_map = {
            "actor_obs": config_dict["input_actor_obs_names"],
            "gt_heightmap_obs": config_dict["input_heightmap_obs_names"],
        }
    content += "input_obs_names_map:\n  {\n"
    for key, obs_list in input_obs_names_map.items():
        content += f"    {key}: ["
        content += ", ".join(f'"{o}"' for o in obs_list)
        content += "],\n"
    content += "  }\n"

    # input_obs_scales_map 多行缩进，并区分标量／列表
    if use_estimator_net:
        input_obs_scales_map = {
            "estimator_net_obs": config_dict["input_estimator_net_obs_scales"],
            "actor_obs": config_dict["input_actor_obs_scales"],
            "gt_heightmap_obs": config_dict["input_heightmap_obs_scales"],
        }
    else:
        input_obs_scales_map = {
            "actor_obs": config_dict["input_actor_obs_scales"],
            "gt_heightmap_obs": config_dict["input_heightmap_obs_scales"],
        }
    content += "input_obs_scales_map:\n  {\n"
    for key, scales in input_obs_scales_map.items():
        content += f"    {key}: {{ "
        parts = []
        for obs, val in scales.items():
            if isinstance(val, list):
                sval = "[" + ", ".join(f"{x}" for x in val) + "]"
            else:
                sval = f"{val}"
            parts.append(f"{obs}: {sval}")
        content += ", ".join(parts)
        content += " },\n"
    content += "  }\n"

    content += "input_obs_size_map:\n  {\n"
    for key, scales in config_dict["input_obs_size_map"].items():
        content += f"    {key}: {scales},\n"
    content += "  }\n"

    # 其余字段
    content += f"action_scale: {config_dict['action_scale']}\n"
    content += f"clip_actions: {config_dict['clip_actions']}\n"
    content += f"clip_obs: {config_dict['clip_obs']}\n"

    # obs_history_length
    content += "obs_history_length: { "
    content += ", ".join(f"{k}: {v}" for k, v in config_dict["obs_history_length"].items())
    content += " }\n"
    content += f"joint_kp: {config_dict['joint_kp']}\n"
    content += f"joint_kd: {config_dict['joint_kd']}\n"

    # 添加固定的速度scales
    content += "velocity_x_forward_scale: 1.0\n"
    content += "velocity_x_backward_scale: 0.7\n"
    content += "velocity_y_scale: 0.5\n"
    content += "velocity_yaw_scale: 1.0\n"

    # 添加固定的速度、加速度和加加速度上限
    content += "max_velocity: [1.0, 0.6, 1.5]\n"
    content += "max_acceleration: [1.5, 1.5, 6]\n"
    content += "max_jerk: [5, 5, 30]\n"

    # 添加固定的 threshold 配置
    content += "threshold:\n"
    content += "  limit_lower: -0.0\n"
    content += "  limit_upper: 0.0\n"
    content += "  damping: 5.0\n"

    # 添加固定的max_torques
    content += f"max_torques: {config_dict['torque_limit']}\n"

    with open(readme_file_path, "w", encoding="utf-8") as f:
        f.write(content)

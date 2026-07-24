from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch.nn as nn
from torch import Tensor
import torch


class GRUEncoder(nn.Module):
    """单层 GRU 编码器。

    输入张量的形状应为 ``(seq_len, batch_size, input_dim)``，输出形状为
    一个以分支名称为键、张量切片为值的字典，每个张量的形状为
    ``(seq_len, batch_size, gru_encoder_out_dict[key])``。

    分支包含：
    - base_lin_vel
    - foot_clearance
    - elevation_map_est
    - latent
    """

    def __init__(
        self,
        gru_encoder_in_dim: int,
        gru_encoder_hidden_dim: int,
        gru_encoder_out_dict: Mapping[str, int],
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if not isinstance(gru_encoder_out_dict, Mapping) or not gru_encoder_out_dict:
            raise TypeError("hidden_dims 必须是非空的映射类型。")

        ordered_hidden_dims = OrderedDict()
        for name, dim in gru_encoder_out_dict.items():
            if not isinstance(name, str):
                raise TypeError("gru_encoder_out_dict 的键必须是字符串。")
            dim = int(dim)
            if dim <= 0:
                raise ValueError(f"隐藏维度 '{name}' 必须为正整数，收到 {dim}。")
            ordered_hidden_dims[name] = dim

        self.gru_encoder_in_dim = int(gru_encoder_in_dim)
        self.output_dims = ordered_hidden_dims
        self.gru_encoder_hidden_dim = gru_encoder_hidden_dim
        # 明确要求的分支键，需与 config_summary.py 中保持一致
        required_keys = ("base_lin_vel", "foot_clearance", "elevation_map_est", "latent")
        for key in required_keys:
            if key not in self.output_dims:
                raise KeyError(f"gru_encoder_out_dict 必须包含 '{key}' 键。")

        self.num_layers = int(num_layers)
        self.gru = nn.GRU(
            input_size=self.gru_encoder_in_dim,
            hidden_size=self.gru_encoder_hidden_dim,
            batch_first=False,
            num_layers=self.num_layers,
        )
        self.output_mlp = nn.ModuleDict()
        for name, dim in self.output_dims.items():
            self.output_mlp[name] = nn.Sequential(
                nn.Linear(self.gru_encoder_hidden_dim, dim),
                nn.ELU(),
            )
        latent_dim = self.output_dims["latent"]
        self.latent_mean = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        self.latent_logvar = nn.Linear(in_features=latent_dim, out_features=latent_dim)

    def forward(
        self,
        gru_input: Tensor,
        deterministic: bool = False,
        gru_out_hidden_states: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> dict[str, Tensor]:
        """前向传播。

        Args:
            gru_input: (seq_len, batch_size, input_dim)
            deterministic: 是否在重参数化采样时使用均值（True）或采样（False）
            gru_out_hidden_states: 传入的上一时刻 hidden state（num_layers, batch_size, hidden_dim）或 None
            masks: 可选的时间掩码，形状为 (seq_len, batch_size)，True/1 表示该时间步有效。
                - 如果为 None，则按固定长度序列处理；
                - 如果不为 None，则使用 pack_padded_sequence 实现变长序列 GRU。
        """
        seq_len, batch_size, _ = gru_input.shape

        # gru_input: (seq_len, batch, input_dim)
        outputs: dict[str, Tensor] = {}

        device = gru_input.device

        # 判断是否需要使用变长序列的 pack/pad 路径：
        # - masks 为 None 或 全为 True: 视为定长序列，直接走普通 GRU
        # - 否则: 使用 pack_padded_sequence / pad_packed_sequence
        use_packed = False
        if masks is not None:
            # masks: (seq_len, batch) -> lengths: (batch,)
            lengths = masks.long().sum(dim=0)
            # 如果所有样本的有效长度都等于 seq_len，则不需要 pack/pad
            if not torch.all(lengths == seq_len):
                use_packed = True

        if not use_packed:
            # 普通 GRU 路径：不使用 pack/pad，直接按 (seq_len, batch, dim) 计算
            if gru_out_hidden_states is None:
                h0 = torch.zeros(
                    self.num_layers,
                    batch_size,
                    self.gru_encoder_hidden_dim,
                    device=device,
                    dtype=gru_input.dtype,
                )
            else:
                h0 = gru_out_hidden_states

            h_all, h_last = self.gru(gru_input, h0.contiguous())
        else:
            # 变长序列路径：仅当 masks 真实包含 0/1（存在不同长度）时才启用
            lengths = masks.long().sum(dim=0)  # 每个 batch 样本的有效时间步数
            nonzero_masks = lengths > 0
            if nonzero_masks.sum().item() == 0:
                raise ValueError(f"masks 中所有样本长度都为 0，收到 {lengths}")
            keep_indices = nonzero_masks.nonzero(as_tuple=False).squeeze(1)
            inputs_nz = gru_input[:, keep_indices, :]
            lengths_nz = lengths[keep_indices]
            h0_nz = gru_out_hidden_states[:, keep_indices, :]
            # 降序排序
            lengths_sorted, indices_sorted = lengths_nz.sort(
                dim=0, descending=True
            )
            inputs_sorted = inputs_nz[:, indices_sorted, :]
            h0_sorted = h0_nz[:, indices_sorted, :]
            # pack_padded_sequence 不允许长度为 0，这里做下限截断，后面再将这些样本的输出清零
            lengths_clamped = torch.clamp(lengths_sorted, min=1)
            # 注意：pack_padded_sequence 要求 lengths 在 CPU 上且为 long
            packed_input = nn.utils.rnn.pack_padded_sequence(
                inputs_sorted,
                lengths_clamped.cpu(),
                batch_first=False,
                enforce_sorted=True,
            )
            packed_out, h_last_sorted = self.gru(
                packed_input, h0_sorted.contiguous()
            )
            # 恢复到与输入相同的时间/批次维度：(seq_len, batch, hidden_dim)
            out_padded, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=False,
                total_length=seq_len,
            )
            # 恢复原有排序
            _, indices_unsorted = indices_sorted.sort()
            out_unsorted = out_padded[:, indices_unsorted, :]
            h_last_unsorted = h_last_sorted[:, indices_unsorted, :]

            # 将剔除的 length==0 segment 插回到原始 batch 大小的位置
            h_all = torch.zeros(
                seq_len,
                batch_size,
                self.gru_encoder_hidden_dim,
                device=device,
                dtype=out_unsorted.dtype,
            )
            h_last = torch.zeros(
                self.num_layers,
                batch_size,
                self.gru_encoder_hidden_dim,
                device=device,
                dtype=h_last_unsorted.dtype,
            )

            h_all[:, keep_indices, :] = out_unsorted
            h_last[:, keep_indices, :] = h_last_unsorted

        # 每个分支输出形状: (seq_len, batch, dim)
        for name, dim in self.output_dims.items():
            outputs[name] = self.output_mlp[name](h_all)
        # h_last: (num_layers, batch, dim)，按最后一维拼回整体 hidden_state
        new_gru_out_hidden_states = h_last

        # latent 也按时间步计算: (seq_len, batch, latent_dim)
        latent_input = outputs["latent"]
        mean_latent = self.latent_mean(latent_input)
        logvar_latent = self.latent_logvar(latent_input)
        logvar_latent = torch.clamp(logvar_latent, min=-10, max=10)
        latent = self.reparameterise(mean_latent, logvar_latent, deterministic=deterministic)
        outputs["latent"] = latent
        outputs["mean_latent"] = mean_latent
        outputs["logvar_latent"] = logvar_latent

        # 按照 output_dims 的内容和顺序，将各分支张量在最后一维上拼接到 hidden_all
        hidden_list = [outputs[name] for name in self.output_dims.keys()]
        outputs["hidden_all"] = torch.cat(hidden_list, dim=-1)  # (seq_len, batch, sum(gru_encoder_out_dict.values()))

        return outputs, new_gru_out_hidden_states

    def reparameterise(self, mean, logvar, deterministic=False):
        if deterministic:
            return mean
        else:
            std = torch.exp(logvar * 0.5)
            distribution = mean + std * torch.randn_like(std)
            return distribution

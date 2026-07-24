from __future__ import annotations

from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor


class TransformerEncoderLayer(nn.Module):
    """单层 Transformer Encoder，由自注意力与前馈网络两部分组成。"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim 必须为正整数。")
        if num_heads <= 0:
            raise ValueError("num_heads 必须为正整数。")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除。")
        if feedforward_dim <= 0:
            raise ValueError("feedforward_dim 必须为正整数。")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout 必须位于 [0, 1) 区间。")

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.feedforward_dim = int(feedforward_dim)
        self.dropout = float(dropout)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )

        self.linear1 = nn.Linear(self.embed_dim, self.feedforward_dim)
        self.linear2 = nn.Linear(self.feedforward_dim, self.embed_dim)

        self.dropout_attn = nn.Dropout(self.dropout)
        self.dropout_ffn = nn.Dropout(self.dropout)

        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError("activation 仅支持 'relu' 与 'gelu'。")

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_attn_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """前向传播。

        Args:
            x: 输入序列，形状为 (batch_size, seq_len, embed_dim)。
            attn_mask: 可选的注意力 mask，形状为 (seq_len, seq_len) 或 (batch_size * num_heads, seq_len, seq_len)。
            key_padding_mask: 可选的 padding mask，形状为 (batch_size, seq_len)。
            need_attn_weights: 是否返回平均后的注意力权重。

        Returns:
            一个元组 (输出张量, 注意力权重或 None)。
        """
        if x.dim() != 3:
            raise ValueError(f"输入 x 的维度应为 3，当前形状为 {tuple(x.shape)}。")
        if x.size(-1) != self.embed_dim:
            raise ValueError(f"输入特征维度应为 {self.embed_dim}，收到 {x.size(-1)}。")

        residual = x
        attn_output, attn_weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_attn_weights,
        )
        x = self.norm1(residual + self.dropout_attn(attn_output))

        residual = x
        x = self.linear2(self.dropout_ffn(self.activation(self.linear1(x))))
        x = self.norm2(residual + self.dropout_ffn(x))

        return x, attn_weights if need_attn_weights else None


class TransformerEncoder(nn.Module):
    """基于自注意力的 Transformer Encoder 堆栈，仅提供输入输出接口。"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feedforward_dim: int,
        num_layers: int,
        dropout: float = 0.1,
        activation: str = "relu",
        return_attn_weights: bool = False,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers 必须为正整数。")

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.return_attn_weights = bool(return_attn_weights)

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, ...]]]:
        """执行多层编码。

        Args:
            x: 输入序列，形状为 (batch_size, seq_len, embed_dim)。
            attn_mask: 可选的注意力 mask。
            key_padding_mask: 可选的 padding mask。

        Returns:
            (输出张量, 各层注意力权重或 None)。
        """
        attn_weights_collection = [] if self.return_attn_weights else None

        for layer in self.layers:
            x, attn_weights = layer(
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_attn_weights=self.return_attn_weights,
            )
            if attn_weights_collection is not None:
                attn_weights_collection.append(attn_weights)

        if attn_weights_collection is not None:
            return x, tuple(attn_weights_collection)
        return x, None

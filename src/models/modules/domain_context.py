"""
Domain Context Encoder（域上下文编码器）。

设计目标
────────
在 ADS-lite 中，我们希望把“当前场景信息”（例如 tab/hour/day）编码成一个紧凑向量 d_ctx，
后续供 PSRG（历史重映射）与 PCRG（多 query 生成）共同使用。

为什么单独做一个模块？
──────────────────
1) 便于复用：PSRG 与 PCRG 都依赖同一份域上下文表示。
2) 便于可解释：可以清楚看到 d_ctx 来源于哪些上下文字段。
3) 便于 ablation：通过配置开关控制是否包含 hour/day/user_ctx。
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def _build_mlp(input_dim: int, hidden_units: List[int], output_dim: int, dropout: float) -> nn.Sequential:
    """构建通用 MLP：Linear + ReLU (+ Dropout) × N，再接输出层。"""
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_units:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class DomainContextEncoder(nn.Module):
    """
    将拼接后的上下文特征编码为 d_ctx。

    输入：
      ctx_concat: [B, D_in]

    输出：
      d_ctx: [B, D_out]

    说明：
      - 这里不直接做 embedding lookup，而是接收上游已经准备好的上下文拼接向量。
      - 这样可以复用主模型已有 embedding（避免重复参数），并保持模块职责单一。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_units: List[int] | None = None,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()
        hidden_units = hidden_units or []
        self.encoder = _build_mlp(
            input_dim=input_dim,
            hidden_units=hidden_units,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.layernorm = nn.LayerNorm(output_dim) if use_layernorm else nn.Identity()

    def forward(self, ctx_concat: torch.Tensor) -> torch.Tensor:
        """前向编码。"""
        d_ctx = self.encoder(ctx_concat)
        d_ctx = self.layernorm(d_ctx)
        return d_ctx

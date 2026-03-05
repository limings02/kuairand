"""
PSRG-lite（Personalized Sequence Representation Generation）模块。

为什么是 "lite" 版本？
────────────────────
论文式完整 PSRG 往往会做更重的“实例级参数生成”（例如为每个样本动态生成完整 MLP 权重），
在工程上会带来较大计算与稳定性成本。

这里采用可训练、低成本的近似：
1) FiLM: h' = LN(h + gamma(h,d)*h + beta(h,d))
2) Gated Residual: h' = LN(h + sigmoid(gate(h,d)) * delta(h,d))

它们保留了“历史行为在不同域上下文下语义可变”的核心思想，同时更适合 DIN baseline 增量实验。
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


def _build_mlp(input_dim: int, hidden_units: List[int], output_dim: int, dropout: float) -> nn.Sequential:
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


class PSRGLite(nn.Module):
    """ADS-lite 的历史序列重映射模块。"""

    def __init__(
        self,
        item_dim: int,
        d_ctx_dim: int,
        mode: str = "gated_residual",
        hidden_units: List[int] | None = None,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()
        hidden_units = hidden_units or [64]
        self.mode = mode
        in_dim = item_dim + d_ctx_dim

        if mode == "film":
            self.gamma_mlp = _build_mlp(in_dim, hidden_units, item_dim, dropout)
            self.beta_mlp = _build_mlp(in_dim, hidden_units, item_dim, dropout)
        elif mode == "gated_residual":
            self.gate_mlp = _build_mlp(in_dim, hidden_units, item_dim, dropout)
            self.delta_mlp = _build_mlp(in_dim, hidden_units, item_dim, dropout)
        else:
            raise ValueError(f"不支持的 psrg_mode: {mode}，仅支持 film / gated_residual")

        self.layernorm = nn.LayerNorm(item_dim) if use_layernorm else nn.Identity()

    def forward(
        self,
        hist_repr: torch.Tensor,
        d_ctx_seq: torch.Tensor,
        hist_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, int]]:
        """
        Args:
            hist_repr: [B, L, D]
            d_ctx_seq: [B, L, D_ctx]
            hist_mask: [B, L]，1=有效，0=padding

        Returns:
            hist_repr_psrg: [B, L, D]
            aux_info: 防御性统计信息
        """
        if hist_repr.ndim != 3 or d_ctx_seq.ndim != 3:
            raise ValueError("PSRG 输入必须为 3D 张量：hist_repr/d_ctx_seq 形状应分别为 [B,L,D] / [B,L,D_ctx]")
        if hist_repr.shape[:2] != d_ctx_seq.shape[:2]:
            raise ValueError(
                f"PSRG 输入 shape 不匹配：hist_repr={list(hist_repr.shape)} vs d_ctx_seq={list(d_ctx_seq.shape)}"
            )

        B, L, D = hist_repr.shape
        merged = torch.cat([hist_repr, d_ctx_seq], dim=-1).reshape(B * L, -1)

        if self.mode == "film":
            gamma = self.gamma_mlp(merged).reshape(B, L, D)
            beta = self.beta_mlp(merged).reshape(B, L, D)
            hist_out = hist_repr + gamma * hist_repr + beta
        else:
            gate = torch.sigmoid(self.gate_mlp(merged)).reshape(B, L, D)
            delta = self.delta_mlp(merged).reshape(B, L, D)
            hist_out = hist_repr + gate * delta

        hist_out = self.layernorm(hist_out)

        # padding 位保持原始值（通常是接近零向量），避免无意义扰动传播。
        mask_f = hist_mask.to(hist_out.dtype).unsqueeze(-1)  # [B, L, 1]
        hist_out = hist_out * mask_f + hist_repr * (1.0 - mask_f)

        all_pad_count = int((hist_mask.sum(dim=-1) == 0).sum().item())
        return hist_out, {"all_pad_count": all_pad_count}

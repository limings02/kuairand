"""
DIN 风格的 DNN Target-Attention 模块。

该模块被两处复用：
1) 原始 DIN 的单 query 兴趣抽取
2) TransformerFusion 的候选相关性增强步骤

核心思想：
- 只做点积会偏线性，难以显式建模 query/key 的交互模式
- DIN 风格 MLP 使用 [q, k, q-k, q*k] 作为输入，可以同时编码相似、差异与逐维交互
- 在推荐场景中，这类打分方式通常比纯 dot-product 更稳健
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对 attention score 做数值稳定的 mask softmax。

    Args:
        scores: [..., N]
        mask:   [..., N]，1=有效，0=padding
        dim:    softmax 维度

    Returns:
        probs:   [..., N]
        all_pad: [...]，该样本在目标维是否全为 padding
    """
    mask_bool = mask > 0
    scores = scores.masked_fill(~mask_bool, -1e4)
    probs = torch.softmax(scores, dim=dim)
    probs = probs * mask_bool.to(probs.dtype)

    denom = probs.sum(dim=dim, keepdim=True)
    probs = torch.where(denom > 0, probs / denom.clamp_min(1e-12), torch.zeros_like(probs))
    all_pad = mask_bool.sum(dim=dim) == 0
    return probs, all_pad


def masked_mean_pool(tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    对 token 序列做 mask-aware mean pooling。

    这里主要用于关闭 target attention 的 ablation：
    - sequence 模式下对历史序列做有效位均值
    - interest 模式下对多兴趣 token 做简单均值
    """
    if mask is None:
        return tokens.mean(dim=1)

    mask_f = mask.to(tokens.dtype).unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (tokens * mask_f).sum(dim=1) / denom


def _build_activation(activation: str) -> nn.Module:
    act = activation.lower()
    if act == "relu":
        return nn.ReLU()
    if act == "gelu":
        return nn.GELU()
    if act == "prelu":
        return nn.PReLU()
    raise ValueError(f"不支持的 activation={activation}，仅支持 relu / gelu / prelu")


class TargetAttentionDNN(nn.Module):
    """
    DIN-style target attention。

    打分公式：
        score(q, t_i) = MLP([q, t_i, q - t_i, q * t_i])

    之所以保留这一层而不是只用 Self-Attention，是因为：
    - Self-Attention 更关注 token 内部关系
    - 推荐打分仍然需要“候选 item 当前最相关哪几个兴趣”这一显式对齐过程
    - 因此这里用 candidate 作为 query，再做一次面向目标的注意力收缩
    """

    def __init__(
        self,
        item_dim: int,
        hidden_units: List[int] | None = None,
        dropout: float = 0.0,
        activation: str = "prelu",
    ):
        super().__init__()
        hidden_units = hidden_units or [64, 32]

        input_dim = 4 * item_dim
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(_build_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            query: [B, D]
            keys:  [B, N, D]
            mask:  [B, N]，若为 None 则视为全有效
        """
        batch_size, num_tokens, _ = keys.shape
        if mask is None:
            mask = torch.ones(batch_size, num_tokens, device=keys.device, dtype=keys.dtype)

        q = query.unsqueeze(1).expand(-1, num_tokens, -1)
        att_input = torch.cat([q, keys, q - keys, q * keys], dim=-1)
        att_scores = self.mlp(att_input).squeeze(-1)

        att_weights, all_pad = masked_softmax(att_scores, mask, dim=-1)
        user_interest = torch.bmm(att_weights.unsqueeze(1), keys).squeeze(1)

        if not return_debug:
            return user_interest

        entropy = -(att_weights * att_weights.clamp_min(1e-12).log()).sum(dim=-1)
        debug: Dict[str, Any] = {
            "all_pad_count": int(all_pad.sum().item()),
            "attn_entropy_mean": float(entropy.mean().item()),
            "attn_weights": att_weights.detach(),
        }
        return user_interest, debug

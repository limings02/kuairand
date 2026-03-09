"""
PCRG-lite（Personalized Candidate Representation Generation）模块。

核心思想
────────
由候选 item + 域上下文 d_ctx 生成 G 个 query，分别关注历史序列的不同兴趣子空间：
  Q = MLP([cand_item_repr, d_ctx]) -> [B, G, D_q]

随后做 target attention：
  score(q_g, h_l) -> alpha_{g,l}
  z_g = Σ_l alpha_{g,l} * h_l

最后按配置把 [B, G, D] 聚合回 [B, D]。
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


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对最后一维做 mask softmax。

    Args:
        scores: [..., L]
        mask:   [..., L]，1=有效，0=padding

    Returns:
        probs:      [..., L]
        all_pad:    [...]，该位置是否全是 padding
    """
    mask_bool = mask > 0
    scores = scores.masked_fill(~mask_bool, -1e4)
    probs = torch.softmax(scores, dim=dim)
    probs = probs * mask_bool.to(probs.dtype)

    denom = probs.sum(dim=dim, keepdim=True)
    probs = torch.where(denom > 0, probs / denom.clamp_min(1e-12), torch.zeros_like(probs))
    all_pad = (mask_bool.sum(dim=dim) == 0)
    return probs, all_pad


class PCRGLite(nn.Module):
    """ADS-lite 的候选多 query 生成与聚合模块。"""

    def __init__(
        self,
        item_dim: int,
        d_ctx_dim: int,
        num_queries: int = 4,
        query_dim: int = 48,
        score_type: str = "din_mlp",
        hidden_units: List[int] | None = None,
        aggregation: str = "mean_pool",
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_units = hidden_units or [64, 32]

        self.item_dim = item_dim
        self.query_dim = query_dim
        self.num_queries = num_queries
        self.score_type = score_type
        self.aggregation = aggregation

        # 由候选 + 域上下文生成 G 个 query
        self.query_gen = _build_mlp(
            input_dim=item_dim + d_ctx_dim,
            hidden_units=hidden_units,
            output_dim=num_queries * query_dim,
            dropout=dropout,
        )

        # 若 query_dim 与 item_dim 不同，自动投影到统一空间后再与历史做 attention
        self.query_to_item = nn.Identity()
        if query_dim != item_dim:
            self.query_to_item = nn.Linear(query_dim, item_dim)

        # 打分函数
        if score_type == "din_mlp":
            self.score_mlp = _build_mlp(
                input_dim=4 * item_dim,
                hidden_units=hidden_units,
                output_dim=1,
                dropout=dropout,
            )
        elif score_type == "dot":
            self.score_mlp = None
        else:
            raise ValueError(f"不支持的 score_type={score_type}，仅支持 dot / din_mlp")

        # G 维聚合器
        if aggregation == "attn_pool_over_G":
            self.g_pool_proj = nn.Linear(item_dim, item_dim)
        elif aggregation == "concat_then_proj":
            self.concat_proj = nn.Linear(num_queries * item_dim, item_dim)
        elif aggregation == "mean_pool":
            pass
        else:
            raise ValueError(
                f"不支持的 aggregation={aggregation}，仅支持 mean_pool / attn_pool_over_G / concat_then_proj"
            )

    def _score_queries_with_hist(self, q_item: torch.Tensor, hist_repr: torch.Tensor) -> torch.Tensor:
        """计算 [B,G,L] 注意力打分。"""
        if self.score_type == "dot":
            return torch.einsum("bgd,bld->bgl", q_item, hist_repr)

        B, G, D = q_item.shape
        L = hist_repr.shape[1]
        q_expand = q_item.unsqueeze(2).expand(-1, -1, L, -1)        # [B,G,L,D]
        k_expand = hist_repr.unsqueeze(1).expand(-1, G, -1, -1)      # [B,G,L,D]
        pair = torch.cat([q_expand, k_expand, q_expand - k_expand, q_expand * k_expand], dim=-1)
        return self.score_mlp(pair).squeeze(-1)                       # [B,G,L]

    def _aggregate(self, z: torch.Tensor, cand_item_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """将 [B,G,D] 聚合为 [B,D]。"""
        if self.aggregation == "mean_pool":
            return z.mean(dim=1), None

        if self.aggregation == "attn_pool_over_G":
            q = self.g_pool_proj(cand_item_repr)                      # [B,D]
            score_g = torch.einsum("bgd,bd->bg", z, q)             # [B,G]
            alpha_g = torch.softmax(score_g, dim=-1)                 # [B,G]
            pooled = torch.einsum("bg,bgd->bd", alpha_g, z)         # [B,D]
            return pooled, alpha_g

        pooled = self.concat_proj(z.reshape(z.shape[0], -1))         # [B,D]
        return pooled, None

    def forward(
        self,
        cand_item_repr: torch.Tensor,
        d_ctx: torch.Tensor,
        hist_repr: torch.Tensor,
        hist_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor | int]]:
        """
        Args:
            cand_item_repr: [B, D]
            d_ctx:          [B, D_ctx]
            hist_repr:      [B, L, D]
            hist_mask:      [B, L]

        Returns:
            user_interest_ads: [B, D]
            aux: 调试统计
        """
        B, L, D = hist_repr.shape

        # 1) 生成多 query
        q_input = torch.cat([cand_item_repr, d_ctx], dim=-1)         # [B, D + D_ctx]
        q = self.query_gen(q_input).reshape(B, self.num_queries, self.query_dim)  # [B,G,D_q]
        q_item = self.query_to_item(q)                                # [B,G,D]

        # 2) 每个 query 对历史做 attention
        scores = self._score_queries_with_hist(q_item, hist_repr)     # [B,G,L]
        mask_3d = hist_mask.unsqueeze(1).expand(-1, self.num_queries, -1)  # [B,G,L]
        attn, all_pad = _masked_softmax(scores, mask_3d, dim=-1)

        # 3) 得到 G 个兴趣向量
        z = torch.einsum("bgl,bld->bgd", attn, hist_repr)            # [B,G,D]

        # 4) 聚合 G 个兴趣
        pooled, alpha_g = self._aggregate(z, cand_item_repr)

        # 5) 可选调试指标
        entropy = -(attn * (attn.clamp_min(1e-12).log())).sum(dim=-1)  # [B,G]
        interest_var = z.var(dim=1, unbiased=False).mean()              # scalar

        aux: Dict[str, torch.Tensor | int] = {
            "attn_entropy_mean": entropy.mean().detach(),
            "query_interest_var": interest_var.detach(),
            "all_pad_count": int(all_pad.sum().item()),
            # 将多兴趣 token 显式返回给上层，便于后续 TransformerFusion 继续建模。
            "interest_tokens": z,
            "interest_mask": torch.ones(B, self.num_queries, device=hist_repr.device, dtype=hist_mask.dtype),
        }
        if alpha_g is not None:
            aux["pool_alpha_mean"] = alpha_g.mean().detach()

        return pooled, aux

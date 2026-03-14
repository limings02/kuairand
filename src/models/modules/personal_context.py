"""
Personal Context Encoder（个性化上下文编码器）。

设计目标
--------
PPNet 不是直接看主干表示，而是先看一个更轻量、更可解释的个性化条件向量 p_ctx。
这里显式把三类信息拆开建模：
1) 场景信息：tab / hour_of_day / day_of_week / is_weekend
2) 用户属性：user sparse embeddings + user dense
3) 活跃度代理：hist_len、is_lowactive_period、user_active_degree embedding

这样做的原因：
- 场景信息决定“现在处于什么推荐环境”，同一个用户在不同 tab / 时间段兴趣会漂移；
- 用户属性决定“这个样本属于什么类型的人”，是千人千面的基础；
- 活跃度代理让 PPNet 对长历史/低活跃尾部样本做更细的条件化，增益通常不大，但更稳定。

该模块本身不做 embedding lookup，而是复用主模型已经查好的 embedding / dense tensor，
避免重复参数与重复显存占用。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _build_mlp(
    input_dim: int,
    hidden_units: Sequence[int],
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    """构建轻量投影 MLP。"""
    layers: List[nn.Module] = []
    prev_dim = int(input_dim)
    for hidden_dim in hidden_units:
        hidden_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, int(output_dim)))
    return nn.Sequential(*layers)


class PersonalContextEncoder(nn.Module):
    """
    将场景 + 用户 + 活跃度代理压缩为 p_ctx。

    输入约定
    --------
    - context_embs:         [B, D_ctx_raw]，通常来自 tab/time/is_weekend embedding 拼接
    - user_sparse_embs:     [B, D_user_sparse]
    - user_dense:           [B, D_user_dense]
    - activity_dense_feats: [B, D_activity_dense]，如 hist_len_norm / is_lowactive_period
    - activity_sparse_embs: [B, D_activity_sparse]，如 user_active_degree embedding

    输出
    ----
    - p_ctx: [B, D_ctx]
    """

    def __init__(
        self,
        context_input_dim: int,
        user_sparse_input_dim: int,
        user_dense_dim: int,
        activity_dense_dim: int,
        activity_sparse_dim: int,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.config = config or {}
        self.use_user_dense = bool(self.config.get("use_user_dense", True))
        self.dropout = float(self.config.get("dropout", 0.0))
        self.use_layernorm = bool(self.config.get("layernorm", True))

        self.context_input_dim = int(context_input_dim)
        self.user_sparse_input_dim = int(user_sparse_input_dim)
        self.user_dense_dim = int(user_dense_dim)
        self.activity_dense_dim = int(activity_dense_dim)
        self.activity_sparse_dim = int(activity_sparse_dim)

        user_input_dim = self.user_sparse_input_dim
        if self.use_user_dense:
            user_input_dim += self.user_dense_dim
        self.user_ctx_dim = int(
            self.config.get(
                "user_ctx_dim",
                min(64, max(16, user_input_dim)) if user_input_dim > 0 else 0,
            )
        )
        self.activity_ctx_dim = int(
            self.config.get(
                "activity_ctx_dim",
                min(32, max(8, self.activity_dense_dim + self.activity_sparse_dim))
                if (self.activity_dense_dim + self.activity_sparse_dim) > 0
                else 0,
            )
        )

        user_hidden_dim = int(
            self.config.get("user_ctx_hidden_dim", max(self.user_ctx_dim, min(64, max(16, user_input_dim))))
        )
        activity_hidden_dim = int(
            self.config.get(
                "activity_ctx_hidden_dim",
                max(self.activity_ctx_dim, min(32, max(8, self.activity_dense_dim + self.activity_sparse_dim))),
            )
        )

        self.user_proj: Optional[nn.Module] = None
        if user_input_dim > 0 and self.user_ctx_dim > 0:
            self.user_proj = _build_mlp(
                input_dim=user_input_dim,
                hidden_units=[user_hidden_dim],
                output_dim=self.user_ctx_dim,
                dropout=self.dropout,
            )

        self.activity_proj: Optional[nn.Module] = None
        activity_input_dim = self.activity_dense_dim + self.activity_sparse_dim
        if activity_input_dim > 0 and self.activity_ctx_dim > 0:
            self.activity_proj = _build_mlp(
                input_dim=activity_input_dim,
                hidden_units=[activity_hidden_dim],
                output_dim=self.activity_ctx_dim,
                dropout=self.dropout,
            )

        total_output_dim = self.context_input_dim
        if self.user_proj is not None:
            total_output_dim += self.user_ctx_dim
        if self.activity_proj is not None:
            total_output_dim += self.activity_ctx_dim

        if total_output_dim <= 0:
            raise ValueError("PersonalContextEncoder 输出维度为 0，请检查上下文输入配置。")

        self.output_dim = int(total_output_dim)
        self.output_norm = nn.LayerNorm(self.output_dim) if self.use_layernorm else nn.Identity()
        self.last_debug_stats: Dict[str, Any] = {}

    def forward(
        self,
        context_embs: Optional[torch.Tensor],
        user_sparse_embs: Optional[torch.Tensor],
        user_dense: Optional[torch.Tensor],
        activity_dense_feats: Optional[torch.Tensor],
        activity_sparse_embs: Optional[torch.Tensor],
        return_debug: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        parts: List[torch.Tensor] = []
        stats: Dict[str, Any] = {}

        if context_embs is not None and context_embs.shape[-1] > 0:
            parts.append(context_embs)
            stats["personal_context_scene_dim"] = int(context_embs.shape[-1])
        else:
            stats["personal_context_scene_dim"] = 0

        if self.user_proj is not None:
            user_parts: List[torch.Tensor] = []
            if user_sparse_embs is not None and user_sparse_embs.shape[-1] > 0:
                user_parts.append(user_sparse_embs)
            if self.use_user_dense and user_dense is not None and user_dense.shape[-1] > 0:
                user_parts.append(user_dense.float())
            if user_parts:
                u_ctx = self.user_proj(torch.cat(user_parts, dim=-1))
                parts.append(u_ctx)
                stats["personal_context_user_dim"] = int(u_ctx.shape[-1])
            else:
                stats["personal_context_user_dim"] = 0

        if self.activity_proj is not None:
            activity_parts: List[torch.Tensor] = []
            if activity_dense_feats is not None and activity_dense_feats.shape[-1] > 0:
                activity_parts.append(activity_dense_feats.float())
            if activity_sparse_embs is not None and activity_sparse_embs.shape[-1] > 0:
                activity_parts.append(activity_sparse_embs)
            if activity_parts:
                act_ctx = self.activity_proj(torch.cat(activity_parts, dim=-1))
                parts.append(act_ctx)
                stats["personal_context_activity_dim"] = int(act_ctx.shape[-1])
            else:
                stats["personal_context_activity_dim"] = 0

        if not parts:
            raise RuntimeError("构造 p_ctx 失败：没有可用的场景/用户/活跃度输入。")

        p_ctx = torch.cat(parts, dim=-1)
        p_ctx = self.output_norm(p_ctx)

        stats["personal_context_output_dim"] = int(p_ctx.shape[-1])
        stats["personal_context_norm_mean"] = float(p_ctx.norm(dim=-1).mean().detach().cpu().item())
        self.last_debug_stats = stats

        if return_debug:
            return p_ctx, stats
        return p_ctx

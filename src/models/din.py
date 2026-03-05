"""
DIN 及 ADS-lite（PSRG + PCRG）实现。

本文件在原 DIN baseline 基础上做增量扩展，支持三种变体：
1) din               : 原始 DIN
2) din_psrg          : 历史先经 PSRG-lite，再走单 query DIN attention
3) din_psrg_pcrg     : 历史经 PSRG-lite + PCRG-lite 多 query attention

设计原则
────────
- 保留原训练/评估循环接口（forward(batch) -> logits）
- 保留 video/author 共享 embedding（hist 与 cand 同表）
- 通过 config 开关控制 PSRG/PCRG，便于做 ablation
- 对 padding 严格 mask，防止无效位置污染注意力
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.models.modules.domain_context import DomainContextEncoder
from src.models.modules.pcrg import PCRGLite
from src.models.modules.psrg import PSRGLite

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 辅助：列名 → vocab 名称映射
# ─────────────────────────────────────────────────────────────

def col_to_vocab_name(col_name: str) -> str:
    """
    从 parquet 列名推导 vocab 名称。
    规则：
      - cand_xxx -> xxx
      - hist_xxx -> xxx（用于 hist_tab / hist_delta_t_bucket / hist_play_ratio_bucket）
      - 其余保持原名
    """
    if col_name.startswith("cand_"):
        return col_name[len("cand_"):]
    if col_name.startswith("hist_"):
        return col_name[len("hist_"):]
    return col_name


# ─────────────────────────────────────────────────────────────
# HashEmbedding
# ─────────────────────────────────────────────────────────────

class HashEmbedding(nn.Module):
    """哈希 Embedding：将大词表映射到较小桶，显著降低显存占用。"""

    def __init__(self, num_buckets: int, emb_dim: int, padding_idx: int = 0):
        super().__init__()
        self.num_buckets = num_buckets
        self.emb = nn.Embedding(num_buckets, emb_dim, padding_idx=padding_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_pad = (x == 0)
        hashed = (x % (self.num_buckets - 1)) + 1
        hashed = hashed.masked_fill(is_pad, 0)
        return self.emb(hashed)


# ─────────────────────────────────────────────────────────────
# DIN Attention
# ─────────────────────────────────────────────────────────────

def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    mask softmax（fp16 友好版本）。

    Args:
        scores: [B, L]
        mask:   [B, L]，1=有效，0=padding

    Returns:
        probs:   [B, L]
        all_pad: [B]，是否该样本全为 padding
    """
    mask_bool = mask > 0
    scores = scores.masked_fill(~mask_bool, -1e4)
    probs = torch.softmax(scores, dim=dim)
    probs = probs * mask_bool.to(probs.dtype)

    denom = probs.sum(dim=dim, keepdim=True)
    probs = torch.where(denom > 0, probs / denom.clamp_min(1e-12), torch.zeros_like(probs))
    all_pad = (mask_bool.sum(dim=dim) == 0)
    return probs, all_pad


class DINAttention(nn.Module):
    """
    DIN 单 query target attention。

    打分公式（经典 DIN）：
      score(q, k) = MLP([q, k, q-k, q*k])
    """

    def __init__(self, item_dim: int, hidden_units: List[int]):
        super().__init__()
        input_dim = 4 * item_dim
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.PReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        mask: torch.Tensor,
        return_debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            query: [B, D]
            keys:  [B, L, D]
            mask:  [B, L]
        """
        B, L, _ = keys.shape
        q = query.unsqueeze(1).expand(-1, L, -1)
        att_input = torch.cat([q, keys, q - keys, q * keys], dim=-1)
        att_scores = self.mlp(att_input).squeeze(-1)

        att_weights, all_pad = _masked_softmax(att_scores, mask, dim=-1)
        user_interest = torch.bmm(att_weights.unsqueeze(1), keys).squeeze(1)

        if return_debug:
            return user_interest, {
                "all_pad_count": int(all_pad.sum().item()),
                "attn_entropy_mean": float(
                    (-(att_weights * att_weights.clamp_min(1e-12).log()).sum(dim=-1)).mean().item()
                ),
            }
        return user_interest


# ─────────────────────────────────────────────────────────────
# DIN + ADS-lite
# ─────────────────────────────────────────────────────────────

class DINModel(nn.Module):
    """
    兼容版 DIN 模型。

    兼容两种配置风格：
    - 旧版：model.video_id_emb_dim / model.att_hidden_units / ...
    - 新版：model.emb_dims.* / model.din.* / model.psrg.* / model.pcrg.*
    """

    def __init__(self, config: dict, vocab_sizes: Dict[str, int]):
        super().__init__()
        self.config = config
        self.vocab_sizes = vocab_sizes

        mcfg = config["model"]
        fcfg = config["fields"]
        self.fcfg = fcfg

        emb_cfg = mcfg.get("emb_dims", {})
        self.video_emb_dim = int(emb_cfg.get("video_id", mcfg.get("video_id_emb_dim", 32)))
        self.author_emb_dim = int(emb_cfg.get("author_id", mcfg.get("author_id_emb_dim", 16)))
        self.item_repr_dim = self.video_emb_dim + self.author_emb_dim

        self.sparse_emb_dim = int(emb_cfg.get("small_cat", mcfg.get("sparse_emb_dim", 4)))
        self.large_sparse_emb_dim = int(mcfg.get("large_sparse_emb_dim", 8))
        self.large_sparse_threshold = int(mcfg.get("large_sparse_threshold", 100))

        self.variant = str(mcfg.get("variant", "din")).lower()
        if self.variant not in {"din", "din_psrg", "din_psrg_pcrg"}:
            raise ValueError(f"不支持的 model.variant={self.variant}")

        self.psrg_cfg = mcfg.get("psrg", {})
        self.pcrg_cfg = mcfg.get("pcrg", {})
        self.fusion_cfg = mcfg.get("fusion", {})
        self.domain_cfg = mcfg.get("domain_context", {})
        self.hist_repr_cfg = mcfg.get("hist_repr", {})
        self.cand_repr_cfg = mcfg.get("cand_repr", {})
        self.debug_cfg = mcfg.get("debug", {})

        # 仅打印一次的 warning/shape 调试标记
        self._warned_messages: set[str] = set()
        self._printed_shape_once = False
        self._last_debug_stats: Dict[str, Any] = {}

        # ── 1) 共享 video/author embedding（必须共享）──
        if mcfg.get("use_hash_embedding", True):
            self.video_id_emb = HashEmbedding(
                int(mcfg.get("video_id_hash_buckets", 1_000_000)),
                self.video_emb_dim,
            )
            self.author_id_emb = HashEmbedding(
                int(mcfg.get("author_id_hash_buckets", 500_000)),
                self.author_emb_dim,
            )
            logger.info(
                "使用 HashEmbedding: video=%d(dim=%d), author=%d(dim=%d)",
                int(mcfg.get("video_id_hash_buckets", 1_000_000)),
                self.video_emb_dim,
                int(mcfg.get("author_id_hash_buckets", 500_000)),
                self.author_emb_dim,
            )
        else:
            self.video_id_emb = nn.Embedding(vocab_sizes["video_id"], self.video_emb_dim, padding_idx=0)
            self.author_id_emb = nn.Embedding(vocab_sizes["author_id"], self.author_emb_dim, padding_idx=0)
            logger.info(
                "使用全量 Embedding: video=%d(dim=%d), author=%d(dim=%d)",
                vocab_sizes["video_id"],
                self.video_emb_dim,
                vocab_sizes["author_id"],
                self.author_emb_dim,
            )

        # ── 2) 其余 sparse embedding ──
        self.sparse_embeddings = nn.ModuleDict()

        cand_cols = fcfg["cand_cols"]
        self.cand_video_col = cand_cols["video_id"]
        self.cand_author_col = cand_cols["author_id"]

        self.cand_side_cols: List[str] = []
        for feat_name, col_name in cand_cols.items():
            if feat_name in {"video_id", "author_id"}:
                continue
            self.cand_side_cols.append(col_name)
            self._register_sparse_emb(col_name)

        self.context_sparse_cols: List[str] = list(fcfg.get("context_sparse_cols", []))
        for col_name in self.context_sparse_cols:
            self._register_sparse_emb(col_name)

        self.user_sparse_cols: List[str] = list(fcfg.get("user_sparse_cols", []))
        for col_name in self.user_sparse_cols:
            self._register_sparse_emb(col_name)

        self.user_dense_cols: List[str] = list(fcfg.get("user_dense_cols", []))
        self.optional_hist_seq_cols: List[str] = list(fcfg.get("optional_hist_seq_cols", []))
        for col_name in self.optional_hist_seq_cols:
            self._register_sparse_emb(col_name)

        # ── 3) DIN attention（主干保留）──
        din_cfg = mcfg.get("din", {})
        self.din_attention = DINAttention(
            item_dim=self.item_repr_dim,
            hidden_units=list(din_cfg.get("att_hidden_units", mcfg.get("att_hidden_units", [64, 32]))),
        )

        # ── 4) 历史 optional 特征融合（可选）──
        self.hist_feature_proj = nn.ModuleDict()
        hist_feature_flags = {
            "hist_tab": bool(self.hist_repr_cfg.get("use_hist_tab", False)),
            "hist_delta_t_bucket": bool(self.hist_repr_cfg.get("use_hist_delta_t_bucket", False)),
            "hist_play_ratio_bucket": bool(self.hist_repr_cfg.get("use_hist_play_ratio_bucket", False)),
        }
        for col_name, enabled in hist_feature_flags.items():
            if enabled and col_name in self.sparse_embeddings:
                in_dim = self.sparse_embeddings[col_name].embedding_dim
                self.hist_feature_proj[col_name] = nn.Linear(in_dim, self.item_repr_dim, bias=False)

        # ── 5) 候选 side 融合到 cand_item_repr（可选）──
        self.cand_fuse_side_into_item = bool(self.cand_repr_cfg.get("fuse_side_into_item", False))
        self.cand_side_proj = nn.ModuleDict()
        if self.cand_fuse_side_into_item:
            cand_side_flags = {
                "cand_video_type": bool(self.cand_repr_cfg.get("use_video_type", True)),
                "cand_upload_type": bool(self.cand_repr_cfg.get("use_upload_type", True)),
                "cand_video_duration_bucket": bool(self.cand_repr_cfg.get("use_video_duration_bucket", True)),
            }
            for col_name in self.cand_side_cols:
                if cand_side_flags.get(col_name, True) and col_name in self.sparse_embeddings:
                    in_dim = self.sparse_embeddings[col_name].embedding_dim
                    self.cand_side_proj[col_name] = nn.Linear(in_dim, self.item_repr_dim, bias=False)

        # ── 6) 域上下文编码器（供 PSRG/PCRG 共享）──
        self.domain_ctx_fields: List[str] = ["tab"]
        if bool(self.domain_cfg.get("use_hour_of_day", False)):
            self.domain_ctx_fields.append("hour_of_day")
        if bool(self.domain_cfg.get("use_day_of_week", False)):
            self.domain_ctx_fields.append("day_of_week")

        domain_input_dim = 0
        for col_name in self.domain_ctx_fields:
            if col_name in self.sparse_embeddings:
                domain_input_dim += self.sparse_embeddings[col_name].embedding_dim
            else:
                self._warn_once(f"domain_missing_{col_name}", f"域上下文字段 '{col_name}' 缺少 embedding，将自动忽略")

        self.use_user_ctx = bool(self.domain_cfg.get("use_user_context", False))
        self.user_ctx_mlp: Optional[nn.Sequential] = None
        if self.use_user_ctx:
            user_ctx_in_dim = 0
            for col_name in self.user_sparse_cols:
                if col_name in self.sparse_embeddings:
                    user_ctx_in_dim += self.sparse_embeddings[col_name].embedding_dim
            user_ctx_in_dim += len(self.user_dense_cols)

            if user_ctx_in_dim > 0:
                user_ctx_hidden = int(self.domain_cfg.get("user_ctx_hidden_dim", 64))
                user_ctx_dim = int(self.domain_cfg.get("user_ctx_dim", 32))
                self.user_ctx_mlp = nn.Sequential(
                    nn.Linear(user_ctx_in_dim, user_ctx_hidden),
                    nn.ReLU(),
                    nn.Linear(user_ctx_hidden, user_ctx_dim),
                )
                domain_input_dim += user_ctx_dim
            else:
                self._warn_once("user_ctx_empty", "use_user_context=true 但没有可用 user 特征，已自动禁用")
                self.use_user_ctx = False

        if domain_input_dim <= 0:
            raise ValueError("DomainContextEncoder 输入维度为 0，请检查 tab/hour/day/user_ctx 配置")

        self.d_ctx_dim = int(self.domain_cfg.get("output_dim", self.item_repr_dim))
        self.domain_context_encoder = DomainContextEncoder(
            input_dim=domain_input_dim,
            output_dim=self.d_ctx_dim,
            hidden_units=list(self.domain_cfg.get("hidden_units", [64])),
            dropout=float(self.domain_cfg.get("dropout", 0.0)),
            use_layernorm=bool(self.domain_cfg.get("layernorm", True)),
        )

        # ── 7) PSRG/PCRG 分支配置 ──
        self.psrg_enabled = (
            self.variant in {"din_psrg", "din_psrg_pcrg"}
            and bool(self.psrg_cfg.get("enabled", True))
        )
        self.pcrg_enabled = (
            self.variant == "din_psrg_pcrg"
            and bool(self.pcrg_cfg.get("enabled", True))
        )

        self.use_hist_tab_in_psrg = bool(self.psrg_cfg.get("use_hist_tab_in_psrg", False))
        self.use_current_tab_always = bool(self.psrg_cfg.get("use_current_tab_always", True))

        self.hist_tab_to_dctx: Optional[nn.Linear] = None
        if self.use_hist_tab_in_psrg and "hist_tab" in self.sparse_embeddings:
            self.hist_tab_to_dctx = nn.Linear(self.sparse_embeddings["hist_tab"].embedding_dim, self.d_ctx_dim)

        self.psrg: Optional[PSRGLite] = None
        if self.psrg_enabled:
            self.psrg = PSRGLite(
                item_dim=self.item_repr_dim,
                d_ctx_dim=self.d_ctx_dim,
                mode=str(self.psrg_cfg.get("mode", "gated_residual")),
                hidden_units=list(self.psrg_cfg.get("hidden_units", [64])),
                dropout=float(self.psrg_cfg.get("dropout", 0.0)),
                use_layernorm=bool(self.psrg_cfg.get("layernorm", True)),
            )

        self.pcrg: Optional[PCRGLite] = None
        if self.pcrg_enabled:
            self.pcrg = PCRGLite(
                item_dim=self.item_repr_dim,
                d_ctx_dim=self.d_ctx_dim,
                num_queries=int(self.pcrg_cfg.get("num_queries", 4)),
                query_dim=int(self.pcrg_cfg.get("query_dim", self.item_repr_dim)),
                score_type=str(self.pcrg_cfg.get("score_type", "din_mlp")),
                hidden_units=list(self.pcrg_cfg.get("hidden_units", [64, 32])),
                aggregation=str(self.pcrg_cfg.get("aggregation", "mean_pool")),
                dropout=float(self.pcrg_cfg.get("dropout", 0.0)),
            )
            if int(self.pcrg_cfg.get("query_dim", self.item_repr_dim)) != self.item_repr_dim:
                logger.info(
                    "PCRG query_dim(%d) != item_dim(%d)，已自动启用 query_to_item 投影层",
                    int(self.pcrg_cfg.get("query_dim", self.item_repr_dim)),
                    self.item_repr_dim,
                )

        # ── 8) 输出融合策略 ──
        self.fusion_mode = str(self.fusion_cfg.get("mode", "concat"))
        self.proj_after_concat = bool(self.fusion_cfg.get("proj_after_concat", True))
        self.fusion_concat_proj: Optional[nn.Linear] = None

        if self.pcrg_enabled and self.fusion_mode == "concat" and self.proj_after_concat:
            self.fusion_concat_proj = nn.Linear(2 * self.item_repr_dim, self.item_repr_dim)

        # ── 9) DNN 塔 ──
        dnn_input_dim = self._calc_dnn_input_dim()
        dnn_hidden_units = list(mcfg.get("dnn_hidden_units", [256, 128, 64]))
        dnn_dropout = float(mcfg.get("dnn_dropout", 0.1))
        dnn_use_bn = bool(mcfg.get("dnn_use_bn", True))

        layers: list[nn.Module] = []
        prev_dim = dnn_input_dim
        for hidden_dim in dnn_hidden_units:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if dnn_use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dnn_dropout > 0:
                layers.append(nn.Dropout(dnn_dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.dnn = nn.Sequential(*layers)

        logger.info(
            "模型变体: variant=%s | psrg=%s | pcrg=%s | fusion=%s",
            self.variant,
            self.psrg_enabled,
            self.pcrg_enabled,
            self.fusion_mode if self.pcrg_enabled else "none",
        )
        logger.info("DNN 输入维度: %d", dnn_input_dim)

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("模型参数量: total=%s, trainable=%s", f"{total_params:,}", f"{trainable_params:,}")

    # ─────────────────────────────────────────────────────────
    # 公共调试接口
    # ─────────────────────────────────────────────────────────

    def get_and_reset_debug_stats(self) -> Dict[str, Any]:
        """供训练循环读取最近一次 forward 的 ADS 调试统计。"""
        stats = self._last_debug_stats
        self._last_debug_stats = {}
        return stats

    # ─────────────────────────────────────────────────────────
    # 私有工具方法
    # ─────────────────────────────────────────────────────────

    def _warn_once(self, key: str, message: str):
        if key not in self._warned_messages:
            logger.warning(message)
            self._warned_messages.add(key)

    def _get_emb_dim(self, vocab_name: str) -> int:
        vs = self.vocab_sizes.get(vocab_name, 1)
        if vs > self.large_sparse_threshold:
            return self.large_sparse_emb_dim
        return self.sparse_emb_dim

    def _register_sparse_emb(self, col_name: str):
        vocab_name = col_to_vocab_name(col_name)
        vs = self.vocab_sizes.get(vocab_name)
        if vs is None:
            self._warn_once(
                f"missing_vocab_{col_name}",
                f"找不到列 '{col_name}' 对应的 vocab('{vocab_name}')，将跳过该 embedding",
            )
            return
        emb_dim = self._get_emb_dim(vocab_name)
        self.sparse_embeddings[col_name] = nn.Embedding(vs, emb_dim, padding_idx=0)

    def _interest_output_dim(self) -> int:
        if self.pcrg_enabled and self.fusion_mode == "concat" and not self.proj_after_concat:
            return 2 * self.item_repr_dim
        return self.item_repr_dim

    def _calc_dnn_input_dim(self) -> int:
        dim = 0
        # user_interest（DIN 或 ADS 融合后）
        dim += self._interest_output_dim()
        # cand_item_repr
        dim += self.item_repr_dim

        # cand side
        for col_name in self.cand_side_cols:
            if col_name in self.sparse_embeddings:
                dim += self.sparse_embeddings[col_name].embedding_dim

        # context sparse
        for col_name in self.context_sparse_cols:
            if col_name in self.sparse_embeddings:
                dim += self.sparse_embeddings[col_name].embedding_dim

        # user sparse
        for col_name in self.user_sparse_cols:
            if col_name in self.sparse_embeddings:
                dim += self.sparse_embeddings[col_name].embedding_dim

        # user dense
        dim += len(self.user_dense_cols)
        return dim

    def _validate_hist_shape(self, batch: Dict[str, torch.Tensor]):
        """健壮性检查：hist_video_id / hist_author_id / hist_mask shape 必须一致。"""
        hv = batch["hist_video_id"]
        ha = batch["hist_author_id"]
        hm = batch["hist_mask"]

        if hv.shape != ha.shape or hv.shape != hm.shape:
            raise ValueError(
                "历史字段 shape 不一致: "
                f"hist_video_id={list(hv.shape)}, "
                f"hist_author_id={list(ha.shape)}, "
                f"hist_mask={list(hm.shape)}"
            )
        if hv.ndim != 2:
            raise ValueError(
                f"hist_video_id/hist_author_id/hist_mask 必须是 [B,L]，当前 hist_video_id={list(hv.shape)}"
            )

    def _get_user_dense(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if "user_dense" in batch:
            return batch["user_dense"].float()

        parts = []
        for col_name in self.user_dense_cols:
            if col_name in batch:
                t = batch[col_name].float()
                if t.ndim == 1:
                    t = t.unsqueeze(1)
                parts.append(t)
        if not parts:
            return None
        return torch.cat(parts, dim=-1)

    def _build_cand_item_repr(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        cand_vid_emb = self.video_id_emb(batch[self.cand_video_col])
        cand_aid_emb = self.author_id_emb(batch[self.cand_author_col])
        cand_repr = torch.cat([cand_vid_emb, cand_aid_emb], dim=-1)  # [B, D_item]

        if self.cand_fuse_side_into_item:
            for col_name, proj in self.cand_side_proj.items():
                if col_name in batch and col_name in self.sparse_embeddings:
                    side_emb = self.sparse_embeddings[col_name](batch[col_name])
                    cand_repr = cand_repr + proj(side_emb)
        return cand_repr

    def _build_hist_item_repr(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        hist_vid_emb = self.video_id_emb(batch["hist_video_id"])      # [B, L, Dv]
        hist_aid_emb = self.author_id_emb(batch["hist_author_id"])    # [B, L, Da]
        hist_repr = torch.cat([hist_vid_emb, hist_aid_emb], dim=-1)    # [B, L, D_item]

        # 可选历史辅助特征（tab/delta_t/play_ratio）通过投影后残差加到 item 表示。
        for col_name, proj in self.hist_feature_proj.items():
            if col_name in batch and col_name in self.sparse_embeddings:
                hist_feat_emb = self.sparse_embeddings[col_name](batch[col_name])
                hist_repr = hist_repr + proj(hist_feat_emb)

        return hist_repr

    def _build_domain_context(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """编码全局域上下文 d_ctx: [B, D_ctx]。"""
        parts = []

        for col_name in self.domain_ctx_fields:
            if col_name in batch and col_name in self.sparse_embeddings:
                parts.append(self.sparse_embeddings[col_name](batch[col_name]))
            else:
                self._warn_once(
                    f"missing_domain_col_{col_name}",
                    f"domain context 字段 '{col_name}' 不在 batch 中，自动跳过",
                )

        if self.use_user_ctx and self.user_ctx_mlp is not None:
            user_ctx_parts = []
            for col_name in self.user_sparse_cols:
                if col_name in batch and col_name in self.sparse_embeddings:
                    user_ctx_parts.append(self.sparse_embeddings[col_name](batch[col_name]))
            dense = self._get_user_dense(batch)
            if dense is not None:
                user_ctx_parts.append(dense)

            if user_ctx_parts:
                user_ctx = torch.cat(user_ctx_parts, dim=-1)
                parts.append(self.user_ctx_mlp(user_ctx))

        if not parts:
            raise RuntimeError("构建 d_ctx 失败：没有可用的上下文输入")

        ctx_concat = torch.cat(parts, dim=-1)
        return self.domain_context_encoder(ctx_concat)

    def _build_psrg_context_seq(
        self,
        batch: Dict[str, torch.Tensor],
        d_ctx: torch.Tensor,
        hist_len: int,
    ) -> torch.Tensor:
        """
        构造 PSRG 使用的逐位置域上下文 [B, L, D_ctx]。

        规则：
        - use_current_tab_always=true: 先用当前 tab 对应 d_ctx 广播到 L
        - use_hist_tab_in_psrg=true 且 hist_tab 存在: 再加上 hist_tab 的位置编码
        - 若 hist_tab 缺失：自动回退，不报错，仅 warning
        """
        B = d_ctx.shape[0]
        device = d_ctx.device

        if self.use_current_tab_always:
            d_ctx_seq = d_ctx.unsqueeze(1).expand(-1, hist_len, -1)
        else:
            d_ctx_seq = torch.zeros(B, hist_len, self.d_ctx_dim, device=device, dtype=d_ctx.dtype)

        if self.use_hist_tab_in_psrg:
            if "hist_tab" in batch and "hist_tab" in self.sparse_embeddings and self.hist_tab_to_dctx is not None:
                hist_tab_emb = self.sparse_embeddings["hist_tab"](batch["hist_tab"])     # [B,L,D_tab]
                hist_tab_ctx = self.hist_tab_to_dctx(hist_tab_emb)                         # [B,L,D_ctx]
                d_ctx_seq = d_ctx_seq + hist_tab_ctx
            else:
                self._warn_once(
                    "hist_tab_fallback",
                    "use_hist_tab_in_psrg=true 但 batch 不含 hist_tab（或无对应 embedding），"
                    "已自动回退为当前 tab 广播上下文",
                )

        return d_ctx_seq

    def _fuse_interest(self, user_interest_din: torch.Tensor, user_interest_ads: torch.Tensor) -> torch.Tensor:
        """DIN 与 ADS 多兴趣输出融合。"""
        if self.fusion_mode == "replace":
            return user_interest_ads

        if self.fusion_mode == "concat":
            fused = torch.cat([user_interest_din, user_interest_ads], dim=-1)
            if self.fusion_concat_proj is not None:
                fused = self.fusion_concat_proj(fused)
            return fused

        if self.fusion_mode == "residual_add":
            if user_interest_din.shape[-1] != user_interest_ads.shape[-1]:
                raise ValueError(
                    "fusion_mode=residual_add 要求 DIN/ADS 兴趣向量维度一致，"
                    f"当前 {user_interest_din.shape[-1]} vs {user_interest_ads.shape[-1]}"
                )
            return user_interest_din + user_interest_ads

        raise ValueError(f"不支持的 fusion.mode={self.fusion_mode}")

    # ─────────────────────────────────────────────────────────
    # forward
    # ─────────────────────────────────────────────────────────

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self._validate_hist_shape(batch)

        hist_mask = batch["hist_mask"].float()                       # [B, L]
        cand_item_repr = self._build_cand_item_repr(batch)            # [B, D]
        hist_item_repr = self._build_hist_item_repr(batch)            # [B, L, D]

        # d_ctx 仅在 PSRG/PCRG 路径需要时构建
        need_domain_ctx = self.psrg_enabled or self.pcrg_enabled
        d_ctx = self._build_domain_context(batch) if need_domain_ctx else None

        # ── 路径 A: PSRG 处理历史 ──
        hist_for_attn = hist_item_repr
        psrg_all_pad_count = 0
        if self.psrg_enabled and self.psrg is not None:
            d_ctx_seq = self._build_psrg_context_seq(batch, d_ctx, hist_item_repr.shape[1])
            hist_for_attn, psrg_aux = self.psrg(hist_item_repr, d_ctx_seq, hist_mask)
            psrg_all_pad_count = int(psrg_aux.get("all_pad_count", 0))

        # ── 路径 B: 原 DIN 单 query attention（基线主干保留）──
        user_interest_din, din_aux = self.din_attention(
            query=cand_item_repr,
            keys=hist_for_attn,
            mask=hist_mask,
            return_debug=True,
        )

        # ── 路径 C: PCRG 多 query attention（可选）──
        pcrg_aux: Dict[str, Any] = {}
        if self.pcrg_enabled and self.pcrg is not None:
            user_interest_ads, pcrg_aux = self.pcrg(
                cand_item_repr=cand_item_repr,
                d_ctx=d_ctx,
                hist_repr=hist_for_attn,
                hist_mask=hist_mask,
            )
            user_interest = self._fuse_interest(user_interest_din, user_interest_ads)
        else:
            user_interest = user_interest_din

        # ── 拼接进入最终 DNN ──
        parts = [user_interest, cand_item_repr]

        for col_name in self.cand_side_cols:
            if col_name in self.sparse_embeddings and col_name in batch:
                parts.append(self.sparse_embeddings[col_name](batch[col_name]))

        for col_name in self.context_sparse_cols:
            if col_name in self.sparse_embeddings and col_name in batch:
                parts.append(self.sparse_embeddings[col_name](batch[col_name]))

        for col_name in self.user_sparse_cols:
            if col_name in self.sparse_embeddings and col_name in batch:
                parts.append(self.sparse_embeddings[col_name](batch[col_name]))

        user_dense = self._get_user_dense(batch)
        if user_dense is not None:
            parts.append(user_dense)

        dnn_input = torch.cat(parts, dim=-1)
        logits = self.dnn(dnn_input).squeeze(-1)

        # 记录调试统计，供 trainer 可选打印
        self._last_debug_stats = {
            "variant": self.variant,
            "din_attn_entropy_mean": din_aux.get("attn_entropy_mean", 0.0),
            "din_all_pad_count": int(din_aux.get("all_pad_count", 0)),
            "psrg_all_pad_count": int(psrg_all_pad_count),
            "pcrg_attn_entropy_mean": float(
                pcrg_aux.get("attn_entropy_mean", torch.tensor(0.0)).item()
            ) if pcrg_aux else 0.0,
            "pcrg_query_interest_var": float(
                pcrg_aux.get("query_interest_var", torch.tensor(0.0)).item()
            ) if pcrg_aux else 0.0,
            "pcrg_all_pad_count": int(pcrg_aux.get("all_pad_count", 0)) if pcrg_aux else 0,
        }

        # debug 模式下仅首个 batch 打印关键 shape
        if bool(self.debug_cfg.get("print_shapes_once", False)) and (not self._printed_shape_once):
            self._printed_shape_once = True
            logger.info(
                "[Debug Shapes] variant=%s | cand_item=%s | hist_item=%s | user_interest=%s | logits=%s",
                self.variant,
                list(cand_item_repr.shape),
                list(hist_for_attn.shape),
                list(user_interest.shape),
                list(logits.shape),
            )

        return logits

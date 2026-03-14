"""
DIN 及 ADS-lite（PSRG + PCRG）实现。

本文件在原 DIN baseline 基础上做增量扩展，支持四种变体：
1) din               : 原始 DIN
2) din_psrg          : 历史先经 PSRG-lite，再走单 query DIN attention
3) din_psrg_pcrg     : 历史经 PSRG-lite + PCRG-lite 多 query attention
4) din_psrg_pcrg_transformer : 在 PSRG + PCRG 之上追加 TransformerFusion

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
from src.models.modules.feature_slices import build_feature_slices
from src.models.modules.mbcnet import MBCNetHead
from src.models.modules.personal_context import PersonalContextEncoder
from src.models.modules.pcrg import PCRGLite
from src.models.modules.ppnet import PPNet
from src.models.modules.psrg import PSRGLite
from src.models.modules.target_attention_dnn import TargetAttentionDNN
from src.models.modules.transformer_fusion import TransformerFusion

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
        if self.variant not in {"din", "din_psrg", "din_psrg_pcrg", "din_psrg_pcrg_transformer"}:
            raise ValueError(f"不支持的 model.variant={self.variant}")
        self.max_hist_len = int(mcfg.get("max_hist_len", 50))

        self.psrg_cfg = mcfg.get("psrg", {})
        self.pcrg_cfg = mcfg.get("pcrg", {})
        self.fusion_cfg = mcfg.get("fusion", {})
        self.transformer_cfg = mcfg.get("transformer_fusion", {})
        self.domain_cfg = mcfg.get("domain_context", {})
        self.hist_repr_cfg = mcfg.get("hist_repr", {})
        self.cand_repr_cfg = mcfg.get("cand_repr", {})
        self.head_cfg = mcfg.get("head", {})
        self.head_type = str(self.head_cfg.get("type", "mlp")).lower()
        self.mbcnet_cfg = self.head_cfg.get("mbcnet", {})
        self.ppnet_cfg = mcfg.get("ppnet", {})
        self.ppnet_enabled = bool(self.ppnet_cfg.get("enabled", False))
        self.ppnet_context_cfg = self.ppnet_cfg.get("context", {})
        self.debug_cfg = mcfg.get("debug", {})
        if self.head_type not in {"mlp", "mbcnet"}:
            raise ValueError(f"不支持的 model.head.type={self.head_type}，仅支持 mlp / mbcnet")

        # 仅打印一次的 warning/shape 调试标记
        self._warned_messages: set[str] = set()
        self._printed_shape_once = False
        self._last_debug_stats: Dict[str, Any] = {}
        self.personal_context_encoder: Optional[PersonalContextEncoder] = None
        self.ppnet: Optional[PPNet] = None
        self.ppnet_context_cols: List[str] = []
        self.ppnet_activity_dense_dim = 0
        self.ppnet_activity_sparse_dim = 0

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
        self.din_attention = TargetAttentionDNN(
            item_dim=self.item_repr_dim,
            hidden_units=list(din_cfg.get("att_hidden_units", mcfg.get("att_hidden_units", [64, 32]))),
            activation="prelu",
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
            self.variant in {"din_psrg", "din_psrg_pcrg", "din_psrg_pcrg_transformer"}
            and bool(self.psrg_cfg.get("enabled", True))
        )
        self.pcrg_enabled = (
            self.variant in {"din_psrg_pcrg", "din_psrg_pcrg_transformer"}
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

        # ── 9) TransformerFusion（可开关，默认推荐 interest 模式）──
        self.transformer_enabled = bool(
            self.transformer_cfg.get("enabled", self.variant == "din_psrg_pcrg_transformer")
        )
        self.transformer_input = str(self.transformer_cfg.get("fusion_input", "interest")).lower()
        self.transformer_merge_mode = str(self.transformer_cfg.get("fusion_mode", "concat")).lower()
        self.transformer_proj_after_concat = bool(self.transformer_cfg.get("proj_after_concat", True))
        self.transformer_output_dim = int(self.transformer_cfg.get("output_dim", self.item_repr_dim))
        self.transformer_use_target_attention = bool(self.transformer_cfg.get("use_target_attention", True))
        self.transformer_fusion: Optional[TransformerFusion] = None
        self.transformer_concat_proj: Optional[nn.Linear] = None

        if self.transformer_enabled:
            if self.transformer_input not in {"sequence", "interest"}:
                raise ValueError(
                    f"transformer_fusion.fusion_input={self.transformer_input} 非法，仅支持 sequence / interest"
                )
            if self.transformer_merge_mode not in {"replace", "concat", "residual_add"}:
                raise ValueError(
                    "transformer_fusion.fusion_mode 非法，仅支持 replace / concat / residual_add"
                )
            if self.transformer_input == "interest" and not self.pcrg_enabled:
                raise ValueError("fusion_input=interest 依赖 PCRG 产生多兴趣 token，请启用 pcrg")

            base_interest_dim = self._base_interest_output_dim()
            if self.transformer_merge_mode == "residual_add" and base_interest_dim != self.transformer_output_dim:
                raise ValueError(
                    "transformer_fusion.fusion_mode=residual_add 要求 original_interest 与 u_fused 维度一致，"
                    f"当前 {base_interest_dim} vs {self.transformer_output_dim}"
                )

            self.transformer_fusion = TransformerFusion(
                input_dim=self.item_repr_dim,
                query_dim=self.item_repr_dim,
                d_model=int(self.transformer_cfg.get("d_model", self.item_repr_dim)),
                output_dim=self.transformer_output_dim,
                n_layers=int(self.transformer_cfg.get("n_layers", 1)),
                n_heads=int(self.transformer_cfg.get("n_heads", 2)),
                dropout=float(self.transformer_cfg.get("dropout", 0.1)),
                target_att_hidden_units=list(self.transformer_cfg.get("target_att_hidden_units", [64, 32])),
                target_att_dropout=float(self.transformer_cfg.get("target_att_dropout", 0.1)),
                ffn_hidden=int(self.transformer_cfg.get("ffn_hidden", 256)),
                activation=str(self.transformer_cfg.get("activation", "gelu")),
                use_layernorm=bool(self.transformer_cfg.get("layernorm", True)),
                use_target_attention=self.transformer_use_target_attention,
            )
            if self.transformer_merge_mode == "concat" and self.transformer_proj_after_concat:
                self.transformer_concat_proj = nn.Linear(
                    base_interest_dim + self.transformer_output_dim,
                    base_interest_dim,
                )

        # ── 10) 统一 head 输入向量 x 的子块切片 ──
        # 这里显式维护各子块在 x 中的范围，便于：
        # 1) MBCNet 按语义字段分组交叉；
        # 2) debug 时快速定位维度错误；
        # 3) 保持上游特征接口不变，仅替换最后 head。
        self.feature_block_dims = self._calc_feature_block_dims()
        self.feature_slices = build_feature_slices(self.feature_block_dims)
        self.head_input_dim = sum(self.feature_block_dims.values())
        if self.ppnet_enabled:
            self.personal_context_encoder = self._build_personal_context_encoder()

        # ── 11) 最终 head：mlp（基线）或 mbcnet（三分支）──
        self.dnn: Optional[nn.Module] = None
        self.mbcnet_head: Optional[MBCNetHead] = None
        if self.head_type == "mlp":
            self.dnn = self._build_mlp_head(mcfg, self.head_input_dim)
            self.head: nn.Module = self.dnn
        else:
            self.mbcnet_head = MBCNetHead(
                input_dim=self.head_input_dim,
                config=self.mbcnet_cfg,
                feature_slices=self.feature_slices,
            )
            self.head = self.mbcnet_head

        if self.ppnet_enabled:
            if self.head_type != "mbcnet" and str(self.ppnet_cfg.get("apply_to", "head_input")).lower() in {
                "mbcnet_branches",
                "both",
            }:
                raise ValueError("PPNet branch gate 仅支持 head.type=mbcnet，请检查 ppnet.apply_to 配置。")
            self.ppnet = PPNet(
                context_dim=self.personal_context_encoder.output_dim if self.personal_context_encoder is not None else 0,
                input_dim=self.head_input_dim,
                config=self.ppnet_cfg,
                feature_slices=self.feature_slices,
                branch_names=self.mbcnet_head.enabled_branch_names if self.mbcnet_head is not None else None,
            )

        logger.info(
            "模型变体: variant=%s | psrg=%s | pcrg=%s | fusion=%s | transformer=%s",
            self.variant,
            self.psrg_enabled,
            self.pcrg_enabled,
            self.fusion_mode if self.pcrg_enabled else "none",
            self.transformer_enabled,
        )
        if self.transformer_enabled:
            logger.info(
                "TransformerFusion 配置: input=%s | layers=%d | heads=%d | fusion_mode=%s | target_att=%s",
                self.transformer_input,
                int(self.transformer_cfg.get("n_layers", 1)),
                int(self.transformer_cfg.get("n_heads", 2)),
                self.transformer_merge_mode,
                self.transformer_use_target_attention,
            )
        logger.info(
            "Head 配置: type=%s | input_dim=%d | slices=%s",
            self.head_type,
            self.head_input_dim,
            self.feature_slices,
        )
        if self.head_type == "mbcnet" and self.mbcnet_head is not None:
            logger.info(
                "MBCNet 配置: branches=%s | fusion=%s | groups=%s",
                self.mbcnet_head.enabled_branch_names,
                self.mbcnet_head.fusion_mode,
                [name for name, _ in self.mbcnet_head.group_slices],
            )
        if self.ppnet_enabled and self.personal_context_encoder is not None and self.ppnet is not None:
            logger.info(
                "PPNet 配置: mode=%s | apply_to=%s | p_ctx_dim=%d | group_film=%s | branch_gate=%s",
                self.ppnet.mode,
                self.ppnet.apply_to,
                self.personal_context_encoder.output_dim,
                self.ppnet.group_film.group_names if self.ppnet.group_film is not None else [],
                self.ppnet.branch_gate.branch_names if self.ppnet.branch_gate is not None else [],
            )

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

    def _ensure_finite(self, tensor: torch.Tensor, name: str):
        """模型内部的有限值检查，便于在调制阶段更早暴露异常。"""
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"检测到 NaN/Inf！位置: {name}")

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

    def _build_personal_context_encoder(self) -> PersonalContextEncoder:
        """
        初始化 p_ctx 编码器。

        p_ctx = 场景信息 + 用户属性 + 活跃度代理
        这样 PPNet 学到的是“在什么场景、对什么人、在什么活跃度状态下如何调制”，
        而不是盲目地为每个样本生成重型参数。
        """
        self.ppnet_use_time_features = bool(self.ppnet_context_cfg.get("use_time_features", True))
        self.ppnet_use_user_dense = bool(self.ppnet_context_cfg.get("use_user_dense", True))
        self.ppnet_use_hist_len = bool(self.ppnet_context_cfg.get("use_hist_len", True))
        self.ppnet_use_user_active_proxy = bool(self.ppnet_context_cfg.get("use_user_active_proxy", True))

        self.ppnet_context_cols = ["tab"]
        if self.ppnet_use_time_features:
            for col_name in ["hour_of_day", "day_of_week", "is_weekend"]:
                if col_name in self.context_sparse_cols and col_name not in self.ppnet_context_cols:
                    self.ppnet_context_cols.append(col_name)

        context_input_dim = 0
        for col_name in self.ppnet_context_cols:
            if col_name in self.sparse_embeddings:
                context_input_dim += self.sparse_embeddings[col_name].embedding_dim

        user_sparse_input_dim = 0
        for col_name in self.user_sparse_cols:
            if col_name in self.sparse_embeddings:
                user_sparse_input_dim += self.sparse_embeddings[col_name].embedding_dim
        user_dense_dim = len(self.user_dense_cols) if self.ppnet_use_user_dense else 0

        self.ppnet_activity_dense_dim = 0
        if self.ppnet_use_hist_len:
            self.ppnet_activity_dense_dim += 1
        if self.ppnet_use_user_active_proxy and "is_lowactive_period" in self.user_sparse_cols:
            self.ppnet_activity_dense_dim += 1

        self.ppnet_activity_sparse_dim = 0
        if self.ppnet_use_user_active_proxy and "user_active_degree" in self.sparse_embeddings:
            self.ppnet_activity_sparse_dim = self.sparse_embeddings["user_active_degree"].embedding_dim

        if context_input_dim <= 0:
            raise ValueError("PPNet 需要场景信息（至少 tab embedding），当前 context_input_dim=0。")
        if (user_sparse_input_dim + user_dense_dim) <= 0:
            raise ValueError("PPNet 需要用户属性输入，当前 user sparse + dense 维度都为 0。")
        if (self.ppnet_activity_dense_dim + self.ppnet_activity_sparse_dim) <= 0:
            raise ValueError("PPNet 需要活跃度代理输入，当前 activity 维度为 0。")

        return PersonalContextEncoder(
            context_input_dim=context_input_dim,
            user_sparse_input_dim=user_sparse_input_dim,
            user_dense_dim=user_dense_dim,
            activity_dense_dim=self.ppnet_activity_dense_dim,
            activity_sparse_dim=self.ppnet_activity_sparse_dim,
            config=self.ppnet_context_cfg,
        )

    def _base_interest_output_dim(self) -> int:
        if self.pcrg_enabled and self.fusion_mode == "concat" and not self.proj_after_concat:
            return 2 * self.item_repr_dim
        return self.item_repr_dim

    def _interest_output_dim(self) -> int:
        base_dim = self._base_interest_output_dim()
        if not self.transformer_enabled:
            return base_dim

        if self.transformer_merge_mode == "replace":
            return self.transformer_output_dim

        if self.transformer_merge_mode == "concat":
            if self.transformer_proj_after_concat:
                return base_dim
            return base_dim + self.transformer_output_dim

        if self.transformer_merge_mode == "residual_add":
            return base_dim

        raise ValueError(f"不支持的 transformer_fusion.fusion_mode={self.transformer_merge_mode}")

    def _calc_feature_block_dims(self) -> Dict[str, int]:
        """计算最终 head 输入 x 的各语义子块维度。"""

        def _sum_sparse_dims(cols: List[str]) -> int:
            total = 0
            for col_name in cols:
                if col_name in self.sparse_embeddings:
                    total += self.sparse_embeddings[col_name].embedding_dim
            return total

        # 默认按 interest/item/user/context/candidate_side 语义分组。
        return {
            "user_interest": self._interest_output_dim(),
            "cand_repr": self.item_repr_dim,
            "user_profile_sparse_embs": _sum_sparse_dims(self.user_sparse_cols),
            "user_dense": len(self.user_dense_cols),
            "context_embs": _sum_sparse_dims(self.context_sparse_cols),
            "candidate_side_embs": _sum_sparse_dims(self.cand_side_cols),
        }

    def _build_mlp_head(self, mcfg: Dict[str, Any], input_dim: int) -> nn.Module:
        """保留原 baseline MLP head，用于与 MBCNet 做配置级 ablation。"""
        dnn_hidden_units = list(mcfg.get("dnn_hidden_units", [256, 128, 64]))
        dnn_dropout = float(mcfg.get("dnn_dropout", 0.1))
        dnn_use_bn = bool(mcfg.get("dnn_use_bn", True))

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in dnn_hidden_units:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if dnn_use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dnn_dropout > 0:
                layers.append(nn.Dropout(dnn_dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def _collect_sparse_block(
        self,
        batch: Dict[str, torch.Tensor],
        cols: List[str],
        ref: torch.Tensor,
        block_name: str,
    ) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        for col_name in cols:
            if col_name not in self.sparse_embeddings:
                continue

            emb = self.sparse_embeddings[col_name]
            if col_name in batch:
                parts.append(emb(batch[col_name]))
            else:
                self._warn_once(
                    f"missing_head_col_{block_name}_{col_name}",
                    f"head 输入构造缺少字段 '{col_name}'，将用 0 向量占位。",
                )
                parts.append(emb.weight.new_zeros((ref.shape[0], emb.embedding_dim)))

        if not parts:
            return ref.new_zeros((ref.shape[0], 0))
        return torch.cat(parts, dim=-1)

    def _get_user_dense_fixed(self, batch: Dict[str, torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
        """固定 user_dense 维度，保证 head 输入稳定。"""
        target_dim = int(self.feature_block_dims.get("user_dense", 0))
        if target_dim <= 0:
            return ref.new_zeros((ref.shape[0], 0))

        dense = self._get_user_dense(batch)
        if dense is None:
            self._warn_once(
                "missing_user_dense_for_head",
                "head 输入构造缺少 user_dense，将用 0 向量占位。",
            )
            return ref.new_zeros((ref.shape[0], target_dim))

        if dense.shape[0] != ref.shape[0]:
            raise ValueError(
                f"user_dense batch size 与主干不一致: {dense.shape[0]} vs {ref.shape[0]}"
            )

        cur_dim = dense.shape[-1]
        if cur_dim == target_dim:
            return dense
        if cur_dim > target_dim:
            self._warn_once(
                "user_dense_truncate_for_head",
                f"user_dense 维度({cur_dim}) > 预期({target_dim})，将截断到预期维度。",
            )
            return dense[:, :target_dim]

        self._warn_once(
            "user_dense_pad_for_head",
            f"user_dense 维度({cur_dim}) < 预期({target_dim})，将右侧补 0。",
        )
        pad = ref.new_zeros((ref.shape[0], target_dim - cur_dim))
        return torch.cat([dense, pad], dim=-1)

    def _build_activity_dense_feats(self, batch: Dict[str, torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
        """构造轻量活跃度代理特征。"""
        target_dim = int(self.ppnet_activity_dense_dim)
        if target_dim <= 0:
            return ref.new_zeros((ref.shape[0], 0))

        parts: List[torch.Tensor] = []
        if getattr(self, "ppnet_use_hist_len", False):
            if "hist_len" in batch:
                hist_len = batch["hist_len"].float().view(-1, 1)
                hist_len = torch.clamp(hist_len / max(self.max_hist_len, 1), 0.0, 1.0)
                parts.append(hist_len)
            else:
                self._warn_once("ppnet_missing_hist_len", "PPNet 活跃度代理缺少 hist_len，将用 0 占位。")
                parts.append(ref.new_zeros((ref.shape[0], 1)))

        if getattr(self, "ppnet_use_user_active_proxy", False) and "is_lowactive_period" in self.user_sparse_cols:
            if "is_lowactive_period" in batch:
                lowactive = batch["is_lowactive_period"].float()
                if lowactive.ndim == 1:
                    lowactive = lowactive.unsqueeze(1)
                else:
                    lowactive = lowactive[:, :1]
                parts.append(lowactive)
            else:
                self._warn_once(
                    "ppnet_missing_is_lowactive_period",
                    "PPNet 活跃度代理缺少 is_lowactive_period，将用 0 占位。",
                )
                parts.append(ref.new_zeros((ref.shape[0], 1)))

        if not parts:
            return ref.new_zeros((ref.shape[0], target_dim))

        out = torch.cat(parts, dim=-1)
        if out.shape[-1] == target_dim:
            return out
        if out.shape[-1] > target_dim:
            return out[:, :target_dim]
        pad = ref.new_zeros((ref.shape[0], target_dim - out.shape[-1]))
        return torch.cat([out, pad], dim=-1)

    def _build_activity_sparse_embs(self, batch: Dict[str, torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
        """构造活跃度相关的 sparse embedding 代理。"""
        target_dim = int(self.ppnet_activity_sparse_dim)
        if target_dim <= 0:
            return ref.new_zeros((ref.shape[0], 0))

        col_name = "user_active_degree"
        if col_name in batch and col_name in self.sparse_embeddings:
            return self.sparse_embeddings[col_name](batch[col_name])

        self._warn_once(
            "ppnet_missing_user_active_degree",
            "PPNet 活跃度代理缺少 user_active_degree embedding，将用 0 占位。",
        )
        return ref.new_zeros((ref.shape[0], target_dim))

    def _build_personal_context(
        self,
        batch: Dict[str, torch.Tensor],
        ref: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """构造 PPNet 使用的 p_ctx。"""
        if self.personal_context_encoder is None:
            raise RuntimeError("PPNet 已启用，但 personal_context_encoder 尚未初始化。")

        context_embs = self._collect_sparse_block(
            batch=batch,
            cols=self.ppnet_context_cols,
            ref=ref,
            block_name="ppnet_context_embs",
        )
        user_sparse_embs = self._collect_sparse_block(
            batch=batch,
            cols=self.user_sparse_cols,
            ref=ref,
            block_name="ppnet_user_sparse_embs",
        )
        user_dense = (
            self._get_user_dense_fixed(batch, ref)
            if getattr(self, "ppnet_use_user_dense", False)
            else ref.new_zeros((ref.shape[0], 0))
        )
        activity_dense_feats = self._build_activity_dense_feats(batch, ref)
        activity_sparse_embs = self._build_activity_sparse_embs(batch, ref)

        p_ctx, stats = self.personal_context_encoder(
            context_embs=context_embs,
            user_sparse_embs=user_sparse_embs,
            user_dense=user_dense,
            activity_dense_feats=activity_dense_feats,
            activity_sparse_embs=activity_sparse_embs,
            return_debug=True,
        )
        return p_ctx, stats

    def _build_head_input(
        self,
        batch: Dict[str, torch.Tensor],
        user_interest: torch.Tensor,
        cand_item_repr: torch.Tensor,
    ) -> torch.Tensor:
        """统一拼接最终 flat feature vector x: [B, D_in]。"""
        user_sparse = self._collect_sparse_block(
            batch=batch,
            cols=self.user_sparse_cols,
            ref=cand_item_repr,
            block_name="user_profile_sparse_embs",
        )
        user_dense = self._get_user_dense_fixed(batch, cand_item_repr)
        context_embs = self._collect_sparse_block(
            batch=batch,
            cols=self.context_sparse_cols,
            ref=cand_item_repr,
            block_name="context_embs",
        )
        cand_side_embs = self._collect_sparse_block(
            batch=batch,
            cols=self.cand_side_cols,
            ref=cand_item_repr,
            block_name="candidate_side_embs",
        )

        parts = [user_interest, cand_item_repr, user_sparse, user_dense, context_embs, cand_side_embs]
        x = torch.cat(parts, dim=-1)
        if x.shape[-1] != self.head_input_dim:
            raise RuntimeError(
                f"head 输入维度不匹配: 实际={x.shape[-1]} 预期={self.head_input_dim} | slices={self.feature_slices}"
            )
        return x

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

    def _fuse_transformer_interest(self, original_interest: torch.Tensor, u_fused: torch.Tensor) -> torch.Tensor:
        """将 TransformerFusion 输出与原 user_interest 再做一次融合。"""
        if self.transformer_merge_mode == "replace":
            return u_fused

        if self.transformer_merge_mode == "concat":
            fused = torch.cat([original_interest, u_fused], dim=-1)
            if self.transformer_concat_proj is not None:
                fused = self.transformer_concat_proj(fused)
            return fused

        if self.transformer_merge_mode == "residual_add":
            if original_interest.shape[-1] != u_fused.shape[-1]:
                raise ValueError(
                    "transformer_fusion.fusion_mode=residual_add 要求维度一致，"
                    f"当前 {original_interest.shape[-1]} vs {u_fused.shape[-1]}"
                )
            return original_interest + u_fused

        raise ValueError(f"不支持的 transformer_fusion.fusion_mode={self.transformer_merge_mode}")

    # ─────────────────────────────────────────────────────────
    # forward
    # ─────────────────────────────────────────────────────────

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self._validate_hist_shape(batch)

        hist_mask = batch["hist_mask"].float()                       # [B, L]
        cand_item_repr = self._build_cand_item_repr(batch)            # [B, D]
        hist_item_repr = self._build_hist_item_repr(batch)            # [B, L, D]
        self._ensure_finite(cand_item_repr, "cand_item_repr")
        self._ensure_finite(hist_item_repr, "hist_item_repr")

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

        # ── 路径 D: TransformerFusion（在现有兴趣表示之上做进一步增强）──
        transformer_aux: Dict[str, Any] = {}
        if self.transformer_enabled and self.transformer_fusion is not None:
            fusion_tokens = hist_for_attn
            fusion_mask = hist_mask

            if self.transformer_input == "interest":
                fusion_tokens = pcrg_aux.get("interest_tokens")
                fusion_mask = pcrg_aux.get("interest_mask")
                if fusion_tokens is None:
                    raise RuntimeError("TransformerFusion interest 模式需要 PCRG 提供 interest_tokens")

            u_fused, transformer_aux = self.transformer_fusion(
                query=cand_item_repr,
                tokens=fusion_tokens,
                token_mask=fusion_mask,
                return_debug=True,
            )
            user_interest = self._fuse_transformer_interest(user_interest, u_fused)

        p_ctx: Optional[torch.Tensor] = None
        personal_ctx_stats: Dict[str, Any] = {}
        ppnet_stats: Dict[str, Any] = {}
        branch_gate: Optional[torch.Tensor] = None
        if self.ppnet_enabled and self.ppnet is not None:
            p_ctx, personal_ctx_stats = self._build_personal_context(batch=batch, ref=cand_item_repr)
            self._ensure_finite(p_ctx, "p_ctx")

        # ── 统一 flat 向量 x，再送入可切换 head（MLP / MBCNet）──
        head_input = self._build_head_input(
            batch=batch,
            user_interest=user_interest,
            cand_item_repr=cand_item_repr,
        )
        self._ensure_finite(head_input, "head_input_before_ppnet")

        if self.ppnet_enabled and self.ppnet is not None and self.ppnet.apply_to in {"head_input", "both"}:
            head_input, head_ppnet_stats = self.ppnet.modulate_head_input(
                x=head_input,
                p_ctx=p_ctx,
                return_debug=True,
            )
            ppnet_stats.update(head_ppnet_stats)
            self._ensure_finite(head_input, "head_input_after_ppnet")

        if self.ppnet_enabled and self.ppnet is not None and self.ppnet.apply_to in {"mbcnet_branches", "both"}:
            branch_gate, branch_ppnet_stats = self.ppnet.build_branch_gate(
                p_ctx=p_ctx,
                return_debug=True,
            )
            ppnet_stats.update(branch_ppnet_stats)
            self._ensure_finite(branch_gate, "ppnet_branch_gate")

        if self.head_type == "mbcnet" and self.mbcnet_head is not None:
            logits = self.mbcnet_head(head_input, branch_gate=branch_gate)
        else:
            logits = self.head(head_input)
        if logits.ndim > 1:
            logits = logits.squeeze(-1)
        self._ensure_finite(logits, "logits")

        # 记录调试统计，供 trainer 可选打印
        self._last_debug_stats = {
            "variant": self.variant,
            "head_type": self.head_type,
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
            "transformer_token_mean": float(transformer_aux.get("token_mean", 0.0)) if transformer_aux else 0.0,
            "transformer_token_var": float(transformer_aux.get("token_var", 0.0)) if transformer_aux else 0.0,
            "transformer_attn_entropy_mean": float(
                transformer_aux.get("target_attn_entropy_mean", 0.0)
            ) if transformer_aux else 0.0,
            "transformer_all_pad_count": int(
                transformer_aux.get("target_all_pad_count", 0)
            ) if transformer_aux else 0,
            "ppnet_enabled": self.ppnet_enabled,
        }
        self._last_debug_stats.update(personal_ctx_stats)
        self._last_debug_stats.update(ppnet_stats)
        if self.head_type == "mbcnet" and self.mbcnet_head is not None:
            self._last_debug_stats.update(self.mbcnet_head.last_debug_stats)

        # debug 模式下仅首个 batch 打印关键 shape
        if bool(self.debug_cfg.get("print_shapes_once", False)) and (not self._printed_shape_once):
            self._printed_shape_once = True
            logger.info(
                "[Debug Shapes] variant=%s | head=%s | cand_item=%s | hist_item=%s | "
                "user_interest=%s | p_ctx=%s | x=%s | branch_gate=%s | logits=%s | slices=%s",
                self.variant,
                self.head_type,
                list(cand_item_repr.shape),
                list(hist_for_attn.shape),
                list(user_interest.shape),
                list(p_ctx.shape) if p_ctx is not None else [],
                list(head_input.shape),
                list(branch_gate.shape) if branch_gate is not None else [],
                list(logits.shape),
                self.feature_slices,
            )

        return logits

"""
Deep Interest Network (DIN) 模型实现。

核心思想（Alibaba, KDD 2018）
─────────────────────────────
传统 CTR 模型对用户历史做 sum/mean pooling，无法区分"不同候选 item 应该关注用户的哪些历史"。
DIN 引入 attention 机制：以当前候选 item 作为 query，对用户历史序列做加权求和，
得到"自适应用户兴趣表示"—— 不同候选激活不同历史。

模型结构
────────
1. Embedding 层
   - video_id / author_id：历史序列与候选 item **共享**同一张 embedding 表。
     为什么共享？共享确保候选 item 与历史 item 在同一语义空间，
     这样 attention 的 query 和 key 是直接可比的，否则 dot-product / MLP
     计算出的相关性没有意义。
   - 其余 sparse 字段（context / user / candidate side）：各自独立 embedding。
   - dense 字段：直接作为连续浮点输入。

2. DIN Attention
   - query = concat(cand_video_emb, cand_author_emb)
   - keys  = concat(hist_video_embs, hist_author_embs)   [B, L, D]
   - 注意力打分：DIN 经典拼接：concat([q, k, q-k, q*k]) → MLP → scalar
   - Mask：padding 位必须 **完全不参与** softmax（设为 -inf → softmax 后为 0）
   - 用户兴趣 = keys 的加权求和

3. DNN 塔
   - 输入 = concat([user_interest, cand_repr, cand_side_embs, ctx_embs, user_sparse_embs, user_dense])
   - 多层 FC + BN + ReLU + Dropout
   - 输出 1 维 logit（配合 BCEWithLogitsLoss）

显存优化
────────
- 支持 HashEmbedding：把 32M video_id 映射到 100 万桶，显存从 ~4GB 降到 ~128MB。
- 所有维度可在 YAML 中配置。
"""

import logging
from typing import Any, Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 辅助：列名 → vocab 名称的映射
# ─────────────────────────────────────────────────────────────

def col_to_vocab_name(col_name: str) -> str:
    """
    从 parquet 列名推导 vocab 名称。
    规则：去掉 'cand_' 前缀（如果有），其余保持原名。
    例：cand_video_type → video_type, user_active_degree → user_active_degree
    """
    if col_name.startswith("cand_"):
        return col_name[len("cand_"):]
    return col_name


# ─────────────────────────────────────────────────────────────
# HashEmbedding
# ─────────────────────────────────────────────────────────────

class HashEmbedding(nn.Module):
    """
    哈希 Embedding：将超大词表通过取模映射到较小的桶中，大幅降低显存。

    例如 video_id 原始 vocab = 32,038,727：
      全量 embedding → 32M × 32 × 4B ≈ 4GB（超出 4GB 显存）
      HashEmbedding(1M, 32)  → 1M × 32 × 4B ≈ 128MB ✓

    Padding 处理：
      - idx = 0 被保留作为 padding，始终映射到 embedding[0]（零向量）。
      - idx > 0 → hash 到 [1, num_buckets - 1] 范围，避免与 padding 冲突。
    """

    def __init__(self, num_buckets: int, emb_dim: int, padding_idx: int = 0):
        super().__init__()
        self.num_buckets = num_buckets
        self.emb = nn.Embedding(num_buckets, emb_dim, padding_idx=padding_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LongTensor，任意 shape（[B] / [B, L] 均可）
        Returns:
            Embedding tensor，shape = x.shape + (emb_dim,)
        """
        # 保留 0 作为 padding（不哈希），非零索引哈希到 [1, num_buckets)
        is_pad = (x == 0)
        hashed = (x % (self.num_buckets - 1)) + 1   # 映射到 [1, num_buckets-1]
        hashed = hashed.masked_fill(is_pad, 0)       # padding 位还原为 0
        return self.emb(hashed)


# ─────────────────────────────────────────────────────────────
# DIN Attention Layer
# ─────────────────────────────────────────────────────────────

class DINAttention(nn.Module):
    """
    DIN 注意力层。

    DIN 的核心创新：
      用候选 item 作为 query，与用户历史序列中的 item（keys）计算注意力权重，
      得到"自适应用户兴趣表示"。不同候选 item 会激活不同的用户历史行为。

    注意力打分方式（经典 DIN 形式）：
      att_input = concat([query, key, query - key, query * key])   维度 = 4D
      score = MLP(att_input) → 标量

    Mask 处理（关键）：
      hist_mask 标记哪些位置是真实历史（1）、哪些是 padding（0）。
      padding 位的 attention score 设为一个极小值（-1e4），确保 softmax 后权重趋近 0，
      不会影响最终的加权求和结果。使用 -1e4 而非 -inf 是为了兼容 fp16 精度。

    Args:
        item_dim: 单个 item 表示的维度（query / key 的维度 D）
        hidden_units: attention MLP 的隐藏层维度列表
    """

    def __init__(self, item_dim: int, hidden_units: List[int]):
        super().__init__()
        # 输入 = [q, k, q-k, q*k] → 4 × D
        input_dim = 4 * item_dim
        layers: list = []
        prev_dim = input_dim
        for h in hidden_units:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.PReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: [B, D] 候选 item 表示
            keys:  [B, L, D] 历史 item 表示序列
            mask:  [B, L] 1=有效, 0=padding

        Returns:
            user_interest: [B, D] 加权求和得到的用户兴趣向量
        """
        B, L, D = keys.shape

        # 扩展 query 到 [B, L, D] 以便与 keys 逐位置比较
        q = query.unsqueeze(1).expand(-1, L, -1)  # [B, L, D]

        # DIN 经典拼接
        att_input = torch.cat([q, keys, q - keys, q * keys], dim=-1)  # [B, L, 4D]

        # MLP 打分
        att_scores = self.mlp(att_input).squeeze(-1)  # [B, L]

        # ── Mask 处理 ──
        # padding 位设为极大负值，softmax 后权重为 ~0
        # 使用 -1e4 而非 float('-inf')：兼容 fp16 精度
        att_scores = att_scores.masked_fill(mask == 0, -1e4)

        # Softmax 得到注意力权重
        att_weights = torch.softmax(att_scores, dim=-1)  # [B, L]

        # 极端情况：如果某行的 mask 全为 0（完全没有历史），
        # softmax(-1e4, -1e4, ...) ≈ 均匀分布而非 nan，但加权求和的结果
        # 实际上是 padding embedding 的均值（接近零向量），语义上合理。
        # 为安全起见，把全 padding 行的权重清零
        att_weights = att_weights.masked_fill(mask == 0, 0.0)

        # 加权求和：[B, 1, L] × [B, L, D] → [B, 1, D] → [B, D]
        user_interest = torch.bmm(att_weights.unsqueeze(1), keys).squeeze(1)

        return user_interest


# ─────────────────────────────────────────────────────────────
# DIN 模型
# ─────────────────────────────────────────────────────────────

class DINModel(nn.Module):
    """
    Deep Interest Network 完整模型。

    Args:
        config: 完整配置字典（包含 model / fields 等节）
        vocab_sizes: {vocab_name: int} 从 field_schema.json 读取
    """

    def __init__(self, config: dict, vocab_sizes: Dict[str, int]):
        super().__init__()

        mcfg = config["model"]
        fcfg = config["fields"]

        self.video_emb_dim = mcfg["video_id_emb_dim"]
        self.author_emb_dim = mcfg["author_id_emb_dim"]

        # ╔══════════════════════════════════════════════════════╗
        # ║ 1. 共享 Embedding（video_id / author_id）           ║
        # ║    hist_video_id 与 cand_video_id 使用同一张表      ║
        # ║    hist_author_id 与 cand_author_id 使用同一张表    ║
        # ╚══════════════════════════════════════════════════════╝
        if mcfg.get("use_hash_embedding", True):
            self.video_id_emb = HashEmbedding(
                mcfg["video_id_hash_buckets"], self.video_emb_dim
            )
            self.author_id_emb = HashEmbedding(
                mcfg["author_id_hash_buckets"], self.author_emb_dim
            )
            logger.info(
                "使用 HashEmbedding: video=%d buckets (dim=%d), author=%d buckets (dim=%d)",
                mcfg["video_id_hash_buckets"], self.video_emb_dim,
                mcfg["author_id_hash_buckets"], self.author_emb_dim,
            )
        else:
            self.video_id_emb = nn.Embedding(
                vocab_sizes["video_id"], self.video_emb_dim, padding_idx=0
            )
            self.author_id_emb = nn.Embedding(
                vocab_sizes["author_id"], self.author_emb_dim, padding_idx=0
            )
            logger.info(
                "使用全量 Embedding: video=%d (dim=%d), author=%d (dim=%d)",
                vocab_sizes["video_id"], self.video_emb_dim,
                vocab_sizes["author_id"], self.author_emb_dim,
            )

        # ╔══════════════════════════════════════════════════════╗
        # ║ 2. 其余 sparse 字段的独立 Embedding                ║
        # ╚══════════════════════════════════════════════════════╝
        self.sparse_embeddings = nn.ModuleDict()

        # 收集需要 embedding 的 sparse 列（排除 video_id / author_id）
        self.cand_side_cols: List[str] = []
        for feat_name, col_name in fcfg["cand_cols"].items():
            if feat_name in ("video_id", "author_id"):
                continue
            self.cand_side_cols.append(col_name)
            self._register_sparse_emb(col_name, vocab_sizes, mcfg)

        self.context_sparse_cols: List[str] = list(fcfg["context_sparse_cols"])
        for col_name in self.context_sparse_cols:
            self._register_sparse_emb(col_name, vocab_sizes, mcfg)

        self.user_sparse_cols: List[str] = list(fcfg["user_sparse_cols"])
        for col_name in self.user_sparse_cols:
            self._register_sparse_emb(col_name, vocab_sizes, mcfg)

        self.user_dense_cols: List[str] = list(fcfg["user_dense_cols"])

        # ╔══════════════════════════════════════════════════════╗
        # ║ 3. DIN Attention                                     ║
        # ╚══════════════════════════════════════════════════════╝
        item_repr_dim = self.video_emb_dim + self.author_emb_dim
        self.din_attention = DINAttention(
            item_dim=item_repr_dim,
            hidden_units=mcfg["att_hidden_units"],
        )

        # ╔══════════════════════════════════════════════════════╗
        # ║ 4. DNN 塔                                           ║
        # ╚══════════════════════════════════════════════════════╝
        dnn_input_dim = self._calc_dnn_input_dim(fcfg, mcfg, vocab_sizes)
        logger.info("DNN 输入维度: %d", dnn_input_dim)

        dnn_layers: list = []
        prev_dim = dnn_input_dim
        for hidden_dim in mcfg["dnn_hidden_units"]:
            dnn_layers.append(nn.Linear(prev_dim, hidden_dim))
            if mcfg.get("dnn_use_bn", True):
                dnn_layers.append(nn.BatchNorm1d(hidden_dim))
            dnn_layers.append(nn.ReLU())
            if mcfg["dnn_dropout"] > 0:
                dnn_layers.append(nn.Dropout(mcfg["dnn_dropout"]))
            prev_dim = hidden_dim
        dnn_layers.append(nn.Linear(prev_dim, 1))
        self.dnn = nn.Sequential(*dnn_layers)

        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("模型参数量: total=%s, trainable=%s", f"{total_params:,}", f"{trainable_params:,}")

    # ─────────── 私有方法 ───────────

    def _get_emb_dim(self, vocab_name: str, vocab_sizes: Dict[str, int], mcfg: dict) -> int:
        """根据 vocab 大小决定 embedding 维度。"""
        vs = vocab_sizes.get(vocab_name, 1)
        threshold = mcfg.get("large_sparse_threshold", 100)
        if vs > threshold:
            return mcfg.get("large_sparse_emb_dim", 8)
        return mcfg.get("sparse_emb_dim", 4)

    def _register_sparse_emb(self, col_name: str, vocab_sizes: Dict[str, int], mcfg: dict):
        """为一个 sparse 列注册 Embedding 到 ModuleDict。"""
        vocab_name = col_to_vocab_name(col_name)
        vs = vocab_sizes.get(vocab_name)
        if vs is None:
            logger.warning("找不到列 '%s' 对应的 vocab (vocab_name='%s')，跳过", col_name, vocab_name)
            return
        emb_dim = self._get_emb_dim(vocab_name, vocab_sizes, mcfg)
        self.sparse_embeddings[col_name] = nn.Embedding(vs, emb_dim, padding_idx=0)

    def _calc_dnn_input_dim(self, fcfg: dict, mcfg: dict, vocab_sizes: Dict[str, int]) -> int:
        """计算 DNN 塔的输入维度（所有特征拼接的总维度）。"""
        dim = 0
        # user_interest: item_repr_dim
        dim += self.video_emb_dim + self.author_emb_dim
        # cand_item_repr: item_repr_dim
        dim += self.video_emb_dim + self.author_emb_dim
        # cand side features
        for col_name in self.cand_side_cols:
            vn = col_to_vocab_name(col_name)
            dim += self._get_emb_dim(vn, vocab_sizes, mcfg)
        # context sparse
        for col_name in self.context_sparse_cols:
            vn = col_to_vocab_name(col_name)
            dim += self._get_emb_dim(vn, vocab_sizes, mcfg)
        # user sparse
        for col_name in self.user_sparse_cols:
            vn = col_to_vocab_name(col_name)
            dim += self._get_emb_dim(vn, vocab_sizes, mcfg)
        # user dense
        dim += len(fcfg["user_dense_cols"])
        return dim

    # ─────────── forward ───────────

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播。

        Args:
            batch: collate_fn 输出的 tensor 字典

        Returns:
            logits: [B] 未经 sigmoid 的预测值（配合 BCEWithLogitsLoss）
        """
        # ── 1. 候选 item embedding（与历史共享 embedding 表）──
        cand_vid_emb = self.video_id_emb(batch["cand_video_id"])       # [B, D_v]
        cand_aid_emb = self.author_id_emb(batch["cand_author_id"])     # [B, D_a]
        cand_item_repr = torch.cat([cand_vid_emb, cand_aid_emb], dim=-1)  # [B, D_item]

        # ── 2. 历史序列 embedding（共享 embedding 表确保 q/k 在同一语义空间）──
        hist_vid_emb = self.video_id_emb(batch["hist_video_id"])       # [B, L, D_v]
        hist_aid_emb = self.author_id_emb(batch["hist_author_id"])     # [B, L, D_a]
        hist_item_repr = torch.cat([hist_vid_emb, hist_aid_emb], dim=-1)  # [B, L, D_item]

        # ── 3. DIN Attention ──
        hist_mask = batch["hist_mask"].float()  # [B, L]  1=有效 0=padding
        user_interest = self.din_attention(cand_item_repr, hist_item_repr, hist_mask)  # [B, D_item]

        # ── 4. 拼接所有特征 ──
        parts = [user_interest, cand_item_repr]

        # 候选 side features
        for col_name in self.cand_side_cols:
            if col_name in self.sparse_embeddings:
                parts.append(self.sparse_embeddings[col_name](batch[col_name]))

        # 上下文 sparse features
        for col_name in self.context_sparse_cols:
            if col_name in self.sparse_embeddings:
                parts.append(self.sparse_embeddings[col_name](batch[col_name]))

        # 用户 sparse features
        for col_name in self.user_sparse_cols:
            if col_name in self.sparse_embeddings:
                parts.append(self.sparse_embeddings[col_name](batch[col_name]))

        # 用户 dense features
        if "user_dense" in batch:
            parts.append(batch["user_dense"])

        # ── 5. DNN 塔 ──
        dnn_input = torch.cat(parts, dim=-1)  # [B, D_total]
        logit = self.dnn(dnn_input).squeeze(-1)  # [B]

        return logit

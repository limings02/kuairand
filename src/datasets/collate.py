"""
Collate 函数：将 DataLoader 的一批样本字典转换为模型所需的 Tensor 字典。

职责：
  - list<int> 历史序列 → LongTensor [B, L]
  - int 标量列（sparse / label / meta） → LongTensor [B]
  - float 标量列 → FloatTensor [B]
  - 将用户 dense 列合并为单个 FloatTensor [B, D_dense]
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# 首次调用时打印一次 batch 结构，帮助 debug
_PRINTED_BATCH_SCHEMA = False


class DINCollateFn:
    """
    可调用的 Collate 函数对象。

    在 DataLoader 中使用：
        DataLoader(dataset, collate_fn=DINCollateFn(user_dense_cols))

    Args:
        user_dense_cols: 用户 dense 列名列表，collate 时会合并成 [B, D] 的 user_dense tensor
    """

    def __init__(self, user_dense_cols: Optional[List[str]] = None):
        self.user_dense_cols = user_dense_cols or []

    def __call__(self, batch: List[dict]) -> Dict[str, torch.Tensor]:
        """
        将一批样本字典列表转换为 Tensor 字典。

        Args:
            batch: list of dicts，每个 dict 是一个样本（来自 Dataset.__getitem__ 或 __iter__）
                   - 1D np.ndarray 项  → list columns（历史序列）
                   - int 项            → 标量 int 列
                   - float 项          → 标量 float 列

        Returns:
            dict of {col_name: Tensor}，外加 'user_dense': [B, D_dense] FloatTensor
        """
        global _PRINTED_BATCH_SCHEMA

        if not batch:
            return {}

        first = batch[0]
        result: Dict[str, torch.Tensor] = {}

        for key in first:
            vals = [sample[key] for sample in batch]
            sample_val = first[key]

            if isinstance(sample_val, np.ndarray):
                # list<int> 历史序列列 → [B, L] LongTensor
                stacked = np.stack(vals)
                result[key] = torch.from_numpy(stacked).long()

            elif isinstance(sample_val, float):
                # 浮点标量 → FloatTensor
                result[key] = torch.tensor(vals, dtype=torch.float32)

            elif isinstance(sample_val, int):
                # 整型标量（label / sparse / meta）→ LongTensor
                result[key] = torch.tensor(vals, dtype=torch.long)

            else:
                # 其他类型（例如字符串）→ 保持为 Python list
                result[key] = vals

        # ── 合并用户 dense 列为 [B, D_dense] ──
        if self.user_dense_cols:
            dense_parts = []
            for col in self.user_dense_cols:
                if col in result and isinstance(result[col], torch.Tensor):
                    dense_parts.append(result[col].unsqueeze(1))  # [B, 1]
            if dense_parts:
                result["user_dense"] = torch.cat(dense_parts, dim=1)  # [B, D_dense]

        # ── 首次打印 batch schema ──
        if not _PRINTED_BATCH_SCHEMA:
            lines = ["Batch schema（首次打印）:"]
            for k, v in result.items():
                if isinstance(v, torch.Tensor):
                    lines.append(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")
                else:
                    lines.append(f"  {k}: type={type(v).__name__}, len={len(v)}")
            logger.info("\n".join(lines))
            _PRINTED_BATCH_SCHEMA = True

        return result


def reset_collate_print_flag():
    """重置 batch schema 打印标记（换数据集评估时调用）。"""
    global _PRINTED_BATCH_SCHEMA
    _PRINTED_BATCH_SCHEMA = False

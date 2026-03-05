"""
Collate 函数：将预组装的 numpy batch 转换为 PyTorch Tensor 字典。

性能优化说明
───────────
旧版 collate 接收 list[dict]（128 个单样本 dict），需要逐字段 list 合并再转 tensor。
  每个 batch 产生 128 × 50 = 6400 次 Python list append + numpy stack。

新版 collate 接收单个 dict[str, np.ndarray]（预组装好的 batch），
  只需对每个 numpy array 调用一次 torch.from_numpy()（零复制转换）。
  每个 batch 仅 ~50 次 torch.from_numpy() 调用，无 Python 循环。

两个入口：
  - BatchCollateFn: 用于 ParquetIterableDataset（预组装 batch）
  - SampleCollateFn: 用于 DebugMapDataset（逐样本 batch，仅 debug）
"""

import logging
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

# 首次调用时打印一次 batch 结构，帮助 debug
_PRINTED_BATCH_SCHEMA = False


class BatchCollateFn:
    """
    高性能 Collate：接收预组装好的 numpy batch dict → 转为 tensor dict。

    ParquetIterableDataset yield 的已经是 {col: np.ndarray[B, ...]} 形式，
    此 collate 只需做 numpy→tensor 转换（torch.from_numpy 走零复制路径），
    以及合并 user_dense 列。

    与 DataLoader(batch_size=None) 搭配使用。

    Args:
        user_dense_cols: 需要合并为 [B, D] 的 dense 列名列表
        float_columns: float 类型列名集合
    """

    def __init__(
        self,
        user_dense_cols: Optional[List[str]] = None,
        float_columns: Optional[Set[str]] = None,
    ):
        self.user_dense_cols = user_dense_cols or []
        self.float_columns = float_columns or set()

    def __call__(self, batch_np: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        将 numpy batch → tensor batch。

        numpy→torch 转换使用 torch.from_numpy()，对连续数组会走零复制路径，
        对非连续数组（fancy indexing 结果）自动做一次 memcpy，也比 Python 循环快几个数量级。
        """
        global _PRINTED_BATCH_SCHEMA

        result: Dict[str, Any] = {}

        for key, arr in batch_np.items():
            if key in self.float_columns:
                # float32 列直接转
                result[key] = torch.from_numpy(np.ascontiguousarray(arr)).float()
            elif arr.dtype.kind in {"U", "S", "O"}:
                # 字符串/对象列（例如 meta_log_source）保留为 Python list。
                # 这些字段仅用于分析/切片，不参与模型前向。
                result[key] = arr.tolist()
            else:
                # int / list<int> 列 → LongTensor
                result[key] = torch.from_numpy(np.ascontiguousarray(arr)).long()

        # ── 合并用户 dense 列为 [B, D_dense] ──
        if self.user_dense_cols:
            dense_parts = []
            for col in self.user_dense_cols:
                if col in result:
                    t = result[col]
                    if isinstance(t, torch.Tensor) and t.dim() == 1:
                        dense_parts.append(t.unsqueeze(1))
                    elif isinstance(t, torch.Tensor):
                        dense_parts.append(t)
            if dense_parts:
                result["user_dense"] = torch.cat(dense_parts, dim=1)

        # ── 首次打印 batch schema ──
        if not _PRINTED_BATCH_SCHEMA:
            lines = ["Batch schema（首次打印）:"]
            for k, v in result.items():
                if isinstance(v, torch.Tensor):
                    lines.append(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")
            logger.info("\n".join(lines))
            _PRINTED_BATCH_SCHEMA = True

        return result


class SampleCollateFn:
    """
    逐样本 Collate（仅用于 DebugMapDataset）。

    接收 list[dict]（每个 dict 是一个样本），合并为 tensor batch。
    只在 debug_rows > 0 时使用，大规模训练不要用。
    """

    def __init__(self, user_dense_cols: Optional[List[str]] = None):
        self.user_dense_cols = user_dense_cols or []

    def __call__(self, batch: List[dict]) -> Dict[str, Any]:
        if not batch:
            return {}

        first = batch[0]
        result: Dict[str, Any] = {}

        for key in first:
            vals = [sample[key] for sample in batch]
            sample_val = first[key]

            if isinstance(sample_val, np.ndarray):
                result[key] = torch.from_numpy(np.stack(vals)).long()
            elif isinstance(sample_val, float):
                result[key] = torch.tensor(vals, dtype=torch.float32)
            elif isinstance(sample_val, int):
                result[key] = torch.tensor(vals, dtype=torch.long)
            else:
                result[key] = vals

        # 合并 dense
        if self.user_dense_cols:
            dense_parts = []
            for col in self.user_dense_cols:
                if col in result and isinstance(result[col], torch.Tensor):
                    dense_parts.append(result[col].unsqueeze(1))
            if dense_parts:
                result["user_dense"] = torch.cat(dense_parts, dim=1)

        return result


def reset_collate_print_flag():
    """重置 batch schema 打印标记（换数据集评估时调用）。"""
    global _PRINTED_BATCH_SCHEMA
    _PRINTED_BATCH_SCHEMA = False

"""
内存友好的 Parquet 流式数据集（高性能版）。

性能优化核心思路
────────────────
旧版设计：Dataset 逐行 yield 单个样本 dict → DataLoader collate 逐样本拼装 tensor
  瓶颈：130M 行 × 50 列 = 65 亿次 Python dict 操作，训练 1 epoch 需要 ~10 小时

新版设计：Dataset 按 row group 读取后直接用 **numpy 向量化切片** 产出预组装好的 batch
  - 读 row group → {col: np.ndarray[n_rows, ...]}
  - shuffle 行索引（numpy 操作，微秒级）
  - 按 batch_size 切片 → {col: np.ndarray[B, ...]}  ← 全部是 numpy fancy indexing
  - DataLoader 用 batch_size=None，collate 只做 numpy→tensor（零 Python 循环）

预期加速：数据管道从 ~3000 samp/s → 10 万+ samp/s（~30-50 倍）

为什么用 IterableDataset 而非 MapDataset？
───────────────────────────────────────────
训练集 130M 行，全量加载内存需要 ~380GB，IterableDataset 按 row group 逐批读取，
任何时刻只保留 1 个 row group（~186K 行 ≈ 500MB Arrow）。

打乱策略
────────
  1. 每个 epoch 随机排列 row group 顺序
  2. 每个 row group 内部 shuffle 行索引
  3. 按 batch_size 切分后直接 yield

多 Worker 分片
──────────────
  按 row group id 轮询分片：worker_i 处理 rg_id % num_workers == worker_i 的 row group。
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, IterableDataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def _detect_list_columns(schema: pa.Schema) -> Set[str]:
    """从 Arrow schema 中识别所有 list 类型列。"""
    return {field.name for field in schema if pa.types.is_list(field.type)}


def _detect_float_columns(schema: pa.Schema) -> Set[str]:
    """从 Arrow schema 中识别所有浮点列。"""
    return {
        field.name
        for field in schema
        if pa.types.is_floating(field.type) or pa.types.is_decimal(field.type)
    }


def _detect_string_columns(schema: pa.Schema) -> Set[str]:
    """从 Arrow schema 中识别所有字符串列。"""
    return {
        field.name
        for field in schema
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
    }


def resolve_columns(
    parquet_path: str,
    fields_config: dict,
    is_train: bool = True,
) -> List[str]:
    """
    根据配置与 parquet schema 解析实际需要读取的列。

    对于 optional 列（如 hist_delta_t_bucket），若 parquet 中存在则包含，否则跳过并打印日志。
    """
    pf = pq.ParquetFile(parquet_path)
    available = set(pf.schema_arrow.names)

    # 必需列
    columns: List[str] = []
    # 标签
    columns.append(fields_config["label_col"])
    # 历史序列
    columns.extend(fields_config["hist_seq_cols"])
    columns.append(fields_config["hist_mask_col"])
    columns.append(fields_config["hist_len_col"])
    # 候选
    columns.extend(fields_config["cand_cols"].values())
    # 上下文
    columns.extend(fields_config["context_sparse_cols"])
    # 用户 sparse
    columns.extend(fields_config["user_sparse_cols"])
    # 用户 dense
    columns.extend(fields_config["user_dense_cols"])

    # 可选历史序列
    for col in fields_config.get("optional_hist_seq_cols", []):
        if col in available:
            columns.append(col)
        else:
            logger.info("可选列 '%s' 在 parquet 中不存在，已跳过", col)

    # 元数据列（评估时需要）
    if not is_train:
        for col in fields_config.get("meta_cols", []):
            if col in available and col not in columns:
                columns.append(col)

    # 训练时也读 user_id_col 以备评估（开销很小）
    uid_col = fields_config.get("user_id_col", "user_id_raw")
    if uid_col in available and uid_col not in columns:
        columns.append(uid_col)

    # 校验必需列
    missing = [c for c in columns if c not in available]
    if missing:
        raise ValueError(
            f"Parquet 缺少必要列: {missing}\n"
            f"可用列: {sorted(available)}"
        )

    # 去重并保持顺序
    seen: set = set()
    unique_cols = []
    for c in columns:
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)
    return unique_cols


# ─────────────────────────────────────────────────────────────
# IterableDataset（高性能版：预组装 batch）
# ─────────────────────────────────────────────────────────────

class ParquetIterableDataset(IterableDataset):
    """
    高性能 Parquet 流式数据集。

    核心优化：直接 yield 预组装好的 batch（dict of numpy arrays），
    避免逐样本 Python 循环。DataLoader 使用 batch_size=None 配合。

    Args:
        parquet_path: parquet 文件路径
        columns: 需要读取的列名列表
        batch_size: 每个 batch 的样本数
        max_hist_len: 历史序列定长
        list_columns: set，list<int> 列名（自动检测）
        float_columns: set，float 列名（自动检测）
        shuffle: 是否打乱
        base_seed: 随机种子基数
    """

    def __init__(
        self,
        parquet_path: str,
        columns: List[str],
        batch_size: int = 128,
        max_hist_len: int = 50,
        list_columns: Optional[Set[str]] = None,
        float_columns: Optional[Set[str]] = None,
        shuffle: bool = False,
        base_seed: int = 42,
    ):
        super().__init__()
        self.parquet_path = str(parquet_path)
        self.columns = columns
        self.batch_size = batch_size
        self.max_hist_len = max_hist_len
        self.shuffle = shuffle
        self.base_seed = base_seed
        self._epoch = 0

        # 读取元数据（不加载实际数据）
        pf = pq.ParquetFile(self.parquet_path)
        self.num_row_groups = pf.metadata.num_row_groups
        self.num_rows = pf.metadata.num_rows

        # 自动检测列类型
        if list_columns is None:
            self.list_columns = _detect_list_columns(pf.schema_arrow) & set(columns)
        else:
            self.list_columns = list_columns & set(columns)
        if float_columns is None:
            self.float_columns = _detect_float_columns(pf.schema_arrow) & set(columns)
        else:
            self.float_columns = float_columns & set(columns)
        self.string_columns = _detect_string_columns(pf.schema_arrow) & set(columns)

        logger.info(
            "ParquetIterableDataset 初始化: path=%s, rows=%d, row_groups=%d, cols=%d, "
            "batch_size=%d, list_cols=%d, float_cols=%d, shuffle=%s",
            self.parquet_path, self.num_rows, self.num_row_groups,
            len(self.columns), self.batch_size,
            len(self.list_columns), len(self.float_columns),
            self.shuffle,
        )

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch（影响 shuffle 的随机种子，确保每个 epoch 打乱不同）。"""
        self._epoch = epoch

    def _read_row_group(self, pf: pq.ParquetFile, rg_idx: int) -> Dict[str, np.ndarray]:
        """
        读取一个 row group，返回 {列名: numpy 数组} 字典。

        全部使用 numpy 向量化操作，无 Python 逐行循环。
        - list 列 → 2D array [n_rows, max_hist_len]
        - float 列 → 1D float32 array [n_rows]
        - int 列  → 1D int64 array [n_rows]
        """
        table = pf.read_row_group(rg_idx, columns=self.columns)
        n_rows = table.num_rows
        arrays: Dict[str, np.ndarray] = {}

        for col_name in self.columns:
            column = table.column(col_name)

            if col_name in self.list_columns:
                # 列表列：将 ListArray 展平后 reshape 为 [n_rows, list_len]
                # 这比 to_pylist() 快一个数量级
                combined = column.combine_chunks()
                flat_values = combined.values.to_numpy()
                expected_len = n_rows * self.max_hist_len
                if len(flat_values) != expected_len:
                    raise ValueError(
                        f"列 '{col_name}' 展平后长度 {len(flat_values)} "
                        f"!= 期望 {n_rows} × {self.max_hist_len} = {expected_len}。"
                        f"请确认所有历史序列已 pad 到定长 {self.max_hist_len}。"
                    )
                arrays[col_name] = flat_values.reshape(n_rows, self.max_hist_len).astype(np.int64)

            elif col_name in self.float_columns:
                arrays[col_name] = column.to_numpy(zero_copy_only=False).astype(np.float32)

            elif col_name in self.string_columns:
                # 字符串元数据列（如 meta_log_source）保持 object/string，
                # 下游 collate 会保留为 Python list，不参与 tensor 化。
                arrays[col_name] = column.to_numpy(zero_copy_only=False).astype(object)

            else:
                # 整型（含 label / sparse / meta 等）→ int64
                arrays[col_name] = column.to_numpy(zero_copy_only=False).astype(np.int64)

        return arrays

    def _slice_batch(
        self,
        arrays: Dict[str, np.ndarray],
        indices: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        用 numpy fancy indexing 向量化切片出一个 batch。

        这是性能优化的关键：
          旧版：for i in range(batch_size): for col in columns: sample[col] = arr[i]
          新版：for col in columns: batch[col] = arr[indices]  ← 全部向量化

        一个 batch 只需 ~50 次 numpy 操作（每列一次），
        而非旧版的 batch_size × 50 = 6400 次 Python dict 操作。
        """
        batch: Dict[str, np.ndarray] = {}
        for col_name in self.columns:
            batch[col_name] = arrays[col_name][indices]
        return batch

    def __iter__(self):
        """
        迭代逻辑（高性能版）：
        1. 确定当前 worker 负责哪些 row group
        2. 可选打乱 row group 顺序
        3. 对每个 row group：numpy shuffle 索引 → 按 batch_size 切片 → yield 预组装 batch
        4. 全程无逐样本 Python 循环
        """
        # ── Worker 分片 ──
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        # 本 worker 负责的 row group 索引
        rg_indices = [i for i in range(self.num_row_groups) if i % num_workers == worker_id]

        # ── 随机种子（每个 epoch × worker 不同）──
        seed = self.base_seed + self._epoch * 100_000 + worker_id
        np_rng = np.random.RandomState(seed)
        py_rng = random.Random(seed)

        if self.shuffle:
            py_rng.shuffle(rg_indices)

        # ── 每个 worker 独立打开文件句柄（多进程安全）──
        pf = pq.ParquetFile(self.parquet_path)

        for rg_idx in rg_indices:
            arrays = self._read_row_group(pf, rg_idx)
            n_rows = arrays[self.columns[0]].shape[0]

            # row group 内部 shuffle 索引（numpy 操作，微秒级）
            if self.shuffle:
                indices = np_rng.permutation(n_rows).astype(np.int64)
            else:
                indices = np.arange(n_rows, dtype=np.int64)

            # ── 按 batch_size 向量化切片，直接 yield 预组装好的 batch ──
            # 无逐样本 Python 循环！每 yield 只做 ~50 次 numpy fancy indexing。
            for start in range(0, n_rows, self.batch_size):
                end = min(start + self.batch_size, n_rows)
                batch_indices = indices[start:end]
                yield self._slice_batch(arrays, batch_indices)


# ─────────────────────────────────────────────────────────────
# Map-style Dataset（仅用于 debug 小规模数据）
# ─────────────────────────────────────────────────────────────

class DebugMapDataset(Dataset):
    """
    ⚠ 仅限 debug 用途 ⚠

    把 parquet 前 N 行全量载入 pandas，转为 Map-style Dataset。
    大规模训练绝对不要用它——内存会爆。

    Args:
        parquet_path: parquet 文件路径
        columns: 读取的列
        max_rows: 最多加载行数（默认 10000）
        max_hist_len: 列表定长
    """

    def __init__(
        self,
        parquet_path: str,
        columns: List[str],
        max_rows: int = 10_000,
        max_hist_len: int = 50,
    ):
        import pandas as pd

        logger.warning("DebugMapDataset 仅限小规模 debug，大规模训练请使用 ParquetIterableDataset！")
        pf = pq.ParquetFile(parquet_path)
        list_cols = _detect_list_columns(pf.schema_arrow) & set(columns)
        float_cols = _detect_float_columns(pf.schema_arrow) & set(columns)
        string_cols = _detect_string_columns(pf.schema_arrow) & set(columns)

        # 只读前 N 行
        table = pf.read_row_groups(list(range(min(1, pf.metadata.num_row_groups))), columns=columns)
        df = table.to_pandas().head(max_rows)
        self.n = len(df)
        self.max_hist_len = max_hist_len

        # 预转为 numpy 字典
        self.data: Dict[str, np.ndarray] = {}
        for col in columns:
            if col in list_cols:
                # list -> 2D array
                self.data[col] = np.stack(df[col].values).astype(np.int64)
            elif col in float_cols:
                self.data[col] = df[col].values.astype(np.float32)
            elif col in string_cols:
                self.data[col] = df[col].astype(str).values
            else:
                self.data[col] = df[col].values.astype(np.int64)

        logger.info("DebugMapDataset 加载了 %d 行 from %s", self.n, parquet_path)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        sample = {}
        for col, arr in self.data.items():
            if arr.ndim == 2:
                sample[col] = arr[idx]
            else:
                val = arr[idx]
                sample[col] = val.item() if hasattr(val, "item") else val
        return sample

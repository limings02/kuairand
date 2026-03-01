from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import maybe_to_numeric, normalize_path_for_sql, safe_div

try:
    import duckdb

    HAS_DUCKDB = True
except Exception:
    HAS_DUCKDB = False


CANDIDATE_KEYS = [
    ["user_id", "time_ms"],
    ["video_id", "time_ms"],
    ["user_id", "video_id", "time_ms"],
    ["user_id"],
    ["video_id"],
]


@dataclass
class FileProfile:
    """
    单个 CSV 的画像结果。

    属性
    ----
    summary_row : Dict[str, object]
        文件级概要信息（行数、列数、重复率、排序性等）。
    column_stats : pd.DataFrame
        列级别统计（缺失率、类型、数值范围）。
    key_stats : pd.DataFrame
        主键候选组合的唯一性与重复率。
    """

    summary_row: Dict[str, object]
    column_stats: pd.DataFrame
    key_stats: pd.DataFrame


def _read_head(path: str, nrows: int = 50000) -> pd.DataFrame:
    """
    读取 CSV 头部样本用于字段推断。
    """
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def _guess_numeric_cols(sample_df: pd.DataFrame, threshold: float = 0.8) -> List[str]:
    """
    根据样本判断哪些列可近似视为数值列。
    """
    numeric_cols: List[str] = []
    if sample_df.empty:
        return numeric_cols
    for c in sample_df.columns:
        s = maybe_to_numeric(sample_df[c])
        ratio = s.notna().mean()
        if ratio >= threshold:
            numeric_cols.append(c)
    return numeric_cols


def _check_monotonic_user_time(path: str, chunksize: int) -> Tuple[Optional[bool], Optional[int]]:
    """
    检查日志是否按 (user_id, time_ms) 单调不降。

    返回
    ----
    (is_monotonic, violation_count)
    """
    try:
        first_chunk = pd.read_csv(path, nrows=5, low_memory=False)
    except Exception:
        return None, None

    if "user_id" not in first_chunk.columns or "time_ms" not in first_chunk.columns:
        return None, None

    violation_count = 0
    prev_user = None
    prev_time = None

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        if chunk.empty:
            continue
        u = pd.to_numeric(chunk["user_id"], errors="coerce").to_numpy()
        t = pd.to_numeric(chunk["time_ms"], errors="coerce").to_numpy()

        # 边界检查（当前块首行 vs 上一块末行）
        if prev_user is not None and len(u) > 0:
            if (u[0] < prev_user) or (u[0] == prev_user and t[0] < prev_time):
                violation_count += 1

        # 块内向量化检查
        if len(u) > 1:
            bad = (u[1:] < u[:-1]) | ((u[1:] == u[:-1]) & (t[1:] < t[:-1]))
            violation_count += int(np.sum(bad))

        prev_user = u[-1]
        prev_time = t[-1]

    return violation_count == 0, violation_count


def _compute_key_stats_duckdb(path: str, key_candidates: List[List[str]]) -> pd.DataFrame:
    """
    使用 DuckDB 计算候选主键的唯一性与重复率。
    """
    if not HAS_DUCKDB:
        return pd.DataFrame(columns=["key", "distinct_cnt", "total_cnt", "uniqueness_ratio", "duplicate_ratio"])

    con = duckdb.connect(database=":memory:")
    quoted_path = normalize_path_for_sql(path)
    con.execute(
        f"""
        CREATE VIEW v AS
        SELECT * FROM read_csv_auto('{quoted_path}', HEADER=TRUE, UNION_BY_NAME=TRUE, IGNORE_ERRORS=TRUE)
        """
    )

    # DuckDB DESCRIBE 返回列名
    desc = con.execute("DESCRIBE v").fetchdf()
    cols = set(desc["column_name"].tolist())

    total_cnt = int(con.execute("SELECT COUNT(*) FROM v").fetchone()[0])
    rows = []
    for key_cols in key_candidates:
        if not all(c in cols for c in key_cols):
            continue
        key_expr = ", ".join(key_cols)
        distinct_cnt = int(
            con.execute(f"SELECT COUNT(DISTINCT ({key_expr})) FROM v").fetchone()[0]
        )
        uniq_ratio = safe_div(distinct_cnt, total_cnt, default=float("nan"))
        rows.append(
            {
                "key": "+".join(key_cols),
                "distinct_cnt": distinct_cnt,
                "total_cnt": total_cnt,
                "uniqueness_ratio": uniq_ratio,
                "duplicate_ratio": 1.0 - uniq_ratio if pd.notna(uniq_ratio) else float("nan"),
            }
        )
    con.close()
    return pd.DataFrame(rows)


def profile_single_csv(
    path: str,
    chunksize: int = 200000,
    median_sample_size: int = 50000,
) -> FileProfile:
    """
    对单个 CSV 做文件级+列级画像。

    说明
    ----
    - 支持大文件分块读取；
    - 数值中位数使用抽样近似，避免全量排序占用大量内存。
    """
    sample_df = _read_head(path, nrows=min(max(chunksize, 10000), 50000))
    columns = list(sample_df.columns)
    col_cnt = len(columns)
    numeric_cols = set(_guess_numeric_cols(sample_df, threshold=0.8))

    missing_cnt: Dict[str, int] = {c: 0 for c in columns}
    dtype_guess: Dict[str, str] = {c: str(sample_df[c].dtype) for c in columns}
    num_min: Dict[str, float] = {c: float("inf") for c in numeric_cols}
    num_max: Dict[str, float] = {c: float("-inf") for c in numeric_cols}
    num_samples: Dict[str, List[float]] = {c: [] for c in numeric_cols}
    num_valid_cnt: Dict[str, int] = {c: 0 for c in numeric_cols}

    total_rows = 0
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        total_rows += len(chunk)
        miss = chunk.isna().sum()
        for c in columns:
            missing_cnt[c] += int(miss.get(c, 0))

        for c in numeric_cols:
            s = maybe_to_numeric(chunk[c]).dropna()
            if s.empty:
                continue
            num_valid_cnt[c] += int(s.shape[0])
            cmin = float(s.min())
            cmax = float(s.max())
            if cmin < num_min[c]:
                num_min[c] = cmin
            if cmax > num_max[c]:
                num_max[c] = cmax

            remain = median_sample_size - len(num_samples[c])
            if remain > 0:
                num_samples[c].extend(s.iloc[:remain].tolist())

    # 列统计表
    col_rows = []
    for c in columns:
        row = {
            "column": c,
            "dtype_inferred": dtype_guess.get(c, "unknown"),
            "missing_count": missing_cnt[c],
            "missing_ratio": safe_div(missing_cnt[c], total_rows, default=float("nan")),
            "is_numeric_like": c in numeric_cols,
            "min": float("nan"),
            "median_approx": float("nan"),
            "max": float("nan"),
        }
        if c in numeric_cols and num_valid_cnt[c] > 0:
            row["min"] = num_min[c]
            row["max"] = num_max[c]
            if len(num_samples[c]) > 0:
                row["median_approx"] = float(np.median(np.asarray(num_samples[c], dtype=float)))
        col_rows.append(row)
    column_stats = pd.DataFrame(col_rows)

    key_stats = _compute_key_stats_duckdb(path, CANDIDATE_KEYS)
    key_candidates = ",".join(key_stats["key"].tolist()) if not key_stats.empty else ""

    best_key = ""
    best_dup_ratio = float("nan")
    if not key_stats.empty:
        best_idx = key_stats["uniqueness_ratio"].idxmax()
        best_key = str(key_stats.loc[best_idx, "key"])
        best_dup_ratio = float(key_stats.loc[best_idx, "duplicate_ratio"])

    # 单调性检查仅对日志类文件有意义，这里统一检测；没有对应列则返回空
    mono, vio = _check_monotonic_user_time(path, chunksize=chunksize)

    summary_row = {
        "file_path": path,
        "file_name": Path(path).name,
        "row_count": int(total_rows),
        "col_count": int(col_cnt),
        "candidate_keys": key_candidates,
        "best_key": best_key,
        "best_key_duplicate_ratio": best_dup_ratio,
        "is_monotonic_user_time": mono,
        "monotonic_violation_count": vio,
    }

    return FileProfile(
        summary_row=summary_row,
        column_stats=column_stats,
        key_stats=key_stats,
    )


def profile_many_files(
    file_paths: List[str],
    chunksize: int = 200000,
    median_sample_size: int = 50000,
) -> Tuple[pd.DataFrame, Dict[str, FileProfile]]:
    """
    批量画像多个文件。

    返回
    ----
    summary_df : pd.DataFrame
        文件级汇总。
    detail : Dict[str, FileProfile]
        文件路径到详细画像对象。
    """
    detail: Dict[str, FileProfile] = {}
    summary_rows = []
    for p in file_paths:
        prof = profile_single_csv(
            path=p,
            chunksize=chunksize,
            median_sample_size=median_sample_size,
        )
        detail[p] = prof
        summary_rows.append(prof.summary_row)
    summary_df = pd.DataFrame(summary_rows)
    return summary_df, detail


from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def ensure_output_dirs(out_dir: str) -> Dict[str, str]:
    """
    创建并返回报告输出目录结构。

    参数
    ----------
    out_dir : str
        输出根目录。

    返回
    -------
    Dict[str, str]
        包含根目录、figures、tables、notebooks、src 子目录路径。
    """
    base = Path(out_dir)
    dirs = {
        "base": str(base),
        "figures": str(base / "figures"),
        "tables": str(base / "tables"),
        "notebooks": str(base / "notebooks"),
        "src": str(base / "src"),
    }
    for p in dirs.values():
        Path(p).mkdir(parents=True, exist_ok=True)
    return dirs


def normalize_path_for_sql(path: str) -> str:
    """
    将 Windows 路径标准化为 DuckDB SQL 可安全引用的形式。
    """
    return path.replace("\\", "/").replace("'", "''")


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    安全除法，避免除零。
    """
    if denominator == 0:
        return default
    return float(numerator) / float(denominator)


def to_pct(x: float, digits: int = 4) -> float:
    """
    将比例转为百分数（数值型，不带 %）。
    """
    return round(100.0 * float(x), digits)


def write_csv(df: pd.DataFrame, path: str, index: bool = False) -> str:
    """
    将 DataFrame 写入 CSV，并确保父目录存在。
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig")
    return path


def choose_first_existing(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
    """
    在 candidates 中返回第一个存在于 columns 的列名。
    """
    col_set = set(columns)
    for c in candidates:
        if c in col_set:
            return c
    return None


def calc_hhi(counts: Iterable[float]) -> float:
    """
    计算 HHI 指标（Herfindahl-Hirschman Index）。
    """
    arr = np.asarray(list(counts), dtype=float)
    if arr.size == 0:
        return float("nan")
    s = arr.sum()
    if s <= 0:
        return float("nan")
    p = arr / s
    return float(np.sum(p * p))


def calc_gini(counts: Iterable[float]) -> float:
    """
    计算 Gini 系数（用于衡量曝光分配集中度）。
    """
    x = np.asarray(list(counts), dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan")
    if np.any(x < 0):
        x = x - np.min(x)
    s = np.sum(x)
    if s == 0:
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    idx = np.arange(1, n + 1)
    gini = (2.0 * np.sum(idx * x_sorted) / (n * s)) - ((n + 1.0) / n)
    return float(gini)


def bootstrap_ctr_diff_from_counts(
    click_a: int,
    expo_a: int,
    click_b: int,
    expo_b: int,
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    基于二项分布近似进行 CTR 差异 bootstrap。

    说明
    ----
    - 不直接重采样原始样本，而是用 (n, p) 二项分布采样近似。
    - 对超大数据集更省内存，且统计上可接受。
    """
    if expo_a <= 0 or expo_b <= 0:
        return (float("nan"), float("nan"), float("nan"))

    p_a = click_a / expo_a
    p_b = click_b / expo_b

    rng = np.random.default_rng(seed)
    sample_a = rng.binomial(expo_a, p_a, size=n_boot) / expo_a
    sample_b = rng.binomial(expo_b, p_b, size=n_boot) / expo_b
    diff = sample_a - sample_b
    return (
        float(np.quantile(diff, 0.025)),
        float(np.quantile(diff, 0.5)),
        float(np.quantile(diff, 0.975)),
    )


def normal_cdf(x: float) -> float:
    """
    标准正态分布 CDF（不依赖 scipy）。
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_proportion_ztest(
    click_a: int,
    expo_a: int,
    click_b: int,
    expo_b: int,
) -> Tuple[float, float]:
    """
    两比例 z 检验，返回 z 值与双侧 p 值。
    """
    if expo_a <= 0 or expo_b <= 0:
        return float("nan"), float("nan")
    p_a = click_a / expo_a
    p_b = click_b / expo_b
    se = math.sqrt((p_a * (1 - p_a) / expo_a) + (p_b * (1 - p_b) / expo_b))
    if se == 0:
        return float("nan"), float("nan")
    z = (p_a - p_b) / se
    p = 2.0 * (1.0 - normal_cdf(abs(z)))
    return float(z), float(p)


def infer_role_from_filename(filename: str) -> str:
    """
    根据文件名关键词识别角色。
    """
    name = filename.lower()
    if "log" in name and "standard" in name:
        return "log_standard"
    if "log" in name and "random" in name:
        return "log_random"
    if "user" in name and "feature" in name:
        return "user_features"
    if "video" in name and "basic" in name:
        return "video_features_basic"
    if "video" in name and ("statistic" in name or "statistics" in name):
        return "video_features_statistic"
    return "unknown"


def maybe_to_numeric(series: pd.Series) -> pd.Series:
    """
    尝试把列转为数值，失败项为 NaN。
    """
    return pd.to_numeric(series, errors="coerce")


def copy_source_files_to_report(src_dir: str, report_src_dir: str) -> None:
    """
    将 src 代码复制到报告目录，便于结果归档和复现。
    """
    src_root = Path(src_dir)
    dst_root = Path(report_src_dir)
    dst_root.mkdir(parents=True, exist_ok=True)

    for py_file in src_root.glob("*.py"):
        target = dst_root / py_file.name
        target.write_text(py_file.read_text(encoding="utf-8"), encoding="utf-8")

    run_file = Path("run_eda.py")
    if run_file.exists():
        (dst_root / "run_eda.py").write_text(
            run_file.read_text(encoding="utf-8"), encoding="utf-8"
        )


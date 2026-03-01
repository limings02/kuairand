from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .utils import infer_role_from_filename


@dataclass
class ScanResult:
    """
    数据目录扫描结果。

    属性
    ----
    role_to_files : Dict[str, List[str]]
        文件角色到路径列表的映射。
    inventory_df : pd.DataFrame
        每个文件的清单（角色、大小、路径等）。
    summary_df : pd.DataFrame
        每个角色的文件数量与总体积汇总。
    """

    role_to_files: Dict[str, List[str]]
    inventory_df: pd.DataFrame
    summary_df: pd.DataFrame


def scan_data_dir(data_dir: str) -> ScanResult:
    """
    扫描目录下全部 CSV，并根据文件名自动识别角色。

    参数
    ----------
    data_dir : str
        KuaiRand 解压目录（可以是 data 子目录，也可以是更上层目录）。
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    csv_files = sorted(root.rglob("*.csv"))
    role_to_files: Dict[str, List[str]] = {
        "log_standard": [],
        "log_random": [],
        "user_features": [],
        "video_features_basic": [],
        "video_features_statistic": [],
        "unknown": [],
    }

    rows = []
    for fp in csv_files:
        role = infer_role_from_filename(fp.name)
        role_to_files.setdefault(role, []).append(str(fp))
        size_mb = fp.stat().st_size / 1024 / 1024
        rows.append(
            {
                "role": role,
                "file_name": fp.name,
                "file_path": str(fp),
                "size_mb": round(size_mb, 4),
            }
        )

    inventory_df = pd.DataFrame(rows).sort_values(["role", "file_name"]).reset_index(drop=True)

    summary_df = (
        inventory_df.groupby("role", as_index=False)
        .agg(file_count=("file_name", "count"), total_size_mb=("size_mb", "sum"))
        .sort_values("role")
        .reset_index(drop=True)
    )
    if not summary_df.empty:
        summary_df["total_size_mb"] = summary_df["total_size_mb"].round(4)

    return ScanResult(
        role_to_files=role_to_files,
        inventory_df=inventory_df,
        summary_df=summary_df,
    )


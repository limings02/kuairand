from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq

    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


logger = logging.getLogger("src.analysis.tab_click_stats")


DEFAULT_GLOB_STANDARD_PRE = "log_standard_4_08_to_4_21*.csv"
DEFAULT_GLOB_STANDARD_EXP = "log_standard_4_22_to_5_08*.csv"
DEFAULT_GLOB_RANDOM_EXP = "log_random_4_22_to_5_08*.csv"


@dataclass
class SourceStats:
    """每个 source 的质量与计数统计。"""

    rows_seen: int = 0
    sample_count: int = 0
    click_positive_count: int = 0
    missing_tab_count: int = 0
    missing_is_click_count: int = 0
    invalid_tab_count: int = 0
    invalid_is_click_count: int = 0
    rows_dropped: int = 0
    files_processed: int = 0
    chunks_processed: int = 0


def parse_bool(value: str) -> bool:
    """将 true/false 字符串解析成布尔值。"""
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value}")


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 tab 统计样本量与 is_click 正样本率（内存友好分块实现）"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["raw_logs", "processed_parquet"],
        default="raw_logs",
        help="统计模式：raw_logs 或 processed_parquet",
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="data/KuaiRand-27K",
        help="原始日志目录（raw_logs 模式使用）",
    )
    parser.add_argument(
        "--processed_root",
        type=str,
        default="output/processed",
        help="processed parquet 目录（processed_parquet 模式使用）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/analysis/tab_click_stats",
        help="输出目录",
    )

    parser.add_argument(
        "--include_standard_pre",
        type=parse_bool,
        default=False,
        help="是否包含 standard_pre（4_08_to_4_21）",
    )
    parser.add_argument(
        "--include_standard_exp",
        type=parse_bool,
        default=True,
        help="是否包含 standard_exp（4_22_to_5_08）",
    )
    parser.add_argument(
        "--include_random_exp",
        type=parse_bool,
        default=True,
        help="是否包含 random_exp（4_22_to_5_08）",
    )

    parser.add_argument(
        "--glob_pattern_standard_pre",
        type=str,
        default=DEFAULT_GLOB_STANDARD_PRE,
        help="standard_pre 的 glob 模式（可逗号分隔多个）",
    )
    parser.add_argument(
        "--glob_pattern_standard_exp",
        type=str,
        default=DEFAULT_GLOB_STANDARD_EXP,
        help="standard_exp 的 glob 模式（可逗号分隔多个）",
    )
    parser.add_argument(
        "--glob_pattern_random_exp",
        type=str,
        default=DEFAULT_GLOB_RANDOM_EXP,
        help="random_exp 的 glob 模式（可逗号分隔多个）",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=200000,
        help="CSV/parquet 批次大小（16GB 内存建议 100000~500000）",
    )
    parser.add_argument(
        "--debug_max_chunks",
        type=int,
        default=0,
        help="调试时每个文件最多处理多少个 chunk（0 表示不限制）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出更详细的过程日志",
    )
    return parser.parse_args()


def split_patterns(pattern_text: str) -> List[str]:
    """支持逗号/分号分隔多个 glob。"""
    if not pattern_text:
        return []
    text = pattern_text.replace(";", ",")
    return [item.strip() for item in text.split(",") if item.strip()]


def expand_glob_patterns(base_dir: Path, pattern_text: str) -> List[Path]:
    """展开 glob 并去重排序。"""
    results: List[Path] = []
    seen: set[str] = set()
    for pattern in split_patterns(pattern_text):
        pattern_path = Path(pattern)
        if pattern_path.is_absolute():
            matched = [Path(item) for item in glob.glob(str(pattern_path))]
        else:
            matched = list(base_dir.glob(pattern))
        for file_path in sorted(matched):
            normalized = str(file_path.resolve())
            if normalized not in seen:
                seen.add(normalized)
                results.append(file_path)
    return sorted(results)


def make_source_order(log_sources: Iterable[str]) -> Dict[str, int]:
    """用于结果排序：固定 source 顺序 + 兜底字母序。"""
    preferred_order = [
        "standard_pre",
        "standard_exp",
        "random_exp",
        "train",
        "val",
        "test_standard",
        "test_random",
        "all_selected",
    ]
    order_map = {name: idx for idx, name in enumerate(preferred_order)}
    next_rank = len(preferred_order)
    for source in sorted(set(log_sources)):
        if source not in order_map:
            order_map[source] = next_rank
            next_rank += 1
    return order_map


def _safe_rate(positive: int, total: int) -> float:
    return float(positive) / float(total) if total > 0 else 0.0


def _compress_tab_dtype(tab_values: pd.Series) -> pd.Series:
    """
    tab 列尽早压缩 dtype：优先 int8，其次 int16，最后 int32。

    说明：
    - 这里在每个 chunk 内做压缩，避免在内存里保留更宽的类型；
    - 对 16GB RAM 机器，长时间扫描大 CSV 时可显著降低峰值占用。
    """
    if tab_values.empty:
        return tab_values.astype(np.int16)

    min_val = int(tab_values.min())
    max_val = int(tab_values.max())
    if np.iinfo(np.int8).min <= min_val <= max_val <= np.iinfo(np.int8).max:
        return tab_values.astype(np.int8)
    if np.iinfo(np.int16).min <= min_val <= max_val <= np.iinfo(np.int16).max:
        return tab_values.astype(np.int16)
    return tab_values.astype(np.int32)


def update_stats_with_chunk(
    chunk: pd.DataFrame,
    source: str,
    source_stats: Dict[str, SourceStats],
    tab_counters: Dict[str, DefaultDict[int, List[int]]],
) -> Tuple[int, int]:
    """
    分块累计核心逻辑：
    - 只依赖 `tab` 和 `is_click` 两列；
    - 对每个 chunk 做缺失与异常过滤；
    - 仅累计 group 结果，不保留 chunk 常驻内存。

    返回
    ----
    (valid_rows, dropped_rows)
    """
    stats = source_stats[source]

    row_count = len(chunk)
    stats.rows_seen += row_count
    if row_count == 0:
        return 0, 0

    tab_num = pd.to_numeric(chunk["tab"], errors="coerce")
    click_num = pd.to_numeric(chunk["is_click"], errors="coerce")

    missing_tab_mask = tab_num.isna()
    missing_click_mask = click_num.isna()
    stats.missing_tab_count += int(missing_tab_mask.sum())
    stats.missing_is_click_count += int(missing_click_mask.sum())

    base_valid_mask = ~(missing_tab_mask | missing_click_mask)
    if not bool(base_valid_mask.any()):
        stats.rows_dropped += row_count
        return 0, row_count

    tab_valid = tab_num[base_valid_mask]
    click_valid = click_num[base_valid_mask]

    tab_int_like_mask = np.isclose(tab_valid.values, np.round(tab_valid.values))
    invalid_tab = int((~tab_int_like_mask).sum())
    stats.invalid_tab_count += invalid_tab

    click_binary_mask = click_valid.isin([0, 1]).values
    invalid_click = int((~click_binary_mask).sum())
    stats.invalid_is_click_count += invalid_click

    final_mask = tab_int_like_mask & click_binary_mask
    valid_rows = int(final_mask.sum())
    dropped_rows = row_count - valid_rows
    stats.rows_dropped += dropped_rows

    if valid_rows == 0:
        return 0, dropped_rows

    clean_tab = pd.Series(np.round(tab_valid.values[final_mask]), index=None, dtype=np.int64)
    clean_tab = _compress_tab_dtype(clean_tab)
    clean_click = pd.Series(click_valid.values[final_mask], index=None).astype(np.int8)

    grouped = (
        pd.DataFrame({"tab": clean_tab, "is_click": clean_click})
        .groupby("tab", sort=False)["is_click"]
        .agg(["count", "sum"])
    )

    total_positive = int(clean_click.sum())
    stats.sample_count += valid_rows
    stats.click_positive_count += total_positive

    for tab_value, row in grouped.iterrows():
        tab_key = int(tab_value)
        tab_counters[source][tab_key][0] += int(row["count"])
        tab_counters[source][tab_key][1] += int(row["sum"])

    return valid_rows, dropped_rows


def process_csv_file(
    file_path: Path,
    source: str,
    chunksize: int,
    debug_max_chunks: int,
    source_stats: Dict[str, SourceStats],
    tab_counters: Dict[str, DefaultDict[int, List[int]]],
    verbose: bool,
) -> None:
    """分块处理单个 CSV 文件。"""
    logger.info("开始处理 CSV | source=%s | file=%s", source, file_path)
    stats = source_stats[source]
    stats.files_processed += 1

    try:
        chunk_iter = pd.read_csv(
            file_path,
            usecols=["tab", "is_click"],
            chunksize=chunksize,
            low_memory=True,
        )
    except ValueError as exc:
        raise ValueError(
            f"CSV 缺少必要列(tab/is_click): {file_path}，错误信息: {exc}"
        ) from exc

    for chunk_idx, chunk in enumerate(chunk_iter, start=1):
        if debug_max_chunks > 0 and chunk_idx > debug_max_chunks:
            logger.warning(
                "debug_max_chunks=%d 生效，提前停止文件处理: %s",
                debug_max_chunks,
                file_path,
            )
            break

        valid_rows, dropped_rows = update_stats_with_chunk(
            chunk=chunk,
            source=source,
            source_stats=source_stats,
            tab_counters=tab_counters,
        )
        stats.chunks_processed += 1

        if verbose or chunk_idx == 1 or chunk_idx % 10 == 0:
            logger.info(
                "CSV 进度 | source=%s | file=%s | chunk=%d | chunk_rows=%d | valid=%d | dropped=%d | cum_valid=%d",
                source,
                file_path.name,
                chunk_idx,
                len(chunk),
                valid_rows,
                dropped_rows,
                stats.sample_count,
            )


def process_parquet_file(
    file_path: Path,
    source: str,
    chunksize: int,
    debug_max_chunks: int,
    source_stats: Dict[str, SourceStats],
    tab_counters: Dict[str, DefaultDict[int, List[int]]],
    verbose: bool,
) -> None:
    """
    使用 pyarrow 分批读取 parquet。

    说明：
    - 为满足 16GB 内存约束，不直接 `pd.read_parquet` 整表；
    - 仅取 `tab/is_click` 两列并按 batch 累计。
    """
    if not HAS_PYARROW:
        raise ImportError("未安装 pyarrow，无法使用 processed_parquet 模式")

    logger.info("开始处理 Parquet | source=%s | file=%s", source, file_path)
    stats = source_stats[source]
    stats.files_processed += 1

    parquet_file = pq.ParquetFile(file_path)
    schema_columns = set(parquet_file.schema_arrow.names)
    required_cols = {"tab", "is_click"}
    missing_cols = required_cols - schema_columns
    if missing_cols:
        missing_text = ",".join(sorted(missing_cols))
        raise ValueError(
            f"Parquet 缺少必要列: {file_path}，缺失列: {missing_text}。"
            "请确认 processed 数据中包含 tab 与 is_click。"
        )

    for batch_idx, batch in enumerate(
        parquet_file.iter_batches(batch_size=chunksize, columns=["tab", "is_click"]),
        start=1,
    ):
        if debug_max_chunks > 0 and batch_idx > debug_max_chunks:
            logger.warning(
                "debug_max_chunks=%d 生效，提前停止文件处理: %s",
                debug_max_chunks,
                file_path,
            )
            break

        chunk = batch.to_pandas()
        valid_rows, dropped_rows = update_stats_with_chunk(
            chunk=chunk,
            source=source,
            source_stats=source_stats,
            tab_counters=tab_counters,
        )
        stats.chunks_processed += 1

        if verbose or batch_idx == 1 or batch_idx % 10 == 0:
            logger.info(
                "Parquet 进度 | source=%s | file=%s | batch=%d | batch_rows=%d | valid=%d | dropped=%d | cum_valid=%d",
                source,
                file_path.name,
                batch_idx,
                len(chunk),
                valid_rows,
                dropped_rows,
                stats.sample_count,
            )


def build_result_dataframe(
    tab_counters: Dict[str, DefaultDict[int, List[int]]],
    include_all_selected: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """构建 by_source 与 overall 两张结果表。"""
    by_source_rows: List[Dict[str, float]] = []
    overall_counter: DefaultDict[int, List[int]] = defaultdict(lambda: [0, 0])

    for source, tab_counter in tab_counters.items():
        for tab, (sample_count, positive_count) in tab_counter.items():
            rate = _safe_rate(positive_count, sample_count)
            by_source_rows.append(
                {
                    "log_source": source,
                    "tab": int(tab),
                    "sample_count": int(sample_count),
                    "click_positive_count": int(positive_count),
                    "click_positive_rate": rate,
                }
            )
            overall_counter[int(tab)][0] += int(sample_count)
            overall_counter[int(tab)][1] += int(positive_count)

    overall_rows: List[Dict[str, float]] = []
    for tab, (sample_count, positive_count) in overall_counter.items():
        overall_rows.append(
            {
                "tab": int(tab),
                "sample_count": int(sample_count),
                "click_positive_count": int(positive_count),
                "click_positive_rate": _safe_rate(positive_count, sample_count),
            }
        )

    by_source_df = pd.DataFrame(by_source_rows)
    overall_df = pd.DataFrame(overall_rows)

    if by_source_df.empty:
        by_source_df = pd.DataFrame(
            columns=[
                "log_source",
                "tab",
                "sample_count",
                "click_positive_count",
                "click_positive_rate",
            ]
        )
    if overall_df.empty:
        overall_df = pd.DataFrame(
            columns=["tab", "sample_count", "click_positive_count", "click_positive_rate"]
        )

    if include_all_selected and not overall_df.empty:
        all_selected_df = overall_df.copy()
        all_selected_df.insert(0, "log_source", "all_selected")
        by_source_df = pd.concat([by_source_df, all_selected_df], ignore_index=True)

    source_order = make_source_order(by_source_df["log_source"].tolist())
    if not by_source_df.empty:
        by_source_df["_source_order"] = by_source_df["log_source"].map(source_order)
        by_source_df = by_source_df.sort_values(
            ["_source_order", "tab"], ascending=[True, True]
        ).drop(columns=["_source_order"])
        by_source_df = by_source_df.reset_index(drop=True)

    if not overall_df.empty:
        overall_df = overall_df.sort_values("tab").reset_index(drop=True)

    return by_source_df, overall_df


def dataframe_to_markdown(df: pd.DataFrame, float_cols: Optional[List[str]] = None) -> str:
    """无 tabulate 依赖的 DataFrame -> Markdown 表格。"""
    if df.empty:
        return "（空表）"

    float_cols = float_cols or []
    headers = list(df.columns)

    def _format_value(col: str, value: object) -> str:
        if pd.isna(value):
            return ""
        if col in float_cols:
            return f"{float(value):.6f}"
        if isinstance(value, (float, np.floating)) and (math.isfinite(float(value))):
            if abs(float(value) - round(float(value))) < 1e-12:
                return str(int(round(float(value))))
            return f"{float(value):.6f}"
        return str(value)

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        values = [_format_value(col, row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def build_rate_delta_table(by_source_df: pd.DataFrame) -> pd.DataFrame:
    """
    生成 standard_exp vs random_exp 的 tab CTR 差值表。

    差值定义：standard_exp_rate - random_exp_rate。
    """
    need_sources = {"standard_exp", "random_exp"}
    available_sources = set(by_source_df["log_source"].unique())
    if not need_sources.issubset(available_sources):
        return pd.DataFrame()

    standard_df = by_source_df[by_source_df["log_source"] == "standard_exp"].copy()
    random_df = by_source_df[by_source_df["log_source"] == "random_exp"].copy()

    standard_df = standard_df[["tab", "sample_count", "click_positive_rate"]].rename(
        columns={
            "sample_count": "standard_exp_sample_count",
            "click_positive_rate": "standard_exp_rate",
        }
    )
    random_df = random_df[["tab", "sample_count", "click_positive_rate"]].rename(
        columns={
            "sample_count": "random_exp_sample_count",
            "click_positive_rate": "random_exp_rate",
        }
    )

    merged = standard_df.merge(random_df, on="tab", how="outer")
    merged = merged.fillna(
        {
            "standard_exp_sample_count": 0,
            "standard_exp_rate": 0.0,
            "random_exp_sample_count": 0,
            "random_exp_rate": 0.0,
        }
    )
    merged["rate_diff_standard_minus_random"] = (
        merged["standard_exp_rate"] - merged["random_exp_rate"]
    )
    merged = merged.sort_values("rate_diff_standard_minus_random", ascending=False).reset_index(drop=True)
    return merged


def plot_click_rate_by_source(by_source_df: pd.DataFrame, output_path: Path) -> Optional[Path]:
    """绘制分 source 的 tab 正样本率柱状图。"""
    if not HAS_MATPLOTLIB:
        logger.warning("未安装 matplotlib，跳过条形图输出")
        return None

    draw_df = by_source_df[by_source_df["log_source"] != "all_selected"].copy()
    if draw_df.empty:
        logger.warning("无可绘图数据，跳过条形图输出")
        return None

    pivot_df = draw_df.pivot_table(
        index="tab",
        columns="log_source",
        values="click_positive_rate",
        aggfunc="first",
        fill_value=0.0,
    ).sort_index()
    if pivot_df.empty:
        logger.warning("透视结果为空，跳过条形图输出")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "sans-serif",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    tabs = pivot_df.index.astype(str).tolist()
    sources = pivot_df.columns.tolist()

    x = np.arange(len(tabs))
    width = 0.8 / max(len(sources), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(tabs) * 1.2), 5))
    for idx, source in enumerate(sources):
        offset = (idx - (len(sources) - 1) / 2) * width
        ax.bar(x + offset, pivot_df[source].values, width=width, label=source)

    ax.set_title("各 source 在不同 tab 的 is_click 正样本率")
    ax.set_xlabel("tab")
    ax.set_ylabel("click_positive_rate")
    ax.set_xticks(x)
    ax.set_xticklabels(tabs)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("已输出图表: %s", output_path)
    return output_path


def build_markdown_report(
    output_path: Path,
    mode: str,
    args: argparse.Namespace,
    source_files: Dict[str, List[str]],
    by_source_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    source_stats: Dict[str, SourceStats],
    runtime_seconds: float,
    rate_delta_df: pd.DataFrame,
    figure_path: Optional[Path],
) -> None:
    """生成 Markdown 报告。"""
    total_valid_samples = int(overall_df["sample_count"].sum()) if not overall_df.empty else 0
    source_sample_rows = []
    for source, stats in source_stats.items():
        source_sample_rows.append(
            {
                "log_source": source,
                "rows_seen": stats.rows_seen,
                "sample_count": stats.sample_count,
                "click_positive_count": stats.click_positive_count,
                "click_positive_rate": _safe_rate(stats.click_positive_count, stats.sample_count),
            }
        )
    source_sample_df = pd.DataFrame(source_sample_rows).sort_values("log_source")

    anomaly_rows = []
    for source, stats in source_stats.items():
        anomaly_rows.append(
            {
                "log_source": source,
                "missing_tab_count": stats.missing_tab_count,
                "missing_is_click_count": stats.missing_is_click_count,
                "invalid_tab_count": stats.invalid_tab_count,
                "invalid_is_click_count": stats.invalid_is_click_count,
                "rows_dropped": stats.rows_dropped,
            }
        )
    anomaly_df = pd.DataFrame(anomaly_rows).sort_values("log_source")

    top_tab_df = overall_df.sort_values("click_positive_rate", ascending=False).head(5)
    low_tab_df = overall_df.sort_values("click_positive_rate", ascending=True).head(5)

    text_parts: List[str] = []
    text_parts.append("# Tab Click 统计报告")
    text_parts.append("")
    text_parts.append("## 运行信息")
    text_parts.append(f"- mode: {mode}")
    text_parts.append(f"- output_dir: {args.output_dir}")
    text_parts.append(f"- chunksize: {args.chunksize}")
    text_parts.append(f"- debug_max_chunks: {args.debug_max_chunks}")
    text_parts.append(f"- runtime_seconds: {runtime_seconds:.3f}")
    text_parts.append(f"- total_valid_samples: {total_valid_samples}")
    text_parts.append("")

    text_parts.append("## 输入文件")
    for source, files in source_files.items():
        text_parts.append(f"- {source}: {len(files)} 个文件")
        for file_path in files:
            text_parts.append(f"  - {file_path}")
    text_parts.append("")

    text_parts.append("## 各 source 样本汇总")
    text_parts.append(
        dataframe_to_markdown(
            source_sample_df,
            float_cols=["click_positive_rate"],
        )
    )
    text_parts.append("")

    text_parts.append("## 异常与缺失行统计")
    text_parts.append(dataframe_to_markdown(anomaly_df))
    text_parts.append("")

    text_parts.append("## tab 统计（按 source）")
    text_parts.append(
        dataframe_to_markdown(
            by_source_df,
            float_cols=["click_positive_rate"],
        )
    )
    text_parts.append("")

    text_parts.append("## tab 统计（overall）")
    text_parts.append(
        dataframe_to_markdown(
            overall_df,
            float_cols=["click_positive_rate"],
        )
    )
    text_parts.append("")

    text_parts.append("## click_positive_rate 排行")
    text_parts.append("### Top 5")
    text_parts.append(
        dataframe_to_markdown(
            top_tab_df,
            float_cols=["click_positive_rate"],
        )
    )
    text_parts.append("")
    text_parts.append("### Bottom 5")
    text_parts.append(
        dataframe_to_markdown(
            low_tab_df,
            float_cols=["click_positive_rate"],
        )
    )
    text_parts.append("")

    if not rate_delta_df.empty:
        text_parts.append("## standard_exp vs random_exp（同 tab 差值）")
        text_parts.append(
            dataframe_to_markdown(
                rate_delta_df,
                float_cols=[
                    "standard_exp_rate",
                    "random_exp_rate",
                    "rate_diff_standard_minus_random",
                ],
            )
        )
        text_parts.append("")

    if figure_path is not None:
        rel_fig = figure_path.name if figure_path.parent == output_path.parent else str(figure_path)
        text_parts.append("## 可视化")
        text_parts.append(f"- 图表路径: {rel_fig}")
        text_parts.append("")

    output_path.write_text("\n".join(text_parts), encoding="utf-8")
    logger.info("已输出 Markdown 报告: %s", output_path)


def write_summary_json(
    output_path: Path,
    mode: str,
    args: argparse.Namespace,
    source_files: Dict[str, List[str]],
    source_stats: Dict[str, SourceStats],
    by_source_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    runtime_seconds: float,
) -> None:
    """写出 JSON 汇总（运行参数 + 计数 + 耗时）。"""
    payload = {
        "mode": mode,
        "runtime_seconds": runtime_seconds,
        "arguments": {
            "data_root": args.data_root,
            "processed_root": args.processed_root,
            "output_dir": args.output_dir,
            "chunksize": args.chunksize,
            "debug_max_chunks": args.debug_max_chunks,
            "include_standard_pre": args.include_standard_pre,
            "include_standard_exp": args.include_standard_exp,
            "include_random_exp": args.include_random_exp,
            "glob_pattern_standard_pre": args.glob_pattern_standard_pre,
            "glob_pattern_standard_exp": args.glob_pattern_standard_exp,
            "glob_pattern_random_exp": args.glob_pattern_random_exp,
            "verbose": args.verbose,
        },
        "source_file_counts": {k: len(v) for k, v in source_files.items()},
        "source_files": source_files,
        "source_stats": {k: asdict(v) for k, v in source_stats.items()},
        "total_rows_by_source": int(
            by_source_df[by_source_df["log_source"] != "all_selected"]["sample_count"].sum()
        )
        if not by_source_df.empty
        else 0,
        "total_rows_overall": int(overall_df["sample_count"].sum()) if not overall_df.empty else 0,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("已输出 JSON 汇总: %s", output_path)


def resolve_raw_log_sources(args: argparse.Namespace) -> Dict[str, List[Path]]:
    """根据配置展开 raw log 文件列表并打 source 标签。"""
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root 不存在: {data_root}")

    source_configs = [
        ("standard_pre", args.include_standard_pre, args.glob_pattern_standard_pre),
        ("standard_exp", args.include_standard_exp, args.glob_pattern_standard_exp),
        ("random_exp", args.include_random_exp, args.glob_pattern_random_exp),
    ]

    resolved: Dict[str, List[Path]] = {}
    for source, include, pattern in source_configs:
        if not include:
            logger.info("跳过 source=%s（include=false）", source)
            continue
        files = expand_glob_patterns(data_root, pattern)
        if not files:
            logger.warning("source=%s 未匹配到文件，pattern=%s", source, pattern)
            continue
        resolved[source] = files

    if not resolved:
        raise FileNotFoundError(
            "raw_logs 模式下未找到任何可用日志文件，请检查 include_* 和 glob_pattern_* 参数。"
        )
    return resolved


def resolve_processed_sources(args: argparse.Namespace) -> Dict[str, List[Path]]:
    """解析 processed parquet 模式的输入文件。"""
    processed_root = Path(args.processed_root)
    if not processed_root.exists():
        raise FileNotFoundError(f"processed_root 不存在: {processed_root}")

    file_map = {
        "train": processed_root / "train.parquet",
        "val": processed_root / "val.parquet",
        "test_standard": processed_root / "test_standard.parquet",
        "test_random": processed_root / "test_random.parquet",
    }
    resolved: Dict[str, List[Path]] = {}
    for source, path in file_map.items():
        if path.exists():
            resolved[source] = [path]
        else:
            logger.warning("processed 文件不存在，跳过: %s", path)

    if not resolved:
        raise FileNotFoundError("processed_parquet 模式下未找到任何 parquet 文件")
    return resolved


def run_statistics(args: argparse.Namespace) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, SourceStats],
    Dict[str, List[str]],
    pd.DataFrame,
    Optional[Path],
    float,
]:
    """执行主统计流程并返回结果对象。"""
    start_time = time.time()

    source_stats: Dict[str, SourceStats] = defaultdict(SourceStats)
    tab_counters: Dict[str, DefaultDict[int, List[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )

    if args.mode == "raw_logs":
        source_files = resolve_raw_log_sources(args)
    else:
        source_files = resolve_processed_sources(args)

    source_files_str = {src: [str(p.resolve()) for p in paths] for src, paths in source_files.items()}

    for source, files in source_files.items():
        for file_path in files:
            if args.mode == "raw_logs":
                process_csv_file(
                    file_path=file_path,
                    source=source,
                    chunksize=args.chunksize,
                    debug_max_chunks=args.debug_max_chunks,
                    source_stats=source_stats,
                    tab_counters=tab_counters,
                    verbose=args.verbose,
                )
            else:
                process_parquet_file(
                    file_path=file_path,
                    source=source,
                    chunksize=args.chunksize,
                    debug_max_chunks=args.debug_max_chunks,
                    source_stats=source_stats,
                    tab_counters=tab_counters,
                    verbose=args.verbose,
                )

    by_source_df, overall_df = build_result_dataframe(tab_counters, include_all_selected=True)
    rate_delta_df = build_rate_delta_table(by_source_df)

    output_dir = Path(args.output_dir)
    fig_path = plot_click_rate_by_source(
        by_source_df=by_source_df,
        output_path=output_dir / "tab_click_rate_by_source.png",
    )

    runtime_seconds = time.time() - start_time
    logger.info(
        "统计完成 | mode=%s | runtime=%.3fs | total_valid_samples=%d",
        args.mode,
        runtime_seconds,
        int(overall_df["sample_count"].sum()) if not overall_df.empty else 0,
    )

    return (
        by_source_df,
        overall_df,
        source_stats,
        source_files_str,
        rate_delta_df,
        fig_path,
        runtime_seconds,
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("启动 tab click 统计 | mode=%s", args.mode)

    try:
        (
            by_source_df,
            overall_df,
            source_stats,
            source_files,
            rate_delta_df,
            figure_path,
            runtime_seconds,
        ) = run_statistics(args)
    except Exception as exc:
        logger.error("统计失败: %s", exc)
        raise SystemExit(1) from exc

    by_source_csv = output_dir / "tab_click_stats_by_source.csv"
    overall_csv = output_dir / "tab_click_stats_overall.csv"
    report_md = output_dir / "tab_click_stats_report.md"
    summary_json = output_dir / "tab_click_stats_summary.json"

    by_source_df.to_csv(by_source_csv, index=False, encoding="utf-8-sig")
    overall_df.to_csv(overall_csv, index=False, encoding="utf-8-sig")
    logger.info("已输出 CSV: %s", by_source_csv)
    logger.info("已输出 CSV: %s", overall_csv)

    build_markdown_report(
        output_path=report_md,
        mode=args.mode,
        args=args,
        source_files=source_files,
        by_source_df=by_source_df,
        overall_df=overall_df,
        source_stats=source_stats,
        runtime_seconds=runtime_seconds,
        rate_delta_df=rate_delta_df,
        figure_path=figure_path,
    )

    write_summary_json(
        output_path=summary_json,
        mode=args.mode,
        args=args,
        source_files=source_files,
        source_stats=source_stats,
        by_source_df=by_source_df,
        overall_df=overall_df,
        runtime_seconds=runtime_seconds,
    )

    logger.info("全部输出完成，目录: %s", output_dir)


if __name__ == "__main__":
    main()

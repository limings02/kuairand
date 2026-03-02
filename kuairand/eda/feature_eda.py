from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import normalize_path_for_sql, safe_div, write_csv

try:
    import duckdb

    HAS_DUCKDB = True
except Exception:
    HAS_DUCKDB = False


@dataclass
class FeatureAnalysisResult:
    """
    特征分析结果容器。
    """

    section_name: str
    table_paths: Dict[str, str]
    figure_paths: Dict[str, str]
    notes: List[str]


def _setup_cn_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "sans-serif",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _read_concat(paths: List[str]) -> pd.DataFrame:
    """
    读取并拼接多个 CSV（适合用户/视频特征表这类中等规模数据）。
    """
    if not paths:
        return pd.DataFrame()
    dfs = []
    for p in paths:
        dfs.append(pd.read_csv(p, low_memory=False))
    return pd.concat(dfs, axis=0, ignore_index=True, sort=False)


def _save_bar_from_counts(counts: pd.Series, path: str, title: str, xlabel: str, ylabel: str) -> None:
    if counts.empty:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _setup_cn_font()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    if len(counts) > 20:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_hist(series: pd.Series, path: str, title: str, xlabel: str) -> None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _setup_cn_font()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(s, bins=60, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("频数")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyze_user_features(
    user_paths: List[str],
    out_tables_dir: str,
    out_figures_dir: str,
) -> FeatureAnalysisResult:
    """
    D1: user_features 分析。
    """
    notes: List[str] = []
    table_paths: Dict[str, str] = {}
    figure_paths: Dict[str, str] = {}

    if not user_paths:
        notes.append("缺少 user_features 文件。")
        return FeatureAnalysisResult("user_features", table_paths, figure_paths, notes)

    df = _read_concat(user_paths)
    if df.empty:
        notes.append("user_features 文件为空。")
        return FeatureAnalysisResult("user_features", table_paths, figure_paths, notes)

    miss_df = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": [int(df[c].isna().sum()) for c in df.columns],
            "missing_ratio": [float(df[c].isna().mean()) for c in df.columns],
        }
    )
    table_paths["missing"] = write_csv(miss_df, f"{out_tables_dir}/user_features_missing.csv")

    card_rows = []
    for c in df.columns:
        card_rows.append(
            {
                "column": c,
                "non_null_count": int(df[c].notna().sum()),
                "unique_count": int(df[c].nunique(dropna=True)),
            }
        )
    card_df = pd.DataFrame(card_rows).sort_values("unique_count", ascending=False)
    table_paths["cardinality"] = write_csv(card_df, f"{out_tables_dir}/user_features_cardinality.csv")

    onehot_cols = [c for c in df.columns if c.startswith("onehot_feat")]
    if onehot_cols:
        sparse_rows = []
        for c in onehot_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            zero_ratio = float((s.fillna(0) == 0).mean())
            nonzero_ratio = float((s.fillna(0) != 0).mean())
            sparse_rows.append(
                {
                    "column": c,
                    "missing_ratio": float(s.isna().mean()),
                    "zero_ratio": zero_ratio,
                    "non_zero_ratio": nonzero_ratio,
                    "unique_count": int(df[c].nunique(dropna=True)),
                }
            )
        onehot_sparse_df = pd.DataFrame(sparse_rows).sort_values("column")
        table_paths["onehot_sparsity"] = write_csv(
            onehot_sparse_df, f"{out_tables_dir}/user_onehot_sparsity.csv"
        )

        combo = df[onehot_cols].astype(str).agg("|".join, axis=1)
        combo_df = combo.value_counts().head(20).reset_index()
        combo_df.columns = ["onehot_combo", "count"]
        table_paths["onehot_top_combo"] = write_csv(
            combo_df, f"{out_tables_dir}/user_onehot_top_combinations.csv"
        )

        _save_bar_from_counts(
            onehot_sparse_df.set_index("column")["non_zero_ratio"],
            path=f"{out_figures_dir}/user_onehot_nonzero_ratio.png",
            title="onehot_feat 非零占比",
            xlabel="特征列",
            ylabel="非零比例",
        )
        figure_paths["onehot_nonzero_ratio"] = f"{out_figures_dir}/user_onehot_nonzero_ratio.png"
    else:
        notes.append("未找到 onehot_feat0..17 列。")

    # 用户活跃度与注册天数分布
    if "user_active_degree" in df.columns:
        vc = df["user_active_degree"].value_counts(dropna=False)
        _save_bar_from_counts(
            vc,
            path=f"{out_figures_dir}/user_active_degree_dist.png",
            title="user_active_degree 分布",
            xlabel="活跃度分桶",
            ylabel="人数",
        )
        figure_paths["user_active_degree_dist"] = f"{out_figures_dir}/user_active_degree_dist.png"

    if "register_days_range" in df.columns:
        vc = df["register_days_range"].value_counts(dropna=False)
        _save_bar_from_counts(
            vc,
            path=f"{out_figures_dir}/user_register_days_range_dist.png",
            title="register_days_range 分布",
            xlabel="注册天数分桶",
            ylabel="人数",
        )
        figure_paths["register_days_range_dist"] = f"{out_figures_dir}/user_register_days_range_dist.png"

    return FeatureAnalysisResult("user_features", table_paths, figure_paths, notes)


def analyze_video_basic(
    video_basic_paths: List[str],
    out_tables_dir: str,
    out_figures_dir: str,
) -> FeatureAnalysisResult:
    """
    D2: video_features_basic 分析。
    """
    notes: List[str] = []
    table_paths: Dict[str, str] = {}
    figure_paths: Dict[str, str] = {}

    if not video_basic_paths:
        notes.append("缺少 video_features_basic 文件。")
        return FeatureAnalysisResult("video_basic", table_paths, figure_paths, notes)

    df = _read_concat(video_basic_paths)
    if df.empty:
        notes.append("video_features_basic 文件为空。")
        return FeatureAnalysisResult("video_basic", table_paths, figure_paths, notes)

    card = []
    for c in ["author_id", "music_id", "video_type", "tag"]:
        if c in df.columns:
            card.append({"column": c, "unique_count": int(df[c].nunique(dropna=True))})
    card_df = pd.DataFrame(card)
    table_paths["cardinality"] = write_csv(card_df, f"{out_tables_dir}/video_basic_cardinality.csv")

    for col in ["video_duration", "server_width", "server_height"]:
        if col in df.columns:
            summary = pd.DataFrame(
                [
                    {
                        "column": col,
                        "min": pd.to_numeric(df[col], errors="coerce").min(),
                        "p50": pd.to_numeric(df[col], errors="coerce").quantile(0.5),
                        "p90": pd.to_numeric(df[col], errors="coerce").quantile(0.9),
                        "max": pd.to_numeric(df[col], errors="coerce").max(),
                    }
                ]
            )
            table_paths[f"{col}_summary"] = write_csv(
                summary, f"{out_tables_dir}/video_basic_{col}_summary.csv"
            )
            _save_hist(
                pd.to_numeric(df[col], errors="coerce"),
                path=f"{out_figures_dir}/video_basic_{col}_hist.png",
                title=f"{col} 分布",
                xlabel=col,
            )
            figure_paths[f"{col}_hist"] = f"{out_figures_dir}/video_basic_{col}_hist.png"

    if "video_type" in df.columns:
        vc = df["video_type"].value_counts(dropna=False)
        type_df = vc.reset_index()
        type_df.columns = ["video_type", "count"]
        type_df["ratio"] = type_df["count"] / type_df["count"].sum()
        table_paths["video_type_ratio"] = write_csv(
            type_df, f"{out_tables_dir}/video_basic_type_ratio.csv"
        )
        _save_bar_from_counts(
            vc,
            path=f"{out_figures_dir}/video_basic_type_ratio.png",
            title="AD vs NORMAL 占比",
            xlabel="video_type",
            ylabel="数量",
        )
        figure_paths["video_type_ratio"] = f"{out_figures_dir}/video_basic_type_ratio.png"

    if "tag" in df.columns:
        def _tag_count(x: object) -> int:
            if pd.isna(x):
                return 0
            s = str(x).strip()
            if not s:
                return 0
            if "|" in s:
                return len([t for t in s.split("|") if t])
            if "," in s:
                return len([t for t in s.split(",") if t])
            return 1

        tag_cnt = df["tag"].apply(_tag_count)
        tag_dist = tag_cnt.value_counts().sort_index().reset_index()
        tag_dist.columns = ["tag_num", "video_count"]
        table_paths["tag_count_dist"] = write_csv(
            tag_dist, f"{out_tables_dir}/video_basic_tag_count_distribution.csv"
        )
        _save_bar_from_counts(
            tag_cnt.value_counts().sort_index(),
            path=f"{out_figures_dir}/video_basic_tag_count_dist.png",
            title="每视频 tag 数量分布",
            xlabel="tag 数量",
            ylabel="视频数",
        )
        figure_paths["tag_count_dist"] = f"{out_figures_dir}/video_basic_tag_count_dist.png"

    return FeatureAnalysisResult("video_basic", table_paths, figure_paths, notes)


def analyze_video_statistic(
    video_stat_paths: List[str],
    standard_log_paths: List[str],
    out_tables_dir: str,
    out_figures_dir: str,
    engine: str,
) -> FeatureAnalysisResult:
    """
    D3: video_features_statistic 分析 + 与日志曝光/点击相关性。
    """
    notes: List[str] = []
    table_paths: Dict[str, str] = {}
    figure_paths: Dict[str, str] = {}

    if not video_stat_paths:
        notes.append("缺少 video_features_statistic 文件。")
        return FeatureAnalysisResult("video_statistic", table_paths, figure_paths, notes)

    if not HAS_DUCKDB or (engine or "duckdb").lower() != "duckdb":
        notes.append("video_statistic 相关性分析需要 duckdb，当前已跳过，仅输出缺失说明。")
        return FeatureAnalysisResult("video_statistic", table_paths, figure_paths, notes)

    con = duckdb.connect(database=":memory:")
    stat_files_sql = "[" + ", ".join([f"'{normalize_path_for_sql(p)}'" for p in video_stat_paths]) + "]"
    con.execute(
        f"""
        CREATE OR REPLACE VIEW video_stat AS
        SELECT * FROM read_csv_auto(
            {stat_files_sql},
            HEADER=TRUE,
            UNION_BY_NAME=TRUE,
            IGNORE_ERRORS=TRUE
        )
        """
    )
    stat_cols = con.execute("DESCRIBE video_stat").fetchdf()["column_name"].tolist()
    if "video_id" not in stat_cols:
        notes.append("video_statistic 中缺少 video_id，无法做相关性。")
        con.close()
        return FeatureAnalysisResult("video_statistic", table_paths, figure_paths, notes)

    # 选取数值列做分布与相关（限制数量，避免报表过长）
    numeric_cols = []
    for c in stat_cols:
        if c == "video_id":
            continue
        tp = con.execute(f"DESCRIBE video_stat").fetchdf()
        trow = tp[tp["column_name"] == c]
        if trow.empty:
            continue
        t = str(trow.iloc[0]["column_type"]).upper()
        if any(k in t for k in ["INT", "DOUBLE", "FLOAT", "DECIMAL", "BIGINT"]):
            numeric_cols.append(c)
    numeric_cols = numeric_cols[:30]

    summary_rows = []
    for c in numeric_cols:
        row = con.execute(
            f"""
            SELECT
                MIN(CAST({c} AS DOUBLE)) AS min_v,
                quantile_cont(CAST({c} AS DOUBLE), 0.5) AS p50_v,
                quantile_cont(CAST({c} AS DOUBLE), 0.9) AS p90_v,
                MAX(CAST({c} AS DOUBLE)) AS max_v
            FROM video_stat
            WHERE {c} IS NOT NULL
            """
        ).fetchone()
        summary_rows.append(
            {"column": c, "min": row[0], "p50": row[1], "p90": row[2], "max": row[3]}
        )
    summary_df = pd.DataFrame(summary_rows)
    table_paths["numeric_summary"] = write_csv(
        summary_df, f"{out_tables_dir}/video_statistic_numeric_summary.csv"
    )

    # 与 standard log 曝光/点击的相关性
    if standard_log_paths:
        log_files_sql = "[" + ", ".join([f"'{normalize_path_for_sql(p)}'" for p in standard_log_paths]) + "]"
        con.execute(
            f"""
            CREATE OR REPLACE VIEW log_std AS
            SELECT * FROM read_csv_auto(
                {log_files_sql},
                HEADER=TRUE,
                UNION_BY_NAME=TRUE,
                IGNORE_ERRORS=TRUE
            )
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE log_video_stats AS
            SELECT
                video_id,
                COUNT(*) AS exposures,
                SUM(CASE WHEN CAST(is_click AS DOUBLE) > 0 THEN 1 ELSE 0 END) AS clicks
            FROM log_std
            GROUP BY 1
            """
        )
        corr_rows = []
        for c in numeric_cols:
            corr = con.execute(
                f"""
                SELECT
                    corr(CAST(s.{c} AS DOUBLE), CAST(l.exposures AS DOUBLE)) AS corr_exposure,
                    corr(CAST(s.{c} AS DOUBLE), CAST(l.clicks AS DOUBLE)) AS corr_click
                FROM video_stat s
                INNER JOIN log_video_stats l USING(video_id)
                WHERE s.{c} IS NOT NULL
                """
            ).fetchone()
            corr_rows.append(
                {"column": c, "corr_exposure": corr[0], "corr_click": corr[1]}
            )
        corr_df = pd.DataFrame(corr_rows).sort_values(
            "corr_exposure", key=lambda s: s.abs(), ascending=False
        )
        table_paths["corr_with_log"] = write_csv(
            corr_df, f"{out_tables_dir}/video_statistic_corr_with_log.csv"
        )

        top_corr = corr_df.head(15)
        if not top_corr.empty:
            _setup_cn_font()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(top_corr["column"].astype(str), top_corr["corr_exposure"])
            ax.set_title("video_statistic 与曝光相关性（Top15）")
            ax.set_xlabel("字段")
            ax.set_ylabel("Pearson Corr(exposure)")
            ax.grid(axis="y", alpha=0.25)
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig.savefig(f"{out_figures_dir}/video_statistic_corr_exposure_top15.png", dpi=150)
            plt.close(fig)
            figure_paths["corr_exposure_top15"] = (
                f"{out_figures_dir}/video_statistic_corr_exposure_top15.png"
            )
    else:
        notes.append("缺少 standard log，无法计算 video_statistic 与曝光/点击相关性。")

    con.close()
    return FeatureAnalysisResult("video_statistic", table_paths, figure_paths, notes)

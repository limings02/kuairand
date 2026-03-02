from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import (
    bootstrap_ctr_diff_from_counts,
    calc_gini,
    normalize_path_for_sql,
    safe_div,
    two_proportion_ztest,
    write_csv,
)

try:
    import duckdb

    HAS_DUCKDB = True
except Exception:
    HAS_DUCKDB = False


@dataclass
class LogAnalysisResult:
    """
    单类日志分析结果。
    """

    name: str
    files: List[str]
    basic_metrics: Dict[str, float]
    table_paths: Dict[str, str]
    figure_paths: Dict[str, str]
    notes: List[str]


def _setup_cn_font() -> None:
    """设置 matplotlib 中文字体。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "sans-serif",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _plot_line(df: pd.DataFrame, x: str, y: str, path: str, title: str, xlabel: str, ylabel: str) -> None:
    """保存折线图。"""
    if df.empty:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _setup_cn_font()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df[x].astype(str), df[y], marker="o", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if len(df) > 20:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_bar(df: pd.DataFrame, x: str, y: str, path: str, title: str, xlabel: str, ylabel: str) -> None:
    """保存柱状图。"""
    if df.empty:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _setup_cn_font()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df[x].astype(str), df[y])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    if len(df) > 20:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_hist(series: pd.Series, path: str, title: str, xlabel: str, log_x: bool = False) -> None:
    """保存直方图。"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _setup_cn_font()
    fig, ax = plt.subplots(figsize=(8, 4))
    if log_x:
        ax.hist(np.log1p(s), bins=60, alpha=0.85)
        ax.set_xlabel(f"log1p({xlabel})")
    else:
        ax.hist(s, bins=60, alpha=0.85)
        ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_ylabel("频数")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _duckdb_file_list(files: List[str]) -> str:
    """把文件列表渲染为 DuckDB read_csv_auto 的参数字符串。"""
    return "[" + ", ".join([f"'{normalize_path_for_sql(p)}'" for p in files]) + "]"


def _table_cols(con: "duckdb.DuckDBPyConnection", table: str) -> List[str]:
    """读取表字段名。"""
    return con.execute(f"DESCRIBE {table}").fetchdf()["column_name"].tolist()


def _empty_df(columns: List[str]) -> pd.DataFrame:
    """创建空 DataFrame。"""
    return pd.DataFrame(columns=columns)


def _analyze_log_duckdb(
    log_files: List[str],
    log_name: str,
    out_tables_dir: str,
    out_figures_dir: str,
    sample_users: Optional[int],
    seed: int,
) -> LogAnalysisResult:
    """
    DuckDB 日志分析主流程。
    """
    notes: List[str] = []
    table_paths: Dict[str, str] = {}
    figure_paths: Dict[str, str] = {}

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")

    source_view = f"{log_name}_source"
    con.execute(
        f"""
        CREATE OR REPLACE VIEW {source_view} AS
        SELECT * FROM read_csv_auto(
            {_duckdb_file_list(log_files)},
            HEADER=TRUE,
            UNION_BY_NAME=TRUE,
            IGNORE_ERRORS=TRUE
        )
        """
    )
    source_cols = set(_table_cols(con, source_view))
    base_table = f"{log_name}_base"
    if sample_users and sample_users > 0 and "user_id" in source_cols:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {base_table} AS
            WITH users AS (SELECT DISTINCT user_id FROM {source_view}),
            sampled AS (
                SELECT user_id FROM users USING SAMPLE reservoir({int(sample_users)} ROWS) REPEATABLE({int(seed)})
            )
            SELECT s.* FROM {source_view} s INNER JOIN sampled u USING(user_id)
            """
        )
        notes.append(f"已按 user_id 抽样 sample_users={sample_users}。")
    else:
        con.execute(f"CREATE OR REPLACE TEMP TABLE {base_table} AS SELECT * FROM {source_view}")

    cols = set(_table_cols(con, base_table))
    click_val = "CASE WHEN CAST(is_click AS DOUBLE) > 0 THEN 1 ELSE 0 END" if "is_click" in cols else "0"
    click_avg = "AVG(CAST(is_click AS DOUBLE))" if "is_click" in cols else "NULL"
    user_cnt = "COUNT(DISTINCT user_id)" if "user_id" in cols else "NULL"
    video_cnt = "COUNT(DISTINCT video_id)" if "video_id" in cols else "NULL"

    date_expr = "CAST(date AS BIGINT)" if "date" in cols else (
        "CAST(strftime(to_timestamp(CAST(time_ms AS DOUBLE)/1000.0), '%Y%m%d') AS BIGINT)" if "time_ms" in cols else "NULL"
    )
    hour_expr = "CAST(CAST(hourmin AS BIGINT)/100 AS BIGINT)" if "hourmin" in cols else (
        "CAST(strftime(to_timestamp(CAST(time_ms AS DOUBLE)/1000.0), '%H') AS BIGINT)" if "time_ms" in cols else "NULL"
    )

    # B1 基本统计
    basic_df = con.execute(
        f"""
        SELECT
            COUNT(*) AS exposure_cnt,
            {user_cnt} AS user_cnt,
            {video_cnt} AS video_cnt,
            SUM({click_val}) AS click_cnt,
            {click_avg} AS ctr,
            MIN({date_expr}) AS min_date,
            MAX({date_expr}) AS max_date
        FROM {base_table}
        """
    ).fetchdf()
    table_paths["basic"] = write_csv(basic_df, f"{out_tables_dir}/log_{log_name}_basic_stats.csv")

    if date_expr != "NULL":
        daily_df = con.execute(
            f"""
            SELECT {date_expr} AS date, COUNT(*) AS exposures, SUM({click_val}) AS clicks, {click_avg} AS ctr
            FROM {base_table}
            GROUP BY 1
            ORDER BY 1
            """
        ).fetchdf()
    else:
        daily_df = _empty_df(["date", "exposures", "clicks", "ctr"])
        notes.append("缺少 date/time_ms，无法做按天统计。")
    table_paths["daily"] = write_csv(daily_df, f"{out_tables_dir}/log_{log_name}_daily.csv")
    _plot_line(
        daily_df,
        x="date",
        y="exposures",
        path=f"{out_figures_dir}/log_{log_name}_daily_exposures.png",
        title=f"{log_name} 日曝光趋势",
        xlabel="日期",
        ylabel="曝光量",
    )
    figure_paths["daily_exposures"] = f"{out_figures_dir}/log_{log_name}_daily_exposures.png"

    if hour_expr != "NULL":
        hourly_df = con.execute(
            f"""
            SELECT {hour_expr} AS hour, COUNT(*) AS exposures, SUM({click_val}) AS clicks, {click_avg} AS ctr
            FROM {base_table}
            GROUP BY 1
            ORDER BY 1
            """
        ).fetchdf()
    else:
        hourly_df = _empty_df(["hour", "exposures", "clicks", "ctr"])
        notes.append("缺少 hourmin/time_ms，无法做按小时统计。")
    table_paths["hourly"] = write_csv(hourly_df, f"{out_tables_dir}/log_{log_name}_hourly.csv")
    _plot_bar(
        hourly_df,
        x="hour",
        y="exposures",
        path=f"{out_figures_dir}/log_{log_name}_hourly_exposures.png",
        title=f"{log_name} 小时曝光分布",
        xlabel="小时",
        ylabel="曝光量",
    )
    figure_paths["hourly_exposures"] = f"{out_figures_dir}/log_{log_name}_hourly_exposures.png"

    # B2 tab 统计 + top videos
    if "tab" in cols:
        tab_df = con.execute(
            f"""
            SELECT tab, COUNT(*) AS exposures, SUM({click_val}) AS clicks, {click_avg} AS ctr
            FROM {base_table}
            GROUP BY 1
            ORDER BY 1
            """
        ).fetchdf()
        if "user_id" in cols:
            total_users = con.execute(f"SELECT COUNT(DISTINCT user_id) FROM {base_table}").fetchone()[0]
            active_df = con.execute(
                f"SELECT tab, COUNT(DISTINCT user_id) AS active_users FROM {base_table} GROUP BY 1 ORDER BY 1"
            ).fetchdf()
            tab_df = tab_df.merge(active_df, on="tab", how="left")
            tab_df["active_user_ratio"] = tab_df["active_users"].apply(
                lambda x: safe_div(x, total_users, default=float("nan"))
            )
        else:
            tab_df["active_users"] = np.nan
            tab_df["active_user_ratio"] = np.nan
        if "video_id" in cols:
            tab_top_df = con.execute(
                f"""
                WITH grouped AS (
                    SELECT tab, video_id, COUNT(*) AS exposures
                    FROM {base_table}
                    GROUP BY 1,2
                ),
                ranked AS (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY tab ORDER BY exposures DESC) AS rk
                    FROM grouped
                )
                SELECT * FROM ranked WHERE rk <= 5 ORDER BY tab, rk
                """
            ).fetchdf()
        else:
            tab_top_df = _empty_df(["tab", "video_id", "exposures", "rk"])
    else:
        tab_df = _empty_df(["tab", "exposures", "clicks", "ctr", "active_users", "active_user_ratio"])
        tab_top_df = _empty_df(["tab", "video_id", "exposures", "rk"])
        notes.append("缺少 tab，无法完成 tab 分析。")
    table_paths["tab_stats"] = write_csv(tab_df, f"{out_tables_dir}/log_{log_name}_tab_stats.csv")
    table_paths["tab_top_videos"] = write_csv(tab_top_df, f"{out_tables_dir}/log_{log_name}_tab_top_videos.csv")
    _plot_bar(
        tab_df,
        x="tab",
        y="ctr",
        path=f"{out_figures_dir}/log_{log_name}_tab_ctr.png",
        title=f"{log_name} 各 Tab CTR",
        xlabel="tab",
        ylabel="CTR",
    )
    figure_paths["tab_ctr"] = f"{out_figures_dir}/log_{log_name}_tab_ctr.png"

    # B3 用户维度
    if "user_id" in cols:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {log_name}_user_stats AS
            SELECT user_id, COUNT(*) AS exposures, SUM({click_val}) AS clicks, {click_avg} AS ctr
            FROM {base_table}
            GROUP BY 1
            """
        )
        user_dist_df = con.execute(f"SELECT exposures, clicks, ctr FROM {log_name}_user_stats").fetchdf()
        user_sum_df = con.execute(
            f"""
            SELECT
                COUNT(*) AS user_cnt,
                MIN(exposures) AS expo_min, quantile_cont(exposures,0.5) AS expo_p50,
                quantile_cont(exposures,0.9) AS expo_p90, MAX(exposures) AS expo_max,
                MIN(clicks) AS click_min, quantile_cont(clicks,0.5) AS click_p50,
                quantile_cont(clicks,0.9) AS click_p90, MAX(clicks) AS click_max,
                MIN(ctr) AS ctr_min, quantile_cont(ctr,0.5) AS ctr_p50,
                quantile_cont(ctr,0.9) AS ctr_p90, MAX(ctr) AS ctr_max
            FROM {log_name}_user_stats
            """
        ).fetchdf()
    else:
        user_dist_df = _empty_df(["exposures", "clicks", "ctr"])
        user_sum_df = _empty_df(["user_cnt"])
        notes.append("缺少 user_id，无法完成用户维度分析。")
    table_paths["user_distribution"] = write_csv(
        user_dist_df, f"{out_tables_dir}/log_{log_name}_user_stats_distribution.csv"
    )
    table_paths["user_summary"] = write_csv(user_sum_df, f"{out_tables_dir}/log_{log_name}_user_stats_summary.csv")

    _plot_hist(
        user_dist_df["exposures"] if "exposures" in user_dist_df.columns else pd.Series(dtype=float),
        path=f"{out_figures_dir}/log_{log_name}_user_exposure_hist.png",
        title=f"{log_name} 用户曝光次数分布",
        xlabel="每用户曝光次数",
        log_x=True,
    )
    _plot_hist(
        user_dist_df["ctr"] if "ctr" in user_dist_df.columns else pd.Series(dtype=float),
        path=f"{out_figures_dir}/log_{log_name}_user_ctr_hist.png",
        title=f"{log_name} 用户 CTR 分布",
        xlabel="用户 CTR",
        log_x=False,
    )
    figure_paths["user_exposure_hist"] = f"{out_figures_dir}/log_{log_name}_user_exposure_hist.png"
    figure_paths["user_ctr_hist"] = f"{out_figures_dir}/log_{log_name}_user_ctr_hist.png"

    # B4 视频维度 + 长尾 + 集中度
    top1_share = float("nan")
    hhi = float("nan")
    gini = float("nan")
    if "video_id" in cols:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {log_name}_video_stats AS
            SELECT video_id, COUNT(*) AS exposures, SUM({click_val}) AS clicks, {click_avg} AS ctr
            FROM {base_table}
            GROUP BY 1
            """
        )
        video_sum_df = con.execute(
            f"""
            SELECT
                COUNT(*) AS video_cnt,
                MIN(exposures) AS expo_min, quantile_cont(exposures,0.5) AS expo_p50,
                quantile_cont(exposures,0.9) AS expo_p90, MAX(exposures) AS expo_max,
                MIN(clicks) AS click_min, quantile_cont(clicks,0.5) AS click_p50,
                quantile_cont(clicks,0.9) AS click_p90, MAX(clicks) AS click_max,
                MIN(ctr) AS ctr_min, quantile_cont(ctr,0.5) AS ctr_p50,
                quantile_cont(ctr,0.9) AS ctr_p90, MAX(ctr) AS ctr_max
            FROM {log_name}_video_stats
            """
        ).fetchdf()
        video_freq_df = con.execute(
            f"""
            SELECT exposures AS exposure_per_video, COUNT(*) AS video_count
            FROM {log_name}_video_stats
            GROUP BY 1
            ORDER BY 1
            """
        ).fetchdf()
        conc_df = con.execute(
            f"""
            WITH ranked AS (
                SELECT
                    exposures,
                    ROW_NUMBER() OVER (ORDER BY exposures DESC) AS rn,
                    COUNT(*) OVER () AS n_video,
                    SUM(exposures) OVER () AS total_expo
                FROM {log_name}_video_stats
            )
            SELECT
                SUM(POWER(exposures * 1.0 / total_expo, 2)) AS hhi,
                SUM(CASE WHEN rn <= CEIL(n_video * 0.01) THEN exposures ELSE 0 END) * 1.0 / MAX(total_expo) AS top1pct_exposure_share
            FROM ranked
            """
        ).fetchdf()
        hhi = float(conc_df.loc[0, "hhi"]) if not conc_df.empty else float("nan")
        top1_share = float(conc_df.loc[0, "top1pct_exposure_share"]) if not conc_df.empty else float("nan")
        # gini: 频数表近似采样
        rng = np.random.default_rng(seed)
        samples = []
        for _, r in video_freq_df.iterrows():
            expo = int(r["exposure_per_video"])
            cnt = int(r["video_count"])
            take = min(cnt, 2000)
            if cnt > take:
                _ = rng.choice(cnt, size=take, replace=False)
            samples.extend([expo] * take)
        gini = calc_gini(samples) if samples else float("nan")
    else:
        video_sum_df = _empty_df(["video_cnt"])
        video_freq_df = _empty_df(["exposure_per_video", "video_count"])
        notes.append("缺少 video_id，无法完成视频维度分析。")

    table_paths["video_summary"] = write_csv(video_sum_df, f"{out_tables_dir}/log_{log_name}_video_stats_summary.csv")
    table_paths["video_exposure_freq"] = write_csv(
        video_freq_df, f"{out_tables_dir}/log_{log_name}_video_exposure_freq.csv"
    )
    table_paths["concentration"] = write_csv(
        pd.DataFrame(
            [{"log_name": log_name, "hhi": hhi, "gini_approx": gini, "top1pct_exposure_share": top1_share}]
        ),
        f"{out_tables_dir}/log_{log_name}_video_concentration.csv",
    )

    if not video_freq_df.empty:
        _setup_cn_font()
        fig, ax = plt.subplots(figsize=(8, 4))
        x = video_freq_df["exposure_per_video"].to_numpy(dtype=float)
        y = video_freq_df["video_count"].to_numpy(dtype=float)
        mask = (x > 0) & (y > 0)
        ax.plot(np.log10(x[mask]), np.log10(y[mask]), "o", alpha=0.5)
        ax.set_title(f"{log_name} 视频长尾（log-log）")
        ax.set_xlabel("log10(视频曝光次数)")
        ax.set_ylabel("log10(视频数量)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(f"{out_figures_dir}/log_{log_name}_video_long_tail_loglog.png", dpi=150)
        plt.close(fig)
    figure_paths["video_long_tail_loglog"] = f"{out_figures_dir}/log_{log_name}_video_long_tail_loglog.png"

    # C1 序列长度（整体、按 tab）
    if "user_id" in cols:
        seq_sum_df = con.execute(
            f"""
            SELECT
                COUNT(*) AS user_cnt,
                MIN(seq_len) AS seq_len_min,
                quantile_cont(seq_len, 0.5) AS seq_len_p50,
                quantile_cont(seq_len, 0.9) AS seq_len_p90,
                MAX(seq_len) AS seq_len_max
            FROM (
                SELECT user_id, COUNT(*) AS seq_len
                FROM {base_table}
                GROUP BY 1
            )
            """
        ).fetchdf()
        if "tab" in cols:
            seq_tab_df = con.execute(
                f"""
                SELECT
                    tab,
                    COUNT(*) AS user_cnt,
                    MIN(seq_len) AS seq_len_min,
                    quantile_cont(seq_len, 0.5) AS seq_len_p50,
                    quantile_cont(seq_len, 0.9) AS seq_len_p90,
                    MAX(seq_len) AS seq_len_max
                FROM (
                    SELECT tab, user_id, COUNT(*) AS seq_len
                    FROM {base_table}
                    GROUP BY 1,2
                )
                GROUP BY 1
                ORDER BY 1
                """
            ).fetchdf()
        else:
            seq_tab_df = _empty_df(["tab", "user_cnt", "seq_len_min", "seq_len_p50", "seq_len_p90", "seq_len_max"])
    else:
        seq_sum_df = _empty_df(["user_cnt", "seq_len_min", "seq_len_p50", "seq_len_p90", "seq_len_max"])
        seq_tab_df = _empty_df(["tab", "user_cnt", "seq_len_min", "seq_len_p50", "seq_len_p90", "seq_len_max"])

    table_paths["sequence_summary"] = write_csv(
        seq_sum_df, f"{out_tables_dir}/log_{log_name}_sequence_len_summary.csv"
    )
    table_paths["sequence_tab_summary"] = write_csv(
        seq_tab_df, f"{out_tables_dir}/log_{log_name}_sequence_len_by_tab.csv"
    )

    _plot_hist(
        user_dist_df["exposures"] if "exposures" in user_dist_df.columns else pd.Series(dtype=float),
        path=f"{out_figures_dir}/log_{log_name}_sequence_len_hist.png",
        title=f"{log_name} 用户序列长度分布",
        xlabel="序列长度（曝光数）",
        log_x=True,
    )
    figure_paths["sequence_len_hist"] = f"{out_figures_dir}/log_{log_name}_sequence_len_hist.png"

    # C2 时间间隔与 session（gap > 30 min）
    if "user_id" in cols and "time_ms" in cols:
        gap_sum_df = con.execute(
            f"""
            WITH ordered AS (
                SELECT
                    user_id,
                    CAST(time_ms AS BIGINT) AS t,
                    LAG(CAST(time_ms AS BIGINT)) OVER(PARTITION BY user_id ORDER BY CAST(time_ms AS BIGINT)) AS prev_t
                FROM {base_table}
            ),
            gaps AS (
                SELECT (t - prev_t)/1000.0 AS gap_s
                FROM ordered
                WHERE prev_t IS NOT NULL AND t >= prev_t
            )
            SELECT
                COUNT(*) AS pair_cnt,
                AVG(gap_s) AS gap_mean_s,
                quantile_cont(gap_s, 0.5) AS gap_p50_s,
                quantile_cont(gap_s, 0.9) AS gap_p90_s,
                quantile_cont(gap_s, 0.99) AS gap_p99_s,
                MAX(gap_s) AS gap_max_s
            FROM gaps
            """
        ).fetchdf()
        gap_bin_df = con.execute(
            f"""
            WITH ordered AS (
                SELECT
                    user_id,
                    CAST(time_ms AS BIGINT) AS t,
                    LAG(CAST(time_ms AS BIGINT)) OVER(PARTITION BY user_id ORDER BY CAST(time_ms AS BIGINT)) AS prev_t
                FROM {base_table}
            ),
            gaps AS (
                SELECT (t - prev_t)/1000.0 AS gap_s
                FROM ordered
                WHERE prev_t IS NOT NULL AND t >= prev_t
            ),
            b AS (
                SELECT
                    CASE
                        WHEN gap_s < 1 THEN '<1s'
                        WHEN gap_s < 10 THEN '1-10s'
                        WHEN gap_s < 60 THEN '10-60s'
                        WHEN gap_s < 300 THEN '1-5min'
                        WHEN gap_s < 1800 THEN '5-30min'
                        WHEN gap_s < 7200 THEN '30-120min'
                        ELSE '>=120min'
                    END AS gap_bin
                FROM gaps
            )
            SELECT gap_bin, COUNT(*) AS cnt
            FROM b
            GROUP BY 1
            """
        ).fetchdf()
        order = ["<1s", "1-10s", "10-60s", "1-5min", "5-30min", "30-120min", ">=120min"]
        if not gap_bin_df.empty:
            gap_bin_df["ord"] = gap_bin_df["gap_bin"].map({v: i for i, v in enumerate(order)})
            gap_bin_df = gap_bin_df.sort_values("ord").drop(columns=["ord"])
        sess_df = con.execute(
            f"""
            WITH ordered AS (
                SELECT
                    user_id,
                    CAST(time_ms AS BIGINT) AS t,
                    LAG(CAST(time_ms AS BIGINT)) OVER(PARTITION BY user_id ORDER BY CAST(time_ms AS BIGINT)) AS prev_t
                FROM {base_table}
            ),
            marks AS (
                SELECT
                    user_id,
                    CASE WHEN prev_t IS NULL OR (t - prev_t) > 1800000 THEN 1 ELSE 0 END AS new_sess
                FROM ordered
            ),
            sess AS (
                SELECT user_id, SUM(new_sess) AS session_cnt
                FROM marks
                GROUP BY 1
            )
            SELECT
                COUNT(*) AS user_cnt,
                AVG(session_cnt) AS session_mean,
                quantile_cont(session_cnt, 0.5) AS session_p50,
                quantile_cont(session_cnt, 0.9) AS session_p90,
                MAX(session_cnt) AS session_max
            FROM sess
            """
        ).fetchdf()
    else:
        gap_sum_df = _empty_df(["pair_cnt", "gap_mean_s", "gap_p50_s", "gap_p90_s", "gap_p99_s", "gap_max_s"])
        gap_bin_df = _empty_df(["gap_bin", "cnt"])
        sess_df = _empty_df(["user_cnt", "session_mean", "session_p50", "session_p90", "session_max"])
        notes.append("缺少 user_id/time_ms，无法进行时间间隔和 session 分析。")

    table_paths["time_gap_summary"] = write_csv(
        gap_sum_df, f"{out_tables_dir}/log_{log_name}_time_gap_summary.csv"
    )
    table_paths["time_gap_bins"] = write_csv(
        gap_bin_df, f"{out_tables_dir}/log_{log_name}_time_gap_bins.csv"
    )
    table_paths["session_summary"] = write_csv(
        sess_df, f"{out_tables_dir}/log_{log_name}_session_summary.csv"
    )

    _plot_bar(
        gap_bin_df,
        x="gap_bin",
        y="cnt",
        path=f"{out_figures_dir}/log_{log_name}_time_gap_bins.png",
        title=f"{log_name} 相邻曝光时间间隔",
        xlabel="间隔桶",
        ylabel="样本数",
    )
    figure_paths["time_gap_bins"] = f"{out_figures_dir}/log_{log_name}_time_gap_bins.png"

    # C3 历史窗口示例
    if "user_id" in cols:
        users = con.execute(
            f"SELECT user_id FROM (SELECT DISTINCT user_id FROM {base_table}) ORDER BY hash(user_id) LIMIT 5"
        ).fetchdf()
        if not users.empty:
            user_list = ",".join([f"'{str(x)}'" for x in users['user_id'].astype(str).tolist()])
            order_expr = "CAST(time_ms AS BIGINT)" if "time_ms" in cols else ("CAST(date AS BIGINT)" if "date" in cols else "1")
            out_cols = ["user_id"] + [c for c in ["time_ms", "date", "hourmin", "video_id", "is_click", "tab"] if c in cols]
            out_sql = ", ".join(out_cols)
            hist_df = con.execute(
                f"""
                WITH x AS (
                    SELECT {out_sql},
                    ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY {order_expr}) AS seq_index
                    FROM {base_table}
                    WHERE CAST(user_id AS VARCHAR) IN ({user_list})
                )
                SELECT * FROM x WHERE seq_index <= 20 ORDER BY user_id, seq_index
                """
            ).fetchdf()
        else:
            hist_df = _empty_df(["user_id", "seq_index"])
    else:
        hist_df = _empty_df(["user_id", "seq_index"])
    table_paths["history_window"] = write_csv(
        hist_df, f"{out_tables_dir}/log_{log_name}_history_window_examples.csv"
    )

    basic = basic_df.iloc[0].to_dict() if not basic_df.empty else {}
    metrics = {
        "exposure_cnt": float(basic.get("exposure_cnt", np.nan)),
        "user_cnt": float(basic.get("user_cnt", np.nan)),
        "video_cnt": float(basic.get("video_cnt", np.nan)),
        "click_cnt": float(basic.get("click_cnt", np.nan)),
        "ctr": float(basic.get("ctr", np.nan)),
        "min_date": float(basic.get("min_date", np.nan)),
        "max_date": float(basic.get("max_date", np.nan)),
        "top1pct_exposure_share": top1_share,
        "hhi": hhi,
        "gini_approx": gini,
    }

    con.close()
    return LogAnalysisResult(
        name=log_name,
        files=log_files,
        basic_metrics=metrics,
        table_paths=table_paths,
        figure_paths=figure_paths,
        notes=notes,
    )


def _analyze_log_pandas_basic(
    log_files: List[str],
    log_name: str,
    out_tables_dir: str,
    out_figures_dir: str,
    chunksize: int,
) -> LogAnalysisResult:
    """
    pandas 分块降级分析。

    说明
    ----
    - 覆盖基础统计；
    - 复杂窗口分析（时间间隔/session）在此路径仅输出空表并给出提示。
    """
    notes = ["当前为 pandas 分块降级路径，建议使用 duckdb 获取完整分析。"]
    table_paths: Dict[str, str] = {}
    figure_paths: Dict[str, str] = {}

    expo, click = 0, 0
    user_set, video_set = set(), set()
    day_cnt, day_click = {}, {}
    hour_cnt = {}
    tab_cnt, tab_click = {}, {}

    for fp in log_files:
        for chunk in pd.read_csv(fp, chunksize=chunksize, low_memory=False):
            expo += len(chunk)
            if "is_click" in chunk.columns:
                clk = (pd.to_numeric(chunk["is_click"], errors="coerce").fillna(0) > 0).astype(int)
                click += int(clk.sum())
            else:
                clk = pd.Series([0] * len(chunk))

            if "user_id" in chunk.columns:
                user_set.update(chunk["user_id"].dropna().astype(str).tolist())
            if "video_id" in chunk.columns:
                video_set.update(chunk["video_id"].dropna().astype(str).tolist())

            if "date" in chunk.columns:
                d = chunk["date"].astype(str)
                g = pd.DataFrame({"d": d, "c": clk}).groupby("d").agg(exposures=("d", "count"), clicks=("c", "sum"))
                for k, r in g.iterrows():
                    day_cnt[k] = day_cnt.get(k, 0) + int(r["exposures"])
                    day_click[k] = day_click.get(k, 0) + int(r["clicks"])

            if "hourmin" in chunk.columns:
                h = (pd.to_numeric(chunk["hourmin"], errors="coerce") // 100).astype("Int64").dropna()
                for x, n in h.value_counts().items():
                    hour_cnt[int(x)] = hour_cnt.get(int(x), 0) + int(n)

            if "tab" in chunk.columns:
                t = chunk["tab"].astype(str)
                g = pd.DataFrame({"t": t, "c": clk}).groupby("t").agg(exposures=("t", "count"), clicks=("c", "sum"))
                for k, r in g.iterrows():
                    tab_cnt[k] = tab_cnt.get(k, 0) + int(r["exposures"])
                    tab_click[k] = tab_click.get(k, 0) + int(r["clicks"])

    basic_df = pd.DataFrame(
        [
            {
                "exposure_cnt": expo,
                "user_cnt": len(user_set),
                "video_cnt": len(video_set),
                "click_cnt": click,
                "ctr": safe_div(click, expo, default=float("nan")),
            }
        ]
    )
    table_paths["basic"] = write_csv(basic_df, f"{out_tables_dir}/log_{log_name}_basic_stats.csv")

    daily_df = pd.DataFrame(
        [{"date": d, "exposures": day_cnt[d], "clicks": day_click.get(d, 0), "ctr": safe_div(day_click.get(d, 0), day_cnt[d], np.nan)} for d in sorted(day_cnt.keys())]
    )
    table_paths["daily"] = write_csv(daily_df, f"{out_tables_dir}/log_{log_name}_daily.csv")
    _plot_line(
        daily_df,
        x="date",
        y="exposures",
        path=f"{out_figures_dir}/log_{log_name}_daily_exposures.png",
        title=f"{log_name} 日曝光趋势",
        xlabel="日期",
        ylabel="曝光量",
    )
    figure_paths["daily_exposures"] = f"{out_figures_dir}/log_{log_name}_daily_exposures.png"

    hourly_df = pd.DataFrame([{"hour": h, "exposures": hour_cnt[h]} for h in sorted(hour_cnt.keys())])
    table_paths["hourly"] = write_csv(hourly_df, f"{out_tables_dir}/log_{log_name}_hourly.csv")
    _plot_bar(
        hourly_df,
        x="hour",
        y="exposures",
        path=f"{out_figures_dir}/log_{log_name}_hourly_exposures.png",
        title=f"{log_name} 小时曝光分布",
        xlabel="小时",
        ylabel="曝光量",
    )
    figure_paths["hourly_exposures"] = f"{out_figures_dir}/log_{log_name}_hourly_exposures.png"

    tab_df = pd.DataFrame(
        [{"tab": t, "exposures": tab_cnt[t], "clicks": tab_click.get(t, 0), "ctr": safe_div(tab_click.get(t, 0), tab_cnt[t], np.nan)} for t in sorted(tab_cnt.keys())]
    )
    table_paths["tab_stats"] = write_csv(tab_df, f"{out_tables_dir}/log_{log_name}_tab_stats.csv")
    table_paths["tab_top_videos"] = write_csv(_empty_df(["tab", "video_id", "exposures", "rk"]), f"{out_tables_dir}/log_{log_name}_tab_top_videos.csv")
    _plot_bar(
        tab_df,
        x="tab",
        y="ctr",
        path=f"{out_figures_dir}/log_{log_name}_tab_ctr.png",
        title=f"{log_name} 各 Tab CTR",
        xlabel="tab",
        ylabel="CTR",
    )
    figure_paths["tab_ctr"] = f"{out_figures_dir}/log_{log_name}_tab_ctr.png"

    # 复杂统计输出空表，保证报告引用一致
    for key, cols in [
        ("user_distribution", ["exposures", "clicks", "ctr"]),
        ("user_summary", ["user_cnt"]),
        ("video_summary", ["video_cnt"]),
        ("video_exposure_freq", ["exposure_per_video", "video_count"]),
        ("concentration", ["log_name", "hhi", "gini_approx", "top1pct_exposure_share"]),
        ("sequence_summary", ["user_cnt", "seq_len_p50"]),
        ("sequence_tab_summary", ["tab", "seq_len_p50"]),
        ("time_gap_summary", ["pair_cnt", "gap_p50_s"]),
        ("time_gap_bins", ["gap_bin", "cnt"]),
        ("session_summary", ["user_cnt", "session_p50"]),
        ("history_window", ["user_id", "seq_index"]),
    ]:
        table_paths[key] = write_csv(_empty_df(cols), f"{out_tables_dir}/log_{log_name}_{key}.csv")
    notes.append("pandas 降级未执行用户/视频明细、序列时间间隔与 session 统计。")

    metrics = {
        "exposure_cnt": float(expo),
        "user_cnt": float(len(user_set)),
        "video_cnt": float(len(video_set)),
        "click_cnt": float(click),
        "ctr": float(safe_div(click, expo, default=float("nan"))),
        "min_date": float("nan"),
        "max_date": float("nan"),
        "top1pct_exposure_share": float("nan"),
        "hhi": float("nan"),
        "gini_approx": float("nan"),
    }
    return LogAnalysisResult(
        name=log_name,
        files=log_files,
        basic_metrics=metrics,
        table_paths=table_paths,
        figure_paths=figure_paths,
        notes=notes,
    )


def analyze_single_log(
    log_files: List[str],
    log_name: str,
    out_tables_dir: str,
    out_figures_dir: str,
    engine: str,
    chunksize: int,
    sample_users: Optional[int],
    seed: int,
) -> LogAnalysisResult:
    """
    单类日志统一分析入口。
    """
    if (engine or "duckdb").lower() == "duckdb" and HAS_DUCKDB:
        return _analyze_log_duckdb(
            log_files=log_files,
            log_name=log_name,
            out_tables_dir=out_tables_dir,
            out_figures_dir=out_figures_dir,
            sample_users=sample_users,
            seed=seed,
        )
    # engine=polars/pandas 或 duckdb 不可用，统一走 pandas 分块降级
    return _analyze_log_pandas_basic(
        log_files=log_files,
        log_name=log_name,
        out_tables_dir=out_tables_dir,
        out_figures_dir=out_figures_dir,
        chunksize=chunksize,
    )


def compare_standard_random(
    standard_res: Optional[LogAnalysisResult],
    random_res: Optional[LogAnalysisResult],
    out_tables_dir: str,
    out_figures_dir: str,
    seed: int,
) -> Optional[str]:
    """
    对比 standard 与 random 的 CTR 和曝光集中度。
    """
    if standard_res is None or random_res is None:
        return None

    s = standard_res.basic_metrics
    r = random_res.basic_metrics

    s_click, s_expo = int(s.get("click_cnt", 0)), int(s.get("exposure_cnt", 0))
    r_click, r_expo = int(r.get("click_cnt", 0)), int(r.get("exposure_cnt", 0))

    ci_low, ci_mid, ci_high = bootstrap_ctr_diff_from_counts(
        click_a=s_click,
        expo_a=s_expo,
        click_b=r_click,
        expo_b=r_expo,
        n_boot=3000,
        seed=seed,
    )
    z, p = two_proportion_ztest(s_click, s_expo, r_click, r_expo)

    comp_df = pd.DataFrame(
        [
            {"metric": "ctr_standard", "value": s.get("ctr", np.nan)},
            {"metric": "ctr_random", "value": r.get("ctr", np.nan)},
            {"metric": "ctr_diff_standard_minus_random", "value": s.get("ctr", np.nan) - r.get("ctr", np.nan)},
            {"metric": "ctr_diff_bootstrap_ci_low", "value": ci_low},
            {"metric": "ctr_diff_bootstrap_ci_median", "value": ci_mid},
            {"metric": "ctr_diff_bootstrap_ci_high", "value": ci_high},
            {"metric": "ctr_diff_z_value", "value": z},
            {"metric": "ctr_diff_p_value", "value": p},
            {"metric": "top1pct_exposure_share_standard", "value": s.get("top1pct_exposure_share", np.nan)},
            {"metric": "top1pct_exposure_share_random", "value": r.get("top1pct_exposure_share", np.nan)},
            {"metric": "hhi_standard", "value": s.get("hhi", np.nan)},
            {"metric": "hhi_random", "value": r.get("hhi", np.nan)},
            {"metric": "gini_standard", "value": s.get("gini_approx", np.nan)},
            {"metric": "gini_random", "value": r.get("gini_approx", np.nan)},
        ]
    )
    path = write_csv(comp_df, f"{out_tables_dir}/random_vs_standard_comparison.csv")

    plot_df = pd.DataFrame(
        [
            {"metric": "top1pct_exposure_share", "standard": s.get("top1pct_exposure_share", np.nan), "random": r.get("top1pct_exposure_share", np.nan)},
            {"metric": "hhi", "standard": s.get("hhi", np.nan), "random": r.get("hhi", np.nan)},
            {"metric": "gini_approx", "standard": s.get("gini_approx", np.nan), "random": r.get("gini_approx", np.nan)},
        ]
    )
    _setup_cn_font()
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(plot_df))
    w = 0.35
    ax.bar(x - w / 2, plot_df["standard"], w, label="standard")
    ax.bar(x + w / 2, plot_df["random"], w, label="random")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["metric"].tolist())
    ax.set_title("standard vs random 集中度对比")
    ax.set_ylabel("值")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_figures_dir}/random_vs_standard_concentration.png", dpi=150)
    plt.close(fig)

    return path

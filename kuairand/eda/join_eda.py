from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .utils import normalize_path_for_sql, write_csv

try:
    import duckdb

    HAS_DUCKDB = True
except Exception:
    HAS_DUCKDB = False


def analyze_join_success(
    log_file_map: Dict[str, List[str]],
    user_paths: List[str],
    video_basic_paths: List[str],
    video_stat_paths: List[str],
    out_tables_dir: str,
    engine: str,
) -> Dict[str, str]:
    """
    E1: Join 成功率分析。

    参数
    ----------
    log_file_map : Dict[str, List[str]]
        {'standard': [...], 'random': [...]} 格式。
    """
    out_paths: Dict[str, str] = {}

    if (engine or "duckdb").lower() == "duckdb" and HAS_DUCKDB:
        con = duckdb.connect(database=":memory:")

        if user_paths:
            con.execute(
                f"""
                CREATE OR REPLACE VIEW user_feat AS
                SELECT * FROM read_csv_auto(
                    [{", ".join([f"'{normalize_path_for_sql(p)}'" for p in user_paths])}],
                    HEADER=TRUE,
                    UNION_BY_NAME=TRUE,
                    IGNORE_ERRORS=TRUE
                )
                """
            )
        if video_basic_paths:
            con.execute(
                f"""
                CREATE OR REPLACE VIEW video_basic AS
                SELECT * FROM read_csv_auto(
                    [{", ".join([f"'{normalize_path_for_sql(p)}'" for p in video_basic_paths])}],
                    HEADER=TRUE,
                    UNION_BY_NAME=TRUE,
                    IGNORE_ERRORS=TRUE
                )
                """
            )
        if video_stat_paths:
            con.execute(
                f"""
                CREATE OR REPLACE VIEW video_stat AS
                SELECT * FROM read_csv_auto(
                    [{", ".join([f"'{normalize_path_for_sql(p)}'" for p in video_stat_paths])}],
                    HEADER=TRUE,
                    UNION_BY_NAME=TRUE,
                    IGNORE_ERRORS=TRUE
                )
                """
            )

        for log_name, files in log_file_map.items():
            if not files:
                continue
            con.execute(
                f"""
                CREATE OR REPLACE VIEW log_{log_name} AS
                SELECT * FROM read_csv_auto(
                    [{", ".join([f"'{normalize_path_for_sql(p)}'" for p in files])}],
                    HEADER=TRUE,
                    UNION_BY_NAME=TRUE,
                    IGNORE_ERRORS=TRUE
                )
                """
            )

            # 按可用表动态构建 SQL
            has_user = bool(user_paths)
            has_basic = bool(video_basic_paths)
            has_stat = bool(video_stat_paths)
            user_rate_expr = (
                "AVG(CASE WHEN uf.user_id IS NOT NULL THEN 1.0 ELSE 0.0 END)"
                if has_user
                else "NULL"
            )
            basic_rate_expr = (
                "AVG(CASE WHEN vb.video_id IS NOT NULL THEN 1.0 ELSE 0.0 END)"
                if has_basic
                else "NULL"
            )
            stat_rate_expr = (
                "AVG(CASE WHEN vs.video_id IS NOT NULL THEN 1.0 ELSE 0.0 END)"
                if has_stat
                else "NULL"
            )
            join_user = "LEFT JOIN user_feat uf ON l.user_id = uf.user_id" if has_user else ""
            join_basic = "LEFT JOIN video_basic vb ON l.video_id = vb.video_id" if has_basic else ""
            join_stat = "LEFT JOIN video_stat vs ON l.video_id = vs.video_id" if has_stat else ""

            df = con.execute(
                f"""
                SELECT
                    COUNT(*) AS exposure_cnt,
                    {user_rate_expr} AS user_join_success_rate,
                    {basic_rate_expr} AS video_basic_join_success_rate,
                    {stat_rate_expr} AS video_stat_join_success_rate
                FROM log_{log_name} l
                {join_user}
                {join_basic}
                {join_stat}
                """
            ).fetchdf()
            out_paths[f"{log_name}_join"] = write_csv(
                df, f"{out_tables_dir}/log_{log_name}_join_success.csv"
            )

        con.close()
    else:
        # pandas 降级：仅给出占位说明
        for log_name, files in log_file_map.items():
            if not files:
                continue
            df = pd.DataFrame(
                [
                    {
                        "exposure_cnt": np.nan,
                        "user_join_success_rate": np.nan,
                        "video_basic_join_success_rate": np.nan,
                        "video_stat_join_success_rate": np.nan,
                    }
                ]
            )
            out_paths[f"{log_name}_join"] = write_csv(
                df, f"{out_tables_dir}/log_{log_name}_join_success.csv"
            )

    # E2 推荐 CTR 训练 schema
    schema_df = pd.DataFrame(
        [
            {"column": "user_id", "source": "log", "role": "key/feature", "note": "用户主键"},
            {"column": "video_id", "source": "log", "role": "key/feature", "note": "视频主键"},
            {"column": "time_ms/date", "source": "log", "role": "split_key", "note": "时间切分键，避免穿越"},
            {"column": "tab", "source": "log", "role": "feature/group", "note": "场景分桶与分群评估"},
            {"column": "is_click", "source": "log", "role": "label", "note": "pointwise CTR 标签（注意语义差异）"},
            {"column": "user_*", "source": "user_features", "role": "feature", "note": "用户画像特征"},
            {"column": "video_basic_*", "source": "video_features_basic", "role": "feature", "note": "视频静态内容特征"},
            {"column": "video_stat_*", "source": "video_features_statistic", "role": "feature (慎用)", "note": "可能有时间泄漏风险"},
        ]
    )
    out_paths["recommended_schema"] = write_csv(
        schema_df, f"{out_tables_dir}/recommended_ctr_schema.csv"
    )

    # E3 负采样策略建议
    neg_df = pd.DataFrame(
        [
            {
                "method": "无需显式负采样(pointwise)",
                "applicable_to": "CTR pointwise",
                "description": "曝光未点样本天然为负例（is_click=0）",
                "bias_risk": "主要受曝光策略偏置影响",
            },
            {
                "method": "曝光未点负采样",
                "applicable_to": "ranking/pairwise",
                "description": "从同用户曝光未点击集中抽负例",
                "bias_risk": "偏向已曝光候选，受日志策略影响",
            },
            {
                "method": "Uniform 全局负采样",
                "applicable_to": "ranking/pairwise",
                "description": "从全局视频池均匀采负",
                "bias_risk": "可能产生过易负例，分布偏离线上",
            },
            {
                "method": "同类 hard negative",
                "applicable_to": "ranking/pairwise",
                "description": "按同作者/同标签等语义近邻采负",
                "bias_risk": "更难但更易引入选择偏差，需校准",
            },
        ]
    )
    out_paths["negative_sampling"] = write_csv(
        neg_df, f"{out_tables_dir}/negative_sampling_recommendations.csv"
    )

    return out_paths


from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

from kuairand.eda.data_scan import scan_data_dir
from kuairand.eda.feature_eda import analyze_user_features, analyze_video_basic, analyze_video_statistic
from kuairand.eda.join_eda import analyze_join_success
from kuairand.eda.log_eda import analyze_single_log, compare_standard_random
from kuairand.eda.notebook_builder import build_notebook
from kuairand.eda.profiling import profile_many_files
from kuairand.eda.reporting import generate_report
from kuairand.eda.utils import copy_source_files_to_report, ensure_output_dirs, write_csv


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="KuaiRand 系统性 EDA 主入口")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="KuaiRand 数据目录（可指向 data 子目录或上层目录）",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./kuairand_report",
        help="输出目录，默认 ./kuairand_report",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="duckdb",
        choices=["duckdb", "polars", "pandas"],
        help="计算引擎",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200000,
        help="pandas 分块大小（duckdb 模式下仍用于部分流程）",
    )
    parser.add_argument(
        "--sample_users",
        type=int,
        default=None,
        help="可选：按用户抽样数量（用于快速模式）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，保证可复现",
    )
    return parser.parse_args()


def main() -> int:
    """
    KuaiRand EDA 总流程。
    """
    args = parse_args()
    out_dirs = ensure_output_dirs(args.out_dir)
    global_notes: List[str] = []

    # 1) 扫描数据目录
    print("[1/8] 扫描数据目录...")
    scan_res = scan_data_dir(args.data_dir)
    inventory_path = write_csv(scan_res.inventory_df, f"{out_dirs['tables']}/file_inventory.csv")
    scan_summary_path = write_csv(scan_res.summary_df, f"{out_dirs['tables']}/file_role_summary.csv")

    role_map = scan_res.role_to_files
    log_standard_files = role_map.get("log_standard", [])
    log_random_files = role_map.get("log_random", [])
    user_files = role_map.get("user_features", [])
    video_basic_files = role_map.get("video_features_basic", [])
    video_stat_files = role_map.get("video_features_statistic", [])

    # 2) A: 全文件画像
    print("[2/8] 文件画像（行列数/缺失/候选主键/单调性）...")
    all_csv_files = scan_res.inventory_df["file_path"].tolist() if not scan_res.inventory_df.empty else []
    if all_csv_files:
        profile_summary_df, profile_detail = profile_many_files(
            all_csv_files,
            chunksize=args.chunksize,
            median_sample_size=50000,
        )
    else:
        profile_summary_df, profile_detail = pd.DataFrame(), {}
        global_notes.append("目录下未扫描到 CSV 文件。")

    profile_summary_path = write_csv(
        profile_summary_df, f"{out_dirs['tables']}/file_profiles_summary.csv"
    )

    # 每个文件落盘列级统计与候选 key 统计
    profile_note_rows = []
    for fp, prof in profile_detail.items():
        safe_name = Path(fp).name.replace(".csv", "")
        col_path = write_csv(
            prof.column_stats,
            f"{out_dirs['tables']}/profile__{safe_name}__columns.csv",
        )
        key_path = write_csv(
            prof.key_stats,
            f"{out_dirs['tables']}/profile__{safe_name}__keys.csv",
        )
        row = dict(prof.summary_row)
        row["column_profile_table"] = col_path
        row["key_profile_table"] = key_path
        profile_note_rows.append(row)

    profile_note_df = pd.DataFrame(profile_note_rows)
    profile_note_path = write_csv(
        profile_note_df, f"{out_dirs['tables']}/file_profiles_details_index.csv"
    )

    # 缺失文件说明
    if not log_standard_files:
        global_notes.append("未发现 log_standard 文件，B/C/F 中标准日志相关结论将缺失。")
    if not log_random_files:
        global_notes.append("未发现 log_random 文件，无法完成 random vs standard 对比。")
    if not user_files:
        global_notes.append("未发现 user_features 文件，用户画像对比与 join 成功率受限。")
    if not video_basic_files:
        global_notes.append("未发现 video_features_basic 文件，视频静态特征分析受限。")
    if not video_stat_files:
        global_notes.append("未发现 video_features_statistic 文件，泄漏风险实证与相关性分析受限。")

    # 3) B/C: 日志分析（standard/random）
    print("[3/8] 日志核心统计 + 序列性质分析...")
    standard_res = None
    random_res = None
    if log_standard_files:
        standard_res = analyze_single_log(
            log_files=log_standard_files,
            log_name="standard",
            out_tables_dir=out_dirs["tables"],
            out_figures_dir=out_dirs["figures"],
            engine=args.engine,
            chunksize=args.chunksize,
            sample_users=args.sample_users,
            seed=args.seed,
        )
    if log_random_files:
        random_res = analyze_single_log(
            log_files=log_random_files,
            log_name="random",
            out_tables_dir=out_dirs["tables"],
            out_figures_dir=out_dirs["figures"],
            engine=args.engine,
            chunksize=args.chunksize,
            sample_users=args.sample_users,
            seed=args.seed,
        )

    random_vs_standard_path = compare_standard_random(
        standard_res=standard_res,
        random_res=random_res,
        out_tables_dir=out_dirs["tables"],
        out_figures_dir=out_dirs["figures"],
        seed=args.seed,
    )

    # 4) D: 特征分析
    print("[4/8] 用户/视频特征分析...")
    user_feat_res = analyze_user_features(
        user_paths=user_files,
        out_tables_dir=out_dirs["tables"],
        out_figures_dir=out_dirs["figures"],
    )
    video_basic_res = analyze_video_basic(
        video_basic_paths=video_basic_files,
        out_tables_dir=out_dirs["tables"],
        out_figures_dir=out_dirs["figures"],
    )
    video_stat_res = analyze_video_statistic(
        video_stat_paths=video_stat_files,
        standard_log_paths=log_standard_files,
        out_tables_dir=out_dirs["tables"],
        out_figures_dir=out_dirs["figures"],
        engine=args.engine,
    )

    # 5) E: Join 与 schema
    print("[5/8] Join 成功率与训练 schema...")
    join_paths = analyze_join_success(
        log_file_map={"standard": log_standard_files, "random": log_random_files},
        user_paths=user_files,
        video_basic_paths=video_basic_files,
        video_stat_paths=video_stat_files,
        out_tables_dir=out_dirs["tables"],
        engine=args.engine,
    )

    # 6) Notebook
    print("[6/8] 生成 Notebook...")
    notebook_path = build_notebook(
        notebook_path=f"{out_dirs['notebooks']}/01_eda.ipynb",
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        engine=args.engine,
        chunksize=args.chunksize,
        sample_users=args.sample_users,
        seed=args.seed,
    )
    global_notes.append(f"Notebook 已生成: {notebook_path}")

    # 7) 报告
    print("[7/8] 生成 Markdown 报告...")
    report_path = generate_report(
        out_dir=args.out_dir,
        data_dir=args.data_dir,
        engine=args.engine,
        chunksize=args.chunksize,
        sample_users=args.sample_users,
        seed=args.seed,
        inventory_path=inventory_path,
        scan_summary_path=scan_summary_path,
        profile_summary_path=profile_summary_path,
        profile_note_path=profile_note_path,
        log_standard=standard_res,
        log_random=random_res,
        random_vs_standard_path=random_vs_standard_path,
        user_feat_res=user_feat_res,
        video_basic_res=video_basic_res,
        video_stat_res=video_stat_res,
        join_paths=join_paths,
        global_notes=global_notes,
    )

    # 8) 复制代码快照到报告目录
    print("[8/8] 复制代码快照...")
    copy_source_files_to_report(src_dir="src", report_src_dir=out_dirs["src"])

    print(f"\n完成。报告路径: {report_path}")
    print(f"图表目录: {out_dirs['figures']}")
    print(f"表格目录: {out_dirs['tables']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from .feature_eda import FeatureAnalysisResult
from .log_eda import LogAnalysisResult


def _rel(path: Optional[str], out_dir: str) -> str:
    """
    把绝对路径转成相对报告目录的相对路径。
    """
    if not path:
        return "（未生成）"
    try:
        return str(Path(path).resolve().relative_to(Path(out_dir).resolve()))
    except Exception:
        return str(Path(path))


def _fmt_num(x: float) -> str:
    if x is None:
        return "NA"
    try:
        if abs(float(x)) >= 1000:
            return f"{float(x):,.0f}"
        return f"{float(x):.6f}"
    except Exception:
        return str(x)


def _log_metric_block(log_res: Optional[LogAnalysisResult]) -> str:
    if log_res is None:
        return "- 未检测到该日志文件。\n"
    m = log_res.basic_metrics
    return (
        f"- 曝光量: {_fmt_num(m.get('exposure_cnt'))}\n"
        f"- 用户数: {_fmt_num(m.get('user_cnt'))}\n"
        f"- 视频数: {_fmt_num(m.get('video_cnt'))}\n"
        f"- 点击数: {_fmt_num(m.get('click_cnt'))}\n"
        f"- CTR: {_fmt_num(m.get('ctr'))}\n"
        f"- 日期范围: {_fmt_num(m.get('min_date'))} ~ {_fmt_num(m.get('max_date'))}\n"
        f"- 集中度 HHI: {_fmt_num(m.get('hhi'))}\n"
        f"- Top1% 曝光占比: {_fmt_num(m.get('top1pct_exposure_share'))}\n"
        f"- Gini(近似): {_fmt_num(m.get('gini_approx'))}\n"
    )


def _feature_paths_block(feature_res: FeatureAnalysisResult, out_dir: str) -> str:
    lines = []
    for k, v in feature_res.table_paths.items():
        lines.append(f"- 表 `{k}`: `{_rel(v, out_dir)}`")
    for k, v in feature_res.figure_paths.items():
        lines.append(f"- 图 `{k}`: `{_rel(v, out_dir)}`")
    if feature_res.notes:
        lines.append("- 备注:")
        for n in feature_res.notes:
            lines.append(f"  - {n}")
    return "\n".join(lines) if lines else "- 无输出。"


def generate_report(
    out_dir: str,
    data_dir: str,
    engine: str,
    chunksize: int,
    sample_users: Optional[int],
    seed: int,
    inventory_path: str,
    scan_summary_path: str,
    profile_summary_path: str,
    profile_note_path: str,
    log_standard: Optional[LogAnalysisResult],
    log_random: Optional[LogAnalysisResult],
    random_vs_standard_path: Optional[str],
    user_feat_res: FeatureAnalysisResult,
    video_basic_res: FeatureAnalysisResult,
    video_stat_res: FeatureAnalysisResult,
    join_paths: Dict[str, str],
    global_notes: Optional[list] = None,
) -> str:
    """
    生成 Markdown 分析报告。
    """
    out = Path(out_dir)
    report_path = out / "report.md"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    global_notes = global_notes or []

    report = f"""# KuaiRand 系统性 EDA 报告
生成时间：{now}

## 一键运行
```bash
python run_eda.py --data_dir "{data_dir}" --out_dir "{out_dir}" --engine {engine} --chunksize {chunksize} --seed {seed}
```
```bash
# 快速模式（按用户抽样）
python run_eda.py --data_dir "{data_dir}" --out_dir "{out_dir}" --engine {engine} --chunksize {chunksize} --sample_users {sample_users if sample_users else 5000} --seed {seed}
```

## 关键背景与口径声明（必须阅读）
1. KuaiRand 包含 `standard log` 与 `random log`。`random log` 来自随机干预曝光子集，可作为更接近无偏评测锚点。  
2. `is_click` 在不同 UI/场景语义可能不同：双列更接近点击，单列更接近 valid_play。将其直接解释为 CTR 标签会影响建模口径与实验结论。  
3. `video_features_statistic` 是聚合统计特征，存在时间信息泄漏风险。建议仅在训练窗口内重算或先不使用，再做消融验证。

## A. 数据总览与质量检查
- 文件清单：`{_rel(inventory_path, out_dir)}`
- 角色汇总：`{_rel(scan_summary_path, out_dir)}`
- 文件画像汇总：`{_rel(profile_summary_path, out_dir)}`
- 单调性与缺失说明：`{_rel(profile_note_path, out_dir)}`

## B. 日志核心统计
### B1-B4 Standard Log
{_log_metric_block(log_standard)}

{f"- 关键表：`{_rel(log_standard.table_paths.get('basic'), out_dir)}`、`{_rel(log_standard.table_paths.get('daily'), out_dir)}`、`{_rel(log_standard.table_paths.get('hourly'), out_dir)}`、`{_rel(log_standard.table_paths.get('tab_stats'), out_dir)}`" if log_standard else "- 关键表：未生成"}
{f"- 用户/视频分布表：`{_rel(log_standard.table_paths.get('user_summary'), out_dir)}`、`{_rel(log_standard.table_paths.get('video_summary'), out_dir)}`、`{_rel(log_standard.table_paths.get('video_exposure_freq'), out_dir)}`" if log_standard else ""}
{f"- 图：`{_rel(log_standard.figure_paths.get('daily_exposures'), out_dir)}`、`{_rel(log_standard.figure_paths.get('tab_ctr'), out_dir)}`、`{_rel(log_standard.figure_paths.get('video_long_tail_loglog'), out_dir)}`" if log_standard else ""}

### B1-B4 Random Log
{_log_metric_block(log_random)}

{f"- 关键表：`{_rel(log_random.table_paths.get('basic'), out_dir)}`、`{_rel(log_random.table_paths.get('daily'), out_dir)}`、`{_rel(log_random.table_paths.get('hourly'), out_dir)}`、`{_rel(log_random.table_paths.get('tab_stats'), out_dir)}`" if log_random else "- 关键表：未生成"}
{f"- 用户/视频分布表：`{_rel(log_random.table_paths.get('user_summary'), out_dir)}`、`{_rel(log_random.table_paths.get('video_summary'), out_dir)}`、`{_rel(log_random.table_paths.get('video_exposure_freq'), out_dir)}`" if log_random else ""}
{f"- 图：`{_rel(log_random.figure_paths.get('daily_exposures'), out_dir)}`、`{_rel(log_random.figure_paths.get('tab_ctr'), out_dir)}`、`{_rel(log_random.figure_paths.get('video_long_tail_loglog'), out_dir)}`" if log_random else ""}

### B5 Random vs Standard 对比
{f"- 对比表（含 CTR bootstrap 95% CI 与 z-test）：`{_rel(random_vs_standard_path, out_dir)}`" if random_vs_standard_path else "- 对比表：未生成（缺少 standard 或 random）"}
- 对比图：`figures/random_vs_standard_concentration.png`

## C. 序列性质（推荐/CTR 序列特征）
{f"- Standard 序列表：`{_rel(log_standard.table_paths.get('sequence_summary'), out_dir)}`、`{_rel(log_standard.table_paths.get('sequence_tab_summary'), out_dir)}`、`{_rel(log_standard.table_paths.get('time_gap_summary'), out_dir)}`、`{_rel(log_standard.table_paths.get('session_summary'), out_dir)}`、`{_rel(log_standard.table_paths.get('history_window'), out_dir)}`" if log_standard else "- Standard 序列表：未生成"}
{f"- Random 序列表：`{_rel(log_random.table_paths.get('sequence_summary'), out_dir)}`、`{_rel(log_random.table_paths.get('sequence_tab_summary'), out_dir)}`、`{_rel(log_random.table_paths.get('time_gap_summary'), out_dir)}`、`{_rel(log_random.table_paths.get('session_summary'), out_dir)}`、`{_rel(log_random.table_paths.get('history_window'), out_dir)}`" if log_random else "- Random 序列表：未生成"}
{f"- 序列图（standard）：`{_rel(log_standard.figure_paths.get('sequence_len_hist'), out_dir)}`、`{_rel(log_standard.figure_paths.get('time_gap_bins'), out_dir)}`" if log_standard else ""}
{f"- 序列图（random）：`{_rel(log_random.figure_paths.get('sequence_len_hist'), out_dir)}`、`{_rel(log_random.figure_paths.get('time_gap_bins'), out_dir)}`" if log_random else ""}

## D. 特征表分析
### D1 user_features
{_feature_paths_block(user_feat_res, out_dir)}

### D2 video_features_basic
{_feature_paths_block(video_basic_res, out_dir)}

### D3 video_features_statistic（含泄漏风险）
{_feature_paths_block(video_stat_res, out_dir)}

泄漏风险建议：
1. 若使用 `video_features_statistic`，必须以训练窗口重算或对齐特征快照时间。  
2. 初版可先不使用该类特征，建立无泄漏 baseline。  
3. 在线/离线结果显著偏离时，优先排查该类特征是否引入未来信息。

## E. Join 与建模可用性
- Join 成功率（standard）：`{_rel(join_paths.get('standard_join'), out_dir)}`
- Join 成功率（random）：`{_rel(join_paths.get('random_join'), out_dir)}`
- 推荐 CTR 训练 schema：`{_rel(join_paths.get('recommended_schema'), out_dir)}`
- 负采样策略建议：`{_rel(join_paths.get('negative_sampling'), out_dir)}`

结论：pointwise CTR 通常不需要显式负采样（`is_click=0` 即负例）；pairwise/ranking 才需要负采样，并应在报告中注明采样偏差来源。

## F. 评测与实验设计建议（必须）
1. 时间切分：使用早期 standard log 构建历史和训练，后期 standard 做验证；random log 作为更可信离线评测集之一。  
2. 指标：pointwise CTR 推荐 AUC / LogLoss；并按 tab、用户活跃度分桶上报。  
3. 风险点：  
   - `is_click` 语义在不同 UI 下不完全一致；  
   - random 子集更稀疏，估计方差更大；  
   - `video_features_statistic` 可能泄漏；  
   - 候选池支持集与线上真实召回池可能不一致。

## 附：运行说明与复现
- 运行参数：engine={engine}, chunksize={chunksize}, sample_users={sample_users}, seed={seed}
- Notebook：`notebooks/01_eda.ipynb`

## 备注
"""
    if log_standard and log_standard.notes:
        report += "\n- standard 备注:\n"
        for n in log_standard.notes:
            report += f"  - {n}\n"
    if log_random and log_random.notes:
        report += "\n- random 备注:\n"
        for n in log_random.notes:
            report += f"  - {n}\n"
    if global_notes:
        report += "\n- 全局备注:\n"
        for n in global_notes:
            report += f"  - {n}\n"

    report_path.write_text(report, encoding="utf-8")
    return str(report_path)


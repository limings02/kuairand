# KuaiRand 系统性 EDA 报告
生成时间：2026-02-23 16:20:23

## 一键运行
```bash
python run_eda.py --data_dir "KuaiRand-27K/KuaiRand-27K/data" --out_dir "./kuairand_report" --engine duckdb --chunksize 200000 --seed 42
```
```bash
# 快速模式（按用户抽样）
python run_eda.py --data_dir "KuaiRand-27K/KuaiRand-27K/data" --out_dir "./kuairand_report" --engine duckdb --chunksize 200000 --sample_users 5000 --seed 42
```

## 关键背景与口径声明（必须阅读）
1. KuaiRand 包含 `standard log` 与 `random log`。`random log` 来自随机干预曝光子集，可作为更接近无偏评测锚点。  
2. `is_click` 在不同 UI/场景语义可能不同：双列更接近点击，单列更接近 valid_play。将其直接解释为 CTR 标签会影响建模口径与实验结论。  
3. `video_features_statistic` 是聚合统计特征，存在时间信息泄漏风险。建议仅在训练窗口内重算或先不使用，再做消融验证。

## A. 数据总览与质量检查
- 文件清单：`tables\file_inventory.csv`
- 角色汇总：`tables\file_role_summary.csv`
- 文件画像汇总：`tables\file_profiles_summary.csv`
- 单调性与缺失说明：`tables\file_profiles_details_index.csv`

## B. 日志核心统计
### B1-B4 Standard Log
- 曝光量: 322,278,385
- 用户数: 27,285
- 视频数: 32,038,693
- 点击数: 122,052,542
- CTR: 0.378718
- 日期范围: 20,220,408 ~ 20,220,508
- 集中度 HHI: 0.000001
- Top1% 曝光占比: 0.441106
- Gini(近似): 0.531400


- 关键表：`tables\log_standard_basic_stats.csv`、`tables\log_standard_daily.csv`、`tables\log_standard_hourly.csv`、`tables\log_standard_tab_stats.csv`
- 用户/视频分布表：`tables\log_standard_user_stats_summary.csv`、`tables\log_standard_video_stats_summary.csv`、`tables\log_standard_video_exposure_freq.csv`
- 图：`figures\log_standard_daily_exposures.png`、`figures\log_standard_tab_ctr.png`、`figures\log_standard_video_long_tail_loglog.png`

### B1-B4 Random Log
- 曝光量: 1,186,059
- 用户数: 27,285
- 视频数: 7,583
- 点击数: 208,934
- CTR: 0.176158
- 日期范围: 20,220,422 ~ 20,220,508
- 集中度 HHI: 0.000137
- Top1% 曝光占比: 0.012701
- Gini(近似): 0.080809


- 关键表：`tables\log_random_basic_stats.csv`、`tables\log_random_daily.csv`、`tables\log_random_hourly.csv`、`tables\log_random_tab_stats.csv`
- 用户/视频分布表：`tables\log_random_user_stats_summary.csv`、`tables\log_random_video_stats_summary.csv`、`tables\log_random_video_exposure_freq.csv`
- 图：`figures\log_random_daily_exposures.png`、`figures\log_random_tab_ctr.png`、`figures\log_random_video_long_tail_loglog.png`

### B5 Random vs Standard 对比
- 对比表（含 CTR bootstrap 95% CI 与 z-test）：`tables\random_vs_standard_comparison.csv`
- 对比图：`figures/random_vs_standard_concentration.png`

## C. 序列性质（推荐/CTR 序列特征）
- Standard 序列表：`tables\log_standard_sequence_len_summary.csv`、`tables\log_standard_sequence_len_by_tab.csv`、`tables\log_standard_time_gap_summary.csv`、`tables\log_standard_session_summary.csv`、`tables\log_standard_history_window_examples.csv`
- Random 序列表：`tables\log_random_sequence_len_summary.csv`、`tables\log_random_sequence_len_by_tab.csv`、`tables\log_random_time_gap_summary.csv`、`tables\log_random_session_summary.csv`、`tables\log_random_history_window_examples.csv`
- 序列图（standard）：`figures\log_standard_sequence_len_hist.png`、`figures\log_standard_time_gap_bins.png`
- 序列图（random）：`figures\log_random_sequence_len_hist.png`、`figures\log_random_time_gap_bins.png`

## D. 特征表分析
### D1 user_features
- 表 `missing`: `tables\user_features_missing.csv`
- 表 `cardinality`: `tables\user_features_cardinality.csv`
- 表 `onehot_sparsity`: `tables\user_onehot_sparsity.csv`
- 表 `onehot_top_combo`: `tables\user_onehot_top_combinations.csv`
- 图 `onehot_nonzero_ratio`: `figures\user_onehot_nonzero_ratio.png`
- 图 `user_active_degree_dist`: `figures\user_active_degree_dist.png`
- 图 `register_days_range_dist`: `figures\user_register_days_range_dist.png`

### D2 video_features_basic
- 表 `cardinality`: `tables\video_basic_cardinality.csv`
- 表 `video_duration_summary`: `tables\video_basic_video_duration_summary.csv`
- 表 `server_width_summary`: `tables\video_basic_server_width_summary.csv`
- 表 `server_height_summary`: `tables\video_basic_server_height_summary.csv`
- 表 `video_type_ratio`: `tables\video_basic_type_ratio.csv`
- 表 `tag_count_dist`: `tables\video_basic_tag_count_distribution.csv`
- 图 `video_duration_hist`: `figures\video_basic_video_duration_hist.png`
- 图 `server_width_hist`: `figures\video_basic_server_width_hist.png`
- 图 `server_height_hist`: `figures\video_basic_server_height_hist.png`
- 图 `video_type_ratio`: `figures\video_basic_type_ratio.png`
- 图 `tag_count_dist`: `figures\video_basic_tag_count_dist.png`

### D3 video_features_statistic（含泄漏风险）
- 表 `numeric_summary`: `tables\video_statistic_numeric_summary.csv`
- 表 `corr_with_log`: `tables\video_statistic_corr_with_log.csv`
- 图 `corr_exposure_top15`: `figures\video_statistic_corr_exposure_top15.png`

泄漏风险建议：
1. 若使用 `video_features_statistic`，必须以训练窗口重算或对齐特征快照时间。  
2. 初版可先不使用该类特征，建立无泄漏 baseline。  
3. 在线/离线结果显著偏离时，优先排查该类特征是否引入未来信息。

## E. Join 与建模可用性
- Join 成功率（standard）：`tables\log_standard_join_success.csv`
- Join 成功率（random）：`tables\log_random_join_success.csv`
- 推荐 CTR 训练 schema：`tables\recommended_ctr_schema.csv`
- 负采样策略建议：`tables\negative_sampling_recommendations.csv`

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
- 运行参数：engine=duckdb, chunksize=200000, sample_users=None, seed=42
- Notebook：`notebooks/01_eda.ipynb`

## 备注

- 全局备注:
  - Notebook 已生成: kuairand_report\notebooks\01_eda.ipynb

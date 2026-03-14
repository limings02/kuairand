# 改进跨域多场景建模的快手精排推荐（KuaiRand）

基于 KuaiRand `standard log` / `random log` 的多场景精排实验仓库。主线是基于一个 DIN baseline，在同一套训练和评估接口上，逐步把 ADS 风格的跨域多兴趣建模、Transformer 兴趣融合、MBCNet 和 PPNet 接到 DIN 上。

## 最核心的创新

- 任务：快手短视频精排，重点处理多 tab / 多场景下的跨域信息共享和用户多兴趣建模。
- baseline：DIN，README 全文按 `is_click` 标签口径描述，输入包含 50 长度行为序列、候选 item、场景特征、用户画像特征。
- 升级路线：`DIN -> PSRG/PCRG -> TransformerFusion -> MBCNet -> PPNet`
- 代码入口：训练在 `src/main_train_din.py`，模型在 `src/models/din.py` 和 `src/models/modules/*`，实验脚本在 `scripts/`。
- 结果口径：按项目汇报记录，最终平均 GAUC 约为 `0.692`，相对 baseline 累计提升接近 `2.7%`。仓库已经给出对应配置、模块实现和测试；完整 run 级实验表目前没有全部随仓库提交。

## 项目概览

| 维度 | 当前仓库里的证据 |
| --- | --- |
| 用户规模 | `user_features_27k.csv` 对应 `27,285` 个用户 |
| 行为规模 | `reports/eda_report/report.md` 统计出 `standard log` 有 `322,278,385` 条曝光，`random log` 有 `1,186,059` 条曝光 |
| 训练切分 | `output/meta/split_summary.json` 中有 `train / val / test_standard / test_random` 四个切片 |
| 序列长度 | `configs/*` 和 `output/meta/field_schema.json` 都写死了 `max_hist_len=50` |
| 候选空间 | `output/meta/field_schema.json` 里 `video_id` vocab 为 `32,038,727`，`author_id` vocab 为 `8,839,737` |
| 场景信息 | `tab`、`hour_of_day`、`day_of_week`、`is_weekend` 都进入训练字段 |

这里说的“域”，主要是 tab / 场景上下文 / 曝光来源这类场景域，不是把两个完全不同业务线硬拼在一起。真正难的地方在于：

- 同一个用户在不同 tab 下兴趣分布并不一样，单一兴趣向量很容易把信息压扁。
- `standard log` 适合拿来训练，但带分发偏置；`random log` 更适合作为离线对照切片，不能混着理解。
- 词表大、样本多、历史序列长，训练和评估都得考虑内存和显存约束。

## 方法演进 / 模型升级路线

下面这条线，是当前仓库真正落地的模型演进路径。

### 0. DIN baseline

- 想解决什么：先把“候选 item 和用户历史的匹配”打通，建立一个可复现的精排底座。
- 现在怎么做：`src/models/din.py` 里保留了标准 DIN attention，视频和作者 embedding 在历史序列与候选侧共享；用户 sparse / dense 特征、场景 sparse 特征和候选侧特征一起进 DNN head。
- 工程取舍：由于 `video_id` 和 `author_id` 词表很大，默认走 `HashEmbedding`，避免直接上全量 embedding 把显存打满。
- 对应代码：
  - 配置：`configs/train_din_mem16gb.yaml`
  - 训练入口：`src/main_train_din.py`
  - 主模型：`src/models/din.py`
  - 数据读取：`src/datasets/parquet_iterable_dataset.py`
  - 冒烟检查：`scripts/smoke_test_din.py`

### 1. ADS：PSRG + PCRG

- 想解决什么：普通 DIN 本质上还是“单 query 对整段历史做一次聚合”，对多场景、多兴趣用户不太够。尤其在多 tab 场景里，同一段历史在不同场景下语义不该完全一样。
- 核心做法：
  - `DomainContextEncoder` 先把 tab / hour / day 这类场景信息编码成 `d_ctx`
  - `PSRGLite` 用域条件对历史序列做动态重映射
  - `PCRGLite` 用候选 item 和 `d_ctx` 生成多个 query，再对历史做多兴趣 attention
- 当前仓库里是“ADS-lite”的工程实现，不是论文里那种重型参数生成网络，重点放在能跑、能做消融、能稳定接到现有训练链路上。
- 对应代码：
  - 配置：`configs/train_din_psrg_pcrg_mem16gb.yaml`
  - 代码：`src/models/modules/domain_context.py`、`src/models/modules/psrg.py`、`src/models/modules/pcrg.py`
  - 集成位置：`src/models/din.py`
  - 测试：`tests/test_din_psrg_pcrg_shapes.py`
- 结果口径：按项目汇报记录，平均 GAUC 相对 baseline 提升 `1.17%`。

### 2. Transformer 兴趣融合

- 想解决什么：PCRG 产生的多兴趣 token 之间默认彼此独立，候选和兴趣之间也只做了一层聚合，还不够细。
- 核心做法：
  - 先用 Self-Attention 做兴趣 token 内部上下文建模
  - 再用 `TargetAttentionDNN` 强化候选 item 和 token 的相关性
  - 最后用额外 FFN 抽更高阶兴趣表征
- 代码里默认用 `fusion_input=interest`，不是直接对整段历史做 Transformer，主要是考虑 4GB 显存下更稳。
- 对应代码：
  - 配置：`configs/train_din_psrg_pcrg_transformer.yaml`
  - 代码：`src/models/modules/transformer_fusion.py`、`src/models/modules/target_attention_dnn.py`
  - 测试：`src/tests/test_transformer_fusion_shapes.py`
- 结果口径：按项目汇报记录，平均 GAUC 再提升 `0.83%`。

### 3. MBCNet

- 想解决什么：把所有特征直接 `concat -> MLP`，对显式交叉的记忆能力不够强，泛化也比较普通。
- 核心做法：
  - 用 FGC 分支做特征分组交叉
  - 用 Low-rank Cross 分支做低秩显式交叉
  - 保留 Deep 分支承接隐式高阶组合
- 当前分组不是瞎切的，`feature_slices` 会把 `user_interest`、`cand_repr`、`user_profile_sparse_embs`、`user_dense`、`context_embs`、`candidate_side_embs` 这些块按语义切开。
- 对应代码：
  - 配置：`configs/train_din_psrg_pcrg_transformer_mbcnet.yaml`
  - 代码：`src/models/modules/mbcnet.py`、`src/models/modules/feature_slices.py`
  - 测试：`src/tests/test_mbcnet_shapes.py`
- 结果口径：按项目汇报记录，平均 GAUC 再提升 `0.52%`。

### 4. PPNet

- 想解决什么：同一套主干表示，在不同场景、不同用户属性、不同活跃度下，最好有轻量的条件化调制，而不是完全共用同一个 head。
- 核心做法：
  - `PersonalContextEncoder` 把场景时间特征、用户 dense 特征、`hist_len`、用户活跃度代理等拼成 `p_ctx`
  - `PPNet` 对 head 输入做 group-wise FiLM 调制
  - 分支 gate 逻辑也已经写进代码里，不过当前主配置用的是 `apply_to=head_input`
- 对应代码：
  - 配置：`configs/train_din_psrg_pcrg_transformer_mbcnet_ppnet.yaml`
  - 代码：`src/models/modules/personal_context.py`、`src/models/modules/ppnet.py`
  - 测试：`src/tests/test_ppnet_shapes.py`
- 结果口径：按项目汇报记录，平均 GAUC 再提升 `0.17%`。

## 项目结果概览


| 阶段 / 模块 | 核心改动 | GAUC 增益 | 备注 |
| --- | --- | --- | --- |
| DIN baseline | 单 query DIN + hash embedding + 50 长度历史序列 | 基线 | `configs/train_din_mem16gb.yaml` |
| ADS | PSRG 动态历史重映射 + PCRG 多 query 兴趣建模 | `+1.17%` | 项目汇报口径，仓库有配置 / 实现 / 测试 |
| Transformer 兴趣融合 | Self-Attention + Target-Attention DNN + FFN | `+0.83%` | `configs/train_din_psrg_pcrg_transformer.yaml` |
| MBCNet | 分组交叉分支 + 低秩交叉分支 + Deep 分支 | `+0.52%` | `configs/train_din_psrg_pcrg_transformer_mbcnet.yaml` |
| PPNet | 场景 / 用户 / 活跃度条件化调制 | `+0.17%` | `configs/train_din_psrg_pcrg_transformer_mbcnet_ppnet.yaml` |
| 最终结果 | 任务平均 GAUC | `0.692` | 相对 baseline 累计提升接近 `2.7%`，按项目汇报口径记录 |

## 想顺着 README 看代码，可以先看这些

| 想看什么 | 代码位置 |
| --- | --- |
| 训练主入口 | `src/main_train_din.py` |
| 训练 / 验证循环 | `src/trainers/train_din.py` |
| 数据流式读取 | `src/datasets/parquet_iterable_dataset.py` |
| DIN 主模型和特征拼接 | `src/models/din.py` |
| ADS 的域上下文、PSRG、PCRG | `src/models/modules/domain_context.py`、`src/models/modules/psrg.py`、`src/models/modules/pcrg.py` |
| Transformer 兴趣融合 | `src/models/modules/transformer_fusion.py`、`src/models/modules/target_attention_dnn.py` |
| MBCNet | `src/models/modules/mbcnet.py`、`src/models/modules/feature_slices.py` |
| PPNet | `src/models/modules/personal_context.py`、`src/models/modules/ppnet.py` |
| GAUC / AUC / LogLoss | `src/metrics/metrics.py` |
| 实验汇总 | `src/analysis/summarize_experiments.py` |
| tab 统计 | `src/analysis/tab_click_stats.py` |
| EDA | `scripts/run_eda.py`、`kuairand/eda/*` |

## 仓库结构

```text
.
├── configs/
│   ├── din_baseline_mem16gb.yaml
│   ├── train_din_mem16gb.yaml
│   ├── train_din_psrg_pcrg_mem16gb.yaml
│   ├── train_din_psrg_pcrg_transformer.yaml
│   ├── train_din_psrg_pcrg_transformer_mbcnet.yaml
│   └── train_din_psrg_pcrg_transformer_mbcnet_ppnet.yaml
├── kuairand/
│   └── eda/                        # 原始日志扫描、特征分析、报表生成
├── src/
│   ├── datasets/                   # ParquetIterableDataset / collate
│   ├── models/                     # DIN 主模型与各增强模块
│   │   └── modules/
│   ├── trainers/                   # train / eval / sanity check
│   ├── metrics/                    # AUC / LogLoss / GAUC
│   ├── analysis/                   # run 汇总、tab 统计
│   ├── utils/                      # seed / checkpoint
│   └── main_train_din.py           # 训练入口
├── scripts/
│   ├── smoke_test_din.py
│   ├── run_ads_ablation.py
│   ├── bench_train_throughput.py
│   └── run_eda.py
├── reports/
│   ├── eda_report/                 # 已生成的 EDA 报告
│   └── interview/                  # Quarto 面试讲解 deck
├── tests/                          # PSRG / PCRG shape tests
├── src/tests/                      # Transformer / MBCNet / PPNet tests
├── output/                         # 本地训练数据、词表、实验产物
└── pyproject.toml / requirements.txt
```

`src/` 这一层是训练主线。`kuairand/eda/` 是原始日志分析工具，两块代码风格和职责是分开的。

## 数据处理与训练流程

先说明一个事实：当前仓库已经能直接训练，但“旧 README 里那套 Stage A-F 构数脚本”不在现在的代码树里了。

- 现在真正能跑通的是：`output/processed/*.parquet` -> `src/main_train_din.py` -> `checkpoint / final_metrics / exp_summary`
- 旧 README 里提到的 `kuairand.din_baseline.main_build_dataset`、`resume_from_stage`、`force_streaming_split` 等入口，当前仓库里没有对应实现，`pyproject.toml` 里相关 entry 也还是旧的
- 因此这份 README 不再把 Stage A-F 当成现成入口介绍，只保留现有产物约定和训练链路

当前训练消费的核心产物是这些：

| 目录 / 文件 | 作用 |
| --- | --- |
| `output/processed/train.parquet` | 训练集 |
| `output/processed/val.parquet` | 验证集 |
| `output/processed/test_standard.parquet` | 标准曝光测试集 |
| `output/processed/test_random.parquet` | 随机曝光测试集 |
| `output/meta/field_schema.json` | 字段 schema、最大历史长度、vocab 大小 |
| `output/meta/split_summary.json` | 各切片样本量和时间范围 |
| `output/meta/sanity_checks.json` | 数据检查结果 |
| `output/vocabs/*.json` | 各字段词表 |

`output/meta/split_summary.json` 里的样本量如下：

| split | 样本数 | 说明 |
| --- | --- | --- |
| train | `130,181,455` | `standard_exp` 的训练切片 |
| val | `18,597,355` | `standard_exp` 的验证切片 |
| test_standard | `37,194,695` | `standard_exp` 的标准测试切片 |
| test_random | `1,186,031` | `random_exp` 单独保留的测试切片 |

训练时的数据流大致是这样：

1. `src/main_train_din.py` 根据 YAML 读取 `processed / meta / vocabs`
2. `src/datasets/parquet_iterable_dataset.py` 按 parquet row group 流式读取，并直接产出预组装 batch
3. `src/datasets/collate.py` 只做 `numpy -> torch.Tensor`
4. `src/models/din.py` 根据 `variant / head / ppnet` 组装模型
5. `src/trainers/train_din.py` 跑训练、验证和最终评估
6. `src/metrics/metrics.py` 计算 AUC / LogLoss / GAUC，并在必要时把中间结果落盘防止内存炸掉
7. `src/utils/checkpoint.py` 保存 `config_snapshot.yaml`、`checkpoint_best.pt`、`final_metrics.json`

## 快速开始 / 复现

### 1. 安装

```bash
pip install -r requirements.txt
pip install torch scikit-learn
```

如果你想用可编辑安装，也可以再执行：

```bash
pip install -e ".[eda,monitoring,dev]"
```

说明：

- 当前 `requirements.txt` / `pyproject.toml` 没把训练依赖声明完整，训练前要额外装 `torch` 和 `scikit-learn`
- EDA 直接用脚本路径执行时容易遇到导入路径问题，下面统一用模块方式

### 2. 数据放置

```text
data/
└── KuaiRand-27K/
    ├── log_standard_4_08_to_4_21_27k_part1.csv
    ├── log_standard_4_08_to_4_21_27k_part2.csv
    ├── log_standard_4_22_to_5_08_27k_part1.csv
    ├── log_standard_4_22_to_5_08_27k_part2.csv
    ├── log_random_4_22_to_5_08_27k.csv
    ├── user_features_27k.csv
    └── video_features_basic_27k.csv
```

训练入口额外需要：

```text
output/
├── processed/
├── meta/
└── vocabs/
```

如果你现在只有原始 CSV，没有 `output/processed` 这些产物，当前仓库还不能直接从 raw log 重新构训练样本，这一点放在下面的“限制”里单独说。

### 3. 先跑 smoke test

```bash
python scripts/smoke_test_din.py
```

这个脚本会读取 `val.parquet` 的一小部分数据，做一次 forward + backward，确认字段、shape 和梯度都没问题。

### 4. 跑一个最小 baseline

```bash
python -m src.main_train_din --config configs/train_din_mem16gb.yaml --debug_rows 512 --epochs 1 --run_dir output/exp_runs/debug_din
```

`--debug_rows` 会切到 `DebugMapDataset`，适合先把训练链路跑通。

### 5. 跑一个全量增强版本的最小检查

```bash
python -m src.main_train_din --config configs/train_din_psrg_pcrg_transformer_mbcnet_ppnet.yaml --debug_rows 256 --epochs 1 --run_dir output/exp_runs/debug_fullstack
```

这个命令会把 `PSRG + PCRG + Transformer + MBCNet + PPNet` 全部接起来，适合验证配置和模块装配是否正常。

### 6. 跑 baseline / ablation

baseline：

```bash
python -m src.main_train_din --config configs/train_din_mem16gb.yaml --run_dir output/exp_runs/din_baseline
```

ADS 消融脚本：

```bash
python scripts/run_ads_ablation.py --base_config configs/train_din_psrg_pcrg_mem16gb.yaml --data_root output/processed --meta_root output/meta --vocabs_root output/vocabs --run_root output/exp_runs/ads_ablation --dry_run
```

先加 `--dry_run` 看自动生成的命令，确认没问题后去掉它再正式跑。

### 7. 汇总实验结果

```bash
python -m src.analysis.summarize_experiments --runs_root output/exp_runs --output_dir output/analysis/exp_summary
```

会生成：

- `overall_summary.csv / .md`
- `delta_vs_baseline.csv / .md`
- 图表对比

### 8. 跑 EDA

```bash
python -m scripts.run_eda --data_dir data/KuaiRand-27K --out_dir output/eda_report --engine duckdb --chunksize 200000 --sample_users 5000 --seed 42
```

已有报告可以直接看：

- `reports/eda_report/report.md`
- `reports/interview/interview_deck.qmd`

### 9. 跑 tab 统计

快速检查原始日志：

```bash
python -m src.analysis.tab_click_stats --mode raw_logs --data_root data/KuaiRand-27K --output_dir output/analysis/tab_click_stats --debug_max_chunks 1
```

去掉 `--debug_max_chunks 1` 就是完整扫描。

### 10. 跑吞吐 benchmark

```bash
python scripts/bench_train_throughput.py
```

这个脚本会固定读取 `configs/train_din_mem16gb.yaml`，比较不同 `num_workers` 下的数据吞吐。

### 11. 跑测试

```bash
pytest -q tests src/tests
```

## 工程实现亮点

- `standard` 和 `random` 两类日志分开处理，`test_random` 单独保留，不把更接近无偏的评测切片混进训练集。
- `ParquetIterableDataset` 不是逐样本吐 dict，而是按 row group 读取后直接预组装 batch，DataLoader 用 `batch_size=None` 透传，少了一层大 Python 循环。
- 大词表默认走 `HashEmbedding`。对当前 `video_id` / `author_id` 规模，这是个很实际的工程选择。
- 评估阶段不是一次性把全量预测拉进内存，`StreamingMetricCollector` 和 tab collector 都支持超过阈值自动落盘。
- 训练入口完全配置驱动，`variant / num_queries / transformer / head / ppnet` 都可以通过 YAML 和 CLI 覆盖。
- 每个 run 都会自动保存 `config_snapshot.yaml`、`checkpoint_best.pt`、`final_metrics.json`，回看实验不会靠手记。
- `scripts/run_ads_ablation.py` 会为每组实验自动生成独立 config，适合按模块做增量验证。
- 测试不是只测 happy path。当前测试覆盖了全 padding、mask fallback、query 维度不一致、MBCNet 分组兜底、PPNet 广播调制这些比较容易踩坑的点。
- `scripts/bench_train_throughput.py` 还把 Windows 下 `num_workers` 的实际吞吐差异单独测了一遍，方便做加载器调参。

## 当前限制与后续计划

- 当前仓库的训练主线依赖现成的 `output/processed` / `output/meta` / `output/vocabs`。旧 README 里的构数入口已经不在代码树里，所以现在更像“训练 + 模型改造 + 实验分析仓库”，不是一条从 raw csv 到 parquet 的完整新链路。
- `pyproject.toml` 里还保留着旧的 `kuairand-din-build` entry，指向的模块当前不存在。README 已经不再用这条命令。
- 训练依赖声明还不完整，`torch` 和 `scikit-learn` 需要手动补装。
- `output/analysis/exp_summary/` 现在能看到的更多是 smoke / 检查性 run。完整实验表没有全部跟着仓库一起沉淀，所以结果部分只能按项目汇报口径写。
- `PPNet` 的 branch gate 路径已经在代码和测试里打通，但当前主配置还是 `apply_to=head_input`，这一块还缺更完整的 run 记录。
- `video_features_statistic` 相关特征没有接进主训练链路，主要是因为时间泄漏风险还没在当前仓库里做完严格对照。
- `src.analysis.tab_click_stats` 的 `processed_parquet` 模式默认读 `is_click`。如果你本地的 processed 数据也是按 `is_click` 导出的，就可以直接用；如果还保留旧版 `label_long_view` 字段名，需要先做一次字段对齐。


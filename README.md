# KuaiRand-27K 数据分析与建模工具包

> **KuaiRand-27K** 数据集的 EDA 分析 + DIN Baseline（严格版）数据预处理 Pipeline。

---

## 项目结构

```
kuairand/
├── README.md                      ← 本文件
├── pyproject.toml                 ← 项目元数据 & 依赖声明
├── requirements.txt               ← pip 依赖（兼容传统安装）
├── .gitignore
│
├── configs/                       # 配置文件
│   └── din_baseline_mem16gb.yaml  # DIN Pipeline 全参数配置（16GB RAM）
│
├── data/                          # 原始数据（gitignored）
│   └── KuaiRand-27K/             # 将解压后的数据放在此处
│       ├── log_standard_*.csv
│       ├── log_random_*.csv
│       ├── user_features_27k.csv
│       ├── video_features_basic_27k.csv
│       └── ...
│
├── kuairand/                      # Python 包
│   ├── __init__.py
│   ├── eda/                       # EDA 分析模块
│   │   ├── data_scan.py           #   数据目录扫描 & 文件角色识别
│   │   ├── profiling.py           #   CSV 质量画像
│   │   ├── log_eda.py             #   日志核心统计 & 序列分析
│   │   ├── feature_eda.py         #   用户/视频特征分析
│   │   ├── join_eda.py            #   Join 可用性检查
│   │   ├── notebook_builder.py    #   Jupyter Notebook 生成器
│   │   ├── reporting.py           #   Markdown 报告生成器
│   │   └── utils.py               #   公共工具函数
│   └── din_baseline/              # DIN Baseline 数据预处理
│       ├── main_build_dataset.py  #   CLI 主入口 & 6 阶段编排
│       ├── loaders.py             #   CSV 加载 & VideoFeatureLookup
│       ├── feature_engineering.py #   特征衍生 & 标准化
│       ├── vocab_builder.py       #   PAD=0/UNK=1 词表构建
│       ├── history_builder.py     #   DIN 历史序列（防泄漏/去重）
│       ├── sample_builder.py      #   逐桶流式样本构造
│       ├── splitter.py            #   严格时序切分
│       ├── sanity_checks.py       #   7 类数据质量检查
│       └── utils_memory.py        #   内存监控 & dtype 优化
│
├── scripts/                       # 入口脚本
│   └── run_eda.py                 #   EDA 主入口
│
├── notebooks/                     # Jupyter Notebooks
│   └── 01_eda.ipynb
│
├── reports/                       # 分析报告 & 演示文稿
│   ├── eda_report/                #   EDA 报告（report.md + figures + tables）
│   └── interview/                 #   面试演示（Quarto deck）
│
├── output/                        # Pipeline 产出（gitignored）
│   ├── processed/                 #   train/val/test parquet
│   ├── vocabs/                    #   词表 JSON
│   ├── meta/                      #   sanity checks & schema
│   └── intermediate/              #   中间产物（可删除）
│
└── tests/                         # 单元测试
    └── __init__.py
```

---

## 环境准备

### 安装

```bash
# 方式一：pip（推荐）
pip install -r requirements.txt

# 方式二：可编辑安装（开发模式）
pip install -e ".[all]"
```

### 数据放置

将 KuaiRand-27K 数据集解压到 `data/KuaiRand-27K/` 目录：

```
data/
└── KuaiRand-27K/
    ├── log_standard_4_08_to_4_21_27k_part1.csv
    ├── log_standard_4_08_to_4_21_27k_part2.csv
    ├── log_standard_4_22_to_5_08_27k_part1.csv
    ├── log_standard_4_22_to_5_08_27k_part2.csv
    ├── log_random_4_22_to_5_08_27k.csv
    ├── user_features_27k.csv
    ├── video_features_basic_27k.csv
    └── ...
```

---

## 一、DIN Baseline 数据预处理

为 **KuaiRand-27K** 生成 DIN baseline（严格版）的索引化特征样本（embedding 原始输入），
16GB RAM 约束下端到端运行。

### 快速开始

```bash
# 调试模式（推荐先跑通逻辑）
python -m kuairand.din_baseline.main_build_dataset `
  --config configs/din_baseline_mem16gb.yaml `
  --data_root data/KuaiRand-27K `
  --output_root output `
  --debug_rows 50000 `
  --max_users 100

# 全量运行
python -m kuairand.din_baseline.main_build_dataset `
  --config configs/din_baseline_mem16gb.yaml `
  --data_root data/KuaiRand-27K `
  --output_root output

# 当 Stage D 已完成、Stage E/F 失败时，从 Stage E 断点续跑（复用已有中间产物）
python -m kuairand.din_baseline.main_build_dataset `
  --config configs/din_baseline_mem16gb.yaml `
  --data_root data/KuaiRand-27K `
  --output_root output `
  --resume_from_stage E `
  --force_streaming_split
```

### CLI 参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--config` | YAML 配置文件路径 | 必填 |
| `--data_root` | KuaiRand-27K 数据目录 | 配置文件中的值 |
| `--output_root` | 输出目录 | 配置文件中的值 |
| `--debug_rows` | 每个 CSV 最多读取行数（调试用） | None（全量） |
| `--max_users` | 最多保留用户数（调试用） | None（全部） |
| `--disable_dedup` | 禁用历史去重 | False |
| `--rebuild_vocab` | 强制重建 vocab | False |
| `--seed` | 随机种子 | 42 |
| `--num_user_buckets` | 用户分桶数 | 100 |
| `--chunk_size` | CSV 分块读取行数 | 500000 |
| `--resume_from_stage` | 断点续跑起点：`A`/`E`/`F` | `A` |
| `--force_streaming_split` | Stage E 强制流式切分（推荐全量开启） | False |

### Pipeline 阶段

| 阶段 | 说明 | 预估峰值内存 |
|---|---|---|
| Stage A | CSV → 标准化 → 分桶 parquet | ~200MB |
| Stage B | 流式构建所有 vocab | ~300MB |
| Stage C | 用户/视频特征预处理 | ~200MB |
| Stage D | 逐桶构造 DIN 样本 | ~400MB |
| Stage E | 流式切分 + 合并分片 | ~1-3GB（与 batch 大小相关） |
| Stage F | Sanity checks | ~2-4GB |

### Stage E 断点续跑机制

- Stage E 会在 `output/processed/_stage_e_resume_state.json` 持久化进度。
- 分片结果写入 `output/processed/_stage_e_parts/<split>/bucket_*.parquet`。
- 中断后重跑 `--resume_from_stage E` 会跳过已完成桶，继续未完成桶。
- 最终会自动合并为：`train.parquet` / `val.parquet` / `test_standard.parquet` / `test_random.parquet`。

### 输出产物

```
output/
├── processed/              # 最终训练数据
│   ├── train.parquet
│   ├── val.parquet
│   ├── test_standard.parquet
│   └── test_random.parquet
├── vocabs/                 # 词表映射（JSON）
│   ├── video_id_vocab.json
│   ├── author_id_vocab.json
│   └── ...
├── meta/                   # 元数据与检查报告
│   ├── sanity_checks.json
│   ├── split_summary.json
│   ├── field_schema.json
│   └── feature_config_snapshot.yaml
└── intermediate/           # 中间产物（可删除）
```

### Parquet Schema（53 列）

<details>
<summary>点击展开完整 Schema</summary>

#### 元数据
- `sample_id` (int64): 全局唯一递增 ID
- `sample_time_ms` (int64): 样本时间戳
- `user_id_raw` (int32): 原始 user_id
- `cand_video_id_raw` (int32): 原始候选 video_id
- `meta_is_rand` (int8): 是否随机曝光
- `meta_log_source` (string): 日志来源

#### 标签
- `label_long_view` (int8): 长观看二分类标签

#### 历史序列（定长 list, 长度 = max_hist_len = 50）
- `hist_video_id` (list\<int32\>): 历史视频 ID（vocab 索引）
- `hist_author_id` (list\<int32\>): 历史作者 ID（vocab 索引）
- `hist_mask` (list\<int8\>): 1=有效, 0=填充
- `hist_len` (int32): 实际历史长度
- `hist_delta_t_bucket` (list\<int8\>): 时间差分桶
- `hist_play_ratio_bucket` (list\<int8\>): 播放比例分桶
- `hist_tab` (list\<int8\>): 历史事件的 tab

#### 候选字段
- `cand_video_id` (int32): 候选视频 vocab 索引
- `cand_author_id` (int32): 候选作者 vocab 索引
- `cand_video_type` (int32): 视频类型 vocab 索引
- `cand_upload_type` (int32): 上传类型 vocab 索引
- `cand_video_duration_bucket` (int8): 时长分桶

#### 上下文字段
- `tab` (int8), `hour_of_day` (int8), `day_of_week` (int8), `is_weekend` (int8)

#### 用户离散特征（vocab 索引化）
- `user_active_degree`, `is_lowactive_period`, `is_live_streamer`, `is_video_author`
- `follow_user_num_range`, `fans_user_num_range`, `friend_user_num_range`, `register_days_range`
- `onehot_feat0` ~ `onehot_feat17`

#### 用户连续特征
- `log1p_follow_user_num`, `log1p_fans_user_num`, `log1p_friend_user_num`, `log1p_register_days` (float32)

</details>

### Vocab 说明

所有 vocab 采用 `{原始值: 整数索引}` 映射，预留位：
- `__PAD__` = 0（序列填充）
- `__UNK__` = 1（未见过的值）

**共享关系**：`video_id` vocab 同时用于 `hist_video_id` / `cand_video_id`；`author_id` 同理。

### 内存优化策略

1. **分块 CSV 读取**：每次仅读 50 万行
2. **列裁剪 (usecols)**：只读需要的列
3. **dtype 降级**：int64→int8/16/32, float64→float32
4. **用户哈希分桶**：同用户事件落入同一桶，逐桶处理
5. **中间结果落盘 (parquet)**：每阶段完成即释放内存
6. **流式 vocab 构建**：Counter 逐桶累加，避免全量 unique()
7. **视频特征 numpy sorted array + searchsorted**：比 dict 省 4 倍内存
8. **逐用户流式写 parquet**：不累积所有样本在内存中

### PyTorch Dataset 对接

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class DINDataset(Dataset):
    def __init__(self, parquet_path):
        self.df = pd.read_parquet(parquet_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "hist_video_id": torch.tensor(row["hist_video_id"], dtype=torch.long),
            "hist_author_id": torch.tensor(row["hist_author_id"], dtype=torch.long),
            "hist_mask": torch.tensor(row["hist_mask"], dtype=torch.float),
            "hist_len": torch.tensor(row["hist_len"], dtype=torch.long),
            "cand_video_id": torch.tensor(row["cand_video_id"], dtype=torch.long),
            "cand_author_id": torch.tensor(row["cand_author_id"], dtype=torch.long),
            "label": torch.tensor(row["label_long_view"], dtype=torch.float),
            # ... 添加其他字段
        }

dataset = DINDataset("output/processed/train.parquet")
loader = DataLoader(dataset, batch_size=256, shuffle=True)
```

### 严格版设计取舍

| 决策 | 原因 |
|---|---|
| 不使用 `video_features_statistic_*.csv` | 统计特征时间点不确定，可能引入信息泄漏 |
| `is_rand` 不作为模型输入 | 仅作为元数据切片字段，避免模型学到分布差异 |
| 不接入文本/类目补充文件 | 基础 baseline 保持简洁，增强版可扩展 |
| 默认不将 random_exp 加入历史池 | 随机曝光行为信号可能有偏 |

---

## 二、EDA 分析

对 KuaiRand-27K 数据集进行系统性探索性数据分析。

### 运行 EDA

```bash
python scripts/run_eda.py \
  --data_dir data/KuaiRand-27K \
  --out_dir reports/eda_report
```

### EDA 模块说明

| 模块 | 功能 |
|---|---|
| `data_scan` | 数据目录扫描、文件角色识别 |
| `profiling` | CSV 质量画像（行数、列类型、缺失率） |
| `log_eda` | 日志核心统计、时序分析、序列长度分布 |
| `feature_eda` | 用户/视频特征分布分析 |
| `join_eda` | 日志与特征表 Join 成功率检查 |
| `notebook_builder` | 自动生成 Jupyter Notebook |
| `reporting` | 生成 Markdown 分析报告 |

### 产出文件

- `reports/eda_report/report.md` — 完整分析报告
- `reports/eda_report/figures/` — 可视化图表
- `reports/eda_report/tables/` — 统计汇总表（CSV）
- `notebooks/01_eda.ipynb` — 交互式 Notebook

---

## 后续扩展建议

1. **增强版历史特征**：打开 `hist_delta_t_bucket` / `hist_play_ratio_bucket` 参与 attention
2. **接入 tag 特征**：从 `video_features_basic` 的 `tag` 列提取多值特征
3. **接入 `video_features_statistic`**：需要解决时点对齐问题
4. **多任务标签**：同时预测 `long_view` + `is_like` + `is_follow`
5. **更大模型**：SIM (Search-based Interest Model) 支持更长历史 (1000+)

---

## 已知限制

1. Stage E 全量合并可能在极端情况下（数亿样本）超出 16GB 内存，此时自动切换流式模式
2. 历史构造使用 Python 循环，对于极高频用户（100K+ 事件）可能较慢
3. vocab 使用 Counter 字典，对于 10M+ 基数特征会占用较多内存

---

## 约定

- Line endings 统一管理（`.gitattributes`）
- Editor 配置统一（`.editorconfig`）
- 原始数据和产出通过 `.gitignore` 排除

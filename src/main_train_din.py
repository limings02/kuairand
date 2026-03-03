"""
DIN Baseline 训练主入口。

使用方式:
    python -m src.main_train_din \\
        --config configs/train_din_mem16gb.yaml \\
        --data_root output/processed \\
        --meta_root output/meta \\
        --vocabs_root output/vocabs \\
        --run_dir output/exp_runs/din_baseline

所有路径参数均可在 YAML 中配置，CLI 参数会覆盖 YAML 中的对应值。
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

# ── 项目内部导入 ──
from src.datasets.collate import DINCollateFn, reset_collate_print_flag
from src.datasets.parquet_iterable_dataset import (
    ParquetIterableDataset,
    resolve_columns,
)
from src.metrics.metrics import compute_all_metrics
from src.models.din import DINModel
from src.trainers.train_din import (
    evaluate,
    is_better,
    sanity_check_forward,
    train_one_epoch,
)
from src.utils.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    save_config_snapshot,
    save_final_metrics,
)
from src.utils.seed import set_seed

logger = logging.getLogger("src")


# ─────────────────────────────────────────────────────────────
# CLI 参数
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DIN Baseline 训练")
    p.add_argument("--config", type=str, default="configs/train_din_mem16gb.yaml", help="YAML 配置文件路径")
    p.add_argument("--data_root", type=str, default=None, help="覆盖 data.data_root")
    p.add_argument("--meta_root", type=str, default=None, help="覆盖 data.meta_root")
    p.add_argument("--vocabs_root", type=str, default=None, help="覆盖 data.vocabs_root")
    p.add_argument("--run_dir", type=str, default=None, help="实验输出目录")
    p.add_argument("--device", type=str, default=None, help="覆盖 device (auto/cuda/cpu)")
    p.add_argument("--epochs", type=int, default=None, help="覆盖训练 epoch 数")
    p.add_argument("--batch_size", type=int, default=None, help="覆盖训练 batch_size")
    p.add_argument("--eval_only", action="store_true", help="仅评估（跳过训练，加载 best checkpoint）")
    p.add_argument("--debug_rows", type=int, default=0, help=">0 时使用 DebugMapDataset 加载前 N 行")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# 配置加载与合并
# ─────────────────────────────────────────────────────────────

def load_config(args) -> dict:
    """加载 YAML 配置并用 CLI 参数覆盖对应字段。"""
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # CLI 覆盖
    if args.data_root:
        config["data"]["data_root"] = args.data_root
    if args.meta_root:
        config["data"]["meta_root"] = args.meta_root
    if args.vocabs_root:
        config["data"]["vocabs_root"] = args.vocabs_root
    if args.device:
        config["device"] = args.device
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    # run_dir
    run_name = config.get("run_name", "din_baseline")
    if args.run_dir:
        config["run_dir"] = args.run_dir
    else:
        config["run_dir"] = str(Path("output") / "exp_runs" / run_name)

    return config


# ─────────────────────────────────────────────────────────────
# Vocab Sizes 加载
# ─────────────────────────────────────────────────────────────

def load_vocab_sizes(config: dict) -> Dict[str, int]:
    """
    从 field_schema.json 的 vocab_sizes 字段读取各 vocab 的大小。
    """
    meta_root = config["data"]["meta_root"]
    schema_file = Path(meta_root) / config["data"]["field_schema_file"]

    if not schema_file.exists():
        raise FileNotFoundError(f"field_schema 文件不存在: {schema_file}")

    with open(schema_file, "r", encoding="utf-8") as f:
        schema = json.load(f)

    vocab_sizes = schema.get("vocab_sizes", {})
    if not vocab_sizes:
        raise ValueError(f"field_schema.json 中未找到 vocab_sizes 字段: {schema_file}")

    logger.info("已加载 vocab_sizes（共 %d 个 vocab）:", len(vocab_sizes))
    for name, size in sorted(vocab_sizes.items()):
        logger.info("  %-30s size=%d", name, size)

    return vocab_sizes


# ─────────────────────────────────────────────────────────────
# DataLoader 构建
# ─────────────────────────────────────────────────────────────

def _worker_init_fn(worker_id: int):
    """
    DataLoader worker 初始化函数。
    每个 worker 使用 torch 分配的不同种子初始化 numpy/random，避免 shuffle 重复。
    """
    import random
    import numpy as np
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def build_dataloader(
    config: dict,
    split: str,
    shuffle: bool = False,
    debug_rows: int = 0,
) -> DataLoader:
    """
    构建指定 split 的 DataLoader。

    Args:
        config: 完整配置
        split: 'train' / 'val' / 'test_standard' / 'test_random'
        shuffle: 是否打乱
        debug_rows: >0 时使用 DebugMapDataset

    Returns:
        DataLoader
    """
    data_cfg = config["data"]
    fields_cfg = config["fields"]
    dl_cfg = config.get("dataloader", {})
    eval_cfg = config.get("eval", {})

    # 文件路径
    file_map = {
        "train": data_cfg["train_file"],
        "val": data_cfg["val_file"],
        "test_standard": data_cfg["test_standard_file"],
        "test_random": data_cfg["test_random_file"],
    }
    parquet_path = str(Path(data_cfg["data_root"]) / file_map[split])

    if not Path(parquet_path).exists():
        raise FileNotFoundError(f"Parquet 文件不存在: {parquet_path}")

    # 解析列
    is_train = (split == "train")
    columns = resolve_columns(parquet_path, fields_cfg, is_train=is_train)
    logger.info("[%s] 实际使用字段清单 (%d 列): %s", split, len(columns), columns)

    # 选择 batch_size 和 workers
    if is_train:
        batch_size = config["training"]["batch_size"]
        num_workers = dl_cfg.get("num_workers", 2)
    else:
        batch_size = eval_cfg.get("batch_size", dl_cfg.get("batch_size", 512))
        num_workers = eval_cfg.get("num_workers", dl_cfg.get("num_workers", 2))

    max_hist_len = config["model"]["max_hist_len"]
    collate_fn = DINCollateFn(user_dense_cols=fields_cfg["user_dense_cols"])

    if debug_rows > 0:
        from src.datasets.parquet_iterable_dataset import DebugMapDataset
        dataset = DebugMapDataset(
            parquet_path=parquet_path,
            columns=columns,
            max_rows=debug_rows,
            max_hist_len=max_hist_len,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # DebugMapDataset 通常不需要多 worker
            collate_fn=collate_fn,
            pin_memory=dl_cfg.get("pin_memory", False),
        )

    # ── 默认：IterableDataset ──
    dataset = ParquetIterableDataset(
        parquet_path=parquet_path,
        columns=columns,
        max_hist_len=max_hist_len,
        shuffle=shuffle,
        shuffle_buffer_size=dl_cfg.get("shuffle_buffer_size", 0),
        base_seed=config.get("seed", 42),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=dl_cfg.get("pin_memory", True) and torch.cuda.is_available(),
        prefetch_factor=dl_cfg.get("prefetch_factor", 2) if num_workers > 0 else None,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    return loader


# ─────────────────────────────────────────────────────────────
# 设备选择
# ─────────────────────────────────────────────────────────────

def get_device(config: dict) -> torch.device:
    dev_str = config.get("device", "auto")
    if dev_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info("使用 GPU: %s (%.1f GB)", name, mem)
        else:
            device = torch.device("cpu")
            logger.info("CUDA 不可用，使用 CPU")
    else:
        device = torch.device(dev_str)
        logger.info("使用设备: %s", device)
    return device


# ─────────────────────────────────────────────────────────────
# 优化器构建
# ─────────────────────────────────────────────────────────────

def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    tcfg = config["training"]
    opt_name = tcfg.get("optimizer", "adam").lower()
    lr = tcfg["learning_rate"]
    wd = tcfg.get("weight_decay", 0.0)

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    logger.info("优化器: %s (lr=%.6f, weight_decay=%.2e)", opt_name, lr, wd)
    return optimizer


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 日志设置 ──
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # ── 加载配置 ──
    config = load_config(args)
    run_dir = config["run_dir"]
    os.makedirs(run_dir, exist_ok=True)

    # 同时写日志到文件
    fh = logging.FileHandler(os.path.join(run_dir, "train.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info("=" * 70)
    logger.info("DIN Baseline 训练启动")
    logger.info("实验目录: %s", run_dir)
    logger.info("=" * 70)

    # 保存配置快照
    save_config_snapshot(config, run_dir)

    # ── 随机种子 ──
    set_seed(config.get("seed", 42))

    # ── 设备 ──
    device = get_device(config)

    # ── Vocab Sizes ──
    vocab_sizes = load_vocab_sizes(config)

    # ── 模型 ──
    model = DINModel(config, vocab_sizes).to(device)

    # ── DataLoader ──
    debug_rows = args.debug_rows
    logger.info("构建 DataLoader...")

    train_loader = build_dataloader(config, "train", shuffle=True, debug_rows=debug_rows)
    val_loader = build_dataloader(config, "val", shuffle=False, debug_rows=debug_rows)

    # 预估 batch 数（用于进度条）
    def est_batches(split_name: str, batch_size: int) -> Optional[int]:
        """估算 batch 数量。"""
        file_map = {
            "train": config["data"]["train_file"],
            "val": config["data"]["val_file"],
            "test_standard": config["data"]["test_standard_file"],
            "test_random": config["data"]["test_random_file"],
        }
        path = Path(config["data"]["data_root"]) / file_map[split_name]
        import pyarrow.parquet as pq
        try:
            n = pq.ParquetFile(str(path)).metadata.num_rows
            return math.ceil(n / batch_size)
        except Exception:
            return None

    train_est = est_batches("train", config["training"]["batch_size"])
    val_est = est_batches("val", config.get("eval", {}).get("batch_size", 512))

    # ── Sanity Check ──
    sanity_check_forward(model, val_loader, device, config)

    if args.eval_only:
        # 仅评估模式
        logger.info("== Eval Only 模式 ==")
        load_checkpoint(model, run_dir, device=device)
        _run_final_eval(model, config, device, run_dir, debug_rows)
        return

    # ── 训练 ──
    optimizer = build_optimizer(model, config)
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=config["training"]["use_amp"] and device.type == "cuda")

    tcfg = config["training"]
    best_metric_val = None
    patience_counter = 0
    global_step = 0

    for epoch in range(tcfg["epochs"]):
        # 为 IterableDataset 设置 epoch（影响 shuffle 种子）
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)

        # 重置 collate 打印标记
        reset_collate_print_flag()

        # ── 训练 ──
        avg_loss, global_step = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            config=config,
            scaler=scaler,
            global_step=global_step,
            epoch=epoch,
            total_batches_est=train_est,
        )

        # ── 验证 ──
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            config=config,
            split_name="val",
            total_batches_est=val_est,
        )

        # ── 指标比较与 checkpoint ──
        monitor = tcfg["monitor_metric"]
        current_val = val_metrics[monitor]

        if is_better(current_val, best_metric_val, monitor):
            logger.info(
                "指标提升: %s %.6f → %.6f，保存 checkpoint",
                monitor,
                best_metric_val if best_metric_val is not None else 0.0,
                current_val,
            )
            best_metric_val = current_val
            save_checkpoint(model, optimizer, epoch, global_step, val_metrics, run_dir)
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(
                "指标未提升 (%s=%.6f, best=%.6f), patience=%d/%d",
                monitor, current_val, best_metric_val, patience_counter, tcfg["patience"],
            )
            if patience_counter >= tcfg["patience"]:
                logger.info("Early stopping 触发！")
                break

    # ── 加载 best checkpoint 进行最终评估 ──
    logger.info("=" * 70)
    logger.info("训练完成，加载 best checkpoint 进行 test 评估...")
    load_checkpoint(model, run_dir, device=device)
    _run_final_eval(model, config, device, run_dir, debug_rows)


def _run_final_eval(model, config, device, run_dir, debug_rows):
    """在 val / test_standard / test_random 上运行最终评估。"""
    all_results = {}

    for split in ["val", "test_standard", "test_random"]:
        logger.info("─" * 40)
        logger.info("评估 %s ...", split)

        loader = build_dataloader(config, split, shuffle=False, debug_rows=debug_rows)
        bs = config.get("eval", {}).get("batch_size", 512)
        est = None
        try:
            import pyarrow.parquet as pq
            path = Path(config["data"]["data_root"]) / config["data"][f"{split}_file"]
            est = math.ceil(pq.ParquetFile(str(path)).metadata.num_rows / bs)
        except Exception:
            pass

        reset_collate_print_flag()
        metrics = evaluate(
            model=model,
            dataloader=loader,
            device=device,
            config=config,
            split_name=split,
            total_batches_est=est,
        )
        all_results[split] = metrics

    # 汇总打印
    logger.info("=" * 70)
    logger.info("最终评估结果汇总:")
    logger.info("%-20s %10s %10s %10s %10s", "Split", "AUC", "LogLoss", "GAUC", "N")
    for split, m in all_results.items():
        logger.info(
            "%-20s %10.6f %10.6f %10.6f %10d",
            split, m["auc"], m["logloss"], m["gauc"], m["n_samples"],
        )
    logger.info("=" * 70)

    # 保存
    save_final_metrics(all_results, run_dir)


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()

"""
快速冒烟测试：加载少量数据，构建模型，跑一个 forward + backward，确认无报错。
用法: python scripts/smoke_test_din.py
"""
import sys
sys.path.insert(0, ".")

import logging
import json
import math
import torch
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("smoke_test")

from src.datasets.parquet_iterable_dataset import ParquetIterableDataset, resolve_columns
from src.datasets.collate import BatchCollateFn, SampleCollateFn
from src.models.din import DINModel
from src.utils.seed import set_seed

def main():
    set_seed(42)

    # 加载配置
    with open("configs/train_din_mem16gb.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    fields_cfg = config["fields"]
    model_cfg = config["model"]

    # 加载 vocab_sizes
    schema_path = Path(config["data"]["meta_root"]) / config["data"]["field_schema_file"]
    with open(schema_path, "r", encoding="utf-8") as f:
        vocab_sizes = json.load(f)["vocab_sizes"]
    logger.info("Vocab sizes loaded: %d vocabs", len(vocab_sizes))

    # 读取 val.parquet（较小）的前几个 row group
    parquet_path = str(Path(config["data"]["data_root"]) / "val.parquet")
    columns = resolve_columns(parquet_path, fields_cfg, is_train=False)
    logger.info("Columns: %s", columns)

    # 构建 IterableDataset（预组装 batch）
    batch_size = 4
    dataset = ParquetIterableDataset(
        parquet_path=parquet_path,
        columns=columns,
        batch_size=batch_size,
        max_hist_len=model_cfg["max_hist_len"],
        shuffle=False,
        base_seed=42,
    )
    logger.info("Dataset: %d rows, %d row_groups", dataset.num_rows, dataset.num_row_groups)

    # DataLoader（batch_size=None 因为 Dataset 已预组装 batch）
    collate_fn = BatchCollateFn(
        user_dense_cols=fields_cfg["user_dense_cols"],
        float_columns=dataset.float_columns,
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=None, num_workers=0, collate_fn=collate_fn)

    # 取一个 batch
    batch = next(iter(loader))
    logger.info("Batch keys: %s", list(batch.keys()))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            logger.info("  %s: shape=%s dtype=%s", k, list(v.shape), v.dtype)

    # 构建模型
    device = torch.device("cpu")
    model = DINModel(config, vocab_sizes).to(device)
    logger.info("Model created.")

    # Forward
    model.eval()
    with torch.no_grad():
        logits = model(batch)
    logger.info("Forward OK! logits shape=%s, values=%s", list(logits.shape), logits.tolist())

    # Backward
    model.train()
    label = batch[fields_cfg["label_col"]].float()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    logits2 = model(batch)
    loss = loss_fn(logits2, label)
    loss.backward()
    logger.info("Backward OK! loss=%.6f", loss.item())

    # 检查梯度
    total_grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.data.norm(2).item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    logger.info("Total grad norm: %.6f", total_grad_norm)

    logger.info("=" * 50)
    logger.info("冒烟测试全部通过！")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()

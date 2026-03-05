import sys
import time
import torch
import yaml
from pathlib import Path

sys.path.insert(0, ".")

from src.main_train_din import load_config, build_dataloader, load_vocab_sizes
from src.models.din import DINModel
from src.trainers.train_din import to_device


class _Args:
    config = "configs/train_din_mem16gb.yaml"
    data_root = None
    meta_root = None
    vocabs_root = None
    run_dir = None
    device = None
    epochs = None
    batch_size = None


def benchmark_data_only(cfg: dict, n_batches: int = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_dataloader(cfg, "train", shuffle=True, debug_rows=0)

    start = time.time()
    total_samples = 0
    for i, batch in enumerate(loader, start=1):
        _ = to_device(batch, device)
        total_samples += batch[cfg["fields"]["label_col"]].shape[0]
        if i >= n_batches:
            break

    elapsed = time.time() - start
    print(f"DATA_ONLY: workers={cfg['dataloader']['num_workers']}, batches={n_batches}, samples={total_samples}, sec={elapsed:.3f}, samp/s={total_samples/elapsed:.1f}")


def benchmark_fwd_bwd(cfg: dict, n_batches: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_dataloader(cfg, "train", shuffle=True, debug_rows=0)

    vocab_sizes = load_vocab_sizes(cfg)
    model = DINModel(cfg, vocab_sizes).to(device)
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()

    # 预热
    warmup = 1
    it = iter(loader)
    for _ in range(warmup):
        b = next(it)
        b = to_device(b, device)
        y = b[cfg["fields"]["label_col"]].float()
        logits = model(b)
        loss = criterion(logits, y)
        loss.backward()
        model.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    total_samples = 0
    for _ in range(n_batches):
        b = next(it)
        b = to_device(b, device)
        y = b[cfg["fields"]["label_col"]].float()
        logits = model(b)
        loss = criterion(logits, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
        total_samples += y.shape[0]

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start
    print(f"FWD_BWD: batches={n_batches}, samples={total_samples}, sec={elapsed:.3f}, samp/s={total_samples/elapsed:.1f}")


def main():
    with open("configs/train_din_mem16gb.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("== Data throughput benchmark (num_workers) ==")
    for w in [0, 1, 2, 4]:
        cfg_w = yaml.safe_load(yaml.safe_dump(cfg))
        cfg_w["dataloader"]["num_workers"] = w
        benchmark_data_only(cfg_w, n_batches=20)


if __name__ == "__main__":
    main()

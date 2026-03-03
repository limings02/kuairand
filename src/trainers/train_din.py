"""
DIN Baseline 训练 / 验证 / 评估循环。

关键设计：
  - 训练：支持 AMP 混合精度 + 梯度累积 + 梯度裁剪，适配 4GB 显存
  - 评估：逐 batch 前向 → 流式收集预测结果 → 统一计算 AUC/LogLoss/GAUC
  - NaN/Inf 检测：遇到异常值立即报警并中止
"""

import logging
import math
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics.metrics import StreamingMetricCollector, compute_all_metrics

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

# 这些列不需要送到 GPU，在 CPU 上保留即可
_META_KEYS = {"user_id_raw", "meta_is_rand", "sample_time_ms", "meta_log_source", "sample_id"}


def to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    将 batch 字典中的 Tensor 搬到目标 device。
    元数据列（user_id_raw 等）保留在 CPU，避免无谓的显存消耗。
    """
    result = {}
    for key, val in batch.items():
        if key in _META_KEYS:
            result[key] = val  # 保留在 CPU
        elif isinstance(val, torch.Tensor):
            result[key] = val.to(device, non_blocking=True)
        else:
            result[key] = val
    return result


def _check_nan_inf(tensor: torch.Tensor, name: str):
    """检查 tensor 是否包含 NaN 或 Inf。"""
    if torch.isnan(tensor).any():
        raise RuntimeError(f"检测到 NaN！位置: {name}")
    if torch.isinf(tensor).any():
        raise RuntimeError(f"检测到 Inf！位置: {name}")


# ─────────────────────────────────────────────────────────────
# 训练一个 epoch
# ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: dict,
    scaler: GradScaler,
    global_step: int,
    epoch: int,
    total_batches_est: Optional[int] = None,
) -> tuple:
    """
    训练一个 epoch。

    Args:
        model: DIN 模型
        dataloader: 训练 DataLoader
        optimizer: 优化器
        criterion: BCEWithLogitsLoss
        device: 训练设备
        config: 完整配置
        scaler: AMP GradScaler
        global_step: 当前全局步数
        epoch: 当前 epoch 编号
        total_batches_est: 预估的总 batch 数（用于进度条）

    Returns:
        (avg_loss: float, global_step: int)
    """
    model.train()
    tcfg = config["training"]
    accum_steps = tcfg["gradient_accum_steps"]
    use_amp = tcfg["use_amp"] and device.type == "cuda"
    grad_clip = tcfg["grad_clip_norm"]
    log_every = tcfg["log_every_n_steps"]

    running_loss = 0.0
    batch_count = 0
    epoch_loss_sum = 0.0
    epoch_batch_count = 0
    t0 = time.time()

    pbar = tqdm(
        dataloader,
        total=total_batches_est,
        desc=f"Epoch {epoch + 1} [Train]",
        dynamic_ncols=True,
    )

    for batch_idx, batch in enumerate(pbar):
        batch = to_device(batch, device)
        label = batch[config["fields"]["label_col"]].float()  # [B]

        # ── 前向 ──
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(batch)  # [B]
            _check_nan_inf(logits, "logits")
            loss = criterion(logits, label) / accum_steps

        _check_nan_inf(loss, "loss")

        # ── 反向（AMP 缩放）──
        scaler.scale(loss).backward()

        # ── 梯度累积到位后更新参数 ──
        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

        step_loss = loss.item() * accum_steps
        running_loss += step_loss
        batch_count += 1
        epoch_loss_sum += step_loss
        epoch_batch_count += 1

        # ── 定期日志 ──
        if batch_count > 0 and global_step % log_every == 0 and (batch_idx + 1) % accum_steps == 0:
            avg = running_loss / batch_count
            elapsed = time.time() - t0
            speed = batch_count * tcfg["batch_size"] / elapsed
            pbar.set_postfix(loss=f"{avg:.5f}", speed=f"{speed:.0f} samp/s", step=global_step)
            logger.info(
                "Epoch %d | Step %d | Loss %.6f | Speed %.0f samp/s",
                epoch + 1, global_step, avg, speed,
            )
            running_loss = 0.0
            batch_count = 0
            t0 = time.time()

    # 处理末尾不足 accum_steps 的残留梯度
    if (batch_idx + 1) % accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        global_step += 1

    avg_epoch_loss = epoch_loss_sum / max(epoch_batch_count, 1)
    logger.info("Epoch %d 训练完成 | Avg Loss: %.6f | Total batches: %d", epoch + 1, avg_epoch_loss, epoch_batch_count)
    return avg_epoch_loss, global_step


# ─────────────────────────────────────────────────────────────
# 评估
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: dict,
    split_name: str = "val",
    total_batches_est: Optional[int] = None,
) -> Dict[str, float]:
    """
    在指定数据集上评估模型。

    流程：
      1. 逐 batch 前向传播，收集 sigmoid 概率、标签、user_id
      2. 全部收集完毕后，一次性计算 AUC / LogLoss / GAUC
      3. 支持内存保护（StreamingMetricCollector 超阈值自动落盘）

    Args:
        model: DIN 模型
        dataloader: 评估 DataLoader
        device: 设备
        config: 完整配置
        split_name: 数据集名称（用于日志）
        total_batches_est: 用于进度条的预估 batch 数

    Returns:
        metrics dict: {auc, logloss, gauc, ...}
    """
    model.eval()
    use_amp = config["training"]["use_amp"] and device.type == "cuda"
    user_id_col = config["fields"]["user_id_col"]
    label_col = config["fields"]["label_col"]
    threshold_gb = config.get("eval", {}).get("memory_threshold_gb", 2.0)

    collector = StreamingMetricCollector(threshold_gb=threshold_gb)

    pbar = tqdm(
        dataloader,
        total=total_batches_est,
        desc=f"[Eval {split_name}]",
        dynamic_ncols=True,
    )

    for batch in pbar:
        batch = to_device(batch, device)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(batch)

        # sigmoid 得到概率
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        labels = batch[label_col].cpu().numpy().astype(np.int32)

        # user_id 在 CPU
        if isinstance(batch[user_id_col], torch.Tensor):
            uids = batch[user_id_col].numpy().astype(np.int64)
        else:
            uids = np.array(batch[user_id_col], dtype=np.int64)

        collector.add(probs, labels, uids)

    logger.info(
        "[%s] 共收集 %d 个样本 (%.2f MB)",
        split_name,
        collector.n_samples,
        collector._accumulated_bytes / (1024 ** 2),
    )

    all_preds, all_labels, all_uids = collector.get_all()

    if len(all_preds) == 0:
        logger.warning("[%s] 没有收集到任何样本！", split_name)
        return {"auc": 0.0, "logloss": 999.0, "gauc": 0.0, "n_samples": 0}

    metrics = compute_all_metrics(all_labels, all_preds, all_uids)
    logger.info(
        "[%s] AUC=%.6f | LogLoss=%.6f | GAUC=%.6f | N=%d | Pos%%=%.4f",
        split_name,
        metrics["auc"],
        metrics["logloss"],
        metrics["gauc"],
        metrics["n_samples"],
        metrics["positive_rate"],
    )

    collector.cleanup()
    return metrics


# ─────────────────────────────────────────────────────────────
# Sanity Check
# ─────────────────────────────────────────────────────────────

def sanity_check_forward(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: dict,
):
    """
    训练前进行一次小 batch 前向传播，确保没有 shape / dtype 错误。

    会打印每个输入 tensor 的 shape 和 dtype，以及输出 logit 的范围。
    """
    model.eval()
    use_amp = config["training"]["use_amp"] and device.type == "cuda"

    logger.info("=" * 60)
    logger.info("Sanity Check: 进行一次小 batch 前向传播...")

    for batch in dataloader:
        batch = to_device(batch, device)

        # 打印输入 shape
        logger.info("输入 batch 字段:")
        for key, val in sorted(batch.items()):
            if isinstance(val, torch.Tensor):
                logger.info("  %-35s shape=%-20s dtype=%s", key, str(list(val.shape)), val.dtype)

        with torch.no_grad():
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch)

        logger.info("输出 logits: shape=%s, dtype=%s", list(logits.shape), logits.dtype)
        logger.info(
            "  范围: [%.4f, %.4f], mean=%.4f, std=%.4f",
            logits.min().item(),
            logits.max().item(),
            logits.mean().item(),
            logits.std().item(),
        )

        # 检查 NaN/Inf
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise RuntimeError("Sanity Check 失败：输出包含 NaN 或 Inf！")

        logger.info("Sanity Check 通过！")
        logger.info("=" * 60)
        break


# ─────────────────────────────────────────────────────────────
# 指标比较
# ─────────────────────────────────────────────────────────────

def is_better(current: float, best: float, metric_name: str) -> bool:
    """判断当前指标是否优于历史最佳。logloss 越小越好，其余越大越好。"""
    if best is None:
        return True
    if metric_name == "logloss":
        return current < best
    return current > best

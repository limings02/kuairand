"""
Checkpoint 保存与加载工具。

功能：
- 保存 best checkpoint（模型权重 + 优化器状态 + epoch + metric）
- 加载 best checkpoint 用于评估
- 保存训练配置快照与最终指标
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

logger = logging.getLogger(__name__)


def ensure_run_dir(run_dir: str) -> Path:
    """确保实验输出目录存在。"""
    p = Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    metrics: Dict[str, float],
    run_dir: str,
    filename: str = "checkpoint_best.pt",
) -> str:
    """
    保存训练 checkpoint。

    保存内容包括：模型权重、优化器状态、当前 epoch/step、验证指标。
    """
    p = ensure_run_dir(run_dir)
    filepath = p / filename
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(state, filepath)
    logger.info("Checkpoint 已保存至 %s  (epoch=%d, metrics=%s)", filepath, epoch, metrics)
    return str(filepath)


def load_checkpoint(
    model: torch.nn.Module,
    run_dir: str,
    filename: str = "checkpoint_best.pt",
    device: Optional[torch.device] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """
    加载 checkpoint 到模型（可选地恢复优化器）。

    Returns:
        保存时的 epoch / global_step / metrics 信息。
    """
    filepath = Path(run_dir) / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint 不存在: {filepath}")

    map_location = device if device else "cpu"
    state = torch.load(filepath, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    logger.info(
        "已加载 checkpoint %s  (epoch=%d, step=%d)",
        filepath,
        state["epoch"],
        state["global_step"],
    )
    return {
        "epoch": state["epoch"],
        "global_step": state["global_step"],
        "metrics": state["metrics"],
    }


def save_config_snapshot(config: dict, run_dir: str) -> None:
    """把本次实验使用的完整配置保存到 run_dir/config_snapshot.yaml。"""
    p = ensure_run_dir(run_dir)
    filepath = p / "config_snapshot.yaml"
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    logger.info("配置快照已保存至 %s", filepath)


def save_final_metrics(metrics: Dict[str, Any], run_dir: str) -> None:
    """把最终评估指标保存为 JSON。"""
    p = ensure_run_dir(run_dir)
    filepath = p / "final_metrics.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("最终指标已保存至 %s", filepath)

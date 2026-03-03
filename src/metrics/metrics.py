"""
评估指标：AUC / LogLoss / GAUC。

支持"分批累积 + 内存保护"：
  - 评估阶段逐 batch 收集 preds / labels / user_ids
  - 监控累积数据量，若超过阈值则自动落盘（np.save）
  - 最终一次性计算全局指标

GAUC（Group AUC）说明
─────────────────────
按 user_id 分组，对每个用户独立计算 AUC，再按该用户的样本数加权平均。

为什么要跳过"单边样本"用户？
  AUC 的定义需要同时存在正样本和负样本。
  如果某个用户只有正样本（全是 1）或只有负样本（全是 0），
  其 AUC 是未定义的（sklearn 会报错或返回不可靠值）。
  强行计入这些用户会扭曲整体 GAUC，因此我们跳过只有单侧标签的用户。
"""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 内存友好的流式指标收集器
# ─────────────────────────────────────────────────────────────

class StreamingMetricCollector:
    """
    流式收集预测结果，支持内存保护与自动落盘。

    使用方式：
        collector = StreamingMetricCollector(threshold_gb=2.0)
        for batch in eval_loader:
            preds, labels, uids = ...
            collector.add(preds, labels, uids)
        all_preds, all_labels, all_uids = collector.get_all()
        metrics = compute_all_metrics(all_labels, all_preds, all_uids)

    内存估算与保护阈值：
        每个样本 ≈ 16 字节（4B pred + 4B label + 8B user_id）
        18M 样本 ≈ 288MB，37M 样本 ≈ 592MB，远低于默认 2GB 阈值。
        但保险起见，超过阈值时自动将已有数据写入临时文件。
    """

    def __init__(self, threshold_gb: float = 2.0, tmp_dir: Optional[str] = None):
        self.threshold_bytes = threshold_gb * (1024 ** 3)
        self.tmp_dir = tmp_dir
        self.reset()

    def reset(self):
        """清空所有累积数据。"""
        self._preds_chunks: List[np.ndarray] = []
        self._labels_chunks: List[np.ndarray] = []
        self._uids_chunks: List[np.ndarray] = []
        self._accumulated_bytes: int = 0
        self._disk_files: List[str] = []
        self._use_disk: bool = False
        self._n_samples: int = 0

    def add(self, preds: np.ndarray, labels: np.ndarray, user_ids: np.ndarray):
        """
        追加一批预测结果。

        Args:
            preds: [batch_size] float32 概率
            labels: [batch_size] int/float 标签
            user_ids: [batch_size] int64 用户 ID
        """
        chunk_bytes = preds.nbytes + labels.nbytes + user_ids.nbytes
        self._accumulated_bytes += chunk_bytes
        self._n_samples += len(preds)

        # 检查是否超过内存阈值
        if not self._use_disk and self._accumulated_bytes > self.threshold_bytes:
            logger.warning(
                "累积数据量 %.2f GB 超过阈值 %.2f GB，切换到磁盘落盘模式",
                self._accumulated_bytes / (1024 ** 3),
                self.threshold_bytes / (1024 ** 3),
            )
            self._use_disk = True
            self._flush_memory_to_disk()

        if self._use_disk:
            self._save_chunk_to_disk(preds, labels, user_ids)
        else:
            self._preds_chunks.append(preds.astype(np.float32))
            self._labels_chunks.append(labels.astype(np.int32))
            self._uids_chunks.append(user_ids.astype(np.int64))

    def get_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回所有累积的 (preds, labels, user_ids)。"""
        if self._use_disk:
            return self._load_all_from_disk()
        else:
            if not self._preds_chunks:
                return np.array([]), np.array([]), np.array([])
            return (
                np.concatenate(self._preds_chunks),
                np.concatenate(self._labels_chunks),
                np.concatenate(self._uids_chunks),
            )

    @property
    def n_samples(self) -> int:
        return self._n_samples

    # ──── 磁盘落盘 ────

    def _flush_memory_to_disk(self):
        """把当前内存中的所有 chunks 合并后写入一个临时文件。"""
        if self._preds_chunks:
            self._save_chunk_to_disk(
                np.concatenate(self._preds_chunks),
                np.concatenate(self._labels_chunks),
                np.concatenate(self._uids_chunks),
            )
            self._preds_chunks.clear()
            self._labels_chunks.clear()
            self._uids_chunks.clear()

    def _save_chunk_to_disk(self, preds, labels, uids):
        fd, path = tempfile.mkstemp(suffix=".npz", dir=self.tmp_dir)
        os.close(fd)
        np.savez_compressed(path, preds=preds, labels=labels, uids=uids)
        self._disk_files.append(path)

    def _load_all_from_disk(self):
        all_p, all_l, all_u = [], [], []
        for path in self._disk_files:
            data = np.load(path)
            all_p.append(data["preds"])
            all_l.append(data["labels"])
            all_u.append(data["uids"])
        # 加上还留在内存中的尾巴
        if self._preds_chunks:
            all_p.extend(self._preds_chunks)
            all_l.extend(self._labels_chunks)
            all_u.extend(self._uids_chunks)
        return np.concatenate(all_p), np.concatenate(all_l), np.concatenate(all_u)

    def cleanup(self):
        """删除临时文件。"""
        for path in self._disk_files:
            try:
                os.remove(path)
            except OSError:
                pass
        self._disk_files.clear()


# ─────────────────────────────────────────────────────────────
# 指标计算函数
# ─────────────────────────────────────────────────────────────

def compute_auc(labels: np.ndarray, preds: np.ndarray) -> float:
    """计算全局 AUC。"""
    if len(np.unique(labels)) < 2:
        logger.warning("标签中只有一类值，AUC 无法计算，返回 0.0")
        return 0.0
    return float(roc_auc_score(labels, preds))


def compute_logloss(labels: np.ndarray, preds: np.ndarray, eps: float = 1e-7) -> float:
    """
    计算 LogLoss（对数损失）。
    preds 应为概率值 ∈ (0, 1)，此处做 clip 防止 log(0)。
    """
    preds_clipped = np.clip(preds, eps, 1 - eps)
    return float(log_loss(labels, preds_clipped))


def compute_gauc(
    labels: np.ndarray,
    preds: np.ndarray,
    user_ids: np.ndarray,
    min_samples: int = 2,
) -> Tuple[float, int, int]:
    """
    计算 GAUC（Group AUC）：按 user_id 分组计算 AUC，再按样本数加权平均。

    为什么要跳过单边样本用户？
      AUC 需要至少一个正样本和一个负样本才有统计意义。
      如果某个用户只有正样本或只有负样本，其 AUC 未定义。
      强行计入会扭曲整体 GAUC。因此这些用户被跳过。

    Args:
        labels: [N] 0/1 标签
        preds:  [N] 概率预测
        user_ids: [N] 用户 ID
        min_samples: 参与 GAUC 的最小样本数

    Returns:
        (gauc, n_valid_users, n_total_users)
    """
    # 按 user_id 排序，然后用 numpy 操作高效分组（避免 pandas 额外内存开销）
    sort_idx = np.argsort(user_ids, kind="mergesort")
    user_ids_sorted = user_ids[sort_idx]
    labels_sorted = labels[sort_idx]
    preds_sorted = preds[sort_idx]

    # 找分组边界
    diff_mask = np.diff(user_ids_sorted) != 0
    boundaries = np.where(diff_mask)[0] + 1
    boundaries = np.concatenate([[0], boundaries, [len(user_ids_sorted)]])

    n_total_users = len(boundaries) - 1
    total_auc_weighted = 0.0
    total_weight = 0
    n_valid = 0

    for i in range(n_total_users):
        start = boundaries[i]
        end = boundaries[i + 1]
        n = end - start
        if n < min_samples:
            continue

        grp_labels = labels_sorted[start:end]
        n_pos = grp_labels.sum()
        n_neg = n - n_pos

        # 跳过单边样本用户
        if n_pos == 0 or n_neg == 0:
            continue

        try:
            auc = roc_auc_score(grp_labels, preds_sorted[start:end])
        except ValueError:
            continue

        total_auc_weighted += auc * n
        total_weight += n
        n_valid += 1

    gauc = total_auc_weighted / total_weight if total_weight > 0 else 0.0

    logger.info(
        "GAUC 计算完成: gauc=%.6f, 有效用户=%d/%d, 参与计算样本=%d/%d",
        gauc, n_valid, n_total_users, total_weight, len(labels),
    )
    return gauc, n_valid, n_total_users


def compute_all_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    user_ids: np.ndarray,
) -> Dict[str, Any]:
    """
    计算完整指标集合：AUC、LogLoss、GAUC。

    Returns:
        dict: {auc, logloss, gauc, gauc_valid_users, gauc_total_users}
    """
    auc_val = compute_auc(labels, preds)
    logloss_val = compute_logloss(labels, preds)
    gauc_val, valid_users, total_users = compute_gauc(labels, preds, user_ids)

    metrics = {
        "auc": round(auc_val, 6),
        "logloss": round(logloss_val, 6),
        "gauc": round(gauc_val, 6),
        "gauc_valid_users": valid_users,
        "gauc_total_users": total_users,
        "n_samples": len(labels),
        "n_positive": int(labels.sum()),
        "positive_rate": round(float(labels.mean()), 6),
    }
    return metrics

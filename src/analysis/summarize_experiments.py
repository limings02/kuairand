"""
实验结果汇总脚本：对多个 run 自动汇总、对比、产出报表与图表。

为什么要做这个脚本
----------------
训练结束后更关键的问题是“不同改动之间谁更好”，而不是单个 run 的孤立指标。
该脚本把 run 级 overall 指标与 tab 级细粒度指标放在同一张报表里，形成实验闭环。

baseline 选择策略
----------------
1) 优先使用 --baseline_run 显式指定；
2) 若未指定，则自动选择“最像基础版”的 run：优先 head=mlp、variant 更基础、复杂模块更少。

16GB 内存约束下的评估策略
-----------------------
当 --recompute_tab_metrics=true 时，不全量读入 pandas。
脚本复用 IterableDataset + DataLoader 流式评估，并用按 tab 的流式收集器累计
(pred, label, user_id)；超过内存阈值自动落盘 npz，避免 OOM。

tab 分桶 GAUC 说明
----------------
GAUC=按 user 分组计算 AUC，再按样本数加权平均。
tab 分桶时是在 tab 内再做同样计算。样本太少或标签单边的 tab 会标记为 unreliable。
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import logging
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from src.metrics.metrics import compute_auc, compute_gauc, compute_logloss

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


logger = logging.getLogger("src.analysis.summarize_experiments")

SPLITS = ("val", "test_standard", "test_random")
TAB_SPLIT_ALIAS_TO_NAME = {"standard": "test_standard", "random": "test_random"}

RUN_INDICATOR_PATTERNS = [
    "**/config_snapshot.yaml",
    "**/config_snapshot.yml",
    "**/config_snapshot.json",
    "**/checkpoint*.pt",
    "**/*.ckpt",
    "**/train.log",
    "**/final_metrics.json",
    "**/metrics.json",
    "**/eval.json",
    "**/*metrics*.json",
]

RUN_SUBDIR_NAMES = {"metrics", "logs", "preds", "checkpoints", "ckpts", "models", "weights"}

SPLIT_ALIASES = {
    "val": {"val", "validation", "valid"},
    "test_standard": {"test_standard", "standard"},
    "test_random": {"test_random", "random"},
}

METRIC_ALIASES = {
    "auc": {"auc", "rocauc"},
    "gauc": {"gauc", "groupauc"},
    "logloss": {"logloss", "loglossvalue", "loglossmetric", "loglossscore", "loglossmean", "loglossavg", "loglossv", "loglossval", "loglosses", "loglossvalueavg", "loglossmetricavg", "log_loss", "loss", "bceloss", "binarycrossentropy"},
    "n_samples": {"nsamples", "samplecount", "count", "n"},
    "n_positive": {"npositive", "positivecount", "poscount"},
    "positive_rate": {"positiverate", "posrate", "clickrate", "longviewrate"},
    "gauc_valid_users": {"gaucvalidusers", "validusers"},
    "gauc_total_users": {"gauctotalusers", "totalusers"},
}


@dataclass
class RunPaths:
    run_dir: Path
    run_name: str
    config_path: Optional[Path]
    checkpoint_path: Optional[Path]
    train_log_path: Optional[Path]
    final_metrics_path: Optional[Path]
    split_metric_paths: Dict[str, Optional[Path]]
    tab_metric_paths: Dict[str, Optional[Path]]


class TabStreamingMetricStore:
    """按 tab 分桶的流式收集器，带内存阈值保护与临时落盘。"""

    def __init__(self, threshold_gb: float = 2.0, tmp_dir: Optional[str] = None):
        self.threshold_bytes = int(threshold_gb * (1024**3))
        self.tmp_dir = tmp_dir
        self._in_memory: Dict[Any, Dict[str, List[np.ndarray]]] = defaultdict(
            lambda: {"preds": [], "labels": [], "uids": []}
        )
        self._disk_files: DefaultDict[Any, List[str]] = defaultdict(list)
        self._use_disk = False
        self._accumulated_bytes = 0
        self.n_samples = 0

    def add_batch(self, tabs: np.ndarray, preds: np.ndarray, labels: np.ndarray, uids: np.ndarray) -> None:
        buckets: DefaultDict[Any, List[int]] = defaultdict(list)
        for idx, tab in enumerate(tabs):
            buckets[normalize_tab_value(tab)].append(idx)

        for tab, idxs in buckets.items():
            index = np.asarray(idxs, dtype=np.int64)
            self.add_tab_chunk(tab, preds[index], labels[index], uids[index])

    def add_tab_chunk(self, tab: Any, preds: np.ndarray, labels: np.ndarray, uids: np.ndarray) -> None:
        preds_chunk = preds.astype(np.float32, copy=False)
        labels_chunk = labels.astype(np.int32, copy=False)
        uids_chunk = uids.astype(np.int64, copy=False)

        chunk_bytes = preds_chunk.nbytes + labels_chunk.nbytes + uids_chunk.nbytes
        self.n_samples += int(preds_chunk.shape[0])

        if self._use_disk:
            self._save_chunk_to_disk(tab, preds_chunk, labels_chunk, uids_chunk)
            return

        self._accumulated_bytes += chunk_bytes
        bucket = self._in_memory[tab]
        bucket["preds"].append(preds_chunk)
        bucket["labels"].append(labels_chunk)
        bucket["uids"].append(uids_chunk)

        if self._accumulated_bytes > self.threshold_bytes:
            logger.warning(
                "tab 收集累计内存 %.2f GB 超过阈值 %.2f GB，切换落盘模式",
                self._accumulated_bytes / (1024**3),
                self.threshold_bytes / (1024**3),
            )
            self._use_disk = True
            self._flush_memory_to_disk()

    def iter_tabs(self) -> List[Any]:
        tabs = set(self._in_memory.keys()) | set(self._disk_files.keys())
        return sorted(tabs, key=tab_sort_key)

    def get_tab_arrays(self, tab: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        preds_parts: List[np.ndarray] = []
        labels_parts: List[np.ndarray] = []
        uids_parts: List[np.ndarray] = []

        for path in self._disk_files.get(tab, []):
            data = np.load(path)
            preds_parts.append(data["preds"])
            labels_parts.append(data["labels"])
            uids_parts.append(data["uids"])

        bucket = self._in_memory.get(tab)
        if bucket:
            preds_parts.extend(bucket["preds"])
            labels_parts.extend(bucket["labels"])
            uids_parts.extend(bucket["uids"])

        if not preds_parts:
            return np.array([]), np.array([]), np.array([])
        return np.concatenate(preds_parts), np.concatenate(labels_parts), np.concatenate(uids_parts)

    def cleanup(self) -> None:
        for file_list in self._disk_files.values():
            for path in file_list:
                try:
                    os.remove(path)
                except OSError:
                    pass
        self._disk_files.clear()

    def _flush_memory_to_disk(self) -> None:
        for tab, bucket in self._in_memory.items():
            for preds, labels, uids in zip(bucket["preds"], bucket["labels"], bucket["uids"]):
                self._save_chunk_to_disk(tab, preds, labels, uids)
        self._in_memory.clear()
        self._accumulated_bytes = 0

    def _save_chunk_to_disk(self, tab: Any, preds: np.ndarray, labels: np.ndarray, uids: np.ndarray) -> None:
        fd, path = tempfile.mkstemp(
            suffix=f".tab_{str(tab).replace(os.sep, '_')}.npz",
            dir=self.tmp_dir,
        )
        os.close(fd)
        np.savez_compressed(path, preds=preds, labels=labels, uids=uids)
        self._disk_files[tab].append(path)


def parse_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value}")


def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总多个实验 run 的 overall/tab 指标并生成报表")
    parser.add_argument("--runs_root", type=str, default="output/exp_runs")
    parser.add_argument("--output_dir", type=str, default="output/analysis/exp_summary")
    parser.add_argument("--baseline_run", type=str, default="")
    parser.add_argument("--recompute_tab_metrics", type=parse_bool, default=False)
    parser.add_argument("--processed_root", type=str, default="")
    parser.add_argument("--vocabs_root", type=str, default="")
    parser.add_argument("--meta_root", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--include_patterns", nargs="*", default=[])
    parser.add_argument("--exclude_patterns", nargs="*", default=[])
    parser.add_argument("--group_a_patterns", nargs="*", default=[])
    parser.add_argument("--group_b_patterns", nargs="*", default=[])
    parser.add_argument("--max_runs", type=int, default=0)
    parser.add_argument("--debug", action="store_true", help="仅处理前几个 run（默认 5）")
    parser.add_argument("--debug_max_runs", type=int, default=5)
    parser.add_argument("--save_plots", type=parse_bool, default=True)
    parser.add_argument("--tab_metrics_overwrite", type=parse_bool, default=False)
    parser.add_argument("--eval_memory_threshold_gb", type=float, default=2.0)
    parser.add_argument("--min_tab_samples_unreliable", type=int, default=200)
    parser.add_argument("--min_valid_users_unreliable", type=int, default=20)
    parser.add_argument("--tmp_dir", type=str, default="")
    return parser.parse_args()


def split_patterns(raw_patterns: Sequence[str]) -> List[str]:
    patterns: List[str] = []
    for raw in raw_patterns:
        if not raw:
            continue
        for part in raw.replace(";", ",").split(","):
            p = part.strip()
            if p:
                patterns.append(p)
    return patterns


def normalize_tab_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    try:
        return int(value)
    except Exception:
        return str(value)


def tab_sort_key(value: Any) -> Tuple[int, Any]:
    if isinstance(value, int):
        return (0, value)
    text = str(value)
    try:
        return (0, int(text))
    except Exception:
        return (1, text)


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def to_text_path(path: Optional[Path]) -> str:
    return str(path) if path else "-"


def should_keep_run(run_name: str, include_patterns: Sequence[str], exclude_patterns: Sequence[str]) -> bool:
    target = run_name.lower()
    if include_patterns and not any(fnmatch.fnmatch(target, p.lower()) for p in include_patterns):
        return False
    if exclude_patterns and any(fnmatch.fnmatch(target, p.lower()) for p in exclude_patterns):
        return False
    return True


def normalize_run_dir_from_file(file_path: Path) -> Path:
    run_dir = file_path.parent
    while run_dir.name.lower() in RUN_SUBDIR_NAMES and run_dir.parent != run_dir:
        run_dir = run_dir.parent
    return run_dir


def discover_run_dirs(
    runs_root: Path,
    include_patterns: Sequence[str],
    exclude_patterns: Sequence[str],
    max_runs: int,
    debug: bool,
    debug_max_runs: int,
) -> List[Path]:
    if not runs_root.exists():
        raise FileNotFoundError(f"runs_root 不存在: {runs_root}")

    candidates: set[Path] = set()
    for pattern in RUN_INDICATOR_PATTERNS:
        for file_path in runs_root.glob(pattern):
            if not file_path.is_file():
                continue
            run_dir = normalize_run_dir_from_file(file_path).resolve()
            if run_dir == runs_root.resolve():
                continue
            try:
                run_dir.relative_to(runs_root.resolve())
            except ValueError:
                continue
            candidates.add(run_dir)

    if not candidates:
        for child in runs_root.iterdir():
            if child.is_dir():
                candidates.add(child.resolve())

    filtered = []
    for run_dir in candidates:
        run_name = run_dir.relative_to(runs_root.resolve()).as_posix()
        if should_keep_run(run_name, include_patterns, exclude_patterns):
            filtered.append(run_dir)

    filtered.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if max_runs > 0:
        filtered = filtered[:max_runs]
    if debug:
        filtered = filtered[: max(1, debug_max_runs)]
    return filtered


def pick_first_by_patterns(root: Path, patterns: Sequence[str]) -> Optional[Path]:
    for pattern in patterns:
        files = [p for p in root.glob(pattern) if p.is_file()]
        if files:
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return files[0]
    return None


def pick_checkpoint_file(run_dir: Path) -> Optional[Path]:
    return pick_first_by_patterns(
        run_dir,
        [
            "checkpoint_best.pt",
            "best*.pt",
            "checkpoint*.pt",
            "*.ckpt",
            "**/checkpoint_best.pt",
            "**/best*.pt",
            "**/checkpoint*.pt",
            "**/*.ckpt",
        ],
    )


def score_split_metric_candidate(path: Path, split: str) -> int:
    name = path.name.lower()
    if "tab" in name:
        return -1
    score = 0
    if name == f"{split}_metrics.json":
        score += 120
    if "metrics" in name:
        score += 30
    if path.parent.name.lower() == "metrics":
        score += 20
    if any(alias in name for alias in SPLIT_ALIASES[split]):
        score += 40
    if "eval" in name:
        score += 10
    return score


def pick_split_metric_file(run_dir: Path, split: str) -> Optional[Path]:
    best: Optional[Path] = None
    best_score = -1
    for path in run_dir.rglob("*.json"):
        score = score_split_metric_candidate(path, split)
        if score <= 0:
            continue
        if score > best_score or (score == best_score and best is not None and path.stat().st_mtime > best.stat().st_mtime):
            best = path
            best_score = score
    return best


def score_tab_metric_candidate(path: Path, alias: str) -> int:
    name = path.name.lower()
    split_name = TAB_SPLIT_ALIAS_TO_NAME[alias]
    if "tab" not in name:
        return -1
    score = 0
    if name == f"tab_metrics_{alias}.json":
        score += 120
    if "metrics" in name:
        score += 20
    if path.parent.name.lower() == "metrics":
        score += 20
    if alias in name:
        score += 40
    if split_name in name:
        score += 30
    return score


def pick_tab_metric_file(run_dir: Path, alias: str) -> Optional[Path]:
    best: Optional[Path] = None
    best_score = -1
    for path in run_dir.rglob("*.json"):
        score = score_tab_metric_candidate(path, alias)
        if score <= 0:
            continue
        if score > best_score or (score == best_score and best is not None and path.stat().st_mtime > best.stat().st_mtime):
            best = path
            best_score = score
    return best


def detect_run_paths(run_dir: Path, runs_root: Path) -> RunPaths:
    run_name = run_dir.relative_to(runs_root.resolve()).as_posix()
    config_path = pick_first_by_patterns(
        run_dir,
        [
            "config_snapshot.yaml",
            "config_snapshot.yml",
            "config_snapshot.json",
            "**/config_snapshot.yaml",
            "**/config_snapshot.yml",
            "**/config_snapshot.json",
            "*config*.yaml",
            "*config*.yml",
            "*config*.json",
            "**/*config*.yaml",
            "**/*config*.yml",
            "**/*config*.json",
        ],
    )
    checkpoint_path = pick_checkpoint_file(run_dir)
    train_log_path = pick_first_by_patterns(run_dir, ["train.log", "**/train.log", "logs/*.log"])
    final_metrics_path = pick_first_by_patterns(
        run_dir,
        [
            "final_metrics.json",
            "metrics/final_metrics.json",
            "metrics.json",
            "eval.json",
            "**/final_metrics.json",
            "**/metrics.json",
            "**/eval.json",
        ],
    )
    split_metric_paths = {split: pick_split_metric_file(run_dir, split) for split in SPLITS}
    tab_metric_paths = {alias: pick_tab_metric_file(run_dir, alias) for alias in TAB_SPLIT_ALIAS_TO_NAME}

    logger.info(
        "识别 run=%s | config=%s | ckpt=%s | final=%s | val=%s | test_standard=%s | test_random=%s | tab_standard=%s | tab_random=%s",
        run_name,
        to_text_path(config_path),
        to_text_path(checkpoint_path),
        to_text_path(final_metrics_path),
        to_text_path(split_metric_paths["val"]),
        to_text_path(split_metric_paths["test_standard"]),
        to_text_path(split_metric_paths["test_random"]),
        to_text_path(tab_metric_paths["standard"]),
        to_text_path(tab_metric_paths["random"]),
    )

    return RunPaths(
        run_dir=run_dir,
        run_name=run_name,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        train_log_path=train_log_path,
        final_metrics_path=final_metrics_path,
        split_metric_paths=split_metric_paths,
        tab_metric_paths=tab_metric_paths,
    )


def safe_read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            logger.warning("读取 JSON 编码失败: %s", path)
            return None
    except Exception as exc:
        logger.warning("读取 JSON 失败: %s, err=%s", path, exc)
        return None


def safe_read_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
        if isinstance(payload, dict):
            return payload
    except Exception as exc:
        logger.warning("读取配置失败: %s, err=%s", path, exc)
    return {}


def normalize_metric_block(block: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in block.items():
        norm_key = normalize_key(str(key))
        fval = safe_float(value)
        ival = safe_int(value)
        for target, aliases in METRIC_ALIASES.items():
            if norm_key in aliases:
                if target in {"n_samples", "n_positive", "gauc_valid_users", "gauc_total_users"} and ival is not None:
                    out[target] = ival
                elif fval is not None:
                    out[target] = fval
                break
    return out


def extract_split_metrics(payload: Any, split_hint: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    if not isinstance(payload, Mapping):
        return result

    metrics_node = payload.get("metrics")
    if isinstance(metrics_node, Mapping):
        nested = extract_split_metrics(metrics_node, split_hint=split_hint)
        result.update(nested)

    alias_norm: Dict[str, set[str]] = {
        split: {normalize_key(alias) for alias in aliases} for split, aliases in SPLIT_ALIASES.items()
    }
    for split, aliases in alias_norm.items():
        for key, value in payload.items():
            if normalize_key(str(key)) in aliases and isinstance(value, Mapping):
                metrics = normalize_metric_block(value)
                if metrics:
                    result[split] = metrics

    if not result and split_hint:
        metrics = normalize_metric_block(payload)
        if metrics:
            result[split_hint] = metrics

    return result


def merge_split_metrics(run_paths: RunPaths) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Optional[Path]]]:
    merged = {split: {} for split in SPLITS}
    source: Dict[str, Optional[Path]] = {split: None for split in SPLITS}

    if run_paths.final_metrics_path and run_paths.final_metrics_path.exists():
        payload = safe_read_json(run_paths.final_metrics_path)
        parsed = extract_split_metrics(payload)
        for split, metrics in parsed.items():
            if split in merged:
                merged[split].update(metrics)
                source[split] = run_paths.final_metrics_path

    for split in SPLITS:
        split_path = run_paths.split_metric_paths.get(split)
        if not split_path or not split_path.exists():
            continue
        payload = safe_read_json(split_path)
        parsed = extract_split_metrics(payload, split_hint=split)
        metrics = parsed.get(split, {})
        if metrics:
            merged[split].update(metrics)
            source[split] = split_path

    return merged, source


def infer_run_timestamp(run_paths: RunPaths) -> str:
    candidates = [run_paths.run_dir]
    for path in [run_paths.config_path, run_paths.train_log_path, run_paths.final_metrics_path, run_paths.checkpoint_path]:
        if path is not None and path.exists():
            candidates.append(path)
    ts = max(path.stat().st_mtime for path in candidates)
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def bool_to_text(value: Optional[bool]) -> str:
    if value is None:
        return ""
    return "on" if value else "off"


def extract_model_summary(config: Dict[str, Any], run_paths: RunPaths) -> Dict[str, Any]:
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    variant = str(model_cfg.get("variant", "unknown")).lower()

    head_cfg = model_cfg.get("head", {}) if isinstance(model_cfg, dict) else {}
    head_type = str(head_cfg.get("type", "mlp")).lower() if isinstance(head_cfg, dict) else "mlp"

    pcrg_cfg = model_cfg.get("pcrg", {}) if isinstance(model_cfg, dict) else {}
    tf_cfg = model_cfg.get("transformer_fusion", {}) if isinstance(model_cfg, dict) else {}
    mbcnet_cfg = head_cfg.get("mbcnet", {}) if isinstance(head_cfg, dict) else {}
    lowrank_cfg = mbcnet_cfg.get("lowrank_cross", {}) if isinstance(mbcnet_cfg, dict) else {}

    num_queries = safe_int(pcrg_cfg.get("num_queries")) if isinstance(pcrg_cfg, dict) else None
    tf_layers = safe_int(tf_cfg.get("n_layers")) if isinstance(tf_cfg, dict) else None
    tf_heads = safe_int(tf_cfg.get("n_heads")) if isinstance(tf_cfg, dict) else None
    mbc_rank = safe_int(lowrank_cfg.get("rank")) if isinstance(lowrank_cfg, dict) else None

    pcrg_enabled = bool(pcrg_cfg.get("enabled")) if isinstance(pcrg_cfg, dict) else None
    tf_enabled = bool(tf_cfg.get("enabled")) if isinstance(tf_cfg, dict) else None
    fgc_enabled = bool(mbcnet_cfg.get("enable_fgc")) if isinstance(mbcnet_cfg, dict) else None
    lr_enabled = bool(mbcnet_cfg.get("enable_lowrank_cross")) if isinstance(mbcnet_cfg, dict) else None
    deep_enabled = bool(mbcnet_cfg.get("enable_deep")) if isinstance(mbcnet_cfg, dict) else None

    notes = []
    if config.get("run_name") and str(config.get("run_name")) != run_paths.run_name:
        notes.append(f"config_run_name={config.get('run_name')}")
    if config.get("notes"):
        notes.append(str(config.get("notes")))

    return {
        "model_variant": variant,
        "head_type": head_type,
        "num_queries": num_queries,
        "transformer_layers": tf_layers,
        "transformer_heads": tf_heads,
        "mbcnet_rank": mbc_rank,
        "mbcnet_fgc": fgc_enabled,
        "mbcnet_lowrank": lr_enabled,
        "mbcnet_deep": deep_enabled,
        "pcrg_enabled": pcrg_enabled,
        "transformer_enabled": tf_enabled,
        "hyperparam_summary": (
            f"Q={num_queries if num_queries is not None else '-'}; "
            f"TF={tf_layers if tf_layers is not None else '-'}x{tf_heads if tf_heads is not None else '-'}; "
            f"MBC_rank={mbc_rank if mbc_rank is not None else '-'}; "
            f"FGC={bool_to_text(fgc_enabled)}"
        ),
        "notes": " | ".join(notes),
    }


def build_overall_row(run_paths: RunPaths, config: Dict[str, Any]) -> Dict[str, Any]:
    merged_metrics, metric_sources = merge_split_metrics(run_paths)
    model_summary = extract_model_summary(config, run_paths)

    row: Dict[str, Any] = {
        "run_name": run_paths.run_name,
        "run_dir": str(run_paths.run_dir),
        "timestamp": infer_run_timestamp(run_paths),
        "model_variant": model_summary["model_variant"],
        "head_type": model_summary["head_type"],
        "num_queries": model_summary["num_queries"],
        "transformer_layers": model_summary["transformer_layers"],
        "transformer_heads": model_summary["transformer_heads"],
        "mbcnet_rank": model_summary["mbcnet_rank"],
        "mbcnet_fgc": bool_to_text(model_summary["mbcnet_fgc"]),
        "mbcnet_lowrank": bool_to_text(model_summary["mbcnet_lowrank"]),
        "mbcnet_deep": bool_to_text(model_summary["mbcnet_deep"]),
        "pcrg_enabled": bool_to_text(model_summary["pcrg_enabled"]),
        "transformer_enabled": bool_to_text(model_summary["transformer_enabled"]),
        "hyperparam_summary": model_summary["hyperparam_summary"],
        "notes": model_summary["notes"],
        "config_path": to_text_path(run_paths.config_path),
        "checkpoint_path": to_text_path(run_paths.checkpoint_path),
    }

    for split in SPLITS:
        metrics = merged_metrics.get(split, {})
        row[f"{split}_auc"] = metrics.get("auc")
        row[f"{split}_gauc"] = metrics.get("gauc")
        row[f"{split}_logloss"] = metrics.get("logloss")
        row[f"{split}_n_samples"] = metrics.get("n_samples")
        row[f"{split}_positive_rate"] = metrics.get("positive_rate")
        row[f"{split}_metrics_path"] = to_text_path(metric_sources.get(split))
    return row


def build_baseline_score(row: pd.Series) -> float:
    """
    自动 baseline 打分，分数越高越接近“基础版”。

    规则：
    1) head=mlp 优先；
    2) variant 越基础越高分；
    3) transformer/mbcnet/pcrg query 多会扣分。
    """
    score = 0.0
    head_type = str(row.get("head_type", "")).lower()
    variant = str(row.get("model_variant", "")).lower()
    num_queries = safe_int(row.get("num_queries"))

    if head_type == "mlp":
        score += 100
    elif head_type == "mbcnet":
        score -= 40

    if variant == "din":
        score += 60
    elif variant == "din_psrg":
        score += 40
    elif variant == "din_psrg_pcrg":
        score += 20
    elif "transformer" in variant:
        score -= 20

    if row.get("transformer_enabled") == "on":
        score -= 20
    if row.get("pcrg_enabled") == "on":
        score -= 10
    if num_queries is not None and num_queries > 0:
        score -= min(num_queries, 16)

    return score


def select_baseline_run(overall_df: pd.DataFrame, baseline_run: str) -> Optional[str]:
    if overall_df.empty:
        return None

    if baseline_run:
        target = baseline_run.strip().lower()
        exact = overall_df[overall_df["run_name"].str.lower() == target]
        if not exact.empty:
            return str(exact.iloc[0]["run_name"])

        by_basename = overall_df[
            overall_df["run_name"].str.split("/").str[-1].str.lower() == target
        ]
        if not by_basename.empty:
            return str(by_basename.iloc[0]["run_name"])

        logger.warning("未找到显式 baseline_run=%s，将使用自动策略", baseline_run)

    scored = overall_df.copy()
    scored["_baseline_score"] = scored.apply(build_baseline_score, axis=1)
    scored = scored.sort_values(["_baseline_score", "timestamp", "run_name"], ascending=[False, False, True])
    selected = str(scored.iloc[0]["run_name"])
    logger.info("自动选择 baseline=%s (score=%.2f)", selected, float(scored.iloc[0]["_baseline_score"]))
    return selected


def build_delta_vs_baseline(overall_df: pd.DataFrame, baseline_run_name: Optional[str]) -> pd.DataFrame:
    if overall_df.empty or not baseline_run_name:
        return pd.DataFrame()

    baseline = overall_df[overall_df["run_name"] == baseline_run_name]
    if baseline.empty:
        logger.warning("baseline=%s 不在 overall 表中，无法计算 delta", baseline_run_name)
        return pd.DataFrame()
    baseline_row = baseline.iloc[0]

    out = overall_df.copy()
    out.insert(0, "baseline_run", baseline_run_name)

    metric_cols = [
        "val_auc",
        "val_gauc",
        "val_logloss",
        "test_standard_auc",
        "test_standard_gauc",
        "test_standard_logloss",
        "test_random_auc",
        "test_random_gauc",
        "test_random_logloss",
    ]
    for col in metric_cols:
        if col in out.columns:
            out[f"delta_{col}"] = out[col] - baseline_row[col]

    keep_cols = [
        "baseline_run",
        "run_name",
        "head_type",
        "model_variant",
        "val_gauc",
        "test_standard_gauc",
        "test_random_gauc",
        "delta_val_gauc",
        "delta_test_standard_gauc",
        "delta_test_random_gauc",
        "delta_val_auc",
        "delta_test_standard_auc",
        "delta_test_random_auc",
        "delta_val_logloss",
        "delta_test_standard_logloss",
        "delta_test_random_logloss",
    ]
    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols].sort_values("delta_test_standard_gauc", ascending=False, na_position="last")
    return out.reset_index(drop=True)


def read_tab_metrics_rows(path: Path) -> List[Dict[str, Any]]:
    payload = safe_read_json(path)
    if payload is None:
        return []

    items: List[Dict[str, Any]] = []
    if isinstance(payload, Mapping):
        if isinstance(payload.get("metrics_by_tab"), list):
            items = [x for x in payload["metrics_by_tab"] if isinstance(x, Mapping)]
        elif isinstance(payload.get("tab_metrics"), list):
            items = [x for x in payload["tab_metrics"] if isinstance(x, Mapping)]
        elif isinstance(payload.get("tabs"), list):
            items = [x for x in payload["tabs"] if isinstance(x, Mapping)]
        elif all(isinstance(v, Mapping) for v in payload.values()):
            for tab_key, metrics in payload.items():
                row = dict(metrics)
                row.setdefault("tab", tab_key)
                items.append(row)
    elif isinstance(payload, list):
        items = [x for x in payload if isinstance(x, Mapping)]

    rows: List[Dict[str, Any]] = []
    for item in items:
        tab = item.get("tab", item.get("tab_id", item.get("bucket")))
        if tab is None:
            continue
        norm = normalize_metric_block(item)

        sample_count = (
            safe_int(item.get("sample_count"))
            or safe_int(item.get("count"))
            or safe_int(norm.get("n_samples"))
            or 0
        )
        n_positive = safe_int(item.get("n_positive")) or safe_int(norm.get("n_positive"))
        pos_rate = safe_float(item.get("pos_rate"))
        if pos_rate is None:
            pos_rate = safe_float(item.get("positive_rate")) or safe_float(norm.get("positive_rate"))
        if pos_rate is None and sample_count > 0 and n_positive is not None:
            pos_rate = float(n_positive) / float(sample_count)

        rows.append(
            {
                "tab": normalize_tab_value(tab),
                "sample_count": sample_count,
                "pos_rate": pos_rate,
                "auc": safe_float(item.get("auc")) or safe_float(norm.get("auc")),
                "gauc": safe_float(item.get("gauc")) or safe_float(norm.get("gauc")),
                "logloss": safe_float(item.get("logloss")) or safe_float(norm.get("logloss")),
                "gauc_valid_users": safe_int(item.get("gauc_valid_users")) or safe_int(norm.get("gauc_valid_users")),
                "gauc_total_users": safe_int(item.get("gauc_total_users")) or safe_int(norm.get("gauc_total_users")),
                "is_unreliable": bool(item.get("is_unreliable", False)),
                "unreliable_reason": str(item.get("unreliable_reason", "")),
            }
        )

    rows.sort(key=lambda x: tab_sort_key(x["tab"]))
    return rows


def mark_tab_unreliable(
    n_samples: int,
    label_unique_count: int,
    gauc_valid_users: int,
    min_tab_samples: int,
    min_valid_users: int,
) -> Tuple[bool, str]:
    reasons: List[str] = []
    if n_samples < min_tab_samples:
        reasons.append(f"sample_count<{min_tab_samples}")
    if label_unique_count < 2:
        reasons.append("single_class_label")
    if gauc_valid_users < min_valid_users:
        reasons.append(f"valid_users<{min_valid_users}")
    return (len(reasons) > 0, ",".join(reasons))


def to_numpy_1d(value: Any, dtype: np.dtype) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value
    elif hasattr(value, "detach") and hasattr(value, "cpu"):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    return arr.astype(dtype, copy=False).reshape(-1)


def infer_tab_col(config: Dict[str, Any], batch: Mapping[str, Any]) -> str:
    fields = config.get("fields", {})
    candidates: List[str] = []

    tab_col = fields.get("tab_col")
    if isinstance(tab_col, str):
        candidates.append(tab_col)

    context_cols = fields.get("context_sparse_cols")
    if isinstance(context_cols, list):
        for col in context_cols:
            if isinstance(col, str) and "tab" in col.lower():
                candidates.append(col)

    candidates.extend(["tab", "cand_tab"])

    for col in candidates:
        if col in batch:
            return col

    for key in batch.keys():
        key_str = str(key).lower()
        if "tab" in key_str and "hist_" not in key_str:
            return str(key)

    raise KeyError("未在 batch 中找到 tab 列，请检查 fields.context_sparse_cols / tab_col")


def resolve_device_name(device_name: str) -> Any:
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def recompute_tab_metrics_for_split(
    run_paths: RunPaths,
    config: Dict[str, Any],
    split: str,
    args: argparse.Namespace,
) -> Optional[Path]:
    split_alias = "standard" if split == "test_standard" else "random"

    if run_paths.checkpoint_path is None or not run_paths.checkpoint_path.exists():
        logger.error("run=%s 缺少 checkpoint，无法重算 %s tab 指标", run_paths.run_name, split)
        return None

    eval_config = copy.deepcopy(config)
    if "training" not in eval_config and "train" in eval_config:
        eval_config["training"] = eval_config["train"]
    eval_config.setdefault("training", {})
    eval_config.setdefault("data", {})
    eval_config.setdefault("fields", {})

    if args.processed_root:
        eval_config["data"]["data_root"] = args.processed_root
    if args.vocabs_root:
        eval_config["data"]["vocabs_root"] = args.vocabs_root
    if args.meta_root:
        eval_config["data"]["meta_root"] = args.meta_root

    split_file_key = f"{split}_file"
    if split_file_key not in eval_config["data"]:
        eval_config["data"][split_file_key] = "test_standard.parquet" if split == "test_standard" else "test_random.parquet"

    if not eval_config["data"].get("data_root"):
        logger.error("run=%s 缺少 data.data_root，无法重算", run_paths.run_name)
        return None

    try:
        import torch
        from torch.amp.autocast_mode import autocast
        from tqdm import tqdm

        from src.main_train_din import build_dataloader, load_vocab_sizes
        from src.models.din import DINModel
        from src.trainers.train_din import to_device
        from src.utils.checkpoint import load_checkpoint
    except Exception as exc:
        logger.error("导入重算依赖失败: %s", exc)
        return None

    device = resolve_device_name(args.device)
    use_amp = bool(eval_config["training"].get("use_amp", False)) and str(device).startswith("cuda")

    try:
        vocab_sizes = load_vocab_sizes(eval_config)
        model = DINModel(eval_config, vocab_sizes).to(device)
        load_checkpoint(
            model=model,
            run_dir=str(run_paths.run_dir),
            filename=run_paths.checkpoint_path.name,
            device=device,
            optimizer=None,
        )
        dataloader = build_dataloader(eval_config, split=split, shuffle=False, debug_rows=0)
    except Exception as exc:
        logger.error("run=%s 重算 %s 准备失败: %s", run_paths.run_name, split, exc)
        return None

    label_col = eval_config["fields"].get("label_col")
    user_id_col = eval_config["fields"].get("user_id_col", "user_id_raw")
    if not label_col:
        logger.error("run=%s 缺少 fields.label_col，无法重算", run_paths.run_name)
        return None

    store = TabStreamingMetricStore(
        threshold_gb=args.eval_memory_threshold_gb,
        tmp_dir=args.tmp_dir or None,
    )
    model.eval()
    tab_col: Optional[str] = None

    try:
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"[Recompute {run_paths.run_name} {split}]", dynamic_ncols=True)
            for batch in pbar:
                if tab_col is None:
                    tab_col = infer_tab_col(eval_config, batch)
                    logger.info("run=%s split=%s 识别 tab 列: %s", run_paths.run_name, split, tab_col)

                batch = to_device(batch, device)
                with autocast(device_type=device.type, enabled=use_amp):
                    logits = model(batch)

                probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
                labels = to_numpy_1d(batch[label_col], np.int32)
                user_ids = to_numpy_1d(batch[user_id_col], np.int64)
                tabs = to_numpy_1d(batch[tab_col], np.int64)
                store.add_batch(tabs=tabs, preds=probs, labels=labels, uids=user_ids)
    except Exception as exc:
        store.cleanup()
        logger.error("run=%s split=%s 重算失败: %s", run_paths.run_name, split, exc)
        return None

    tab_rows: List[Dict[str, Any]] = []
    for tab in store.iter_tabs():
        preds, labels, uids = store.get_tab_arrays(tab)
        if preds.size == 0:
            continue
        n_samples = int(preds.shape[0])
        n_positive = int(labels.sum())
        pos_rate = float(labels.mean()) if n_samples > 0 else 0.0
        auc = compute_auc(labels, preds)
        logloss = compute_logloss(labels, preds)
        gauc, valid_users, total_users = compute_gauc(labels, preds, uids)

        is_unreliable, reason = mark_tab_unreliable(
            n_samples=n_samples,
            label_unique_count=int(np.unique(labels).shape[0]),
            gauc_valid_users=valid_users,
            min_tab_samples=args.min_tab_samples_unreliable,
            min_valid_users=args.min_valid_users_unreliable,
        )

        tab_rows.append(
            {
                "tab": tab,
                "sample_count": n_samples,
                "n_positive": n_positive,
                "pos_rate": round(pos_rate, 6),
                "auc": round(float(auc), 6),
                "gauc": round(float(gauc), 6),
                "logloss": round(float(logloss), 6),
                "gauc_valid_users": int(valid_users),
                "gauc_total_users": int(total_users),
                "is_unreliable": bool(is_unreliable),
                "unreliable_reason": reason,
            }
        )

    store.cleanup()
    if not tab_rows:
        logger.error("run=%s split=%s 重算后未得到 tab 指标", run_paths.run_name, split)
        return None

    tab_rows.sort(key=lambda x: tab_sort_key(x["tab"]))

    out_path = run_paths.tab_metric_paths.get(split_alias)
    if out_path is None:
        out_path = run_paths.run_dir / "metrics" / f"tab_metrics_{split_alias}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "split": split,
        "generated_by": "src.analysis.summarize_experiments",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tab_col": tab_col or "tab",
        "n_tabs": len(tab_rows),
        "n_samples_total": int(sum(row["sample_count"] for row in tab_rows)),
        "metrics_by_tab": tab_rows,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("run=%s split=%s 重算 tab 指标保存: %s", run_paths.run_name, split, out_path)
    return out_path


def enrich_tab_rows_with_baseline_delta(tab_df: pd.DataFrame, baseline_run_name: Optional[str]) -> pd.DataFrame:
    if tab_df.empty:
        return tab_df
    out = tab_df.copy()
    out["baseline_gauc"] = np.nan
    out["delta_gauc_vs_baseline"] = np.nan
    if not baseline_run_name:
        return out

    baseline_rows = out[out["run_name"] == baseline_run_name][["tab", "gauc"]]
    if baseline_rows.empty:
        return out

    baseline_map = {row["tab"]: row["gauc"] for _, row in baseline_rows.iterrows()}
    out["baseline_gauc"] = out["tab"].map(baseline_map)
    out["delta_gauc_vs_baseline"] = out["gauc"] - out["baseline_gauc"]
    return out


def dataframe_to_markdown(df: pd.DataFrame, float_cols: Optional[Sequence[str]] = None) -> str:
    if df.empty:
        return "（空表）"
    float_cols = list(float_cols or [])
    headers = list(df.columns)

    def _fmt(col: str, value: Any) -> str:
        if pd.isna(value):
            return ""
        if col in float_cols:
            return f"{float(value):.6f}"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [_fmt(col, row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_table_outputs(
    df: pd.DataFrame,
    csv_path: Path,
    md_path: Path,
    float_cols: Optional[Sequence[str]] = None,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(dataframe_to_markdown(df, float_cols=float_cols), encoding="utf-8")
    logger.info("已输出: %s", csv_path)
    logger.info("已输出: %s", md_path)


def build_head_type_compare(overall_df: pd.DataFrame) -> pd.DataFrame:
    if overall_df.empty or "head_type" not in overall_df.columns:
        return pd.DataFrame()
    metrics = [c for c in ["val_gauc", "test_standard_gauc", "test_random_gauc"] if c in overall_df.columns]
    if not metrics:
        return pd.DataFrame()
    grouped = overall_df.groupby("head_type", dropna=False)[metrics].mean(numeric_only=True).reset_index()
    grouped["run_count"] = overall_df.groupby("head_type").size().values
    return grouped


def build_custom_group_compare(
    overall_df: pd.DataFrame,
    group_a_patterns: Sequence[str],
    group_b_patterns: Sequence[str],
) -> pd.DataFrame:
    if overall_df.empty or not group_a_patterns or not group_b_patterns:
        return pd.DataFrame()

    def _mask(series: pd.Series, patterns: Sequence[str]) -> pd.Series:
        mask = pd.Series(False, index=series.index)
        for pattern in patterns:
            p = pattern.lower()
            mask = mask | series.str.lower().map(lambda text: fnmatch.fnmatch(text, p))
        return mask

    run_names = overall_df["run_name"].astype(str)
    group_a = overall_df[_mask(run_names, group_a_patterns)]
    group_b = overall_df[_mask(run_names, group_b_patterns)]
    if group_a.empty or group_b.empty:
        return pd.DataFrame()

    metrics = [c for c in ["val_gauc", "test_standard_gauc", "test_random_gauc"] if c in overall_df.columns]
    row: Dict[str, Any] = {
        "group_a_patterns": ",".join(group_a_patterns),
        "group_b_patterns": ",".join(group_b_patterns),
        "group_a_run_count": int(group_a.shape[0]),
        "group_b_run_count": int(group_b.shape[0]),
    }
    for m in metrics:
        a_mean = group_a[m].mean()
        b_mean = group_b[m].mean()
        row[f"group_a_{m}_mean"] = a_mean
        row[f"group_b_{m}_mean"] = b_mean
        row[f"delta_group_b_minus_a_{m}"] = b_mean - a_mean
    return pd.DataFrame([row])


def plot_overall_gauc_compare(overall_df: pd.DataFrame, output_path: Path) -> None:
    if overall_df.empty or not HAS_MATPLOTLIB:
        return
    need = ["run_name", "val_gauc", "test_standard_gauc", "test_random_gauc"]
    if not all(c in overall_df.columns for c in need):
        return

    plot_df = overall_df[need].copy()
    plot_df = plot_df.sort_values("test_standard_gauc", ascending=False, na_position="last").head(25)
    if plot_df.empty:
        return

    x = np.arange(plot_df.shape[0], dtype=np.float32)
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(10, plot_df.shape[0] * 0.55), 5.8))
    ax.bar(x - width, plot_df["val_gauc"], width=width, label="val_gauc")
    ax.bar(x, plot_df["test_standard_gauc"], width=width, label="test_standard_gauc")
    ax.bar(x + width, plot_df["test_random_gauc"], width=width, label="test_random_gauc")
    ax.set_title("Overall GAUC Compare")
    ax.set_xlabel("run")
    ax.set_ylabel("GAUC")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["run_name"], rotation=55, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    logger.info("已输出图表: %s", output_path)


def plot_tab_delta_gauc(tab_df: pd.DataFrame, output_path: Path, title: str) -> None:
    if tab_df.empty or not HAS_MATPLOTLIB:
        return
    if not {"run_name", "tab", "delta_gauc_vs_baseline"}.issubset(tab_df.columns):
        return

    plot_df = tab_df.dropna(subset=["delta_gauc_vs_baseline"]).copy()
    if plot_df.empty:
        return

    mean_delta = plot_df.groupby("run_name")["delta_gauc_vs_baseline"].mean().sort_values(ascending=False)
    selected_runs = list(mean_delta.head(6).index)
    plot_df = plot_df[plot_df["run_name"].isin(selected_runs)]
    if plot_df.empty:
        return

    pivot = plot_df.pivot_table(
        index="tab",
        columns="run_name",
        values="delta_gauc_vs_baseline",
        aggfunc="mean",
    )
    if pivot.empty:
        return

    pivot = pivot.sort_index(key=lambda idx: [tab_sort_key(x) for x in idx])
    fig, ax = plt.subplots(figsize=(max(10, pivot.shape[0] * 0.55), 6.0))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("tab")
    ax.set_ylabel("ΔGAUC vs baseline")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(title="run", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    logger.info("已输出图表: %s", output_path)


def run_main(args: argparse.Namespace) -> int:
    runs_root = Path(args.runs_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    include_patterns = split_patterns(args.include_patterns)
    exclude_patterns = split_patterns(args.exclude_patterns)
    group_a_patterns = split_patterns(args.group_a_patterns)
    group_b_patterns = split_patterns(args.group_b_patterns)

    run_dirs = discover_run_dirs(
        runs_root=runs_root,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        max_runs=args.max_runs,
        debug=args.debug,
        debug_max_runs=args.debug_max_runs,
    )
    if not run_dirs:
        logger.error("在 runs_root=%s 下没有找到可用 run", runs_root)
        return 1

    overall_rows: List[Dict[str, Any]] = []
    tab_standard_rows: List[Dict[str, Any]] = []
    tab_random_rows: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        run_paths = detect_run_paths(run_dir, runs_root)
        config = safe_read_config(run_paths.config_path)
        row = build_overall_row(run_paths, config)
        overall_rows.append(row)

        for alias, split in TAB_SPLIT_ALIAS_TO_NAME.items():
            tab_path = run_paths.tab_metric_paths.get(alias)
            need_recompute = args.recompute_tab_metrics and (tab_path is None or args.tab_metrics_overwrite)
            if need_recompute:
                recomputed = recompute_tab_metrics_for_split(run_paths, config, split, args)
                if recomputed is not None:
                    run_paths.tab_metric_paths[alias] = recomputed
                    tab_path = recomputed

            if tab_path is None or not tab_path.exists():
                continue

            rows = read_tab_metrics_rows(tab_path)
            for r in rows:
                r["run_name"] = run_paths.run_name
                r["run_dir"] = str(run_paths.run_dir)
                r["head_type"] = row.get("head_type")
                r["model_variant"] = row.get("model_variant")
                r["tab_metrics_path"] = str(tab_path)

            if alias == "standard":
                tab_standard_rows.extend(rows)
            else:
                tab_random_rows.extend(rows)

    overall_df = pd.DataFrame(overall_rows)
    if overall_df.empty:
        logger.error("未生成 overall summary")
        return 1

    sort_cols = [c for c in ["test_standard_gauc", "val_gauc", "timestamp"] if c in overall_df.columns]
    if sort_cols:
        overall_df = overall_df.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
    overall_df = overall_df.reset_index(drop=True)

    baseline_run_name = select_baseline_run(overall_df, args.baseline_run)
    if baseline_run_name:
        logger.info("baseline run: %s", baseline_run_name)

    delta_df = build_delta_vs_baseline(overall_df, baseline_run_name)

    tab_standard_df = pd.DataFrame(tab_standard_rows)
    tab_random_df = pd.DataFrame(tab_random_rows)

    if not tab_standard_df.empty:
        tab_standard_df = enrich_tab_rows_with_baseline_delta(tab_standard_df, baseline_run_name)
        tab_standard_df = tab_standard_df.sort_values(
            ["tab", "delta_gauc_vs_baseline", "run_name"],
            ascending=[True, False, True],
            na_position="last",
        ).reset_index(drop=True)

    if not tab_random_df.empty:
        tab_random_df = enrich_tab_rows_with_baseline_delta(tab_random_df, baseline_run_name)
        tab_random_df = tab_random_df.sort_values(
            ["tab", "delta_gauc_vs_baseline", "run_name"],
            ascending=[True, False, True],
            na_position="last",
        ).reset_index(drop=True)

    write_table_outputs(
        overall_df,
        csv_path=output_dir / "overall_summary.csv",
        md_path=output_dir / "overall_summary.md",
        float_cols=[c for c in overall_df.columns if any(k in c for k in ["auc", "gauc", "logloss", "rate"])],
    )

    write_table_outputs(
        tab_standard_df,
        csv_path=output_dir / "tab_summary_standard.csv",
        md_path=output_dir / "tab_summary_standard.md",
        float_cols=["pos_rate", "auc", "gauc", "logloss", "baseline_gauc", "delta_gauc_vs_baseline"],
    )

    if not tab_random_df.empty:
        write_table_outputs(
            tab_random_df,
            csv_path=output_dir / "tab_summary_random.csv",
            md_path=output_dir / "tab_summary_random.md",
            float_cols=["pos_rate", "auc", "gauc", "logloss", "baseline_gauc", "delta_gauc_vs_baseline"],
        )
    else:
        logger.warning("未发现 random tab 指标，跳过 tab_summary_random 输出")

    write_table_outputs(
        delta_df,
        csv_path=output_dir / "delta_vs_baseline.csv",
        md_path=output_dir / "delta_vs_baseline.md",
        float_cols=[c for c in delta_df.columns if any(k in c for k in ["auc", "gauc", "logloss"])],
    )

    head_compare_df = build_head_type_compare(overall_df)
    custom_group_df = build_custom_group_compare(overall_df, group_a_patterns, group_b_patterns)
    if not head_compare_df.empty or not custom_group_df.empty:
        delta_md = output_dir / "delta_vs_baseline.md"
        sections = [delta_md.read_text(encoding="utf-8"), ""]
        if not head_compare_df.empty:
            sections.append("## Head Type 平均对比")
            sections.append(
                dataframe_to_markdown(
                    head_compare_df,
                    float_cols=[c for c in head_compare_df.columns if "gauc" in c],
                )
            )
            sections.append("")
        if not custom_group_df.empty:
            sections.append("## 自定义两组对比")
            sections.append(
                dataframe_to_markdown(
                    custom_group_df,
                    float_cols=[c for c in custom_group_df.columns if "gauc" in c or "delta_" in c],
                )
            )
            sections.append("")
        delta_md.write_text("\n".join(sections), encoding="utf-8")
        logger.info("已追加分组对比到: %s", delta_md)

    if args.save_plots:
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib 不可用，跳过图表")
        else:
            plot_overall_gauc_compare(overall_df, output_dir / "overall_gauc_compare.png")
            if not tab_standard_df.empty:
                plot_tab_delta_gauc(
                    tab_standard_df,
                    output_path=output_dir / "tab_delta_gauc_standard.png",
                    title="Tab ΔGAUC (standard) vs baseline",
                )
            if not tab_random_df.empty:
                plot_tab_delta_gauc(
                    tab_random_df,
                    output_path=output_dir / "tab_delta_gauc_random.png",
                    title="Tab ΔGAUC (random) vs baseline",
                )

    logger.info("实验汇总完成，输出目录: %s", output_dir)
    return 0


def main() -> None:
    args = parse_args()
    setup_logging(args.debug)
    code = run_main(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()

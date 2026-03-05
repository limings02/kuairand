"""
ADS-lite ablation 运行脚本。

用途：
  一次性跑以下实验，不用手工改代码：
  - DIN
  - DIN + PSRG
  - DIN + PSRG + PCRG (G=2/4/8)
  - DIN + PSRG + PCRG + mean_pool vs attn_pool_over_G

用法示例：
  python scripts/run_ads_ablation.py \
    --base_config configs/train_din_psrg_pcrg_mem16gb.yaml \
    --data_root output/processed \
    --meta_root output/meta \
    --run_root output/exp_runs/ads_ablation
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ADS-lite ablations")
    parser.add_argument("--base_config", type=str, default="configs/train_din_psrg_pcrg_mem16gb.yaml")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--meta_root", type=str, default=None)
    parser.add_argument("--vocabs_root", type=str, default=None)
    parser.add_argument("--run_root", type=str, default="output/exp_runs/ads_ablation")
    parser.add_argument("--dry_run", action="store_true", help="仅打印命令，不实际执行")
    return parser.parse_args()


def deep_update(base: dict, updates: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def build_experiments() -> list[dict]:
    return [
        {
            "name": "din",
            "updates": {
                "model": {
                    "variant": "din",
                    "psrg": {"enabled": False},
                    "pcrg": {"enabled": False},
                },
            },
        },
        {
            "name": "din_psrg",
            "updates": {
                "model": {
                    "variant": "din_psrg",
                    "psrg": {"enabled": True},
                    "pcrg": {"enabled": False},
                },
            },
        },
        {
            "name": "din_psrg_pcrg_g2_mean",
            "updates": {
                "model": {
                    "variant": "din_psrg_pcrg",
                    "psrg": {"enabled": True},
                    "pcrg": {"enabled": True, "num_queries": 2, "aggregation": "mean_pool"},
                },
            },
        },
        {
            "name": "din_psrg_pcrg_g4_mean",
            "updates": {
                "model": {
                    "variant": "din_psrg_pcrg",
                    "psrg": {"enabled": True},
                    "pcrg": {"enabled": True, "num_queries": 4, "aggregation": "mean_pool"},
                },
            },
        },
        {
            "name": "din_psrg_pcrg_g8_mean",
            "updates": {
                "model": {
                    "variant": "din_psrg_pcrg",
                    "psrg": {"enabled": True},
                    "pcrg": {"enabled": True, "num_queries": 8, "aggregation": "mean_pool"},
                },
            },
        },
        {
            "name": "din_psrg_pcrg_g4_attn_pool_over_G",
            "updates": {
                "model": {
                    "variant": "din_psrg_pcrg",
                    "psrg": {"enabled": True},
                    "pcrg": {"enabled": True, "num_queries": 4, "aggregation": "attn_pool_over_G"},
                },
            },
        },
    ]


def main() -> None:
    args = parse_args()

    with open(args.base_config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    if "training" not in base_cfg and "train" in base_cfg:
        base_cfg["training"] = base_cfg["train"]

    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    generated_cfg_dir = run_root / "generated_configs"
    generated_cfg_dir.mkdir(parents=True, exist_ok=True)

    experiments = build_experiments()

    for exp in experiments:
        exp_name = exp["name"]
        cfg = deep_update(base_cfg, exp["updates"])

        if args.data_root:
            cfg["data"]["data_root"] = args.data_root
        if args.meta_root:
            cfg["data"]["meta_root"] = args.meta_root
        if args.vocabs_root:
            cfg["data"]["vocabs_root"] = args.vocabs_root

        run_dir = run_root / exp_name
        cfg["run_name"] = f"ads_ablation_{exp_name}"
        cfg["run_dir"] = str(run_dir)

        cfg_path = generated_cfg_dir / f"{exp_name}.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

        cmd = [
            sys.executable,
            "-m",
            "src.main_train_din",
            "--config",
            str(cfg_path),
            "--run_dir",
            str(run_dir),
        ]

        print("=" * 80)
        print(f"[Ablation] {exp_name}")
        print("Command:", " ".join(cmd))

        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

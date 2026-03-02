from __future__ import annotations

import json
from pathlib import Path


def build_notebook(
    notebook_path: str,
    data_dir: str,
    out_dir: str,
    engine: str,
    chunksize: int,
    sample_users: int | None,
    seed: int,
) -> str:
    """
    生成可一键重跑的 EDA Notebook（轻量模板）。
    """
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# KuaiRand EDA Notebook\\n",
                    "\\n",
                    "该 Notebook 调用 `run_eda.py` 一键生成报告与图表。\\n",
                    "\\n",
                    "请确认本机已安装依赖，并把 `data_dir` 改为你的 KuaiRand 数据目录。\\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    f"data_dir = r\"{data_dir}\"\\n",
                    f"out_dir = r\"{out_dir}\"\\n",
                    f"engine = \"{engine}\"\\n",
                    f"chunksize = {chunksize}\\n",
                    f"sample_users = {sample_users if sample_users else 'None'}\\n",
                    f"seed = {seed}\\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import subprocess\\n",
                    "cmd = [\\n",
                    "    \"python\", \"run_eda.py\",\\n",
                    "    \"--data_dir\", data_dir,\\n",
                    "    \"--out_dir\", out_dir,\\n",
                    "    \"--engine\", engine,\\n",
                    "    \"--chunksize\", str(chunksize),\\n",
                    "    \"--seed\", str(seed),\\n",
                    "]\\n",
                    "if sample_users is not None:\\n",
                    "    cmd += [\"--sample_users\", str(sample_users)]\\n",
                    "print(\" \".join(cmd))\\n",
                    "subprocess.run(cmd, check=True)\\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\\n",
                    "report = Path(out_dir) / \"report.md\"\\n",
                    "print(\"Report:\", report.resolve())\\n",
                    "print(report.read_text(encoding='utf-8')[:2000])\\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    p = Path(notebook_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


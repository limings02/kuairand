"""
随机种子设定工具。

固定 Python / NumPy / PyTorch / CUDA 的随机种子以确保实验可复现。
"""

import os
import random
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    统一设置所有随机种子，确保实验可复现。

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 关闭 cuDNN 非确定性算法（牺牲少量速度换取可复现性）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("随机种子已设置为 %d", seed)

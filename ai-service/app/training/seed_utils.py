"""
Shared seeding utilities for training and evaluation code.

This module centralises seeding of Python's ``random`` module, NumPy and
PyTorch so that training/evaluation jobs can enable reproducible runs
from a single integer seed.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_all(seed: int, *, enable_cudnn_determinism: bool = True) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible experiments.

    Args:
        seed: Integer seed to use for all RNGs.
        enable_cudnn_determinism: When True (default), configure cuDNN
            for deterministic behaviour at the cost of some performance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if enable_cudnn_determinism:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
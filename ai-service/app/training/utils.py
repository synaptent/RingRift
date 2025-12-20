"""
Shared utilities for training infrastructure.

This module provides commonly-used utility functions to avoid duplication
across the training codebase.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Lazy torch import cache
_torch: Any | None = None


def get_torch():
    """
    Lazy import of torch module.

    This pattern avoids importing torch at module load time, which can
    prevent OOM issues in orchestrator processes that don't need torch.

    Returns:
        The torch module
    """
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def get_device(prefer_cuda: bool = True, device_id: int | None = None):
    """
    Get the best available compute device.

    Args:
        prefer_cuda: If True, prefer CUDA if available
        device_id: Specific CUDA device ID to use (if CUDA available)

    Returns:
        torch.device for computation
    """
    torch = get_torch()

    if prefer_cuda and torch.cuda.is_available():
        if device_id is not None:
            return torch.device(f'cuda:{device_id}')
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_device_info() -> dict:
    """
    Get information about available compute devices.

    Returns:
        Dictionary with device info including:
        - cuda_available: bool
        - cuda_device_count: int
        - mps_available: bool
        - recommended_device: str
    """
    torch = get_torch()

    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
    }

    if info['cuda_available']:
        info['recommended_device'] = 'cuda'
        info['cuda_devices'] = [
            {
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(info['cuda_device_count'])
        ]
    elif info['mps_available']:
        info['recommended_device'] = 'mps'
    else:
        info['recommended_device'] = 'cpu'

    return info


def clear_cuda_cache():
    """Clear CUDA cache to free memory."""
    torch = get_torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def seed_all(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    torch = get_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

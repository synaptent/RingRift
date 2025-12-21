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
    """Get the best available compute device.

    .. deprecated:: 2025-12
        Use ``app.utils.torch_utils.get_device`` instead, which provides
        a unified interface with distributed training support.

    Args:
        prefer_cuda: If True, prefer CUDA if available
        device_id: Specific CUDA device ID to use (if CUDA available)

    Returns:
        torch.device for computation
    """
    import warnings
    warnings.warn(
        "app.training.utils.get_device is deprecated since 2025-12. "
        "Use app.utils.torch_utils.get_device instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Delegate to canonical implementation
    from app.utils.torch_utils import get_device as _get_device
    return _get_device(prefer_gpu=prefer_cuda, device_id=device_id)


def get_device_info() -> dict:
    """Get information about available compute devices.

    .. deprecated:: 2025-12
        Use ``app.utils.torch_utils.get_device_info`` instead.

    Returns:
        Dictionary with device info including:
        - cuda_available: bool
        - cuda_device_count: int
        - mps_available: bool
        - recommended_device: str
    """
    import warnings
    warnings.warn(
        "app.training.utils.get_device_info is deprecated since 2025-12. "
        "Use app.utils.torch_utils.get_device_info instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Delegate to canonical implementation
    from app.utils.torch_utils import get_device_info as _get_device_info
    return _get_device_info()


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

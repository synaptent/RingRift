"""Utility modules for RingRift AI.

This package provides reusable utilities harvested from archived debug scripts
and consolidated for maintainability.

Modules:
    debug_utils: State comparison and parity debugging utilities
    torch_utils: Safe PyTorch operations including device detection (canonical)

Device Management (Canonical Exports):
    get_device: Auto-detect best compute device (CUDA/MPS/CPU)
    get_device_info: Get detailed device information
"""

from __future__ import annotations

# Canonical device management exports
from app.utils.torch_utils import get_device, get_device_info

__all__ = [
    "debug_utils",
    "torch_utils",
    # Device management
    "get_device",
    "get_device_info",
]

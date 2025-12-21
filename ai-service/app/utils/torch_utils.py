"""Utilities for safe PyTorch operations.

This module provides secure wrappers around PyTorch functions that may
have security implications, particularly around model loading.

Security Note:
    torch.load with weights_only=False can execute arbitrary code during
    unpickling. This module provides safe_load_checkpoint which:
    1. First tries weights_only=True (safe mode)
    2. Falls back to weights_only=False only when necessary
    3. Logs a warning when using unsafe mode
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Try to import torch - this module should work even without torch
skip_torch = os.getenv("RINGRIFT_SKIP_TORCH_IMPORT", "").strip().lower()
skip_optional = os.getenv("RINGRIFT_SKIP_OPTIONAL_IMPORTS", "").strip().lower()
if skip_torch in ("1", "true", "yes", "on") or skip_optional in ("1", "true", "yes", "on"):
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]
else:
    try:
        import torch
        HAS_TORCH = True
    except Exception as exc:
        HAS_TORCH = False
        torch = None  # type: ignore[assignment]
        logger.debug("PyTorch import failed: %s", exc)


def safe_load_checkpoint(
    path: str | Path,
    *,
    map_location: str | None = "cpu",
    allow_unsafe: bool = True,
    warn_on_unsafe: bool = True,
) -> dict[str, Any]:
    """Safely load a PyTorch checkpoint.

    This function attempts to load checkpoints in the safest way possible:
    1. First tries with weights_only=True (prevents arbitrary code execution)
    2. If that fails and allow_unsafe=True, falls back to weights_only=False

    Args:
        path: Path to the checkpoint file
        map_location: Device to map tensors to (default: "cpu")
        allow_unsafe: Whether to allow fallback to unsafe loading
        warn_on_unsafe: Whether to log a warning when using unsafe loading

    Returns:
        The loaded checkpoint dictionary

    Raises:
        ImportError: If PyTorch is not installed
        FileNotFoundError: If the checkpoint file doesn't exist
        RuntimeError: If loading fails and allow_unsafe=False
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for checkpoint loading")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Try safe loading first
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        return checkpoint
    except Exception as safe_error:
        if not allow_unsafe:
            raise RuntimeError(
                f"Failed to load checkpoint with weights_only=True: {safe_error}. "
                "Set allow_unsafe=True to allow unsafe loading."
            ) from safe_error

        # Fall back to unsafe loading for legacy checkpoints
        if warn_on_unsafe:
            logger.warning(
                "Loading checkpoint with weights_only=False (unsafe mode). "
                "This checkpoint may contain non-tensor data. Path: %s",
                path,
            )

        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=False)
            return checkpoint
        except TypeError:
            # Very old PyTorch versions don't support weights_only
            checkpoint = torch.load(path, map_location=map_location)
            return checkpoint


def load_state_dict_only(
    path: str | Path,
    *,
    map_location: str | None = "cpu",
) -> dict[str, Any]:
    """Load only the state_dict from a checkpoint (safest mode).

    This is the safest way to load model weights - it only loads the
    state_dict and ignores any other data in the checkpoint.

    Args:
        path: Path to the checkpoint file
        map_location: Device to map tensors to

    Returns:
        The model state_dict

    Raises:
        KeyError: If the checkpoint doesn't contain a state_dict
    """
    checkpoint = safe_load_checkpoint(
        path,
        map_location=map_location,
        allow_unsafe=True,  # May need for legacy checkpoints
        warn_on_unsafe=False,  # We're only extracting state_dict anyway
    )

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        elif "model" in checkpoint:
            return checkpoint["model"]
        else:
            # Assume the whole dict is the state_dict
            return checkpoint
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")


def save_checkpoint_safe(
    checkpoint: dict[str, Any],
    path: str | Path,
    *,
    use_new_format: bool = True,
) -> None:
    """Save a checkpoint in a secure format.

    Args:
        checkpoint: The checkpoint dictionary to save
        path: Path to save the checkpoint
        use_new_format: If True, use torch.save with _use_new_zipfile_serialization
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for checkpoint saving")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if use_new_format:
        torch.save(checkpoint, path, _use_new_zipfile_serialization=True)
    else:
        torch.save(checkpoint, path)


# =============================================================================
# Device Detection and Management (Canonical Implementation)
# =============================================================================
#
# This is the canonical implementation for device detection across the codebase.
# Other modules should import from here:
#   from app.utils.torch_utils import get_device, get_device_info
#
# This consolidates duplicate implementations from:
# - app/ai/gpu_batch.py (deprecated, delegates here)
# - app/ai/gpu_kernels.py (deprecated, delegates here)
# - app/training/utils.py (deprecated, delegates here)
# - app/training/train_setup.py (uses local_rank variant)
# =============================================================================


def get_device(
    prefer_gpu: bool = True,
    device_id: int | None = None,
    local_rank: int = -1,
) -> Any:
    """Get the best available compute device (canonical implementation).

    This is the consolidated device detection function. It supports:
    - CUDA (NVIDIA GPUs) with device selection
    - MPS (Apple Silicon)
    - CPU fallback
    - Distributed training (local_rank)

    Priority order when prefer_gpu=True:
    1. CUDA with specified device_id or local_rank
    2. MPS (Apple Silicon)
    3. CPU (fallback)

    Args:
        prefer_gpu: Whether to prefer GPU over CPU
        device_id: Specific CUDA device ID to use (ignored for MPS/CPU)
        local_rank: Local rank for distributed training (-1 for single GPU).
                    If >= 0, overrides device_id.

    Returns:
        torch.device for the selected compute device

    Example:
        # Simple usage - auto-detect best device
        device = get_device()

        # Force CPU
        device = get_device(prefer_gpu=False)

        # Specific GPU
        device = get_device(device_id=1)

        # Distributed training
        device = get_device(local_rank=int(os.environ.get('LOCAL_RANK', -1)))
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for device detection")

    # Distributed training takes precedence
    if local_rank >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")

    if prefer_gpu:
        if torch.cuda.is_available():
            cuda_id = device_id if device_id is not None else 0
            device = torch.device(f"cuda:{cuda_id}")
            try:
                props = torch.cuda.get_device_properties(cuda_id)
                logger.debug(
                    "Using CUDA device %d: %s (%.1fGB)",
                    cuda_id, props.name, props.total_memory / 1024**3
                )
            except Exception:
                pass  # Logging is optional
            return device

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.debug("Using MPS (Apple Silicon)")
            return torch.device("mps")

    logger.debug("Using CPU")
    return torch.device("cpu")


def get_device_info() -> dict[str, Any]:
    """Get information about available compute devices.

    Returns:
        Dictionary with device information including:
        - cuda_available: bool
        - cuda_device_count: int
        - cuda_devices: list of device info dicts (if CUDA available)
        - mps_available: bool
        - recommended_device: str ('cuda', 'mps', or 'cpu')
        - torch_available: bool

    Example:
        info = get_device_info()
        if info['cuda_available']:
            print(f"Found {info['cuda_device_count']} CUDA devices")
            for dev in info['cuda_devices']:
                print(f"  GPU {dev['id']}: {dev['name']} ({dev['memory_gb']:.1f}GB)")
    """
    info: dict[str, Any] = {
        "torch_available": HAS_TORCH,
        "cuda_available": False,
        "cuda_device_count": 0,
        "mps_available": False,
        "recommended_device": "cpu",
    }

    if not HAS_TORCH:
        return info

    info["cuda_available"] = torch.cuda.is_available()
    info["cuda_device_count"] = torch.cuda.device_count() if info["cuda_available"] else 0
    info["mps_available"] = (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )

    if info["cuda_available"]:
        info["recommended_device"] = "cuda"
        info["cuda_devices"] = []
        for i in range(info["cuda_device_count"]):
            try:
                props = torch.cuda.get_device_properties(i)
                info["cuda_devices"].append({
                    "id": i,
                    "name": props.name,
                    "memory_gb": props.total_memory / 1024**3,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count,
                })
            except Exception:
                info["cuda_devices"].append({"id": i, "name": "Unknown", "memory_gb": 0})
    elif info["mps_available"]:
        info["recommended_device"] = "mps"

    return info

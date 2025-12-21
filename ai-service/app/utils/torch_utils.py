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


# =============================================================================
# Checkpoint Version Validation
# =============================================================================

# Current checkpoint schema version
# Increment when checkpoint format changes in breaking ways
CHECKPOINT_SCHEMA_VERSION = "2.0"

# Minimum compatible schema version for loading
CHECKPOINT_MIN_COMPATIBLE_VERSION = "1.0"


class CheckpointVersionError(Exception):
    """Raised when checkpoint version is incompatible."""
    pass


class CheckpointVersionWarning(UserWarning):
    """Warning for checkpoint version mismatches."""
    pass


def validate_checkpoint_version(
    checkpoint: dict[str, Any],
    *,
    expected_version: str | None = None,
    min_version: str | None = None,
    strict: bool = False,
    log_info: bool = True,
) -> dict[str, Any]:
    """Validate checkpoint version compatibility.

    Checks if a loaded checkpoint is compatible with the current version.
    This helps catch issues when loading old checkpoints with new code
    or vice versa.

    Args:
        checkpoint: Loaded checkpoint dictionary
        expected_version: If set, require exact version match
        min_version: If set, require version >= this (default: CHECKPOINT_MIN_COMPATIBLE_VERSION)
        strict: If True, raise error on version mismatch. If False, warn only.
        log_info: If True, log checkpoint version info

    Returns:
        The checkpoint dictionary (unchanged, for chaining)

    Raises:
        CheckpointVersionError: If strict=True and version is incompatible

    Example:
        checkpoint = safe_load_checkpoint("model.pt")
        validate_checkpoint_version(checkpoint, min_version="1.5")
    """
    import warnings

    # Get version from checkpoint
    ckpt_version = checkpoint.get("schema_version") or checkpoint.get("version") or "unknown"
    ckpt_created = checkpoint.get("created_at") or checkpoint.get("timestamp") or "unknown"
    ckpt_model_class = checkpoint.get("model_class") or checkpoint.get("architecture") or "unknown"

    if log_info:
        logger.info(
            "[Checkpoint] Version: %s, Created: %s, Model: %s",
            ckpt_version, ckpt_created, ckpt_model_class
        )

    # If version is unknown, we can't validate
    if ckpt_version == "unknown":
        if strict:
            raise CheckpointVersionError(
                "Checkpoint has no version information. "
                "Cannot validate compatibility with strict=True."
            )
        logger.warning(
            "[Checkpoint] No version information found. "
            "This checkpoint may be from an older version."
        )
        return checkpoint

    # Parse version for comparison
    try:
        ckpt_parts = _parse_version(ckpt_version)
    except ValueError:
        if strict:
            raise CheckpointVersionError(f"Invalid checkpoint version format: {ckpt_version}")
        logger.warning(f"[Checkpoint] Invalid version format: {ckpt_version}")
        return checkpoint

    # Check exact version match if requested
    if expected_version is not None:
        expected_parts = _parse_version(expected_version)
        if ckpt_parts != expected_parts:
            msg = (
                f"Checkpoint version mismatch: expected {expected_version}, "
                f"got {ckpt_version}"
            )
            if strict:
                raise CheckpointVersionError(msg)
            warnings.warn(msg, CheckpointVersionWarning)
            logger.warning(f"[Checkpoint] {msg}")

    # Check minimum version
    min_ver = min_version or CHECKPOINT_MIN_COMPATIBLE_VERSION
    min_parts = _parse_version(min_ver)
    if ckpt_parts < min_parts:
        msg = (
            f"Checkpoint version {ckpt_version} is older than minimum "
            f"compatible version {min_ver}. Loading may fail or produce "
            "incorrect results."
        )
        if strict:
            raise CheckpointVersionError(msg)
        warnings.warn(msg, CheckpointVersionWarning)
        logger.warning(f"[Checkpoint] {msg}")

    return checkpoint


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse version string to tuple of integers for comparison.

    Args:
        version_str: Version string like "1.0" or "2.1.3"

    Returns:
        Tuple of integers, e.g., (2, 1, 3)

    Raises:
        ValueError: If version string is invalid
    """
    parts = version_str.strip().split(".")
    return tuple(int(p) for p in parts)


def add_checkpoint_metadata(
    checkpoint: dict[str, Any],
    *,
    model_class: str | None = None,
    board_type: str | None = None,
    num_players: int | None = None,
    training_config: dict | None = None,
) -> dict[str, Any]:
    """Add version and metadata to a checkpoint before saving.

    This should be called before saving a checkpoint to ensure
    it contains version information for future compatibility checks.

    Args:
        checkpoint: Checkpoint dictionary to augment
        model_class: Name of the model class (e.g., "RingRiftCNN_v4")
        board_type: Board type string (e.g., "square8", "hexagonal")
        num_players: Number of players the model was trained for
        training_config: Optional training configuration dict

    Returns:
        Augmented checkpoint dictionary

    Example:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = add_checkpoint_metadata(
            checkpoint,
            model_class="RingRiftCNN_v4",
            board_type="square8",
            num_players=2,
        )
        save_checkpoint_safe(checkpoint, "model.pt")
    """
    from datetime import datetime

    checkpoint["schema_version"] = CHECKPOINT_SCHEMA_VERSION
    checkpoint["created_at"] = datetime.now().isoformat()

    if model_class is not None:
        checkpoint["model_class"] = model_class
    if board_type is not None:
        checkpoint["board_type"] = board_type
    if num_players is not None:
        checkpoint["num_players"] = num_players
    if training_config is not None:
        checkpoint["training_config"] = training_config

    return checkpoint


def get_checkpoint_info(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Extract metadata from a checkpoint.

    Args:
        checkpoint: Loaded checkpoint dictionary

    Returns:
        Dictionary with checkpoint metadata:
        - schema_version: Version string or "unknown"
        - created_at: Creation timestamp or "unknown"
        - model_class: Model class name or "unknown"
        - board_type: Board type or "unknown"
        - num_players: Number of players or None
        - has_optimizer: Whether optimizer state is present
        - has_scheduler: Whether scheduler state is present
        - has_training_config: Whether training config is present
        - keys: List of top-level keys in checkpoint
    """
    return {
        "schema_version": checkpoint.get("schema_version") or checkpoint.get("version") or "unknown",
        "created_at": checkpoint.get("created_at") or checkpoint.get("timestamp") or "unknown",
        "model_class": checkpoint.get("model_class") or checkpoint.get("architecture") or "unknown",
        "board_type": checkpoint.get("board_type") or "unknown",
        "num_players": checkpoint.get("num_players"),
        "has_optimizer": "optimizer_state_dict" in checkpoint or "optimizer" in checkpoint,
        "has_scheduler": "scheduler_state_dict" in checkpoint or "scheduler" in checkpoint,
        "has_training_config": "training_config" in checkpoint,
        "keys": list(checkpoint.keys()),
    }


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

"""Checkpoint utility functions for atomic file operations.

This module consolidates the atomic save pattern that was previously
duplicated across:
- app/training/checkpointing.py (lines 204-218, 361-370)
- app/training/model_versioning.py (lines 529-557, 885-896)
- app/training/checkpoint_unified.py (lines 370-382)

Usage:
    from app.training.checkpoint_utils import atomic_save, atomic_torch_save

    # Low-level atomic save for any data
    atomic_save(
        save_func=lambda p: torch.save(data, p),
        file_path=Path("model.pth"),
        verify_hash=True,
    )

    # Higher-level atomic torch.save
    atomic_torch_save(checkpoint_dict, Path("model.pth"))

December 2025: Centralized atomic save pattern.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default temp file suffix
TEMP_SUFFIX = '.tmp'

# Hash algorithms supported
HASH_SHA256 = 'sha256'


def compute_file_hash(file_path: Path, algorithm: str = HASH_SHA256) -> str:
    """Compute hash of a file.

    Args:
        file_path: Path to the file
        algorithm: Hash algorithm (default: sha256)

    Returns:
        Hex-encoded hash string
    """
    if algorithm != HASH_SHA256:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Read in chunks for memory efficiency with large files
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def fsync_file(file_path: Path) -> bool:
    """Ensure file is flushed to disk.

    Important for network filesystems (NFS) and preventing data loss
    on sudden power failure.

    Args:
        file_path: Path to the file to sync

    Returns:
        True if fsync succeeded, False if fell back to global sync
    """
    try:
        with open(file_path, 'rb') as f:
            os.fsync(f.fileno())
        return True
    except OSError as e:
        # fsync may fail on some filesystems (e.g., NFS without sync option)
        # Fall back to global sync as a last resort
        logger.debug(f"fsync failed ({e}), using os.sync() fallback")
        try:
            os.sync()
        except (OSError, AttributeError):
            # os.sync() not available on all platforms (e.g., Windows)
            pass
        return False


def atomic_save(
    save_func: Callable[[Path], None],
    file_path: Path,
    temp_suffix: str = TEMP_SUFFIX,
    sync_to_disk: bool = True,
    verify_hash: bool = False,
) -> str | None:
    """Atomically save a file using temp file + rename pattern.

    This prevents file corruption from:
    - Interrupted writes (power failure, process kill)
    - Concurrent readers seeing partial data
    - Network filesystem issues

    Args:
        save_func: Function that writes data to a path (receives temp path)
        file_path: Final destination path
        temp_suffix: Suffix for temporary file (default: '.tmp')
        sync_to_disk: Whether to fsync before rename (default: True)
        verify_hash: Whether to compute and return hash after save

    Returns:
        File hash if verify_hash=True, else None

    Raises:
        RuntimeError: If save fails (temp file is cleaned up)

    Example:
        >>> atomic_save(
        ...     save_func=lambda p: torch.save(data, p),
        ...     file_path=Path("model.pth"),
        ... )
    """
    file_path = Path(file_path)
    temp_path = file_path.with_suffix(file_path.suffix + temp_suffix)

    try:
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to temp file
        save_func(temp_path)

        # Sync to disk to ensure durability
        if sync_to_disk:
            fsync_file(temp_path)

        # Atomically rename temp to final
        temp_path.rename(file_path)

        # Compute hash if requested
        if verify_hash:
            return compute_file_hash(file_path)
        return None

    except Exception as e:
        # Clean up temp file on failure
        if temp_path.exists():
            with contextlib.suppress(OSError):
                temp_path.unlink()
        raise RuntimeError(f"Failed to save {file_path}: {e}") from e


def atomic_torch_save(
    data: dict[str, Any],
    file_path: Path,
    sync_to_disk: bool = True,
    verify_hash: bool = False,
) -> str | None:
    """Atomically save PyTorch data using temp file + rename pattern.

    Convenience wrapper around atomic_save for torch.save operations.

    Args:
        data: Dictionary to save with torch.save
        file_path: Final destination path
        sync_to_disk: Whether to fsync before rename
        verify_hash: Whether to compute and return hash after save

    Returns:
        File hash if verify_hash=True, else None

    Example:
        >>> checkpoint = {'model': model.state_dict(), 'epoch': 10}
        >>> atomic_torch_save(checkpoint, Path("checkpoint.pth"))
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch required for atomic_torch_save")

    return atomic_save(
        save_func=lambda p: torch.save(data, p),
        file_path=file_path,
        sync_to_disk=sync_to_disk,
        verify_hash=verify_hash,
    )


def load_with_validation(
    file_path: Path,
    expected_hash: str | None = None,
) -> tuple[Any, str]:
    """Load a PyTorch checkpoint with optional hash validation.

    Args:
        file_path: Path to the checkpoint file
        expected_hash: Expected SHA256 hash (if provided, validates before loading)

    Returns:
        Tuple of (loaded data, actual hash)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If hash doesn't match expected
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch required for load_with_validation")

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {file_path}")

    # Compute hash before loading
    actual_hash = compute_file_hash(file_path)

    # Validate hash if expected
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            f"Checkpoint hash mismatch: expected {expected_hash[:16]}..., "
            f"got {actual_hash[:16]}..."
        )

    # Load the checkpoint
    data = torch.load(file_path, weights_only=False)

    return data, actual_hash


__all__ = [
    'HASH_SHA256',
    # Constants
    'TEMP_SUFFIX',
    # Core functions
    'atomic_save',
    'atomic_torch_save',
    # Utilities
    'compute_file_hash',
    'fsync_file',
    'load_with_validation',
]

"""Utilities for safe NumPy operations.

This module provides secure wrappers around NumPy functions that may
have security implications, particularly around NPZ file loading.

Security Note:
    np.load with allow_pickle=True can execute arbitrary code during
    unpickling of object arrays. This module provides safe_load_npz which:
    1. First tries allow_pickle=False (safe mode)
    2. Falls back to allow_pickle=True only when necessary for object arrays
    3. Logs a warning when using unsafe mode

Usage:
    from app.utils.numpy_utils import safe_load_npz

    # Safe by default - tries without pickle first
    data = safe_load_npz("training_data.npz")

    # For untrusted files - enforce safe mode (will fail on object arrays)
    data = safe_load_npz(external_file, allow_unsafe=False)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "safe_load_npz",
    "EXPECTED_TRAINING_KEYS",
]

logger = logging.getLogger(__name__)


# Expected keys in training NPZ files - these should all be numeric arrays
EXPECTED_TRAINING_KEYS = frozenset({
    # Core training data
    "features",
    "policy",
    "policies",  # Alternative name
    "value",
    "values",  # Alternative name

    # Optional arrays
    "game_ids",
    "move_numbers",
    "weights",
    "sample_weights",
    "quality_scores",

    # Metadata (may require pickle if stored as object arrays)
    "board_type",
    "num_players",
    "metadata",
})


def safe_load_npz(
    path: str | Path,
    *,
    mmap_mode: str | None = None,
    allow_unsafe: bool = True,
    warn_on_unsafe: bool = True,
    expected_keys: frozenset[str] | None = None,
) -> np.lib.npyio.NpzFile:
    """Safely load an NPZ file with security checks.

    This function attempts to load NPZ files in the safest way possible:
    1. First tries with allow_pickle=False (prevents arbitrary code execution)
    2. If that fails and allow_unsafe=True, falls back to allow_pickle=True

    Args:
        path: Path to the NPZ file
        mmap_mode: Memory-map mode (None, 'r', 'r+', 'c'). Use 'r' for large files.
        allow_unsafe: Whether to allow fallback to pickle-enabled loading
        warn_on_unsafe: Whether to log a warning when using pickle loading
        expected_keys: If provided, validate that only these keys are present

    Returns:
        NpzFile object (use as context manager or access arrays via indexing)

    Raises:
        FileNotFoundError: If the file doesn't exist
        RuntimeError: If loading fails and allow_unsafe=False
        ValueError: If unexpected keys are found and expected_keys is provided

    Example:
        # Basic usage
        with safe_load_npz("training.npz") as data:
            features = data["features"]
            policy = data["policy"]

        # For large files, use memory mapping
        with safe_load_npz("large_training.npz", mmap_mode="r") as data:
            # Arrays are memory-mapped, not fully loaded
            features = data["features"]

        # Strict mode for untrusted files
        data = safe_load_npz(untrusted_path, allow_unsafe=False)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NPZ file not found: {path}")

    # Try safe loading first (no pickle)
    try:
        data = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)

        # Force eager loading of all arrays to detect object arrays that need pickle.
        # NpzFile loads arrays lazily, so ValueError is raised when accessing, not loading.
        for key in data.files:
            _ = data[key]  # This triggers the actual load

        # Validate keys if requested
        if expected_keys is not None:
            _validate_npz_keys(data, expected_keys, path)

        return data

    except ValueError as safe_error:
        # ValueError is raised when pickle is required but disallowed
        if "allow_pickle=False" not in str(safe_error) and "pickle" not in str(safe_error).lower():
            # Not a pickle-related error, re-raise
            raise

        if not allow_unsafe:
            raise RuntimeError(
                f"NPZ file requires pickle for object arrays: {path}. "
                "This file may contain Python objects that need pickle to load. "
                "Set allow_unsafe=True to allow loading (security risk for untrusted files)."
            ) from safe_error

        # Fall back to pickle-enabled loading
        if warn_on_unsafe:
            logger.warning(
                "[NumPy] Loading NPZ with allow_pickle=True (contains object arrays). "
                "This is potentially unsafe for untrusted files. Path: %s",
                path,
            )

        data = np.load(path, mmap_mode=mmap_mode, allow_pickle=True)

        # Validate keys if requested
        if expected_keys is not None:
            _validate_npz_keys(data, expected_keys, path)

        return data


def _validate_npz_keys(
    data: np.lib.npyio.NpzFile,
    expected_keys: frozenset[str],
    path: Path,
) -> None:
    """Validate that NPZ file contains only expected keys.

    Args:
        data: Loaded NPZ file
        expected_keys: Set of allowed key names
        path: Path for error messages

    Raises:
        ValueError: If unexpected keys are found
    """
    actual_keys = set(data.files)
    unexpected = actual_keys - expected_keys

    if unexpected:
        logger.warning(
            "[NumPy] NPZ file contains unexpected keys: %s (file: %s)",
            unexpected, path
        )


def load_training_npz(
    path: str | Path,
    *,
    mmap_mode: str | None = None,
    validate_keys: bool = True,
) -> np.lib.npyio.NpzFile:
    """Load a training data NPZ file with appropriate defaults.

    Convenience wrapper around safe_load_npz specifically for training data.
    Training NPZ files typically contain only numeric arrays (features, policy,
    value) and should load without pickle.

    Args:
        path: Path to the training NPZ file
        mmap_mode: Memory-map mode for large files (None, 'r', 'r+', 'c')
        validate_keys: Whether to warn about unexpected keys

    Returns:
        NpzFile object containing training arrays

    Example:
        with load_training_npz("data/training/hex8_2p.npz", mmap_mode="r") as data:
            print(f"Features shape: {data['features'].shape}")
            print(f"Policy shape: {data['policy'].shape}")
    """
    return safe_load_npz(
        path,
        mmap_mode=mmap_mode,
        allow_unsafe=True,  # Legacy files may have object arrays
        warn_on_unsafe=True,
        expected_keys=EXPECTED_TRAINING_KEYS if validate_keys else None,
    )

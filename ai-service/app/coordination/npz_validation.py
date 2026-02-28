"""NPZ-specific validation beyond checksum verification.

This module provides deep validation of NPZ training data files:
1. File can be opened as valid NPZ
2. Required arrays exist (features, policy*, values)
3. Array shapes are consistent (all have same sample count)
4. Data types are correct
5. No corrupted/truncated arrays

This catches corruption issues that checksum-only validation misses,
such as the December 2025 incident where rsync --partial stitched together
corrupted segments resulting in an array with 22 billion elements instead
of the expected 6.3 million.

Usage:
    from app.coordination.npz_validation import validate_npz_structure, NPZValidationResult

    result = validate_npz_structure(Path("data/training/hex8_2p.npz"))
    if not result.valid:
        print(f"Validation failed: {result.errors}")
    else:
        print(f"Valid NPZ with {result.sample_count} samples")

Integration:
    This module is used by:
    - app/distributed/resilient_transfer.py (post-transfer validation)
    - app/coordination/npz_distribution_daemon.py (before distribution)
    - scripts/export_replay_dataset.py (after export)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Maximum reasonable sample count for training data
# 100M samples is extremely large; anything larger is likely corruption
MAX_REASONABLE_SAMPLES = 100_000_000

# Maximum reasonable array dimension
# Arrays with dimensions > 1B are almost certainly corrupted
MAX_REASONABLE_DIMENSION = 1_000_000_000

# Required arrays for training data
REQUIRED_ARRAYS = ["features", "values"]

# Policy array prefixes (policy_logits, policy_mask, etc.)
POLICY_PREFIXES = ["policy"]

# Expected data types for each array type
EXPECTED_DTYPES = {
    "features": ["float32", "float16", "int8", "uint8"],
    "values": ["float32", "float16"],
    "policy_logits": ["float32", "float16"],
    "policy_mask": ["bool", "int8", "uint8", "float32"],
    "move_indices": ["int32", "int64"],
}


@dataclass
class NPZValidationResult:
    """Result of NPZ file validation.

    Attributes:
        valid: Whether the file passed all validation checks
        sample_count: Number of samples in the file (0 if invalid)
        errors: List of critical errors that make the file unusable
        warnings: List of non-critical issues
        array_shapes: Dictionary of array names to their shapes
        array_dtypes: Dictionary of array names to their data types
        file_size: Size of the NPZ file in bytes
    """

    valid: bool
    sample_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    array_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    array_dtypes: dict[str, str] = field(default_factory=dict)
    file_size: int = 0

    def summary(self) -> str:
        """Return a human-readable summary of the validation result."""
        if self.valid:
            return (
                f"Valid NPZ: {self.sample_count} samples, "
                f"{len(self.array_shapes)} arrays, {self.file_size / 1024 / 1024:.1f}MB"
            )
        else:
            return f"Invalid NPZ: {'; '.join(self.errors)}"


def validate_npz_structure(
    path: str | Path,
    require_policy: bool = True,
    max_samples: int = MAX_REASONABLE_SAMPLES,
) -> NPZValidationResult:
    """Validate NPZ file structure and content.

    This function performs deep validation of NPZ training data:
    1. File opens successfully as valid NPZ
    2. Required arrays present (features, values, optionally policy_*)
    3. Sample counts match across all arrays
    4. Array dimensions are reasonable (not corrupted)
    5. Data types are appropriate

    Args:
        path: Path to the NPZ file
        require_policy: Whether to require policy arrays (default True)
        max_samples: Maximum reasonable sample count (default 100M)

    Returns:
        NPZValidationResult with validation status and details

    Example:
        >>> result = validate_npz_structure(Path("training.npz"))
        >>> if not result.valid:
        ...     for error in result.errors:
        ...         print(f"ERROR: {error}")
    """
    result = NPZValidationResult(valid=False)
    path = Path(path)  # Normalize str -> Path

    # Check file exists
    if not path.exists():
        result.errors.append(f"File not found: {path}")
        return result

    # Get file size
    try:
        result.file_size = path.stat().st_size
    except Exception as e:
        result.errors.append(f"Cannot stat file: {e}")
        return result

    # Check file is not empty
    if result.file_size == 0:
        result.errors.append("File is empty (0 bytes)")
        return result

    # Try to open as NPZ
    try:
        import numpy as np

        # Use mmap_mode='r' for memory-efficient reading of large files
        data = np.load(path, allow_pickle=True, mmap_mode="r")
    except Exception as e:
        result.errors.append(f"Cannot open as NPZ: {e}")
        return result

    try:
        # Get list of arrays
        array_names = list(data.files)

        if not array_names:
            result.errors.append("NPZ file contains no arrays")
            return result

        # Check required arrays
        for required in REQUIRED_ARRAYS:
            if required not in array_names:
                result.errors.append(f"Missing required array: {required}")

        # Check for at least one policy array if required
        if require_policy:
            has_policy = any(
                name.startswith(prefix) for name in array_names for prefix in POLICY_PREFIXES
            )
            if not has_policy:
                result.warnings.append("No policy arrays found (policy_logits, policy_mask, etc.)")

        # If we have critical errors, return early
        if result.errors:
            return result

        # Validate each array and collect sample counts
        sample_counts: dict[str, int] = {}

        for name in array_names:
            try:
                arr = data[name]

                # Check if it's actually an array
                if not hasattr(arr, "shape"):
                    result.warnings.append(f"Array '{name}' has no shape attribute")
                    continue

                shape = arr.shape
                dtype = str(arr.dtype)

                result.array_shapes[name] = shape
                result.array_dtypes[name] = dtype

                # Check for corrupted dimensions
                for i, dim in enumerate(shape):
                    if dim > MAX_REASONABLE_DIMENSION:
                        result.errors.append(
                            f"Array '{name}' has unreasonable dimension {i}: {dim} "
                            f"(max {MAX_REASONABLE_DIMENSION}). This indicates corruption."
                        )

                # Get sample count (first dimension)
                if len(shape) > 0:
                    sample_counts[name] = shape[0]

                    # Check for corrupted sample count
                    if shape[0] > max_samples:
                        result.errors.append(
                            f"Array '{name}' has {shape[0]} samples, exceeding maximum "
                            f"of {max_samples}. This indicates corruption."
                        )

                # Validate dtype for known array types
                for array_type, valid_dtypes in EXPECTED_DTYPES.items():
                    if name == array_type or name.startswith(array_type):
                        if dtype not in valid_dtypes:
                            result.warnings.append(
                                f"Array '{name}' has dtype {dtype}, expected one of {valid_dtypes}"
                            )

            except Exception as e:
                result.errors.append(f"Corrupted array '{name}': {e}")

        # If we have critical errors, return early
        if result.errors:
            return result

        # Check sample count consistency
        if sample_counts:
            unique_counts = set(sample_counts.values())

            if len(unique_counts) > 1:
                # Build detailed mismatch message
                count_groups: dict[int, list[str]] = {}
                for name, count in sample_counts.items():
                    if count not in count_groups:
                        count_groups[count] = []
                    count_groups[count].append(name)

                mismatch_details = ", ".join(
                    f"{count}: [{', '.join(names)}]" for count, names in count_groups.items()
                )
                result.errors.append(f"Inconsistent sample counts across arrays: {mismatch_details}")
            else:
                result.sample_count = list(sample_counts.values())[0]

        # If we still have errors, return
        if result.errors:
            return result

        # All checks passed
        result.valid = True

        logger.debug(
            f"NPZ validation passed: {path.name} - "
            f"{result.sample_count} samples, {len(array_names)} arrays"
        )

        return result

    finally:
        # Ensure we close the memory-mapped file
        if hasattr(data, "close"):
            data.close()


def validate_npz_for_training(
    path: Path,
    board_type: str | None = None,
    num_players: int | None = None,
) -> NPZValidationResult:
    """Validate NPZ file for use in training.

    This is a higher-level validation that checks the file is suitable
    for training with specific board type and player count.

    Args:
        path: Path to the NPZ file
        board_type: Expected board type (hex8, square8, etc.)
        num_players: Expected number of players (2, 3, or 4)

    Returns:
        NPZValidationResult with validation status
    """
    # First do basic structure validation
    result = validate_npz_structure(path, require_policy=True)

    if not result.valid:
        return result

    # Additional training-specific checks
    try:
        import numpy as np

        data = np.load(path, allow_pickle=True, mmap_mode="r")

        # Check features array shape matches expected board
        if "features" in data.files and board_type:
            features = data["features"]
            expected_cells = _get_expected_cells(board_type)

            if expected_cells and len(features.shape) >= 2:
                # Features shape is typically (samples, cells, channels) or (samples, channels, cells)
                if features.shape[1] != expected_cells and (
                    len(features.shape) < 3 or features.shape[2] != expected_cells
                ):
                    result.warnings.append(
                        f"Features shape {features.shape} may not match board type {board_type} "
                        f"(expected {expected_cells} cells)"
                    )

        # Check values array shape matches player count
        if "values" in data.files and num_players:
            values = data["values"]
            if len(values.shape) >= 2 and values.shape[1] != num_players:
                result.warnings.append(
                    f"Values shape {values.shape} does not match num_players={num_players}"
                )

        data.close()

    except Exception as e:
        result.warnings.append(f"Could not perform training-specific validation: {e}")

    return result


def _get_expected_cells(board_type: str) -> int | None:
    """Get expected cell count for a board type."""
    cell_counts = {
        "hex8": 61,  # radius 4
        "square8": 64,  # 8x8
        "square19": 361,  # 19x19
        "hexagonal": 469,  # radius 12
    }
    return cell_counts.get(board_type)


def quick_npz_check(path: str | Path) -> tuple[bool, str]:
    """Quick check if NPZ file is likely valid.

    This is a fast check that doesn't fully validate the file,
    suitable for use in hot paths.

    Args:
        path: Path to the NPZ file (str or Path)

    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(path)  # Normalize str -> Path
    if not path.exists():
        return False, "File not found"

    if path.stat().st_size == 0:
        return False, "File is empty"

    try:
        import numpy as np

        # Just try to open and get file list
        data = np.load(path, allow_pickle=True, mmap_mode="r")
        array_names = list(data.files)
        data.close()

        if not array_names:
            return False, "No arrays in file"

        if "features" not in array_names:
            return False, "Missing 'features' array"

        if "values" not in array_names:
            return False, "Missing 'values' array"

        return True, ""

    except Exception as e:
        return False, str(e)

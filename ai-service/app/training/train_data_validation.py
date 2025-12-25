"""
Training data validation utilities.

Extracted from train.py (December 2025) to reduce module size.
Handles NPZ file validation and quality checks before training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

# Try to import validation module
try:
    from .data_validation import validate_npz_file, ValidationResult

    HAS_DATA_VALIDATION = True
except ImportError:
    HAS_DATA_VALIDATION = False
    ValidationResult = None  # type: ignore


@dataclass
class DataValidationResult:
    """Aggregated result of validating multiple data files."""

    all_valid: bool
    total_samples: int
    files_validated: int
    files_with_issues: int
    issues: list[str] = field(default_factory=list)


def validate_training_data(
    data_paths: str | list[str],
    fail_on_invalid: bool = False,
    log_issues: bool = True,
    max_issues_to_log: int = 5,
) -> DataValidationResult:
    """Validate training data files before training.

    Args:
        data_paths: Single path or list of paths to NPZ files
        fail_on_invalid: If True, raise ValueError on validation failure
        log_issues: If True, log validation issues
        max_issues_to_log: Maximum number of issues to log per file

    Returns:
        DataValidationResult with aggregated validation status

    Raises:
        ValueError: If fail_on_invalid=True and validation fails
    """
    if not HAS_DATA_VALIDATION:
        logger.warning("Data validation module not available")
        return DataValidationResult(
            all_valid=True,
            total_samples=0,
            files_validated=0,
            files_with_issues=0,
        )

    # Normalize to list
    if isinstance(data_paths, str):
        paths = [data_paths]
    else:
        paths = list(data_paths)

    # Filter to existing files
    existing_paths = [p for p in paths if Path(p).exists()]

    if not existing_paths:
        logger.warning("No valid data paths found to validate")
        return DataValidationResult(
            all_valid=True,
            total_samples=0,
            files_validated=0,
            files_with_issues=0,
        )

    if log_issues:
        logger.info(f"Validating {len(existing_paths)} training data file(s)...")

    total_samples = 0
    files_with_issues = 0
    all_issues: list[str] = []
    all_valid = True

    for path in existing_paths:
        result = validate_npz_file(path)
        total_samples += result.total_samples

        if result.valid:
            if log_issues:
                logger.info(f"  ✓ {path}: {result.total_samples} samples OK")
        else:
            all_valid = False
            files_with_issues += 1

            if log_issues:
                logger.warning(
                    f"  ✗ {path}: {len(result.issues)} issues in "
                    f"{result.samples_with_issues}/{result.total_samples} samples"
                )
                # Log first few issues
                for issue in result.issues[:max_issues_to_log]:
                    logger.warning(f"    - {issue}")
                    all_issues.append(f"{path}: {issue}")
                if len(result.issues) > max_issues_to_log:
                    logger.warning(f"    ... and {len(result.issues) - max_issues_to_log} more issues")

    if not all_valid:
        if fail_on_invalid:
            raise ValueError(
                "Training data validation failed. Set fail_on_invalid=False "
                "to proceed despite validation issues (not recommended)."
            )
        elif log_issues:
            logger.warning(
                "Proceeding with training despite validation issues. "
                "Set fail_on_invalid=True to enforce data quality."
            )

    return DataValidationResult(
        all_valid=all_valid,
        total_samples=total_samples,
        files_validated=len(existing_paths),
        files_with_issues=files_with_issues,
        issues=all_issues,
    )


def infer_in_channels_from_npz(npz_path: str) -> int | None:
    """Infer the number of input channels from an NPZ file.

    Args:
        npz_path: Path to NPZ file with 'features' array

    Returns:
        Number of input channels (feature planes), or None if cannot infer
    """
    import numpy as np

    try:
        with np.load(npz_path, mmap_mode="r") as data:
            if "features" in data:
                features = data["features"]
                if len(features.shape) >= 2:
                    # features shape: (N, C, H, W) or (N, C*H*W)
                    if len(features.shape) == 4:
                        return features.shape[1]
                    elif len(features.shape) == 2:
                        # Flattened, try to infer from total size
                        # Common board sizes: 8x8=64, 9x9=81, 19x19=361, 25x25=625
                        total = features.shape[1]
                        for board_cells in [64, 81, 361, 625]:
                            if total % board_cells == 0:
                                return total // board_cells
    except Exception as e:
        logger.warning(f"Could not infer in_channels from {npz_path}: {e}")

    return None


def validate_hex_policy_indices(
    npz_path: str,
    hex_radius: int,
    expected_policy_size: int,
) -> bool:
    """Validate that policy indices in NPZ are within expected range for hex board.

    Args:
        npz_path: Path to NPZ file
        hex_radius: Hex board radius
        expected_policy_size: Expected policy output size

    Returns:
        True if all policy indices are valid
    """
    import numpy as np

    try:
        with np.load(npz_path, mmap_mode="r") as data:
            if "policy" not in data:
                return True

            policy = data["policy"]
            if len(policy.shape) == 1:
                # Sparse format: indices
                max_idx = int(np.max(policy))
                if max_idx >= expected_policy_size:
                    logger.warning(
                        f"Policy index {max_idx} exceeds expected size {expected_policy_size} "
                        f"for hex radius {hex_radius}"
                    )
                    return False
            elif len(policy.shape) == 2:
                # Dense format: probabilities
                if policy.shape[1] != expected_policy_size:
                    logger.warning(
                        f"Policy shape {policy.shape[1]} != expected {expected_policy_size}"
                    )
                    return False

        return True

    except Exception as e:
        logger.warning(f"Could not validate hex policy indices: {e}")
        return True  # Don't fail on validation errors


def get_sample_count(npz_path: str) -> int:
    """Get the number of samples in an NPZ file.

    Args:
        npz_path: Path to NPZ file

    Returns:
        Number of samples, or 0 if cannot determine
    """
    import numpy as np

    try:
        with np.load(npz_path, mmap_mode="r") as data:
            if "features" in data:
                return data["features"].shape[0]
            elif "states" in data:
                return data["states"].shape[0]
    except Exception as e:
        logger.warning(f"Could not get sample count from {npz_path}: {e}")

    return 0

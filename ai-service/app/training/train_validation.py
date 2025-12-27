"""Training data validation utilities.

Extracted from train.py to reduce complexity and improve testability.
These functions handle pre-training validation of data quality and freshness.

Usage:
    from app.training.train_validation import (
        validate_training_data_freshness,
        validate_training_data_files,
        validate_data_checksums,
    )

    # Check data freshness
    validate_training_data_freshness(
        board_type="square8",
        num_players=2,
        max_age_hours=1.0,
    )

    # Validate data files
    validate_training_data_files(
        data_paths=["data/training/sq8_2p.npz"],
        fail_on_invalid=True,
    )
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Lazy import flags
_HAS_FRESHNESS_CHECK = None
_HAS_DATA_VALIDATION = None
_HAS_CHECKSUM_VERIFICATION = None
_HAS_STRUCTURE_VALIDATION = None


def _check_freshness_available() -> bool:
    """Check if freshness check module is available."""
    global _HAS_FRESHNESS_CHECK
    if _HAS_FRESHNESS_CHECK is None:
        try:
            from app.coordination.training_freshness import check_freshness_sync
            _HAS_FRESHNESS_CHECK = True
        except ImportError:
            _HAS_FRESHNESS_CHECK = False
    return _HAS_FRESHNESS_CHECK


def _check_validation_available() -> bool:
    """Check if data validation module is available."""
    global _HAS_DATA_VALIDATION
    if _HAS_DATA_VALIDATION is None:
        try:
            from app.training.data_validation import validate_npz_file
            _HAS_DATA_VALIDATION = True
        except ImportError:
            _HAS_DATA_VALIDATION = False
    return _HAS_DATA_VALIDATION


def _check_checksum_available() -> bool:
    """Check if checksum verification module is available."""
    global _HAS_CHECKSUM_VERIFICATION
    if _HAS_CHECKSUM_VERIFICATION is None:
        try:
            from app.training.npz_checksum import verify_npz_checksums
            _HAS_CHECKSUM_VERIFICATION = True
        except ImportError:
            _HAS_CHECKSUM_VERIFICATION = False
    return _HAS_CHECKSUM_VERIFICATION


def _check_structure_available() -> bool:
    """Check if NPZ structure validation module is available."""
    global _HAS_STRUCTURE_VALIDATION
    if _HAS_STRUCTURE_VALIDATION is None:
        try:
            from app.training.npz_structure_validation import validate_npz_structure
            _HAS_STRUCTURE_VALIDATION = True
        except ImportError:
            _HAS_STRUCTURE_VALIDATION = False
    return _HAS_STRUCTURE_VALIDATION


@dataclass
class FreshnessResult:
    """Result of data freshness check."""

    is_fresh: bool
    data_age_hours: float
    games_available: int
    message: str


@dataclass
class ValidationResult:
    """Result of data validation."""

    valid: bool
    total_samples: int
    samples_with_issues: int
    issues: list[str]


@dataclass
class StructureValidationResult:
    """Result of NPZ structure validation."""

    valid: bool
    sample_count: int
    array_shapes: dict[str, tuple]
    errors: list[str]


@dataclass
class DataValidationResult:
    """Combined result of all training data validation checks.

    This aggregates results from freshness, structure, content, and checksum
    validation into a single result object.
    """

    all_valid: bool
    freshness: FreshnessResult | None = None
    structure_results: dict[str, StructureValidationResult] | None = None
    file_validations: list[ValidationResult] | None = None
    checksum_results: dict[str, tuple[bool, list[str]]] | None = None
    errors: list[str] | None = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def validate_training_data_freshness(
    board_type: str,
    num_players: int,
    max_age_hours: float = 1.0,
    allow_stale: bool = False,
    emit_events: bool = True,
) -> FreshnessResult:
    """Validate that training data is fresh enough for training.

    Args:
        board_type: Board type (e.g., "square8")
        num_players: Number of players
        max_age_hours: Maximum acceptable data age in hours
        allow_stale: If True, warn but don't fail on stale data
        emit_events: If True, emit TRAINING_BLOCKED_BY_QUALITY on stale data

    Returns:
        FreshnessResult with freshness status

    Raises:
        ValueError: If data is stale and allow_stale is False
    """
    if not _check_freshness_available():
        logger.warning("Data freshness check module not available - skipping")
        return FreshnessResult(
            is_fresh=True,
            data_age_hours=0.0,
            games_available=0,
            message="Freshness check skipped (module unavailable)",
        )

    from app.coordination.training_freshness import check_freshness_sync

    logger.info(f"Checking training data freshness (max_age={max_age_hours}h)...")

    try:
        freshness_result = check_freshness_sync(
            board_type=board_type,
            num_players=num_players,
            max_age_hours=max_age_hours,
        )

        if freshness_result.is_fresh:
            msg = (
                f"Training data is fresh: {freshness_result.games_available} games, "
                f"age={freshness_result.data_age_hours:.1f}h"
            )
            logger.info(msg)
            return FreshnessResult(
                is_fresh=True,
                data_age_hours=freshness_result.data_age_hours,
                games_available=freshness_result.games_available,
                message=msg,
            )

        # Data is stale
        msg = (
            f"Training data is stale: age={freshness_result.data_age_hours:.1f}h "
            f"(threshold={max_age_hours}h), games={freshness_result.games_available}"
        )

        if allow_stale:
            logger.warning(msg)
            logger.warning(
                "Proceeding with stale data (--allow-stale-data specified). "
                "Consider running data sync: python scripts/run_training_loop.py --sync-only"
            )
            return FreshnessResult(
                is_fresh=False,
                data_age_hours=freshness_result.data_age_hours,
                games_available=freshness_result.games_available,
                message=f"{msg} (allowed)",
            )

        # Emit event if requested
        if emit_events:
            _emit_training_blocked_event(
                board_type=board_type,
                num_players=num_players,
                data_age_hours=freshness_result.data_age_hours,
                threshold_hours=max_age_hours,
                games_available=freshness_result.games_available,
            )

        # Fail on stale data
        raise ValueError(
            f"{msg}. Use --allow-stale-data to proceed anyway, or "
            "run data sync: python scripts/run_training_loop.py --sync-only"
        )

    except ValueError:
        raise
    except Exception as e:
        logger.warning(f"Data freshness check failed: {e}")
        return FreshnessResult(
            is_fresh=True,  # Assume fresh on error
            data_age_hours=0.0,
            games_available=0,
            message=f"Freshness check failed: {e}",
        )


def _emit_training_blocked_event(
    board_type: str,
    num_players: int,
    data_age_hours: float,
    threshold_hours: float,
    games_available: int,
) -> None:
    """Emit TRAINING_BLOCKED_BY_QUALITY event."""
    try:
        from app.coordination.event_router import DataEventType, get_event_bus

        config_key = f"{board_type}_{num_players}p"
        bus = get_event_bus()
        if bus:
            bus.emit(DataEventType.TRAINING_BLOCKED_BY_QUALITY, {
                "config_key": config_key,
                "reason": "stale_data",
                "data_age_hours": data_age_hours,
                "threshold_hours": threshold_hours,
                "games_available": games_available,
            })
            logger.info(
                f"Emitted TRAINING_BLOCKED_BY_QUALITY for {config_key} "
                f"(age={data_age_hours:.1f}h)"
            )
    except Exception as e:
        logger.debug(f"Failed to emit training blocked event: {e}")


def validate_training_data_files(
    data_paths: list[str],
    fail_on_invalid: bool = False,
) -> list[ValidationResult]:
    """Validate training data files for corruption or issues.

    Args:
        data_paths: List of paths to .npz files
        fail_on_invalid: If True, raise on validation failure

    Returns:
        List of ValidationResult for each file

    Raises:
        ValueError: If fail_on_invalid is True and any file fails validation
    """
    if not _check_validation_available():
        logger.warning("Data validation module not available")
        return []

    from app.training.data_validation import validate_npz_file

    # Filter to existing paths
    valid_paths = [p for p in data_paths if p and os.path.exists(p)]
    if not valid_paths:
        return []

    logger.info(f"Validating {len(valid_paths)} training data file(s)...")

    results = []
    any_failed = False

    for path in valid_paths:
        result = validate_npz_file(path)
        vr = ValidationResult(
            valid=result.valid,
            total_samples=result.total_samples,
            samples_with_issues=result.samples_with_issues,
            issues=list(result.issues) if hasattr(result, 'issues') else [],
        )
        results.append(vr)

        if result.valid:
            logger.info(f"  ✓ {path}: {result.total_samples} samples OK")
        else:
            logger.warning(
                f"  ✗ {path}: {len(vr.issues)} issues in "
                f"{result.samples_with_issues}/{result.total_samples} samples"
            )
            for issue in vr.issues[:5]:
                logger.warning(f"    - {issue}")
            if len(vr.issues) > 5:
                logger.warning(f"    ... and {len(vr.issues) - 5} more issues")
            any_failed = True

    if any_failed and fail_on_invalid:
        raise ValueError(
            "Training data validation failed. Set fail_on_invalid=False "
            "to proceed despite validation issues (not recommended)."
        )

    return results


def validate_data_checksums(
    data_paths: list[str],
    fail_on_mismatch: bool = False,
) -> dict[str, tuple[bool, list[str]]]:
    """Verify embedded checksums in training data files.

    Args:
        data_paths: List of paths to .npz files
        fail_on_mismatch: If True, raise on checksum mismatch

    Returns:
        Dict mapping path to (valid, errors) tuple

    Raises:
        ValueError: If fail_on_mismatch is True and checksums don't match
    """
    if not _check_checksum_available():
        logger.debug("Checksum verification module not available")
        return {}

    from app.training.npz_checksum import verify_npz_checksums

    valid_paths = [p for p in data_paths if p and os.path.exists(p)]
    if not valid_paths:
        return {}

    logger.info("Verifying data checksums...")

    results = {}
    any_failed = False

    for path in valid_paths:
        all_valid, computed, errors = verify_npz_checksums(path)

        if all_valid and not errors:
            if computed:
                logger.info(f"  ✓ {path}: checksums verified ({len(computed)} arrays)")
            else:
                logger.debug(f"  ○ {path}: no embedded checksums (legacy file)")
            results[path] = (True, [])
        else:
            logger.warning(f"  ✗ {path}: checksum verification failed")
            for error in errors[:3]:
                logger.warning(f"    - {error}")
            if len(errors) > 3:
                logger.warning(f"    ... and {len(errors) - 3} more errors")
            results[path] = (False, errors)
            any_failed = True

    if any_failed and fail_on_mismatch:
        raise ValueError(
            "Training data checksum verification failed. "
            "Data may be corrupted - consider re-exporting."
        )

    return results


def validate_npz_structure_files(
    data_paths: list[str],
    require_policy: bool = True,
    fail_on_invalid: bool = False,
) -> dict[str, StructureValidationResult]:
    """Validate NPZ file structure for corruption or dimension issues.

    This validation catches issues like rsync --partial creating files with
    unreasonable dimensions (e.g., 22 billion elements instead of 6.3 million).

    Args:
        data_paths: List of paths to .npz files
        require_policy: If True, require policy arrays to be present
        fail_on_invalid: If True, raise on validation failure

    Returns:
        Dict mapping path to StructureValidationResult

    Raises:
        ValueError: If fail_on_invalid is True and any file fails validation
    """
    if not _check_structure_available():
        logger.debug("NPZ structure validation module not available")
        return {}

    from pathlib import Path

    from app.training.npz_structure_validation import validate_npz_structure

    valid_paths = [p for p in data_paths if p and os.path.exists(p)]
    if not valid_paths:
        return {}

    logger.info(f"Validating NPZ structure for {len(valid_paths)} file(s)...")

    results = {}
    any_failed = False

    for path in valid_paths:
        struct_result = validate_npz_structure(Path(path), require_policy=require_policy)

        # Convert to our dataclass
        result = StructureValidationResult(
            valid=struct_result.valid,
            sample_count=getattr(struct_result, 'sample_count', 0),
            array_shapes=dict(getattr(struct_result, 'array_shapes', {})),
            errors=list(getattr(struct_result, 'errors', [])),
        )
        results[path] = result

        if result.valid:
            logger.info(
                f"  ✓ {path}: {result.sample_count} samples, "
                f"{len(result.array_shapes)} arrays"
            )
        else:
            logger.error(f"  ✗ {path}: CORRUPTED")
            for error in result.errors:
                logger.error(f"    - {error}")
            any_failed = True

    if any_failed and fail_on_invalid:
        raise ValueError(
            "NPZ structure validation FAILED - files may be corrupted. "
            "Do NOT proceed with training. Check rsync/transfer logs for issues."
        )

    return results


# Default size limit for checksum verification (MB)
CHECKSUM_SIZE_LIMIT_MB = 500


def validate_training_data(
    data_paths: list[str],
    board_type: str | None = None,
    num_players: int | None = None,
    check_freshness: bool = True,
    check_structure: bool = True,
    check_content: bool = True,
    check_checksums: bool = True,
    max_data_age_hours: float = 1.0,
    allow_stale_data: bool = False,
    fail_on_invalid: bool = False,
    checksum_size_limit_mb: float = CHECKSUM_SIZE_LIMIT_MB,
) -> DataValidationResult:
    """Unified validation of all training data quality checks.

    This function orchestrates freshness, structure, content, and checksum
    validation into a single call, returning a combined result.

    Args:
        data_paths: List of paths to .npz training data files.
        board_type: Board type for freshness check (e.g., "square8").
            Required if check_freshness is True.
        num_players: Number of players for freshness check.
            Required if check_freshness is True.
        check_freshness: If True, validate data freshness.
        check_structure: If True, validate NPZ file structure.
        check_content: If True, validate data content (policy sums, values).
        check_checksums: If True, verify embedded checksums.
        max_data_age_hours: Maximum acceptable data age in hours.
        allow_stale_data: If True, warn but don't fail on stale data.
        fail_on_invalid: If True, raise on any validation failure.
        checksum_size_limit_mb: Skip checksum for files larger than this (MB).

    Returns:
        DataValidationResult with aggregated results from all checks.

    Raises:
        ValueError: If fail_on_invalid is True and any validation fails.

    Example:
        >>> from app.training.train_validation import validate_training_data
        >>> result = validate_training_data(
        ...     data_paths=["data/training/sq8_2p.npz"],
        ...     board_type="square8",
        ...     num_players=2,
        ...     fail_on_invalid=True,
        ... )
        >>> if result.all_valid:
        ...     print("All validations passed!")
    """
    errors: list[str] = []
    all_valid = True

    # Freshness check
    freshness_result = None
    if check_freshness:
        if board_type is None or num_players is None:
            logger.warning(
                "Freshness check skipped: board_type and num_players required"
            )
        else:
            try:
                freshness_result = validate_training_data_freshness(
                    board_type=board_type,
                    num_players=num_players,
                    max_age_hours=max_data_age_hours,
                    allow_stale=allow_stale_data,
                    emit_events=True,
                )
                if not freshness_result.is_fresh and not allow_stale_data:
                    all_valid = False
                    errors.append(f"Data is stale: {freshness_result.message}")
            except ValueError as e:
                all_valid = False
                errors.append(str(e))
                if fail_on_invalid:
                    raise

    # Filter to existing paths for file-based checks
    valid_paths = [p for p in data_paths if p and os.path.exists(p)]
    if not valid_paths and (check_structure or check_content or check_checksums):
        logger.warning("No valid data paths found for file validation")

    # Structure validation
    structure_results = None
    if check_structure and valid_paths:
        structure_results = validate_npz_structure_files(
            data_paths=valid_paths,
            require_policy=True,
            fail_on_invalid=False,  # We'll handle failure below
        )
        for path, result in structure_results.items():
            if not result.valid:
                all_valid = False
                errors.append(f"Structure validation failed: {path}")

    # Content validation
    file_validations = None
    if check_content and valid_paths:
        file_validations = validate_training_data_files(
            data_paths=valid_paths,
            fail_on_invalid=False,  # We'll handle failure below
        )
        for result in file_validations:
            if not result.valid:
                all_valid = False
                errors.append(
                    f"Content validation failed: {result.samples_with_issues} issues"
                )

    # Checksum verification (skip large files)
    checksum_results = None
    if check_checksums and valid_paths:
        # Filter out files larger than size limit
        paths_for_checksum = []
        for path in valid_paths:
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            if file_size_mb <= checksum_size_limit_mb:
                paths_for_checksum.append(path)
            else:
                logger.info(
                    f"  ○ {path}: skipping checksum "
                    f"(file size {file_size_mb:.0f}MB > {checksum_size_limit_mb}MB limit)"
                )

        if paths_for_checksum:
            checksum_results = validate_data_checksums(
                data_paths=paths_for_checksum,
                fail_on_mismatch=False,  # We'll handle failure below
            )
            for path, (valid, errs) in checksum_results.items():
                if not valid:
                    all_valid = False
                    errors.append(f"Checksum verification failed: {path}")

    # Create combined result
    result = DataValidationResult(
        all_valid=all_valid,
        freshness=freshness_result,
        structure_results=structure_results,
        file_validations=file_validations,
        checksum_results=checksum_results,
        errors=errors,
    )

    # Raise if requested and validation failed
    if fail_on_invalid and not all_valid:
        error_summary = "; ".join(errors[:3])
        if len(errors) > 3:
            error_summary += f" (and {len(errors) - 3} more errors)"
        raise ValueError(f"Training data validation failed: {error_summary}")

    return result


__all__ = [
    # Dataclasses
    "DataValidationResult",
    "FreshnessResult",
    "StructureValidationResult",
    "ValidationResult",
    # Individual validators
    "validate_data_checksums",
    "validate_npz_structure_files",
    "validate_training_data_files",
    "validate_training_data_freshness",
    # Unified validator
    "validate_training_data",
    # Constants
    "CHECKSUM_SIZE_LIMIT_MB",
]

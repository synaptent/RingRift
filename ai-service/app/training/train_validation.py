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
        from app.distributed.data_events import DataEventType
        from app.coordination.event_router import get_event_bus

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


__all__ = [
    "FreshnessResult",
    "ValidationResult",
    "validate_training_data_freshness",
    "validate_training_data_files",
    "validate_data_checksums",
]

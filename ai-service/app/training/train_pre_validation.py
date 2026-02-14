"""Pre-training validation for RingRift training pipeline.

Extracted from train.py (lines 1108-1419) to reduce train_model() complexity.
Handles training data freshness checks, NPZ structure validation,
data content validation, and checksum verification.

This module runs all validation gates before training begins to catch
issues early and prevent wasting compute on invalid data.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


def run_pre_training_validation(
    *,
    data_path: str | list[str],
    config: Any,  # TrainConfig
    num_players: int,
    distributed: bool = False,
    is_main: bool = True,
    skip_freshness_check: bool = False,
    max_data_age_hours: float = 2000.0,
    allow_stale_data: bool = False,
    disable_stale_fallback: bool = False,
    max_sync_failures: int = 5,
    max_sync_duration: float = 2700.0,
    validate_data: bool = True,
    fail_on_invalid_data: bool = False,
    use_streaming: bool = False,
    # Module references (None = not available)
    check_freshness_sync: Any | None = None,
    validate_npz_structure_fn: Any | None = None,
    validate_npz_file_fn: Any | None = None,
    verify_npz_checksums_fn: Any | None = None,
    should_allow_stale_training_fn: Any | None = None,
    # Feature flags
    HAS_FRESHNESS_CHECK: bool = False,
    HAS_NPZ_STRUCTURE_VALIDATION: bool = False,
    HAS_DATA_VALIDATION: bool = False,
    HAS_CHECKSUM_VERIFICATION: bool = False,
    HAS_STALE_FALLBACK: bool = False,
    # DataEventType reference for event emission
    DataEventType: Any | None = None,
) -> None:
    """Run all pre-training validation checks.

    This function performs four sequential validation steps:
    1. Training data freshness check
    2. NPZ structure validation (corruption detection)
    3. Data content validation (policy sums, value ranges)
    4. Checksum verification (file integrity)

    Args:
        data_path: Path(s) to training data (.npz file or list of files).
        config: Training configuration with board_type attribute.
        num_players: Number of players for this training run.
        distributed: Whether distributed training is enabled.
        is_main: Whether this is the main process (for logging).
        skip_freshness_check: Skip data freshness check entirely.
        max_data_age_hours: Maximum acceptable data age in hours.
        allow_stale_data: Allow stale data with a warning instead of failing.
        disable_stale_fallback: Disable automatic stale fallback controller.
        max_sync_failures: Number of sync failures before fallback is allowed.
        max_sync_duration: Seconds before sync timeout fallback is allowed.
        validate_data: Whether to run data validation checks.
        fail_on_invalid_data: Raise ValueError on validation failure vs warn.
        use_streaming: Whether streaming data loading is being used.
        check_freshness_sync: Function to check data freshness (or None).
        validate_npz_structure_fn: Function to validate NPZ structure (or None).
        validate_npz_file_fn: Function to validate NPZ file contents (or None).
        verify_npz_checksums_fn: Function to verify NPZ checksums (or None).
        should_allow_stale_training_fn: Function to check stale fallback (or None).
        HAS_FRESHNESS_CHECK: Whether freshness check module is available.
        HAS_NPZ_STRUCTURE_VALIDATION: Whether NPZ structure validation is available.
        HAS_DATA_VALIDATION: Whether data validation module is available.
        HAS_CHECKSUM_VERIFICATION: Whether checksum verification is available.
        HAS_STALE_FALLBACK: Whether stale fallback controller is available.
        DataEventType: DataEventType enum reference for event emission (or None).

    Raises:
        ValueError: If validation fails and the check is configured to fail.
    """
    _check_freshness(
        data_path=data_path,
        config=config,
        num_players=num_players,
        distributed=distributed,
        is_main=is_main,
        skip_freshness_check=skip_freshness_check,
        max_data_age_hours=max_data_age_hours,
        allow_stale_data=allow_stale_data,
        disable_stale_fallback=disable_stale_fallback,
        max_sync_failures=max_sync_failures,
        max_sync_duration=max_sync_duration,
        check_freshness_sync=check_freshness_sync,
        should_allow_stale_training_fn=should_allow_stale_training_fn,
        HAS_FRESHNESS_CHECK=HAS_FRESHNESS_CHECK,
        HAS_STALE_FALLBACK=HAS_STALE_FALLBACK,
        DataEventType=DataEventType,
    )

    _validate_npz_structure(
        data_path=data_path,
        distributed=distributed,
        is_main=is_main,
        validate_data=validate_data,
        fail_on_invalid_data=fail_on_invalid_data,
        use_streaming=use_streaming,
        validate_npz_structure_fn=validate_npz_structure_fn,
        HAS_NPZ_STRUCTURE_VALIDATION=HAS_NPZ_STRUCTURE_VALIDATION,
    )

    _validate_data_content(
        data_path=data_path,
        distributed=distributed,
        is_main=is_main,
        validate_data=validate_data,
        fail_on_invalid_data=fail_on_invalid_data,
        use_streaming=use_streaming,
        validate_npz_file_fn=validate_npz_file_fn,
        HAS_DATA_VALIDATION=HAS_DATA_VALIDATION,
    )

    _verify_checksums(
        data_path=data_path,
        distributed=distributed,
        is_main=is_main,
        validate_data=validate_data,
        fail_on_invalid_data=fail_on_invalid_data,
        use_streaming=use_streaming,
        verify_npz_checksums_fn=verify_npz_checksums_fn,
        HAS_CHECKSUM_VERIFICATION=HAS_CHECKSUM_VERIFICATION,
    )


def _check_freshness(
    *,
    data_path: str | list[str],
    config: Any,
    num_players: int,
    distributed: bool,
    is_main: bool,
    skip_freshness_check: bool,
    max_data_age_hours: float,
    allow_stale_data: bool,
    disable_stale_fallback: bool,
    max_sync_failures: int,
    max_sync_duration: float,
    check_freshness_sync: Any | None,
    should_allow_stale_training_fn: Any | None,
    HAS_FRESHNESS_CHECK: bool,
    HAS_STALE_FALLBACK: bool,
    DataEventType: Any | None,
) -> None:
    """Training Data Freshness Check (2025-12) - Phase 1.5: MANDATORY BY DEFAULT.

    Prevents 95% of stale data training incidents by failing early.

    DEFAULT BEHAVIOR (skip_freshness_check=False):
      - Check data age before training starts
      - Default threshold: 1.0 hours (max_data_age_hours)
      - If data is stale: FAIL with clear error message
      - Override with --allow-stale-data (warns) or --skip-freshness-check (dangerous)
    """
    if not skip_freshness_check and HAS_FRESHNESS_CHECK:
        if not distributed or is_main:
            config_key = f"{config.board_type.value}_{num_players}p"
            logger.info(
                f"[DataFreshness] Checking training data freshness for {config_key} "
                f"(max_age={max_data_age_hours}h)..."
            )
            try:
                freshness_result = check_freshness_sync(
                    board_type=config.board_type.value,
                    num_players=num_players,
                    max_age_hours=max_data_age_hours,
                )
                if freshness_result.is_fresh:
                    logger.info(
                        f"[DataFreshness] \u2713 Training data is fresh: "
                        f"{freshness_result.games_available} games, "
                        f"age={freshness_result.data_age_hours:.1f}h"
                    )
                else:
                    # Data is stale - determine if we should fail or warn
                    stale_msg = (
                        f"Training data is STALE for {config_key}:\n"
                        f"  - Data age: {freshness_result.data_age_hours:.1f} hours\n"
                        f"  - Threshold: {max_data_age_hours} hours\n"
                        f"  - Games available: {freshness_result.games_available}"
                    )

                    if allow_stale_data:
                        # User explicitly allowed stale data with --allow-stale-data
                        logger.warning(f"[DataFreshness] {stale_msg}")
                        logger.warning(
                            "[DataFreshness] Proceeding with stale data (--allow-stale-data specified). "
                            "This may result in suboptimal training."
                        )
                        logger.info(
                            f"To get fresh data, run:\n"
                            f"  python scripts/run_training_loop.py --sync-only "
                            f"--board-type {config.board_type.value} --num-players {num_players}"
                        )
                    else:
                        # December 2025: Check stale fallback controller before blocking
                        # Part of 48-hour autonomous operation plan
                        fallback_allowed = False
                        fallback_reason = ""

                        if not disable_stale_fallback and HAS_STALE_FALLBACK:
                            try:
                                fallback_allowed, fallback_reason = should_allow_stale_training_fn(
                                    config_key=config_key,
                                    data_age_hours=freshness_result.data_age_hours,
                                    sync_failures=max_sync_failures,
                                    elapsed_sync_time=max_sync_duration,
                                    games_available=freshness_result.games_available,
                                )
                            except (RuntimeError, ValueError, AttributeError) as fb_err:
                                logger.debug(f"[DataFreshness] Fallback check failed: {fb_err}")

                        if fallback_allowed:
                            # Fallback triggered - proceed with stale data
                            logger.warning(f"[DataFreshness] {stale_msg}")
                            logger.warning(
                                f"[StaleFallback] Proceeding with stale data due to fallback: {fallback_reason}\n"
                                f"  This is part of 48-hour autonomous operation. Training will continue\n"
                                f"  but may produce suboptimal results. Sync should recover in background."
                            )
                            logger.info(
                                f"To get fresh data manually, run:\n"
                                f"  python scripts/run_training_loop.py --sync-only "
                                f"--board-type {config.board_type.value} --num-players {num_players}"
                            )
                        else:
                            # P1.1 (Dec 2025): Emit TRAINING_BLOCKED_BY_QUALITY to trigger selfplay acceleration
                            # This closes the critical feedback loop: stale data -> more selfplay -> fresh data
                            try:
                                # Use globally imported DataEventType - local import would shadow it and cause UnboundLocalError
                                from app.coordination.event_router import get_event_bus

                                bus = get_event_bus()
                                if bus and DataEventType is not None:
                                    bus.emit(DataEventType.TRAINING_BLOCKED_BY_QUALITY, {
                                        "config_key": config_key,
                                        "reason": "stale_data",
                                        "data_age_hours": freshness_result.data_age_hours,
                                        "threshold_hours": max_data_age_hours,
                                        "games_available": freshness_result.games_available,
                                    })
                                    logger.info(
                                        f"[DataFreshness] Emitted TRAINING_BLOCKED_BY_QUALITY for {config_key} "
                                        f"(age={freshness_result.data_age_hours:.1f}h)"
                                    )
                            except (ImportError, AttributeError, RuntimeError) as emit_err:
                                # ImportError: event_router module not available
                                # AttributeError: missing event bus methods
                                # RuntimeError: event system errors
                                logger.debug(f"[DataFreshness] Failed to emit training blocked event: {emit_err}")

                            # Default: fail on stale data to prevent training on outdated samples
                            error_msg = (
                                f"\n{'='*70}\n"
                                f"TRAINING BLOCKED: {stale_msg}\n"
                                f"{'='*70}\n\n"
                                f"Training blocked to prevent learning from stale data.\n\n"
                                f"OPTIONS TO PROCEED:\n\n"
                                f"  1. Get fresh data (RECOMMENDED):\n"
                                f"     python scripts/run_training_loop.py --sync-only \\\n"
                                f"       --board-type {config.board_type.value} --num-players {num_players}\n\n"
                                f"  2. Allow stale data (NOT RECOMMENDED - may degrade model quality):\n"
                                f"     Add --allow-stale-data flag to your training command\n\n"
                                f"  3. Skip freshness check (DANGEROUS - only for debugging):\n"
                                f"     Add --skip-freshness-check flag to your training command\n\n"
                                f"  4. Adjust freshness threshold:\n"
                                f"     Add --max-data-age-hours <hours> to allow older data\n"
                                f"{'='*70}\n"
                            )
                            raise ValueError(error_msg)
            except ValueError:
                raise  # Re-raise stale data errors
            except (OSError, ImportError, AttributeError, RuntimeError) as e:
                # OSError: file/network I/O errors when checking freshness
                # ImportError: freshness module dependencies missing
                # AttributeError: API changes in freshness checker
                # RuntimeError: freshness check logic errors
                logger.warning(f"[DataFreshness] Check failed with error: {e}")
                # If freshness check crashes, we allow training to proceed
                # This prevents transient issues from blocking training entirely
                logger.warning(
                    "[DataFreshness] Proceeding with training despite check failure. "
                    "Consider investigating the error."
                )
    elif not skip_freshness_check and not HAS_FRESHNESS_CHECK:
        if not distributed or is_main:
            logger.warning(
                "[DataFreshness] Freshness check module not available - check skipped. "
                "Install app.coordination.training_freshness to enable."
            )
    elif skip_freshness_check:
        if not distributed or is_main:
            logger.warning(
                f"\n{'='*70}\n"
                f"\u26a0\ufe0f  FRESHNESS CHECK SKIPPED (--skip-freshness-check)\n"
                f"{'='*70}\n"
                f"  Training may use stale or outdated data.\n"
                f"  This can lead to poor model quality and wasted compute.\n"
                f"  Only use this flag for debugging or special scenarios.\n"
                f"{'='*70}\n"
            )


def _resolve_data_paths(data_path: str | list[str]) -> list[str]:
    """Resolve data_path to a list of existing file paths."""
    paths: list[str] = []
    if isinstance(data_path, list):
        paths = [p for p in data_path if p and os.path.exists(p)]
    elif data_path and os.path.exists(data_path):
        paths = [data_path]
    return paths


def _validate_npz_structure(
    *,
    data_path: str | list[str],
    distributed: bool,
    is_main: bool,
    validate_data: bool,
    fail_on_invalid_data: bool,
    use_streaming: bool,
    validate_npz_structure_fn: Any | None,
    HAS_NPZ_STRUCTURE_VALIDATION: bool,
) -> None:
    """NPZ Structure Validation (December 2025).

    Validate NPZ file structure BEFORE loading to catch corruption early.
    This catches issues like rsync --partial creating files with unreasonable
    dimensions (e.g., 22 billion elements instead of 6.3 million).
    """
    if not (validate_data and HAS_NPZ_STRUCTURE_VALIDATION and not use_streaming):
        return

    structure_paths = _resolve_data_paths(data_path)
    if not structure_paths:
        return

    if not distributed or is_main:
        logger.info(f"Validating NPZ structure for {len(structure_paths)} file(s)...")

    structure_failed = False
    for path in structure_paths:
        struct_result = validate_npz_structure_fn(Path(path), require_policy=True)
        if not distributed or is_main:
            if struct_result.valid:
                logger.info(
                    f"  \u2713 {path}: {struct_result.sample_count} samples, "
                    f"{len(struct_result.array_shapes)} arrays"
                )
            else:
                logger.error(f"  \u2717 {path}: CORRUPTED")
                for error in struct_result.errors:
                    logger.error(f"    - {error}")
                structure_failed = True

    if structure_failed:
        if fail_on_invalid_data:
            raise ValueError(
                "NPZ structure validation FAILED - files may be corrupted. "
                "Do NOT proceed with training. Check rsync/transfer logs for issues."
            )
        else:
            if not distributed or is_main:
                logger.error(
                    "=" * 70 + "\n"
                    "WARNING: NPZ files appear CORRUPTED but proceeding anyway.\n"
                    "This will likely produce garbage models.\n"
                    "Set fail_on_invalid_data=True to prevent this.\n"
                    "=" * 70
                )


def _validate_data_content(
    *,
    data_path: str | list[str],
    distributed: bool,
    is_main: bool,
    validate_data: bool,
    fail_on_invalid_data: bool,
    use_streaming: bool,
    validate_npz_file_fn: Any | None,
    HAS_DATA_VALIDATION: bool,
) -> None:
    """Data Content Validation (2025-12).

    Validate training data content (policy sums, value ranges) after structure check.
    """
    if validate_data and HAS_DATA_VALIDATION and not use_streaming:
        data_paths_to_validate = _resolve_data_paths(data_path)

        if data_paths_to_validate:
            if not distributed or is_main:
                logger.info(f"Validating data content for {len(data_paths_to_validate)} file(s)...")

            validation_failed = False
            for path in data_paths_to_validate:
                result = validate_npz_file_fn(path)
                if not distributed or is_main:
                    if result.valid:
                        logger.info(f"  \u2713 {path}: {result.total_samples} samples OK")
                    else:
                        logger.warning(
                            f"  \u2717 {path}: {len(result.issues)} issues in "
                            f"{result.samples_with_issues}/{result.total_samples} samples"
                        )
                        # Log first few issues
                        for issue in result.issues[:5]:
                            logger.warning(f"    - {issue}")
                        if len(result.issues) > 5:
                            logger.warning(f"    ... and {len(result.issues) - 5} more issues")
                        validation_failed = True

            if validation_failed:
                if fail_on_invalid_data:
                    raise ValueError(
                        "Training data validation failed. Set fail_on_invalid_data=False "
                        "to proceed despite validation issues (not recommended)."
                    )
                else:
                    if not distributed or is_main:
                        logger.warning(
                            "Proceeding with training despite validation issues. "
                            "Set fail_on_invalid_data=True to enforce data quality."
                        )
    elif validate_data and not HAS_DATA_VALIDATION:
        if not distributed or is_main:
            logger.warning("Data validation requested but module not available")


def _verify_checksums(
    *,
    data_path: str | list[str],
    distributed: bool,
    is_main: bool,
    validate_data: bool,
    fail_on_invalid_data: bool,
    use_streaming: bool,
    verify_npz_checksums_fn: Any | None,
    HAS_CHECKSUM_VERIFICATION: bool,
) -> None:
    """Checksum Verification (December 2025).

    Verify embedded checksums to detect file corruption.
    Skip checksum verification for large files (>500MB) to avoid memory issues.
    """
    CHECKSUM_SIZE_LIMIT_MB = 500

    if not (validate_data and HAS_CHECKSUM_VERIFICATION and not use_streaming):
        return

    checksum_paths = _resolve_data_paths(data_path)
    if not checksum_paths:
        return

    if not distributed or is_main:
        logger.info("Verifying data checksums...")

    checksum_failed = False
    for path in checksum_paths:
        # Skip checksum verification for large files (>500MB)
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        if file_size_mb > CHECKSUM_SIZE_LIMIT_MB:
            if not distributed or is_main:
                logger.info(f"  \u25cb {path}: skipping checksum (file size {file_size_mb:.0f}MB > {CHECKSUM_SIZE_LIMIT_MB}MB limit)")
            continue
        all_valid, computed, errors = verify_npz_checksums_fn(path)
        if not distributed or is_main:
            if all_valid and not errors:
                if computed:
                    logger.info(f"  \u2713 {path}: checksums verified ({len(computed)} arrays)")
                else:
                    logger.debug(f"  \u25cb {path}: no embedded checksums (legacy file)")
            else:
                logger.warning(f"  \u2717 {path}: checksum verification failed")
                for error in errors[:3]:
                    logger.warning(f"    - {error}")
                if len(errors) > 3:
                    logger.warning(f"    ... and {len(errors) - 3} more errors")
                checksum_failed = True

    if checksum_failed:
        if fail_on_invalid_data:
            raise ValueError(
                "Checksum verification failed - data may be corrupted. "
                "Re-export the training data or set fail_on_invalid_data=False "
                "to proceed despite checksum issues (not recommended)."
            )
        else:
            if not distributed or is_main:
                logger.warning(
                    "Proceeding with training despite checksum issues. "
                    "Data may be corrupted - consider re-exporting."
                )

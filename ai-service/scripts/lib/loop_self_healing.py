"""Self-Healing and Escalation for the minimal AlphaZero loop circuit breaker.

Pattern-specific recovery handlers that attempt automatic fixes before the
circuit breaker stops the loop. Each failure is classified into a known pattern,
and a targeted recovery action is executed.

Usage from minimal_alphazero_loop.py:
    from scripts.lib.loop_self_healing import attempt_recovery, FailureContext

    recovery = attempt_recovery(FailureContext(
        error_message=err_msg,
        stage="training",
        config_key="hex8_2p",
        work_dir=str(wdir),
        model_path=str(best),
        batch_size=batch_size,
        selfplay_randomness=randomness,
    ))
    if recovery.recovered:
        consec_failures = 0
        continue
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger("loop_self_healing")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class FailurePattern(Enum):
    OOM = "oom"
    IDENTICAL_DATA = "identical_data"
    DEAD_MODEL = "dead_model"
    ARCH_MISMATCH = "arch_mismatch"
    UNKNOWN = "unknown"


@dataclass
class FailureContext:
    error_message: str
    stage: str  # "selfplay", "export", "data_quality", "training", "probe"
    config_key: str
    work_dir: str
    model_path: str  # path to best.pth
    batch_size: int = 512
    selfplay_randomness: float = 0.25
    # Model architecture version (e.g. "v2", "v4", "v5-heavy"). When present
    # and non-v2, the arch-mismatch recovery skips the S3 redownload because
    # the canonical_{config}.pth in S3 is the v2 family — pulling it would
    # poison a v4/v5-heavy lane. Left optional for backward compatibility.
    model_version: str | None = None


@dataclass
class RecoveryResult:
    recovered: bool
    action: str
    message: str
    adjustments: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. Failure Pattern Classifier
# ---------------------------------------------------------------------------

def classify_failure(ctx: FailureContext) -> FailurePattern:
    """Classify the last failure into a known pattern based on error content.

    Inspects the error message for signature strings that indicate specific
    root causes. Returns UNKNOWN for unclassified failures.
    """
    msg = ctx.error_message.lower()

    # OOM: CUDA or general out-of-memory
    if "out of memory" in msg or "cuda oom" in msg or "outofmemoryerror" in msg:
        return FailurePattern.OOM

    # Identical/stale data: DQS flagged near-identical training data
    if "near-identical" in msg or "identical to a recent iteration" in msg:
        return FailurePattern.IDENTICAL_DATA

    # Dead model: training probe found zero gradient or dead value head
    if "zero gradient" in msg or "dead value" in msg or "heuristic fallback" in msg:
        return FailurePattern.DEAD_MODEL

    # Architecture mismatch: encoder/channel count mismatch
    if "encoder mismatch" in msg or ("channel" in msg and "mismatch" in msg):
        return FailurePattern.ARCH_MISMATCH
    if "encoding mismatch" in msg:
        return FailurePattern.ARCH_MISMATCH

    return FailurePattern.UNKNOWN


# ---------------------------------------------------------------------------
# 2. Recovery Actions
# ---------------------------------------------------------------------------

S3_MODELS_PREFIX = "s3://ringrift-models-20251214/consolidated/models"


def _recover_oom(ctx: FailureContext) -> RecoveryResult:
    """Halve the batch size for the next training attempt.

    Does not directly retry training -- returns adjustments that the caller
    should apply on the next iteration.
    """
    new_bs = max(ctx.batch_size // 2, 32)
    if new_bs == ctx.batch_size:
        return RecoveryResult(
            recovered=False,
            action="retry_smaller_batch",
            message=f"Batch size already at minimum ({ctx.batch_size}), cannot reduce further",
        )
    logger.info(
        "SELF-HEAL [OOM]: reducing batch_size %d -> %d",
        ctx.batch_size, new_bs,
    )
    return RecoveryResult(
        recovered=True,
        action="retry_smaller_batch",
        message=f"Halved batch_size {ctx.batch_size} -> {new_bs}",
        adjustments={"batch_size": new_bs},
    )


def _recover_identical_data(ctx: FailureContext) -> RecoveryResult:
    """Increase selfplay randomness and delete the last NPZ to force regeneration.

    Caps randomness at 0.5 to avoid degenerate play.
    """
    new_rand = min(ctx.selfplay_randomness + 0.1, 0.5)
    if new_rand <= ctx.selfplay_randomness:
        return RecoveryResult(
            recovered=False,
            action="retry_higher_randomness",
            message=f"Randomness already at max ({ctx.selfplay_randomness:.2f}), cannot increase",
        )

    # Delete the most recent NPZ to force fresh selfplay
    wdir = Path(ctx.work_dir)
    recent_npz = sorted(wdir.glob("iter_*.npz"))
    deleted = None
    if recent_npz:
        last_npz = recent_npz[-1]
        try:
            last_npz.unlink()
            deleted = str(last_npz)
            logger.info("SELF-HEAL [IDENTICAL_DATA]: deleted stale NPZ %s", last_npz.name)
        except OSError as e:
            logger.warning("SELF-HEAL: failed to delete NPZ %s: %s", last_npz, e)

    logger.info(
        "SELF-HEAL [IDENTICAL_DATA]: increasing randomness %.2f -> %.2f",
        ctx.selfplay_randomness, new_rand,
    )
    return RecoveryResult(
        recovered=True,
        action="retry_higher_randomness",
        message=f"Increased randomness {ctx.selfplay_randomness:.2f} -> {new_rand:.2f}"
                + (f", deleted {deleted}" if deleted else ""),
        adjustments={"selfplay_randomness": new_rand},
    )


def _recover_dead_model(ctx: FailureContext) -> RecoveryResult:
    """Roll back to canonical model by copying it over best.pth.

    Resets the loop to the last known good model. Should only fire once
    per loop run -- the caller tracks per-pattern recovery counts.
    """
    # Look for canonical model as the original starting point
    # Convention: canonical models are in models/ directory
    best_path = Path(ctx.model_path)
    if not best_path.exists():
        return RecoveryResult(
            recovered=False,
            action="rollback_model",
            message=f"Best model not found at {ctx.model_path}",
        )

    # Find canonical model for this config
    # Search in the same directory as best.pth (work_dir/models/) and
    # in the project-level models/ directory (relative to work_dir parent).
    models_dir = best_path.parent
    project_models = Path(ctx.work_dir).parent / "models"
    canonical_name = f"canonical_{ctx.config_key}.pth"
    canonical_candidates = [
        models_dir / canonical_name,
        project_models / canonical_name,
    ]
    canonical = None
    for cand in canonical_candidates:
        if cand.exists():
            canonical = cand
            break

    if canonical is None:
        # Try to download from S3
        logger.info("SELF-HEAL [DEAD_MODEL]: no local canonical model, trying S3 download")
        s3_result = _download_canonical_from_s3(ctx.config_key, str(best_path))
        if s3_result:
            return RecoveryResult(
                recovered=True,
                action="rollback_model",
                message=f"Rolled back best.pth from S3 canonical_{ctx.config_key}.pth",
            )
        return RecoveryResult(
            recovered=False,
            action="rollback_model",
            message=f"No canonical model found for {ctx.config_key} (local or S3)",
        )

    try:
        shutil.copy2(str(canonical), str(best_path))
        logger.info(
            "SELF-HEAL [DEAD_MODEL]: rolled back %s -> %s",
            canonical.name, best_path,
        )
        return RecoveryResult(
            recovered=True,
            action="rollback_model",
            message=f"Rolled back best.pth from {canonical.name}",
        )
    except OSError as e:
        return RecoveryResult(
            recovered=False,
            action="rollback_model",
            message=f"Failed to copy canonical model: {e}",
        )


def _download_canonical_from_s3(config_key: str, dest_path: str) -> bool:
    """Best-effort download of canonical model from S3.

    Returns True if download succeeded, False otherwise.
    Never raises -- S3 failures are non-fatal.
    """
    s3_path = f"{S3_MODELS_PREFIX}/canonical_{config_key}.pth"
    try:
        result = subprocess.run(
            ["aws", "s3", "cp", s3_path, dest_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0 and Path(dest_path).exists():
            logger.info("SELF-HEAL: downloaded %s from S3", s3_path)
            return True
        logger.warning(
            "SELF-HEAL: S3 download failed (exit %d): %s",
            result.returncode, result.stderr[:200],
        )
    except FileNotFoundError:
        logger.warning("SELF-HEAL: aws CLI not found, cannot download from S3")
    except subprocess.TimeoutExpired:
        logger.warning("SELF-HEAL: S3 download timed out after 120s")
    except Exception as e:
        logger.warning("SELF-HEAL: S3 download error: %s", e)
    return False


def _recover_arch_mismatch(ctx: FailureContext) -> RecoveryResult:
    """Re-download canonical model from S3 to fix architecture mismatches.

    This typically happens when a model checkpoint was corrupted or when
    the training code was updated with a different encoder version.

    Safety: S3 only carries ``canonical_{config_key}.pth`` (v2-family) for
    each config. For v4/v5-heavy lanes, redownloading that canonical would
    overwrite a 64-channel checkpoint with a 40-channel one, poisoning
    the lane into a persistent crash-restart loop. Gh200-11 v5-heavy hit
    exactly this on 2026-04-21 after iter 1 successfully promoted. If
    ``ctx.model_version`` is set and not v2, refuse the S3 redownload
    and let the circuit breaker trip cleanly instead of corrupting state.
    """
    model_version = (ctx.model_version or "v2").lower()
    if model_version not in ("v2", ""):
        logger.warning(
            "SELF-HEAL [ARCH_MISMATCH]: refusing S3 redownload for model_version=%s; "
            "canonical_%s.pth on S3 is v2-family and would poison this lane. "
            "Manual intervention required.",
            model_version, ctx.config_key,
        )
        return RecoveryResult(
            recovered=False,
            action="redownload_canonical_skipped",
            message=(
                f"Refused S3 canonical redownload for model_version={model_version}: "
                f"canonical_{ctx.config_key}.pth in S3 is v2-family and would poison "
                f"this lane. Fix the underlying arch-mismatch before retry."
            ),
        )

    best_path = Path(ctx.model_path)
    logger.info("SELF-HEAL [ARCH_MISMATCH]: re-downloading canonical model from S3")
    if _download_canonical_from_s3(ctx.config_key, str(best_path)):
        return RecoveryResult(
            recovered=True,
            action="redownload_canonical",
            message=f"Re-downloaded canonical_{ctx.config_key}.pth from S3",
        )
    return RecoveryResult(
        recovered=False,
        action="redownload_canonical",
        message=f"Failed to download canonical model from S3 for {ctx.config_key}",
    )


def _no_auto_fix(ctx: FailureContext) -> RecoveryResult:
    """Log diagnostic context for manual investigation."""
    logger.warning(
        "SELF-HEAL [UNKNOWN]: no auto-fix available. "
        "stage=%s config=%s error=%s",
        ctx.stage, ctx.config_key, ctx.error_message[:300],
    )
    return RecoveryResult(
        recovered=False,
        action="none",
        message="Unknown failure pattern, manual investigation needed",
    )


# Map from pattern to recovery function
RECOVERY_ACTIONS = {
    FailurePattern.OOM: _recover_oom,
    FailurePattern.IDENTICAL_DATA: _recover_identical_data,
    FailurePattern.DEAD_MODEL: _recover_dead_model,
    FailurePattern.ARCH_MISMATCH: _recover_arch_mismatch,
    FailurePattern.UNKNOWN: _no_auto_fix,
}


# ---------------------------------------------------------------------------
# 3. Recovery Coordinator
# ---------------------------------------------------------------------------

# Per-pattern recovery budget per loop run
MAX_RECOVERIES_PER_PATTERN = 2
# ROLLBACK_MODEL is more disruptive -- limit to 1
_PATTERN_LIMITS = {
    FailurePattern.DEAD_MODEL: 1,
}

# Module-level counter tracking recoveries within the current process lifetime.
# Reset via reset_recovery_counts() if the loop restarts.
_recovery_counts: Counter = Counter()


def reset_recovery_counts() -> None:
    """Reset per-pattern recovery counts. Call when starting a new loop run."""
    _recovery_counts.clear()


def attempt_recovery(ctx: FailureContext) -> RecoveryResult:
    """Classify the failure and attempt pattern-specific recovery.

    Returns a RecoveryResult indicating whether recovery succeeded and
    what adjustments (if any) the caller should apply.

    Recovery attempts are rate-limited: max 2 per pattern per loop run,
    with DEAD_MODEL limited to 1.
    """
    pattern = classify_failure(ctx)
    logger.info(
        "SELF-HEAL: classified failure as %s (stage=%s, error=%.100s...)",
        pattern.value, ctx.stage, ctx.error_message,
    )

    if pattern == FailurePattern.UNKNOWN:
        return _no_auto_fix(ctx)

    # Check per-pattern budget
    limit = _PATTERN_LIMITS.get(pattern, MAX_RECOVERIES_PER_PATTERN)
    if _recovery_counts[pattern] >= limit:
        logger.warning(
            "SELF-HEAL: recovery budget exhausted for %s (%d/%d attempts used)",
            pattern.value, _recovery_counts[pattern], limit,
        )
        return RecoveryResult(
            recovered=False,
            action=f"budget_exhausted_{pattern.value}",
            message=f"Recovery budget exhausted for {pattern.value} "
                    f"({_recovery_counts[pattern]}/{limit} attempts used)",
        )

    action_fn = RECOVERY_ACTIONS[pattern]
    result = action_fn(ctx)

    # Track the attempt regardless of outcome
    _recovery_counts[pattern] += 1

    if result.recovered:
        logger.info(
            "SELF-HEAL: recovery succeeded — action=%s message=%s",
            result.action, result.message,
        )
    else:
        logger.warning(
            "SELF-HEAL: recovery failed — action=%s message=%s",
            result.action, result.message,
        )

    return result

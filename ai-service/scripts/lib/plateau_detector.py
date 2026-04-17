"""Plateau Detector for the minimal AlphaZero loop.

Formalises what we'd otherwise do by eye: if the recent window of
iterations has been dominated by rejections and no promotion has
happened in a long time, the model is plateaued.

This is a pure-logic module intentionally independent of the loop so
it can be unit-tested without spinning up selfplay or training. The
loop calls `detect_plateau(history)` at the end of each iteration and
acts on the result (structured log line; optionally bump exploration
or lower threshold when the caller opted in).

Default thresholds were chosen to match the hex8_2p plateau pattern
observed April 2026 (iters 34-36 rejected at 49-50% after iter 33
promoted):

- Minimum window: 10 recent iterations (smaller windows are too noisy)
- Rejection rate trigger: >= 80% of those 10 rejected
- Staleness trigger: >= 15 iterations since the last promotion

Both triggers must fire before we call it a plateau. This avoids
false positives during the normal early-training phase where nothing
promotes yet because the candidate is below threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Tunables (exported for tests + regression tracking)
# ---------------------------------------------------------------------------

PLATEAU_WINDOW = 10
PLATEAU_REJECTION_RATE = 0.80
PLATEAU_STALENESS = 15

# Minimum iterations before plateau detection becomes meaningful. Early-
# training configs can go many iterations without a promotion simply
# because the first promotion hasn't happened yet; that's not a plateau,
# it's a cold start.
PLATEAU_MIN_ITERATIONS = 20


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlateauResult:
    """Outcome of a plateau check at a given point in training."""

    detected: bool
    """True when both triggers fired and the full pre-conditions held."""

    recent_rejection_rate: float
    """Fraction of the recent window that rejected (0.0 – 1.0)."""

    iterations_since_promotion: int | None
    """Iterations since the last promotion, or None if no promotion yet."""

    window_size: int
    """How many iterations were actually in the analysed window."""

    total_iterations: int
    """Total iterations observed so far."""

    last_promoted_iteration: int | None
    """The iteration number of the most recent promotion, if any."""

    reason: str
    """Human-readable one-line summary of why the detector did or did
    not fire.  Suitable for inclusion in a structured log line."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_plateau(
    history: Sequence[dict],
    *,
    window: int = PLATEAU_WINDOW,
    rejection_rate_threshold: float = PLATEAU_REJECTION_RATE,
    staleness_threshold: int = PLATEAU_STALENESS,
    min_iterations: int = PLATEAU_MIN_ITERATIONS,
) -> PlateauResult:
    """Decide whether the training loop has plateaued.

    Args:
        history: Iterable of metrics dicts in chronological order. Each
            entry must contain at least the keys used by the loop's
            metrics.jsonl: ``iteration`` (int) and ``promoted`` (bool).
            Missing keys are treated as "iteration unknown" and
            "not promoted" respectively.
        window: How many most-recent iterations to analyse for
            rejection rate. Default 10.
        rejection_rate_threshold: Trigger fires when rejection rate in
            the window meets or exceeds this value. Default 0.80.
        staleness_threshold: Trigger fires when we've been this many
            iterations without a promotion. Default 15.
        min_iterations: Skip detection below this many total iterations
            to avoid firing during normal cold-start behaviour.
            Default 20.

    Returns:
        PlateauResult.  Never raises.
    """
    if not isinstance(window, int) or window <= 0:
        raise ValueError(f"window must be positive int, got {window!r}")
    if not 0.0 < rejection_rate_threshold <= 1.0:
        raise ValueError(
            f"rejection_rate_threshold must be in (0,1], got "
            f"{rejection_rate_threshold!r}"
        )
    if staleness_threshold <= 0:
        raise ValueError(
            f"staleness_threshold must be > 0, got {staleness_threshold!r}"
        )

    hist = list(history)
    total = len(hist)

    if total == 0:
        return PlateauResult(
            detected=False,
            recent_rejection_rate=0.0,
            iterations_since_promotion=None,
            window_size=0,
            total_iterations=0,
            last_promoted_iteration=None,
            reason="no metrics history yet",
        )

    # Compute last promoted iteration. We trust 'iteration' in metrics
    # entries, but fall back to list index if missing.
    last_promoted_iter: int | None = None
    last_iter_seen: int | None = None
    for idx, m in enumerate(hist):
        iter_num = _get_iteration(m, idx)
        last_iter_seen = iter_num
        if _is_promoted(m):
            last_promoted_iter = iter_num

    # Recent-window rejection rate. Include only iterations that reached
    # an evaluation decision (i.e. have a 'promoted' key); missing-key
    # entries count as "no decision" so they don't inflate the rate.
    recent = hist[-window:] if len(hist) >= window else hist
    decided = [m for m in recent if "promoted" in m]
    window_size = len(decided)
    rejections = sum(1 for m in decided if not _is_promoted(m))
    rejection_rate = rejections / window_size if window_size else 0.0

    # Iterations since last promotion, measured against the most recent
    # iteration seen (not simply len(hist)) to handle gaps in the log.
    if last_promoted_iter is not None and last_iter_seen is not None:
        iter_since_promotion = last_iter_seen - last_promoted_iter
    elif last_iter_seen is not None:
        # Never promoted — staleness is just the iteration counter.
        iter_since_promotion = last_iter_seen
    else:
        iter_since_promotion = None

    # Gate on min_iterations so we don't mistake cold start for plateau.
    if total < min_iterations:
        return PlateauResult(
            detected=False,
            recent_rejection_rate=rejection_rate,
            iterations_since_promotion=iter_since_promotion,
            window_size=window_size,
            total_iterations=total,
            last_promoted_iteration=last_promoted_iter,
            reason=(
                f"below min_iterations ({total} < {min_iterations}); "
                "skipping detection"
            ),
        )

    rate_fired = rejection_rate >= rejection_rate_threshold
    staleness_fired = (
        iter_since_promotion is not None
        and iter_since_promotion >= staleness_threshold
    )

    if rate_fired and staleness_fired:
        reason = (
            f"PLATEAU_DETECTED rejection_rate={rejection_rate:.0%} "
            f"(>= {rejection_rate_threshold:.0%} over last {window_size}); "
            f"iters_since_promotion={iter_since_promotion} "
            f"(>= {staleness_threshold})"
        )
        return PlateauResult(
            detected=True,
            recent_rejection_rate=rejection_rate,
            iterations_since_promotion=iter_since_promotion,
            window_size=window_size,
            total_iterations=total,
            last_promoted_iteration=last_promoted_iter,
            reason=reason,
        )

    # Not yet plateaued; explain why for the log.
    parts: list[str] = []
    if not rate_fired:
        parts.append(
            f"rejection_rate={rejection_rate:.0%} < "
            f"{rejection_rate_threshold:.0%}"
        )
    if not staleness_fired:
        parts.append(
            f"iters_since_promotion={iter_since_promotion} < "
            f"{staleness_threshold}"
        )
    reason = "no plateau; " + ", ".join(parts) if parts else "no plateau"
    return PlateauResult(
        detected=False,
        recent_rejection_rate=rejection_rate,
        iterations_since_promotion=iter_since_promotion,
        window_size=window_size,
        total_iterations=total,
        last_promoted_iteration=last_promoted_iter,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _is_promoted(m: dict) -> bool:
    return bool(m.get("promoted"))


def _get_iteration(m: dict, fallback_idx: int) -> int:
    iter_raw = m.get("iteration")
    if isinstance(iter_raw, int):
        return iter_raw
    # If iteration is missing or malformed, fall back to 1-based index in
    # the history list so downstream math still works.
    return fallback_idx + 1


__all__ = [
    "PLATEAU_MIN_ITERATIONS",
    "PLATEAU_REJECTION_RATE",
    "PLATEAU_STALENESS",
    "PLATEAU_WINDOW",
    "PlateauResult",
    "detect_plateau",
]

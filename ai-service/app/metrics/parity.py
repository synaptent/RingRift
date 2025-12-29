"""Parity healthcheck metrics for RingRift AI service.

This module provides Prometheus metrics for tracking parity healthcheck results,
which validate that Python and TypeScript rules engines produce identical outputs.

December 2025 Update:
- Added semantic divergence tracking by board type and dimension
- Added ANM divergence counter for the hexagonal parity bug investigation
- Added divergence position histogram for understanding failure patterns
- Added structural issue counters by type
- Added parity check duration histogram for performance tracking

Usage:
    from app.metrics.parity import (
        record_parity_mismatch,
        record_parity_case,
        update_parity_pass_rate,
        record_semantic_divergence,
        record_anm_divergence,
        record_parity_check_duration,
    )

    # Record a mismatch
    record_parity_mismatch("hash", "contract_vectors")

    # Record a semantic divergence with dimension
    record_semantic_divergence("hexagonal", 2, "anm_state")

    # Record ANM-specific divergence
    record_anm_divergence("hexagonal", 2, move_index=837)

    # Record check duration
    record_parity_check_duration("square8", 2, 0.5)
"""

from __future__ import annotations

import logging

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# Parity mismatch counter - tracks mismatches by type and suite
PARITY_MISMATCHES_TOTAL = Counter(
    "ringrift_parity_mismatches_total",
    "Total parity mismatches by type and suite.",
    ["mismatch_type", "suite"],
)

# Parity healthcheck cases counter - tracks cases by suite and result
PARITY_HEALTHCHECK_CASES_TOTAL = Counter(
    "ringrift_parity_healthcheck_cases_total",
    "Total parity healthcheck cases executed.",
    ["suite", "result"],
)

# Parity healthcheck pass rate gauge - current pass rate per suite
PARITY_HEALTHCHECK_PASS_RATE = Gauge(
    "ringrift_parity_healthcheck_pass_rate",
    "Parity healthcheck pass rate per suite (0-1).",
    ["suite"],
)


def record_parity_mismatch(mismatch_type: str, suite: str) -> None:
    """Record a parity mismatch.

    Args:
        mismatch_type: Type of mismatch (validation, status, hash, s_invariant, unknown)
        suite: Suite name (contract_vectors, plateau_snapshots)
    """
    try:
        PARITY_MISMATCHES_TOTAL.labels(mismatch_type=mismatch_type, suite=suite).inc()
    except Exception as e:
        logger.warning(f"Failed to record parity mismatch metric: {e}")


def record_parity_case(suite: str, passed: bool) -> None:
    """Record a parity healthcheck case result.

    Args:
        suite: Suite name (contract_vectors, plateau_snapshots)
        passed: Whether the case passed
    """
    try:
        result = "passed" if passed else "failed"
        PARITY_HEALTHCHECK_CASES_TOTAL.labels(suite=suite, result=result).inc()
    except Exception as e:
        logger.warning(f"Failed to record parity case metric: {e}")


def update_parity_pass_rate(suite: str, pass_rate: float) -> None:
    """Update the parity healthcheck pass rate for a suite.

    Args:
        suite: Suite name (contract_vectors, plateau_snapshots)
        pass_rate: Pass rate between 0 and 1
    """
    try:
        PARITY_HEALTHCHECK_PASS_RATE.labels(suite=suite).set(pass_rate)
    except Exception as e:
        logger.warning(f"Failed to update parity pass rate metric: {e}")


# Track previous rates for change detection (December 29, 2025)
_previous_rates: dict[str, float] = {}
_SIGNIFICANT_RATE_CHANGE = 0.05  # 5% change triggers event


def update_parity_pass_rate_with_event(
    suite: str,
    pass_rate: float,
    board_type: str = "",
    num_players: int = 0,
    total_checked: int = 0,
) -> None:
    """Update parity pass rate and emit event if rate changed significantly.

    December 29, 2025: This function wires the previously orphaned
    PARITY_FAILURE_RATE_CHANGED event. Call this instead of update_parity_pass_rate
    when you want event emission.

    Args:
        suite: Suite name (contract_vectors, plateau_snapshots)
        pass_rate: Pass rate between 0 and 1
        board_type: Board type being validated (optional)
        num_players: Number of players (optional)
        total_checked: Total games checked in this batch (optional)
    """
    # Update the gauge
    update_parity_pass_rate(suite, pass_rate)

    # Check for significant rate change and emit event
    key = f"{suite}:{board_type}:{num_players}"
    old_rate = _previous_rates.get(key, pass_rate)
    failure_rate = 1.0 - pass_rate
    old_failure_rate = 1.0 - old_rate

    if abs(failure_rate - old_failure_rate) >= _SIGNIFICANT_RATE_CHANGE:
        try:
            import asyncio
            from app.distributed.data_events import emit_parity_failure_rate_changed
            from app.core.async_context import fire_and_forget

            try:
                asyncio.get_running_loop()
                fire_and_forget(
                    emit_parity_failure_rate_changed(
                        new_rate=failure_rate,
                        old_rate=old_failure_rate,
                        board_type=board_type,
                        num_players=num_players,
                        total_checked=total_checked,
                        source="ParityMetrics",
                    ),
                    name="emit_parity_failure_rate_changed",
                )
            except RuntimeError:
                # No event loop - skip event emission in sync context
                pass

            logger.info(
                f"[ParityMetrics] PARITY_FAILURE_RATE_CHANGED: "
                f"{old_failure_rate:.1%} -> {failure_rate:.1%} for {key}"
            )
        except ImportError:
            pass

    _previous_rates[key] = pass_rate


# =============================================================================
# Semantic Divergence Metrics (December 2025)
# =============================================================================

# Semantic divergence counter by board type and dimension
PARITY_SEMANTIC_DIVERGENCES_TOTAL = Counter(
    "ringrift_parity_semantic_divergences_total",
    "Total semantic divergences between Python and TypeScript by board type and dimension.",
    ["board_type", "num_players", "dimension"],
)

# ANM divergence counter - specifically tracks ANM state mismatches
PARITY_ANM_DIVERGENCES_TOTAL = Counter(
    "ringrift_parity_anm_divergences_total",
    "Total ANM state divergences between Python and TypeScript.",
    ["board_type", "num_players"],
)

# Divergence position histogram - where in the game divergences occur
PARITY_DIVERGENCE_MOVE_INDEX = Histogram(
    "ringrift_parity_divergence_move_index",
    "Move index where parity divergence occurred.",
    ["board_type", "num_players"],
    buckets=[10, 50, 100, 200, 500, 800, 1000, 1500, 2000],
)

# Structural issues counter by type
PARITY_STRUCTURAL_ISSUES_TOTAL = Counter(
    "ringrift_parity_structural_issues_total",
    "Total structural issues in game recordings (invalid, mid_snapshot, non_canonical).",
    ["board_type", "num_players", "issue_type"],
)

# Parity check duration histogram - performance tracking
PARITY_CHECK_DURATION_SECONDS = Histogram(
    "ringrift_parity_check_duration_seconds",
    "Duration of parity check per game in seconds.",
    ["board_type", "num_players"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

# Games processed counter
PARITY_GAMES_CHECKED_TOTAL = Counter(
    "ringrift_parity_games_checked_total",
    "Total games checked for parity by board type and result.",
    ["board_type", "num_players", "result"],
)


def record_semantic_divergence(
    board_type: str,
    num_players: int,
    dimension: str,
    move_index: int | None = None,
) -> None:
    """Record a semantic divergence between Python and TypeScript.

    Args:
        board_type: Board type (square8, square19, hex8, hexagonal)
        num_players: Number of players (2, 3, 4)
        dimension: Mismatch dimension (current_player, current_phase, anm_state, state_hash, etc.)
        move_index: Optional move index where divergence occurred
    """
    try:
        np_str = str(num_players)
        PARITY_SEMANTIC_DIVERGENCES_TOTAL.labels(
            board_type=board_type,
            num_players=np_str,
            dimension=dimension,
        ).inc()

        # Also track move index if provided
        if move_index is not None:
            PARITY_DIVERGENCE_MOVE_INDEX.labels(
                board_type=board_type,
                num_players=np_str,
            ).observe(float(move_index))

    except Exception as e:
        logger.warning(f"Failed to record semantic divergence metric: {e}")


def record_anm_divergence(
    board_type: str,
    num_players: int,
    move_index: int | None = None,
) -> None:
    """Record an ANM state divergence (subset of semantic divergence).

    This is specifically for tracking the hexagonal ANM parity bug and
    similar ANM-related issues.

    Args:
        board_type: Board type (square8, square19, hex8, hexagonal)
        num_players: Number of players (2, 3, 4)
        move_index: Optional move index where divergence occurred
    """
    try:
        np_str = str(num_players)
        PARITY_ANM_DIVERGENCES_TOTAL.labels(
            board_type=board_type,
            num_players=np_str,
        ).inc()

        # Also record as semantic divergence with anm_state dimension
        record_semantic_divergence(board_type, num_players, "anm_state", move_index)

    except Exception as e:
        logger.warning(f"Failed to record ANM divergence metric: {e}")


def record_structural_issue(
    board_type: str,
    num_players: int,
    issue_type: str,
) -> None:
    """Record a structural issue in a game recording.

    Args:
        board_type: Board type (square8, square19, hex8, hexagonal)
        num_players: Number of players (2, 3, 4)
        issue_type: Type of issue (invalid, mid_snapshot, non_canonical_history)
    """
    try:
        PARITY_STRUCTURAL_ISSUES_TOTAL.labels(
            board_type=board_type,
            num_players=str(num_players),
            issue_type=issue_type,
        ).inc()
    except Exception as e:
        logger.warning(f"Failed to record structural issue metric: {e}")


def record_parity_check_duration(
    board_type: str,
    num_players: int,
    duration_seconds: float,
) -> None:
    """Record the duration of a parity check for a single game.

    Args:
        board_type: Board type (square8, square19, hex8, hexagonal)
        num_players: Number of players (2, 3, 4)
        duration_seconds: Time taken to check parity in seconds
    """
    try:
        PARITY_CHECK_DURATION_SECONDS.labels(
            board_type=board_type,
            num_players=str(num_players),
        ).observe(duration_seconds)
    except Exception as e:
        logger.warning(f"Failed to record parity check duration metric: {e}")


def record_game_checked(
    board_type: str,
    num_players: int,
    passed: bool,
) -> None:
    """Record a game that was checked for parity.

    Args:
        board_type: Board type (square8, square19, hex8, hexagonal)
        num_players: Number of players (2, 3, 4)
        passed: Whether the game passed parity check
    """
    try:
        result = "passed" if passed else "failed"
        PARITY_GAMES_CHECKED_TOTAL.labels(
            board_type=board_type,
            num_players=str(num_players),
            result=result,
        ).inc()
    except Exception as e:
        logger.warning(f"Failed to record game checked metric: {e}")


def emit_parity_summary_metrics(summary: dict) -> None:
    """Emit metrics from a parity healthcheck summary.

    This function takes the JSON summary from run_parity_healthcheck.py
    and emits corresponding Prometheus metrics.

    Args:
        summary: Parity healthcheck summary dict with keys:
            - total_cases: int
            - total_mismatches: int
            - mismatches_by_type: dict[str, int]
            - mismatches_by_suite: dict[str, int]
            - pass_rate_by_suite: dict[str, float] (optional)
    """
    try:
        # Emit mismatch counts by type (use suite="all" for aggregate)
        for mismatch_type, count in summary.get("mismatches_by_type", {}).items():
            if count > 0:
                PARITY_MISMATCHES_TOTAL.labels(
                    mismatch_type=mismatch_type, suite="all"
                ).inc(count)

        # Emit mismatch counts by suite (use mismatch_type="all" for aggregate)
        for suite, count in summary.get("mismatches_by_suite", {}).items():
            if count > 0:
                PARITY_MISMATCHES_TOTAL.labels(
                    mismatch_type="all", suite=suite
                ).inc(count)

        # Update pass rates per suite
        total_cases = summary.get("total_cases", 0)
        total_mismatches = summary.get("total_mismatches", 0)

        if total_cases > 0:
            # Overall pass rate
            overall_pass_rate = 1.0 - (total_mismatches / total_cases)
            update_parity_pass_rate("all", overall_pass_rate)

        # Per-suite pass rates if available - use event-emitting version
        for suite, pass_rate in summary.get("pass_rate_by_suite", {}).items():
            update_parity_pass_rate_with_event(
                suite=suite,
                pass_rate=pass_rate,
                board_type=summary.get("board_type", ""),
                num_players=summary.get("num_players", 0),
                total_checked=summary.get("total_games_checked", total_cases),
            )

    except Exception as e:
        logger.warning(f"Failed to emit parity summary metrics: {e}")


__all__ = [
    # Legacy metrics
    "PARITY_HEALTHCHECK_CASES_TOTAL",
    "PARITY_HEALTHCHECK_PASS_RATE",
    "PARITY_MISMATCHES_TOTAL",
    # Semantic divergence metrics (December 2025)
    "PARITY_ANM_DIVERGENCES_TOTAL",
    "PARITY_CHECK_DURATION_SECONDS",
    "PARITY_DIVERGENCE_MOVE_INDEX",
    "PARITY_GAMES_CHECKED_TOTAL",
    "PARITY_SEMANTIC_DIVERGENCES_TOTAL",
    "PARITY_STRUCTURAL_ISSUES_TOTAL",
    # Recording functions
    "emit_parity_summary_metrics",
    "record_anm_divergence",
    "record_game_checked",
    "record_parity_case",
    "record_parity_check_duration",
    "record_parity_mismatch",
    "record_semantic_divergence",
    "record_structural_issue",
    "update_parity_pass_rate",
    "update_parity_pass_rate_with_event",
]

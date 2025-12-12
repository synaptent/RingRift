"""Prometheus metrics for the RingRift AI service.

This module centralises counters and histograms so that /ai/move and
related endpoints can record lightweight telemetry without each handler
having to manage its own metric instances. The metrics are intentionally
minimal but labeled so they can be filtered by AI type and difficulty in
local/dev Prometheus setups.
"""

from __future__ import annotations

from typing import Final

from prometheus_client import Counter, Gauge, Histogram


AI_MOVE_REQUESTS: Final[Counter] = Counter(
    "ai_move_requests_total",
    (
        "Total number of /ai/move requests, labeled by ai_type, "
        "difficulty and outcome."
    ),
    labelnames=("ai_type", "difficulty", "outcome"),
)

AI_MOVE_LATENCY: Final[Histogram] = Histogram(
    "ai_move_latency_seconds",
    (
        "Latency of /ai/move requests in seconds, labeled by ai_type "
        "and difficulty."
    ),
    labelnames=("ai_type", "difficulty"),
    # Buckets chosen to cover sub-100ms up to several seconds while keeping
    # the set small enough for local/dev use. These can be refined later if
    # we deploy a dedicated metrics stack.
    buckets=(
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
    ),
)


PYTHON_INVARIANT_VIOLATIONS: Final[Counter] = Counter(
    "ringrift_python_invariant_violations_total",
    (
        "Total number of Python self-play invariant violations observed in "
        "run_self_play_soak, labeled by high-level invariant_id and "
        "low-level violation type."
    ),
    labelnames=("invariant_id", "type"),
)

AI_INSTANCE_CACHE_LOOKUPS: Final[Counter] = Counter(
    "ai_instance_cache_lookups_total",
    "Total AI instance cache lookups, labeled by ai_type and outcome.",
    labelnames=("ai_type", "outcome"),
)

AI_INSTANCE_CACHE_SIZE: Final[Gauge] = Gauge(
    "ai_instance_cache_size",
    "Current number of cached AI instances in this process.",
)

# Pre-initialize one labeled time series for the core /ai/move metrics so the
# /metrics endpoint exposes histogram buckets even before the first request.
# This keeps smoke tests and local Prometheus setups stable.
#
# Note: we intentionally do NOT call .observe() / .inc() here; creating the
# labeled child is sufficient to emit zero-valued samples.
AI_MOVE_REQUESTS.labels("init", "0", "init")  # type: ignore[arg-type]
AI_MOVE_LATENCY.labels("init", "0")  # type: ignore[arg-type]


def observe_ai_move_start(ai_type: str, difficulty: int) -> tuple[str, str]:
    """Prepare metric label values for a new /ai/move request.

    This helper just normalises difficulty into a string label; callers are
    expected to pass the returned labels into the Counter/Histogram as
    needed. It exists mainly to keep the label-shape logic in one place.
    """

    return ai_type, str(difficulty)


__all__ = [
    "AI_MOVE_REQUESTS",
    "AI_MOVE_LATENCY",
    "AI_INSTANCE_CACHE_LOOKUPS",
    "AI_INSTANCE_CACHE_SIZE",
    "PYTHON_INVARIANT_VIOLATIONS",
    "observe_ai_move_start",
]

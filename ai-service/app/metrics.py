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

# Game outcome metrics
GAME_OUTCOMES: Final[Counter] = Counter(
    "ringrift_game_outcomes_total",
    "Total game outcomes from selfplay, labeled by board_type, num_players, and outcome.",
    labelnames=("board_type", "num_players", "outcome"),
)

GAMES_COMPLETED: Final[Counter] = Counter(
    "ringrift_games_completed_total",
    "Total completed selfplay games, labeled by board_type and num_players.",
    labelnames=("board_type", "num_players"),
)

GAMES_MOVES_TOTAL: Final[Counter] = Counter(
    "ringrift_games_moves_total",
    "Total moves across all selfplay games, labeled by board_type and num_players.",
    labelnames=("board_type", "num_players"),
)

GAME_DURATION_SECONDS: Final[Histogram] = Histogram(
    "ringrift_game_duration_seconds",
    "Duration of selfplay games in seconds.",
    labelnames=("board_type", "num_players"),
    buckets=(1, 5, 10, 30, 60, 120, 300, 600),
)

WIN_RATE_BY_PLAYER: Final[Gauge] = Gauge(
    "ringrift_win_rate_by_player",
    "Win rate per player position (0-indexed), updated periodically.",
    labelnames=("board_type", "num_players", "player_position"),
)

DRAW_RATE: Final[Gauge] = Gauge(
    "ringrift_draw_rate",
    "Draw rate for selfplay games, updated periodically.",
    labelnames=("board_type", "num_players"),
)

# Cluster cost and efficiency metrics
CLUSTER_NODE_UP: Final[Gauge] = Gauge(
    "ringrift_cluster_node_up",
    "Whether a cluster node is active (1=up, 0=down).",
    labelnames=("node", "gpu_type"),
)

CLUSTER_NODE_COST_PER_HOUR: Final[Gauge] = Gauge(
    "ringrift_cluster_node_cost_per_hour",
    "Estimated hourly cost for a cluster node in USD.",
    labelnames=("node", "gpu_type"),
)

CLUSTER_GPU_UTILIZATION: Final[Gauge] = Gauge(
    "ringrift_cluster_gpu_utilization",
    "GPU utilization as a fraction (0-1).",
    labelnames=("node", "gpu_type"),
)

CLUSTER_CPU_UTILIZATION: Final[Gauge] = Gauge(
    "ringrift_cluster_cpu_utilization",
    "CPU utilization as a fraction (0-1).",
    labelnames=("node",),
)

CLUSTER_GPU_MEMORY_USED_BYTES: Final[Gauge] = Gauge(
    "ringrift_cluster_gpu_memory_used_bytes",
    "GPU memory used in bytes.",
    labelnames=("node", "gpu_type"),
)

CLUSTER_MEMORY_USED_BYTES: Final[Gauge] = Gauge(
    "ringrift_cluster_memory_used_bytes",
    "System memory used in bytes.",
    labelnames=("node",),
)

# GPU pricing (Lambda Labs, December 2024)
GPU_HOURLY_RATES: Final[dict] = {
    "GH200": 2.49,
    "H100": 2.49,
    "A100": 1.99,
    "A10": 0.75,
    "RTX_4090": 0.50,
    "unknown": 1.00,
}


def report_cluster_node(
    node: str,
    gpu_type: str,
    is_up: bool = True,
    gpu_utilization: float = 0.0,
    cpu_utilization: float = 0.0,
    gpu_memory_bytes: int = 0,
    system_memory_bytes: int = 0,
) -> None:
    """Report metrics for a cluster node.

    Args:
        node: Node identifier (e.g., '192.222.53.22' or 'lambda-gh200-1')
        gpu_type: GPU type (e.g., 'GH200', 'A100', 'H100')
        is_up: Whether the node is currently active
        gpu_utilization: GPU utilization fraction (0-1)
        cpu_utilization: CPU utilization fraction (0-1)
        gpu_memory_bytes: GPU memory used in bytes
        system_memory_bytes: System memory used in bytes
    """
    CLUSTER_NODE_UP.labels(node, gpu_type).set(1 if is_up else 0)

    hourly_rate = GPU_HOURLY_RATES.get(gpu_type, GPU_HOURLY_RATES["unknown"])
    CLUSTER_NODE_COST_PER_HOUR.labels(node, gpu_type).set(hourly_rate if is_up else 0)

    CLUSTER_GPU_UTILIZATION.labels(node, gpu_type).set(gpu_utilization)
    CLUSTER_CPU_UTILIZATION.labels(node).set(cpu_utilization)
    CLUSTER_GPU_MEMORY_USED_BYTES.labels(node, gpu_type).set(gpu_memory_bytes)
    CLUSTER_MEMORY_USED_BYTES.labels(node).set(system_memory_bytes)


# Training data diversity metrics
TRAINING_SAMPLES_BY_PHASE: Final[Counter] = Counter(
    "ringrift_training_samples_by_phase_total",
    "Training samples by game phase (opening/midgame/endgame).",
    labelnames=("board_type", "num_players", "phase"),
)

TRAINING_SAMPLES_BY_MOVE_NUMBER: Final[Histogram] = Histogram(
    "ringrift_training_sample_move_number",
    "Distribution of move numbers in training samples.",
    labelnames=("board_type", "num_players"),
    buckets=(5, 10, 20, 30, 50, 75, 100, 150, 200, 300),
)

TRAINING_DATA_RECENCY: Final[Gauge] = Gauge(
    "ringrift_training_data_recency_hours",
    "Age of oldest training sample in hours.",
    labelnames=("board_type", "num_players"),
)

TRAINING_UNIQUE_POSITIONS: Final[Gauge] = Gauge(
    "ringrift_training_unique_positions",
    "Number of unique positions in training buffer.",
    labelnames=("board_type", "num_players"),
)

TRAINING_POSITION_ENTROPY: Final[Gauge] = Gauge(
    "ringrift_training_position_entropy",
    "Entropy of position distribution in training buffer (higher = more diverse).",
    labelnames=("board_type", "num_players"),
)

# Model promotion metrics
PROMOTION_DECISIONS: Final[Counter] = Counter(
    "ringrift_promotion_decisions_total",
    "Total promotion decisions, labeled by type and outcome.",
    labelnames=("promotion_type", "outcome"),  # outcome: approved/rejected
)

PROMOTION_EXECUTIONS: Final[Counter] = Counter(
    "ringrift_promotion_executions_total",
    "Total promotion executions, labeled by type and result.",
    labelnames=("promotion_type", "result"),  # result: success/failure/dry_run
)

PROMOTION_ELO_IMPROVEMENT: Final[Histogram] = Histogram(
    "ringrift_promotion_elo_improvement",
    "Elo improvement at time of promotion decision.",
    labelnames=("promotion_type",),
    buckets=(-50, -25, 0, 10, 25, 50, 75, 100, 150, 200),
)

# Elo reconciliation metrics
ELO_SYNC_OPERATIONS: Final[Counter] = Counter(
    "ringrift_elo_sync_operations_total",
    "Total Elo sync operations, labeled by result.",
    labelnames=("remote_host", "result"),  # result: success/failure
)

ELO_SYNC_MATCHES_ADDED: Final[Counter] = Counter(
    "ringrift_elo_sync_matches_added_total",
    "Total matches added via Elo sync.",
    labelnames=("remote_host",),
)

ELO_SYNC_CONFLICTS: Final[Counter] = Counter(
    "ringrift_elo_sync_conflicts_total",
    "Total conflicts detected during Elo sync.",
    labelnames=("remote_host",),
)

ELO_DRIFT_MAX: Final[Gauge] = Gauge(
    "ringrift_elo_drift_max",
    "Maximum Elo drift detected across participants.",
    labelnames=("board_type", "num_players"),
)

ELO_DRIFT_AVG: Final[Gauge] = Gauge(
    "ringrift_elo_drift_avg",
    "Average Elo drift detected across participants.",
    labelnames=("board_type", "num_players"),
)

ELO_DRIFT_SIGNIFICANT: Final[Gauge] = Gauge(
    "ringrift_elo_drift_significant",
    "Whether significant Elo drift is detected (1=yes, 0=no).",
    labelnames=("board_type", "num_players"),
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


def record_game_outcome(
    board_type: str,
    num_players: int,
    winner: int | None,  # None for draw, player index (0-based) for win
    move_count: int,
    duration_seconds: float,
) -> None:
    """Record metrics for a completed selfplay game.

    Args:
        board_type: Board type (e.g., 'square8', 'hexagonal')
        num_players: Number of players (2, 3, or 4)
        winner: Player index (0-based) who won, or None for draw
        move_count: Total moves in the game
        duration_seconds: Game duration in seconds
    """
    np_str = str(num_players)

    # Record game completion
    GAMES_COMPLETED.labels(board_type, np_str).inc()

    # Record outcome
    if winner is None:
        GAME_OUTCOMES.labels(board_type, np_str, "draw").inc()
    else:
        GAME_OUTCOMES.labels(board_type, np_str, f"player_{winner}_win").inc()

    # Record moves
    GAMES_MOVES_TOTAL.labels(board_type, np_str).inc(move_count)

    # Record duration
    GAME_DURATION_SECONDS.labels(board_type, np_str).observe(duration_seconds)


def record_training_sample(
    board_type: str,
    num_players: int,
    move_number: int,
    total_moves: int,
) -> None:
    """Record metrics for a training sample.

    Args:
        board_type: Board type (e.g., 'square8', 'hexagonal')
        num_players: Number of players
        move_number: Move number in the game (1-indexed)
        total_moves: Total moves in the game
    """
    np_str = str(num_players)

    # Determine game phase
    if total_moves > 0:
        progress = move_number / total_moves
        if progress < 0.25:
            phase = "opening"
        elif progress < 0.75:
            phase = "midgame"
        else:
            phase = "endgame"
    else:
        phase = "unknown"

    TRAINING_SAMPLES_BY_PHASE.labels(board_type, np_str, phase).inc()
    TRAINING_SAMPLES_BY_MOVE_NUMBER.labels(board_type, np_str).observe(move_number)


def record_promotion_decision(
    promotion_type: str,
    approved: bool,
    elo_improvement: float | None = None,
) -> None:
    """Record metrics for a promotion decision.

    Args:
        promotion_type: Type of promotion (staging, production, tier, champion, rollback)
        approved: Whether the promotion was approved
        elo_improvement: Elo improvement at decision time (if available)
    """
    outcome = "approved" if approved else "rejected"
    PROMOTION_DECISIONS.labels(promotion_type, outcome).inc()

    if elo_improvement is not None:
        PROMOTION_ELO_IMPROVEMENT.labels(promotion_type).observe(elo_improvement)


def record_promotion_execution(
    promotion_type: str,
    success: bool,
    dry_run: bool = False,
) -> None:
    """Record metrics for a promotion execution.

    Args:
        promotion_type: Type of promotion
        success: Whether the execution succeeded
        dry_run: Whether this was a dry run
    """
    if dry_run:
        result = "dry_run"
    elif success:
        result = "success"
    else:
        result = "failure"
    PROMOTION_EXECUTIONS.labels(promotion_type, result).inc()


def record_elo_sync(
    remote_host: str,
    success: bool,
    matches_added: int = 0,
    conflicts: int = 0,
) -> None:
    """Record metrics for an Elo sync operation.

    Args:
        remote_host: Remote host synced from
        success: Whether the sync succeeded
        matches_added: Number of matches added
        conflicts: Number of conflicts detected
    """
    result = "success" if success else "failure"
    ELO_SYNC_OPERATIONS.labels(remote_host, result).inc()

    if matches_added > 0:
        ELO_SYNC_MATCHES_ADDED.labels(remote_host).inc(matches_added)

    if conflicts > 0:
        ELO_SYNC_CONFLICTS.labels(remote_host).inc(conflicts)


def record_elo_drift(
    board_type: str,
    num_players: int,
    max_drift: float,
    avg_drift: float,
    is_significant: bool,
) -> None:
    """Record metrics for Elo drift detection.

    Args:
        board_type: Board type
        num_players: Number of players
        max_drift: Maximum rating drift
        avg_drift: Average rating drift
        is_significant: Whether drift is significant
    """
    np_str = str(num_players)
    ELO_DRIFT_MAX.labels(board_type, np_str).set(max_drift)
    ELO_DRIFT_AVG.labels(board_type, np_str).set(avg_drift)
    ELO_DRIFT_SIGNIFICANT.labels(board_type, np_str).set(1 if is_significant else 0)


__all__ = [
    "AI_MOVE_REQUESTS",
    "AI_MOVE_LATENCY",
    "AI_INSTANCE_CACHE_LOOKUPS",
    "AI_INSTANCE_CACHE_SIZE",
    "PYTHON_INVARIANT_VIOLATIONS",
    "GAME_OUTCOMES",
    "GAMES_COMPLETED",
    "GAMES_MOVES_TOTAL",
    "GAME_DURATION_SECONDS",
    "WIN_RATE_BY_PLAYER",
    "DRAW_RATE",
    # Cluster cost metrics
    "CLUSTER_NODE_UP",
    "CLUSTER_NODE_COST_PER_HOUR",
    "CLUSTER_GPU_UTILIZATION",
    "CLUSTER_CPU_UTILIZATION",
    "CLUSTER_GPU_MEMORY_USED_BYTES",
    "CLUSTER_MEMORY_USED_BYTES",
    "GPU_HOURLY_RATES",
    # Training data diversity metrics
    "TRAINING_SAMPLES_BY_PHASE",
    "TRAINING_SAMPLES_BY_MOVE_NUMBER",
    "TRAINING_DATA_RECENCY",
    "TRAINING_UNIQUE_POSITIONS",
    "TRAINING_POSITION_ENTROPY",
    # Promotion metrics
    "PROMOTION_DECISIONS",
    "PROMOTION_EXECUTIONS",
    "PROMOTION_ELO_IMPROVEMENT",
    # Elo reconciliation metrics
    "ELO_SYNC_OPERATIONS",
    "ELO_SYNC_MATCHES_ADDED",
    "ELO_SYNC_CONFLICTS",
    "ELO_DRIFT_MAX",
    "ELO_DRIFT_AVG",
    "ELO_DRIFT_SIGNIFICANT",
    # Helper functions
    "observe_ai_move_start",
    "record_game_outcome",
    "record_training_sample",
    "report_cluster_node",
    "record_promotion_decision",
    "record_promotion_execution",
    "record_elo_sync",
    "record_elo_drift",
]

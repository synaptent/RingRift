"""Unified Threshold Constants for RingRift AI Service.

This module provides SINGLE SOURCE OF TRUTH for all threshold values
used across training, evaluation, promotion, and rollback systems.

Import these constants instead of hardcoding values:

    from app.config.thresholds import (
        TRAINING_TRIGGER_GAMES,
        ELO_DROP_ROLLBACK,
        ELO_IMPROVEMENT_PROMOTE,
    )

See docs/CONSOLIDATION_ROADMAP.md for consolidation context.
"""

# =============================================================================
# Training Thresholds
# =============================================================================

# Games needed to trigger training (per config)
TRAINING_TRIGGER_GAMES = 500

# Minimum interval between training runs (seconds)
TRAINING_MIN_INTERVAL_SECONDS = 1200  # 20 minutes

# Hours before a config is considered "stale" (no recent training)
TRAINING_STALENESS_HOURS = 6.0

# Bootstrap threshold for new configs (0 models)
TRAINING_BOOTSTRAP_GAMES = 50

# Maximum concurrent training jobs
TRAINING_MAX_CONCURRENT = 3

# =============================================================================
# Regression & Rollback Thresholds
# =============================================================================

# Elo drop that triggers rollback consideration
ELO_DROP_ROLLBACK = 50

# Win rate drop percentage that triggers rollback
WIN_RATE_DROP_ROLLBACK = 0.10  # 10%

# Error rate threshold for rollback
ERROR_RATE_ROLLBACK = 0.05  # 5%

# Minimum games for reliable regression detection
MIN_GAMES_REGRESSION = 50

# Consecutive regressions before forced rollback
CONSECUTIVE_REGRESSIONS_FORCE = 3

# =============================================================================
# Promotion Thresholds
# =============================================================================

# Elo improvement required for promotion
ELO_IMPROVEMENT_PROMOTE = 20

# Minimum games before eligible for promotion
MIN_GAMES_PROMOTE = 100

# Minimum win rate for promotion consideration
MIN_WIN_RATE_PROMOTE = 0.45

# Cooldown between promotion attempts (seconds)
PROMOTION_COOLDOWN_SECONDS = 900  # 15 minutes

# =============================================================================
# Elo Rating System
# =============================================================================

# Initial Elo rating for new models/players
INITIAL_ELO_RATING = 1500.0

# Minimum Elo rating (floor)
MIN_ELO_RATING = 100.0

# Maximum Elo rating (ceiling)
MAX_ELO_RATING = 3000.0

# =============================================================================
# Baseline Gating Thresholds
# =============================================================================
# Checkpoints must beat baselines at these rates to be considered "qualified"
# This prevents selecting checkpoints strong in neural-vs-neural but weak vs basics

# Minimum win rate against random AI for checkpoint qualification
MIN_WIN_RATE_VS_RANDOM = 0.85  # 85%

# Minimum win rate against heuristic AI for checkpoint qualification
MIN_WIN_RATE_VS_HEURISTIC = 0.60  # 60%

# Baseline Elo estimates for Elo calculation from win rates
BASELINE_ELO_RANDOM = 400
BASELINE_ELO_HEURISTIC = 1200

# =============================================================================
# Evaluation Thresholds
# =============================================================================

# Shadow tournament interval (seconds)
SHADOW_TOURNAMENT_INTERVAL = 900  # 15 minutes

# Games per config in shadow tournaments
SHADOW_GAMES_PER_CONFIG = 15

# Full tournament interval (seconds)
FULL_TOURNAMENT_INTERVAL = 3600  # 1 hour

# Games in full tournaments
FULL_TOURNAMENT_GAMES = 50

# Minimum games for Elo calculation
MIN_GAMES_FOR_ELO = 30

# Elo K-factor for rating updates
ELO_K_FACTOR = 32

# =============================================================================
# Signal Weights (for training triggers)
# =============================================================================

# Weight for data freshness signal
SIGNAL_WEIGHT_FRESHNESS = 1.0

# Weight for model staleness signal
SIGNAL_WEIGHT_STALENESS = 0.8

# Weight for performance regression signal (higher = more urgent)
SIGNAL_WEIGHT_REGRESSION = 1.5

# Bootstrap priority multiplier
SIGNAL_BOOTSTRAP_PRIORITY = 10.0

# =============================================================================
# NNUE-Specific Thresholds
# =============================================================================

# Minimum games for NNUE value training
NNUE_MIN_GAMES = 10000

# Minimum games for NNUE policy training
NNUE_POLICY_MIN_GAMES = 5000

# Minimum games for CMA-ES heuristic optimization
CMAES_MIN_GAMES = 20000

# =============================================================================
# Plateau Detection
# =============================================================================

# Hours of no Elo improvement before plateau detected
PLATEAU_HOURS = 24

# Minimum Elo change to not be considered plateau
PLATEAU_MIN_ELO_CHANGE = 5

# Consecutive plateaus before architecture search
PLATEAU_TRIGGER_NAS = 3

# =============================================================================
# Resource Limits
# =============================================================================

# CPU utilization warning threshold
CPU_WARNING_PERCENT = 70

# CPU utilization critical threshold
CPU_CRITICAL_PERCENT = 80

# GPU utilization warning threshold
GPU_WARNING_PERCENT = 70

# GPU utilization critical threshold
GPU_CRITICAL_PERCENT = 80

# Memory utilization warning threshold
MEMORY_WARNING_PERCENT = 70

# Memory utilization critical threshold
MEMORY_CRITICAL_PERCENT = 80

# Disk utilization warning threshold
DISK_WARNING_PERCENT = 65

# Disk utilization critical threshold
DISK_CRITICAL_PERCENT = 70

# =============================================================================
# Network/SSH Timeouts
# =============================================================================

# SSH connection timeout (seconds)
SSH_CONNECT_TIMEOUT = 10

# SSH command timeout (seconds)
SSH_COMMAND_TIMEOUT = 30

# HTTP request timeout (seconds)
HTTP_TIMEOUT = 30

# P2P operations timeout (seconds)
P2P_TIMEOUT = 30

# =============================================================================
# Training Pipeline Timeouts (December 2025)
# =============================================================================

# SQLite connection timeout (seconds)
SQLITE_TIMEOUT = 30

# SQLite short operations timeout (seconds)
SQLITE_SHORT_TIMEOUT = 10

# URL open timeout for quick health checks (seconds)
URLOPEN_SHORT_TIMEOUT = 5

# URL open timeout for data operations (seconds)
URLOPEN_TIMEOUT = 10

# Rsync transfer timeout (seconds)
RSYNC_TIMEOUT = 30

# Async subprocess wait timeout (seconds)
ASYNC_SUBPROCESS_TIMEOUT = 180

# Process/thread join timeout (seconds)
THREAD_JOIN_TIMEOUT = 5

# Future result timeout for parallel operations (seconds)
FUTURE_RESULT_TIMEOUT = 300

# Checkpoint future timeout (seconds)
CHECKPOINT_FUTURE_TIMEOUT = 120

# Long training job timeout (seconds) - 4 hours
TRAINING_JOB_TIMEOUT = 14400

# Training lock timeout (seconds) - 2 hours
TRAINING_LOCK_TIMEOUT = 7200

# Resource wait timeout (seconds)
RESOURCE_WAIT_TIMEOUT = 300

# =============================================================================
# Cluster Health & Monitoring
# =============================================================================

# Heartbeat interval for health checks (seconds)
HEARTBEAT_INTERVAL = 30

# Peer timeout - no heartbeat means dead (seconds)
PEER_TIMEOUT = 90

# Election timeout for leader election (seconds)
ELECTION_TIMEOUT = 10

# Leader lease renewal interval (seconds)
LEADER_LEASE_RENEW_INTERVAL = 10

# Job status check interval (seconds)
JOB_CHECK_INTERVAL = 60

# Peer discovery broadcast interval (seconds)
DISCOVERY_INTERVAL = 120

# Stale entry cleanup interval (seconds)
STALE_CLEANUP_INTERVAL = 60

# =============================================================================
# Data Sync Intervals
# =============================================================================

# Main sync cycle interval (seconds)
SYNC_INTERVAL = 60

# Transport health check interval (seconds)
TRANSPORT_HEALTH_CHECK_INTERVAL = 300  # 5 minutes

# Freshness check interval (seconds)
FRESHNESS_CHECK_INTERVAL = 60

# Checkpoint interval for ephemeral data (seconds)
CHECKPOINT_INTERVAL = 60

# Stale data thresholds (December 2025)
STALE_DATA_THRESHOLD_SECONDS = 1800  # 30 minutes - data older than this is stale
CRITICAL_STALE_THRESHOLD_SECONDS = 3600  # 1 hour - urgent sync needed

# Max items in sync queue
MAX_SYNC_QUEUE_SIZE = 20

# =============================================================================
# Lock & Mutex Settings
# =============================================================================

# Default lock timeout (seconds)
DEFAULT_LOCK_TIMEOUT = 3600  # 1 hour

# Lock acquisition timeout (seconds)
DEFAULT_ACQUIRE_TIMEOUT = 60

# Lock poll interval when waiting (seconds)
LOCK_POLL_INTERVAL = 0.5

# Crash detection threshold - no heartbeat (seconds)
CRASH_DETECTION_THRESHOLD = 60

# =============================================================================
# Circuit Breaker & Retry
# =============================================================================

# Number of failures before circuit opens
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 3

# Time to wait before attempting recovery (seconds)
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 300  # 5 minutes

# Per-operation circuit breaker configurations (December 2025)
# Keys: failure_threshold (int), recovery_timeout (float in seconds)
CIRCUIT_BREAKER_CONFIGS = {
    "ssh": {"failure_threshold": 3, "recovery_timeout": 60.0},
    "http": {"failure_threshold": 5, "recovery_timeout": 30.0},
    "p2p": {"failure_threshold": 3, "recovery_timeout": 45.0},
    "aria2": {"failure_threshold": 2, "recovery_timeout": 120.0},
    "rsync": {"failure_threshold": 2, "recovery_timeout": 90.0},
    "database": {"failure_threshold": 3, "recovery_timeout": 30.0},
    "external_api": {"failure_threshold": 5, "recovery_timeout": 60.0},
}

# Maximum retry attempts for recoverable errors
MAX_RETRY_ATTEMPTS = 3

# Base delay between retries (seconds)
RETRY_BASE_DELAY = 1.0

# Maximum delay between retries (seconds)
RETRY_MAX_DELAY = 60.0

# =============================================================================
# Model Management
# =============================================================================

# Model count threshold to trigger culling
MODEL_CULL_THRESHOLD = 100

# Win rate threshold for model promotion
PROMOTION_WIN_RATE_THRESHOLD = 0.55  # 55%

# ELO underserved threshold (fewer games = underserved)
ELO_UNDERSERVED_THRESHOLD = 100

# =============================================================================
# File Transfer
# =============================================================================

# Maximum timeout for batch operations (seconds)
MAX_BATCH_TIMEOUT = 1800  # 30 minutes

# Maximum timeout per file (seconds)
MAX_PER_FILE_TIMEOUT = 120  # 2 minutes

# =============================================================================
# Shadow Validation
# =============================================================================

# Maximum divergence allowed between implementations
DIVERGENCE_THRESHOLD = 0.001  # 0.1%


# =============================================================================
# Distillation & Temperature (December 2025)
# =============================================================================

# Knowledge distillation temperature (higher = softer)
DISTILL_TEMPERATURE = 3.0

# Distillation loss weight (alpha between student/teacher)
DISTILL_ALPHA = 0.7

# Value head distillation temperature
VALUE_TEMPERATURE = 1.0

# Policy head distillation temperature
POLICY_TEMPERATURE = 3.0

# EMA decay for model averaging
EMA_DECAY = 0.999

# Focal loss gamma for hard example mining
FOCAL_GAMMA = 2.0

# Policy label smoothing factor
POLICY_LABEL_SMOOTHING = 0.05


# =============================================================================
# Quality Thresholds (December 2025)
# =============================================================================

# Minimum quality score for training data
MIN_QUALITY_FOR_TRAINING = 0.3

# Minimum quality score for priority sync (higher than training minimum)
MIN_QUALITY_FOR_PRIORITY_SYNC = 0.5

# High quality threshold (affects priority)
HIGH_QUALITY_THRESHOLD = 0.7

# Overfit detection threshold (train-val gap)
OVERFIT_THRESHOLD = 0.15


# =============================================================================
# Selfplay & Gameplay (December 2025)
# =============================================================================

# Maximum moves per game before termination
SELFPLAY_MAX_MOVES = 10000

# Base temperature for move selection
SELFPLAY_TEMPERATURE = 1.0

# Higher temperature for opening moves (diversity)
SELFPLAY_OPENING_TEMPERATURE = 1.5

# Quality threshold for adaptive engine selection
SELFPLAY_QUALITY_THRESHOLD = 0.7


# =============================================================================
# GPU Batch Scaling (December 2025)
# =============================================================================

# Reserved GPU memory for system overhead (GB)
GPU_RESERVED_MEMORY_GB = 8.0

# Maximum batch size regardless of memory
GPU_MAX_BATCH_SIZE = 16384

# Batch multipliers by GPU type (base batch = 64)
GH200_BATCH_MULTIPLIER = 64  # NVIDIA GH200 (96GB)
H100_BATCH_MULTIPLIER = 32   # NVIDIA H100 (80GB)
A100_BATCH_MULTIPLIER = 16   # NVIDIA A100 (40/80GB)
A10_BATCH_MULTIPLIER = 8     # NVIDIA A10 (24GB)
RTX_BATCH_MULTIPLIER = 4     # RTX 4090/3090 (24GB)


# =============================================================================
# Coordination & Concurrency (December 2025)
# =============================================================================

# Idle GPU utilization threshold (below = available)
IDLE_GPU_THRESHOLD = 10

# Default P2P communication port
P2P_DEFAULT_PORT = 8770

# Ephemeral data evacuation threshold (games)
EVACUATION_THRESHOLD = 50

# Maximum concurrent syncs per host
MAX_CONCURRENT_SYNCS_PER_HOST = 1

# Maximum global concurrent syncs
MAX_CONCURRENT_SYNCS_GLOBAL = 5

# Target GPU utilization range
TARGET_GPU_UTILIZATION_MIN = 60
TARGET_GPU_UTILIZATION_MAX = 80


# =============================================================================
# Data Streaming (December 2025)
# =============================================================================

# Streaming buffer size (games)
DATA_STREAMING_BUFFER_SIZE = 10000

# Minimum buffer fill before training starts (0-1)
DATA_STREAMING_MIN_BUFFER_FILL = 0.2

# Deduplication window size (games)
DATA_STREAMING_DEDUPE_WINDOW = 50000

# Weighting factors for data sampling
DATA_SAMPLING_RECENCY_WEIGHT = 0.3
DATA_SAMPLING_QUALITY_WEIGHT = 0.4


# =============================================================================
# Feedback Loop Thresholds (December 2025)
# =============================================================================

# ELO momentum lookback (number of updates)
ELO_MOMENTUM_LOOKBACK = 5

# Strong ELO improvement threshold (per update)
ELO_STRONG_IMPROVEMENT = 25.0

# Moderate ELO improvement threshold
ELO_MODERATE_IMPROVEMENT = 12.0

# ELO plateau threshold (below = stagnant)
ELO_PLATEAU_THRESHOLD = 5.0

# Training intensity multiplier range
MAX_INTENSITY_MULTIPLIER = 2.5
MIN_INTENSITY_MULTIPLIER = 0.5


# =============================================================================
# Alert Levels and Monitoring Thresholds
# =============================================================================
# Consolidated from app/monitoring/thresholds.py

from enum import Enum
from typing import Any, Dict, Optional


class AlertLevel(str, Enum):
    """Alert severity levels for monitoring."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


# Master threshold configuration for monitoring systems
# All monitoring scripts should reference these values instead of hardcoding
MONITORING_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    # Disk space monitoring
    "disk": {
        "warning": DISK_WARNING_PERCENT,
        "critical": DISK_CRITICAL_PERCENT,
        "fatal": 95,
        "unit": "percent",
        "description": "Disk space utilization thresholds",
    },

    # GPU monitoring
    "gpu_utilization": {
        "idle": 5,
        "low": 20,
        "normal": 50,
        "warning": GPU_WARNING_PERCENT,
        "critical": GPU_CRITICAL_PERCENT,
        "unit": "percent",
        "description": "GPU utilization levels",
    },
    "gpu_memory": {
        "warning": 85,
        "critical": 95,
        "unit": "percent",
        "description": "GPU memory utilization",
    },

    # Training monitoring
    "training": {
        "stale_hours": TRAINING_STALENESS_HOURS,
        "model_stale_hours": 48,
        "min_batch_rate": 10,
        "max_loss_increase": 0.5,
        "description": "Training progress thresholds",
    },

    # Data quality monitoring
    "data_quality": {
        "draw_rate_threshold": 0.20,
        "min_game_length": 10,
        "max_game_length": 500,
        "nan_threshold": 0.001,
        "zero_feature_threshold": 0.05,
        "description": "Data quality metrics",
    },

    # Cluster health
    "cluster": {
        "min_nodes_online": 5,
        "node_timeout_seconds": PEER_TIMEOUT,
        "heartbeat_interval": HEARTBEAT_INTERVAL,
        "max_coordinator_lag_seconds": 300,
        "description": "Cluster health requirements",
    },

    # Selfplay monitoring
    "selfplay": {
        "min_games_per_hour": 100,
        "max_game_duration_seconds": 600,
        "min_move_count": 5,
        "description": "Selfplay generation metrics",
    },

    # Network/P2P monitoring
    "network": {
        "ping_timeout_ms": 5000,
        "max_relay_latency_ms": 200,
        "reconnect_interval_seconds": 30,
        "description": "Network health thresholds",
    },

    # Memory monitoring
    "memory": {
        "warning": MEMORY_WARNING_PERCENT,
        "critical": MEMORY_CRITICAL_PERCENT,
        "unit": "percent",
        "description": "System memory thresholds",
    },
}

# Backwards compatibility alias
THRESHOLDS = MONITORING_THRESHOLDS


def get_threshold(
    category: str,
    key: str,
    default: Optional[Any] = None,
) -> Any:
    """Get a specific monitoring threshold value.

    Args:
        category: Threshold category (e.g., "disk", "gpu_utilization")
        key: Specific threshold key (e.g., "warning", "critical")
        default: Default value if not found

    Returns:
        Threshold value or default

    Example:
        disk_warning = get_threshold("disk", "warning")  # Returns 65
    """
    if category not in MONITORING_THRESHOLDS:
        return default
    return MONITORING_THRESHOLDS[category].get(key, default)


def should_alert(
    category: str,
    value: float,
    level: str = "warning",
    comparison: str = "gte",
) -> bool:
    """Check if a value exceeds the threshold for alert.

    Args:
        category: Threshold category
        value: Current value to check
        level: Alert level to check against
        comparison: Comparison type (gte=>=, lte=<=, gt=>, lt=<, eq===)

    Returns:
        True if value triggers alert at specified level

    Example:
        if should_alert("disk", 75, "warning"):
            send_warning()
    """
    threshold = get_threshold(category, level)
    if threshold is None:
        return False

    comparisons = {
        "gte": lambda v, t: v >= t,
        "lte": lambda v, t: v <= t,
        "gt": lambda v, t: v > t,
        "lt": lambda v, t: v < t,
        "eq": lambda v, t: v == t,
    }

    compare_fn = comparisons.get(comparison, comparisons["gte"])
    return compare_fn(value, threshold)


def get_all_thresholds() -> Dict[str, Dict[str, Any]]:
    """Get all monitoring thresholds for display/documentation."""
    return MONITORING_THRESHOLDS.copy()


def update_threshold(category: str, key: str, value: Any) -> None:
    """Update a threshold value at runtime.

    Use sparingly - primarily for testing or dynamic configuration.
    """
    if category in MONITORING_THRESHOLDS:
        MONITORING_THRESHOLDS[category][key] = value

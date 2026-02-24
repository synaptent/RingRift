"""Centralized threshold constants for coordination daemons.

This module consolidates hardcoded numeric thresholds used across coordination,
sync, and daemon modules. Use these constants instead of magic numbers.

December 2025: Created to reduce technical debt from 200+ scattered threshold values.

Usage:
    from app.config.daemon_thresholds import (
        TimeoutDefaults,
        RetryDefaults,
        ResourceThresholds,
        QueueThresholds,
    )

    # Use constants
    timeout = TimeoutDefaults.HTTP_REQUEST
    max_retries = RetryDefaults.STANDARD

Categories:
    - TimeoutDefaults: Network, database, and operation timeouts
    - RetryDefaults: Retry counts and circuit breaker thresholds
    - ResourceThresholds: CPU, memory, disk, GPU utilization levels
    - QueueThresholds: Backpressure and queue size limits
    - HealthThresholds: Node health and failure detection
    - DataThresholds: Quality, freshness, and sync thresholds
    - IntervalDefaults: Daemon scheduling intervals
    - SpawnLimits: Job and process spawn constraints
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = [
    "TimeoutDefaults",
    "RetryDefaults",
    "ResourceThresholds",
    "QueueThresholds",
    "HealthThresholds",
    "DataThresholds",
    "IntervalDefaults",
    "SpawnLimits",
    "CircuitBreakerDefaults",
    "ExplorationBoosts",
]


# =============================================================================
# TIMEOUT DEFAULTS
# =============================================================================


@dataclass(frozen=True)
class TimeoutDefaults:
    """Timeout values in seconds for various operations."""

    # Database operations
    DB_OPERATION: Final[float] = 5.0
    DB_TRANSACTION: Final[float] = 10.0
    DB_LONG_QUERY: Final[float] = 30.0

    # Network operations
    HTTP_REQUEST: Final[float] = 15.0
    HTTP_CONNECT: Final[float] = 5.0
    HTTP_READ: Final[float] = 30.0

    # SSH operations
    SSH_CONNECT: Final[float] = 10.0
    SSH_COMMAND: Final[float] = 30.0
    SSH_LONG_COMMAND: Final[float] = 120.0

    # Lock operations
    LOCK_ACQUIRE: Final[float] = 30.0
    LOCK_HOLD_MAX: Final[float] = 3600.0  # 1 hour max hold

    # Async task operations
    ASYNC_TASK_SHORT: Final[float] = 5.0
    ASYNC_TASK_STANDARD: Final[float] = 30.0
    ASYNC_TASK_LONG: Final[float] = 120.0

    # Sync operations
    SYNC_STALL_DETECTION: Final[float] = 300.0  # 5 minutes
    SYNC_STALL_CRITICAL: Final[float] = 600.0  # 10 minutes

    # Process operations
    PROCESS_CRASH_DETECTION: Final[float] = 60.0
    NODE_RECOVERY: Final[float] = 120.0


# =============================================================================
# RETRY DEFAULTS
# =============================================================================


@dataclass(frozen=True)
class RetryDefaults:
    """Retry counts for various operations."""

    # Retry counts
    MINIMAL: Final[int] = 1
    QUICK: Final[int] = 2
    STANDARD: Final[int] = 3
    PATIENT: Final[int] = 5
    PERSISTENT: Final[int] = 10

    # Retry delays (seconds)
    BASE_DELAY: Final[float] = 1.0
    SSH_DELAY: Final[float] = 5.0
    HTTP_DELAY: Final[float] = 2.0
    MAX_DELAY: Final[float] = 60.0

    # Jitter factor (0-1)
    DEFAULT_JITTER: Final[float] = 0.1
    HIGH_JITTER: Final[float] = 0.25


# CircuitBreakerDefaults moved to app/config/coordination_defaults.py
# to have a single source of truth for circuit breaker configuration.
# See coordination_defaults.CircuitBreakerDefaults for the canonical implementation.


# =============================================================================
# RESOURCE THRESHOLDS
# =============================================================================


@dataclass(frozen=True)
class ResourceThresholds:
    """Resource utilization thresholds (as percentages 0.0-1.0)."""

    # Disk usage - values from app.config.thresholds (canonical source)
    # 0.70 = DISK_SYNC_TARGET_PERCENT, 0.85 = DISK_PRODUCTION_HALT_PERCENT, 0.90 = DISK_CRITICAL_PERCENT
    DISK_WARNING: Final[float] = 0.70
    DISK_CRITICAL: Final[float] = 0.85
    DISK_EMERGENCY: Final[float] = 0.90

    # Memory usage
    MEMORY_WARNING: Final[float] = 0.70
    MEMORY_CRITICAL: Final[float] = 0.85
    MEMORY_EMERGENCY: Final[float] = 0.95

    # CPU usage
    CPU_WARNING: Final[float] = 0.70
    CPU_CRITICAL: Final[float] = 0.80
    LOAD_CRITICAL_MULTIPLIER: Final[float] = 1.5  # x CPU cores

    # GPU usage
    GPU_IDLE: Final[float] = 0.20
    GPU_WARNING: Final[float] = 0.70
    GPU_CRITICAL: Final[float] = 0.80
    GPU_FULL: Final[float] = 0.90


# =============================================================================
# QUEUE & BACKPRESSURE THRESHOLDS
# =============================================================================


@dataclass(frozen=True)
class QueueThresholds:
    """Queue depth and backpressure thresholds."""

    # Queue depth
    QUEUE_LOW: Final[int] = 10
    QUEUE_MEDIUM: Final[int] = 50
    QUEUE_HIGH: Final[int] = 100
    QUEUE_CRITICAL: Final[int] = 200

    # Training job pressure
    TRAINING_LOW: Final[int] = 2
    TRAINING_HIGH: Final[int] = 10

    # Sync pending items
    SYNC_LOW: Final[int] = 5
    SYNC_HIGH: Final[int] = 50

    # Backpressure thresholds
    BACKPRESSURE_LIGHT: Final[float] = 0.5
    BACKPRESSURE_MODERATE: Final[float] = 0.7
    BACKPRESSURE_HEAVY: Final[float] = 0.8
    BACKPRESSURE_CRITICAL: Final[float] = 0.9

    # Node overload
    NODE_OVERLOAD_JOBS: Final[int] = 5

    # Cache TTL
    SIGNAL_CACHE_TTL: Final[float] = 10.0


# =============================================================================
# HEALTH THRESHOLDS
# =============================================================================


@dataclass(frozen=True)
class HealthThresholds:
    """Node health and failure detection thresholds."""

    # Failure counts for health state transitions
    DEGRADED_FAILURE_COUNT: Final[int] = 1
    UNHEALTHY_FAILURE_COUNT: Final[int] = 3
    EVICTION_FAILURE_COUNT: Final[int] = 5

    # Error rate thresholds
    ERROR_WARNING: Final[float] = 0.01  # 1%
    ERROR_CRITICAL: Final[float] = 0.05  # 5%

    # Recovery timing
    RECOVERY_ATTEMPT_COOLDOWN: Final[float] = 300.0  # 5 minutes
    ESCALATION_COOLDOWN: Final[float] = 3600.0  # 1 hour

    # Health score weights
    NODE_AVAILABILITY_WEIGHT: Final[float] = 0.40
    CIRCUIT_HEALTH_WEIGHT: Final[float] = 0.25
    ERROR_RATE_WEIGHT: Final[float] = 0.20
    RECOVERY_WEIGHT: Final[float] = 0.15


# =============================================================================
# DATA QUALITY & FRESHNESS THRESHOLDS
# =============================================================================


@dataclass(frozen=True)
class DataThresholds:
    """Data quality and freshness thresholds."""

    # Quality scores (0.0-1.0)
    QUALITY_POOR: Final[float] = 0.50
    QUALITY_ADEQUATE: Final[float] = 0.65
    QUALITY_GOOD: Final[float] = 0.80
    QUALITY_EXCELLENT: Final[float] = 0.90

    # Win rate requirements
    WIN_RATE_VS_RANDOM: Final[float] = 0.85
    WIN_RATE_VS_HEURISTIC: Final[float] = 0.60
    MASTERY_WIN_RATE: Final[float] = 0.85

    # Policy accuracy
    POLICY_ACCURACY_MINIMUM: Final[float] = 0.75

    # Data freshness (hours)
    # Dec 29: Relaxed freshness gates for better training throughput
    # Fresh data: 2h (was 1h) - triggers priority sync if older
    # Stale warning: 8h (was 4h) - logs warning but continues
    # Stale critical: 48h (was 24h) - blocks training
    FRESH_DATA_MAX_AGE: Final[float] = 2.0
    STALE_DATA_WARNING: Final[float] = 8.0
    STALE_DATA_CRITICAL: Final[float] = 48.0

    # Curriculum weights stale (seconds)
    CURRICULUM_WEIGHTS_STALE: Final[float] = 7200.0  # 2 hours

    # Ephemeral data thresholds
    EPHEMERAL_EVACUATION_GAMES: Final[int] = 50
    EPHEMERAL_CRITICAL_GAMES: Final[int] = 10

    # State history size
    MAX_QUALITY_HISTORY: Final[int] = 100


# =============================================================================
# INTERVAL DEFAULTS
# =============================================================================


@dataclass(frozen=True)
class IntervalDefaults:
    """Daemon scheduling intervals in seconds."""

    # Sync intervals
    AUTO_SYNC: Final[float] = 30.0
    FULL_DISCOVERY: Final[float] = 300.0
    EPHEMERAL_POLL: Final[float] = 5.0

    # Priority updates
    PRIORITY_UPDATE: Final[float] = 15.0
    NODE_CAPABILITY_UPDATE: Final[float] = 60.0

    # Health checks
    DAEMON_HEALTH_CHECK: Final[float] = 60.0
    QUICK_HEALTH_CHECK: Final[float] = 10.0

    # Data freshness
    FRESHNESS_CHECK: Final[float] = 300.0  # 5 minutes

    # Cooldowns
    FEEDBACK_ADJUSTMENT: Final[float] = 300.0  # 5 minutes
    CURRICULUM_ADVANCEMENT: Final[float] = 600.0  # 10 minutes
    BACKUP_DEBOUNCE: Final[float] = 60.0

    # Mastery check
    MASTERY_CHECK: Final[float] = 120.0


# =============================================================================
# SPAWN LIMITS
# =============================================================================


@dataclass(frozen=True)
class SpawnLimits:
    """Job and process spawn constraints."""

    # Per-node limits
    MAX_SELFPLAY_PER_NODE: Final[int] = 32
    MAX_TRAINING_PER_NODE: Final[int] = 2
    FALLBACK_MAX_JOBS_PER_GPU: Final[int] = 4

    # Cluster-wide limits
    MAX_TOTAL_SELFPLAY: Final[int] = 200
    MAX_TOTAL_TRAINING: Final[int] = 20

    # Spawn rate limiting
    MAX_SPAWNS_PER_WINDOW: Final[int] = 30
    SPAWN_WINDOW_SECONDS: Final[float] = 60.0

    # Spawn delays (seconds)
    NORMAL_SPAWN_DELAY: Final[float] = 0.0
    WARNING_SPAWN_DELAY: Final[float] = 1.0
    CRITICAL_SPAWN_DELAY: Final[float] = 5.0


# =============================================================================
# EXPLORATION & CURRICULUM BOOSTS
# =============================================================================


@dataclass(frozen=True)
class ExplorationBoosts:
    """Exploration boost and penalty factors."""

    # Exploration boost multipliers
    BOOST_LOW: Final[float] = 1.1
    BOOST_NORMAL: Final[float] = 1.3
    BOOST_HIGH: Final[float] = 1.5
    BOOST_AGGRESSIVE: Final[float] = 2.0

    # Reduction factors
    SUCCESS_REDUCTION: Final[float] = 0.9
    PLATEAU_REDUCTION: Final[float] = 0.8

    # Penalty factors
    QUALITY_PENALTY: Final[float] = 0.15
    WEIGHT_REDUCTION_PER_PENALTY: Final[float] = 0.15
    WEIGHT_INCREASE_PER_FAILURE: Final[float] = 0.20

    # Anomaly handling
    ANOMALY_BOOST_PER_ANOMALY: Final[float] = 1.15
    MOMENTUM_WEIGHT_BOOST: Final[float] = 0.3

    # Plateau detection
    PLATEAU_VARIANCE: Final[float] = 0.01
    PLATEAU_WINDOW_SIZE: Final[int] = 5
    PLATEAU_EXTRA_SELFPLAY: Final[int] = 5000
    PLATEAU_EPOCH_EXTENSION: Final[float] = 1.5

    # ELO thresholds for regression detection
    REGRESSION_ELO_DROP: Final[int] = 50
    SEVERE_REGRESSION_ELO_DROP: Final[int] = 100
    ELO_UNDERSERVED: Final[int] = 100

    # Model adjustments
    WEAK_MODEL_SEARCH_BUDGET_MULTIPLIER: Final[float] = 0.9
    QUALITY_THRESHOLD_BOOST: Final[float] = 0.05

    # Games for mastery
    MIN_GAMES_FOR_MASTERY: Final[int] = 20


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_timeout(category: str) -> float:
    """Get a timeout value by category name.

    Args:
        category: One of 'http', 'ssh', 'db', 'lock', 'async'

    Returns:
        Default timeout in seconds
    """
    category_map = {
        "http": TimeoutDefaults.HTTP_REQUEST,
        "ssh": TimeoutDefaults.SSH_COMMAND,
        "db": TimeoutDefaults.DB_OPERATION,
        "lock": TimeoutDefaults.LOCK_ACQUIRE,
        "async": TimeoutDefaults.ASYNC_TASK_STANDARD,
    }
    return category_map.get(category, TimeoutDefaults.ASYNC_TASK_STANDARD)


def get_retry_config(mode: str = "standard") -> tuple[int, float, float]:
    """Get retry configuration by mode.

    Args:
        mode: One of 'quick', 'standard', 'patient', 'ssh', 'http'

    Returns:
        Tuple of (max_attempts, base_delay, max_delay)
    """
    configs = {
        "quick": (RetryDefaults.QUICK, 0.5, 5.0),
        "standard": (RetryDefaults.STANDARD, 1.0, 30.0),
        "patient": (RetryDefaults.PATIENT, 2.0, 60.0),
        "ssh": (RetryDefaults.STANDARD, RetryDefaults.SSH_DELAY, 30.0),
        "http": (RetryDefaults.PATIENT, RetryDefaults.HTTP_DELAY, 15.0),
    }
    return configs.get(mode, configs["standard"])


def is_resource_critical(disk: float = 0, memory: float = 0, cpu: float = 0) -> bool:
    """Check if any resource is at critical level.

    Args:
        disk: Disk usage ratio (0.0-1.0)
        memory: Memory usage ratio (0.0-1.0)
        cpu: CPU usage ratio (0.0-1.0)

    Returns:
        True if any resource is critical
    """
    return (
        disk >= ResourceThresholds.DISK_CRITICAL
        or memory >= ResourceThresholds.MEMORY_CRITICAL
        or cpu >= ResourceThresholds.CPU_CRITICAL
    )

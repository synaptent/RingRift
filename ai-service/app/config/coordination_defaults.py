"""Centralized Coordination Defaults for RingRift AI.

This module provides SINGLE SOURCE OF TRUTH for all coordination-related
constants used across distributed locking, transport, sync, and scheduling.

Import these constants instead of hardcoding values:

    from app.config.coordination_defaults import (
        LockDefaults,
        TransportDefaults,
        SyncDefaults,
        HeartbeatDefaults,
    )

    # Use in your code
    timeout = LockDefaults.ACQUIRE_TIMEOUT
    max_syncs = SyncDefaults.MAX_CONCURRENT_PER_HOST

Environment variables can override defaults:
    RINGRIFT_LOCK_TIMEOUT=7200  # Override lock timeout to 2 hours
    RINGRIFT_HEARTBEAT_INTERVAL=45  # Override heartbeat to 45s
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(key: str, default: int) -> int:
    """Get integer from environment or use default."""
    return int(os.environ.get(key, default))


def _env_float(key: str, default: float) -> float:
    """Get float from environment or use default."""
    return float(os.environ.get(key, default))


# =============================================================================
# Distributed Locking Defaults
# =============================================================================

@dataclass(frozen=True)
class LockDefaults:
    """Default values for distributed locking.

    Used by: app/coordination/distributed_lock.py
    """
    # Maximum time a lock can be held (seconds)
    LOCK_TIMEOUT: int = _env_int("RINGRIFT_LOCK_TIMEOUT", 3600)  # 1 hour

    # Time to wait when trying to acquire a lock (seconds)
    ACQUIRE_TIMEOUT: int = _env_int("RINGRIFT_LOCK_ACQUIRE_TIMEOUT", 60)

    # Retry interval when waiting for lock (seconds)
    RETRY_INTERVAL: float = _env_float("RINGRIFT_LOCK_RETRY_INTERVAL", 1.0)

    # Training-specific lock timeout (longer for training jobs)
    TRAINING_LOCK_TIMEOUT: int = _env_int("RINGRIFT_TRAINING_LOCK_TIMEOUT", 7200)  # 2 hours


# =============================================================================
# Network Transport Defaults
# =============================================================================

@dataclass(frozen=True)
class TransportDefaults:
    """Default values for network transport operations.

    Used by: app/coordination/cluster_transport.py
    """
    # Connection timeout (seconds) - increased for Tailscale VPN stability
    # Dec 2025: Increased from 30 to 45 for better Tailscale reconnection handling
    CONNECT_TIMEOUT: int = _env_int("RINGRIFT_CONNECT_TIMEOUT", 45)

    # Operation timeout (seconds) - for large transfers
    OPERATION_TIMEOUT: int = _env_int("RINGRIFT_OPERATION_TIMEOUT", 180)

    # HTTP request timeout (seconds)
    HTTP_TIMEOUT: int = _env_int("RINGRIFT_HTTP_TIMEOUT", 30)

    # Circuit breaker recovery timeout (seconds)
    CIRCUIT_BREAKER_RECOVERY: int = _env_int("RINGRIFT_CIRCUIT_BREAKER_RECOVERY", 300)

    # SSH timeout for remote operations
    # Dec 2025: Increased from 30 to 60 for Tailscale userland stability
    SSH_TIMEOUT: int = _env_int("RINGRIFT_SSH_TIMEOUT", 60)

    # Maximum retries for failed operations
    MAX_RETRIES: int = _env_int("RINGRIFT_MAX_RETRIES", 3)


# =============================================================================
# Sync Operation Defaults
# =============================================================================

@dataclass(frozen=True)
class SyncDefaults:
    """Default values for sync operations.

    Used by: app/coordination/sync_mutex.py, app/distributed/sync_orchestrator.py

    Dec 2025: Increased concurrency and reduced intervals for 43-node cluster:
    - MAX_CONCURRENT_CLUSTER: 5 → 10 (3 was bottleneck with 40+ nodes)
    - MAX_CONCURRENT_PER_HOST: 1 → 2 (high-bandwidth nodes can handle more)
    - DATA_SYNC_INTERVAL: 300 → 120 (faster data availability)
    """
    # Sync lock timeout (seconds)
    LOCK_TIMEOUT: int = _env_int("RINGRIFT_SYNC_LOCK_TIMEOUT", 120)

    # Maximum concurrent syncs per host (Dec 2025: 1 → 2 for faster throughput)
    MAX_CONCURRENT_PER_HOST: int = _env_int("RINGRIFT_MAX_SYNCS_PER_HOST", 2)

    # Maximum concurrent syncs cluster-wide (Dec 2025: 5 → 10 for 43-node cluster)
    MAX_CONCURRENT_CLUSTER: int = _env_int("RINGRIFT_MAX_SYNCS_CLUSTER", 10)

    # Data sync interval (seconds) (Dec 2025: 300 → 120 for faster availability)
    DATA_SYNC_INTERVAL: float = _env_float("RINGRIFT_DATA_SYNC_INTERVAL", 120.0)

    # Model sync interval (seconds)
    MODEL_SYNC_INTERVAL: float = _env_float("RINGRIFT_MODEL_SYNC_INTERVAL", 600.0)

    # Elo sync interval (seconds)
    ELO_SYNC_INTERVAL: float = _env_float("RINGRIFT_ELO_SYNC_INTERVAL", 60.0)

    # Registry sync interval (seconds)
    REGISTRY_SYNC_INTERVAL: float = _env_float("RINGRIFT_REGISTRY_SYNC_INTERVAL", 120.0)


# =============================================================================
# Heartbeat Defaults
# =============================================================================

@dataclass(frozen=True)
class HeartbeatDefaults:
    """Default values for heartbeat and liveness detection.

    Used by: app/coordination/orchestrator_registry.py, training_coordinator.py
    """
    # How often to send heartbeats (seconds)
    INTERVAL: int = _env_int("RINGRIFT_HEARTBEAT_INTERVAL", 30)

    # Consider peer dead after this many seconds without heartbeat
    TIMEOUT: int = _env_int("RINGRIFT_HEARTBEAT_TIMEOUT", 90)

    # Cleanup stale entries this often (seconds)
    STALE_CLEANUP_INTERVAL: int = _env_int("RINGRIFT_STALE_CLEANUP_INTERVAL", 60)

    # Multiplier for timeout (dead after INTERVAL * TIMEOUT_MULTIPLIER)
    TIMEOUT_MULTIPLIER: int = 3


# =============================================================================
# Training Coordination Defaults
# =============================================================================

@dataclass(frozen=True)
class TrainingDefaults:
    """Default values for training coordination.

    Used by: app/coordination/training_coordinator.py
    """
    # Maximum concurrent training jobs for same config
    MAX_CONCURRENT_SAME_CONFIG: int = _env_int("RINGRIFT_MAX_TRAINING_SAME_CONFIG", 1)

    # Maximum total concurrent training jobs
    MAX_CONCURRENT_TOTAL: int = _env_int("RINGRIFT_MAX_TRAINING_TOTAL", 3)

    # Training job timeout (hours)
    TIMEOUT_HOURS: float = _env_float("RINGRIFT_TRAINING_TIMEOUT_HOURS", 24.0)

    # Minimum interval between training runs (seconds)
    MIN_INTERVAL: int = _env_int("RINGRIFT_TRAINING_MIN_INTERVAL", 1200)


# =============================================================================
# Task Scheduling Defaults
# =============================================================================

@dataclass(frozen=True)
class SchedulerDefaults:
    """Default values for task scheduling.

    Used by: app/coordination/job_scheduler.py, task_coordinator.py
    """
    # Minimum memory (GB) required for task assignment
    MIN_MEMORY_GB: int = _env_int("RINGRIFT_MIN_MEMORY_GB", 64)

    # Maximum queue size for pending tasks
    MAX_QUEUE_SIZE: int = _env_int("RINGRIFT_MAX_QUEUE_SIZE", 1000)

    # Maximum selfplay tasks cluster-wide
    MAX_SELFPLAY_CLUSTER: int = _env_int("RINGRIFT_MAX_SELFPLAY_CLUSTER", 500)

    # Health check cache TTL (seconds)
    HEALTH_CACHE_TTL: int = _env_int("RINGRIFT_HEALTH_CACHE_TTL", 30)


# =============================================================================
# Ephemeral Data Guard Defaults
# =============================================================================

@dataclass(frozen=True)
class EphemeralDefaults:
    """Default values for ephemeral data protection.

    Used by: app/coordination/ephemeral_data_guard.py
    """
    # Checkpoint interval (seconds)
    CHECKPOINT_INTERVAL: int = _env_int("RINGRIFT_CHECKPOINT_INTERVAL", 300)

    # Host considered dead after this timeout (seconds)
    HEARTBEAT_TIMEOUT: int = _env_int("RINGRIFT_EPHEMERAL_HEARTBEAT_TIMEOUT", 120)


# =============================================================================
# Circuit Breaker Defaults
# =============================================================================

@dataclass(frozen=True)
class CircuitBreakerDefaults:
    """Default values for circuit breaker pattern.

    Used by: app/distributed/circuit_breaker.py
    """
    # Default failure threshold before circuit opens
    FAILURE_THRESHOLD: int = _env_int("RINGRIFT_CB_FAILURE_THRESHOLD", 5)

    # Default recovery timeout (seconds)
    RECOVERY_TIMEOUT: float = _env_float("RINGRIFT_CB_RECOVERY_TIMEOUT", 60.0)

    # Maximum backoff timeout (seconds)
    MAX_BACKOFF: float = _env_float("RINGRIFT_CB_MAX_BACKOFF", 600.0)

    # Half-open state max calls for testing recovery
    HALF_OPEN_MAX_CALLS: int = _env_int("RINGRIFT_CB_HALF_OPEN_MAX_CALLS", 1)

    # Per-transport type configs
    SSH_FAILURE_THRESHOLD: int = 3
    SSH_RECOVERY_TIMEOUT: float = 60.0

    HTTP_FAILURE_THRESHOLD: int = 5
    HTTP_RECOVERY_TIMEOUT: float = 30.0

    P2P_FAILURE_THRESHOLD: int = 3
    P2P_RECOVERY_TIMEOUT: float = 45.0

    ARIA2_FAILURE_THRESHOLD: int = 2
    ARIA2_RECOVERY_TIMEOUT: float = 120.0

    RSYNC_FAILURE_THRESHOLD: int = 2
    RSYNC_RECOVERY_TIMEOUT: float = 90.0


# =============================================================================
# Health Check Defaults (December 2025)
# =============================================================================

@dataclass(frozen=True)
class HealthDefaults:
    """Default values for host health checking.

    Used by: app/coordination/host_health_policy.py
    """
    # SSH timeout for health checks (seconds)
    SSH_TIMEOUT: int = _env_int("RINGRIFT_HEALTH_SSH_TIMEOUT", 5)

    # Cache TTL for healthy results (seconds)
    HEALTHY_CACHE_TTL: int = _env_int("RINGRIFT_HEALTHY_CACHE_TTL", 60)

    # Cache TTL for unhealthy results (seconds)
    UNHEALTHY_CACHE_TTL: int = _env_int("RINGRIFT_UNHEALTHY_CACHE_TTL", 30)

    # Maximum concurrent health checks
    MAX_CONCURRENT_CHECKS: int = _env_int("RINGRIFT_MAX_HEALTH_CHECKS", 10)

    # Minimum healthy hosts required
    MIN_HEALTHY_HOSTS: int = _env_int("RINGRIFT_MIN_HEALTHY_HOSTS", 2)

    # Cluster health cache TTL (seconds)
    CLUSTER_HEALTH_CACHE_TTL: int = _env_int("RINGRIFT_CLUSTER_HEALTH_CACHE_TTL", 120)


# =============================================================================
# Utilization Target Defaults (December 2025)
# =============================================================================

@dataclass(frozen=True)
class UtilizationDefaults:
    """Default values for resource utilization targets.

    Used by: app/coordination/job_scheduler.py, resource_optimizer.py
    """
    # GPU utilization targets (%)
    GPU_TARGET_MIN: int = _env_int("RINGRIFT_GPU_TARGET_MIN", 60)
    GPU_TARGET_MAX: int = _env_int("RINGRIFT_GPU_TARGET_MAX", 80)

    # CPU utilization targets (%)
    CPU_TARGET_MIN: int = _env_int("RINGRIFT_CPU_TARGET_MIN", 60)
    CPU_TARGET_MAX: int = _env_int("RINGRIFT_CPU_TARGET_MAX", 80)

    # Memory thresholds
    MIN_MEMORY_GB: int = _env_int("RINGRIFT_MIN_MEMORY_GB", 64)

    # Utilization update interval (seconds)
    UPDATE_INTERVAL: int = _env_int("RINGRIFT_UTILIZATION_UPDATE_INTERVAL", 10)

    # Optimization interval (Dec 2025: 30s → 15s for faster response)
    OPTIMIZATION_INTERVAL: int = _env_int("RINGRIFT_OPTIMIZATION_INTERVAL", 15)


# =============================================================================
# Bandwidth Manager Defaults (December 2025)
# =============================================================================

@dataclass(frozen=True)
class BandwidthDefaults:
    """Default values for bandwidth management.

    Used by: app/coordination/bandwidth_manager.py
    """
    # Maximum concurrent transfers per host
    MAX_CONCURRENT_TRANSFERS: int = _env_int("RINGRIFT_MAX_CONCURRENT_TRANSFERS", 3)

    # Bandwidth measurement window (seconds)
    MEASUREMENT_WINDOW: int = _env_int("RINGRIFT_BANDWIDTH_MEASUREMENT_WINDOW", 300)

    # Default host bandwidth limits (MB/s)
    DEFAULT_UPLOAD_MBPS: int = _env_int("RINGRIFT_DEFAULT_UPLOAD_MBPS", 100)
    DEFAULT_DOWNLOAD_MBPS: int = _env_int("RINGRIFT_DEFAULT_DOWNLOAD_MBPS", 1000)


# =============================================================================
# Resource Limits Defaults (December 2025)
# =============================================================================

@dataclass(frozen=True)
class ResourceLimitsDefaults:
    """Default values for per-tier resource limits.

    Used by: app/coordination/resource_optimizer.py
    """
    # Maximum concurrent selfplay by GPU tier
    CONSUMER_MAX: int = _env_int("RINGRIFT_CONSUMER_MAX_SELFPLAY", 16)
    PROSUMER_MAX: int = _env_int("RINGRIFT_PROSUMER_MAX_SELFPLAY", 32)
    DATACENTER_MAX: int = _env_int("RINGRIFT_DATACENTER_MAX_SELFPLAY", 64)
    HIGH_CPU_MAX: int = _env_int("RINGRIFT_HIGH_CPU_MAX_SELFPLAY", 128)


# =============================================================================
# PID Controller Defaults (December 2025)
# =============================================================================

@dataclass(frozen=True)
class PIDDefaults:
    """Default values for PID controller tuning.

    Used by: app/coordination/resource_optimizer.py

    Dec 2025: Increased gains for faster responsiveness:
    - KP: 0.3 → 0.5 (faster error correction)
    - KI: 0.05 → 0.1 (faster steady-state convergence)
    - KD unchanged (0.1 provides good damping)
    """
    # Proportional gain (Dec 2025: 0.3 → 0.5 for faster response)
    KP: float = _env_float("RINGRIFT_PID_KP", 0.5)

    # Integral gain (Dec 2025: 0.05 → 0.1 for faster convergence)
    KI: float = _env_float("RINGRIFT_PID_KI", 0.1)

    # Derivative gain
    KD: float = _env_float("RINGRIFT_PID_KD", 0.1)


# =============================================================================
# Resource Monitoring Defaults (December 2025)
# =============================================================================

@dataclass(frozen=True)
class ResourceMonitoringDefaults:
    """Default values for resource monitoring and backpressure.

    Used by: app/coordination/resource_monitoring_coordinator.py
    """
    # Backpressure thresholds (%)
    BACKPRESSURE_GPU_THRESHOLD: float = _env_float("RINGRIFT_BACKPRESSURE_GPU_THRESHOLD", 90.0)
    BACKPRESSURE_MEMORY_THRESHOLD: float = _env_float("RINGRIFT_BACKPRESSURE_MEMORY_THRESHOLD", 85.0)
    BACKPRESSURE_DISK_THRESHOLD: float = _env_float("RINGRIFT_BACKPRESSURE_DISK_THRESHOLD", 90.0)

    # Resource update interval (seconds)
    UPDATE_INTERVAL: int = _env_int("RINGRIFT_RESOURCE_UPDATE_INTERVAL", 10)

    # Backpressure cooldown before release (seconds)
    BACKPRESSURE_COOLDOWN: int = _env_int("RINGRIFT_BACKPRESSURE_COOLDOWN", 30)


# =============================================================================
# Optimization Coordinator Defaults (December 2025)
# =============================================================================

@dataclass(frozen=True)
class OptimizationDefaults:
    """Default values for optimization coordination (CMA-ES, NAS).

    Used by: app/coordination/optimization_coordinator.py
    """
    # Plateau detection window (epochs)
    PLATEAU_WINDOW: int = _env_int("RINGRIFT_OPTIMIZATION_PLATEAU_WINDOW", 10)

    # Plateau detection threshold
    PLATEAU_THRESHOLD: float = _env_float("RINGRIFT_OPTIMIZATION_PLATEAU_THRESHOLD", 0.001)

    # Minimum epochs between optimization runs
    MIN_EPOCHS_BETWEEN: int = _env_int("RINGRIFT_OPTIMIZATION_MIN_EPOCHS", 20)

    # Cooldown after optimization completes (seconds)
    COOLDOWN_SECONDS: float = _env_float("RINGRIFT_OPTIMIZATION_COOLDOWN", 300.0)

    # Maximum optimization history to retain
    MAX_HISTORY: int = _env_int("RINGRIFT_OPTIMIZATION_MAX_HISTORY", 100)


# =============================================================================
# Task Lifecycle Defaults (December 2025)
# =============================================================================

@dataclass(frozen=True)
class TaskLifecycleDefaults:
    """Default values for task lifecycle coordination.

    Used by: app/coordination/task_lifecycle_coordinator.py
    """
    # Task heartbeat timeout - consider orphaned after (seconds)
    HEARTBEAT_TIMEOUT: float = _env_float("RINGRIFT_TASK_HEARTBEAT_TIMEOUT", 60.0)

    # Grace period for new tasks before orphan check (seconds)
    ORPHAN_GRACE_PERIOD: float = _env_float("RINGRIFT_TASK_ORPHAN_GRACE", 30.0)

    # Maximum tasks to track in history
    MAX_HISTORY: int = _env_int("RINGRIFT_TASK_MAX_HISTORY", 1000)

    # Task cleanup interval (seconds)
    CLEANUP_INTERVAL: int = _env_int("RINGRIFT_TASK_CLEANUP_INTERVAL", 60)


# =============================================================================
# Daemon Loop Defaults (December 2025)
# =============================================================================

@dataclass(frozen=True)
class DaemonLoopDefaults:
    """Default values for daemon main loop operation.

    Used by: app/coordination/base_daemon.py, daemon_manager.py
    """
    # Default check interval for daemon loops (seconds)
    CHECK_INTERVAL: int = _env_int("RINGRIFT_DAEMON_CHECK_INTERVAL", 300)

    # Error backoff base delay (seconds)
    ERROR_BACKOFF_BASE: float = _env_float("RINGRIFT_DAEMON_ERROR_BACKOFF_BASE", 5.0)

    # Maximum error backoff (seconds)
    ERROR_BACKOFF_MAX: float = _env_float("RINGRIFT_DAEMON_ERROR_BACKOFF_MAX", 300.0)

    # Maximum consecutive errors before alerting
    MAX_CONSECUTIVE_ERRORS: int = _env_int("RINGRIFT_DAEMON_MAX_CONSECUTIVE_ERRORS", 5)

    # Shutdown grace period (seconds)
    SHUTDOWN_GRACE_PERIOD: float = _env_float("RINGRIFT_DAEMON_SHUTDOWN_GRACE", 10.0)

    # Health check timeout (seconds)
    HEALTH_CHECK_TIMEOUT: float = _env_float("RINGRIFT_DAEMON_HEALTH_TIMEOUT", 5.0)

    # Error rate threshold for unhealthy status (fraction)
    ERROR_RATE_THRESHOLD: float = _env_float("RINGRIFT_DAEMON_ERROR_RATE_THRESHOLD", 0.5)

    # Minimum cycles before error rate check
    MIN_CYCLES_FOR_ERROR_CHECK: int = _env_int("RINGRIFT_DAEMON_MIN_CYCLES_ERROR_CHECK", 10)


# =============================================================================
# Network Retry Defaults (December 2025)
# =============================================================================

@dataclass(frozen=True)
class NetworkRetryDefaults:
    """Default values for network operation retries.

    Extends RetryDefaults with network-specific settings.

    Used by: app/coordination/cluster_transport.py, sync operations
    """
    # Connection retry settings
    CONNECT_MAX_RETRIES: int = _env_int("RINGRIFT_CONNECT_MAX_RETRIES", 3)
    CONNECT_BASE_DELAY: float = _env_float("RINGRIFT_CONNECT_BASE_DELAY", 1.0)
    CONNECT_MAX_DELAY: float = _env_float("RINGRIFT_CONNECT_MAX_DELAY", 30.0)

    # Transfer retry settings
    TRANSFER_MAX_RETRIES: int = _env_int("RINGRIFT_TRANSFER_MAX_RETRIES", 3)
    TRANSFER_BASE_DELAY: float = _env_float("RINGRIFT_TRANSFER_BASE_DELAY", 5.0)
    TRANSFER_MAX_DELAY: float = _env_float("RINGRIFT_TRANSFER_MAX_DELAY", 60.0)

    # SSH-specific retry settings
    SSH_MAX_RETRIES: int = _env_int("RINGRIFT_SSH_MAX_RETRIES", 2)
    SSH_BASE_DELAY: float = _env_float("RINGRIFT_SSH_BASE_DELAY", 2.0)
    SSH_MAX_DELAY: float = _env_float("RINGRIFT_SSH_MAX_DELAY", 30.0)

    # P2P/HTTP retry settings
    P2P_MAX_RETRIES: int = _env_int("RINGRIFT_P2P_MAX_RETRIES", 3)
    P2P_BASE_DELAY: float = _env_float("RINGRIFT_P2P_BASE_DELAY", 1.0)
    P2P_MAX_DELAY: float = _env_float("RINGRIFT_P2P_MAX_DELAY", 15.0)

    # DNS resolution retry
    DNS_MAX_RETRIES: int = _env_int("RINGRIFT_DNS_MAX_RETRIES", 2)
    DNS_TIMEOUT: float = _env_float("RINGRIFT_DNS_TIMEOUT", 5.0)


# =============================================================================
# Monitoring Thresholds (December 2025)
# =============================================================================

@dataclass(frozen=True)
class MonitoringDefaults:
    """Default values for system and cluster monitoring.

    Used by: app/coordination/system_health_monitor.py, cluster_monitor.py
    """
    # Disk space warning thresholds (%)
    DISK_WARNING_THRESHOLD: float = _env_float("RINGRIFT_DISK_WARNING_THRESHOLD", 70.0)
    DISK_CRITICAL_THRESHOLD: float = _env_float("RINGRIFT_DISK_CRITICAL_THRESHOLD", 90.0)

    # Memory usage thresholds (%)
    MEMORY_WARNING_THRESHOLD: float = _env_float("RINGRIFT_MEMORY_WARNING_THRESHOLD", 80.0)
    MEMORY_CRITICAL_THRESHOLD: float = _env_float("RINGRIFT_MEMORY_CRITICAL_THRESHOLD", 95.0)

    # GPU memory thresholds (%)
    GPU_MEMORY_WARNING: float = _env_float("RINGRIFT_GPU_MEMORY_WARNING", 90.0)
    GPU_MEMORY_CRITICAL: float = _env_float("RINGRIFT_GPU_MEMORY_CRITICAL", 98.0)

    # Node offline detection (seconds)
    NODE_OFFLINE_THRESHOLD: int = _env_int("RINGRIFT_NODE_OFFLINE_THRESHOLD", 300)

    # Cluster health check interval (seconds)
    CLUSTER_CHECK_INTERVAL: int = _env_int("RINGRIFT_CLUSTER_CHECK_INTERVAL", 60)

    # Minimum healthy node fraction for cluster health
    MIN_HEALTHY_FRACTION: float = _env_float("RINGRIFT_MIN_HEALTHY_FRACTION", 0.6)

    # Alert cooldown (seconds) - avoid spam
    ALERT_COOLDOWN: int = _env_int("RINGRIFT_ALERT_COOLDOWN", 300)


# =============================================================================
# Metrics Analysis Defaults (December 2025)
# =============================================================================

@dataclass(frozen=True)
class MetricsAnalysisDefaults:
    """Default values for metrics analysis and trend detection.

    Used by: app/coordination/metrics_analysis_orchestrator.py
    """
    # Sliding window size for metric tracking
    WINDOW_SIZE: int = _env_int("RINGRIFT_METRICS_WINDOW_SIZE", 100)

    # Plateau detection threshold
    PLATEAU_THRESHOLD: float = _env_float("RINGRIFT_METRICS_PLATEAU_THRESHOLD", 0.001)

    # Plateau detection window (data points)
    PLATEAU_WINDOW: int = _env_int("RINGRIFT_METRICS_PLATEAU_WINDOW", 10)

    # Regression severity threshold (fraction)
    REGRESSION_THRESHOLD: float = _env_float("RINGRIFT_METRICS_REGRESSION_THRESHOLD", 0.05)

    # Anomaly detection threshold (standard deviations)
    ANOMALY_THRESHOLD: float = _env_float("RINGRIFT_METRICS_ANOMALY_THRESHOLD", 3.0)


# =============================================================================
# Cache Coordination Defaults (December 2025)
# =============================================================================

@dataclass(frozen=True)
class CacheDefaults:
    """Default values for cache coordination.

    Used by: app/coordination/cache_coordination_orchestrator.py
    """
    # Default cache TTL (seconds)
    DEFAULT_TTL: int = _env_int("RINGRIFT_CACHE_DEFAULT_TTL", 3600)

    # Maximum cache entries per node
    MAX_ENTRIES_PER_NODE: int = _env_int("RINGRIFT_CACHE_MAX_ENTRIES_PER_NODE", 100)

    # Cache cleanup interval (seconds)
    CLEANUP_INTERVAL: int = _env_int("RINGRIFT_CACHE_CLEANUP_INTERVAL", 300)

    # Stale cache threshold (seconds since last access)
    STALE_THRESHOLD: int = _env_int("RINGRIFT_CACHE_STALE_THRESHOLD", 7200)


# =============================================================================
# Queue Monitoring Defaults (December 2025)
# =============================================================================

@dataclass(frozen=True)
class QueueDefaults:
    """Default values for queue depth monitoring and backpressure.

    Used by: app/coordination/queue_monitor.py
    """
    # Training data queue thresholds
    TRAINING_DATA_SOFT_LIMIT: int = _env_int("RINGRIFT_QUEUE_TRAINING_SOFT", 100000)
    TRAINING_DATA_HARD_LIMIT: int = _env_int("RINGRIFT_QUEUE_TRAINING_HARD", 500000)
    TRAINING_DATA_TARGET: int = _env_int("RINGRIFT_QUEUE_TRAINING_TARGET", 50000)

    # Pending games queue thresholds
    PENDING_GAMES_SOFT_LIMIT: int = _env_int("RINGRIFT_QUEUE_GAMES_SOFT", 1000)
    PENDING_GAMES_HARD_LIMIT: int = _env_int("RINGRIFT_QUEUE_GAMES_HARD", 5000)
    PENDING_GAMES_TARGET: int = _env_int("RINGRIFT_QUEUE_GAMES_TARGET", 500)

    # Evaluation queue thresholds
    EVALUATION_SOFT_LIMIT: int = _env_int("RINGRIFT_QUEUE_EVAL_SOFT", 50)
    EVALUATION_HARD_LIMIT: int = _env_int("RINGRIFT_QUEUE_EVAL_HARD", 200)
    EVALUATION_TARGET: int = _env_int("RINGRIFT_QUEUE_EVAL_TARGET", 20)

    # Sync queue thresholds
    SYNC_SOFT_LIMIT: int = _env_int("RINGRIFT_QUEUE_SYNC_SOFT", 100)
    SYNC_HARD_LIMIT: int = _env_int("RINGRIFT_QUEUE_SYNC_HARD", 500)
    SYNC_TARGET: int = _env_int("RINGRIFT_QUEUE_SYNC_TARGET", 50)


# =============================================================================
# Auto-Scaling Defaults (December 2025)
# =============================================================================

@dataclass(frozen=True)
class ScalingDefaults:
    """Default values for auto-scaling operations.

    Used by: app/coordination/auto_scaler.py
    """
    # Queue depth thresholds for scaling
    SCALE_UP_QUEUE_DEPTH: int = _env_int("RINGRIFT_SCALE_UP_QUEUE_DEPTH", 100)
    SCALE_DOWN_QUEUE_DEPTH: int = _env_int("RINGRIFT_SCALE_DOWN_QUEUE_DEPTH", 10)

    # Time thresholds (minutes)
    SCALE_DOWN_IDLE_MINUTES: int = _env_int("RINGRIFT_SCALE_DOWN_IDLE_MINUTES", 30)
    SCALE_UP_COOLDOWN_MINUTES: int = _env_int("RINGRIFT_SCALE_UP_COOLDOWN_MINUTES", 5)
    SCALE_DOWN_COOLDOWN_MINUTES: int = _env_int("RINGRIFT_SCALE_DOWN_COOLDOWN_MINUTES", 10)

    # Instance limits
    MAX_INSTANCES: int = _env_int("RINGRIFT_MAX_INSTANCES", 10)
    MIN_INSTANCES: int = _env_int("RINGRIFT_MIN_INSTANCES", 1)

    # GPU utilization targets for scaling (%)
    GPU_SCALE_UP_THRESHOLD: int = _env_int("RINGRIFT_GPU_SCALE_UP_THRESHOLD", 85)
    GPU_SCALE_DOWN_THRESHOLD: int = _env_int("RINGRIFT_GPU_SCALE_DOWN_THRESHOLD", 30)

    # Cost optimization
    MAX_HOURLY_COST: float = _env_float("RINGRIFT_MAX_HOURLY_COST", 10.0)


# =============================================================================
# Duration Scheduling Defaults (December 2025)
# =============================================================================

@dataclass(frozen=True)
class DurationDefaults:
    """Default values for duration-aware task scheduling.

    Used by: app/coordination/duration_scheduler.py
    """
    # Default task durations (seconds)
    SELFPLAY_DURATION: int = _env_int("RINGRIFT_DURATION_SELFPLAY", 3600)  # 1 hour
    GPU_SELFPLAY_DURATION: int = _env_int("RINGRIFT_DURATION_GPU_SELFPLAY", 7200)  # 2 hours
    TRAINING_DURATION: int = _env_int("RINGRIFT_DURATION_TRAINING", 14400)  # 4 hours
    CMAES_DURATION: int = _env_int("RINGRIFT_DURATION_CMAES", 28800)  # 8 hours
    TOURNAMENT_DURATION: int = _env_int("RINGRIFT_DURATION_TOURNAMENT", 1800)  # 30 min
    EVALUATION_DURATION: int = _env_int("RINGRIFT_DURATION_EVALUATION", 3600)  # 1 hour
    SYNC_DURATION: int = _env_int("RINGRIFT_DURATION_SYNC", 600)  # 10 min
    EXPORT_DURATION: int = _env_int("RINGRIFT_DURATION_EXPORT", 300)  # 5 min
    PIPELINE_DURATION: int = _env_int("RINGRIFT_DURATION_PIPELINE", 21600)  # 6 hours
    IMPROVEMENT_LOOP_DURATION: int = _env_int("RINGRIFT_DURATION_IMPROVEMENT", 43200)  # 12 hours

    # Peak hours (UTC) - avoid intensive tasks
    PEAK_HOURS_START: int = _env_int("RINGRIFT_PEAK_HOURS_START", 14)  # 2 PM UTC
    PEAK_HOURS_END: int = _env_int("RINGRIFT_PEAK_HOURS_END", 22)  # 10 PM UTC


# =============================================================================
# Sync Coordinator Extended Defaults (December 2025)
# =============================================================================

@dataclass(frozen=True)
class SyncCoordinatorDefaults:
    """Default values for sync coordinator operations.

    Used by: app/coordination/sync_coordinator.py
    """
    # Critical stale threshold for data freshness (seconds)
    CRITICAL_STALE_THRESHOLD: int = _env_int("RINGRIFT_SYNC_CRITICAL_STALE", 3600)

    # Freshness check interval (seconds)
    FRESHNESS_CHECK_INTERVAL: int = _env_int("RINGRIFT_SYNC_FRESHNESS_INTERVAL", 60)

    # Full sync interval (seconds)
    FULL_SYNC_INTERVAL: int = _env_int("RINGRIFT_SYNC_FULL_INTERVAL", 3600)


def get_circuit_breaker_configs() -> dict:
    """Get circuit breaker configs per transport type."""
    return {
        "ssh": {
            "failure_threshold": CircuitBreakerDefaults.SSH_FAILURE_THRESHOLD,
            "recovery_timeout": CircuitBreakerDefaults.SSH_RECOVERY_TIMEOUT,
        },
        "http": {
            "failure_threshold": CircuitBreakerDefaults.HTTP_FAILURE_THRESHOLD,
            "recovery_timeout": CircuitBreakerDefaults.HTTP_RECOVERY_TIMEOUT,
        },
        "p2p": {
            "failure_threshold": CircuitBreakerDefaults.P2P_FAILURE_THRESHOLD,
            "recovery_timeout": CircuitBreakerDefaults.P2P_RECOVERY_TIMEOUT,
        },
        "aria2": {
            "failure_threshold": CircuitBreakerDefaults.ARIA2_FAILURE_THRESHOLD,
            "recovery_timeout": CircuitBreakerDefaults.ARIA2_RECOVERY_TIMEOUT,
        },
        "rsync": {
            "failure_threshold": CircuitBreakerDefaults.RSYNC_FAILURE_THRESHOLD,
            "recovery_timeout": CircuitBreakerDefaults.RSYNC_RECOVERY_TIMEOUT,
        },
    }


# =============================================================================
# SQLite Database Defaults (December 2025)
# =============================================================================

@dataclass(frozen=True)
class SQLiteDefaults:
    """Default values for SQLite database connections.

    Consolidates 40+ scattered timeout values across the codebase.

    Used by: app/db/game_replay.py, app/training/model_registry.py,
             app/coordination/*.py, scripts/p2p_orchestrator.py

    Timeout tiers:
    - QUICK (2s): Health checks, existence tests, short reads
    - READ (5s): Standard read operations, game discovery
    - STANDARD (10s): Normal read/write operations
    - WRITE (30s): Registry updates, Elo calculations, multi-step writes
    - HEAVY (60s): Database consolidation, migration, bulk inserts
    """
    # Quick timeout for health checks and existence tests (seconds)
    QUICK_TIMEOUT: float = _env_float("RINGRIFT_SQLITE_QUICK_TIMEOUT", 2.0)

    # Read-only operations timeout (seconds)
    READ_TIMEOUT: float = _env_float("RINGRIFT_SQLITE_READ_TIMEOUT", 5.0)

    # Standard operations timeout (seconds)
    STANDARD_TIMEOUT: float = _env_float("RINGRIFT_SQLITE_STANDARD_TIMEOUT", 10.0)

    # Write operations timeout (seconds) - registry, Elo, etc.
    WRITE_TIMEOUT: float = _env_float("RINGRIFT_SQLITE_WRITE_TIMEOUT", 30.0)

    # Heavy operations timeout (seconds) - consolidation, migration
    HEAVY_TIMEOUT: float = _env_float("RINGRIFT_SQLITE_HEAVY_TIMEOUT", 60.0)

    # Very long operations (database merge, bulk import)
    MERGE_TIMEOUT: float = _env_float("RINGRIFT_SQLITE_MERGE_TIMEOUT", 120.0)

    # WAL mode settings
    WAL_CHECKPOINT_THRESHOLD: int = _env_int("RINGRIFT_SQLITE_WAL_CHECKPOINT", 1000)

    # Busy timeout for lock contention (milliseconds)
    BUSY_TIMEOUT_MS: int = _env_int("RINGRIFT_SQLITE_BUSY_TIMEOUT_MS", 5000)


def get_sqlite_timeout(operation: str) -> float:
    """Get SQLite timeout for a specific operation type.

    Args:
        operation: Operation type ("quick", "read", "standard", "write", "heavy", "merge")

    Returns:
        Timeout in seconds

    Example:
        timeout = get_sqlite_timeout("read")  # Returns 5.0
        conn = sqlite3.connect(db_path, timeout=timeout)
    """
    timeouts = {
        "quick": SQLiteDefaults.QUICK_TIMEOUT,
        "read": SQLiteDefaults.READ_TIMEOUT,
        "standard": SQLiteDefaults.STANDARD_TIMEOUT,
        "write": SQLiteDefaults.WRITE_TIMEOUT,
        "heavy": SQLiteDefaults.HEAVY_TIMEOUT,
        "merge": SQLiteDefaults.MERGE_TIMEOUT,
        # Aliases for common use cases
        "health": SQLiteDefaults.QUICK_TIMEOUT,
        "discovery": SQLiteDefaults.READ_TIMEOUT,
        "registry": SQLiteDefaults.WRITE_TIMEOUT,
        "elo": SQLiteDefaults.WRITE_TIMEOUT,
        "training": SQLiteDefaults.WRITE_TIMEOUT,
        "consolidate": SQLiteDefaults.HEAVY_TIMEOUT,
        "migrate": SQLiteDefaults.MERGE_TIMEOUT,
    }
    return timeouts.get(operation, SQLiteDefaults.STANDARD_TIMEOUT)


# =============================================================================
# Operation Timeouts (December 2025)
# =============================================================================

@dataclass(frozen=True)
class OperationTimeouts:
    """Centralized timeouts for various operations.

    Use these instead of hardcoding timeout values in code.

    Used by: multi_provider_orchestrator.py, data_sync.py, fault_tolerance.py
    """
    # Quick health check timeout (seconds)
    HEALTH_CHECK: int = _env_int("RINGRIFT_HEALTH_CHECK_TIMEOUT", 5)

    # URL fetch for quick operations (seconds)
    URL_FETCH_QUICK: int = _env_int("RINGRIFT_URL_FETCH_QUICK_TIMEOUT", 5)

    # URL fetch for data operations (seconds)
    URL_FETCH: int = _env_int("RINGRIFT_URL_FETCH_TIMEOUT", 10)

    # Rsync transfer timeout (seconds)
    RSYNC: int = _env_int("RINGRIFT_RSYNC_TIMEOUT", 30)

    # Async subprocess timeout (seconds)
    ASYNC_SUBPROCESS: int = _env_int("RINGRIFT_ASYNC_SUBPROCESS_TIMEOUT", 180)

    # Thread/process join timeout (seconds)
    THREAD_JOIN: int = _env_int("RINGRIFT_THREAD_JOIN_TIMEOUT", 5)

    # Future result timeout (seconds)
    FUTURE_RESULT: int = _env_int("RINGRIFT_FUTURE_RESULT_TIMEOUT", 300)

    # Checkpoint operation timeout (seconds)
    CHECKPOINT: int = _env_int("RINGRIFT_CHECKPOINT_TIMEOUT", 120)

    # Long-running job timeout - training (seconds) - 4 hours
    TRAINING_JOB: int = _env_int("RINGRIFT_TRAINING_JOB_TIMEOUT", 14400)

    # Resource wait timeout (seconds)
    RESOURCE_WAIT: int = _env_int("RINGRIFT_RESOURCE_WAIT_TIMEOUT", 300)

    # Batch operation timeout (seconds) - 30 minutes
    BATCH_OPERATION: int = _env_int("RINGRIFT_BATCH_OPERATION_TIMEOUT", 1800)

    # Per-file operation timeout (seconds)
    PER_FILE: int = _env_int("RINGRIFT_PER_FILE_TIMEOUT", 120)


@dataclass(frozen=True)
class RetryDefaults:
    """Default retry configuration values.

    Use these for consistent retry behavior across the codebase.

    Used by: error_handler.py, fault_tolerance.py, data_sync.py
    """
    # Default retry count for transient failures
    MAX_RETRIES: int = _env_int("RINGRIFT_DEFAULT_MAX_RETRIES", 3)

    # Base delay between retries (seconds)
    BASE_DELAY: float = _env_float("RINGRIFT_RETRY_BASE_DELAY", 1.0)

    # Maximum delay between retries (seconds)
    MAX_DELAY: float = _env_float("RINGRIFT_RETRY_MAX_DELAY", 60.0)

    # Exponential backoff multiplier
    BACKOFF_MULTIPLIER: float = _env_float("RINGRIFT_RETRY_BACKOFF_MULTIPLIER", 2.0)

    # Jitter factor (0-1) to add randomness to delays
    JITTER_FACTOR: float = _env_float("RINGRIFT_RETRY_JITTER_FACTOR", 0.1)

    # Aggressive retry for critical operations
    AGGRESSIVE_MAX_RETRIES: int = 5
    AGGRESSIVE_BASE_DELAY: float = 0.5

    # Fast retry for quick operations
    FAST_MAX_RETRIES: int = 2
    FAST_BASE_DELAY: float = 0.5
    FAST_MAX_DELAY: float = 5.0

    # Sync-specific retry
    SYNC_MAX_RETRIES: int = 3
    SYNC_BASE_DELAY: float = 2.0
    SYNC_MAX_DELAY: float = 30.0


def get_timeout(operation: str) -> int:
    """Get timeout for a specific operation type.

    Args:
        operation: Operation type ("http", "ssh", "health", "rsync", etc.)

    Returns:
        Timeout in seconds

    Example:
        timeout = get_timeout("health")  # Returns 5
        timeout = get_timeout("ssh")     # Returns 30
    """
    timeouts = {
        "http": TransportDefaults.HTTP_TIMEOUT,
        "ssh": TransportDefaults.SSH_TIMEOUT,
        "connect": TransportDefaults.CONNECT_TIMEOUT,
        "operation": TransportDefaults.OPERATION_TIMEOUT,
        "health": OperationTimeouts.HEALTH_CHECK,
        "rsync": OperationTimeouts.RSYNC,
        "subprocess": OperationTimeouts.ASYNC_SUBPROCESS,
        "thread_join": OperationTimeouts.THREAD_JOIN,
        "future": OperationTimeouts.FUTURE_RESULT,
        "checkpoint": OperationTimeouts.CHECKPOINT,
        "training": OperationTimeouts.TRAINING_JOB,
        "resource": OperationTimeouts.RESOURCE_WAIT,
        "batch": OperationTimeouts.BATCH_OPERATION,
        "per_file": OperationTimeouts.PER_FILE,
        "url_quick": OperationTimeouts.URL_FETCH_QUICK,
        "url": OperationTimeouts.URL_FETCH,
    }
    return timeouts.get(operation, TransportDefaults.HTTP_TIMEOUT)


# =============================================================================
# Convenience Functions
# =============================================================================

def get_all_defaults() -> dict:
    """Get all defaults as a dictionary for inspection/logging."""
    return {
        "lock": {
            "lock_timeout": LockDefaults.LOCK_TIMEOUT,
            "acquire_timeout": LockDefaults.ACQUIRE_TIMEOUT,
            "retry_interval": LockDefaults.RETRY_INTERVAL,
            "training_lock_timeout": LockDefaults.TRAINING_LOCK_TIMEOUT,
        },
        "transport": {
            "connect_timeout": TransportDefaults.CONNECT_TIMEOUT,
            "operation_timeout": TransportDefaults.OPERATION_TIMEOUT,
            "http_timeout": TransportDefaults.HTTP_TIMEOUT,
            "circuit_breaker_recovery": TransportDefaults.CIRCUIT_BREAKER_RECOVERY,
            "ssh_timeout": TransportDefaults.SSH_TIMEOUT,
            "max_retries": TransportDefaults.MAX_RETRIES,
        },
        "sync": {
            "lock_timeout": SyncDefaults.LOCK_TIMEOUT,
            "max_concurrent_per_host": SyncDefaults.MAX_CONCURRENT_PER_HOST,
            "max_concurrent_cluster": SyncDefaults.MAX_CONCURRENT_CLUSTER,
            "data_sync_interval": SyncDefaults.DATA_SYNC_INTERVAL,
            "model_sync_interval": SyncDefaults.MODEL_SYNC_INTERVAL,
            "elo_sync_interval": SyncDefaults.ELO_SYNC_INTERVAL,
            "registry_sync_interval": SyncDefaults.REGISTRY_SYNC_INTERVAL,
        },
        "heartbeat": {
            "interval": HeartbeatDefaults.INTERVAL,
            "timeout": HeartbeatDefaults.TIMEOUT,
            "stale_cleanup_interval": HeartbeatDefaults.STALE_CLEANUP_INTERVAL,
        },
        "training": {
            "max_concurrent_same_config": TrainingDefaults.MAX_CONCURRENT_SAME_CONFIG,
            "max_concurrent_total": TrainingDefaults.MAX_CONCURRENT_TOTAL,
            "timeout_hours": TrainingDefaults.TIMEOUT_HOURS,
            "min_interval": TrainingDefaults.MIN_INTERVAL,
        },
        "scheduler": {
            "min_memory_gb": SchedulerDefaults.MIN_MEMORY_GB,
            "max_queue_size": SchedulerDefaults.MAX_QUEUE_SIZE,
            "max_selfplay_cluster": SchedulerDefaults.MAX_SELFPLAY_CLUSTER,
            "health_cache_ttl": SchedulerDefaults.HEALTH_CACHE_TTL,
        },
        "ephemeral": {
            "checkpoint_interval": EphemeralDefaults.CHECKPOINT_INTERVAL,
            "heartbeat_timeout": EphemeralDefaults.HEARTBEAT_TIMEOUT,
        },
        "circuit_breaker": {
            "failure_threshold": CircuitBreakerDefaults.FAILURE_THRESHOLD,
            "recovery_timeout": CircuitBreakerDefaults.RECOVERY_TIMEOUT,
            "max_backoff": CircuitBreakerDefaults.MAX_BACKOFF,
            "half_open_max_calls": CircuitBreakerDefaults.HALF_OPEN_MAX_CALLS,
        },
        # December 2025 additions
        "health": {
            "ssh_timeout": HealthDefaults.SSH_TIMEOUT,
            "healthy_cache_ttl": HealthDefaults.HEALTHY_CACHE_TTL,
            "unhealthy_cache_ttl": HealthDefaults.UNHEALTHY_CACHE_TTL,
            "max_concurrent_checks": HealthDefaults.MAX_CONCURRENT_CHECKS,
            "min_healthy_hosts": HealthDefaults.MIN_HEALTHY_HOSTS,
        },
        "utilization": {
            "gpu_target_min": UtilizationDefaults.GPU_TARGET_MIN,
            "gpu_target_max": UtilizationDefaults.GPU_TARGET_MAX,
            "cpu_target_min": UtilizationDefaults.CPU_TARGET_MIN,
            "cpu_target_max": UtilizationDefaults.CPU_TARGET_MAX,
            "min_memory_gb": UtilizationDefaults.MIN_MEMORY_GB,
        },
        "bandwidth": {
            "max_concurrent_transfers": BandwidthDefaults.MAX_CONCURRENT_TRANSFERS,
            "measurement_window": BandwidthDefaults.MEASUREMENT_WINDOW,
        },
        "resource_limits": {
            "consumer_max": ResourceLimitsDefaults.CONSUMER_MAX,
            "prosumer_max": ResourceLimitsDefaults.PROSUMER_MAX,
            "datacenter_max": ResourceLimitsDefaults.DATACENTER_MAX,
        },
        "pid": {
            "kp": PIDDefaults.KP,
            "ki": PIDDefaults.KI,
            "kd": PIDDefaults.KD,
        },
        "resource_monitoring": {
            "backpressure_gpu_threshold": ResourceMonitoringDefaults.BACKPRESSURE_GPU_THRESHOLD,
            "backpressure_memory_threshold": ResourceMonitoringDefaults.BACKPRESSURE_MEMORY_THRESHOLD,
            "backpressure_disk_threshold": ResourceMonitoringDefaults.BACKPRESSURE_DISK_THRESHOLD,
            "update_interval": ResourceMonitoringDefaults.UPDATE_INTERVAL,
        },
        "optimization": {
            "plateau_window": OptimizationDefaults.PLATEAU_WINDOW,
            "plateau_threshold": OptimizationDefaults.PLATEAU_THRESHOLD,
            "cooldown_seconds": OptimizationDefaults.COOLDOWN_SECONDS,
        },
        "task_lifecycle": {
            "heartbeat_timeout": TaskLifecycleDefaults.HEARTBEAT_TIMEOUT,
            "orphan_grace_period": TaskLifecycleDefaults.ORPHAN_GRACE_PERIOD,
        },
        "metrics_analysis": {
            "window_size": MetricsAnalysisDefaults.WINDOW_SIZE,
            "plateau_threshold": MetricsAnalysisDefaults.PLATEAU_THRESHOLD,
            "regression_threshold": MetricsAnalysisDefaults.REGRESSION_THRESHOLD,
        },
        "cache": {
            "default_ttl": CacheDefaults.DEFAULT_TTL,
            "max_entries_per_node": CacheDefaults.MAX_ENTRIES_PER_NODE,
            "cleanup_interval": CacheDefaults.CLEANUP_INTERVAL,
        },
        "sync_coordinator": {
            "critical_stale_threshold": SyncCoordinatorDefaults.CRITICAL_STALE_THRESHOLD,
            "freshness_check_interval": SyncCoordinatorDefaults.FRESHNESS_CHECK_INTERVAL,
        },
        # December 27, 2025 additions
        "daemon_loop": {
            "check_interval": DaemonLoopDefaults.CHECK_INTERVAL,
            "error_backoff_base": DaemonLoopDefaults.ERROR_BACKOFF_BASE,
            "error_backoff_max": DaemonLoopDefaults.ERROR_BACKOFF_MAX,
            "max_consecutive_errors": DaemonLoopDefaults.MAX_CONSECUTIVE_ERRORS,
            "shutdown_grace_period": DaemonLoopDefaults.SHUTDOWN_GRACE_PERIOD,
            "error_rate_threshold": DaemonLoopDefaults.ERROR_RATE_THRESHOLD,
        },
        "network_retry": {
            "connect_max_retries": NetworkRetryDefaults.CONNECT_MAX_RETRIES,
            "connect_base_delay": NetworkRetryDefaults.CONNECT_BASE_DELAY,
            "transfer_max_retries": NetworkRetryDefaults.TRANSFER_MAX_RETRIES,
            "ssh_max_retries": NetworkRetryDefaults.SSH_MAX_RETRIES,
            "p2p_max_retries": NetworkRetryDefaults.P2P_MAX_RETRIES,
        },
        "monitoring": {
            "disk_warning_threshold": MonitoringDefaults.DISK_WARNING_THRESHOLD,
            "disk_critical_threshold": MonitoringDefaults.DISK_CRITICAL_THRESHOLD,
            "memory_warning_threshold": MonitoringDefaults.MEMORY_WARNING_THRESHOLD,
            "memory_critical_threshold": MonitoringDefaults.MEMORY_CRITICAL_THRESHOLD,
            "node_offline_threshold": MonitoringDefaults.NODE_OFFLINE_THRESHOLD,
            "cluster_check_interval": MonitoringDefaults.CLUSTER_CHECK_INTERVAL,
        },
        "queue": {
            "training_data_soft_limit": QueueDefaults.TRAINING_DATA_SOFT_LIMIT,
            "training_data_hard_limit": QueueDefaults.TRAINING_DATA_HARD_LIMIT,
            "pending_games_soft_limit": QueueDefaults.PENDING_GAMES_SOFT_LIMIT,
            "evaluation_soft_limit": QueueDefaults.EVALUATION_SOFT_LIMIT,
        },
        "scaling": {
            "scale_up_queue_depth": ScalingDefaults.SCALE_UP_QUEUE_DEPTH,
            "scale_down_idle_minutes": ScalingDefaults.SCALE_DOWN_IDLE_MINUTES,
            "max_instances": ScalingDefaults.MAX_INSTANCES,
            "gpu_scale_up_threshold": ScalingDefaults.GPU_SCALE_UP_THRESHOLD,
        },
        "duration": {
            "selfplay_duration": DurationDefaults.SELFPLAY_DURATION,
            "training_duration": DurationDefaults.TRAINING_DURATION,
            "peak_hours_start": DurationDefaults.PEAK_HOURS_START,
            "peak_hours_end": DurationDefaults.PEAK_HOURS_END,
        },
        # December 27, 2025: SQLite database timeouts
        "sqlite": {
            "quick_timeout": SQLiteDefaults.QUICK_TIMEOUT,
            "read_timeout": SQLiteDefaults.READ_TIMEOUT,
            "standard_timeout": SQLiteDefaults.STANDARD_TIMEOUT,
            "write_timeout": SQLiteDefaults.WRITE_TIMEOUT,
            "heavy_timeout": SQLiteDefaults.HEAVY_TIMEOUT,
            "merge_timeout": SQLiteDefaults.MERGE_TIMEOUT,
            "busy_timeout_ms": SQLiteDefaults.BUSY_TIMEOUT_MS,
        },
    }


__all__ = [
    "BandwidthDefaults",
    "CacheDefaults",
    "CircuitBreakerDefaults",
    # December 27, 2025 additions
    "DaemonLoopDefaults",
    "DurationDefaults",
    "EphemeralDefaults",
    # December 2025 additions
    "HealthDefaults",
    "HeartbeatDefaults",
    # Config classes
    "LockDefaults",
    "MetricsAnalysisDefaults",
    # December 27, 2025 additions
    "MonitoringDefaults",
    "NetworkRetryDefaults",
    "OperationTimeouts",
    "OptimizationDefaults",
    "PIDDefaults",
    # December 2025 new defaults
    "QueueDefaults",
    "ResourceLimitsDefaults",
    # December 2025 coordinator defaults
    "ResourceMonitoringDefaults",
    "RetryDefaults",
    "ScalingDefaults",
    "SchedulerDefaults",
    # SQLite database defaults (December 27, 2025)
    "SQLiteDefaults",
    "SyncCoordinatorDefaults",
    "SyncDefaults",
    "TaskLifecycleDefaults",
    "TrainingDefaults",
    "TransportDefaults",
    "UtilizationDefaults",
    # Utilities
    "get_all_defaults",
    "get_circuit_breaker_configs",
    "get_sqlite_timeout",
    "get_timeout",
]

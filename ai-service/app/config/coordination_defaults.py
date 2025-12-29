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
    # Dec 29, 2025: Reduced from 30 to 15 for faster failover (22% timeout rate → <5%)
    HTTP_TIMEOUT: int = _env_int("RINGRIFT_HTTP_TIMEOUT", 15)

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

    # Sync operation timeout (seconds) - for waiting for sync to complete
    SYNC_TIMEOUT: float = _env_float("RINGRIFT_SYNC_TIMEOUT", 300.0)

    # Stall detection timeout (seconds) - consider sync stalled after this
    STALL_DETECTION_TIMEOUT: float = _env_float("RINGRIFT_SYNC_STALL_TIMEOUT", 600.0)

    # Emergency sync cooldown (seconds) - minimum time between emergency syncs
    EMERGENCY_SYNC_COOLDOWN: float = _env_float("RINGRIFT_EMERGENCY_SYNC_COOLDOWN", 600.0)

    # Fast sync interval (seconds) - for training freshness or high-priority syncs
    FAST_SYNC_INTERVAL: float = _env_float("RINGRIFT_FAST_SYNC_INTERVAL", 30.0)


# =============================================================================
# Model Distribution Defaults (December 2025 - Phase 3)
# =============================================================================

@dataclass(frozen=True)
class DistributionDefaults:
    """Default values for model distribution verification.

    Used by: app/coordination/unified_distribution_daemon.py,
             app/coordination/promotion_controller.py,
             app/coordination/evaluation_daemon.py

    These thresholds ensure models are adequately distributed before
    promotion/evaluation to prevent "stuck" high-Elo models.
    """
    # Minimum nodes a model must be distributed to before promotion
    MIN_NODES_FOR_PROMOTION: int = _env_int("RINGRIFT_MIN_NODES_FOR_PROMOTION", 10)

    # Minimum nodes for fair evaluation (lower than promotion to allow testing)
    MIN_NODES_FOR_EVALUATION: int = _env_int("RINGRIFT_MIN_NODES_FOR_EVALUATION", 5)

    # Timeout waiting for distribution to complete (seconds)
    DISTRIBUTION_TIMEOUT_SECONDS: float = _env_float(
        "RINGRIFT_DISTRIBUTION_TIMEOUT", 300.0
    )

    # Retry interval when waiting for distribution (seconds)
    DISTRIBUTION_RETRY_INTERVAL: float = _env_float(
        "RINGRIFT_DISTRIBUTION_RETRY_INTERVAL", 15.0
    )

    # Minimum availability score (0-1) before allowing promotion
    MIN_AVAILABILITY_SCORE: float = _env_float(
        "RINGRIFT_MIN_AVAILABILITY_SCORE", 0.3
    )

    # Whether to block promotion on insufficient distribution
    BLOCK_PROMOTION_ON_INCOMPLETE: bool = _env_bool(
        "RINGRIFT_BLOCK_PROMOTION_INCOMPLETE", False
    )


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
# Data Freshness Defaults (December 2025)
# =============================================================================

def _env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment or use default."""
    val = os.environ.get(key, "")
    if not val:
        return default
    return val.lower() in ("true", "1", "yes")


@dataclass(frozen=True)
class DataFreshnessDefaults:
    """Default values for training data freshness checks.

    December 2025: Single source of truth for data freshness thresholds.
    Previously scattered across training_trigger_daemon.py and master_loop.py.

    Used by:
        - app/coordination/training_trigger_daemon.py
        - scripts/master_loop.py
        - app/coordination/training_freshness.py
    """
    # Maximum acceptable age of training data (hours)
    # Data older than this is considered stale and may trigger sync
    MAX_DATA_AGE_HOURS: float = _env_float("RINGRIFT_MAX_DATA_AGE_HOURS", 4.0)

    # Warning threshold (hours) - emit DATA_STALE warning above this
    FRESHNESS_WARNING_HOURS: float = _env_float("RINGRIFT_FRESHNESS_WARNING_HOURS", 2.0)

    # Strict mode: fail immediately if data is stale (no sync attempt)
    # Useful for high-quality training where only fresh data should be used
    STRICT_FRESHNESS: bool = _env_bool("RINGRIFT_STRICT_DATA_FRESHNESS", False)

    # If True, trigger sync instead of rejecting when data is stale
    ENFORCE_FRESHNESS_WITH_SYNC: bool = _env_bool("RINGRIFT_FRESHNESS_ENFORCE_SYNC", True)

    # Timeout for freshness-triggered sync (seconds)
    FRESHNESS_SYNC_TIMEOUT: float = _env_float("RINGRIFT_FRESHNESS_SYNC_TIMEOUT", 300.0)


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
    # Dec 28, 2025: Reduced from 600 (10 min) to 180 (3 min) to prevent long stalls
    # in training pipelines. With sync decoupled from circuit breaker backpressure,
    # data continues to flow even during circuit-open periods, so shorter recovery
    # attempts are safe and improve overall pipeline throughput.
    MAX_BACKOFF: float = _env_float("RINGRIFT_CB_MAX_BACKOFF", 180.0)

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
    # Dec 2025: Extended from 10s to 30s to prevent premature SIGKILL of slow-shutting daemons
    # (e.g., daemons with large state files, pending sync operations, or cleanup tasks)
    SHUTDOWN_GRACE_PERIOD: float = _env_float("RINGRIFT_DAEMON_SHUTDOWN_GRACE", 30.0)

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
# P2P Network Defaults (December 2025)
# =============================================================================

@dataclass(frozen=True)
class P2PDefaults:
    """Default values for P2P network configuration.

    Used by: scripts/p2p_orchestrator.py, app/coordination/p2p_*.py

    These values control the P2P mesh network for cluster coordination.
    """
    # P2P orchestrator port (the main cluster coordination port)
    DEFAULT_PORT: int = _env_int("RINGRIFT_P2P_PORT", 8770)

    # Health check HTTP endpoint port (same as P2P port)
    HEALTH_PORT: int = _env_int("RINGRIFT_P2P_HEALTH_PORT", 8770)

    # Data server port for file distribution
    DATA_SERVER_PORT: int = _env_int("RINGRIFT_DATA_SERVER_PORT", 8780)

    # Gossip interval (seconds) - how often to exchange state
    GOSSIP_INTERVAL: int = _env_int("RINGRIFT_P2P_GOSSIP_INTERVAL", 15)

    # Heartbeat interval (seconds) - liveness detection
    HEARTBEAT_INTERVAL: int = _env_int("RINGRIFT_P2P_HEARTBEAT_INTERVAL", 15)

    # Peer timeout (seconds) - consider dead after no heartbeat
    PEER_TIMEOUT: int = _env_int("RINGRIFT_P2P_PEER_TIMEOUT", 60)

    # Leader election timeout (seconds)
    ELECTION_TIMEOUT: int = _env_int("RINGRIFT_P2P_ELECTION_TIMEOUT", 30)

    # Startup grace period (seconds) - don't kill processes during startup
    STARTUP_GRACE_PERIOD: int = _env_int("RINGRIFT_P2P_STARTUP_GRACE_PERIOD", 120)

    # Maximum retry attempts for P2P operations
    MAX_RETRIES: int = _env_int("RINGRIFT_P2P_MAX_RETRIES", 3)

    # Quorum size for consensus (minimum voters needed)
    DEFAULT_QUORUM: int = _env_int("RINGRIFT_P2P_QUORUM", 3)

    # Maximum peers to track
    MAX_PEERS: int = _env_int("RINGRIFT_P2P_MAX_PEERS", 100)


def get_p2p_port() -> int:
    """Get the P2P orchestrator port.

    Example:
        port = get_p2p_port()  # Returns 8770 or env override
        url = f"http://{host}:{port}/status"
    """
    return P2PDefaults.DEFAULT_PORT


# =============================================================================
# Job Timeout Defaults (December 2025)
# =============================================================================

@dataclass(frozen=True)
class JobTimeoutDefaults:
    """Default timeout values per job type.

    Used by: app/coordination/job_reaper.py, scripts/p2p_orchestrator.py

    These values control when jobs are considered stuck/stale.
    """
    # GPU selfplay job timeout (seconds) - 1 hour
    GPU_SELFPLAY: int = _env_int("RINGRIFT_JOB_TIMEOUT_GPU_SELFPLAY", 3600)

    # CPU selfplay job timeout (seconds) - 2 hours (slower)
    CPU_SELFPLAY: int = _env_int("RINGRIFT_JOB_TIMEOUT_CPU_SELFPLAY", 7200)

    # Training job timeout (seconds) - 4 hours
    TRAINING: int = _env_int("RINGRIFT_JOB_TIMEOUT_TRAINING", 14400)

    # Tournament job timeout (seconds) - 1 hour
    TOURNAMENT: int = _env_int("RINGRIFT_JOB_TIMEOUT_TOURNAMENT", 3600)

    # Data export job timeout (seconds) - 30 minutes
    DATA_EXPORT: int = _env_int("RINGRIFT_JOB_TIMEOUT_DATA_EXPORT", 1800)

    # Evaluation job timeout (seconds) - 1 hour
    EVALUATION: int = _env_int("RINGRIFT_JOB_TIMEOUT_EVALUATION", 3600)

    # Model sync job timeout (seconds) - 30 minutes
    MODEL_SYNC: int = _env_int("RINGRIFT_JOB_TIMEOUT_MODEL_SYNC", 1800)

    # CMA-ES optimization timeout (seconds) - 8 hours
    CMAES: int = _env_int("RINGRIFT_JOB_TIMEOUT_CMAES", 28800)

    # Pipeline stage timeouts (seconds) - 10 minutes default
    PIPELINE_STAGE: int = _env_int("RINGRIFT_JOB_TIMEOUT_PIPELINE_STAGE", 600)


def get_job_timeout(job_type: str) -> int:
    """Get timeout for a specific job type.

    Args:
        job_type: Job type ("gpu_selfplay", "training", "tournament", etc.)

    Returns:
        Timeout in seconds

    Example:
        timeout = get_job_timeout("training")  # Returns 14400 (4 hours)
    """
    timeouts = {
        "gpu_selfplay": JobTimeoutDefaults.GPU_SELFPLAY,
        "cpu_selfplay": JobTimeoutDefaults.CPU_SELFPLAY,
        "selfplay": JobTimeoutDefaults.GPU_SELFPLAY,  # Alias
        "training": JobTimeoutDefaults.TRAINING,
        "tournament": JobTimeoutDefaults.TOURNAMENT,
        "data_export": JobTimeoutDefaults.DATA_EXPORT,
        "export": JobTimeoutDefaults.DATA_EXPORT,  # Alias
        "evaluation": JobTimeoutDefaults.EVALUATION,
        "eval": JobTimeoutDefaults.EVALUATION,  # Alias
        "model_sync": JobTimeoutDefaults.MODEL_SYNC,
        "sync": JobTimeoutDefaults.MODEL_SYNC,  # Alias
        "cmaes": JobTimeoutDefaults.CMAES,
        "pipeline": JobTimeoutDefaults.PIPELINE_STAGE,
    }
    return timeouts.get(job_type.lower(), JobTimeoutDefaults.GPU_SELFPLAY)


# =============================================================================
# Backpressure Defaults (December 2025)
# =============================================================================

@dataclass(frozen=True)
class BackpressureDefaults:
    """Default values for backpressure calculation.

    Used by: app/coordination/backpressure.py, app/coordination/types.py

    These values control spawn rate throttling across the cluster.
    """
    # Component weights (must sum to 1.0)
    WEIGHT_QUEUE: float = _env_float("RINGRIFT_BP_WEIGHT_QUEUE", 0.30)
    WEIGHT_TRAINING: float = _env_float("RINGRIFT_BP_WEIGHT_TRAINING", 0.25)
    WEIGHT_DISK: float = _env_float("RINGRIFT_BP_WEIGHT_DISK", 0.20)
    WEIGHT_SYNC: float = _env_float("RINGRIFT_BP_WEIGHT_SYNC", 0.15)
    WEIGHT_MEMORY: float = _env_float("RINGRIFT_BP_WEIGHT_MEMORY", 0.10)

    # Queue pressure thresholds (normalized 0-1)
    QUEUE_LOW: float = _env_float("RINGRIFT_BP_QUEUE_LOW", 0.3)
    QUEUE_MEDIUM: float = _env_float("RINGRIFT_BP_QUEUE_MEDIUM", 0.5)
    QUEUE_HIGH: float = _env_float("RINGRIFT_BP_QUEUE_HIGH", 0.7)
    QUEUE_CRITICAL: float = _env_float("RINGRIFT_BP_QUEUE_CRITICAL", 0.9)

    # Spawn rate multipliers per backpressure level
    MULTIPLIER_NONE: float = 1.0       # No backpressure
    MULTIPLIER_LOW: float = 0.75       # Slight reduction
    MULTIPLIER_SOFT: float = 0.50      # Moderate reduction
    MULTIPLIER_MEDIUM: float = 0.25    # Significant reduction
    MULTIPLIER_HARD: float = 0.10      # Heavy reduction
    MULTIPLIER_HIGH: float = 0.05      # Very heavy reduction
    MULTIPLIER_CRITICAL: float = 0.01  # Near-stop
    MULTIPLIER_STOP: float = 0.0       # Full stop

    # Cache TTL for backpressure signals (seconds)
    CACHE_TTL: float = _env_float("RINGRIFT_BP_CACHE_TTL", 10.0)

    # Cooldown between recalculations (seconds)
    COOLDOWN: int = _env_int("RINGRIFT_BP_COOLDOWN", 5)


def get_backpressure_multiplier(level: str) -> float:
    """Get spawn rate multiplier for a backpressure level.

    Args:
        level: Backpressure level ("none", "low", "medium", "high", etc.)

    Returns:
        Multiplier (0.0 to 1.0) to apply to spawn rate

    Example:
        multiplier = get_backpressure_multiplier("medium")  # Returns 0.25
        spawn_rate = base_rate * multiplier
    """
    multipliers = {
        "none": BackpressureDefaults.MULTIPLIER_NONE,
        "low": BackpressureDefaults.MULTIPLIER_LOW,
        "soft": BackpressureDefaults.MULTIPLIER_SOFT,
        "medium": BackpressureDefaults.MULTIPLIER_MEDIUM,
        "hard": BackpressureDefaults.MULTIPLIER_HARD,
        "high": BackpressureDefaults.MULTIPLIER_HIGH,
        "critical": BackpressureDefaults.MULTIPLIER_CRITICAL,
        "stop": BackpressureDefaults.MULTIPLIER_STOP,
    }
    return multipliers.get(level.lower(), BackpressureDefaults.MULTIPLIER_NONE)


# =============================================================================
# Resource Manager Defaults (December 2025)
# =============================================================================

@dataclass(frozen=True)
class ResourceManagerDefaults:
    """Default values for adaptive resource manager operations.

    Used by: app/coordination/adaptive_resource_manager.py

    These control proactive disk/memory/GPU monitoring across the cluster.
    """
    # Check interval - how often to check resource usage (seconds)
    CHECK_INTERVAL: int = _env_int("RINGRIFT_RESOURCE_CHECK_INTERVAL", 300)  # 5 minutes

    # Cleanup cooldown - minimum time between cleanup operations (seconds)
    CLEANUP_COOLDOWN: int = _env_int("RINGRIFT_CLEANUP_COOLDOWN", 1800)  # 30 minutes

    # Aggregation interval - how often to aggregate selfplay data (seconds)
    AGGREGATION_INTERVAL: int = _env_int("RINGRIFT_AGGREGATION_INTERVAL", 600)  # 10 minutes

    # Disk thresholds (percentage)
    DISK_WARNING_THRESHOLD: int = _env_int("RINGRIFT_DISK_WARNING_THRESHOLD", 85)
    DISK_CRITICAL_THRESHOLD: int = _env_int("RINGRIFT_DISK_CRITICAL_THRESHOLD", 92)
    DISK_CLEANUP_THRESHOLD: int = _env_int("RINGRIFT_DISK_CLEANUP_THRESHOLD", 90)

    # Memory thresholds (percentage)
    MEMORY_WARNING_THRESHOLD: int = _env_int("RINGRIFT_MEMORY_WARNING_THRESHOLD", 85)
    MEMORY_CRITICAL_THRESHOLD: int = _env_int("RINGRIFT_MEMORY_CRITICAL_THRESHOLD", 95)


# =============================================================================
# Orphan Detection Defaults (December 2025)
# =============================================================================

@dataclass(frozen=True)
class OrphanDetectionDefaults:
    """Default values for orphan detection daemon.

    Used by: app/coordination/orphan_detection_daemon.py
    """
    # Scan interval - how often to scan for orphaned databases (seconds)
    SCAN_INTERVAL: float = _env_float("RINGRIFT_ORPHAN_SCAN_INTERVAL", 300.0)  # 5 minutes

    # Minimum games required in a DB to auto-register
    MIN_GAMES_TO_REGISTER: int = _env_int("RINGRIFT_ORPHAN_MIN_GAMES", 1)

    # Minimum age before considering cleanup (hours)
    MIN_AGE_BEFORE_CLEANUP_HOURS: float = _env_float("RINGRIFT_ORPHAN_MIN_AGE_HOURS", 24.0)

    # Alert threshold - emit alert if orphan count exceeds this
    MAX_ORPHAN_COUNT_BEFORE_ALERT: int = _env_int("RINGRIFT_ORPHAN_ALERT_THRESHOLD", 100)


# =============================================================================
# Cluster Watchdog Defaults (December 2025)
# =============================================================================

@dataclass(frozen=True)
class ClusterWatchdogDefaults:
    """Default values for cluster watchdog daemon.

    Used by: app/coordination/cluster_watchdog_daemon.py

    These control automatic node activation and GPU utilization monitoring.
    """
    # SSH timeout for node checks (seconds)
    SSH_TIMEOUT: int = _env_int("RINGRIFT_WATCHDOG_SSH_TIMEOUT", 30)

    # Minimum GPU utilization - below this, spawn selfplay (percentage)
    MIN_GPU_UTILIZATION: float = _env_float("RINGRIFT_WATCHDOG_MIN_GPU_UTIL", 20.0)

    # Activation cooldown - time to wait after activating a node (seconds)
    ACTIVATION_COOLDOWN: int = _env_int("RINGRIFT_WATCHDOG_ACTIVATION_COOLDOWN", 600)  # 10 minutes

    # Maximum consecutive failures before escalation
    MAX_CONSECUTIVE_FAILURES: int = _env_int("RINGRIFT_WATCHDOG_MAX_FAILURES", 3)

    # Maximum nodes to activate per cycle
    MAX_ACTIVATIONS_PER_CYCLE: int = _env_int("RINGRIFT_WATCHDOG_MAX_ACTIVATIONS", 10)


# =============================================================================
# Daemon Health Check Defaults (December 2025)
# =============================================================================

@dataclass(frozen=True)
class DaemonHealthDefaults:
    """Default values for daemon health monitoring.

    Used by: app/coordination/daemon_manager.py

    These control how daemons are monitored and when they're restarted.
    """
    # Health check interval (seconds) - how often to check daemon health
    CHECK_INTERVAL: float = _env_float("RINGRIFT_DAEMON_HEALTH_INTERVAL", 60.0)

    # Critical daemon check interval (seconds) - faster for critical daemons
    CRITICAL_CHECK_INTERVAL: float = _env_float(
        "RINGRIFT_DAEMON_CRITICAL_CHECK_INTERVAL", 30.0
    )

    # Maximum failures before restart
    MAX_FAILURES: int = _env_int("RINGRIFT_DAEMON_MAX_FAILURES", 3)

    # Restart backoff base (seconds)
    RESTART_BACKOFF_BASE: float = _env_float("RINGRIFT_DAEMON_RESTART_BACKOFF", 5.0)

    # Maximum restart backoff (seconds)
    RESTART_BACKOFF_MAX: float = _env_float("RINGRIFT_DAEMON_RESTART_BACKOFF_MAX", 300.0)

    # Startup timeout (seconds) - how long to wait for daemon to become healthy
    STARTUP_TIMEOUT: float = _env_float("RINGRIFT_DAEMON_STARTUP_TIMEOUT", 30.0)

    # Shutdown timeout (seconds) - how long to wait for graceful shutdown
    # Dec 2025: Extended from 10s to 30s (matching SHUTDOWN_GRACE_PERIOD)
    SHUTDOWN_TIMEOUT: float = _env_float("RINGRIFT_DAEMON_SHUTDOWN_TIMEOUT", 30.0)


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


def get_aiohttp_timeout(operation: str = "http"):
    """Get aiohttp ClientTimeout for an operation type.

    This is the preferred way to get timeouts for aiohttp sessions.
    ALWAYS use this instead of creating sessions without timeout.

    Args:
        operation: Operation type ("http", "health", "connect", etc.)

    Returns:
        aiohttp.ClientTimeout configured for the operation

    Example:
        from app.config.coordination_defaults import get_aiohttp_timeout

        # Health check (5s timeout)
        async with aiohttp.ClientSession(timeout=get_aiohttp_timeout("health")) as session:
            ...

        # Standard HTTP (30s timeout)
        async with aiohttp.ClientSession(timeout=get_aiohttp_timeout()) as session:
            ...

    Note:
        Import aiohttp in your module - we use lazy import to avoid
        circular dependencies in this config module.
    """
    # Lazy import to avoid circular dependencies
    try:
        import aiohttp
    except ImportError:
        # Return None if aiohttp not available - caller should handle
        return None

    timeout_seconds = get_timeout(operation)
    # Use total timeout - this covers the entire request lifecycle
    return aiohttp.ClientTimeout(total=timeout_seconds)


# =============================================================================
# Job Reaper Defaults (December 27, 2025)
# =============================================================================

@dataclass(frozen=True)
class JobReaperDefaults:
    """Default values for job reaper daemon.

    Used by: job_reaper.py
    """
    # Interval between reaper checks (seconds)
    CHECK_INTERVAL: int = _env_int("RINGRIFT_JOB_REAPER_CHECK_INTERVAL", 30)

    # Default job timeout (seconds)
    DEFAULT_JOB_TIMEOUT: int = _env_int("RINGRIFT_DEFAULT_JOB_TIMEOUT", 3600)

    # Maximum times to reassign a failed job
    MAX_REASSIGN_ATTEMPTS: int = _env_int("RINGRIFT_MAX_REASSIGN_ATTEMPTS", 3)

    # Duration to blacklist failing nodes (seconds)
    NODE_BLACKLIST_DURATION: int = _env_int("RINGRIFT_NODE_BLACKLIST_DURATION", 600)

    # SSH command timeout (seconds)
    SSH_TIMEOUT: int = _env_int("RINGRIFT_JOB_REAPER_SSH_TIMEOUT", 30)

    # P2P leader check timeout (seconds)
    LEADER_CHECK_TIMEOUT: int = _env_int("RINGRIFT_LEADER_CHECK_TIMEOUT", 5)

    # Delay before retrying when not leader (seconds)
    LEADER_RETRY_DELAY: int = _env_int("RINGRIFT_LEADER_RETRY_DELAY", 10)


# =============================================================================
# Coordinator Health Defaults (December 27, 2025)
# =============================================================================

@dataclass(frozen=True)
class CoordinatorHealthDefaults:
    """Default values for coordinator health monitoring.

    Used by: coordinator_health_monitor_daemon.py
    """
    # Threshold for considering heartbeat stale (seconds)
    HEARTBEAT_STALE_THRESHOLD: int = _env_int(
        "RINGRIFT_COORDINATOR_HEARTBEAT_STALE_THRESHOLD", 300
    )

    # Cooldown between degraded alerts (seconds)
    DEGRADED_COOLDOWN: int = _env_int("RINGRIFT_COORDINATOR_DEGRADED_COOLDOWN", 60)

    # Max failures before marking permanently unhealthy
    INIT_FAILURE_MAX_RETRIES: int = _env_int(
        "RINGRIFT_COORDINATOR_INIT_FAILURE_MAX_RETRIES", 3
    )


# =============================================================================
# Work Queue Monitor Defaults (December 27, 2025)
# =============================================================================

@dataclass(frozen=True)
class WorkQueueMonitorDefaults:
    """Default values for work queue monitoring.

    Used by: work_queue_monitor_daemon.py
    """
    # Queue depth threshold for backpressure signal
    BACKPRESSURE_THRESHOLD: int = _env_int(
        "RINGRIFT_QUEUE_BACKPRESSURE_THRESHOLD", 100
    )

    # Time threshold for stuck jobs (seconds)
    STUCK_JOB_THRESHOLD: int = _env_int("RINGRIFT_STUCK_JOB_THRESHOLD", 300)

    # Max concurrent jobs per node
    NODE_OVERLOAD_THRESHOLD: int = _env_int("RINGRIFT_NODE_OVERLOAD_THRESHOLD", 5)

    # Rolling window size for latency calculation
    LATENCY_WINDOW_SIZE: int = _env_int("RINGRIFT_LATENCY_WINDOW_SIZE", 100)


# =============================================================================
# Ephemeral Guard Defaults (December 27, 2025)
# =============================================================================

@dataclass(frozen=True)
class EphemeralGuardDefaults:
    """Default values for ephemeral data guard.

    Used by: ephemeral_data_guard.py
    """
    # Interval between checkpoints (seconds)
    CHECKPOINT_INTERVAL: int = _env_int("RINGRIFT_EPHEMERAL_CHECKPOINT_INTERVAL", 60)

    # Timeout for host heartbeat (seconds)
    HEARTBEAT_TIMEOUT: int = _env_int("RINGRIFT_EPHEMERAL_HEARTBEAT_TIMEOUT", 120)

    # Unsynced games threshold for evacuation
    EVACUATION_THRESHOLD: int = _env_int("RINGRIFT_EVACUATION_THRESHOLD", 50)

    # Threshold for immediate write-through of critical games
    CRITICAL_GAME_THRESHOLD: int = _env_int("RINGRIFT_CRITICAL_GAME_THRESHOLD", 10)


# =============================================================================
# Selfplay Allocation Defaults (December 27, 2025)
# =============================================================================

@dataclass(frozen=True)
class SelfplayAllocationDefaults:
    """Default values for selfplay job allocation.

    Used by: selfplay_scheduler.py, selfplay_orchestrator.py
    """
    # Default games per config allocation
    GAMES_PER_CONFIG: int = _env_int("RINGRIFT_SELFPLAY_GAMES_PER_CONFIG", 500)

    # Minimum games per allocation
    MIN_GAMES_PER_ALLOCATION: int = _env_int(
        "RINGRIFT_MIN_GAMES_PER_ALLOCATION", 100
    )

    # Minimum RAM for task allocation (GB)
    MIN_MEMORY_GB: int = _env_int("RINGRIFT_MIN_MEMORY_GB_FOR_TASKS", 8)

    # Disk usage warning threshold (%)
    DISK_WARNING_THRESHOLD: int = _env_int("RINGRIFT_DISK_WARNING_THRESHOLD", 90)

    # Memory usage warning threshold (%)
    MEMORY_WARNING_THRESHOLD: int = _env_int("RINGRIFT_MEMORY_WARNING_THRESHOLD", 95)

    # Default MCTS budget for large boards
    LARGE_BOARD_BUDGET: int = _env_int("RINGRIFT_LARGE_BOARD_MCTS_BUDGET", 800)


# =============================================================================
# Cross Process Defaults (December 27, 2025)
# =============================================================================

@dataclass(frozen=True)
class CrossProcessDefaults:
    """Default values for cross-process events.

    Used by: cross_process_events.py
    """
    # Default retention for events (hours)
    RETENTION_HOURS: int = _env_int("RINGRIFT_CROSS_PROCESS_RETENTION_HOURS", 24)

    # Timeout for subscriber operations (seconds)
    SUBSCRIBER_TIMEOUT: int = _env_int("RINGRIFT_SUBSCRIBER_TIMEOUT", 300)


# =============================================================================
# Idle Threshold Defaults (December 27, 2025)
# =============================================================================

@dataclass(frozen=True)
class IdleThresholdDefaults:
    """Default values for GPU idle detection and process cleanup.

    Used by: app/coordination/idle_resource_daemon.py, unified_idle_shutdown_daemon.py
    """
    # GPU idle threshold - consider GPU idle after this duration (seconds)
    GPU_IDLE_THRESHOLD: int = _env_int("RINGRIFT_GPU_IDLE_THRESHOLD", 600)  # 10 minutes

    # Maximum selfplay processes per node before cleanup
    MAX_PROCESSES: int = _env_int("RINGRIFT_MAX_PROCESSES", 128)

    # Disk usage threshold for cleanup (percentage)
    CLEANUP_THRESHOLD_PERCENT: int = _env_int("RINGRIFT_CLEANUP_THRESHOLD_PERCENT", 90)

    # Minimum GPU utilization to consider active (percentage)
    MIN_GPU_UTILIZATION: float = _env_float("RINGRIFT_MIN_GPU_UTILIZATION", 10.0)

    # Grace period before terminating idle instance (seconds)
    TERMINATION_GRACE_PERIOD: int = _env_int("RINGRIFT_TERMINATION_GRACE_PERIOD", 300)


# =============================================================================
# Provider Defaults (December 27, 2025)
# =============================================================================

@dataclass(frozen=True)
class ProviderDefaults:
    """Default values for cloud provider operations.

    Used by: app/coordination/multi_provider_orchestrator.py
    """
    # Interval between provider status checks (seconds)
    CHECK_INTERVAL: int = _env_int("RINGRIFT_PROVIDER_CHECK_INTERVAL", 300)  # 5 minutes

    # Timeout for provider API calls (seconds)
    TIMEOUT: int = _env_int("RINGRIFT_PROVIDER_TIMEOUT", 30)

    # Delay between retries on failure (seconds)
    RETRY_DELAY: int = _env_int("RINGRIFT_PROVIDER_RETRY_DELAY", 5)

    # Maximum retries for provider operations
    MAX_RETRIES: int = _env_int("RINGRIFT_PROVIDER_MAX_RETRIES", 3)

    # Cooldown after provider error before retry (seconds)
    ERROR_COOLDOWN: int = _env_int("RINGRIFT_PROVIDER_ERROR_COOLDOWN", 60)


# =============================================================================
# Inventory Defaults (December 27, 2025)
# =============================================================================

@dataclass(frozen=True)
class InventoryDefaults:
    """Default values for unified inventory management.

    Used by: app/coordination/unified_inventory.py
    """
    # Interval between inventory refreshes (seconds)
    REFRESH_INTERVAL: int = _env_int("RINGRIFT_INVENTORY_REFRESH_INTERVAL", 300)

    # Cache TTL for inventory data (seconds)
    CACHE_TTL: int = _env_int("RINGRIFT_INVENTORY_CACHE_TTL", 60)

    # Threshold for considering inventory stale (seconds)
    STALE_THRESHOLD: int = _env_int("RINGRIFT_INVENTORY_STALE_THRESHOLD", 1800)

    # Maximum nodes to track in inventory
    MAX_NODES: int = _env_int("RINGRIFT_INVENTORY_MAX_NODES", 200)

    # Timeout for inventory fetch operations (seconds)
    FETCH_TIMEOUT: int = _env_int("RINGRIFT_INVENTORY_FETCH_TIMEOUT", 10)


# =============================================================================
# Sync Integrity Defaults (December 27, 2025)
# =============================================================================

@dataclass(frozen=True)
class SyncIntegrityDefaults:
    """Default values for sync integrity checking.

    Used by: sync_integrity.py
    """
    # Default chunk size for file hashing (bytes)
    DEFAULT_CHUNK_SIZE: int = _env_int("RINGRIFT_SYNC_DEFAULT_CHUNK_SIZE", 8192)

    # Large chunk size for big files (bytes)
    LARGE_CHUNK_SIZE: int = _env_int("RINGRIFT_SYNC_LARGE_CHUNK_SIZE", 65536)


# =============================================================================
# SSH Defaults (December 28, 2025)
# =============================================================================

@dataclass(frozen=True)
class SSHDefaults:
    """Default values for SSH operations.

    Used by: scripts/p2p_orchestrator.py, app/coordination/auto_sync_daemon.py,
             app/core/ssh.py, scripts/p2p/managers/*.py

    Consolidates 40+ scattered SSH timeout values across the codebase.
    """
    # Command execution timeout (seconds) - for quick commands like nvidia-smi
    COMMAND_TIMEOUT: float = _env_float("RINGRIFT_SSH_COMMAND_TIMEOUT", 30.0)

    # Long command timeout (seconds) - for training, exports, migrations
    LONG_COMMAND_TIMEOUT: float = _env_float("RINGRIFT_SSH_LONG_COMMAND_TIMEOUT", 300.0)

    # SCP file transfer timeout (seconds) - for model/data transfers
    SCP_TIMEOUT: float = _env_float("RINGRIFT_SCP_TIMEOUT", 60.0)

    # Connection timeout (seconds) - initial SSH handshake
    CONNECT_TIMEOUT: float = _env_float("RINGRIFT_SSH_CONNECT_TIMEOUT", 10.0)

    # Rsync timeout (seconds) - for rsync operations
    RSYNC_TIMEOUT: float = _env_float("RINGRIFT_RSYNC_TIMEOUT", 60.0)

    # Health check SSH timeout (seconds) - quick connectivity tests
    HEALTH_CHECK_TIMEOUT: float = _env_float("RINGRIFT_SSH_HEALTH_CHECK_TIMEOUT", 5.0)

    # Maximum retries for SSH operations
    MAX_RETRIES: int = _env_int("RINGRIFT_SSH_MAX_RETRIES", 3)


# =============================================================================
# Job Defaults (December 28, 2025)
# =============================================================================

@dataclass(frozen=True)
class JobDefaults:
    """Default values for job execution timeouts.

    Used by: scripts/p2p_orchestrator.py, scripts/p2p/managers/job_manager.py,
             app/coordination/job_reaper.py

    Consolidates job-specific timeouts that were scattered across P2P code.
    """
    # Selfplay job timeout (seconds) - 2 hours
    SELFPLAY_TIMEOUT: float = _env_float("RINGRIFT_JOB_SELFPLAY_TIMEOUT", 7200.0)

    # Training job timeout (seconds) - 24 hours
    TRAINING_TIMEOUT: float = _env_float("RINGRIFT_JOB_TRAINING_TIMEOUT", 86400.0)

    # Job status check timeout (seconds) - 10 minutes
    JOB_STATUS_TIMEOUT: float = _env_float("RINGRIFT_JOB_STATUS_TIMEOUT", 600.0)

    # Health check timeout (seconds) - for node health checks
    HEALTH_CHECK_TIMEOUT: float = _env_float("RINGRIFT_JOB_HEALTH_CHECK_TIMEOUT", 30.0)

    # Tournament job timeout (seconds) - 4 hours
    TOURNAMENT_TIMEOUT: float = _env_float("RINGRIFT_JOB_TOURNAMENT_TIMEOUT", 14400.0)

    # Gauntlet evaluation timeout (seconds) - 2 hours
    GAUNTLET_TIMEOUT: float = _env_float("RINGRIFT_JOB_GAUNTLET_TIMEOUT", 7200.0)

    # Data export timeout (seconds) - 30 minutes
    EXPORT_TIMEOUT: float = _env_float("RINGRIFT_JOB_EXPORT_TIMEOUT", 1800.0)

    # Model sync timeout (seconds) - 30 minutes
    MODEL_SYNC_TIMEOUT: float = _env_float("RINGRIFT_JOB_MODEL_SYNC_TIMEOUT", 1800.0)


# =============================================================================
# Peer Defaults (December 28, 2025)
# =============================================================================

@dataclass(frozen=True)
class PeerDefaults:
    """Default values for P2P peer management.

    Used by: scripts/p2p_orchestrator.py, scripts/p2p/managers/*.py,
             app/p2p/constants.py

    Consolidates P2P timing constants for peer discovery and health.
    """
    # Heartbeat interval (seconds) - how often to send heartbeats
    HEARTBEAT_INTERVAL: float = _env_float("RINGRIFT_PEER_HEARTBEAT_INTERVAL", 15.0)

    # Peer timeout (seconds) - consider peer dead after no heartbeat
    PEER_TIMEOUT: float = _env_float("RINGRIFT_PEER_TIMEOUT", 60.0)

    # Gossip interval (seconds) - how often to exchange state
    GOSSIP_INTERVAL: float = _env_float("RINGRIFT_PEER_GOSSIP_INTERVAL", 15.0)

    # Manifest collection timeout (seconds) - for data manifest requests
    MANIFEST_TIMEOUT: float = _env_float("RINGRIFT_PEER_MANIFEST_TIMEOUT", 300.0)

    # Election timeout (seconds) - for leader election
    ELECTION_TIMEOUT: float = _env_float("RINGRIFT_PEER_ELECTION_TIMEOUT", 30.0)

    # Bootstrap interval (seconds) - for initial peer discovery
    BOOTSTRAP_INTERVAL: float = _env_float("RINGRIFT_PEER_BOOTSTRAP_INTERVAL", 60.0)

    # Suspect timeout (seconds) - grace period before marking dead
    SUSPECT_TIMEOUT: float = _env_float("RINGRIFT_PEER_SUSPECT_TIMEOUT", 30.0)

    # Retry dead node interval (seconds) - how often to retry dead nodes
    RETRY_DEAD_NODE_INTERVAL: float = _env_float("RINGRIFT_PEER_RETRY_DEAD_INTERVAL", 120.0)


def get_ssh_timeout(operation: str = "command") -> float:
    """Get SSH timeout for a specific operation type.

    Args:
        operation: Operation type ("command", "long", "scp", "connect", "rsync", "health")

    Returns:
        Timeout in seconds

    Example:
        timeout = get_ssh_timeout("rsync")  # Returns 60.0
    """
    timeouts = {
        "command": SSHDefaults.COMMAND_TIMEOUT,
        "long": SSHDefaults.LONG_COMMAND_TIMEOUT,
        "scp": SSHDefaults.SCP_TIMEOUT,
        "connect": SSHDefaults.CONNECT_TIMEOUT,
        "rsync": SSHDefaults.RSYNC_TIMEOUT,
        "health": SSHDefaults.HEALTH_CHECK_TIMEOUT,
    }
    return timeouts.get(operation, SSHDefaults.COMMAND_TIMEOUT)


def get_peer_timeout(timeout_type: str = "peer") -> float:
    """Get P2P peer timeout for a specific type.

    Args:
        timeout_type: Type ("heartbeat", "peer", "gossip", "manifest", "election", "bootstrap")

    Returns:
        Timeout in seconds

    Example:
        timeout = get_peer_timeout("manifest")  # Returns 300.0
    """
    timeouts = {
        "heartbeat": PeerDefaults.HEARTBEAT_INTERVAL,
        "peer": PeerDefaults.PEER_TIMEOUT,
        "gossip": PeerDefaults.GOSSIP_INTERVAL,
        "manifest": PeerDefaults.MANIFEST_TIMEOUT,
        "election": PeerDefaults.ELECTION_TIMEOUT,
        "bootstrap": PeerDefaults.BOOTSTRAP_INTERVAL,
        "suspect": PeerDefaults.SUSPECT_TIMEOUT,
        "retry_dead": PeerDefaults.RETRY_DEAD_NODE_INTERVAL,
    }
    return timeouts.get(timeout_type, PeerDefaults.PEER_TIMEOUT)


# =============================================================================
# Curriculum Integration Defaults (December 28, 2025)
# =============================================================================

@dataclass(frozen=True)
class CurriculumDefaults:
    """Default values for curriculum integration and feedback loops.

    Used by: app/coordination/curriculum_integration.py

    These control the feedback loops between training performance and
    curriculum weight adjustments.
    """
    # Win rate threshold - opponent considered mastered above this (fraction)
    MASTERY_THRESHOLD: float = _env_float("RINGRIFT_MASTERY_THRESHOLD", 0.85)

    # Interval between curriculum checks (seconds)
    CHECK_INTERVAL: float = _env_float("RINGRIFT_CURRICULUM_CHECK_INTERVAL", 120.0)

    # Temperature boost factor for exploration on low quality (1.3 = +30%)
    EXPLORATION_BOOST_FACTOR: float = _env_float(
        "RINGRIFT_EXPLORATION_BOOST_FACTOR", 1.3
    )

    # Quality threshold below which exploration is boosted (fraction)
    LOW_QUALITY_THRESHOLD: float = _env_float(
        "RINGRIFT_LOW_QUALITY_THRESHOLD", 0.3
    )

    # Minimum games before updating curriculum weights
    MIN_GAMES_FOR_UPDATE: int = _env_int("RINGRIFT_MIN_GAMES_FOR_UPDATE", 100)

    # Weight decay for stale configs (per hour)
    WEIGHT_STALE_DECAY: float = _env_float("RINGRIFT_WEIGHT_STALE_DECAY", 0.1)

    # Maximum weight for any single config (prevents monopolization)
    MAX_WEIGHT: float = _env_float("RINGRIFT_MAX_CURRICULUM_WEIGHT", 0.5)

    # Minimum weight for any config (ensures diversity)
    MIN_WEIGHT: float = _env_float("RINGRIFT_MIN_CURRICULUM_WEIGHT", 0.05)


# =============================================================================
# Health Check Orchestrator Defaults (December 28, 2025)
# =============================================================================

@dataclass(frozen=True)
class HealthCheckOrchestratorDefaults:
    """Default values for health check orchestrator.

    Used by: app/coordination/health_check_orchestrator.py

    These control the intervals and timeouts for different types of
    health checks across the cluster.
    """
    # P2P health check interval (seconds) - fastest checks
    P2P_CHECK_INTERVAL: int = _env_int("RINGRIFT_P2P_CHECK_INTERVAL", 60)

    # SSH health check interval (seconds)
    SSH_CHECK_INTERVAL: int = _env_int("RINGRIFT_SSH_CHECK_INTERVAL", 120)

    # Cloud provider API check interval (seconds)
    PROVIDER_CHECK_INTERVAL: int = _env_int("RINGRIFT_PROVIDER_CHECK_INTERVAL", 120)

    # GPU/CPU utilization check interval (seconds)
    UTILIZATION_CHECK_INTERVAL: int = _env_int(
        "RINGRIFT_UTILIZATION_CHECK_INTERVAL", 60
    )

    # Default check interval for the orchestrator main loop (seconds)
    DEFAULT_CHECK_INTERVAL: float = _env_float(
        "RINGRIFT_HEALTH_CHECK_ORCHESTRATOR_INTERVAL", 60.0
    )

    # Timeout for individual health check operations (seconds)
    CHECK_TIMEOUT: int = _env_int("RINGRIFT_HEALTH_CHECK_TIMEOUT", 10)

    # Maximum concurrent health checks to run in parallel
    MAX_CONCURRENT: int = _env_int("RINGRIFT_HEALTH_CHECK_MAX_CONCURRENT", 20)


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
        # December 27, 2025: P2P network defaults
        "p2p": {
            "default_port": P2PDefaults.DEFAULT_PORT,
            "health_port": P2PDefaults.HEALTH_PORT,
            "data_server_port": P2PDefaults.DATA_SERVER_PORT,
            "gossip_interval": P2PDefaults.GOSSIP_INTERVAL,
            "heartbeat_interval": P2PDefaults.HEARTBEAT_INTERVAL,
            "peer_timeout": P2PDefaults.PEER_TIMEOUT,
            "election_timeout": P2PDefaults.ELECTION_TIMEOUT,
            "startup_grace_period": P2PDefaults.STARTUP_GRACE_PERIOD,
        },
        # December 27, 2025: Job timeout defaults
        "job_timeouts": {
            "gpu_selfplay": JobTimeoutDefaults.GPU_SELFPLAY,
            "cpu_selfplay": JobTimeoutDefaults.CPU_SELFPLAY,
            "training": JobTimeoutDefaults.TRAINING,
            "tournament": JobTimeoutDefaults.TOURNAMENT,
            "data_export": JobTimeoutDefaults.DATA_EXPORT,
            "evaluation": JobTimeoutDefaults.EVALUATION,
            "cmaes": JobTimeoutDefaults.CMAES,
        },
        # December 27, 2025: Backpressure defaults
        "backpressure": {
            "weight_queue": BackpressureDefaults.WEIGHT_QUEUE,
            "weight_training": BackpressureDefaults.WEIGHT_TRAINING,
            "weight_disk": BackpressureDefaults.WEIGHT_DISK,
            "queue_low": BackpressureDefaults.QUEUE_LOW,
            "queue_critical": BackpressureDefaults.QUEUE_CRITICAL,
            "cache_ttl": BackpressureDefaults.CACHE_TTL,
        },
        # December 27, 2025: Daemon health defaults
        "daemon_health": {
            "check_interval": DaemonHealthDefaults.CHECK_INTERVAL,
            "critical_check_interval": DaemonHealthDefaults.CRITICAL_CHECK_INTERVAL,
            "max_failures": DaemonHealthDefaults.MAX_FAILURES,
            "startup_timeout": DaemonHealthDefaults.STARTUP_TIMEOUT,
            "shutdown_timeout": DaemonHealthDefaults.SHUTDOWN_TIMEOUT,
        },
        # December 27, 2025: Idle threshold defaults
        "idle_threshold": {
            "gpu_idle_threshold": IdleThresholdDefaults.GPU_IDLE_THRESHOLD,
            "max_processes": IdleThresholdDefaults.MAX_PROCESSES,
            "cleanup_threshold_percent": IdleThresholdDefaults.CLEANUP_THRESHOLD_PERCENT,
            "min_gpu_utilization": IdleThresholdDefaults.MIN_GPU_UTILIZATION,
            "termination_grace_period": IdleThresholdDefaults.TERMINATION_GRACE_PERIOD,
        },
        # December 27, 2025: Provider defaults
        "provider": {
            "check_interval": ProviderDefaults.CHECK_INTERVAL,
            "timeout": ProviderDefaults.TIMEOUT,
            "retry_delay": ProviderDefaults.RETRY_DELAY,
            "max_retries": ProviderDefaults.MAX_RETRIES,
            "error_cooldown": ProviderDefaults.ERROR_COOLDOWN,
        },
        # December 27, 2025: Inventory defaults
        "inventory": {
            "refresh_interval": InventoryDefaults.REFRESH_INTERVAL,
            "cache_ttl": InventoryDefaults.CACHE_TTL,
            "stale_threshold": InventoryDefaults.STALE_THRESHOLD,
            "max_nodes": InventoryDefaults.MAX_NODES,
            "fetch_timeout": InventoryDefaults.FETCH_TIMEOUT,
        },
        # December 28, 2025: Curriculum defaults
        "curriculum": {
            "mastery_threshold": CurriculumDefaults.MASTERY_THRESHOLD,
            "check_interval": CurriculumDefaults.CHECK_INTERVAL,
            "exploration_boost_factor": CurriculumDefaults.EXPLORATION_BOOST_FACTOR,
            "low_quality_threshold": CurriculumDefaults.LOW_QUALITY_THRESHOLD,
            "min_games_for_update": CurriculumDefaults.MIN_GAMES_FOR_UPDATE,
            "max_weight": CurriculumDefaults.MAX_WEIGHT,
            "min_weight": CurriculumDefaults.MIN_WEIGHT,
        },
        # December 28, 2025: Health check orchestrator defaults
        "health_orchestrator": {
            "p2p_check_interval": HealthCheckOrchestratorDefaults.P2P_CHECK_INTERVAL,
            "ssh_check_interval": HealthCheckOrchestratorDefaults.SSH_CHECK_INTERVAL,
            "provider_check_interval": HealthCheckOrchestratorDefaults.PROVIDER_CHECK_INTERVAL,
            "utilization_check_interval": HealthCheckOrchestratorDefaults.UTILIZATION_CHECK_INTERVAL,
            "check_timeout": HealthCheckOrchestratorDefaults.CHECK_TIMEOUT,
            "max_concurrent": HealthCheckOrchestratorDefaults.MAX_CONCURRENT,
        },
        # December 28, 2025: SSH defaults
        "ssh": {
            "command_timeout": SSHDefaults.COMMAND_TIMEOUT,
            "long_command_timeout": SSHDefaults.LONG_COMMAND_TIMEOUT,
            "scp_timeout": SSHDefaults.SCP_TIMEOUT,
            "connect_timeout": SSHDefaults.CONNECT_TIMEOUT,
            "rsync_timeout": SSHDefaults.RSYNC_TIMEOUT,
            "health_check_timeout": SSHDefaults.HEALTH_CHECK_TIMEOUT,
            "max_retries": SSHDefaults.MAX_RETRIES,
        },
        # December 28, 2025: Job defaults
        "job": {
            "selfplay_timeout": JobDefaults.SELFPLAY_TIMEOUT,
            "training_timeout": JobDefaults.TRAINING_TIMEOUT,
            "job_status_timeout": JobDefaults.JOB_STATUS_TIMEOUT,
            "health_check_timeout": JobDefaults.HEALTH_CHECK_TIMEOUT,
            "tournament_timeout": JobDefaults.TOURNAMENT_TIMEOUT,
            "gauntlet_timeout": JobDefaults.GAUNTLET_TIMEOUT,
            "export_timeout": JobDefaults.EXPORT_TIMEOUT,
            "model_sync_timeout": JobDefaults.MODEL_SYNC_TIMEOUT,
        },
        # December 28, 2025: Peer defaults
        "peer": {
            "heartbeat_interval": PeerDefaults.HEARTBEAT_INTERVAL,
            "peer_timeout": PeerDefaults.PEER_TIMEOUT,
            "gossip_interval": PeerDefaults.GOSSIP_INTERVAL,
            "manifest_timeout": PeerDefaults.MANIFEST_TIMEOUT,
            "election_timeout": PeerDefaults.ELECTION_TIMEOUT,
            "bootstrap_interval": PeerDefaults.BOOTSTRAP_INTERVAL,
            "suspect_timeout": PeerDefaults.SUSPECT_TIMEOUT,
            "retry_dead_node_interval": PeerDefaults.RETRY_DEAD_NODE_INTERVAL,
        },
    }


__all__ = [
    # Config classes (alphabetical)
    "BackpressureDefaults",
    "BandwidthDefaults",
    "CacheDefaults",
    "CircuitBreakerDefaults",
    "ClusterWatchdogDefaults",
    "CoordinatorHealthDefaults",
    "CrossProcessDefaults",
    "CurriculumDefaults",  # December 28, 2025
    "DaemonHealthDefaults",
    "DaemonLoopDefaults",
    "DurationDefaults",
    "EphemeralDefaults",
    "EphemeralGuardDefaults",
    "HealthCheckOrchestratorDefaults",  # December 28, 2025
    "HealthDefaults",
    "HeartbeatDefaults",
    "IdleThresholdDefaults",  # December 27, 2025
    "InventoryDefaults",  # December 27, 2025
    "JobDefaults",  # December 28, 2025
    "JobReaperDefaults",
    "JobTimeoutDefaults",
    "LockDefaults",
    "MetricsAnalysisDefaults",
    "MonitoringDefaults",
    "NetworkRetryDefaults",
    "OperationTimeouts",
    "OptimizationDefaults",
    "OrphanDetectionDefaults",
    "P2PDefaults",
    "PeerDefaults",  # December 28, 2025
    "PIDDefaults",
    "ProviderDefaults",  # December 27, 2025
    "QueueDefaults",
    "ResourceLimitsDefaults",
    "ResourceManagerDefaults",
    "ResourceMonitoringDefaults",
    "RetryDefaults",
    "ScalingDefaults",
    "SchedulerDefaults",
    "SelfplayAllocationDefaults",
    "SQLiteDefaults",
    "SSHDefaults",  # December 28, 2025
    "SyncCoordinatorDefaults",
    "SyncDefaults",
    "SyncIntegrityDefaults",
    "TaskLifecycleDefaults",
    "TrainingDefaults",
    "TransportDefaults",
    "UtilizationDefaults",
    "WorkQueueMonitorDefaults",
    # Utility functions
    "get_aiohttp_timeout",
    "get_all_defaults",
    "get_backpressure_multiplier",
    "get_circuit_breaker_configs",
    "get_job_timeout",
    "get_p2p_port",
    "get_peer_timeout",  # December 28, 2025
    "get_sqlite_timeout",
    "get_ssh_timeout",  # December 28, 2025
    "get_timeout",
]

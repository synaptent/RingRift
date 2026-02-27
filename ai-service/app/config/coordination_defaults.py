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


def _env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment or use default."""
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes", "on")


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
    # Dec 30, 2025: Reduced from 300 to 60 for faster recovery during 48h autonomous ops
    # Matches CircuitBreakerDefaults.RECOVERY_TIMEOUT for consistency
    CIRCUIT_BREAKER_RECOVERY: int = _env_int("RINGRIFT_CIRCUIT_BREAKER_RECOVERY", 60)

    # SSH timeout for remote operations
    # Dec 2025: Increased from 30 to 60 for Tailscale userland stability
    SSH_TIMEOUT: int = _env_int("RINGRIFT_SSH_TIMEOUT", 60)

    # Maximum retries for failed operations
    MAX_RETRIES: int = _env_int("RINGRIFT_MAX_RETRIES", 3)


# =============================================================================
# Transfer Memory Defaults
# =============================================================================

@dataclass(frozen=True)
class TransferMemoryDefaults:
    """Memory thresholds for rsync fallback to aria2/HTTP.

    Feb 2026: When memory usage exceeds RSYNC_MEMORY_THRESHOLD on coordinator,
    rsync transfers fall back to aria2/HTTP which use less memory.
    """
    # Memory usage % above which rsync falls back to aria2/HTTP
    RSYNC_MEMORY_THRESHOLD: float = _env_float("RINGRIFT_RSYNC_MEMORY_THRESHOLD", 70.0)
    # Enable/disable memory-aware transfer fallback
    ENABLED: bool = _env_bool("RINGRIFT_MEMORY_AWARE_TRANSFER", True)
    # Only apply memory checks on coordinator nodes
    COORDINATOR_ONLY: bool = _env_bool("RINGRIFT_MEMORY_AWARE_COORDINATOR_ONLY", True)


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

    Jan 2026: Further increased for parallelism-by-default:
    - MAX_CONCURRENT_PER_HOST: 2 → 4 (modern GPUs and networks can handle more)
    - MAX_CONCURRENT_CLUSTER: 10 → 16 (better cluster-wide throughput)
    """
    # Sync lock timeout (seconds)
    LOCK_TIMEOUT: int = _env_int("RINGRIFT_SYNC_LOCK_TIMEOUT", 120)

    # Maximum concurrent syncs per host
    # Feb 2026: 4 → 1 to prevent OOM from parallel rsync processes
    MAX_CONCURRENT_PER_HOST: int = _env_int("RINGRIFT_MAX_SYNCS_PER_HOST", 1)

    # Maximum concurrent syncs cluster-wide
    # Feb 2026: 16 → 1 to prevent OOM from parallel rsync processes
    MAX_CONCURRENT_CLUSTER: int = _env_int("RINGRIFT_MAX_SYNCS_CLUSTER", 1)

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

    # Minimum nodes for fair evaluation.
    # Set to 1: coordinator runs evaluation locally with the candidate model.
    # Higher values block evaluation behind model distribution which times out.
    MIN_NODES_FOR_EVALUATION: int = _env_int("RINGRIFT_MIN_NODES_FOR_EVALUATION", 1)

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
    # Jan 5, 2026 (Phase 11.1): Increased from 3 to 5 for +40% training throughput
    # Jan 12, 2026: Increased from 5 to 8 for +60% training throughput
    # With 11 Lambda GH200 + 3 Nebius H100 nodes, 8 concurrent jobs matches capacity
    MAX_CONCURRENT_TOTAL: int = _env_int("RINGRIFT_MAX_TRAINING_TOTAL", 8)

    # Training job timeout (hours)
    TIMEOUT_HOURS: float = _env_float("RINGRIFT_TRAINING_TIMEOUT_HOURS", 24.0)

    # Minimum interval between training runs (seconds)
    MIN_INTERVAL: int = _env_int("RINGRIFT_TRAINING_MIN_INTERVAL", 1200)


# =============================================================================
# Data Freshness Defaults (December 2025)
# =============================================================================

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
    # December 29, 2025: Relaxed from 4.0 to 48.0 for 48-hour autonomous operation
    # This reduces training gate blocks by allowing older data when fresh data
    # isn't available, prioritizing training throughput
    MAX_DATA_AGE_HOURS: float = _env_float("RINGRIFT_MAX_DATA_AGE_HOURS", 48.0)

    # Warning threshold (hours) - emit DATA_STALE warning above this
    # December 29, 2025: Tightened from 8.0 to 1.0 for earlier warnings
    FRESHNESS_WARNING_HOURS: float = _env_float("RINGRIFT_FRESHNESS_WARNING_HOURS", 1.0)

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
    # Session 17.35: Reduced from 1000 to 800 to trigger backpressure earlier
    # This improves throughput by 5-8% by preventing queue saturation
    MAX_QUEUE_SIZE: int = _env_int("RINGRIFT_MAX_QUEUE_SIZE", 800)

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
    HALF_OPEN_MAX_CALLS: int = _env_int("RINGRIFT_CB_HALF_OPEN_MAX_CALLS", 2)

    # Per-transport type configs
    SSH_FAILURE_THRESHOLD: int = 5
    SSH_RECOVERY_TIMEOUT: float = 60.0

    HTTP_FAILURE_THRESHOLD: int = 5
    HTTP_RECOVERY_TIMEOUT: float = 30.0

    P2P_FAILURE_THRESHOLD: int = 5
    P2P_RECOVERY_TIMEOUT: float = 45.0

    ARIA2_FAILURE_THRESHOLD: int = 3
    ARIA2_RECOVERY_TIMEOUT: float = 120.0

    RSYNC_FAILURE_THRESHOLD: int = 3
    RSYNC_RECOVERY_TIMEOUT: float = 90.0

    # Per-provider configs (December 29, 2025)
    # Vast.ai nodes have higher network latency and more timeouts
    VAST_FAILURE_THRESHOLD: int = 6
    VAST_RECOVERY_TIMEOUT: float = 90.0

    # RunPod nodes are generally stable but can have cold-start latency
    RUNPOD_FAILURE_THRESHOLD: int = 4
    RUNPOD_RECOVERY_TIMEOUT: float = 60.0

    # Lambda nodes are most reliable (dedicated VMs)
    LAMBDA_FAILURE_THRESHOLD: int = 3
    LAMBDA_RECOVERY_TIMEOUT: float = 45.0

    # Nebius nodes have good reliability
    NEBIUS_FAILURE_THRESHOLD: int = 3
    NEBIUS_RECOVERY_TIMEOUT: float = 45.0

    # Vultr nodes are stable but limited bandwidth
    VULTR_FAILURE_THRESHOLD: int = 4
    VULTR_RECOVERY_TIMEOUT: float = 60.0

    # Hetzner CPU-only nodes - most reliable
    HETZNER_FAILURE_THRESHOLD: int = 3
    HETZNER_RECOVERY_TIMEOUT: float = 30.0


def get_circuit_breaker_for_provider(provider: str) -> dict[str, float | int]:
    """Get circuit breaker config for a cloud provider.

    Args:
        provider: Provider name (vast, runpod, lambda, nebius, vultr, hetzner)

    Returns:
        Dict with failure_threshold and recovery_timeout

    December 29, 2025: Added for provider-specific tuning to reduce false
    circuit opens on lossy networks (Vast.ai) while maintaining fast detection
    on reliable networks (Lambda, Nebius).
    """
    provider = provider.lower()
    defaults = CircuitBreakerDefaults

    configs = {
        "vast": {
            "failure_threshold": defaults.VAST_FAILURE_THRESHOLD,
            "recovery_timeout": defaults.VAST_RECOVERY_TIMEOUT,
        },
        "runpod": {
            "failure_threshold": defaults.RUNPOD_FAILURE_THRESHOLD,
            "recovery_timeout": defaults.RUNPOD_RECOVERY_TIMEOUT,
        },
        "lambda": {
            "failure_threshold": defaults.LAMBDA_FAILURE_THRESHOLD,
            "recovery_timeout": defaults.LAMBDA_RECOVERY_TIMEOUT,
        },
        "nebius": {
            "failure_threshold": defaults.NEBIUS_FAILURE_THRESHOLD,
            "recovery_timeout": defaults.NEBIUS_RECOVERY_TIMEOUT,
        },
        "vultr": {
            "failure_threshold": defaults.VULTR_FAILURE_THRESHOLD,
            "recovery_timeout": defaults.VULTR_RECOVERY_TIMEOUT,
        },
        "hetzner": {
            "failure_threshold": defaults.HETZNER_FAILURE_THRESHOLD,
            "recovery_timeout": defaults.HETZNER_RECOVERY_TIMEOUT,
        },
    }

    return configs.get(
        provider,
        {
            "failure_threshold": defaults.FAILURE_THRESHOLD,
            "recovery_timeout": defaults.RECOVERY_TIMEOUT,
        },
    )


# =============================================================================
# Cascade Breaker Defaults (December 30, 2025)
# =============================================================================


@dataclass(frozen=True)
class CascadeBreakerDefaults:
    """Default values for hierarchical cascade circuit breaker.

    Used by: app/coordination/cascade_breaker.py
    See: app/coordination/daemon_types.py for DaemonCategory enum

    The cascade breaker prevents daemon restart cascades from destabilizing
    the system. It provides per-category breakers with independent thresholds,
    plus critical daemon exemptions.

    Categories with exempt_from_global=True (EVENT, PIPELINE, FEEDBACK, AUTONOMOUS)
    can restart even when the global breaker is open.

    Environment variable overrides:
    - RINGRIFT_CASCADE_GLOBAL_THRESHOLD: Max total restarts before global trips
    - RINGRIFT_CASCADE_GLOBAL_COOLDOWN: Global cooldown in seconds
    - RINGRIFT_CASCADE_STARTUP_GRACE: Startup grace period in seconds
    - RINGRIFT_CASCADE_STARTUP_THRESHOLD: Threshold during startup
    """

    # Global breaker settings
    GLOBAL_THRESHOLD: int = _env_int("RINGRIFT_CASCADE_GLOBAL_THRESHOLD", 15)
    GLOBAL_WINDOW_SECONDS: int = 300  # 5 minutes
    GLOBAL_COOLDOWN_SECONDS: int = _env_int("RINGRIFT_CASCADE_GLOBAL_COOLDOWN", 120)

    # Startup grace period (higher threshold during initialization)
    STARTUP_GRACE_PERIOD: int = _env_int("RINGRIFT_CASCADE_STARTUP_GRACE", 180)
    STARTUP_THRESHOLD: int = _env_int("RINGRIFT_CASCADE_STARTUP_THRESHOLD", 50)

    # Per-category thresholds and cooldowns
    # Critical categories (exempt from global)
    EVENT_THRESHOLD: int = 10
    EVENT_COOLDOWN: int = 30

    PIPELINE_THRESHOLD: int = 8
    PIPELINE_COOLDOWN: int = 45

    FEEDBACK_THRESHOLD: int = 8
    FEEDBACK_COOLDOWN: int = 45

    AUTONOMOUS_THRESHOLD: int = 8
    AUTONOMOUS_COOLDOWN: int = 30

    # Standard categories
    SYNC_THRESHOLD: int = 6
    SYNC_COOLDOWN: int = 60

    HEALTH_THRESHOLD: int = 6
    HEALTH_COOLDOWN: int = 60

    QUEUE_THRESHOLD: int = 5
    QUEUE_COOLDOWN: int = 60

    RESOURCE_THRESHOLD: int = 5
    RESOURCE_COOLDOWN: int = 90

    # Less critical categories
    EVALUATION_THRESHOLD: int = 5
    EVALUATION_COOLDOWN: int = 90

    DISTRIBUTION_THRESHOLD: int = 4
    DISTRIBUTION_COOLDOWN: int = 90

    RECOVERY_THRESHOLD: int = 4
    RECOVERY_COOLDOWN: int = 120

    PROVIDER_THRESHOLD: int = 3
    PROVIDER_COOLDOWN: int = 120

    MISC_THRESHOLD: int = 4
    MISC_COOLDOWN: int = 120


# =============================================================================
# Partition Recovery Defaults (December 29, 2025)
# =============================================================================


@dataclass(frozen=True)
class PartitionRecoveryDefaults:
    """Default values for network partition recovery.

    Part of 48-hour autonomous operation optimization.
    Used by: scripts/p2p/loops/resilience_loops.py

    Previous assumption: Partitions heal within ~60s
    New assumption: Extended partitions can last hours; need graceful handling
    """

    # Expected healing time for network partitions (seconds)
    # 2 hours - generous to handle extended cloud provider issues
    HEALING_TIME_SECONDS: int = _env_int("RINGRIFT_PARTITION_HEALING_TIME", 7200)

    # Delay after partition heals before triggering resync (seconds)
    # Wait for quorum to stabilize before syncing
    RESYNC_DELAY_SECONDS: int = _env_int("RINGRIFT_PARTITION_RESYNC_DELAY", 60)

    # Minimum peers required for healthy cluster
    MIN_PEERS_FOR_HEALTHY: int = _env_int("RINGRIFT_MIN_PEERS_HEALTHY", 3)

    # Duration before emitting partition alert (seconds)
    # 30 minutes - enough to distinguish from transient issues
    PARTITION_ALERT_THRESHOLD: int = _env_int("RINGRIFT_PARTITION_ALERT_THRESHOLD", 1800)

    # Maximum retries for resync after partition heals
    MAX_RESYNC_RETRIES: int = 3

    # Backoff multiplier between resync retries
    RESYNC_BACKOFF_MULTIPLIER: float = 2.0


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
# Memory Pressure Defaults (January 2026)
# =============================================================================

@dataclass(frozen=True)
class MemoryPressureDefaults:
    """Default values for proactive memory pressure management.

    Used by: app/coordination/memory_pressure_controller.py

    The 4-tier graduated response prevents memory exhaustion by taking
    increasingly aggressive action as memory usage rises. This was added
    after a cluster failure where RAM reached 100% without triggering
    adequate response (Session 16 cluster resilience plan).

    Tier Progression:
        CAUTION (60%): Log warning, emit event for monitoring
        WARNING (70%): Pause new selfplay jobs, reduce batch sizes
        CRITICAL (80%): Kill non-essential daemons, trigger GC
        EMERGENCY (90%): Notify standby coordinator, graceful shutdown
    """

    # Tier thresholds (RAM percentage)
    TIER_CAUTION: int = _env_int("RINGRIFT_MEMORY_TIER_CAUTION", 60)
    TIER_WARNING: int = _env_int("RINGRIFT_MEMORY_TIER_WARNING", 70)
    TIER_CRITICAL: int = _env_int("RINGRIFT_MEMORY_TIER_CRITICAL", 80)
    TIER_EMERGENCY: int = _env_int("RINGRIFT_MEMORY_TIER_EMERGENCY", 90)

    # Monitoring interval (seconds)
    CHECK_INTERVAL: int = _env_int("RINGRIFT_MEMORY_CHECK_INTERVAL", 10)

    # Hysteresis - must drop this many % below threshold to recover
    # Prevents oscillation between tiers
    # Jan 2026: Reduced from 5% to 3% for faster tier recovery while still preventing oscillation
    HYSTERESIS: int = _env_int("RINGRIFT_MEMORY_HYSTERESIS", 3)

    # Cooldown after taking action (seconds)
    ACTION_COOLDOWN: int = _env_int("RINGRIFT_MEMORY_ACTION_COOLDOWN", 60)

    # How long to wait for GC effect before escalating (seconds)
    GC_WAIT_TIME: int = _env_int("RINGRIFT_MEMORY_GC_WAIT", 30)

    # Batch size reduction factor at WARNING tier
    BATCH_SIZE_REDUCTION: float = 0.5  # Reduce to 50% of normal

    # Number of consecutive samples above threshold before acting
    CONSECUTIVE_SAMPLES_REQUIRED: int = _env_int(
        "RINGRIFT_MEMORY_CONSECUTIVE_SAMPLES", 3
    )


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
    # Feb 2026: 3 → 1 to prevent OOM from parallel rsync processes
    MAX_CONCURRENT_TRANSFERS: int = _env_int("RINGRIFT_MAX_CONCURRENT_TRANSFERS", 1)

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

    Jan 19, 2026: Significantly lowered all limits to prevent node
    saturation and maintain P2P heartbeat responsiveness. Previous
    values (16-128) allowed nodes to hit 100% CPU with 5+ concurrent
    selfplay processes, blocking heartbeats.
    """
    # Maximum concurrent selfplay by GPU tier - lowered for P2P stability
    CONSUMER_MAX: int = _env_int("RINGRIFT_CONSUMER_MAX_SELFPLAY", 4)   # was 16
    PROSUMER_MAX: int = _env_int("RINGRIFT_PROSUMER_MAX_SELFPLAY", 8)   # was 32
    DATACENTER_MAX: int = _env_int("RINGRIFT_DATACENTER_MAX_SELFPLAY", 16)  # was 64
    HIGH_CPU_MAX: int = _env_int("RINGRIFT_HIGH_CPU_MAX_SELFPLAY", 24)  # was 128


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

    Jan 19, 2026: Lowered thresholds to prevent node saturation and
    maintain P2P heartbeat responsiveness. Previous high thresholds
    (85-90%) allowed nodes to hit 100% before throttling, causing
    heartbeat timeouts and cluster instability.
    """
    # Backpressure thresholds (%) - lowered to prevent node saturation
    BACKPRESSURE_GPU_THRESHOLD: float = _env_float("RINGRIFT_BACKPRESSURE_GPU_THRESHOLD", 70.0)  # was 90
    BACKPRESSURE_MEMORY_THRESHOLD: float = _env_float("RINGRIFT_BACKPRESSURE_MEMORY_THRESHOLD", 70.0)  # was 85
    BACKPRESSURE_DISK_THRESHOLD: float = _env_float("RINGRIFT_BACKPRESSURE_DISK_THRESHOLD", 75.0)  # Between DISK_SYNC_TARGET(70) and DISK_PRODUCTION_HALT(85)

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
    # Disk space warning thresholds (%) - aligned with app.config.thresholds canonical values
    DISK_WARNING_THRESHOLD: float = _env_float("RINGRIFT_DISK_WARNING_THRESHOLD", 70.0)   # = DISK_SYNC_TARGET_PERCENT
    DISK_CRITICAL_THRESHOLD: float = _env_float("RINGRIFT_DISK_CRITICAL_THRESHOLD", 90.0)  # = DISK_CRITICAL_PERCENT

    # Memory usage thresholds (%)
    MEMORY_WARNING_THRESHOLD: float = _env_float("RINGRIFT_MEMORY_WARNING_THRESHOLD", 80.0)
    MEMORY_CRITICAL_THRESHOLD: float = _env_float("RINGRIFT_MEMORY_CRITICAL_THRESHOLD", 95.0)

    # GPU memory thresholds (%)
    GPU_MEMORY_WARNING: float = _env_float("RINGRIFT_GPU_MEMORY_WARNING", 90.0)
    GPU_MEMORY_CRITICAL: float = _env_float("RINGRIFT_GPU_MEMORY_CRITICAL", 98.0)

    # Node offline detection (seconds)
    # Jan 13, 2026: Reduced from 300s to 60s for faster quorum detection during 48h autonomous ops
    NODE_OFFLINE_THRESHOLD: int = _env_int("RINGRIFT_NODE_OFFLINE_THRESHOLD", 60)

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


@dataclass(frozen=True)
class MetricsRetentionDefaults:
    """Default values for metrics database retention and cleanup.

    January 2026: Added to prevent unbounded SQLite database growth.

    Used by: scripts/p2p/metrics_manager.py
    """
    # Retention period for old metrics (hours)
    # Default 7 days = 168 hours
    RETENTION_HOURS: int = _env_int("RINGRIFT_METRICS_RETENTION_HOURS", 168)

    # Batch size for cleanup deletions (rows per batch)
    CLEANUP_BATCH_SIZE: int = _env_int("RINGRIFT_METRICS_CLEANUP_BATCH_SIZE", 1000)

    # VACUUM threshold - run VACUUM after deleting this % of total rows
    VACUUM_THRESHOLD_PERCENT: float = _env_float(
        "RINGRIFT_METRICS_VACUUM_THRESHOLD_PERCENT", 10.0
    )

    # Minimum interval between cleanups (seconds)
    # Prevents cleanup running too frequently
    CLEANUP_INTERVAL: int = _env_int("RINGRIFT_METRICS_CLEANUP_INTERVAL", 3600)


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
    # Dec 30, 2025: Reduced from 15 to 10 for faster partition detection (45s → 20s)
    HEARTBEAT_INTERVAL: int = _env_int("RINGRIFT_P2P_HEARTBEAT_INTERVAL", 10)

    # Voter heartbeat interval (seconds) - faster heartbeats for voter nodes
    # Dec 30, 2025: Added for quorum-critical voter nodes to detect failures faster
    VOTER_HEARTBEAT_INTERVAL: int = _env_int("RINGRIFT_P2P_VOTER_HEARTBEAT_INTERVAL", 5)

    # Peer timeout (seconds) - consider dead after no heartbeat
    # Jan 19, 2026: Increased from 30 to 90 to reduce false disconnections
    # Jan 22, 2026: Increased to 180 to align with app/p2p/constants.py for gossip convergence
    PEER_TIMEOUT: int = _env_int("RINGRIFT_P2P_PEER_TIMEOUT", 180)

    # NAT-blocked peer timeout (seconds) - longer timeout for Lambda/Vast NAT-blocked nodes
    # Jan 19, 2026: Added for nodes behind CGNAT or strict NAT (rely on relay/P2PD)
    # Jan 22, 2026: Increased to 180 to align with app/p2p/constants.py
    PEER_TIMEOUT_NAT_BLOCKED: int = _env_int("RINGRIFT_P2P_PEER_TIMEOUT_NAT_BLOCKED", 180)

    # Fast peer timeout (seconds) - for well-connected datacenter nodes
    # Jan 19, 2026: Added for Hetzner, Vultr nodes with direct connectivity
    PEER_TIMEOUT_FAST: int = _env_int("RINGRIFT_P2P_PEER_TIMEOUT_FAST", 60)

    # Leader election timeout (seconds)
    ELECTION_TIMEOUT: int = _env_int("RINGRIFT_P2P_ELECTION_TIMEOUT", 30)

    # Gossip fanout - number of random peers to gossip to each cycle
    # Jan 19, 2026: Increased from 5 to 8 for larger clusters (20+ nodes)
    GOSSIP_FANOUT: int = _env_int("RINGRIFT_P2P_GOSSIP_FANOUT", 8)

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
# P2P Recovery Defaults (January 2026)
# =============================================================================

@dataclass(frozen=True)
class P2PRecoveryDefaults:
    """Default values for P2P cluster recovery operations.

    Used by: app/coordination/p2p_recovery_daemon.py,
             scripts/p2p/loops/remote_p2p_recovery_loop.py

    January 2026: Centralized defaults for 48-hour autonomous operation.
    These values optimize P2P cluster connectivity by enabling faster
    recovery intervals and NAT-aware thresholds.
    """
    # Remote recovery loop interval (seconds) - how often to check for missing nodes
    # Jan 2026: Reduced from 300s to 60s for faster 48h autonomous recovery
    REMOTE_RECOVERY_INTERVAL: float = _env_float("RINGRIFT_REMOTE_P2P_RECOVERY_INTERVAL", 60.0)

    # Maximum nodes to recover per cycle (prevents thundering herd)
    # Jan 2026: Increased from 5 to 10 for faster cluster recovery
    MAX_NODES_PER_CYCLE: int = _env_int("RINGRIFT_REMOTE_P2P_MAX_NODES_PER_CYCLE", 10)

    # Retry cooldown for individual nodes (seconds)
    # Jan 2026: Reduced from 120s to 60s for faster NAT-blocked node recovery
    RETRY_COOLDOWN: float = _env_float("RINGRIFT_REMOTE_P2P_RETRY_COOLDOWN", 60.0)

    # Normal restart threshold (consecutive failures before restart)
    RESTART_THRESHOLD: int = _env_int("RINGRIFT_P2P_RESTART_THRESHOLD", 3)

    # NAT-blocked restart threshold (fewer failures needed for NAT nodes)
    NAT_BLOCKED_RESTART_THRESHOLD: int = _env_int("RINGRIFT_P2P_NAT_RESTART_THRESHOLD", 2)

    # Normal restart cooldown (seconds)
    RESTART_COOLDOWN: int = _env_int("RINGRIFT_P2P_RESTART_COOLDOWN", 300)

    # NAT-blocked restart cooldown (seconds) - faster for NAT nodes
    NAT_BLOCKED_COOLDOWN: int = _env_int("RINGRIFT_P2P_NAT_COOLDOWN", 60)

    # Minimum healthy relays required for NAT-blocked nodes
    MIN_HEALTHY_RELAYS: int = _env_int("RINGRIFT_P2P_MIN_HEALTHY_RELAYS", 3)

    # Verification timeout after recovery (seconds) - wait for node to appear in mesh
    VERIFICATION_TIMEOUT: float = _env_float("RINGRIFT_P2P_VERIFICATION_TIMEOUT", 60.0)

    # SSH timeout for recovery operations (seconds)
    SSH_TIMEOUT: float = _env_float("RINGRIFT_P2P_SSH_TIMEOUT", 30.0)


@dataclass(frozen=True)
class PartitionHealingDefaults:
    """Default values for partition healing automation.

    January 2026: Auto-triggered healing when network partitions are detected.

    Used by: scripts/p2p/partition_healer.py, scripts/p2p_orchestrator.py
    """
    # Minimum interval between healing passes (seconds)
    # Rate limit to prevent healing loops
    MIN_INTERVAL: float = _env_float("RINGRIFT_PARTITION_HEALING_MIN_INTERVAL", 300.0)

    # Timeout for individual peer queries during healing (seconds)
    PEER_TIMEOUT: float = _env_float("RINGRIFT_PARTITION_HEALING_PEER_TIMEOUT", 10.0)

    # Whether auto-healing is enabled
    AUTO_ENABLED: bool = _env_bool("RINGRIFT_PARTITION_HEALING_AUTO_ENABLED", True)

    # Delay before starting healing after partition detection (seconds)
    # Gives time for transient issues to resolve
    DETECTION_DELAY: float = _env_float("RINGRIFT_PARTITION_HEALING_DETECTION_DELAY", 30.0)

    # Maximum bridges to create per partition pair
    MAX_BRIDGES: int = _env_int("RINGRIFT_PARTITION_HEALING_MAX_BRIDGES", 3)

    # Sprint 4 (Jan 2, 2026): Convergence validation after healing
    # Timeout for waiting for gossip convergence after healing (seconds)
    CONVERGENCE_TIMEOUT: float = _env_float(
        "RINGRIFT_PARTITION_HEALING_CONVERGENCE_TIMEOUT", 120.0
    )

    # Convergence check interval (seconds)
    CONVERGENCE_CHECK_INTERVAL: float = _env_float(
        "RINGRIFT_PARTITION_HEALING_CONVERGENCE_CHECK_INTERVAL", 15.0
    )

    # Minimum agreement ratio across quorum for convergence (0.0-1.0)
    # Nodes must agree on at least this ratio of peers to consider converged
    CONVERGENCE_AGREEMENT_THRESHOLD: float = _env_float(
        "RINGRIFT_PARTITION_HEALING_CONVERGENCE_AGREEMENT", 0.8
    )

    # Sprint 10 (Jan 3, 2026): Recovery escalation on convergence failure
    # Number of consecutive convergence failures before escalation
    ESCALATION_THRESHOLD: int = _env_int(
        "RINGRIFT_PARTITION_HEALING_ESCALATION_THRESHOLD", 3
    )

    # Base wait time (seconds) before probe after failure (multiplied by escalation level)
    ESCALATION_BASE_WAIT: float = _env_float(
        "RINGRIFT_PARTITION_HEALING_ESCALATION_BASE_WAIT", 60.0
    )

    # Maximum escalation level (caps wait time growth)
    ESCALATION_MAX_LEVEL: int = _env_int(
        "RINGRIFT_PARTITION_HEALING_ESCALATION_MAX_LEVEL", 5
    )

    # Whether to emit P2P_RECOVERY_NEEDED event at max escalation
    EMIT_RECOVERY_EVENT_AT_MAX: bool = _env_bool(
        "RINGRIFT_PARTITION_HEALING_EMIT_RECOVERY_EVENT", True
    )

    # Jan 3, 2026: Overall timeout for entire healing pass (seconds)
    # Prevents healing operations from hanging indefinitely
    # Default: 300s (5 minutes) - enough for discovery + healing + convergence
    TOTAL_TIMEOUT: float = _env_float(
        "RINGRIFT_PARTITION_HEALING_TOTAL_TIMEOUT", 300.0
    )

    # Sprint 16.1 (Jan 3, 2026): Gossip propagation validation after T0 injection
    # Timeout for checking if injected peers are visible (per-batch, not full convergence)
    INJECTION_PROPAGATION_TIMEOUT: float = _env_float(
        "RINGRIFT_PARTITION_HEALING_INJECTION_PROPAGATION_TIMEOUT", 30.0
    )

    # Minimum ratio of target nodes that must see injected peers for success
    INJECTION_VISIBILITY_THRESHOLD: float = _env_float(
        "RINGRIFT_PARTITION_HEALING_INJECTION_VISIBILITY_THRESHOLD", 0.5
    )


# =============================================================================
# Export Validation Defaults (Sprint 4 - January 2, 2026)
# =============================================================================


@dataclass(frozen=True)
class ExportValidationDefaults:
    """Default values for pre-export validation checks.

    Sprint 4 (Jan 2, 2026): Validates game count and quality before NPZ export
    to prevent low-quality training data from entering the pipeline.

    Used by: app/coordination/auto_export_daemon.py
    """
    # Minimum games required before export is allowed
    # Feb 2026: Lowered from 100 to 25 — was higher than min_games_threshold (50),
    # causing validation to block exports that already passed the threshold check.
    # This produced 3,347 consecutive failures for hexagonal_4p.
    MIN_GAMES: int = _env_int("RINGRIFT_EXPORT_VALIDATION_MIN_GAMES", 25)

    # Minimum average quality score (0.0-1.0)
    # Games are quality-scored based on move diversity, game length, etc.
    MIN_AVG_QUALITY: float = _env_float("RINGRIFT_EXPORT_VALIDATION_MIN_QUALITY", 0.5)

    # Enable/disable validation entirely
    ENABLED: bool = _env_bool("RINGRIFT_EXPORT_VALIDATION_ENABLED", True)

    # Skip validation for bootstrap configs (less than MIN_BOOTSTRAP_GAMES games)
    # Bootstrap mode allows lower quality for initial training data
    BOOTSTRAP_MODE_THRESHOLD: int = _env_int(
        "RINGRIFT_EXPORT_VALIDATION_BOOTSTRAP_THRESHOLD", 500
    )

    # Allow export if either game count OR quality exceeds thresholds
    # False = require both, True = require either
    REQUIRE_BOTH: bool = _env_bool("RINGRIFT_EXPORT_VALIDATION_REQUIRE_BOTH", False)


# =============================================================================
# Frozen Leader Detection Defaults (January 2, 2026)
# =============================================================================


@dataclass(frozen=True)
class FrozenLeaderDefaults:
    """Default values for frozen leader detection.

    January 2, 2026: Detects leaders that heartbeat but can't accept work.

    Used by: scripts/p2p/leader_health.py, app/p2p/constants.py

    Problem: A leader may respond to /status (heartbeat) but fail to accept new work
    because its event loop is blocked (long-running sync, deadlock, stuck callback).
    Solution: Probe /admin/ping_work which requires the event loop to be responsive.
    """
    # Timeout for work acceptance probe (seconds)
    # Should be shorter than heartbeat interval to detect frozen state quickly
    PROBE_TIMEOUT: float = _env_float("RINGRIFT_P2P_FROZEN_LEADER_PROBE_TIMEOUT", 5.0)

    # How long a leader can fail work acceptance probes before triggering election
    # 300s = 5 minutes of unresponsive event loop
    TIMEOUT: int = _env_int("RINGRIFT_P2P_FROZEN_LEADER_TIMEOUT", 300)

    # Consecutive work acceptance failures before declaring leader frozen
    # Requires 3 failures to avoid false positives during transient load spikes
    CONSECUTIVE_FAILURES: int = _env_int("RINGRIFT_P2P_FROZEN_LEADER_CONSECUTIVE_FAILURES", 3)

    # Grace period after leader election before probing for frozen state
    # New leaders need time to initialize before being probed
    GRACE_PERIOD: int = _env_int("RINGRIFT_P2P_FROZEN_LEADER_GRACE_PERIOD", 60)

    # Whether frozen leader detection is enabled
    ENABLED: bool = _env_bool("RINGRIFT_P2P_FROZEN_LEADER_ENABLED", True)


# =============================================================================
# Endpoint Validation Defaults (December 30, 2025)
# =============================================================================

@dataclass(frozen=True)
class EndpointValidationDefaults:
    """Default values for P2P endpoint validation.

    Used by: scripts/p2p/gossip_protocol.py, scripts/p2p_orchestrator.py

    December 30, 2025: Added to prevent P2P partitions from stale IPs.
    When nodes change networks or containers restart, their IPs may change.
    Proactive endpoint validation detects stale IPs and tries alternates.

    Key insight: 5-node isolation was caused by nodes advertising private IPs
    that became unreachable after network changes. This validation catches
    those stale endpoints and triggers alternate IP probing.
    """
    # How long before an endpoint is considered stale (seconds)
    # After this time without successful heartbeat, we probe alternates
    ENDPOINT_TTL: int = _env_int("RINGRIFT_ENDPOINT_TTL", 300)  # 5 minutes

    # How often to run stale endpoint validation (seconds)
    VALIDATION_INTERVAL: int = _env_int("RINGRIFT_ENDPOINT_VALIDATION_INTERVAL", 60)

    # Maximum peers to validate per cycle (avoid thundering herd)
    MAX_VALIDATIONS_PER_CYCLE: int = _env_int("RINGRIFT_MAX_VALIDATIONS_PER_CYCLE", 5)

    # Timeout for endpoint probe (seconds)
    PROBE_TIMEOUT: float = _env_float("RINGRIFT_ENDPOINT_PROBE_TIMEOUT", 5.0)

    # Enable/disable endpoint validation
    ENABLED: bool = _env_bool("RINGRIFT_ENDPOINT_VALIDATION_ENABLED", True)


# =============================================================================
# P2P Protocol Defaults - SWIM/Raft (December 2025)
# =============================================================================

@dataclass(frozen=True)
class P2PProtocolDefaults:
    """Default values for P2P protocol configuration (SWIM/Raft).

    Used by: scripts/p2p_orchestrator.py, scripts/p2p/membership_mixin.py,
             scripts/p2p/consensus_mixin.py, app/p2p/hybrid_coordinator.py

    SWIM Protocol: Gossip-based membership with 5s failure detection
    Raft Protocol: Replicated state machine for work queue consensus

    Note: SWIM and Raft are optional upgrades. Production default is HTTP polling
    for membership and Bully election for consensus. Enable SWIM/Raft only after
    installing dependencies (swim-p2p>=1.2.0, pysyncobj>=0.3.14).
    """
    # ===========================================================================
    # Protocol Mode Selection
    # ===========================================================================

    # SWIM protocol enable flag (requires swim-p2p package)
    SWIM_ENABLED: bool = _env_bool("RINGRIFT_SWIM_ENABLED", False)

    # Raft protocol enable flag (requires pysyncobj package)
    RAFT_ENABLED: bool = _env_bool("RINGRIFT_RAFT_ENABLED", False)

    # Membership mode: "http" (polling), "swim" (gossip), "hybrid" (both)
    MEMBERSHIP_MODE: str = os.getenv("RINGRIFT_MEMBERSHIP_MODE", "http")

    # Consensus mode: "bully" (election), "raft" (replicated), "hybrid" (both)
    CONSENSUS_MODE: str = os.getenv("RINGRIFT_CONSENSUS_MODE", "bully")

    # ===========================================================================
    # SWIM Protocol Settings
    # ===========================================================================
    # December 30, 2025: Tuned for large clusters (~40 nodes).
    # Previous 1-second ping interval caused 40 pings/sec cluster-wide.
    # New 5-second interval reduces to 8 pings/sec while maintaining
    # failure detection under 30 seconds.

    # SWIM suspicion timeout (seconds) - time to suspect before declaring dead
    # December 30, 2025: Increased from 5.0 to 15.0 for reduced false positives
    SWIM_SUSPICION_TIMEOUT: float = _env_float("RINGRIFT_SWIM_SUSPICION_TIMEOUT", 15.0)

    # SWIM ping interval (seconds) - how often to ping random peers
    # December 30, 2025: Increased from 1.0 to 5.0 to reduce network traffic
    SWIM_PING_INTERVAL: float = _env_float("RINGRIFT_SWIM_PING_INTERVAL", 5.0)

    # SWIM ping timeout (seconds) - direct ping timeout before indirect probing
    SWIM_PING_TIMEOUT: float = _env_float("RINGRIFT_SWIM_PING_TIMEOUT", 0.5)

    # SWIM indirect ping count - number of peers to use for indirect probing
    SWIM_INDIRECT_PING_COUNT: int = _env_int("RINGRIFT_SWIM_INDIRECT_PING_COUNT", 3)

    # SWIM gossip fanout - number of peers to gossip to per round
    SWIM_GOSSIP_FANOUT: int = _env_int("RINGRIFT_SWIM_GOSSIP_FANOUT", 3)

    # SWIM message retransmit limit - max times to retransmit update
    SWIM_RETRANSMIT_LIMIT: int = _env_int("RINGRIFT_SWIM_RETRANSMIT_LIMIT", 3)

    # ===========================================================================
    # Raft Protocol Settings
    # ===========================================================================

    # Raft election timeout base (seconds) - randomized between 1x and 2x
    RAFT_ELECTION_TIMEOUT: float = _env_float("RINGRIFT_RAFT_ELECTION_TIMEOUT", 1.0)

    # Raft heartbeat interval (seconds) - leader heartbeat frequency
    RAFT_HEARTBEAT_INTERVAL: float = _env_float("RINGRIFT_RAFT_HEARTBEAT_INTERVAL", 0.3)

    # Raft append entries batch size - max entries per RPC
    RAFT_BATCH_SIZE: int = _env_int("RINGRIFT_RAFT_BATCH_SIZE", 100)

    # Raft log compaction threshold - entries before snapshot
    RAFT_COMPACTION_THRESHOLD: int = _env_int("RINGRIFT_RAFT_COMPACTION_THRESHOLD", 10000)

    # Raft snapshot chunk size (bytes) - for large state transfers
    RAFT_SNAPSHOT_CHUNK_SIZE: int = _env_int("RINGRIFT_RAFT_SNAPSHOT_CHUNK_SIZE", 1048576)

    # ===========================================================================
    # Adaptive Timeout Settings
    # ===========================================================================

    # Enable adaptive timeouts based on network latency
    ADAPTIVE_TIMEOUTS_ENABLED: bool = _env_bool("RINGRIFT_ADAPTIVE_TIMEOUTS", True)

    # Minimum timeout multiplier (prevents timeouts from being too aggressive)
    TIMEOUT_MIN_MULTIPLIER: float = _env_float("RINGRIFT_TIMEOUT_MIN_MULTIPLIER", 0.5)

    # Maximum timeout multiplier (prevents timeouts from being too lax)
    TIMEOUT_MAX_MULTIPLIER: float = _env_float("RINGRIFT_TIMEOUT_MAX_MULTIPLIER", 3.0)

    # Latency measurement window (seconds) - rolling window for RTT stats
    LATENCY_WINDOW_SECONDS: float = _env_float("RINGRIFT_LATENCY_WINDOW", 60.0)

    # Latency percentile for timeout calculation (0-100)
    LATENCY_PERCENTILE: float = _env_float("RINGRIFT_LATENCY_PERCENTILE", 95.0)

    # ===========================================================================
    # Hybrid Mode Settings
    # ===========================================================================

    # Hybrid mode: Use SWIM for fast failure detection, HTTP for stability
    HYBRID_SWIM_WEIGHT: float = _env_float("RINGRIFT_HYBRID_SWIM_WEIGHT", 0.7)
    HYBRID_HTTP_WEIGHT: float = _env_float("RINGRIFT_HYBRID_HTTP_WEIGHT", 0.3)

    # Hybrid mode: Raft for work queue, Bully for leader (simpler)
    HYBRID_RAFT_FALLBACK_ENABLED: bool = _env_bool("RINGRIFT_RAFT_FALLBACK", True)

    # ===========================================================================
    # Quorum and Split-Brain Settings
    # ===========================================================================

    # Minimum voter quorum for critical operations
    MIN_VOTER_QUORUM: int = _env_int("RINGRIFT_MIN_VOTER_QUORUM", 3)

    # Split-brain detection threshold (seconds without consensus)
    SPLIT_BRAIN_DETECTION_TIMEOUT: float = _env_float(
        "RINGRIFT_SPLIT_BRAIN_TIMEOUT", 30.0
    )

    # Split-brain auto-resolution: step down if minority partition
    SPLIT_BRAIN_AUTO_STEPDOWN: bool = _env_bool("RINGRIFT_SPLIT_BRAIN_STEPDOWN", True)


def get_membership_mode() -> str:
    """Get the configured membership mode.

    Returns:
        "http", "swim", or "hybrid"

    Example:
        mode = get_membership_mode()
        if mode == "swim":
            start_swim_protocol()
    """
    return P2PProtocolDefaults.MEMBERSHIP_MODE


def get_consensus_mode() -> str:
    """Get the configured consensus mode.

    Returns:
        "bully", "raft", or "hybrid"

    Example:
        mode = get_consensus_mode()
        if mode == "raft":
            start_raft_cluster()
    """
    return P2PProtocolDefaults.CONSENSUS_MODE


def is_swim_enabled() -> bool:
    """Check if SWIM protocol is enabled.

    Returns:
        True if SWIM is enabled and membership mode uses SWIM
    """
    return P2PProtocolDefaults.SWIM_ENABLED and P2PProtocolDefaults.MEMBERSHIP_MODE in (
        "swim",
        "hybrid",
    )


def is_raft_enabled() -> bool:
    """Check if Raft protocol is enabled.

    Returns:
        True if Raft is enabled and consensus mode uses Raft
    """
    return P2PProtocolDefaults.RAFT_ENABLED and P2PProtocolDefaults.CONSENSUS_MODE in (
        "raft",
        "hybrid",
    )


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

    # Disk thresholds (%) - aligned with app.config.thresholds canonical values
    DISK_WARNING_THRESHOLD: int = _env_int("RINGRIFT_DISK_WARNING_THRESHOLD", 85)   # = DISK_PRODUCTION_HALT_PERCENT
    DISK_CRITICAL_THRESHOLD: int = _env_int("RINGRIFT_DISK_CRITICAL_THRESHOLD", 90)  # = DISK_CRITICAL_PERCENT
    DISK_CLEANUP_THRESHOLD: int = _env_int("RINGRIFT_DISK_CLEANUP_THRESHOLD", 85)   # = DISK_PRODUCTION_HALT_PERCENT

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
    # Jan 2026: Reduced from 60s to 45s for faster signal response
    CHECK_INTERVAL: float = _env_float("RINGRIFT_DAEMON_HEALTH_INTERVAL", 45.0)

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

    # Dec 29, 2025: Health check parallelization settings
    # Individual daemon health check timeout (seconds)
    # Previously hardcoded as 5.0 in daemon_manager.py:2418
    HEALTH_CHECK_TIMEOUT: float = _env_float("RINGRIFT_DAEMON_HEALTH_CHECK_TIMEOUT", 5.0)

    # Enable parallel health checks (run all daemon checks concurrently)
    # Reduces health loop time from O(n*t) to O(t) where n=daemons, t=timeout
    PARALLEL_HEALTH_CHECKS: bool = _env_bool("RINGRIFT_DAEMON_PARALLEL_HEALTH", True)

    # Maximum concurrent health checks (to prevent overwhelming the system)
    # Set to 0 for unlimited (all daemons checked in parallel)
    MAX_PARALLEL_HEALTH_CHECKS: int = _env_int("RINGRIFT_DAEMON_MAX_PARALLEL_HEALTH", 20)

    # Dependency poll interval (seconds) - how often to poll when waiting for dependency
    # Previously hardcoded as 0.5 in daemon_manager.py:1218
    DEPENDENCY_POLL_INTERVAL: float = _env_float("RINGRIFT_DAEMON_DEPENDENCY_POLL", 0.1)

    # Dependency wait timeout (seconds) - max time to wait for a dependency
    # Previously hardcoded as 30.0 in daemon_manager.py:1193
    # Session 17.48: Increased from 30s to 90s to allow slower daemons to initialize
    DEPENDENCY_WAIT_TIMEOUT: float = _env_float("RINGRIFT_DAEMON_DEPENDENCY_TIMEOUT", 90.0)

    # Jan 3, 2026 (Sprint 15.1): Adaptive health check timeout settings
    # Base timeout used when system is idle/low load
    ADAPTIVE_TIMEOUT_BASE: float = _env_float("RINGRIFT_HEALTH_TIMEOUT_BASE", 5.0)

    # Timeout when system is under moderate load
    ADAPTIVE_TIMEOUT_MODERATE: float = _env_float("RINGRIFT_HEALTH_TIMEOUT_MODERATE", 10.0)

    # Timeout when system is under high load
    ADAPTIVE_TIMEOUT_HIGH: float = _env_float("RINGRIFT_HEALTH_TIMEOUT_HIGH", 15.0)

    # CPU threshold (fraction 0-1) for moderate load
    # Jan 19, 2026: Lowered from 0.6 to 0.5 for earlier backpressure response
    CPU_THRESHOLD_MODERATE: float = _env_float("RINGRIFT_CPU_THRESHOLD_MODERATE", 0.5)

    # CPU threshold (fraction 0-1) for high load
    # Jan 19, 2026: Lowered from 0.85 to 0.7 to prevent node saturation
    CPU_THRESHOLD_HIGH: float = _env_float("RINGRIFT_CPU_THRESHOLD_HIGH", 0.7)


def get_adaptive_health_timeout() -> float:
    """Get adaptive health check timeout based on current system load.

    Jan 3, 2026 (Sprint 15.1): Health checks may fail under load due to
    5s timeout being too aggressive. This function returns a timeout
    that scales with system load:

    - Idle/low load (CPU < 50%): 5s (default)
    - Moderate load (CPU 50-70%): 10s
    - High load (CPU > 70%): 15s

    Jan 19, 2026: Lowered thresholds from 60/85 to 50/70 to detect
    load earlier and prevent P2P heartbeat failures.

    Returns:
        Appropriate timeout value in seconds.

    Note:
        Requires psutil for CPU monitoring. Falls back to base timeout
        if psutil is unavailable or on error.
    """
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0

        if cpu_percent >= DaemonHealthDefaults.CPU_THRESHOLD_HIGH:
            return DaemonHealthDefaults.ADAPTIVE_TIMEOUT_HIGH
        elif cpu_percent >= DaemonHealthDefaults.CPU_THRESHOLD_MODERATE:
            return DaemonHealthDefaults.ADAPTIVE_TIMEOUT_MODERATE
        else:
            return DaemonHealthDefaults.ADAPTIVE_TIMEOUT_BASE
    except ImportError:
        # psutil not available - use base timeout
        return DaemonHealthDefaults.ADAPTIVE_TIMEOUT_BASE
    except Exception:
        # Any error - use base timeout
        return DaemonHealthDefaults.ADAPTIVE_TIMEOUT_BASE


# =============================================================================
# Degraded Mode Defaults (December 2025 - 48-Hour Autonomous Operation)
# =============================================================================

@dataclass(frozen=True)
class DegradedModeDefaults:
    """Default values for daemon graceful degradation.

    December 2025: Part of 48-hour autonomous operation plan.
    Instead of blocking daemons for 24 hours when restart limits are exceeded,
    we use tiered restart policies with degraded mode.

    Restart Tiers:
    - NORMAL (1-5 restarts): Standard exponential backoff (5s → 80s)
    - ELEVATED (6-10 restarts): Extended backoff (160s → 320s)
    - DEGRADED (>10 restarts): Keep retrying with longer intervals
      - Critical daemons: 30 min retry interval
      - Non-critical daemons: 4 hour retry interval

    Used by: app/coordination/daemon_manager.py
    """
    # Retry interval for critical daemons in degraded mode (seconds)
    # Critical daemons: EVENT_ROUTER, DATA_PIPELINE, AUTO_SYNC, FEEDBACK_LOOP,
    #                   P2P_BACKEND, QUEUE_POPULATOR, DAEMON_WATCHDOG
    CRITICAL_RETRY_INTERVAL: float = _env_float(
        "RINGRIFT_DEGRADED_CRITICAL_RETRY_INTERVAL", 1800.0  # 30 minutes
    )

    # Retry interval for non-critical daemons in degraded mode (seconds)
    NONCRITICAL_RETRY_INTERVAL: float = _env_float(
        "RINGRIFT_DEGRADED_NONCRITICAL_RETRY_INTERVAL", 14400.0  # 4 hours
    )

    # Restart thresholds for tier transitions
    NORMAL_MAX_RESTARTS: int = _env_int("RINGRIFT_DEGRADED_NORMAL_MAX", 5)
    ELEVATED_MAX_RESTARTS: int = _env_int("RINGRIFT_DEGRADED_ELEVATED_MAX", 10)

    # Backoff delays for each tier (seconds)
    # Normal tier: exponential from 5s to 80s
    NORMAL_BACKOFF_BASE: float = _env_float("RINGRIFT_DEGRADED_NORMAL_BACKOFF_BASE", 5.0)
    NORMAL_BACKOFF_MAX: float = _env_float("RINGRIFT_DEGRADED_NORMAL_BACKOFF_MAX", 80.0)

    # Elevated tier: extended backoff from 160s to 320s
    ELEVATED_BACKOFF_BASE: float = _env_float("RINGRIFT_DEGRADED_ELEVATED_BACKOFF_BASE", 160.0)
    ELEVATED_BACKOFF_MAX: float = _env_float("RINGRIFT_DEGRADED_ELEVATED_BACKOFF_MAX", 320.0)

    # Whether degraded mode is enabled (can disable for strict mode)
    ENABLED: bool = _env_bool("RINGRIFT_DEGRADED_MODE_ENABLED", True)

    # Hours before degraded daemon resets to normal (auto-recovery)
    RESET_AFTER_HOURS: float = _env_float("RINGRIFT_DEGRADED_RESET_HOURS", 2.0)


# =============================================================================
# Stale Data Fallback Defaults (December 2025 - 48-hour autonomous operation)
# =============================================================================

@dataclass(frozen=True)
class StaleFallbackDefaults:
    """Default values for training stale data fallback.

    December 2025: Part of 48-hour autonomous operation plan.
    When sync failures block training indefinitely, allow training to proceed
    with stale data after N failures or timeout.

    Problem: Sync failures can block training indefinitely, freezing progress.
    Solution: After configurable failures or timeout, allow training with
    older data while continuing to attempt sync in background.

    Jan 21, 2026: Phase 5 - Added tiered warnings before fallback.
    - Tier 1 (NOTICE): 15 min OR 2 failures - log info
    - Tier 2 (WARNING): 30 min OR 4 failures - emit warning event
    - Tier 3 (CRITICAL): 45 min OR 6 failures - allow fallback
    Rollback: RINGRIFT_STALE_FALLBACK_LEGACY=true (uses 45min OR 5 failures)

    Used by: app/coordination/stale_fallback.py, app/training/train.py
    """
    # Jan 21, 2026: Phase 5 - Tier 1: NOTICE (early warning)
    TIER_1_DURATION: float = _env_float("RINGRIFT_STALE_TIER1_DURATION", 900.0)  # 15 min
    TIER_1_FAILURES: int = _env_int("RINGRIFT_STALE_TIER1_FAILURES", 2)

    # Jan 21, 2026: Phase 5 - Tier 2: WARNING (escalated warning)
    TIER_2_DURATION: float = _env_float("RINGRIFT_STALE_TIER2_DURATION", 1800.0)  # 30 min
    TIER_2_FAILURES: int = _env_int("RINGRIFT_STALE_TIER2_FAILURES", 4)

    # Jan 21, 2026: Phase 5 - Tier 3: CRITICAL (fallback allowed)
    # Same as MAX_SYNC_DURATION and MAX_SYNC_FAILURES for backward compatibility
    TIER_3_DURATION: float = _env_float("RINGRIFT_STALE_TIER3_DURATION", 2700.0)  # 45 min
    TIER_3_FAILURES: int = _env_int("RINGRIFT_STALE_TIER3_FAILURES", 6)  # Was 5, increased to 6

    # Maximum sync failures before allowing stale fallback
    # Jan 21, 2026: Increased from 5 to 6 to match TIER_3_FAILURES
    MAX_SYNC_FAILURES: int = _env_int("RINGRIFT_MAX_SYNC_FAILURES", 6)

    # Maximum sync duration before fallback (seconds) - 45 minutes
    MAX_SYNC_DURATION: float = _env_float("RINGRIFT_MAX_SYNC_DURATION", 2700.0)

    # Absolute maximum data age allowed even in fallback mode (hours)
    # Training will NEVER proceed if data is older than this
    # December 29, 2025: Increased from 24h to 48h for extended autonomous runs
    ABSOLUTE_MAX_DATA_AGE: float = _env_float("RINGRIFT_ABSOLUTE_MAX_DATA_AGE", 48.0)

    # Whether stale fallback is enabled (can disable for strict mode)
    ENABLE_STALE_FALLBACK: bool = _env_bool("RINGRIFT_ENABLE_STALE_FALLBACK", True)

    # Minimum games required for fallback to proceed
    MIN_GAMES_FOR_FALLBACK: int = _env_int("RINGRIFT_MIN_GAMES_FOR_FALLBACK", 1000)

    # Cooldown between fallback attempts (seconds) - prevent rapid retries
    FALLBACK_COOLDOWN: float = _env_float("RINGRIFT_FALLBACK_COOLDOWN", 300.0)

    # Whether to emit events when falling back to stale data
    EMIT_FALLBACK_EVENTS: bool = _env_bool("RINGRIFT_EMIT_FALLBACK_EVENTS", True)

    # Jan 21, 2026: Phase 5 - Legacy mode flag (disables tier system)
    # When true, uses original 45min OR 5 failures without intermediate warnings
    LEGACY_MODE: bool = _env_bool("RINGRIFT_STALE_FALLBACK_LEGACY", False)


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
    # December 29, 2025: Increased default from 5s to 30s to match thresholds.py
    # This prevents lock timeout errors during concurrent cluster operations
    BUSY_TIMEOUT_MS: int = _env_int("RINGRIFT_SQLITE_BUSY_TIMEOUT_MS", 30000)


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

    # January 2026: Exponential backoff for job reassignment
    # Prevents rapid retry cycles when jobs consistently fail
    REASSIGN_BACKOFF_BASE: float = _env_float("RINGRIFT_REASSIGN_BACKOFF_BASE", 30.0)
    REASSIGN_BACKOFF_MULTIPLIER: float = _env_float("RINGRIFT_REASSIGN_BACKOFF_MULTIPLIER", 2.0)
    REASSIGN_BACKOFF_MAX: float = _env_float("RINGRIFT_REASSIGN_BACKOFF_MAX", 300.0)

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
# Work Queue Cleanup Defaults (January 15, 2026)
# =============================================================================

@dataclass(frozen=True)
class WorkQueueCleanupDefaults:
    """Default values for work queue stale item cleanup.

    SINGLE SOURCE OF TRUTH for cleanup thresholds used by:
    - scripts/p2p/loops/job_loops.py (WorkQueueMaintenanceLoop)
    - app/coordination/maintenance_daemon.py (MaintenanceDaemon)
    - app/coordination/job_reaper.py (JobReaper)
    - app/coordination/work_queue.py (cleanup_stale_items)

    January 2026: Standardized to 2 hours for CLAIMED (was inconsistent 1-4h).
    """
    # Max age for PENDING items before removal (hours)
    # Items stuck in PENDING for this long are likely invalid configs
    # Feb 2026: Lowered from 24h to 8h — stale items degrade claim_work() O(n) perf
    MAX_PENDING_AGE_HOURS: float = _env_float(
        "RINGRIFT_QUEUE_MAX_PENDING_AGE_HOURS", 8.0
    )

    # Max age for CLAIMED items before reset to PENDING (hours)
    # Items claimed but not started within this window are orphaned
    # Feb 2026: Lowered from 2h to 1h — training starts within minutes or not at all
    MAX_CLAIMED_AGE_HOURS: float = _env_float(
        "RINGRIFT_QUEUE_MAX_CLAIMED_AGE_HOURS", 1.0
    )

    # Cleanup check interval (seconds)
    CLEANUP_INTERVAL_SECONDS: float = _env_float(
        "RINGRIFT_QUEUE_CLEANUP_INTERVAL", 300.0  # 5 minutes
    )


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

    # Disk usage warning threshold (%) - aligned with app.config.thresholds: DISK_CRITICAL_PERCENT
    DISK_WARNING_THRESHOLD: int = _env_int("RINGRIFT_DISK_WARNING_THRESHOLD", 90)

    # Memory usage warning threshold (%)
    MEMORY_WARNING_THRESHOLD: int = _env_int("RINGRIFT_MEMORY_WARNING_THRESHOLD", 95)

    # Default MCTS budget for large boards
    LARGE_BOARD_BUDGET: int = _env_int("RINGRIFT_LARGE_BOARD_MCTS_BUDGET", 800)


# =============================================================================
# Selfplay Priority Weight Defaults (December 29, 2025)
# =============================================================================

@dataclass(frozen=True)
class SelfplayPriorityWeightDefaults:
    """Priority weight constants for selfplay allocation.

    These weights control how selfplay jobs are prioritized across configs.
    Environment variables allow runtime tuning without code changes.

    Used by: selfplay_scheduler.py (DynamicWeights class)

    Example:
        # Boost staleness weight for faster data refresh
        export RINGRIFT_STALENESS_WEIGHT=0.40

        # Reduce velocity weight when most configs are at target Elo
        export RINGRIFT_ELO_VELOCITY_WEIGHT=0.15
    """
    # Priority calculation weights (baseline values, adjusted dynamically)
    # Jan 2026: Reduced STALENESS_WEIGHT and ELO_VELOCITY_WEIGHT to prevent
    # freshness bias that was causing underserved configs to get < 100 games/day
    # Jan 5, 2026: Increased STALENESS_WEIGHT 0.15→0.25 to improve node utilization
    # (targeting 85%+ cluster utilization, was 40-60% with 0.15)
    STALENESS_WEIGHT: float = _env_float("RINGRIFT_STALENESS_WEIGHT", 0.25)
    # Jan 7, 2026: Increased from 0.10 to 0.15 to prioritize configs with fastest Elo improvement
    ELO_VELOCITY_WEIGHT: float = _env_float("RINGRIFT_ELO_VELOCITY_WEIGHT", 0.15)
    TRAINING_NEED_WEIGHT: float = _env_float("RINGRIFT_TRAINING_NEED_WEIGHT", 0.10)
    EXPLORATION_BOOST_WEIGHT: float = _env_float("RINGRIFT_EXPLORATION_BOOST_WEIGHT", 0.10)
    # Feb 2026: Swapped CURRICULUM_WEIGHT and DATA_DEFICIT_WEIGHT.
    # With idle detection fixed (41e49eadc), GPU nodes are now active.
    # The old 0.40 curriculum weight caused hex8/square8 to monopolize allocation
    # while square19 got only 5% of games. Boosting data_deficit helps underserved
    # configs (hexagonal_4p: 318 games, square19_4p: 592) catch up.
    CURRICULUM_WEIGHT: float = _env_float("RINGRIFT_CURRICULUM_WEIGHT", 0.25)
    IMPROVEMENT_BOOST_WEIGHT: float = _env_float("RINGRIFT_IMPROVEMENT_BOOST_WEIGHT", 0.15)
    DATA_DEFICIT_WEIGHT: float = _env_float("RINGRIFT_DATA_DEFICIT_WEIGHT", 0.40)
    QUALITY_WEIGHT: float = _env_float("RINGRIFT_QUALITY_WEIGHT", 0.15)
    VOI_WEIGHT: float = _env_float("RINGRIFT_VOI_WEIGHT", 0.20)
    # January 2026 Sprint 10: Diversity weight for maximizing opponent variety
    # Prioritizes configs with less opponent variety to improve training robustness
    DIVERSITY_WEIGHT: float = _env_float("RINGRIFT_DIVERSITY_WEIGHT", 0.10)

    # Dynamic weight bounds - prevent any single factor from dominating
    STALENESS_WEIGHT_MIN: float = _env_float("RINGRIFT_STALENESS_WEIGHT_MIN", 0.15)
    STALENESS_WEIGHT_MAX: float = _env_float("RINGRIFT_STALENESS_WEIGHT_MAX", 0.50)
    VELOCITY_WEIGHT_MIN: float = _env_float("RINGRIFT_VELOCITY_WEIGHT_MIN", 0.10)
    VELOCITY_WEIGHT_MAX: float = _env_float("RINGRIFT_VELOCITY_WEIGHT_MAX", 0.30)
    CURRICULUM_WEIGHT_MIN: float = _env_float("RINGRIFT_CURRICULUM_WEIGHT_MIN", 0.05)
    # Session 17.42: Increased max 0.25→0.50 to allow higher curriculum influence
    CURRICULUM_WEIGHT_MAX: float = _env_float("RINGRIFT_CURRICULUM_WEIGHT_MAX", 0.50)
    DATA_DEFICIT_WEIGHT_MIN: float = _env_float("RINGRIFT_DATA_DEFICIT_WEIGHT_MIN", 0.15)
    DATA_DEFICIT_WEIGHT_MAX: float = _env_float("RINGRIFT_DATA_DEFICIT_WEIGHT_MAX", 0.40)
    QUALITY_WEIGHT_MIN: float = _env_float("RINGRIFT_QUALITY_WEIGHT_MIN", 0.05)
    QUALITY_WEIGHT_MAX: float = _env_float("RINGRIFT_QUALITY_WEIGHT_MAX", 0.25)
    VOI_WEIGHT_MIN: float = _env_float("RINGRIFT_VOI_WEIGHT_MIN", 0.10)
    VOI_WEIGHT_MAX: float = _env_float("RINGRIFT_VOI_WEIGHT_MAX", 0.35)
    # January 2026 Sprint 10: Diversity weight bounds
    DIVERSITY_WEIGHT_MIN: float = _env_float("RINGRIFT_DIVERSITY_WEIGHT_MIN", 0.05)
    DIVERSITY_WEIGHT_MAX: float = _env_float("RINGRIFT_DIVERSITY_WEIGHT_MAX", 0.20)

    # Cluster state thresholds for weight adjustment triggers
    IDLE_GPU_HIGH_THRESHOLD: float = _env_float("RINGRIFT_IDLE_GPU_HIGH_THRESHOLD", 0.50)
    IDLE_GPU_LOW_THRESHOLD: float = _env_float("RINGRIFT_IDLE_GPU_LOW_THRESHOLD", 0.10)
    TRAINING_QUEUE_HIGH_THRESHOLD: int = _env_int("RINGRIFT_TRAINING_QUEUE_HIGH_THRESHOLD", 10)
    CONFIGS_AT_TARGET_THRESHOLD: float = _env_float("RINGRIFT_CONFIGS_AT_TARGET_THRESHOLD", 0.50)
    ELO_HIGH_THRESHOLD: int = _env_int("RINGRIFT_ELO_HIGH_THRESHOLD", 1800)
    ELO_MEDIUM_THRESHOLD: int = _env_int("RINGRIFT_ELO_MEDIUM_THRESHOLD", 1500)

    # Data starvation emergency thresholds
    # Dec 29, 2025: Added ULTRA tier for critically starved configs (< 20 games)
    # Jan 5, 2026 (Session 17.32): Expanded thresholds to catch more underserved configs
    # - ULTRA: 20 → 500 (hex8_3p/4p, hexagonal_3p/4p, square19_3p/4p have <1000 games)
    # - EMERGENCY: 100 → 1500
    # - CRITICAL: 500 → 3000
    # This ensures configs below 3000 games get significant priority boosts
    DATA_STARVATION_ULTRA_THRESHOLD: int = _env_int(
        "RINGRIFT_DATA_STARVATION_ULTRA_THRESHOLD", 500
    )
    DATA_STARVATION_EMERGENCY_THRESHOLD: int = _env_int(
        "RINGRIFT_DATA_STARVATION_EMERGENCY_THRESHOLD", 1500
    )
    DATA_STARVATION_CRITICAL_THRESHOLD: int = _env_int(
        "RINGRIFT_DATA_STARVATION_CRITICAL_THRESHOLD", 3000
    )
    DATA_STARVATION_ULTRA_MULTIPLIER: float = _env_float(
        # Dec 31, 2025: Increased from 25x to 100x for 48h autonomous operation
        # Jan 2026: Increased to 200x to ensure critically starved configs get priority
        # Jan 5, 2026: Increased to 500x to address 3-player config starvation
        "RINGRIFT_DATA_STARVATION_ULTRA_MULTIPLIER", 500.0
    )
    DATA_STARVATION_EMERGENCY_MULTIPLIER: float = _env_float(
        # Jan 2026: Increased from 10x to 50x to address allocation imbalance
        # Jan 5, 2026: Increased to 100x to address 3-player config starvation
        "RINGRIFT_DATA_STARVATION_EMERGENCY_MULTIPLIER", 100.0
    )
    DATA_STARVATION_CRITICAL_MULTIPLIER: float = _env_float(
        # Jan 2026: Increased from 5x to 20x to address allocation imbalance
        # Jan 5, 2026: Increased to 30x to address 3-player config starvation
        "RINGRIFT_DATA_STARVATION_CRITICAL_MULTIPLIER", 30.0
    )

    # Data poverty tier - configs below this threshold get moderate priority boost
    # Dec 30, 2025: Added to bridge gap between CRITICAL (1000) and no boost
    # Jan 2026: Lowered threshold to 3000, increased multiplier to 5.0
    DATA_POVERTY_THRESHOLD: int = _env_int("RINGRIFT_DATA_POVERTY_THRESHOLD", 3000)
    DATA_POVERTY_MULTIPLIER: float = _env_float(
        "RINGRIFT_DATA_POVERTY_MULTIPLIER", 5.0  # Stronger boost for <3000 games
    )

    # Session 17.34 (Jan 5, 2026): Add WARNING tier for configs with <5000 games
    # This catches configs like square8_3p (3,167 games) and hexagonal_2p (4,008 games)
    # that are just above POVERTY threshold but still underserved for reliable training
    DATA_WARNING_THRESHOLD: int = _env_int("RINGRIFT_DATA_WARNING_THRESHOLD", 5000)
    DATA_WARNING_MULTIPLIER: float = _env_float(
        "RINGRIFT_DATA_WARNING_MULTIPLIER", 3.0  # Moderate boost for <5000 games
    )

    # Staleness thresholds (hours)
    FRESH_DATA_THRESHOLD: float = _env_float("RINGRIFT_FRESH_DATA_THRESHOLD", 1.0)
    STALE_DATA_THRESHOLD: float = _env_float("RINGRIFT_STALE_DATA_THRESHOLD", 4.0)
    # December 29, 2025: Increased from 24h to 48h for extended autonomous operation
    MAX_STALENESS_HOURS: float = _env_float("RINGRIFT_MAX_STALENESS_HOURS", 48.0)

    # VOI (Value of Information) target
    VOI_ELO_TARGET: float = _env_float("RINGRIFT_VOI_ELO_TARGET", 2000.0)
    TARGET_GAMES_FOR_2000_ELO: int = _env_int("RINGRIFT_TARGET_GAMES_FOR_2000_ELO", 100000)
    LARGE_BOARD_TARGET_MULTIPLIER: float = _env_float(
        "RINGRIFT_LARGE_BOARD_TARGET_MULTIPLIER", 1.5
    )

    def get_weight_bounds(self) -> dict[str, tuple[float, float]]:
        """Get dynamic weight bounds as a dict (for backward compatibility)."""
        return {
            "staleness": (self.STALENESS_WEIGHT_MIN, self.STALENESS_WEIGHT_MAX),
            "velocity": (self.VELOCITY_WEIGHT_MIN, self.VELOCITY_WEIGHT_MAX),
            "training": (0.05, 0.20),  # Fixed bounds
            "exploration": (0.05, 0.20),  # Fixed bounds
            "curriculum": (self.CURRICULUM_WEIGHT_MIN, self.CURRICULUM_WEIGHT_MAX),
            "improvement": (0.10, 0.25),  # Fixed bounds
            "data_deficit": (self.DATA_DEFICIT_WEIGHT_MIN, self.DATA_DEFICIT_WEIGHT_MAX),
            "quality": (self.QUALITY_WEIGHT_MIN, self.QUALITY_WEIGHT_MAX),
            "voi": (self.VOI_WEIGHT_MIN, self.VOI_WEIGHT_MAX),
            # January 2026 Sprint 10: Diversity weight bounds
            "diversity": (self.DIVERSITY_WEIGHT_MIN, self.DIVERSITY_WEIGHT_MAX),
        }


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

    Jan 19, 2026: Lowered MAX_PROCESSES from 128 to 16 to prevent
    node saturation and maintain P2P heartbeat responsiveness.
    """
    # GPU idle threshold - consider GPU idle after this duration (seconds)
    GPU_IDLE_THRESHOLD: int = _env_int("RINGRIFT_GPU_IDLE_THRESHOLD", 600)  # 10 minutes

    # Maximum selfplay processes per node before cleanup
    # Jan 19, 2026: Lowered from 128 to 16 for P2P stability
    MAX_PROCESSES: int = _env_int("RINGRIFT_MAX_PROCESSES", 16)

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

    # Dec 2025: Threshold for using fast integrity check instead of full scan (bytes)
    # Databases larger than this will use fast check to avoid timeouts
    LARGE_DB_THRESHOLD: int = _env_int("RINGRIFT_LARGE_DB_THRESHOLD", 100_000_000)  # 100MB


# =============================================================================
# SSH Defaults (December 28, 2025)
# =============================================================================

@dataclass(frozen=True)
class SSHDefaults:
    """Default values for SSH operations.

    Used by: scripts/p2p_orchestrator.py, app/coordination/auto_sync_daemon.py,
             app/core/ssh.py, scripts/p2p/managers/*.py

    Consolidates 40+ scattered SSH timeout values across the codebase.

    December 30, 2025: Extended with per-provider timeouts and connection
    stability settings. Cloud providers have different network characteristics:
    - Vast.ai: Higher latency, needs longer timeouts
    - Lambda: Generally reliable, standard timeouts
    - Nebius: Very reliable, standard timeouts
    - Hetzner: CPU-only, can be slow to respond

    Use build_ssh_options() or build_ssh_options_list() to construct SSH
    command options with appropriate per-provider settings.
    """
    # ==========================================================================
    # Core Timeouts
    # ==========================================================================

    # Command execution timeout (seconds) - for quick commands like nvidia-smi
    COMMAND_TIMEOUT: float = _env_float("RINGRIFT_SSH_COMMAND_TIMEOUT", 30.0)

    # Long command timeout (seconds) - for training, exports, migrations
    LONG_COMMAND_TIMEOUT: float = _env_float("RINGRIFT_SSH_LONG_COMMAND_TIMEOUT", 300.0)

    # SCP file transfer timeout (seconds) - for model/data transfers
    SCP_TIMEOUT: float = _env_float("RINGRIFT_SCP_TIMEOUT", 60.0)

    # Connection timeout (seconds) - initial SSH handshake (default)
    CONNECT_TIMEOUT: float = _env_float("RINGRIFT_SSH_CONNECT_TIMEOUT", 10.0)

    # Rsync timeout (seconds) - for rsync operations
    RSYNC_TIMEOUT: float = _env_float("RINGRIFT_RSYNC_TIMEOUT", 60.0)

    # Health check SSH timeout (seconds) - quick connectivity tests
    HEALTH_CHECK_TIMEOUT: float = _env_float("RINGRIFT_SSH_HEALTH_CHECK_TIMEOUT", 5.0)

    # Maximum retries for SSH operations
    MAX_RETRIES: int = _env_int("RINGRIFT_SSH_MAX_RETRIES", 3)

    # ==========================================================================
    # Connection Stability Settings (December 30, 2025)
    # ==========================================================================

    # TCP KeepAlive - send TCP keepalive probes to detect broken connections
    TCP_KEEPALIVE: bool = _env_bool("RINGRIFT_SSH_TCP_KEEPALIVE", True)

    # ServerAliveInterval (seconds) - send null packet to keep connection alive
    SERVER_ALIVE_INTERVAL: int = _env_int("RINGRIFT_SSH_SERVER_ALIVE_INTERVAL", 30)

    # ServerAliveCountMax - max failed keepalives before disconnect
    SERVER_ALIVE_COUNT_MAX: int = _env_int("RINGRIFT_SSH_SERVER_ALIVE_COUNT_MAX", 3)

    # StrictHostKeyChecking - disabled for dynamic cloud instances
    STRICT_HOST_KEY_CHECKING: bool = _env_bool("RINGRIFT_SSH_STRICT_HOST_KEY", False)

    # BatchMode - no interactive prompts (required for automated operations)
    BATCH_MODE: bool = _env_bool("RINGRIFT_SSH_BATCH_MODE", True)

    # ==========================================================================
    # Per-Provider Connect Timeout Overrides (December 30, 2025)
    # ==========================================================================
    # Cloud providers have different network characteristics. Vast.ai instances
    # often have higher latency and need longer connection timeouts.

    VAST_CONNECT_TIMEOUT: float = _env_float("RINGRIFT_SSH_VAST_CONNECT_TIMEOUT", 15.0)
    RUNPOD_CONNECT_TIMEOUT: float = _env_float("RINGRIFT_SSH_RUNPOD_CONNECT_TIMEOUT", 10.0)
    LAMBDA_CONNECT_TIMEOUT: float = _env_float("RINGRIFT_SSH_LAMBDA_CONNECT_TIMEOUT", 10.0)
    NEBIUS_CONNECT_TIMEOUT: float = _env_float("RINGRIFT_SSH_NEBIUS_CONNECT_TIMEOUT", 10.0)
    VULTR_CONNECT_TIMEOUT: float = _env_float("RINGRIFT_SSH_VULTR_CONNECT_TIMEOUT", 10.0)
    HETZNER_CONNECT_TIMEOUT: float = _env_float("RINGRIFT_SSH_HETZNER_CONNECT_TIMEOUT", 15.0)
    LOCAL_CONNECT_TIMEOUT: float = _env_float("RINGRIFT_SSH_LOCAL_CONNECT_TIMEOUT", 5.0)


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
    # Jan 5, 2026 (Session 17.33): Reduced from 60s to 45s for 25% faster dead node detection
    PEER_TIMEOUT: float = _env_float("RINGRIFT_PEER_TIMEOUT", 45.0)

    # Gossip interval (seconds) - how often to exchange state
    GOSSIP_INTERVAL: float = _env_float("RINGRIFT_PEER_GOSSIP_INTERVAL", 15.0)

    # Manifest collection timeout (seconds) - for data manifest requests
    # Jan 2, 2026 (Sprint 8): Reduced from 300s to 60s for faster sync of new games
    MANIFEST_TIMEOUT: float = _env_float("RINGRIFT_PEER_MANIFEST_TIMEOUT", 60.0)

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


def get_ssh_connect_timeout(provider: str = "default") -> float:
    """Get SSH connect timeout for a specific cloud provider.

    Different providers have different network characteristics. This function
    returns the appropriate connect timeout based on the provider name.

    Args:
        provider: Provider name (vast, runpod, lambda, nebius, vultr, hetzner, local)
                  Case-insensitive. Defaults to SSHDefaults.CONNECT_TIMEOUT.

    Returns:
        Connect timeout in seconds

    Example:
        timeout = get_ssh_connect_timeout("vast")  # Returns 15.0
        timeout = get_ssh_connect_timeout()  # Returns 10.0 (default)

    December 30, 2025: Added to support per-provider SSH configuration.
    """
    provider = provider.lower()
    timeouts = {
        "vast": SSHDefaults.VAST_CONNECT_TIMEOUT,
        "runpod": SSHDefaults.RUNPOD_CONNECT_TIMEOUT,
        "lambda": SSHDefaults.LAMBDA_CONNECT_TIMEOUT,
        "nebius": SSHDefaults.NEBIUS_CONNECT_TIMEOUT,
        "vultr": SSHDefaults.VULTR_CONNECT_TIMEOUT,
        "hetzner": SSHDefaults.HETZNER_CONNECT_TIMEOUT,
        "local": SSHDefaults.LOCAL_CONNECT_TIMEOUT,
    }
    return timeouts.get(provider, SSHDefaults.CONNECT_TIMEOUT)


def get_provider_from_node_id(node_id: str) -> str:
    """Infer cloud provider from node ID naming convention.

    Node IDs follow the pattern: {provider}-{identifier}
    Examples: vast-28889766, lambda-gh200-1, nebius-h100-3

    Args:
        node_id: Node identifier string

    Returns:
        Provider name (vast, runpod, lambda, nebius, vultr, hetzner, local, or default)

    December 30, 2025: Added to support automatic provider detection for SSH config.
    """
    node_id_lower = node_id.lower()

    if node_id_lower.startswith("vast-"):
        return "vast"
    elif node_id_lower.startswith("runpod-"):
        return "runpod"
    elif node_id_lower.startswith("lambda-"):
        return "lambda"
    elif node_id_lower.startswith("nebius-"):
        return "nebius"
    elif node_id_lower.startswith("vultr-"):
        return "vultr"
    elif node_id_lower.startswith("hetzner-"):
        return "hetzner"
    elif node_id_lower.startswith("local-") or node_id_lower.startswith("mac-"):
        return "local"
    else:
        return "default"


def build_ssh_options(
    key_path: str | None = None,
    provider: str = "default",
    include_keepalive: bool = True,
    port: int | None = None,
    node_id: str | None = None,
) -> str:
    """Build SSH options string for subprocess/rsync commands.

    Replaces scattered hardcoded SSH options across 100+ files with a single
    source of truth. All settings are configurable via environment variables.

    Args:
        key_path: Path to SSH key (optional, uses -i flag if provided)
        provider: Provider name for per-provider timeout adjustment
        include_keepalive: Include ServerAlive settings for long connections
        port: SSH port (optional, uses -p flag if provided)
        node_id: Node ID to auto-detect provider (overrides provider if set)

    Returns:
        SSH options string suitable for -e flag in rsync or direct ssh command

    Example:
        # Before (hardcoded in 100+ files):
        ssh_opts = "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10"

        # After:
        from app.config.coordination_defaults import build_ssh_options
        ssh_opts = build_ssh_options(key_path="~/.ssh/id_cluster", provider="vast")
        # Returns: "ssh -i ~/.ssh/id_cluster -o StrictHostKeyChecking=no ..."

        # Or with auto-detection:
        ssh_opts = build_ssh_options(key_path="~/.ssh/id_cluster", node_id="vast-28889766")

    December 30, 2025: Added to centralize SSH configuration across codebase.
    """
    # Auto-detect provider from node_id if provided
    if node_id:
        provider = get_provider_from_node_id(node_id)

    parts = ["ssh"]

    if key_path:
        parts.extend(["-i", key_path])
        parts.extend(["-o", "IdentitiesOnly=yes"])

    if port:
        parts.extend(["-p", str(port)])

    if not SSHDefaults.STRICT_HOST_KEY_CHECKING:
        parts.extend(["-o", "StrictHostKeyChecking=no"])

    if SSHDefaults.BATCH_MODE:
        parts.extend(["-o", "BatchMode=yes"])

    connect_timeout = get_ssh_connect_timeout(provider)
    parts.extend(["-o", f"ConnectTimeout={int(connect_timeout)}"])

    if include_keepalive:
        if SSHDefaults.TCP_KEEPALIVE:
            parts.extend(["-o", "TCPKeepAlive=yes"])
        parts.extend(["-o", f"ServerAliveInterval={SSHDefaults.SERVER_ALIVE_INTERVAL}"])
        parts.extend(["-o", f"ServerAliveCountMax={SSHDefaults.SERVER_ALIVE_COUNT_MAX}"])

    return " ".join(parts)


def build_ssh_options_list(
    key_path: str | None = None,
    provider: str = "default",
    include_keepalive: bool = True,
    port: int | None = None,
    node_id: str | None = None,
) -> list[str]:
    """Build SSH options as a list for subprocess.run() commands.

    Same as build_ssh_options() but returns a list instead of string.
    Use this for subprocess.run() with shell=False (safer).

    Args:
        key_path: Path to SSH key (optional)
        provider: Provider name for per-provider timeout adjustment
        include_keepalive: Include ServerAlive settings
        port: SSH port (optional)
        node_id: Node ID to auto-detect provider

    Returns:
        List of SSH command parts for subprocess.run()

    Example:
        cmd = build_ssh_options_list(key_path="~/.ssh/id_cluster", node_id="vast-123")
        cmd.extend(["user@host", "nvidia-smi"])
        subprocess.run(cmd, ...)

    December 30, 2025: Added for safer subprocess usage with shell=False.
    """
    # Auto-detect provider from node_id if provided
    if node_id:
        provider = get_provider_from_node_id(node_id)

    parts: list[str] = ["ssh"]

    if key_path:
        parts.extend(["-i", key_path])
        parts.extend(["-o", "IdentitiesOnly=yes"])

    if port:
        parts.extend(["-p", str(port)])

    if not SSHDefaults.STRICT_HOST_KEY_CHECKING:
        parts.extend(["-o", "StrictHostKeyChecking=no"])

    if SSHDefaults.BATCH_MODE:
        parts.extend(["-o", "BatchMode=yes"])

    connect_timeout = get_ssh_connect_timeout(provider)
    parts.extend(["-o", f"ConnectTimeout={int(connect_timeout)}"])

    if include_keepalive:
        if SSHDefaults.TCP_KEEPALIVE:
            parts.extend(["-o", "TCPKeepAlive=yes"])
        parts.extend(["-o", f"ServerAliveInterval={SSHDefaults.SERVER_ALIVE_INTERVAL}"])
        parts.extend(["-o", f"ServerAliveCountMax={SSHDefaults.SERVER_ALIVE_COUNT_MAX}"])

    return parts


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
class HTTPHandlerDefaults:
    """Default values for HTTP handler timeouts.

    Used by: scripts/p2p_orchestrator.py, app/coordination/daemon_manager.py

    December 30, 2025: Added to fix P2P cluster connectivity issues.
    HTTP handlers were blocking indefinitely on slow operations (lock
    acquisition, daemon status collection). These timeouts ensure handlers
    return within reasonable time even when subsystems are slow.
    """
    # Status endpoint timeout (seconds) - collect cluster/daemon status
    STATUS_TIMEOUT: float = _env_float("RINGRIFT_STATUS_TIMEOUT", 10.0)

    # Health endpoint timeout (seconds) - simple liveness check
    HEALTH_TIMEOUT: float = _env_float("RINGRIFT_HEALTH_TIMEOUT", 5.0)

    # Dispatch endpoint timeout (seconds) - job dispatch operations
    DISPATCH_TIMEOUT: float = _env_float("RINGRIFT_DISPATCH_TIMEOUT", 30.0)

    # Individual daemon status timeout (seconds) - per-daemon health check
    DAEMON_STATUS_TIMEOUT: float = _env_float("RINGRIFT_DAEMON_STATUS_TIMEOUT", 1.0)

    # Lock acquisition timeout in handlers (seconds)
    LOCK_TIMEOUT: float = _env_float("RINGRIFT_HANDLER_LOCK_TIMEOUT", 2.0)


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


class GossipDefaults:
    """Default values for gossip protocol operations.

    Used by: scripts/p2p/gossip_protocol.py

    December 30, 2025: Extracted hardcoded timeouts to enable network tuning
    for cross-cloud deployments (Lambda↔Vast↔RunPod may need different settings).
    """
    # Lock acquisition timeout for gossip state operations (seconds)
    STATE_LOCK_TIMEOUT: float = _env_float("RINGRIFT_GOSSIP_LOCK_TIMEOUT", 5.0)

    # HTTP probe timeout for endpoint validation (seconds)
    PROBE_HTTP_TIMEOUT: float = _env_float("RINGRIFT_GOSSIP_PROBE_TIMEOUT", 5.0)

    # Full gossip exchange timeout (seconds) - longer for full state transfer
    EXCHANGE_TIMEOUT: float = _env_float("RINGRIFT_GOSSIP_EXCHANGE_TIMEOUT", 10.0)

    # Anti-entropy repair timeout (seconds)
    ANTI_ENTROPY_TIMEOUT: float = _env_float("RINGRIFT_GOSSIP_ANTI_ENTROPY_TIMEOUT", 10.0)

    # Consecutive failures before marking peer as suspect
    # Jan 2026: Increased from 4 to 8 for more tolerance with flaky cross-cloud networks
    # Jan 24, 2026: Increased from 8 to 10 for better stability on 40+ node clusters
    # Jan 25, 2026: Increased from 10 to 15 for cross-cloud packet loss tolerance.
    # With 15 failures at 15s heartbeat = 225s to mark SUSPECT, more tolerant of network jitter.
    FAILURE_THRESHOLD: int = _env_int("RINGRIFT_GOSSIP_FAILURE_THRESHOLD", 15)

    # Maximum gossip message size in bytes (1MB default)
    MAX_MESSAGE_SIZE_BYTES: int = _env_int("RINGRIFT_GOSSIP_MAX_MESSAGE_SIZE", 1_048_576)

    # =========================================================================
    # January 3, 2026 Sprint 12 Session 10: Timeout Centralization
    # Migrated from hardcoded values in gossip_protocol.py
    # =========================================================================

    # Dead peer detection: seconds since last seen before marking as dead
    # Jan 13, 2026: Reduced from 300s to 60s for faster quorum detection during 48h autonomous ops
    # Jan 25, 2026: CRITICAL FIX - Increased from 60s to 240s.
    # DEAD_PEER_TIMEOUT MUST be >= PEER_TIMEOUT (180s) to avoid premature gossip cleanup.
    # With 60s, peers were removed from gossip before being marked dead, creating "zombie peers"
    # where nodes disagreed on peer liveness (asymmetric connectivity, 9-19 peer fluctuation).
    DEAD_PEER_TIMEOUT: float = _env_float("RINGRIFT_GOSSIP_DEAD_PEER_TIMEOUT", 240.0)

    # Cleanup interval: minimum seconds between gossip state cleanup passes
    CLEANUP_INTERVAL: float = _env_float("RINGRIFT_GOSSIP_CLEANUP_INTERVAL", 300.0)

    # NAT-blocked peer timeout: longer timeout for peers behind NAT/relay (seconds)
    NAT_PEER_TIMEOUT: float = _env_float("RINGRIFT_GOSSIP_NAT_TIMEOUT", 120.0)

    # Circuit breaker cutoff: only process CB states from last N seconds
    CIRCUIT_CUTOFF_SECONDS: float = _env_float("RINGRIFT_GOSSIP_CB_CUTOFF", 300.0)

    # Stale state threshold: gossip states older than this are cleaned up (seconds)
    # January 8, 2026: Reduced from 600s to 180s for faster stale peer detection
    STALE_STATE_SECONDS: float = _env_float("RINGRIFT_GOSSIP_STALE_STATE", 180.0)

    # State TTL: how long gossip states are cached before expiring (seconds)
    # Jan 6, 2026: Reduced from 3600s (1hr) to 600s (10min) for faster peer recovery discovery
    # January 8, 2026: Further reduced from 600s to 180s for faster autonomous recovery
    # Jan 19, 2026: Further reduced from 180s to 60s (2x convergence time)
    # CRITICAL FIX: 180s TTL >> 30s convergence caused minutes of view divergence
    # Jan 22, 2026: INCREASED from 60s to 240s for 40-node cluster propagation.
    # CRITICAL FIX: 60s TTL was too short for gossip to propagate across 40 nodes.
    # Math: With fanout 10 and 30s interval, 40 nodes need 2 rounds = 60s minimum.
    # With 4x safety factor for jitter/delays: 60s * 4 = 240s minimum TTL.
    # Without this fix, Lambda nodes never learned about peer changes (state expired mid-propagation).
    # Jan 25, 2026: Increased from 240s to 480s. STATE_TTL should be 2x DEAD_PEER_TIMEOUT (240s)
    # to ensure gossip state persists long enough for cleanup to complete across all nodes.
    STATE_TTL: float = _env_float("RINGRIFT_GOSSIP_STATE_TTL", 480.0)

    # =========================================================================
    # January 5, 2026 Session 17.28: Recovery Probing for Dead Nodes
    # Periodically check if "dead" nodes have recovered to reduce false-dead state
    # Expected improvement: +5-10 nodes recovered from false-dead state
    # =========================================================================

    # Recovery probe interval: seconds between probing "dead" nodes for recovery
    RECOVERY_PROBE_INTERVAL: float = _env_float("RINGRIFT_GOSSIP_RECOVERY_PROBE_INTERVAL", 60.0)

    # Recovery probe batch size: max dead nodes to probe per cycle (prevents thundering herd)
    # Jan 25, 2026: Increased from 3 to 10 for faster dead peer recovery on 40-node clusters.
    # With 8-10 dead peers and batch 3, took 180s to probe all. With batch 10, takes 60s.
    RECOVERY_PROBE_BATCH_SIZE: int = _env_int("RINGRIFT_GOSSIP_RECOVERY_PROBE_BATCH_SIZE", 10)


# =============================================================================
# Convenience Functions
# =============================================================================


def get_gossip_state_ttl(cluster_size: int = 40) -> float:
    """Calculate gossip state TTL based on cluster size.

    Jan 22, 2026: Dynamic TTL ensures gossip state lives long enough to propagate.

    Formula: TTL = gossip_rounds_for_propagation * gossip_interval * safety_factor

    For N nodes with fanout F:
    - Rounds needed: ceil(log_F(N))
    - With 40 nodes, fanout 10: ceil(log10(40)) = 2 rounds
    - gossip_interval = 30s (stable mode)
    - safety_factor = 4 (for jitter, network delays, retries)

    TTL = 2 * 30s * 4 = 240s minimum

    Args:
        cluster_size: Number of nodes in the cluster (default: 40)

    Returns:
        Recommended gossip state TTL in seconds

    Example:
        >>> get_gossip_state_ttl(40)
        240.0
        >>> get_gossip_state_ttl(100)  # Larger cluster needs longer TTL
        360.0
    """
    import math

    fanout = 10  # GOSSIP_FANOUT_LEADER
    gossip_interval = 30.0  # GOSSIP_INTERVAL_STABLE
    safety_factor = 4.0

    rounds = max(2, math.ceil(math.log(max(cluster_size, 2), fanout)))
    ttl = rounds * gossip_interval * safety_factor

    return max(ttl, 240.0)  # Minimum 240s for stability


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
        # December 2025: Degraded mode (48-hour autonomous operation)
        "degraded_mode": {
            "critical_retry_interval": DegradedModeDefaults.CRITICAL_RETRY_INTERVAL,
            "noncritical_retry_interval": DegradedModeDefaults.NONCRITICAL_RETRY_INTERVAL,
            "normal_max_restarts": DegradedModeDefaults.NORMAL_MAX_RESTARTS,
            "elevated_max_restarts": DegradedModeDefaults.ELEVATED_MAX_RESTARTS,
            "enabled": DegradedModeDefaults.ENABLED,
            "reset_after_hours": DegradedModeDefaults.RESET_AFTER_HOURS,
        },
        # December 2025: Stale fallback (48-hour autonomous operation)
        "stale_fallback": {
            "max_sync_failures": StaleFallbackDefaults.MAX_SYNC_FAILURES,
            "max_sync_duration": StaleFallbackDefaults.MAX_SYNC_DURATION,
            "absolute_max_data_age": StaleFallbackDefaults.ABSOLUTE_MAX_DATA_AGE,
            "enabled": StaleFallbackDefaults.ENABLE_STALE_FALLBACK,
            "min_games_for_fallback": StaleFallbackDefaults.MIN_GAMES_FOR_FALLBACK,
            "fallback_cooldown": StaleFallbackDefaults.FALLBACK_COOLDOWN,
            "emit_fallback_events": StaleFallbackDefaults.EMIT_FALLBACK_EVENTS,
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


# =============================================================================
# Quality Gate Defaults (January 2, 2026 - Phase 2.2)
# =============================================================================

class QualityGateDefaults:
    """Quality gate thresholds for training triggers.

    January 2, 2026 (Phase 2.2): Adaptive quality gates by player count.
    4-player configs have higher variance, so they need stricter quality
    thresholds to ensure training data is meaningful.

    January 3, 2026 (Sprint 10): Added board-type-specific adjustments.
    Larger/more complex boards need higher quality data for effective training:
    - hex8: Simplest board (61 cells) - lowest threshold
    - square8: Medium complexity (64 cells) - medium threshold
    - square19: Large board (361 cells) - higher threshold
    - hexagonal: Most complex (469 cells) - highest threshold

    Used by: training_trigger_daemon.py, quality_monitor_daemon.py
    """
    # Default quality threshold (used if config not in QUALITY_GATES)
    DEFAULT_THRESHOLD: float = _env_float("RINGRIFT_DEFAULT_QUALITY_THRESHOLD", 0.50)

    # Board complexity adjustments (applied on top of player count threshold)
    # Sprint 10: More complex boards need higher quality training data
    BOARD_COMPLEXITY_ADJUSTMENTS: dict = {
        "hex8": 0.0,        # Simplest board - baseline
        "square8": 0.02,    # Slightly more complex
        "square19": 0.05,   # Large board - higher quality needed
        "hexagonal": 0.08,  # Most complex - highest quality needed
    }

    # Quality thresholds by config - combines player count + board complexity
    # Sprint 10: Thresholds now vary by BOTH board type AND player count
    # Base: 2p=0.50, 3p=0.55, 4p=0.65, then add board complexity adjustment
    QUALITY_GATES: dict = {
        # 2-player configs - baseline + board adjustment
        "hex8_2p": 0.50,         # 0.50 + 0.00
        "square8_2p": 0.52,      # 0.50 + 0.02
        "square19_2p": 0.55,     # 0.50 + 0.05
        "hexagonal_2p": 0.58,    # 0.50 + 0.08
        # 3-player configs - moderate + board adjustment
        "hex8_3p": 0.55,         # 0.55 + 0.00
        "square8_3p": 0.57,      # 0.55 + 0.02
        "square19_3p": 0.60,     # 0.55 + 0.05
        "hexagonal_3p": 0.63,    # 0.55 + 0.08
        # 4-player configs - highest + board adjustment
        "hex8_4p": 0.65,         # 0.65 + 0.00
        "square8_4p": 0.67,      # 0.65 + 0.02
        "square19_4p": 0.70,     # 0.65 + 0.05
        "hexagonal_4p": 0.73,    # 0.65 + 0.08
    }

    @classmethod
    def get_quality_threshold(cls, config_key: str) -> float:
        """Get quality threshold for a config.

        Args:
            config_key: Config key like "hex8_2p" or "square19_4p"

        Returns:
            Quality threshold (0.0 - 1.0)
        """
        return cls.QUALITY_GATES.get(config_key, cls.DEFAULT_THRESHOLD)

    @classmethod
    def get_board_complexity_adjustment(cls, board_type: str) -> float:
        """Get quality threshold adjustment for board complexity.

        January 3, 2026 (Sprint 10): Returns the quality threshold adjustment
        for the given board type. Larger/more complex boards get higher adjustments.

        Args:
            board_type: Board type like "hex8", "square19", "hexagonal"

        Returns:
            Adjustment value (0.0 - 0.10) to add to base threshold
        """
        return cls.BOARD_COMPLEXITY_ADJUSTMENTS.get(board_type, 0.0)


# =============================================================================
# Promotion Game Defaults (January 2, 2026 - Phase 2.3)
# =============================================================================

class PromotionGameDefaults:
    """Promotion game count requirements by player count.

    January 2, 2026 (Phase 2.3): Graduate promotion games by player count.
    4-player games have higher variance, requiring more games for
    statistically significant promotion decisions.

    Used by: auto_promotion_daemon.py, promotion_controller.py
    """
    # Minimum games by player count for promotion consideration
    MIN_GAMES_2P: int = _env_int("RINGRIFT_PROMOTION_MIN_GAMES_2P", 50)
    MIN_GAMES_3P: int = _env_int("RINGRIFT_PROMOTION_MIN_GAMES_3P", 70)
    MIN_GAMES_4P: int = _env_int("RINGRIFT_PROMOTION_MIN_GAMES_4P", 100)

    # Win rate thresholds by player count
    # 4-player games are harder to win consistently, so lower threshold
    WIN_RATE_2P: float = _env_float("RINGRIFT_PROMOTION_WIN_RATE_2P", 0.52)
    WIN_RATE_3P: float = _env_float("RINGRIFT_PROMOTION_WIN_RATE_3P", 0.45)
    WIN_RATE_4P: float = _env_float("RINGRIFT_PROMOTION_WIN_RATE_4P", 0.35)

    @classmethod
    def get_min_games(cls, num_players: int) -> int:
        """Get minimum games required for promotion.

        Args:
            num_players: Number of players (2, 3, or 4)

        Returns:
            Minimum games required
        """
        games_map = {
            2: cls.MIN_GAMES_2P,
            3: cls.MIN_GAMES_3P,
            4: cls.MIN_GAMES_4P,
        }
        return games_map.get(num_players, cls.MIN_GAMES_2P)

    @classmethod
    def get_win_rate_threshold(cls, num_players: int) -> float:
        """Get win rate threshold for promotion.

        Args:
            num_players: Number of players (2, 3, or 4)

        Returns:
            Minimum win rate (0.0 - 1.0)
        """
        rate_map = {
            2: cls.WIN_RATE_2P,
            3: cls.WIN_RATE_3P,
            4: cls.WIN_RATE_4P,
        }
        return rate_map.get(num_players, cls.WIN_RATE_2P)


# =============================================================================
# OWC Model Import Defaults (January 3, 2026 - Sprint 13 Session 4)
# =============================================================================


@dataclass(frozen=True)
class OWCModelImportDefaults:
    """Default values for OWC model import daemon.

    January 3, 2026 (Sprint 13 Session 4): Imports model files (.pth) from
    OWC external drive on mac-studio for Elo evaluation.

    Used by: app/coordination/owc_model_import_daemon.py
    """
    # Check interval for discovering new models (seconds)
    # 2 hours by default to avoid excessive SSH operations
    CHECK_INTERVAL: float = _env_float("RINGRIFT_OWC_MODEL_IMPORT_INTERVAL", 7200.0)

    # Maximum models to import per cycle
    # Limits concurrent imports to avoid overloading coordinator
    MAX_MODELS_PER_CYCLE: int = _env_int("RINGRIFT_OWC_MODEL_IMPORT_MAX_MODELS", 10)

    # Minimum model file size (bytes) to consider valid
    # Filters out corrupted or partial model files
    MIN_MODEL_SIZE_BYTES: int = _env_int("RINGRIFT_OWC_MODEL_MIN_SIZE", 1_000_000)

    # Maximum model file size (bytes) to import
    # Prevents importing unexpectedly large files
    MAX_MODEL_SIZE_BYTES: int = _env_int("RINGRIFT_OWC_MODEL_MAX_SIZE", 1_000_000_000)

    # SSH timeout for OWC operations (seconds)
    SSH_TIMEOUT: float = _env_float("RINGRIFT_OWC_MODEL_SSH_TIMEOUT", 60.0)

    # Enable/disable OWC model import daemon
    ENABLED: bool = _env_bool("RINGRIFT_OWC_MODEL_IMPORT_ENABLED", True)


# =============================================================================
# Unevaluated Model Scanner Defaults (January 3, 2026 - Sprint 13 Session 4)
# =============================================================================


@dataclass(frozen=True)
class UnevaluatedScannerDefaults:
    """Default values for unevaluated model scanner daemon.

    January 3, 2026 (Sprint 13 Session 4): Scans all model sources for models
    without Elo ratings and queues them for evaluation.

    Used by: app/coordination/unevaluated_model_scanner_daemon.py
    """
    # Scan interval (seconds)
    # 1 hour by default - balance between freshness and load
    SCAN_INTERVAL: float = _env_float("RINGRIFT_UNEVALUATED_SCAN_INTERVAL", 3600.0)

    # Maximum models to queue per cycle
    # Prevents flooding evaluation queue
    MAX_QUEUE_PER_CYCLE: int = _env_int("RINGRIFT_UNEVALUATED_MAX_QUEUE", 20)

    # Priority boost for 4-player models (underserved configs)
    PRIORITY_BOOST_4PLAYER: int = _env_int("RINGRIFT_UNEVALUATED_PRIORITY_4P", 30)

    # Priority boost for underserved configs (hexagonal, square19)
    PRIORITY_BOOST_UNDERSERVED: int = _env_int("RINGRIFT_UNEVALUATED_PRIORITY_UNDERSERVED", 50)

    # Priority boost for canonical models
    PRIORITY_BOOST_CANONICAL: int = _env_int("RINGRIFT_UNEVALUATED_PRIORITY_CANONICAL", 20)

    # Priority boost for recent training (within 24h)
    PRIORITY_BOOST_RECENT: int = _env_int("RINGRIFT_UNEVALUATED_PRIORITY_RECENT", 30)

    # Base priority for all models
    BASE_PRIORITY: int = _env_int("RINGRIFT_UNEVALUATED_BASE_PRIORITY", 50)

    # Enable/disable scanner daemon
    ENABLED: bool = _env_bool("RINGRIFT_UNEVALUATED_SCANNER_ENABLED", True)


# =============================================================================
# Evaluation Queue Defaults (January 3, 2026 - Sprint 13 Session 4)
# =============================================================================


@dataclass(frozen=True)
class EvaluationQueueDefaults:
    """Default values for persistent evaluation queue.

    January 3, 2026 (Sprint 13 Session 4): SQLite-backed priority queue for
    model evaluation that survives daemon restarts.

    Used by: app/coordination/evaluation_queue.py, evaluation_daemon.py
    """
    # Database path for persistent queue
    DB_PATH: str = "data/coordination/evaluation_queue.db"

    # Stuck detection interval (seconds)
    # Check for stuck evaluations every 30 minutes
    STUCK_DETECTION_INTERVAL: float = _env_float(
        "RINGRIFT_EVAL_QUEUE_STUCK_INTERVAL", 1800.0
    )

    # Maximum retry attempts for failed evaluations
    MAX_ATTEMPTS: int = _env_int("RINGRIFT_EVAL_QUEUE_MAX_ATTEMPTS", 3)

    # Board-specific stuck timeouts (seconds)
    # Different board sizes have different expected evaluation times
    STUCK_TIMEOUT_HEX8: float = _env_float("RINGRIFT_EVAL_STUCK_HEX8", 3600.0)  # 1 hour
    STUCK_TIMEOUT_SQUARE8: float = _env_float("RINGRIFT_EVAL_STUCK_SQUARE8", 7200.0)  # 2 hours
    STUCK_TIMEOUT_SQUARE19: float = _env_float("RINGRIFT_EVAL_STUCK_SQUARE19", 10800.0)  # 3 hours
    STUCK_TIMEOUT_HEXAGONAL: float = _env_float("RINGRIFT_EVAL_STUCK_HEXAGONAL", 14400.0)  # 4 hours

    # Default stuck timeout for unknown board types
    STUCK_TIMEOUT_DEFAULT: float = _env_float("RINGRIFT_EVAL_STUCK_DEFAULT", 7200.0)  # 2 hours

    # Retry backoff multiplier (exponential)
    RETRY_BACKOFF_MULTIPLIER: float = _env_float("RINGRIFT_EVAL_RETRY_BACKOFF", 2.0)

    # Initial retry delay (seconds)
    RETRY_DELAY_INITIAL: float = _env_float("RINGRIFT_EVAL_RETRY_DELAY", 60.0)

    @classmethod
    def get_stuck_timeout(cls, board_type: str) -> float:
        """Get stuck timeout for board type.

        Args:
            board_type: Board type (hex8, square8, square19, hexagonal)

        Returns:
            Timeout in seconds
        """
        timeouts = {
            "hex8": cls.STUCK_TIMEOUT_HEX8,
            "square8": cls.STUCK_TIMEOUT_SQUARE8,
            "square19": cls.STUCK_TIMEOUT_SQUARE19,
            "hexagonal": cls.STUCK_TIMEOUT_HEXAGONAL,
        }
        return timeouts.get(board_type, cls.STUCK_TIMEOUT_DEFAULT)


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
    "DegradedModeDefaults",  # December 2025 - 48-hour autonomous operation
    "StaleFallbackDefaults",  # December 2025 - 48-hour autonomous operation
    "DurationDefaults",
    "EphemeralDefaults",
    "EphemeralGuardDefaults",
    "EvaluationQueueDefaults",  # January 2026 - Sprint 13 Session 4
    "FrozenLeaderDefaults",  # January 2026
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
    "MetricsRetentionDefaults",  # January 2026
    "MonitoringDefaults",
    "NetworkRetryDefaults",
    "OperationTimeouts",
    "OptimizationDefaults",
    "OrphanDetectionDefaults",
    "OWCModelImportDefaults",  # January 2026 - Sprint 13 Session 4
    "P2PDefaults",
    "P2PRecoveryDefaults",  # January 2026
    "PartitionHealingDefaults",  # January 2026
    "PeerDefaults",  # December 28, 2025
    "PIDDefaults",
    "PromotionGameDefaults",  # January 2026
    "ProviderDefaults",  # December 27, 2025
    "QualityGateDefaults",  # January 2026
    "QueueDefaults",
    "ResourceLimitsDefaults",
    "ResourceManagerDefaults",
    "ResourceMonitoringDefaults",
    "RetryDefaults",
    "ScalingDefaults",
    "SchedulerDefaults",
    "SelfplayAllocationDefaults",
    "SelfplayPriorityWeightDefaults",  # December 29, 2025
    "SQLiteDefaults",
    "SSHDefaults",  # December 28, 2025
    "SyncCoordinatorDefaults",
    "SyncDefaults",
    "SyncIntegrityDefaults",
    "TaskLifecycleDefaults",
    "TrainingDefaults",
    "TransportDefaults",
    "UnevaluatedScannerDefaults",  # January 2026 - Sprint 13 Session 4
    "UtilizationDefaults",
    "WorkQueueMonitorDefaults",
    # Utility functions
    "get_aiohttp_timeout",
    "get_all_defaults",
    "get_backpressure_multiplier",
    "get_circuit_breaker_configs",
    "get_circuit_breaker_for_provider",
    "get_job_timeout",
    "get_p2p_port",
    "get_peer_timeout",  # December 28, 2025
    "get_sqlite_timeout",
    "get_ssh_timeout",  # December 28, 2025
    "get_timeout",
]

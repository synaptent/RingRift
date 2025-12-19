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
from typing import Optional


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
    # Connection timeout (seconds) - increased for VAST.ai and slow networks
    CONNECT_TIMEOUT: int = _env_int("RINGRIFT_CONNECT_TIMEOUT", 30)

    # Operation timeout (seconds) - for large transfers
    OPERATION_TIMEOUT: int = _env_int("RINGRIFT_OPERATION_TIMEOUT", 180)

    # HTTP request timeout (seconds)
    HTTP_TIMEOUT: int = _env_int("RINGRIFT_HTTP_TIMEOUT", 30)

    # Circuit breaker recovery timeout (seconds)
    CIRCUIT_BREAKER_RECOVERY: int = _env_int("RINGRIFT_CIRCUIT_BREAKER_RECOVERY", 300)

    # SSH timeout for remote operations
    SSH_TIMEOUT: int = _env_int("RINGRIFT_SSH_TIMEOUT", 30)

    # Maximum retries for failed operations
    MAX_RETRIES: int = _env_int("RINGRIFT_MAX_RETRIES", 3)


# =============================================================================
# Sync Operation Defaults
# =============================================================================

@dataclass(frozen=True)
class SyncDefaults:
    """Default values for sync operations.

    Used by: app/coordination/sync_mutex.py, app/distributed/sync_orchestrator.py
    """
    # Sync lock timeout (seconds)
    LOCK_TIMEOUT: int = _env_int("RINGRIFT_SYNC_LOCK_TIMEOUT", 120)

    # Maximum concurrent syncs per host
    MAX_CONCURRENT_PER_HOST: int = _env_int("RINGRIFT_MAX_SYNCS_PER_HOST", 1)

    # Maximum concurrent syncs cluster-wide
    MAX_CONCURRENT_CLUSTER: int = _env_int("RINGRIFT_MAX_SYNCS_CLUSTER", 5)

    # Data sync interval (seconds)
    DATA_SYNC_INTERVAL: float = _env_float("RINGRIFT_DATA_SYNC_INTERVAL", 300.0)

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
    }


__all__ = [
    # Config classes
    "LockDefaults",
    "TransportDefaults",
    "SyncDefaults",
    "HeartbeatDefaults",
    "TrainingDefaults",
    "SchedulerDefaults",
    "EphemeralDefaults",
    # Utilities
    "get_all_defaults",
]

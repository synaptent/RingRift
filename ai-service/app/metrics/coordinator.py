"""Coordinator-specific Prometheus metrics.

This module provides metrics for tracking coordinator/manager components,
aggregating stats from CoordinatorBase implementations.

Usage:
    from app.metrics.coordinator import (
        update_coordinator_status,
        record_coordinator_operation,
        get_coordinator_metrics,
    )

    # Update coordinator status
    update_coordinator_status("UnifiedHealthManager", "running")

    # Record an operation
    record_coordinator_operation("BandwidthManager", "request", success=True)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Final

from prometheus_client import Counter, Gauge

logger = logging.getLogger(__name__)


# =============================================================================
# Safe Metric Registration (December 2025: Consolidated)
# =============================================================================
# Use the centralized registry to avoid duplicate metric registration.
from app.metrics.registry import safe_metric as _safe_metric

# =============================================================================
# Coordinator Status Metrics
# =============================================================================

COORDINATOR_STATUS: Final[Gauge] = _safe_metric(Gauge,
    "ringrift_coordinator_status",
    "Current status of coordinators (1=ready, 2=running, 3=paused, 4=draining, 5=error, 0=stopped).",
    labelnames=("coordinator_name",),
)

COORDINATOR_UPTIME: Final[Gauge] = _safe_metric(Gauge,
    "ringrift_coordinator_uptime_seconds",
    "Uptime of coordinators in seconds.",
    labelnames=("coordinator_name",),
)

COORDINATOR_OPERATIONS: Final[Counter] = _safe_metric(Counter,
    "ringrift_coordinator_operations_total",
    "Total operations performed by coordinators.",
    labelnames=("coordinator_name", "operation_type", "status"),
)

COORDINATOR_ERRORS: Final[Counter] = _safe_metric(Counter,
    "ringrift_coordinator_errors_total",
    "Total errors encountered by coordinators.",
    labelnames=("coordinator_name", "error_type"),
)


# =============================================================================
# Health Management Metrics (UnifiedHealthManager)
# =============================================================================

RECOVERY_ATTEMPTS: Final[Counter] = _safe_metric(Counter,
    "ringrift_recovery_attempts_total",
    "Total recovery attempts by UnifiedHealthManager.",
    labelnames=("target_type", "action", "result"),
)

RECOVERY_ESCALATIONS: Final[Counter] = _safe_metric(Counter,
    "ringrift_recovery_escalations_total",
    "Total escalations to human operators.",
    labelnames=("reason",),
)

NODES_TRACKED: Final[Gauge] = _safe_metric(Gauge,
    "ringrift_recovery_nodes_tracked",
    "Number of nodes being tracked by UnifiedHealthManager.",
)

JOBS_TRACKED: Final[Gauge] = _safe_metric(Gauge,
    "ringrift_recovery_jobs_tracked",
    "Number of jobs being tracked by UnifiedHealthManager.",
)


# =============================================================================
# Bandwidth Manager Specific Metrics
# =============================================================================

BANDWIDTH_ALLOCATIONS_ACTIVE: Final[Gauge] = _safe_metric(Gauge,
    "ringrift_bandwidth_allocations_active",
    "Number of active bandwidth allocations.",
    labelnames=("host",),
)

BANDWIDTH_TRANSFERS_TOTAL: Final[Counter] = _safe_metric(Counter,
    "ringrift_bandwidth_transfers_total",
    "Total transfers completed through BandwidthManager.",
    labelnames=("host", "priority"),
)

BANDWIDTH_BYTES_TOTAL: Final[Counter] = _safe_metric(Counter,
    "ringrift_bandwidth_bytes_total",
    "Total bytes transferred through BandwidthManager.",
    labelnames=("host",),
)


# =============================================================================
# Sync Coordinator Specific Metrics
# =============================================================================

SYNC_HOSTS_TOTAL: Final[Gauge] = _safe_metric(Gauge,
    "ringrift_sync_hosts_total",
    "Total number of hosts tracked by SyncCoordinator.",
)

SYNC_HOSTS_HEALTHY: Final[Gauge] = _safe_metric(Gauge,
    "ringrift_sync_hosts_healthy",
    "Number of healthy hosts in SyncCoordinator.",
)

SYNC_HOSTS_STALE: Final[Gauge] = _safe_metric(Gauge,
    "ringrift_sync_hosts_stale",
    "Number of hosts with stale data.",
)

SYNC_HOSTS_CRITICAL: Final[Gauge] = _safe_metric(Gauge,
    "ringrift_sync_hosts_critical",
    "Number of hosts in critical sync state.",
)

SYNC_GAMES_UNSYNCED: Final[Gauge] = _safe_metric(Gauge,
    "ringrift_sync_games_unsynced",
    "Estimated number of unsynced games across cluster.",
)

SYNC_CLUSTER_HEALTH: Final[Gauge] = _safe_metric(Gauge,
    "ringrift_sync_cluster_health_score",
    "Overall cluster health score (0-100).",
)


# =============================================================================
# Status Mapping
# =============================================================================

STATUS_MAP = {
    "initializing": 0.5,
    "ready": 1,
    "running": 2,
    "paused": 3,
    "draining": 4,
    "error": 5,
    "stopped": 0,
}


# =============================================================================
# Update Functions
# =============================================================================

def update_coordinator_status(name: str, status: str) -> None:
    """Update coordinator status metric.

    Args:
        name: Coordinator name
        status: Status string (initializing, ready, running, paused, draining, error, stopped)
    """
    numeric_status = STATUS_MAP.get(status, 0)
    COORDINATOR_STATUS.labels(coordinator_name=name).set(numeric_status)


def update_coordinator_uptime(name: str, uptime_seconds: float) -> None:
    """Update coordinator uptime metric.

    Args:
        name: Coordinator name
        uptime_seconds: Uptime in seconds
    """
    COORDINATOR_UPTIME.labels(coordinator_name=name).set(uptime_seconds)


def record_coordinator_operation(
    name: str,
    operation_type: str,
    success: bool = True,
) -> None:
    """Record a coordinator operation.

    Args:
        name: Coordinator name
        operation_type: Type of operation (e.g., "sync", "request", "recover")
        success: Whether the operation succeeded
    """
    status = "success" if success else "failure"
    COORDINATOR_OPERATIONS.labels(
        coordinator_name=name,
        operation_type=operation_type,
        status=status,
    ).inc()


def record_coordinator_error(
    name: str,
    error_type: str = "unknown",
) -> None:
    """Record a coordinator error.

    Args:
        name: Coordinator name
        error_type: Type of error
    """
    COORDINATOR_ERRORS.labels(
        coordinator_name=name,
        error_type=error_type,
    ).inc()


# =============================================================================
# Health Manager Updates
# =============================================================================

def update_recovery_stats(stats: dict[str, Any]) -> None:
    """Update UnifiedHealthManager metrics from stats dict.

    Args:
        stats: Stats dict from UnifiedHealthManager.get_stats()
    """
    NODES_TRACKED.set(stats.get("nodes_tracked", 0))
    JOBS_TRACKED.set(stats.get("jobs_tracked", 0))

    # Record hourly stats
    hourly = stats.get("recoveries_last_hour", {})
    if hourly:
        for result, count in hourly.items():
            if count > 0:
                RECOVERY_ATTEMPTS.labels(
                    target_type="mixed",
                    action="recover",
                    result=result,
                ).inc(count)


# =============================================================================
# Bandwidth Manager Updates
# =============================================================================

def update_bandwidth_stats(stats: dict[str, Any]) -> None:
    """Update BandwidthManager metrics from stats dict.

    Args:
        stats: Stats dict from BandwidthManager.get_stats()
    """
    active = stats.get("active_allocations", {})
    for host, count in active.items():
        BANDWIDTH_ALLOCATIONS_ACTIVE.labels(host=host).set(count)

    history = stats.get("history_24h", {})
    for host, data in history.items():
        transfers = data.get("transfers", 0)
        if transfers > 0:
            BANDWIDTH_TRANSFERS_TOTAL.labels(
                host=host,
                priority="mixed",
            ).inc(transfers)


# =============================================================================
# Sync Coordinator Updates
# =============================================================================

def update_sync_stats(stats: dict[str, Any]) -> None:
    """Update SyncCoordinator metrics from stats dict.

    Args:
        stats: Stats dict from SyncCoordinator.get_stats()
    """
    SYNC_HOSTS_TOTAL.set(stats.get("total_hosts", 0))
    SYNC_HOSTS_HEALTHY.set(stats.get("healthy_hosts", 0))
    SYNC_HOSTS_STALE.set(stats.get("stale_hosts", 0))
    SYNC_HOSTS_CRITICAL.set(stats.get("critical_hosts", 0))
    SYNC_GAMES_UNSYNCED.set(stats.get("estimated_unsynced_games", 0))
    SYNC_CLUSTER_HEALTH.set(stats.get("cluster_health_score", 0))


# =============================================================================
# Aggregation Function
# =============================================================================

async def collect_all_coordinator_metrics() -> dict[str, Any]:
    """Collect metrics from all coordinators.

    Returns:
        Dict with aggregated metrics from all coordinators
    """
    metrics = {
        "coordinators": {},
        "total_operations": 0,
        "total_errors": 0,
    }

    # Try to get UnifiedHealthManager stats (consolidates RecoveryManager + ErrorRecoveryCoordinator)
    try:
        from app.coordination.unified_health_manager import UnifiedHealthManager
        uhm = UnifiedHealthManager.get_instance()
        stats = await uhm.get_stats()
        metrics["coordinators"]["UnifiedHealthManager"] = {
            "status": stats.get("status", "unknown"),
            "total_errors": stats.get("total_errors", 0),
            "errors_by_severity": stats.get("errors_by_severity", {}),
            "recovery_attempts": stats.get("recovery_attempts", 0),
            "successful_recoveries": stats.get("successful_recoveries", 0),
            "failed_recoveries": stats.get("failed_recoveries", 0),
            "recovery_rate": stats.get("recovery_rate", 0.0),
            "uptime_seconds": stats.get("uptime_seconds", 0),
        }
        update_coordinator_status("UnifiedHealthManager", stats.get("status", "unknown"))
        update_coordinator_uptime("UnifiedHealthManager", stats.get("uptime_seconds", 0))
        update_recovery_stats(stats)
    except Exception as e:
        logger.debug(f"Could not collect UnifiedHealthManager metrics: {e}")

    # Try to get BandwidthManager stats
    try:
        from app.coordination.bandwidth_manager import get_bandwidth_manager
        bm = get_bandwidth_manager()
        stats = await bm.get_stats()
        metrics["coordinators"]["BandwidthManager"] = stats
        update_coordinator_status("BandwidthManager", stats.get("status", "unknown"))
        update_coordinator_uptime("BandwidthManager", stats.get("uptime_seconds", 0))
        update_bandwidth_stats(stats)
    except Exception as e:
        logger.debug(f"Could not collect BandwidthManager metrics: {e}")

    # Try to get SyncScheduler stats (scheduling layer for cluster sync)
    try:
        from app.coordination.cluster.sync import SyncScheduler
        sc = SyncScheduler.get_instance()
        stats = await sc.get_stats()
        metrics["coordinators"]["SyncScheduler"] = stats
        update_coordinator_status("SyncScheduler", stats.get("status", "unknown"))
        update_coordinator_uptime("SyncScheduler", stats.get("uptime_seconds", 0))
        update_sync_stats(stats)
        SyncScheduler.reset_instance()  # Clean up singleton
    except Exception as e:
        logger.debug(f"Could not collect SyncScheduler metrics: {e}")

    return metrics


def collect_all_coordinator_metrics_sync() -> dict[str, Any]:
    """Synchronous wrapper for collect_all_coordinator_metrics."""
    try:
        asyncio.get_running_loop()
        # If there's a running loop, use the sync versions
        return _collect_sync()
    except RuntimeError:
        # No running loop, safe to create one
        return asyncio.run(collect_all_coordinator_metrics())


def _collect_sync() -> dict[str, Any]:
    """Synchronous collection using sync stat methods."""
    metrics = {"coordinators": {}}

    try:
        # Use UnifiedHealthManager (consolidates RecoveryManager + ErrorRecoveryCoordinator)
        from app.coordination.unified_health_manager import UnifiedHealthManager
        uhm = UnifiedHealthManager.get_instance()
        stats = uhm.get_status()
        metrics["coordinators"]["UnifiedHealthManager"] = {
            "status": stats.get("status", "unknown"),
            "total_errors": stats.get("total_errors", 0),
            "errors_by_severity": stats.get("errors_by_severity", {}),
            "recovery_attempts": stats.get("recovery_attempts", 0),
            "successful_recoveries": stats.get("successful_recoveries", 0),
            "failed_recoveries": stats.get("failed_recoveries", 0),
            "recovery_rate": stats.get("recovery_rate", 0.0),
            "uptime_seconds": round(uhm.uptime_seconds, 2),
        }
        update_coordinator_status("UnifiedHealthManager", stats.get("status", "unknown"))
        update_coordinator_uptime("UnifiedHealthManager", round(uhm.uptime_seconds, 2))
        update_recovery_stats(stats)
    except Exception as e:
        logger.debug(f"Could not collect UnifiedHealthManager metrics (sync): {e}")

    try:
        from app.coordination.bandwidth_manager import get_bandwidth_manager
        bm = get_bandwidth_manager()
        stats = bm.get_stats_sync()
        metrics["coordinators"]["BandwidthManager"] = stats
        update_coordinator_status("BandwidthManager", stats.get("status", "unknown"))
        update_bandwidth_stats(stats)
    except Exception as e:
        logger.debug(f"Could not collect BandwidthManager metrics (sync): {e}")

    try:
        from app.coordination.cluster.sync import SyncScheduler
        sc = SyncScheduler.get_instance()
        stats = sc.get_stats_sync()
        metrics["coordinators"]["SyncScheduler"] = stats
        update_coordinator_status("SyncScheduler", stats.get("status", "unknown"))
        update_sync_stats(stats)
        SyncScheduler.reset_instance()
    except Exception as e:
        logger.debug(f"Could not collect SyncScheduler metrics (sync): {e}")

    return metrics

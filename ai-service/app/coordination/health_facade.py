"""Unified Health Facade - Single entry point for all health operations.

This module consolidates health functionality from:
- unified_health_manager.py (system-level health, pipeline pause)
- health_check_orchestrator.py (node-level health, recovery escalation)
- daemon_manager.py (daemon health aggregation)
- sync infrastructure (sync health status)

Replaces deprecated modules (Q2 2026 removal):
- system_health_monitor.py
- node_health_monitor.py

Usage:
    from app.coordination.health_facade import (
        # System-level health
        get_system_health_score,
        get_system_health_level,
        should_pause_pipeline,

        # Node-level health
        get_node_health,
        get_healthy_nodes,
        get_unhealthy_nodes,

        # Job scheduling gate (December 2025)
        should_allow_new_jobs,

        # Dashboard (December 2025)
        ClusterHealthDashboard,
        ClusterHealthStatus,
        get_cluster_health_dashboard,

        # S3 health (January 2026 - Phase 3)
        get_s3_health,
        is_s3_healthy,
        get_s3_replication_lag,

        # Managers (for advanced use)
        get_health_manager,
        get_health_orchestrator,
    )

    # Check if new jobs should be allowed
    if should_allow_new_jobs():
        schedule_new_job()

    # Check if pipeline should pause
    should_pause, reasons = should_pause_pipeline()

    # Get individual node health
    node = get_node_health("node-001")
    if node and node.state == NodeHealthState.HEALTHY:
        # Node is healthy
        pass

    # Get comprehensive dashboard
    dashboard = get_cluster_health_dashboard()
    status = dashboard.get_cluster_health()
    print(f"Overall score: {status.overall_score}")

Created: December 2025
Purpose: Unified health interface (consolidation phase)
Updated: December 27, 2025 - Added ClusterHealthDashboard for job scheduling gates
Updated: January 2026 - Added canonical HealthStatus from Phase 4.1
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)


# =============================================================================
# Canonical Health Status (January 2026 - Phase 4.1)
# =============================================================================

# Re-export canonical HealthStatus for convenience
from app.coordination.health import (
    HealthStatus,
    HealthStatusInfo,
    to_health_status,
    from_legacy_health_state,
    from_legacy_health_level,
    from_legacy_system_health_level,
    from_legacy_node_health_state,
    get_health_score,
    from_health_score,
)


# Re-export system-level health from unified_health_manager
from app.coordination.unified_health_manager import (
    # Core manager
    get_health_manager,
    UnifiedHealthManager,
    # Health levels and config
    SystemHealthLevel,
    SystemHealthConfig,
    SystemHealthScore,
    # Convenience functions
    get_system_health_score,
    get_system_health_level,
    should_pause_pipeline,
    is_pipeline_paused,  # Deprecated but still available
)

# Re-export node-level health from health_check_orchestrator
from app.coordination.health_check_orchestrator import (
    # Core orchestrator
    get_health_orchestrator,
    HealthCheckOrchestrator,
    # Node health types
    NodeHealthState,
    NodeHealthDetails,
)
from app.coordination.singleton_mixin import SingletonMixin

if TYPE_CHECKING:
    from typing import Any


# =============================================================================
# Convenience Functions (unified interface)
# =============================================================================


def get_node_health_status(node_id: str) -> HealthStatus:
    """Get health status for a specific node in canonical form.

    January 2026: Uses canonical HealthStatus from Phase 4.1.

    Args:
        node_id: The node identifier

    Returns:
        HealthStatus (HEALTHY, DEGRADED, UNHEALTHY, OFFLINE, etc.)
    """
    details = get_health_orchestrator().get_node_health(node_id)
    if details is None:
        return HealthStatus.UNKNOWN
    return from_legacy_node_health_state(details.state)


def get_cluster_health_status() -> HealthStatus:
    """Get overall cluster health in canonical form.

    January 2026: Uses canonical HealthStatus from Phase 4.1.

    Returns:
        HealthStatus representing aggregate cluster health
    """
    level = get_system_health_level()
    return from_legacy_system_health_level(level)


def get_node_health(node_id: str) -> NodeHealthDetails | None:
    """Get health details for a specific node.

    Args:
        node_id: The node identifier

    Returns:
        NodeHealthDetails or None if node not tracked
    """
    return get_health_orchestrator().get_node_health(node_id)


def get_healthy_nodes() -> list[str]:
    """Get list of healthy node IDs.

    Returns:
        List of node IDs in HEALTHY state
    """
    orchestrator = get_health_orchestrator()
    return [
        node_id for node_id, details in orchestrator.node_health.items()
        if details.state == NodeHealthState.HEALTHY
    ]


def get_unhealthy_nodes() -> list[str]:
    """Get list of unhealthy node IDs.

    Returns:
        List of node IDs NOT in HEALTHY state
    """
    orchestrator = get_health_orchestrator()
    return [
        node_id for node_id, details in orchestrator.node_health.items()
        if details.state != NodeHealthState.HEALTHY
    ]


def get_degraded_nodes() -> list[str]:
    """Get list of degraded node IDs.

    Returns:
        List of node IDs in DEGRADED state
    """
    orchestrator = get_health_orchestrator()
    return [
        node_id for node_id, details in orchestrator.node_health.items()
        if details.state == NodeHealthState.DEGRADED
    ]


def get_offline_nodes() -> list[str]:
    """Get list of offline node IDs.

    Returns:
        List of node IDs in OFFLINE or RETIRED state
    """
    orchestrator = get_health_orchestrator()
    return [
        node_id for node_id, details in orchestrator.node_health.items()
        if details.state in (NodeHealthState.OFFLINE, NodeHealthState.RETIRED)
    ]


def mark_node_retired(node_id: str) -> bool:
    """Mark a node as retired (removed from active use).

    Args:
        node_id: The node to retire

    Returns:
        True if marked, False if node not found
    """
    return get_health_orchestrator().mark_retired(node_id)


def get_cluster_health_summary() -> dict[str, Any]:
    """Get a summary of cluster health.

    Returns:
        Dict with health counts and overall status
    """
    orchestrator = get_health_orchestrator()
    manager = get_health_manager()

    health_counts = {
        NodeHealthState.HEALTHY.value: 0,
        NodeHealthState.DEGRADED.value: 0,
        NodeHealthState.UNHEALTHY.value: 0,
        NodeHealthState.OFFLINE.value: 0,
        NodeHealthState.RETIRED.value: 0,
    }

    for details in orchestrator.node_health.values():
        state_value = details.state.value
        if state_value in health_counts:
            health_counts[state_value] += 1

    system_score = manager.calculate_system_health_score()
    should_pause, pause_reasons = should_pause_pipeline()

    return {
        "total_nodes": len(orchestrator.node_health),
        "node_counts": health_counts,
        "system_score": system_score.score,
        "system_level": system_score.level.value,
        "pipeline_paused": should_pause,
        "pause_reasons": pause_reasons,
    }


# =============================================================================
# Phase 3 (Jan 2026): S3 Health Monitoring
# =============================================================================


def get_s3_health() -> dict[str, Any]:
    """Get S3 storage tier health status.

    Phase 3 of S3-as-primary-storage: Monitor S3 replication lag and health.

    January 2026: Added as part of S3 first-class storage tier upgrade.

    Returns:
        Dict with S3 health metrics including:
        - enabled: Whether S3 is configured
        - healthy: Whether S3 is reachable
        - replication_lag_seconds: Seconds since last successful push
        - bucket: S3 bucket name
        - last_push_time: Timestamp of last successful push
    """
    try:
        from app.coordination.sync_router import get_sync_router
        from app.coordination.s3_sync_daemon import get_s3_sync_daemon

        router = get_sync_router()
        s3_config = router.get_s3_config()

        if not s3_config.enabled:
            return {
                "enabled": False,
                "healthy": None,
                "replication_lag_seconds": None,
                "bucket": None,
                "last_push_time": None,
            }

        # Get S3 daemon stats
        try:
            daemon = get_s3_sync_daemon()
            stats = daemon.get_stats()
            last_push_time = stats.get("last_event_driven_push_time", 0) or stats.get("last_sync_time", 0)
            event_driven_pushes = stats.get("event_driven_pushes", 0)
            total_files_pushed = stats.get("total_files_pushed", 0)
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.debug(f"Could not get S3 daemon stats: {e}")
            last_push_time = 0
            event_driven_pushes = 0
            total_files_pushed = 0

        # Calculate replication lag
        replication_lag = int(time.time() - last_push_time) if last_push_time > 0 else None

        # Check S3 health via router
        s3_healthy = router._check_s3_health()

        return {
            "enabled": True,
            "healthy": s3_healthy,
            "replication_lag_seconds": replication_lag,
            "bucket": s3_config.bucket,
            "last_push_time": last_push_time,
            "event_driven_pushes": event_driven_pushes,
            "total_files_pushed": total_files_pushed,
            "primary_for_games": s3_config.primary_for_games,
            # Alert if lag > 15 minutes (900 seconds)
            "lag_alert": replication_lag is not None and replication_lag > 900,
        }

    except (ImportError, AttributeError, RuntimeError) as e:
        logger.warning(f"Could not get S3 health: {e}")
        return {
            "enabled": None,
            "healthy": False,
            "replication_lag_seconds": None,
            "bucket": None,
            "last_push_time": None,
            "error": str(e),
        }


def is_s3_healthy() -> bool:
    """Check if S3 storage tier is healthy.

    Returns:
        True if S3 is enabled and reachable
    """
    health = get_s3_health()
    return health.get("enabled", False) and health.get("healthy", False)


def get_s3_replication_lag() -> int | None:
    """Get S3 replication lag in seconds.

    Returns:
        Seconds since last push, or None if unknown
    """
    health = get_s3_health()
    return health.get("replication_lag_seconds")


# =============================================================================
# Backward Compatibility (deprecated functions)
# =============================================================================

def get_node_health_monitor() -> "HealthCheckOrchestrator":
    """DEPRECATED: Use get_health_orchestrator() instead.

    Returns the HealthCheckOrchestrator for backward compatibility.

    Returns:
        HealthCheckOrchestrator instance
    """
    warnings.warn(
        "get_node_health_monitor() is deprecated. Use get_health_orchestrator() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_health_orchestrator()


def get_system_health() -> "UnifiedHealthManager":
    """DEPRECATED: Use get_health_manager() instead.

    Returns the UnifiedHealthManager for backward compatibility.

    Returns:
        UnifiedHealthManager instance
    """
    warnings.warn(
        "get_system_health() is deprecated. Use get_health_manager() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_health_manager()


# =============================================================================
# Cluster Health Dashboard (December 27, 2025)
# =============================================================================

# Default health score threshold for allowing new jobs
DEFAULT_JOB_SCHEDULING_THRESHOLD = 60


@dataclass
class DaemonHealthSummary:
    """Summary of daemon health status."""

    total_daemons: int = 0
    running: int = 0
    stopped: int = 0
    failed: int = 0
    healthy_ratio: float = 0.0
    critical_daemons_healthy: bool = True
    critical_failures: list[str] = field(default_factory=list)


@dataclass
class SyncHealthSummary:
    """Summary of sync infrastructure health."""

    last_sync_time: float = 0.0
    seconds_since_sync: float = 0.0
    sync_healthy: bool = True
    consecutive_failures: int = 0
    data_server_healthy: bool = True
    transport_available: dict[str, bool] = field(default_factory=dict)


@dataclass
class DataSyncHealthSummary:
    """Summary of unified data sync health (January 2026).

    Tracks backup status to S3 and OWC for disaster recovery readiness.
    """

    # Backup status
    s3_healthy: bool = True
    owc_healthy: bool = True
    last_s3_backup_time: float = 0.0
    last_owc_backup_time: float = 0.0

    # Counters
    s3_backups_succeeded: int = 0
    s3_backups_failed: int = 0
    owc_backups_succeeded: int = 0
    owc_backups_failed: int = 0

    # Replication status
    pending_backups: int = 0
    under_replicated_count: int = 0
    games_with_s3_backup: int = 0
    games_with_owc_backup: int = 0

    @property
    def is_healthy(self) -> bool:
        """Check if data sync is healthy."""
        return self.s3_healthy and self.owc_healthy

    @property
    def has_redundancy(self) -> bool:
        """Check if data has redundant backups."""
        return self.games_with_s3_backup > 0 and self.games_with_owc_backup > 0


@dataclass
class ClusterHealthStatus:
    """Comprehensive cluster health status.

    This aggregates health from multiple sources:
    - System-level health score (from UnifiedHealthManager)
    - Node-level health (from HealthCheckOrchestrator)
    - Daemon health (from DaemonManager)
    - Sync health (from sync infrastructure)

    Usage:
        dashboard = get_cluster_health_dashboard()
        status = dashboard.get_cluster_health()
        if status.overall_score >= 60:
            # Cluster is healthy enough for new jobs
            pass
    """

    overall_score: float = 0.0
    overall_level: str = "unknown"
    timestamp: float = field(default_factory=time.time)

    # Component health
    system_score: float = 0.0
    node_score: float = 0.0
    daemon_score: float = 0.0
    sync_score: float = 0.0

    # Detailed summaries
    node_counts: dict[str, int] = field(default_factory=dict)
    daemon_health: DaemonHealthSummary = field(default_factory=DaemonHealthSummary)
    sync_health: SyncHealthSummary = field(default_factory=SyncHealthSummary)
    data_sync_health: DataSyncHealthSummary = field(default_factory=DataSyncHealthSummary)

    # Aggregated status
    total_nodes: int = 0
    healthy_nodes: int = 0
    pipeline_paused: bool = False
    pause_reasons: list[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        """Check if cluster is in healthy state."""
        return self.overall_score >= DEFAULT_JOB_SCHEDULING_THRESHOLD

    @property
    def can_schedule_jobs(self) -> bool:
        """Check if new jobs should be scheduled."""
        return (
            self.overall_score >= DEFAULT_JOB_SCHEDULING_THRESHOLD
            and not self.pipeline_paused
            and self.daemon_health.critical_daemons_healthy
        )


class ClusterHealthDashboard(SingletonMixin):
    """Unified cluster health aggregation (December 2025).

    This class provides a single source of truth for cluster health by
    aggregating health data from multiple sources:

    1. System health (UnifiedHealthManager)
    2. Node health (HealthCheckOrchestrator)
    3. Daemon health (DaemonManager)
    4. Sync health (SyncCoordinator)

    January 2026: Migrated to use SingletonMixin for consistency.

    Usage:
        dashboard = ClusterHealthDashboard.get_instance()
        status = dashboard.get_cluster_health()

        if dashboard.should_allow_new_jobs():
            schedule_new_job()

        # Or with custom threshold
        if dashboard.should_allow_new_jobs(threshold=80):
            schedule_high_priority_job()
    """

    # Critical daemons that must be healthy for job scheduling
    CRITICAL_DAEMONS = frozenset({
        "EVENT_ROUTER",
        "AUTO_SYNC",
        "DATA_PIPELINE",
        "SELFPLAY_COORDINATOR",
    })

    # Health score weights for overall calculation
    WEIGHT_SYSTEM = 0.35
    WEIGHT_NODE = 0.30
    WEIGHT_DAEMON = 0.20
    WEIGHT_SYNC = 0.15

    def __init__(self) -> None:
        """Initialize the dashboard."""
        self._last_status: ClusterHealthStatus | None = None
        self._last_check: float = 0.0
        self._cache_ttl: float = 5.0  # Cache for 5 seconds

    def get_cluster_health(self, force_refresh: bool = False) -> ClusterHealthStatus:
        """Get comprehensive cluster health status.

        Args:
            force_refresh: If True, bypass cache and recalculate

        Returns:
            ClusterHealthStatus with aggregated health data
        """
        now = time.time()

        # Return cached status if still valid
        if (
            not force_refresh
            and self._last_status is not None
            and (now - self._last_check) < self._cache_ttl
        ):
            return self._last_status

        # Calculate fresh status
        status = self._calculate_health_status()
        self._last_status = status
        self._last_check = now
        return status

    def _calculate_health_status(self) -> ClusterHealthStatus:
        """Calculate comprehensive health status."""
        # Get system health
        system_score_obj = get_system_health_score()
        system_score = system_score_obj.score if system_score_obj else 0.0
        system_level = (
            system_score_obj.level.value
            if system_score_obj
            else SystemHealthLevel.CRITICAL.value
        )

        # Get node health
        node_counts = self._get_node_counts()
        node_score = self._calculate_node_score(node_counts)

        # Get daemon health
        daemon_summary = self._get_daemon_health()
        daemon_score = daemon_summary.healthy_ratio * 100

        # Get sync health
        sync_summary = self._get_sync_health()
        sync_score = 100.0 if sync_summary.sync_healthy else 50.0
        if sync_summary.consecutive_failures > 0:
            sync_score = max(0, sync_score - sync_summary.consecutive_failures * 10)

        # Get data sync health (January 2026)
        data_sync_summary = self._get_data_sync_health()

        # Calculate overall score (weighted average)
        overall_score = (
            self.WEIGHT_SYSTEM * system_score
            + self.WEIGHT_NODE * node_score
            + self.WEIGHT_DAEMON * daemon_score
            + self.WEIGHT_SYNC * sync_score
        )

        # Determine overall level
        if overall_score >= 80:
            overall_level = SystemHealthLevel.HEALTHY.value
        elif overall_score >= 60:
            overall_level = SystemHealthLevel.DEGRADED.value
        elif overall_score >= 40:
            overall_level = SystemHealthLevel.UNHEALTHY.value
        else:
            overall_level = SystemHealthLevel.CRITICAL.value

        # Get pipeline status
        should_pause, pause_reasons = should_pause_pipeline()

        return ClusterHealthStatus(
            overall_score=overall_score,
            overall_level=overall_level,
            timestamp=time.time(),
            system_score=system_score,
            node_score=node_score,
            daemon_score=daemon_score,
            sync_score=sync_score,
            node_counts=node_counts,
            daemon_health=daemon_summary,
            sync_health=sync_summary,
            data_sync_health=data_sync_summary,
            total_nodes=sum(node_counts.values()),
            healthy_nodes=node_counts.get(NodeHealthState.HEALTHY.value, 0),
            pipeline_paused=should_pause,
            pause_reasons=pause_reasons,
        )

    def _get_node_counts(self) -> dict[str, int]:
        """Get node counts by health state."""
        orchestrator = get_health_orchestrator()
        counts: dict[str, int] = {
            NodeHealthState.HEALTHY.value: 0,
            NodeHealthState.DEGRADED.value: 0,
            NodeHealthState.UNHEALTHY.value: 0,
            NodeHealthState.OFFLINE.value: 0,
            NodeHealthState.RETIRED.value: 0,
        }
        for details in orchestrator.node_health.values():
            state_value = details.state.value
            if state_value in counts:
                counts[state_value] += 1
        return counts

    def _calculate_node_score(self, node_counts: dict[str, int]) -> float:
        """Calculate node health score (0-100)."""
        total = sum(node_counts.values())
        if total == 0:
            return 0.0

        healthy = node_counts.get(NodeHealthState.HEALTHY.value, 0)
        degraded = node_counts.get(NodeHealthState.DEGRADED.value, 0)

        # Healthy nodes count fully, degraded count half
        score = ((healthy * 1.0) + (degraded * 0.5)) / total * 100
        return min(100.0, max(0.0, score))

    def _get_daemon_health(self) -> DaemonHealthSummary:
        """Get daemon health summary."""
        try:
            from app.coordination.daemon_manager import get_daemon_manager
            from app.coordination.daemon_types import DaemonState

            dm = get_daemon_manager()
            all_health = dm.get_all_daemon_health()

            running = 0
            stopped = 0
            failed = 0
            critical_failures: list[str] = []

            for daemon_type, health in all_health.items():
                state = health.get("state", "unknown")
                if state == DaemonState.RUNNING.value:
                    running += 1
                elif state == DaemonState.STOPPED.value:
                    stopped += 1
                elif state == DaemonState.FAILED.value:
                    failed += 1
                    # Check if it's a critical daemon
                    if daemon_type.name in self.CRITICAL_DAEMONS:
                        critical_failures.append(daemon_type.name)

            total = running + stopped + failed
            healthy_ratio = running / total if total > 0 else 0.0

            return DaemonHealthSummary(
                total_daemons=total,
                running=running,
                stopped=stopped,
                failed=failed,
                healthy_ratio=healthy_ratio,
                critical_daemons_healthy=len(critical_failures) == 0,
                critical_failures=critical_failures,
            )

        except ImportError:
            logger.debug("DaemonManager not available for health check")
            return DaemonHealthSummary()
        except Exception as e:
            logger.debug(f"Error getting daemon health: {e}")
            return DaemonHealthSummary()

    def _get_sync_health(self) -> SyncHealthSummary:
        """Get sync infrastructure health summary."""
        try:
            from app.distributed.sync_coordinator import SyncCoordinator

            sync = SyncCoordinator.get_instance()
            health = sync.health_check()

            return SyncHealthSummary(
                last_sync_time=health.details.get("last_sync_time", 0.0),
                seconds_since_sync=health.details.get("seconds_since_last_sync", 0.0),
                sync_healthy=health.status == "healthy",
                consecutive_failures=health.details.get("consecutive_failures", 0),
                data_server_healthy=health.details.get("data_server_healthy", True),
                transport_available=health.details.get("transport_available", {}),
            )

        except ImportError:
            logger.debug("SyncCoordinator not available for health check")
            return SyncHealthSummary(sync_healthy=True)  # Assume healthy if not available
        except Exception as e:
            logger.debug(f"Error getting sync health: {e}")
            return SyncHealthSummary()

    def _get_data_sync_health(self) -> DataSyncHealthSummary:
        """Get unified data sync health summary (January 2026).

        Collects metrics from UnifiedDataSyncOrchestrator for backup status.
        """
        try:
            from app.coordination.unified_data_sync_orchestrator import (
                get_unified_data_sync_orchestrator,
            )

            orchestrator = get_unified_data_sync_orchestrator()
            metrics = orchestrator.get_metrics()

            return DataSyncHealthSummary(
                s3_healthy=metrics.get("s3_backups_failed", 0) < 5,
                owc_healthy=metrics.get("owc_backups_failed", 0) < 5,
                last_s3_backup_time=metrics.get("last_s3_backup_time", 0.0),
                last_owc_backup_time=metrics.get("last_owc_backup_time", 0.0),
                s3_backups_succeeded=metrics.get("s3_backups_succeeded", 0),
                s3_backups_failed=metrics.get("s3_backups_failed", 0),
                owc_backups_succeeded=metrics.get("owc_backups_succeeded", 0),
                owc_backups_failed=metrics.get("owc_backups_failed", 0),
                pending_backups=metrics.get("pending_backups", 0),
                under_replicated_count=metrics.get("under_replicated_count", 0),
            )

        except ImportError:
            logger.debug("UnifiedDataSyncOrchestrator not available for health check")
            return DataSyncHealthSummary()  # Default healthy state
        except Exception as e:
            logger.debug(f"Error getting data sync health: {e}")
            return DataSyncHealthSummary()

    def should_allow_new_jobs(
        self,
        threshold: float = DEFAULT_JOB_SCHEDULING_THRESHOLD,
    ) -> bool:
        """Check if new jobs should be scheduled based on cluster health.

        This is the primary gate for job scheduling decisions. It considers:
        1. Overall health score vs threshold
        2. Pipeline pause status
        3. Critical daemon health

        Args:
            threshold: Minimum health score to allow jobs (default: 60)

        Returns:
            True if new jobs should be allowed, False otherwise
        """
        status = self.get_cluster_health()
        return (
            status.overall_score >= threshold
            and not status.pipeline_paused
            and status.daemon_health.critical_daemons_healthy
        )

    def get_scheduling_recommendation(self) -> tuple[bool, str]:
        """Get job scheduling recommendation with reason.

        Returns:
            Tuple of (should_schedule, reason_message)
        """
        status = self.get_cluster_health()

        if status.overall_score < DEFAULT_JOB_SCHEDULING_THRESHOLD:
            return (
                False,
                f"Cluster health score ({status.overall_score:.1f}) "
                f"below threshold ({DEFAULT_JOB_SCHEDULING_THRESHOLD})"
            )

        if status.pipeline_paused:
            reasons = ", ".join(status.pause_reasons) if status.pause_reasons else "unknown"
            return False, f"Pipeline is paused: {reasons}"

        if not status.daemon_health.critical_daemons_healthy:
            failures = ", ".join(status.daemon_health.critical_failures)
            return False, f"Critical daemons unhealthy: {failures}"

        return True, f"Cluster healthy (score: {status.overall_score:.1f})"


# Singleton accessor
def get_cluster_health_dashboard() -> ClusterHealthDashboard:
    """Get the singleton ClusterHealthDashboard instance.

    Returns:
        ClusterHealthDashboard instance
    """
    return ClusterHealthDashboard.get_instance()


# Convenience function for job scheduling
def should_allow_new_jobs(
    threshold: float = DEFAULT_JOB_SCHEDULING_THRESHOLD,
) -> bool:
    """Check if new jobs should be scheduled based on cluster health.

    This is a convenience wrapper around ClusterHealthDashboard.should_allow_new_jobs().

    Args:
        threshold: Minimum health score to allow jobs (default: 60)

    Returns:
        True if new jobs should be allowed, False otherwise

    Example:
        if should_allow_new_jobs():
            orchestrator.spawn_selfplay_job(config)
        else:
            logger.warning("Cluster unhealthy, deferring job scheduling")
    """
    return get_cluster_health_dashboard().should_allow_new_jobs(threshold)


def get_daemon_health_summary() -> DaemonHealthSummary:
    """Get a summary of daemon health.

    Returns:
        DaemonHealthSummary with daemon health statistics
    """
    return get_cluster_health_dashboard()._get_daemon_health()


def get_sync_health_summary() -> SyncHealthSummary:
    """Get a summary of sync infrastructure health.

    Returns:
        SyncHealthSummary with sync health statistics
    """
    return get_cluster_health_dashboard()._get_sync_health()


def get_data_sync_health_summary() -> DataSyncHealthSummary:
    """Get a summary of unified data sync health (January 2026).

    Returns:
        DataSyncHealthSummary with backup status to S3 and OWC
    """
    return get_cluster_health_dashboard()._get_data_sync_health()


# =============================================================================
# Coordinator Health Check (January 2026 - Phase 4)
# =============================================================================


@dataclass
class CoordinatorHealthResult:
    """Result of coordinator health check.

    Attributes:
        is_healthy: Overall health status
        owc_mounted: Whether OWC drive is mounted
        ssh_key_exists: Whether cluster SSH key exists
        tailscale_connected: Whether Tailscale is connected
        p2p_reachable: Whether P2P orchestrator is reachable
        details: Additional details about each check
    """

    is_healthy: bool = True
    owc_mounted: bool = False
    ssh_key_exists: bool = False
    tailscale_connected: bool = False
    p2p_reachable: bool = False
    details: dict = field(default_factory=dict)


def check_coordinator_health() -> CoordinatorHealthResult:
    """Verify coordinator-specific requirements.

    Checks:
    1. OWC drive mounted at /Volumes/RingRift-Data
    2. SSH key exists at ~/.ssh/id_cluster
    3. Tailscale connected (has 100.x.x.x IP)
    4. P2P orchestrator reachable

    Returns:
        CoordinatorHealthResult with all check results

    Example:
        from app.coordination.health_facade import check_coordinator_health

        result = check_coordinator_health()
        if not result.is_healthy:
            print(f"Coordinator unhealthy: {result.details}")
    """
    import os
    import subprocess
    from pathlib import Path

    result = CoordinatorHealthResult()
    issues: list[str] = []

    # Check 1: OWC drive mounted
    owc_path = Path("/Volumes/RingRift-Data")
    if owc_path.exists() and owc_path.is_dir():
        result.owc_mounted = True
        result.details["owc_path"] = str(owc_path)
        # Check if it has expected subdirectories
        expected_dirs = ["selfplay_repository", "canonical_models"]
        found_dirs = [d for d in expected_dirs if (owc_path / d).exists()]
        result.details["owc_subdirs"] = found_dirs
        if len(found_dirs) < len(expected_dirs):
            issues.append(f"OWC missing subdirs: {set(expected_dirs) - set(found_dirs)}")
    else:
        issues.append("OWC drive not mounted at /Volumes/RingRift-Data")

    # Check 2: SSH key exists
    ssh_key_path = Path.home() / ".ssh" / "id_cluster"
    if ssh_key_path.exists():
        result.ssh_key_exists = True
        result.details["ssh_key_path"] = str(ssh_key_path)
    else:
        issues.append("SSH key not found at ~/.ssh/id_cluster")

    # Check 3: Tailscale connected
    try:
        ts_result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if ts_result.returncode == 0:
            ts_ip = ts_result.stdout.strip()
            if ts_ip.startswith("100."):
                result.tailscale_connected = True
                result.details["tailscale_ip"] = ts_ip
            else:
                issues.append(f"Tailscale IP not in 100.x range: {ts_ip}")
        else:
            issues.append("Tailscale not running")
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        issues.append(f"Tailscale check failed: {e}")

    # Check 4: P2P orchestrator reachable
    try:
        from app.config.ports import get_local_p2p_url
        import urllib.request

        p2p_url = f"{get_local_p2p_url()}/health"
        with urllib.request.urlopen(p2p_url, timeout=5) as resp:
            if resp.status == 200:
                result.p2p_reachable = True
                result.details["p2p_url"] = p2p_url
    except Exception as e:
        issues.append(f"P2P orchestrator not reachable: {e}")

    # Determine overall health
    result.is_healthy = (
        result.owc_mounted
        and result.ssh_key_exists
        and result.tailscale_connected
        and result.p2p_reachable
    )

    if issues:
        result.details["issues"] = issues

    return result


__all__ = [
    # System-level health
    "get_health_manager",
    "UnifiedHealthManager",
    "SystemHealthLevel",
    "SystemHealthConfig",
    "SystemHealthScore",
    "get_system_health_score",
    "get_system_health_level",
    "should_pause_pipeline",
    "is_pipeline_paused",
    # Node-level health
    "get_health_orchestrator",
    "HealthCheckOrchestrator",
    "NodeHealthState",
    "NodeHealthDetails",
    "get_node_health",
    "get_healthy_nodes",
    "get_unhealthy_nodes",
    "get_degraded_nodes",
    "get_offline_nodes",
    "mark_node_retired",
    # Cluster summary
    "get_cluster_health_summary",
    # Cluster Health Dashboard (December 2025)
    "ClusterHealthDashboard",
    "ClusterHealthStatus",
    "DaemonHealthSummary",
    "SyncHealthSummary",
    "DataSyncHealthSummary",  # January 2026
    "get_cluster_health_dashboard",
    "should_allow_new_jobs",
    "get_daemon_health_summary",
    "get_sync_health_summary",
    "get_data_sync_health_summary",  # January 2026
    "DEFAULT_JOB_SCHEDULING_THRESHOLD",
    # Coordinator health (January 2026 - Phase 4)
    "CoordinatorHealthResult",
    "check_coordinator_health",
    # Backward compat (deprecated)
    "get_node_health_monitor",
    "get_system_health",
]

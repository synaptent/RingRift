"""Shared interfaces/protocols for coordination modules.

These protocols break circular dependencies by providing abstract
interfaces that modules can depend on instead of concrete implementations.

CRITICAL: This module has ZERO dependencies on other coordination modules.
Only standard library and typing imports are allowed.

Circular Dependencies Broken:
- selfplay_scheduler.py:84 <-> backpressure.py (via IBackpressureMonitor)
- resource_optimizer.py:70 <-> resource_targets.py (via IResourceTargetManager)

Usage:
    from app.coordination.interfaces import (
        IBackpressureMonitor,
        IResourceTargetManager,
        IScheduler,
        IHealthChecker,
    )

    class MyService:
        def __init__(self, backpressure: IBackpressureMonitor | None = None):
            self._backpressure = backpressure

        def _get_backpressure_level(self) -> float:
            if self._backpressure is None:
                # Lazy import to break circular dependency
                from app.coordination.backpressure import get_backpressure_monitor
                self._backpressure = get_backpressure_monitor()
            return self._backpressure.get_backpressure_level()

Created: December 2025
Purpose: Break circular dependencies in coordination modules
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable

__all__ = [
    # Backpressure Protocol
    "IBackpressureMonitor",
    "IBackpressureSignal",
    # Resource Target Protocol
    "IResourceTargetManager",
    "IResourceTargets",
    # Scheduler Protocol
    "IScheduler",
    "IJobInfo",
    # Health Protocol
    "IHealthChecker",
    "IHealthResult",
    # Sync Protocol
    "ISyncProvider",
]


def __dir__() -> list[str]:
    """Expose the intended protocol surface for discoverability and tests."""

    return sorted(set(globals()) | set(__all__))


# =============================================================================
# Backpressure Interfaces
# =============================================================================


@runtime_checkable
class IBackpressureSignal(Protocol):
    """Protocol for backpressure signal data.

    Implementations: app.coordination.backpressure.BackpressureSignal
    """

    @property
    def queue_pressure(self) -> float:
        """Selfplay queue depth pressure (0.0-1.0)."""
        ...

    @property
    def training_pressure(self) -> float:
        """Active training jobs pressure (0.0-1.0)."""
        ...

    @property
    def disk_pressure(self) -> float:
        """Cluster disk usage pressure (0.0-1.0)."""
        ...

    @property
    def sync_pressure(self) -> float:
        """Pending data syncs pressure (0.0-1.0)."""
        ...

    @property
    def overall_pressure(self) -> float:
        """Weighted average of all pressure sources (0.0-1.0)."""
        ...

    @property
    def spawn_rate_multiplier(self) -> float:
        """Rate multiplier based on pressure (0.0-1.0)."""
        ...

    @property
    def should_pause(self) -> bool:
        """Whether operations should pause entirely."""
        ...

    @property
    def is_healthy(self) -> bool:
        """Whether system is healthy (no significant pressure)."""
        ...


@runtime_checkable
class IBackpressureMonitor(Protocol):
    """Protocol for backpressure monitoring.

    Implementations: app.coordination.backpressure.BackpressureMonitor

    Used by: SelfplayScheduler, IdleResourceDaemon, JobManager
    """

    async def get_signal(self, force_refresh: bool = False) -> IBackpressureSignal:
        """Get current backpressure signal.

        Args:
            force_refresh: Force refresh even if cache is valid

        Returns:
            Current backpressure signal
        """
        ...

    def get_cached_signal(self) -> Optional[IBackpressureSignal]:
        """Get cached signal without refreshing.

        Returns:
            Cached signal or None if not cached
        """
        ...

    def get_backpressure_level(self) -> float:
        """Get current backpressure level (0.0-1.0).

        Convenience method returning overall_pressure from cached signal.

        Returns:
            Overall pressure level (0.0 = no pressure, 1.0 = max pressure)
        """
        ...

    def is_under_pressure(self) -> bool:
        """Check if system is under backpressure.

        Returns:
            True if overall_pressure > 0.5
        """
        ...


# =============================================================================
# Resource Target Interfaces
# =============================================================================


@runtime_checkable
class IResourceTargets(Protocol):
    """Protocol for utilization targets.

    Implementations: app.coordination.resource_targets.UtilizationTargets
    """

    @property
    def cpu_min(self) -> float:
        """Minimum CPU utilization target (%)."""
        ...

    @property
    def cpu_target(self) -> float:
        """Optimal CPU utilization target (%)."""
        ...

    @property
    def cpu_max(self) -> float:
        """Maximum CPU utilization target (%)."""
        ...

    @property
    def gpu_min(self) -> float:
        """Minimum GPU utilization target (%)."""
        ...

    @property
    def gpu_target(self) -> float:
        """Optimal GPU utilization target (%)."""
        ...

    @property
    def gpu_max(self) -> float:
        """Maximum GPU utilization target (%)."""
        ...


@runtime_checkable
class IResourceTargetManager(Protocol):
    """Protocol for resource target management.

    Implementations: app.coordination.resource_targets.ResourceTargetManager

    Used by: ResourceOptimizer, SelfplayScheduler, IdleResourceDaemon
    """

    def get_resource_targets(self) -> IResourceTargets:
        """Get global resource targets.

        Returns:
            Current utilization targets
        """
        ...

    def get_target_utilization(self, host: str) -> float:
        """Get target utilization for a host.

        Args:
            host: Host name

        Returns:
            Target utilization percentage (0-100)
        """
        ...

    def should_scale_up(self, host: str, current_util: float = 0.0) -> bool:
        """Check if host should scale up.

        Args:
            host: Host name
            current_util: Current utilization (optional)

        Returns:
            True if utilization is below target
        """
        ...

    def should_scale_down(self, host: str, current_util: float = 0.0) -> bool:
        """Check if host should scale down.

        Args:
            host: Host name
            current_util: Current utilization (optional)

        Returns:
            True if utilization is above target
        """
        ...

    def get_utilization_score(self, host: str, current_util: float) -> float:
        """Get utilization score for scheduling.

        Args:
            host: Host name
            current_util: Current utilization

        Returns:
            Score (0.0-1.0) where 1.0 = optimal utilization
        """
        ...


# =============================================================================
# Scheduler Interfaces
# =============================================================================


@runtime_checkable
class IJobInfo(Protocol):
    """Protocol for job information.

    Minimal interface for job metadata needed by schedulers.
    """

    @property
    def job_id(self) -> str:
        """Unique job identifier."""
        ...

    @property
    def config_key(self) -> str:
        """Configuration key (e.g., 'hex8_2p')."""
        ...

    @property
    def node_id(self) -> str:
        """Node where job is running."""
        ...

    @property
    def status(self) -> str:
        """Job status (pending, running, completed, failed)."""
        ...


@runtime_checkable
class IScheduler(Protocol):
    """Protocol for job scheduling.

    Implementations:
        - app.coordination.selfplay_scheduler.SelfplayScheduler
        - app.coordination.unified_scheduler.UnifiedScheduler

    Used by: JobManager, P2POrchestrator, IdleResourceDaemon
    """

    def get_next_job(self) -> Optional[Dict[str, Any]]:
        """Get next job to execute.

        Returns:
            Job specification dict or None if no jobs pending
        """
        ...

    def mark_complete(self, job_id: str) -> None:
        """Mark job as complete.

        Args:
            job_id: Job identifier
        """
        ...

    def mark_failed(self, job_id: str, error: str = "") -> None:
        """Mark job as failed.

        Args:
            job_id: Job identifier
            error: Error message
        """
        ...

    def get_pending_count(self) -> int:
        """Get number of pending jobs.

        Returns:
            Count of pending jobs
        """
        ...


# =============================================================================
# Health Check Interfaces
# =============================================================================


@runtime_checkable
class IHealthResult(Protocol):
    """Protocol for health check results.

    Implementations: app.coordination.contracts.HealthCheckResult
    """

    @property
    def healthy(self) -> bool:
        """Whether the component is healthy."""
        ...

    @property
    def message(self) -> str:
        """Human-readable status message."""
        ...

    @property
    def details(self) -> Dict[str, Any]:
        """Additional details about health status."""
        ...


@runtime_checkable
class IHealthChecker(Protocol):
    """Protocol for health checking.

    Implementations: All coordinators that implement health_check()

    Used by: DaemonManager, HealthCheckOrchestrator
    """

    def health_check(self) -> IHealthResult:
        """Return health check result.

        Returns:
            Health check result with status details
        """
        ...


# =============================================================================
# Sync Interfaces
# =============================================================================


@runtime_checkable
class ISyncProvider(Protocol):
    """Protocol for data synchronization.

    Implementations:
        - app.coordination.auto_sync_daemon.AutoSyncDaemon
        - app.coordination.sync_router.SyncRouter

    Used by: DataPipelineOrchestrator, SelfplayScheduler
    """

    async def trigger_sync(
        self,
        config_key: Optional[str] = None,
        priority: bool = False,
    ) -> bool:
        """Trigger a data sync operation.

        Args:
            config_key: Optional config to sync (None = all)
            priority: Whether this is a priority sync

        Returns:
            True if sync was triggered
        """
        ...

    def is_syncing(self) -> bool:
        """Check if a sync is in progress.

        Returns:
            True if sync operation is running
        """
        ...

    def get_last_sync_time(self) -> Optional[float]:
        """Get timestamp of last completed sync.

        Returns:
            Unix timestamp or None if never synced
        """
        ...

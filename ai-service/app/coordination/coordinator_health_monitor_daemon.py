"""Coordinator Health Monitor Daemon - Tracks coordinator lifecycle events.

Subscribes to all COORDINATOR_* events and provides:
1. Coordinator health tracking (healthy/unhealthy/degraded)
2. Heartbeat freshness monitoring
3. Init failure tracking
4. Shutdown detection
5. Cluster-wide coordinator health summary

Usage:
    from app.coordination.coordinator_health_monitor_daemon import (
        CoordinatorHealthMonitorDaemon,
        get_coordinator_health_monitor,
    )

    monitor = get_coordinator_health_monitor()
    await monitor.start()

    summary = monitor.get_health_summary()
    print(f"Healthy: {summary.healthy_count}/{summary.total_count}")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

from app.config.coordination_defaults import CoordinatorHealthDefaults
from app.coordination.contracts import CoordinatorStatus, HealthCheckResult
from app.coordination.handler_base import HandlerBase
from app.coordination.types import CoordinatorHealthState

HEARTBEAT_STALE_THRESHOLD_SECONDS = CoordinatorHealthDefaults.HEARTBEAT_STALE_THRESHOLD
DEGRADED_COOLDOWN_SECONDS = CoordinatorHealthDefaults.DEGRADED_COOLDOWN
INIT_FAILURE_MAX_RETRIES = CoordinatorHealthDefaults.INIT_FAILURE_MAX_RETRIES

# Backward-compat alias
CoordinatorState = CoordinatorHealthState


@dataclass
class CoordinatorInfo:
    """Tracks a single coordinator's health state."""

    name: str
    state: CoordinatorState = CoordinatorState.UNKNOWN
    last_healthy_at: float = 0.0
    last_unhealthy_at: float = 0.0
    last_degraded_at: float = 0.0
    last_heartbeat_at: float = 0.0
    last_shutdown_at: float = 0.0
    init_failure_count: int = 0
    degraded_reason: str = ""
    node_id: str = ""


@dataclass
class CoordinatorHealthSummary:
    """Summary of all coordinator health states."""

    total_count: int = 0
    healthy_count: int = 0
    unhealthy_count: int = 0
    degraded_count: int = 0
    shutdown_count: int = 0
    stale_count: int = 0  # No recent heartbeat
    unknown_count: int = 0

    # Details
    coordinators: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Overall health
    cluster_healthy: bool = True
    cluster_health_pct: float = 100.0


class CoordinatorHealthMonitorDaemon(HandlerBase):
    """Daemon that monitors coordinator health events.

    Subscribes to all COORDINATOR_* events and tracks:
    - Coordinator health states (healthy/unhealthy/degraded)
    - Heartbeat freshness
    - Init failures
    - Shutdown events
    - Cluster-wide health summary
    """

    _event_source = "CoordinatorHealthMonitor"

    def __init__(self) -> None:
        """Initialize the coordinator health monitor."""
        super().__init__(
            name="coordinator_health_monitor",
            cycle_interval=60.0,
        )

        # Thresholds (from CoordinatorHealthDefaults)
        self._heartbeat_stale_threshold = HEARTBEAT_STALE_THRESHOLD_SECONDS
        self._cluster_unhealthy_threshold_pct = 20.0

        # Coordinator tracking
        self._coordinators: dict[str, CoordinatorInfo] = {}

        # Counters
        self._total_healthy_events = 0
        self._total_unhealthy_events = 0
        self._total_degraded_events = 0
        self._total_heartbeats = 0
        self._total_shutdowns = 0
        self._total_init_failures = 0

        # Lock for thread safety
        self._lock = asyncio.Lock()

    def _get_event_subscriptions(self) -> dict[str, Callable]:
        """Return event handlers to subscribe to."""
        try:
            from app.distributed.data_events import DataEventType

            if DataEventType is None:
                logger.warning("[CoordinatorHealthMonitor] DataEventType not available")
                return {}

            return {
                DataEventType.COORDINATOR_HEALTHY.value: self._on_coordinator_healthy,
                DataEventType.COORDINATOR_UNHEALTHY.value: self._on_coordinator_unhealthy,
                DataEventType.COORDINATOR_HEALTH_DEGRADED.value: self._on_coordinator_degraded,
                DataEventType.COORDINATOR_SHUTDOWN.value: self._on_coordinator_shutdown,
                DataEventType.COORDINATOR_INIT_FAILED.value: self._on_coordinator_init_failed,
                DataEventType.COORDINATOR_HEARTBEAT.value: self._on_coordinator_heartbeat,
            }
        except ImportError as e:
            logger.warning(f"[CoordinatorHealthMonitor] data_events not available: {e}")
            return {}

    async def _run_cycle(self) -> None:
        """Execute one monitoring cycle - check for stale heartbeats."""
        await self._check_stale_heartbeats()

    def _get_or_create_coordinator(self, name: str) -> CoordinatorInfo:
        """Get or create coordinator info."""
        if name not in self._coordinators:
            self._coordinators[name] = CoordinatorInfo(name=name)
        return self._coordinators[name]

    async def _on_coordinator_healthy(self, event: Any) -> None:
        """Handle COORDINATOR_HEALTHY event."""
        payload = event.payload if hasattr(event, "payload") else event

        coordinator_name = payload.get("coordinator_name", "") or payload.get("name", "")
        if not coordinator_name:
            return

        async with self._lock:
            info = self._get_or_create_coordinator(coordinator_name)
            info.state = CoordinatorState.HEALTHY
            info.last_healthy_at = time.time()
            info.node_id = payload.get("node_id", "")
            self._total_healthy_events += 1

        logger.info(f"[CoordinatorHealthMonitor] Coordinator healthy: {coordinator_name}")

    async def _on_coordinator_unhealthy(self, event: Any) -> None:
        """Handle COORDINATOR_UNHEALTHY event."""
        payload = event.payload if hasattr(event, "payload") else event

        coordinator_name = payload.get("coordinator_name", "") or payload.get("name", "")
        if not coordinator_name:
            return

        async with self._lock:
            info = self._get_or_create_coordinator(coordinator_name)
            info.state = CoordinatorState.UNHEALTHY
            info.last_unhealthy_at = time.time()
            info.node_id = payload.get("node_id", "")
            self._total_unhealthy_events += 1

        logger.warning(f"[CoordinatorHealthMonitor] Coordinator unhealthy: {coordinator_name}")
        await self._emit_cluster_health_event()

    async def _on_coordinator_degraded(self, event: Any) -> None:
        """Handle COORDINATOR_HEALTH_DEGRADED event."""
        payload = event.payload if hasattr(event, "payload") else event

        coordinator_name = payload.get("coordinator_name", "") or payload.get("name", "")
        if not coordinator_name:
            return

        async with self._lock:
            info = self._get_or_create_coordinator(coordinator_name)
            info.state = CoordinatorState.DEGRADED
            info.last_degraded_at = time.time()
            info.degraded_reason = payload.get("reason", "") or payload.get("message", "")
            info.node_id = payload.get("node_id", "")
            self._total_degraded_events += 1

        logger.warning(
            f"[CoordinatorHealthMonitor] Coordinator degraded: {coordinator_name} - "
            f"{info.degraded_reason}"
        )

    async def _on_coordinator_shutdown(self, event: Any) -> None:
        """Handle COORDINATOR_SHUTDOWN event."""
        payload = event.payload if hasattr(event, "payload") else event

        coordinator_name = payload.get("coordinator_name", "") or payload.get("name", "")
        if not coordinator_name:
            return

        async with self._lock:
            info = self._get_or_create_coordinator(coordinator_name)
            info.state = CoordinatorState.SHUTDOWN
            info.last_shutdown_at = time.time()
            self._total_shutdowns += 1

        logger.info(f"[CoordinatorHealthMonitor] Coordinator shutdown: {coordinator_name}")

    async def _on_coordinator_init_failed(self, event: Any) -> None:
        """Handle COORDINATOR_INIT_FAILED event."""
        payload = event.payload if hasattr(event, "payload") else event

        coordinator_name = payload.get("coordinator_name", "") or payload.get("name", "")
        if not coordinator_name:
            return

        async with self._lock:
            info = self._get_or_create_coordinator(coordinator_name)
            info.init_failure_count += 1
            self._total_init_failures += 1

            # Mark permanently failed after max retries
            if info.init_failure_count >= INIT_FAILURE_MAX_RETRIES:
                info.state = CoordinatorState.INIT_FAILED

        logger.error(
            f"[CoordinatorHealthMonitor] Coordinator init failed: {coordinator_name} "
            f"(attempt {info.init_failure_count})"
        )

    async def _on_coordinator_heartbeat(self, event: Any) -> None:
        """Handle COORDINATOR_HEARTBEAT event."""
        payload = event.payload if hasattr(event, "payload") else event

        coordinator_name = payload.get("coordinator_name", "") or payload.get("name", "")
        if not coordinator_name:
            return

        async with self._lock:
            info = self._get_or_create_coordinator(coordinator_name)
            info.last_heartbeat_at = time.time()
            self._total_heartbeats += 1

            # If coordinator was unknown/stale, upgrade to healthy
            if info.state == CoordinatorState.UNKNOWN:
                info.state = CoordinatorState.HEALTHY
                info.last_healthy_at = time.time()

    async def _emit_cluster_health_event(self) -> None:
        """Emit event when cluster health changes significantly."""
        try:
            from app.coordination.event_router import publish

            summary = self.get_health_summary()

            await publish(
                "CLUSTER_COORDINATOR_HEALTH_CHANGED",
                {
                    "cluster_healthy": summary.cluster_healthy,
                    "cluster_health_pct": summary.cluster_health_pct,
                    "healthy_count": summary.healthy_count,
                    "unhealthy_count": summary.unhealthy_count,
                    "degraded_count": summary.degraded_count,
                    "total_count": summary.total_count,
                    "timestamp": time.time(),
                },
            )
        except Exception as e:
            logger.debug(f"[CoordinatorHealthMonitor] Failed to emit cluster health event: {e}")

    async def _check_stale_heartbeats(self) -> None:
        """Check for coordinators with stale heartbeats."""
        now = time.time()
        stale_coordinators: list[str] = []
        threshold = self._heartbeat_stale_threshold

        async with self._lock:
            for name, info in self._coordinators.items():
                # Skip already unhealthy/shutdown coordinators
                if info.state in (CoordinatorState.SHUTDOWN, CoordinatorState.INIT_FAILED):
                    continue

                # Check heartbeat freshness
                if info.last_heartbeat_at > 0:
                    stale_duration = now - info.last_heartbeat_at
                    if stale_duration > threshold:
                        stale_coordinators.append(name)

        # Log stale coordinators (outside lock)
        for name in stale_coordinators:
            logger.warning(
                f"[CoordinatorHealthMonitor] Stale heartbeat for {name} "
                f"(no heartbeat for {threshold}s)"
            )

    def get_health_summary(self) -> CoordinatorHealthSummary:
        """Get summary of all coordinator health states."""
        now = time.time()
        summary = CoordinatorHealthSummary()
        heartbeat_threshold = self._heartbeat_stale_threshold
        cluster_unhealthy_threshold = self._cluster_unhealthy_threshold_pct

        for name, info in self._coordinators.items():
            summary.total_count += 1

            # Count by state
            if info.state == CoordinatorState.HEALTHY:
                summary.healthy_count += 1
            elif info.state == CoordinatorState.UNHEALTHY:
                summary.unhealthy_count += 1
            elif info.state == CoordinatorState.DEGRADED:
                summary.degraded_count += 1
            elif info.state == CoordinatorState.SHUTDOWN:
                summary.shutdown_count += 1
            elif info.state == CoordinatorState.UNKNOWN:
                summary.unknown_count += 1

            # Check for stale heartbeat
            if info.last_heartbeat_at > 0:
                stale_duration = now - info.last_heartbeat_at
                if stale_duration > heartbeat_threshold:
                    summary.stale_count += 1

            # Add coordinator details
            summary.coordinators[name] = {
                "state": info.state.value,
                "last_healthy_at": info.last_healthy_at,
                "last_heartbeat_at": info.last_heartbeat_at,
                "init_failure_count": info.init_failure_count,
                "degraded_reason": info.degraded_reason,
                "node_id": info.node_id,
            }

        # Calculate overall health
        if summary.total_count > 0:
            # Healthy = healthy + degraded (degraded still functions)
            operational = summary.healthy_count + summary.degraded_count
            summary.cluster_health_pct = (operational / summary.total_count) * 100

            # Cluster is unhealthy if >threshold% coordinators are down
            unhealthy_pct = (summary.unhealthy_count / summary.total_count) * 100
            summary.cluster_healthy = unhealthy_pct < cluster_unhealthy_threshold
        else:
            summary.cluster_healthy = True
            summary.cluster_health_pct = 100.0

        return summary

    def get_status(self) -> dict[str, Any]:
        """Get daemon status with coordinator-specific data."""
        status = super().get_status()
        summary = self.get_health_summary()
        status.update({
            "total_coordinators": summary.total_count,
            "healthy_count": summary.healthy_count,
            "unhealthy_count": summary.unhealthy_count,
            "degraded_count": summary.degraded_count,
            "stale_count": summary.stale_count,
            "cluster_healthy": summary.cluster_healthy,
            "cluster_health_pct": round(summary.cluster_health_pct, 1),
            "total_events": {
                "healthy": self._total_healthy_events,
                "unhealthy": self._total_unhealthy_events,
                "degraded": self._total_degraded_events,
                "heartbeats": self._total_heartbeats,
                "shutdowns": self._total_shutdowns,
                "init_failures": self._total_init_failures,
            },
        })
        return status

    def health_check(self) -> HealthCheckResult:
        """Check daemon health with coordinator-specific checks."""
        base_result = super().health_check()
        if not base_result.healthy:
            return base_result

        if not self._event_subscribed:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message="Coordinator health monitor not subscribed to events",
                details=self.get_status(),
            )

        summary = self.get_health_summary()

        if not summary.cluster_healthy:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"Cluster unhealthy: {summary.unhealthy_count}/{summary.total_count} coordinators down",
                details=self.get_status(),
            )

        if summary.stale_count > 0:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message=f"{summary.stale_count} coordinators have stale heartbeats",
                details=self.get_status(),
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"Coordinator monitor running ({summary.healthy_count} healthy)",
            details=self.get_status(),
        )


# =============================================================================
# Singleton Accessor Functions
# =============================================================================


def get_coordinator_health_monitor() -> CoordinatorHealthMonitorDaemon:
    """Get or create the singleton CoordinatorHealthMonitorDaemon instance."""
    return CoordinatorHealthMonitorDaemon.get_instance()


def get_coordinator_health_monitor_sync() -> CoordinatorHealthMonitorDaemon:
    """Get the singleton instance synchronously."""
    return CoordinatorHealthMonitorDaemon.get_instance()


def reset_coordinator_health_monitor() -> None:
    """Reset the singleton instance (for testing)."""
    CoordinatorHealthMonitorDaemon.reset_instance()


__all__ = [
    "CoordinatorHealthMonitorDaemon",
    "CoordinatorHealthSummary",
    "CoordinatorInfo",
    "CoordinatorState",
    "get_coordinator_health_monitor",
    "get_coordinator_health_monitor_sync",
    "reset_coordinator_health_monitor",
    "HEARTBEAT_STALE_THRESHOLD_SECONDS",
    "INIT_FAILURE_MAX_RETRIES",
]

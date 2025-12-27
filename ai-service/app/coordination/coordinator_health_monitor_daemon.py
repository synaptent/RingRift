"""Coordinator Health Monitor Daemon - Tracks coordinator lifecycle events.

December 2025: Created to address critical gap in event coordination.
Previously, COORDINATOR_HEALTHY and COORDINATOR_UNHEALTHY events were emitted
but had NO subscribers - making coordinator health visibility impossible.

This daemon subscribes to all COORDINATOR_* events and provides:
1. Coordinator health tracking (healthy/unhealthy/degraded)
2. Heartbeat freshness monitoring
3. Init failure tracking
4. Shutdown detection
5. Cluster-wide coordinator health summary

Event Subscriptions:
- COORDINATOR_HEALTHY: Track coordinator initialization success
- COORDINATOR_UNHEALTHY: Track coordinator initialization failure
- COORDINATOR_HEALTH_DEGRADED: Track degraded coordinator state
- COORDINATOR_SHUTDOWN: Track graceful shutdown
- COORDINATOR_INIT_FAILED: Track persistent initialization failures
- COORDINATOR_HEARTBEAT: Track liveness signals

Usage:
    from app.coordination.coordinator_health_monitor_daemon import (
        CoordinatorHealthMonitorDaemon,
        get_coordinator_health_monitor,
    )

    # Start monitoring
    monitor = get_coordinator_health_monitor()
    await monitor.start()

    # Get coordinator health summary
    summary = monitor.get_health_summary()
    print(f"Healthy: {summary['healthy_count']}/{summary['total_count']}")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Thresholds (December 27, 2025: Centralized in coordination_defaults.py)
from app.config.coordination_defaults import CoordinatorHealthDefaults

HEARTBEAT_STALE_THRESHOLD_SECONDS = CoordinatorHealthDefaults.HEARTBEAT_STALE_THRESHOLD
DEGRADED_COOLDOWN_SECONDS = CoordinatorHealthDefaults.DEGRADED_COOLDOWN
INIT_FAILURE_MAX_RETRIES = CoordinatorHealthDefaults.INIT_FAILURE_MAX_RETRIES


# December 2025: Import CoordinatorHealthState from canonical source
from app.coordination.types import CoordinatorHealthState

# CoordinatorHealthState is now imported from app.coordination.types
# Canonical values: UNKNOWN, HEALTHY, UNHEALTHY, DEGRADED, SHUTDOWN, INIT_FAILED
# Backward-compat alias:
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


class CoordinatorHealthMonitorDaemon:
    """Daemon that monitors coordinator health events.

    Subscribes to all COORDINATOR_* events and tracks:
    - Coordinator health states (healthy/unhealthy/degraded)
    - Heartbeat freshness
    - Init failures
    - Shutdown events
    - Cluster-wide health summary
    """

    def __init__(self):
        """Initialize the coordinator health monitor."""
        self._running = False
        self._subscribed = False

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

    async def start(self) -> bool:
        """Start the monitor daemon."""
        if self._running:
            return True

        self._running = True
        success = await self._subscribe_to_events()

        if success:
            logger.info("[CoordinatorHealthMonitor] Started - monitoring COORDINATOR_* events")

            # Start background monitoring loop
            asyncio.create_task(self._monitoring_loop())
        else:
            logger.warning("[CoordinatorHealthMonitor] Started without event subscriptions")

        return success

    async def stop(self) -> None:
        """Stop the monitor daemon."""
        self._running = False
        await self._unsubscribe_from_events()
        logger.info("[CoordinatorHealthMonitor] Stopped")

    async def _subscribe_to_events(self) -> bool:
        """Subscribe to all COORDINATOR_* events."""
        if self._subscribed:
            return True

        try:
            from app.distributed.data_events import DataEventType
            from app.coordination.event_router import get_router

            # December 27, 2025: Guard against DataEventType being None
            if DataEventType is None:
                logger.warning("[CoordinatorHealthMonitor] DataEventType not available")
                return False

            router = get_router()

            # Subscribe to all coordinator events
            router.subscribe(DataEventType.COORDINATOR_HEALTHY.value, self._on_coordinator_healthy)
            router.subscribe(DataEventType.COORDINATOR_UNHEALTHY.value, self._on_coordinator_unhealthy)
            router.subscribe(DataEventType.COORDINATOR_HEALTH_DEGRADED.value, self._on_coordinator_degraded)
            router.subscribe(DataEventType.COORDINATOR_SHUTDOWN.value, self._on_coordinator_shutdown)
            router.subscribe(DataEventType.COORDINATOR_INIT_FAILED.value, self._on_coordinator_init_failed)
            router.subscribe(DataEventType.COORDINATOR_HEARTBEAT.value, self._on_coordinator_heartbeat)

            self._subscribed = True
            logger.info("[CoordinatorHealthMonitor] Subscribed to 6 COORDINATOR_* events")
            return True

        except ImportError as e:
            logger.warning(f"[CoordinatorHealthMonitor] data_events not available: {e}")
            return False
        except Exception as e:
            logger.error(f"[CoordinatorHealthMonitor] Failed to subscribe: {e}")
            return False

    async def _unsubscribe_from_events(self) -> None:
        """Unsubscribe from all events."""
        if not self._subscribed:
            return

        try:
            from app.distributed.data_events import DataEventType
            from app.coordination.event_router import get_router

            router = get_router()

            router.unsubscribe(DataEventType.COORDINATOR_HEALTHY.value, self._on_coordinator_healthy)
            router.unsubscribe(DataEventType.COORDINATOR_UNHEALTHY.value, self._on_coordinator_unhealthy)
            router.unsubscribe(DataEventType.COORDINATOR_HEALTH_DEGRADED.value, self._on_coordinator_degraded)
            router.unsubscribe(DataEventType.COORDINATOR_SHUTDOWN.value, self._on_coordinator_shutdown)
            router.unsubscribe(DataEventType.COORDINATOR_INIT_FAILED.value, self._on_coordinator_init_failed)
            router.unsubscribe(DataEventType.COORDINATOR_HEARTBEAT.value, self._on_coordinator_heartbeat)

            self._subscribed = False

        except Exception as e:
            logger.warning(f"[CoordinatorHealthMonitor] Error unsubscribing: {e}")

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
            from app.coordination.event_router import get_router

            summary = self.get_health_summary()

            router = get_router()
            await router.publish(
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

    async def _monitoring_loop(self) -> None:
        """Background loop for periodic health checks."""
        while self._running:
            try:
                await self._check_stale_heartbeats()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[CoordinatorHealthMonitor] Monitoring loop error: {e}")
                await asyncio.sleep(60)

    async def _check_stale_heartbeats(self) -> None:
        """Check for coordinators with stale heartbeats."""
        now = time.time()
        stale_coordinators: list[str] = []

        async with self._lock:
            for name, info in self._coordinators.items():
                # Skip already unhealthy/shutdown coordinators
                if info.state in (CoordinatorState.SHUTDOWN, CoordinatorState.INIT_FAILED):
                    continue

                # Check heartbeat freshness
                if info.last_heartbeat_at > 0:
                    stale_duration = now - info.last_heartbeat_at
                    if stale_duration > HEARTBEAT_STALE_THRESHOLD_SECONDS:
                        stale_coordinators.append(name)

        # Log stale coordinators (outside lock)
        for name in stale_coordinators:
            logger.warning(
                f"[CoordinatorHealthMonitor] Stale heartbeat for {name} "
                f"(no heartbeat for {HEARTBEAT_STALE_THRESHOLD_SECONDS}s)"
            )

    def get_health_summary(self) -> CoordinatorHealthSummary:
        """Get summary of all coordinator health states."""
        now = time.time()
        summary = CoordinatorHealthSummary()

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
                if stale_duration > HEARTBEAT_STALE_THRESHOLD_SECONDS:
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

            # Cluster is unhealthy if >20% coordinators are down
            unhealthy_pct = (summary.unhealthy_count / summary.total_count) * 100
            summary.cluster_healthy = unhealthy_pct < 20.0
        else:
            summary.cluster_healthy = True
            summary.cluster_health_pct = 100.0

        return summary

    def get_status(self) -> dict[str, Any]:
        """Get daemon status for health checks."""
        summary = self.get_health_summary()
        return {
            "running": self._running,
            "subscribed": self._subscribed,
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
        }

    def health_check(self) -> "HealthCheckResult":
        """Check daemon health status."""
        try:
            from app.coordination.protocols import HealthCheckResult, CoordinatorStatus

            if not self._running:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.STOPPED,
                    message="Coordinator health monitor not running",
                )

            if not self._subscribed:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.DEGRADED,
                    message="Coordinator health monitor not subscribed to events",
                )

            summary = self.get_health_summary()

            # Check for cluster-level issues
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

        except ImportError:
            return {"healthy": self._running and self._subscribed}


# Singleton instance
_monitor_instance: CoordinatorHealthMonitorDaemon | None = None
_monitor_lock = asyncio.Lock()


async def get_coordinator_health_monitor() -> CoordinatorHealthMonitorDaemon:
    """Get or create the singleton CoordinatorHealthMonitorDaemon instance."""
    global _monitor_instance

    async with _monitor_lock:
        if _monitor_instance is None:
            _monitor_instance = CoordinatorHealthMonitorDaemon()
        return _monitor_instance


def get_coordinator_health_monitor_sync() -> CoordinatorHealthMonitorDaemon:
    """Get the singleton instance synchronously (may create if not exists)."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = CoordinatorHealthMonitorDaemon()
    return _monitor_instance


__all__ = [
    "CoordinatorHealthMonitorDaemon",
    "CoordinatorHealthSummary",
    "CoordinatorInfo",
    "CoordinatorState",
    "get_coordinator_health_monitor",
    "get_coordinator_health_monitor_sync",
    "HEARTBEAT_STALE_THRESHOLD_SECONDS",
    "INIT_FAILURE_MAX_RETRIES",
]

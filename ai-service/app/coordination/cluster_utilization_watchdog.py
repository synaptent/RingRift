"""Cluster Utilization Watchdog Daemon (December 30, 2025).

Monitors cluster GPU utilization and emits alerts when utilization drops
below acceptable thresholds, enabling proactive remediation.

Problem: After P2P mesh recovery, work queue may be empty and GPUs idle
for extended periods (5-10 minutes). This daemon detects underutilization
and triggers remediation actions.

Key Features:
- Polls node utilization every 30s (configurable)
- Emits CLUSTER_UNDERUTILIZED when threshold exceeded
- Tracks duration of underutilization
- Distinguishes warning vs critical levels
- Integrates with IdleResourceDaemon for remediation

Usage:
    from app.coordination.cluster_utilization_watchdog import (
        ClusterUtilizationWatchdog,
        UtilizationWatchdogConfig,
        get_utilization_watchdog,
    )

    # Get singleton instance
    watchdog = get_utilization_watchdog()
    await watchdog.start()

Events Emitted:
    CLUSTER_UNDERUTILIZED: When utilization drops below warning threshold
    CLUSTER_UTILIZATION_RECOVERED: When utilization returns to healthy levels
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.config.ports import get_local_p2p_status_url
from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.contracts import CoordinatorStatus
from app.coordination.event_emission_helpers import safe_emit_event

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class UtilizationLevel(Enum):
    """Cluster utilization health levels."""

    HEALTHY = "healthy"  # >= 40% utilization
    WARNING = "warning"  # 20-40% utilization
    CRITICAL = "critical"  # < 20% utilization


@dataclass
class UtilizationWatchdogConfig:
    """Configuration for cluster utilization watchdog.

    Attributes:
        enabled: Whether the watchdog is enabled
        check_interval_seconds: How often to poll utilization (default: 30s)
        warning_threshold: Fraction of nodes idle before warning (default: 0.6)
        critical_threshold: Fraction of nodes idle before critical (default: 0.8)
        idle_util_threshold: GPU util % below which node is considered idle (default: 10)
        idle_duration_trigger_seconds: Seconds of underutilization before alert (default: 180)
        recovery_threshold: Fraction of nodes active to clear alert (default: 0.5)
        emit_events: Whether to emit events (default: True)
        p2p_status_endpoint: P2P endpoint to query node status
    """

    enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_UTILIZATION_WATCHDOG_ENABLED", "true"
        ).lower()
        in {"1", "true", "yes", "on"}
    )

    check_interval_seconds: int = field(
        default_factory=lambda: int(
            os.environ.get("RINGRIFT_UTILIZATION_CHECK_INTERVAL", "30")
        )
    )

    # Threshold: fraction of nodes that are idle before warning/critical
    # 0.6 = 60% of GPU nodes idle triggers warning
    warning_threshold: float = 0.6
    critical_threshold: float = 0.8

    # GPU utilization percentage below which a node is considered "idle"
    idle_util_threshold: float = 10.0

    # How long underutilization must persist before triggering action
    idle_duration_trigger_seconds: int = 180  # 3 minutes

    # Fraction of nodes that must be active to clear underutilization alert
    recovery_threshold: float = 0.5

    # Event emission
    emit_events: bool = True

    # P2P status endpoint
    p2p_status_endpoint: str = field(default_factory=get_local_p2p_status_url)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")
        if not 0 < self.warning_threshold < 1:
            raise ValueError("warning_threshold must be between 0 and 1")
        if not 0 < self.critical_threshold <= 1:
            raise ValueError("critical_threshold must be between 0 and 1")
        if self.warning_threshold >= self.critical_threshold:
            raise ValueError("warning_threshold must be < critical_threshold")


# =============================================================================
# Watchdog Daemon
# =============================================================================


class ClusterUtilizationWatchdog(HandlerBase):
    """Monitors cluster GPU utilization and emits alerts.

    Key behaviors:
    - Queries P2P /status endpoint for node list
    - Checks GPU utilization per node
    - Tracks duration of underutilization
    - Emits CLUSTER_UNDERUTILIZED when thresholds exceeded
    - Emits CLUSTER_UTILIZATION_RECOVERED when healthy again
    """

    # Singleton instance
    _instance: ClusterUtilizationWatchdog | None = None

    def __init__(self, config: UtilizationWatchdogConfig | None = None):
        """Initialize the watchdog.

        Args:
            config: Optional configuration, uses defaults if not provided
        """
        self.config = config or UtilizationWatchdogConfig()

        super().__init__(
            name="cluster_utilization_watchdog",
            cycle_interval=float(self.config.check_interval_seconds),
        )

        # Tracking state
        self._current_level = UtilizationLevel.HEALTHY
        self._underutilization_start: float | None = None
        self._last_check_time: float = 0.0
        self._total_gpu_nodes: int = 0
        self._idle_gpu_nodes: int = 0
        self._active_gpu_nodes: int = 0

        # Statistics
        self._stats_checks = 0
        self._stats_warnings = 0
        self._stats_criticals = 0
        self._stats_recoveries = 0

    @classmethod
    def get_instance(cls) -> ClusterUtilizationWatchdog:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton for testing."""
        cls._instance = None

    async def _run_cycle(self) -> None:
        """Execute one monitoring cycle."""
        if not self.config.enabled:
            return

        try:
            # Get cluster utilization metrics
            metrics = await self._get_cluster_metrics()
            if metrics is None:
                logger.debug("[UtilizationWatchdog] Could not get cluster metrics")
                return

            self._stats_checks += 1
            self._last_check_time = time.time()

            # Update tracking
            self._total_gpu_nodes = metrics["total_gpu_nodes"]
            self._idle_gpu_nodes = metrics["idle_gpu_nodes"]
            self._active_gpu_nodes = metrics["active_gpu_nodes"]

            # Determine utilization level
            if self._total_gpu_nodes == 0:
                new_level = UtilizationLevel.HEALTHY
            else:
                idle_fraction = self._idle_gpu_nodes / self._total_gpu_nodes

                if idle_fraction >= self.config.critical_threshold:
                    new_level = UtilizationLevel.CRITICAL
                elif idle_fraction >= self.config.warning_threshold:
                    new_level = UtilizationLevel.WARNING
                else:
                    new_level = UtilizationLevel.HEALTHY

            # Handle level transitions
            await self._handle_level_transition(new_level)

        except Exception as e:
            logger.debug(f"[UtilizationWatchdog] Cycle error: {e}")
            self._record_error(f"Cycle failed: {e}", e)

    async def _get_cluster_metrics(self) -> dict[str, Any] | None:
        """Get cluster utilization metrics from P2P status.

        Returns:
            Dict with total_gpu_nodes, idle_gpu_nodes, active_gpu_nodes
            or None if metrics unavailable.
        """
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.config.p2p_status_endpoint,
                    timeout=aiohttp.ClientTimeout(total=10.0),
                ) as resp:
                    if resp.status != 200:
                        return None

                    status = await resp.json()

            # Extract peer list
            peers = status.get("peers", {})
            alive_peers = [p for p in peers.values() if p.get("state") == "alive"]

            # Filter to GPU nodes (have gpu_type or gpu_memory)
            gpu_peers = [
                p for p in alive_peers
                if p.get("gpu_type") or p.get("gpu_memory_gb", 0) > 0
            ]

            if not gpu_peers:
                # Try alternative: any node with selfplay capability
                gpu_peers = [
                    p for p in alive_peers
                    if p.get("capabilities", {}).get("selfplay", False)
                ]

            # Count idle vs active based on current workload
            idle_count = 0
            active_count = 0

            for peer in gpu_peers:
                # Check if peer has active jobs
                active_jobs = peer.get("active_jobs", 0)
                gpu_util = peer.get("gpu_utilization", 0.0)

                if active_jobs == 0 and gpu_util < self.config.idle_util_threshold:
                    idle_count += 1
                else:
                    active_count += 1

            return {
                "total_gpu_nodes": len(gpu_peers),
                "idle_gpu_nodes": idle_count,
                "active_gpu_nodes": active_count,
                "idle_fraction": idle_count / len(gpu_peers) if gpu_peers else 0.0,
            }

        except ImportError:
            logger.debug("[UtilizationWatchdog] aiohttp not available")
            return None
        except Exception as e:
            logger.debug(f"[UtilizationWatchdog] Failed to get metrics: {e}")
            return None

    async def _handle_level_transition(self, new_level: UtilizationLevel) -> None:
        """Handle transition between utilization levels."""
        now = time.time()
        old_level = self._current_level

        if new_level == UtilizationLevel.HEALTHY:
            # Recovered from underutilization
            if old_level != UtilizationLevel.HEALTHY:
                await self._emit_recovery_event()
                self._stats_recoveries += 1

            self._underutilization_start = None
            self._current_level = new_level

        else:
            # Underutilization detected
            if self._underutilization_start is None:
                self._underutilization_start = now

            duration = now - self._underutilization_start

            # Only emit events after duration trigger
            if duration >= self.config.idle_duration_trigger_seconds:
                if new_level == UtilizationLevel.CRITICAL:
                    if old_level != UtilizationLevel.CRITICAL:
                        await self._emit_underutilization_event(new_level, duration)
                        self._stats_criticals += 1
                elif new_level == UtilizationLevel.WARNING:
                    if old_level == UtilizationLevel.HEALTHY:
                        await self._emit_underutilization_event(new_level, duration)
                        self._stats_warnings += 1

            self._current_level = new_level

    async def _emit_underutilization_event(
        self,
        level: UtilizationLevel,
        duration: float,
    ) -> None:
        """Emit CLUSTER_UNDERUTILIZED event."""
        if not self.config.emit_events:
            return

        logger.warning(
            f"[UtilizationWatchdog] Cluster underutilized: level={level.value}, "
            f"idle={self._idle_gpu_nodes}/{self._total_gpu_nodes}, duration={duration:.0f}s"
        )

        safe_emit_event(
            "cluster_underutilized",
            {
                "level": level.value,
                "total_gpu_nodes": self._total_gpu_nodes,
                "idle_gpu_nodes": self._idle_gpu_nodes,
                "active_gpu_nodes": self._active_gpu_nodes,
                "idle_fraction": self._idle_gpu_nodes / max(1, self._total_gpu_nodes),
                "duration_seconds": duration,
                "source": "ClusterUtilizationWatchdog",
            },
            context="UtilizationWatchdog",
        )

    async def _emit_recovery_event(self) -> None:
        """Emit CLUSTER_UTILIZATION_RECOVERED event."""
        if not self.config.emit_events:
            return

        duration = 0.0
        if self._underutilization_start:
            duration = time.time() - self._underutilization_start

        logger.info(
            f"[UtilizationWatchdog] Cluster utilization recovered after {duration:.0f}s"
        )

        safe_emit_event(
            "cluster_utilization_recovered",
            {
                "total_gpu_nodes": self._total_gpu_nodes,
                "active_gpu_nodes": self._active_gpu_nodes,
                "recovery_duration_seconds": duration,
                "source": "ClusterUtilizationWatchdog",
            },
            context="UtilizationWatchdog",
        )

    def health_check(self) -> HealthCheckResult:
        """Return health check result."""
        is_healthy = self._current_level == UtilizationLevel.HEALTHY

        return HealthCheckResult(
            healthy=is_healthy,
            status=CoordinatorStatus.RUNNING if is_healthy else CoordinatorStatus.DEGRADED,
            details={
                "enabled": self.config.enabled,
                "current_level": self._current_level.value,
                "total_gpu_nodes": self._total_gpu_nodes,
                "idle_gpu_nodes": self._idle_gpu_nodes,
                "active_gpu_nodes": self._active_gpu_nodes,
                "last_check_time": self._last_check_time,
                "stats_checks": self._stats_checks,
                "stats_warnings": self._stats_warnings,
                "stats_criticals": self._stats_criticals,
                "stats_recoveries": self._stats_recoveries,
            },
        )

    def get_current_level(self) -> UtilizationLevel:
        """Get current utilization level."""
        return self._current_level

    def get_stats(self) -> dict[str, Any]:
        """Get watchdog statistics."""
        return {
            "checks": self._stats_checks,
            "warnings": self._stats_warnings,
            "criticals": self._stats_criticals,
            "recoveries": self._stats_recoveries,
            "current_level": self._current_level.value,
            "total_gpu_nodes": self._total_gpu_nodes,
            "idle_gpu_nodes": self._idle_gpu_nodes,
        }


# =============================================================================
# Module-level accessors
# =============================================================================


def get_utilization_watchdog() -> ClusterUtilizationWatchdog:
    """Get the singleton utilization watchdog instance."""
    return ClusterUtilizationWatchdog.get_instance()


def reset_utilization_watchdog() -> None:
    """Reset the singleton for testing."""
    ClusterUtilizationWatchdog.reset_instance()

"""Replication Monitor Daemon - Continuous monitoring for data replication health.

.. deprecated:: December 2025
    This module is deprecated. Use unified_replication_daemon.py instead:

    from app.coordination.unified_replication_daemon import (
        UnifiedReplicationDaemon,
        UnifiedReplicationConfig,
        get_replication_daemon,
    )

MIGRATION GUIDE:
    Old (deprecated):
        from app.coordination.replication_monitor import get_replication_monitor
        daemon = get_replication_monitor()

    New (recommended):
        from app.coordination.unified_replication_daemon import get_replication_daemon
        daemon = await get_replication_daemon()

This daemon monitors the ClusterManifest for under-replicated data and triggers
emergency sync operations when data safety is at risk.

Features:
1. Periodic check of replication status (every 5 minutes by default)
2. Alert if any game has <2 replicas for >15 minutes
3. Track replication success rate per node
4. Trigger emergency sync for critical under-replication
5. Emit events for external monitoring systems

Usage (DEPRECATED - see migration guide above):
    from app.coordination.replication_monitor import (
        ReplicationMonitorDaemon,
        ReplicationMonitorConfig,
        get_replication_monitor,
    )

    # Get singleton daemon
    daemon = get_replication_monitor()

    # Start monitoring
    await daemon.start()

    # Get current status
    status = daemon.get_status()

    # Stop daemon
    await daemon.stop()
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Emit deprecation warning on import
warnings.warn(
    "replication_monitor is deprecated. Use unified_replication_daemon instead:\n"
    "  from app.coordination.unified_replication_daemon import get_replication_daemon",
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ReplicationMonitorDaemon",
    "ReplicationMonitorConfig",
    "ReplicationAlert",
    "ReplicationAlertLevel",
    "ReplicationStats",
    "get_replication_monitor",
    "reset_replication_monitor",
]


class ReplicationAlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ReplicationAlert:
    """An alert about replication issues."""
    level: ReplicationAlertLevel
    message: str
    game_count: int = 0
    affected_nodes: list[str] = field(default_factory=list)
    timestamp: float = 0.0
    resolved: bool = False
    resolved_at: float = 0.0


@dataclass
class ReplicationStats:
    """Statistics about replication health."""
    total_games: int = 0
    under_replicated_games: int = 0
    single_copy_games: int = 0
    zero_copy_games: int = 0
    avg_replication_factor: float = 0.0
    nodes_with_data: int = 0
    last_check_time: float = 0.0
    check_duration_seconds: float = 0.0


@dataclass
class ReplicationMonitorConfig:
    """Configuration for the replication monitor."""
    # Check interval
    check_interval_seconds: float = 300.0  # 5 minutes

    # Alert thresholds
    warning_threshold_minutes: float = 15.0  # Warn if under-replicated for 15 min
    critical_threshold_minutes: float = 60.0  # Critical if under-replicated for 1 hour
    single_copy_threshold_games: int = 100  # Alert if this many games have single copy

    # Replication targets
    min_replicas: int = 2
    target_replicas: int = 3

    # Emergency sync
    enable_emergency_sync: bool = True
    emergency_sync_threshold_games: int = 500  # Trigger emergency sync if this many under-replicated

    # Alerting
    emit_events: bool = True
    max_alerts_history: int = 100


class ReplicationMonitorDaemon:
    """Daemon that monitors data replication health across the cluster.

    Periodically checks the ClusterManifest for:
    - Under-replicated games (< min_replicas copies)
    - Single-copy games (highest risk)
    - Zero-copy games (data loss!)
    - Node availability

    Triggers alerts and emergency sync when thresholds are exceeded.
    """

    def __init__(self, config: ReplicationMonitorConfig | None = None):
        """Initialize the replication monitor.

        Args:
            config: Monitor configuration
        """
        self.config = config or ReplicationMonitorConfig()
        self.node_id = socket.gethostname()

        self._running = False
        self._monitor_task: asyncio.Task | None = None
        self._stats = ReplicationStats()
        self._alerts: list[ReplicationAlert] = []
        self._active_alerts: dict[str, ReplicationAlert] = {}  # key -> alert

        # Track under-replication duration
        self._under_replicated_since: dict[str, float] = {}  # game_id -> first_seen_time

        # Emergency sync state
        self._last_emergency_sync: float = 0.0
        self._emergency_sync_cooldown = 600.0  # 10 minutes

    async def start(self) -> None:
        """Start the replication monitor daemon."""
        if self._running:
            logger.warning("ReplicationMonitor already running")
            return

        self._running = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(),
            name="replication_monitor"
        )
        logger.info(
            f"ReplicationMonitor started (interval={self.config.check_interval_seconds}s, "
            f"min_replicas={self.config.min_replicas})"
        )

    async def stop(self) -> None:
        """Stop the replication monitor daemon."""
        if not self._running:
            return

        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await asyncio.wait_for(self._monitor_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._monitor_task = None

        logger.info("ReplicationMonitor stopped")

    def is_running(self) -> bool:
        """Check if daemon is running."""
        return self._running

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_replication()
                await asyncio.sleep(self.config.check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in replication check: {e}")
                await asyncio.sleep(60)  # Back off on error

    async def _check_replication(self) -> None:
        """Perform replication health check."""
        start_time = time.time()

        try:
            from app.distributed.cluster_manifest import get_cluster_manifest
        except ImportError as e:
            logger.error(f"ClusterManifest not available: {e}")
            return

        manifest = get_cluster_manifest()
        now = time.time()

        # Get under-replicated games
        min_replicas = self.config.min_replicas
        under_replicated = manifest.get_under_replicated_games(
            min_copies=min_replicas,
            limit=10000,
        )

        # Categorize by replica count
        single_copy_games = []
        zero_copy_games = []

        for game_id, copies in under_replicated:
            if copies == 0:
                zero_copy_games.append(game_id)
            elif copies == 1:
                single_copy_games.append(game_id)

            # Track how long this game has been under-replicated
            if game_id not in self._under_replicated_since:
                self._under_replicated_since[game_id] = now

        # Clean up games that are now properly replicated
        currently_under = {g[0] for g in under_replicated}
        for game_id in list(self._under_replicated_since.keys()):
            if game_id not in currently_under:
                del self._under_replicated_since[game_id]

        # Get cluster stats
        cluster_stats = manifest.get_cluster_stats()

        # Calculate average replication factor
        total_games = cluster_stats.get("total_games", 0)
        avg_replication = 0.0
        if total_games > 0:
            # Approximate from under-replicated count
            properly_replicated = total_games - len(under_replicated)
            avg_replication = (
                (properly_replicated * min_replicas) +
                sum(copies for _, copies in under_replicated)
            ) / total_games

        # Update stats
        self._stats = ReplicationStats(
            total_games=total_games,
            under_replicated_games=len(under_replicated),
            single_copy_games=len(single_copy_games),
            zero_copy_games=len(zero_copy_games),
            avg_replication_factor=avg_replication,
            nodes_with_data=len(cluster_stats.get("games_by_node", {})),
            last_check_time=now,
            check_duration_seconds=time.time() - start_time,
        )

        # Generate alerts
        await self._evaluate_alerts(now, single_copy_games, zero_copy_games, under_replicated)

        # Check for emergency sync trigger
        if self.config.enable_emergency_sync:
            await self._check_emergency_sync(under_replicated)

        logger.debug(
            f"Replication check: {len(under_replicated)} under-replicated, "
            f"{len(single_copy_games)} single-copy, {len(zero_copy_games)} zero-copy"
        )

    async def _evaluate_alerts(
        self,
        now: float,
        single_copy_games: list[str],
        zero_copy_games: list[str],
        under_replicated: list[tuple[str, int]],
    ) -> None:
        """Evaluate and generate alerts based on current state."""
        new_alerts = []

        # CRITICAL: Zero-copy games (data loss!)
        if zero_copy_games:
            alert = ReplicationAlert(
                level=ReplicationAlertLevel.CRITICAL,
                message=f"DATA LOSS: {len(zero_copy_games)} games have ZERO replicas!",
                game_count=len(zero_copy_games),
                timestamp=now,
            )
            new_alerts.append(alert)
            self._add_alert("zero_copy", alert)

        # CRITICAL: Too many single-copy games
        if len(single_copy_games) >= self.config.single_copy_threshold_games:
            alert = ReplicationAlert(
                level=ReplicationAlertLevel.CRITICAL,
                message=f"HIGH RISK: {len(single_copy_games)} games have only 1 replica",
                game_count=len(single_copy_games),
                timestamp=now,
            )
            new_alerts.append(alert)
            self._add_alert("single_copy_high", alert)

        # WARNING: Long-standing under-replication
        warning_threshold_seconds = self.config.warning_threshold_minutes * 60
        critical_threshold_seconds = self.config.critical_threshold_minutes * 60

        long_under_replicated = []
        critical_under_replicated = []

        for game_id, copies in under_replicated:
            first_seen = self._under_replicated_since.get(game_id, now)
            duration = now - first_seen

            if duration >= critical_threshold_seconds:
                critical_under_replicated.append(game_id)
            elif duration >= warning_threshold_seconds:
                long_under_replicated.append(game_id)

        if critical_under_replicated:
            alert = ReplicationAlert(
                level=ReplicationAlertLevel.CRITICAL,
                message=(
                    f"{len(critical_under_replicated)} games under-replicated for "
                    f">{self.config.critical_threshold_minutes} minutes"
                ),
                game_count=len(critical_under_replicated),
                timestamp=now,
            )
            new_alerts.append(alert)
            self._add_alert("critical_duration", alert)

        if long_under_replicated:
            alert = ReplicationAlert(
                level=ReplicationAlertLevel.WARNING,
                message=(
                    f"{len(long_under_replicated)} games under-replicated for "
                    f">{self.config.warning_threshold_minutes} minutes"
                ),
                game_count=len(long_under_replicated),
                timestamp=now,
            )
            new_alerts.append(alert)
            self._add_alert("warning_duration", alert)

        # Resolve alerts that no longer apply
        if not zero_copy_games and "zero_copy" in self._active_alerts:
            self._resolve_alert("zero_copy", now)

        if len(single_copy_games) < self.config.single_copy_threshold_games:
            if "single_copy_high" in self._active_alerts:
                self._resolve_alert("single_copy_high", now)

        if not critical_under_replicated and "critical_duration" in self._active_alerts:
            self._resolve_alert("critical_duration", now)

        if not long_under_replicated and "warning_duration" in self._active_alerts:
            self._resolve_alert("warning_duration", now)

        # Emit events for new alerts
        if self.config.emit_events and new_alerts:
            await self._emit_alert_events(new_alerts)

    def _add_alert(self, key: str, alert: ReplicationAlert) -> None:
        """Add or update an active alert."""
        if key not in self._active_alerts:
            self._active_alerts[key] = alert
            self._alerts.append(alert)

            # Trim alert history
            if len(self._alerts) > self.config.max_alerts_history:
                self._alerts = self._alerts[-self.config.max_alerts_history:]

            logger.warning(f"REPLICATION ALERT [{alert.level.value.upper()}]: {alert.message}")
        else:
            # Update existing alert
            existing = self._active_alerts[key]
            existing.game_count = alert.game_count
            existing.message = alert.message

    def _resolve_alert(self, key: str, now: float) -> None:
        """Mark an alert as resolved."""
        if key in self._active_alerts:
            alert = self._active_alerts[key]
            alert.resolved = True
            alert.resolved_at = now
            del self._active_alerts[key]
            logger.info(f"REPLICATION ALERT RESOLVED: {alert.message}")

    async def _emit_alert_events(self, alerts: list[ReplicationAlert]) -> None:
        """Emit events for alerts."""
        try:
            from app.coordination.event_router import (
                get_router,
                DataEventType,
            )

            router = get_router()

            for alert in alerts:
                # Phase 22.2 fix: Use publish instead of emit (which doesn't exist)
                await router.publish(
                    event_type=DataEventType.HEALTH_ALERT,
                    payload={
                        "alert": "replication_health",
                        "source": "replication_monitor",
                        "level": alert.level.value,
                        "message": alert.message,
                        "game_count": alert.game_count,
                        "timestamp": alert.timestamp,
                    },
                    source="replication_monitor",
                )
        except ImportError:
            pass  # Event router not available
        except Exception as e:
            logger.debug(f"Failed to publish alert events: {e}")

    async def _check_emergency_sync(
        self,
        under_replicated: list[tuple[str, int]],
    ) -> None:
        """Check if emergency sync should be triggered."""
        now = time.time()

        # Check cooldown
        if now - self._last_emergency_sync < self._emergency_sync_cooldown:
            return

        # Check threshold
        if len(under_replicated) < self.config.emergency_sync_threshold_games:
            return

        logger.warning(
            f"EMERGENCY SYNC: {len(under_replicated)} games under-replicated, "
            f"triggering priority sync"
        )

        try:
            from app.coordination.auto_sync_daemon import get_auto_sync_daemon

            daemon = get_auto_sync_daemon()
            if daemon.is_running():
                await daemon.trigger_sync()
                self._last_emergency_sync = now
                logger.info("Emergency sync triggered successfully")
            else:
                logger.warning("AutoSyncDaemon not running, cannot trigger emergency sync")

        except ImportError:
            logger.warning("AutoSyncDaemon not available for emergency sync")
        except Exception as e:
            logger.error(f"Failed to trigger emergency sync: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get current replication status.

        Returns:
            Status dict with stats, alerts, and health metrics
        """
        return {
            "node_id": self.node_id,
            "running": self._running,
            "stats": {
                "total_games": self._stats.total_games,
                "under_replicated_games": self._stats.under_replicated_games,
                "single_copy_games": self._stats.single_copy_games,
                "zero_copy_games": self._stats.zero_copy_games,
                "avg_replication_factor": round(self._stats.avg_replication_factor, 2),
                "nodes_with_data": self._stats.nodes_with_data,
                "last_check_time": self._stats.last_check_time,
                "check_duration_seconds": round(self._stats.check_duration_seconds, 3),
            },
            "active_alerts": [
                {
                    "level": a.level.value,
                    "message": a.message,
                    "game_count": a.game_count,
                    "timestamp": a.timestamp,
                }
                for a in self._active_alerts.values()
            ],
            "config": {
                "check_interval_seconds": self.config.check_interval_seconds,
                "min_replicas": self.config.min_replicas,
                "warning_threshold_minutes": self.config.warning_threshold_minutes,
                "critical_threshold_minutes": self.config.critical_threshold_minutes,
                "enable_emergency_sync": self.config.enable_emergency_sync,
            },
            "health": self._compute_health_score(),
        }

    def _compute_health_score(self) -> dict[str, Any]:
        """Compute overall replication health score."""
        if self._stats.total_games == 0:
            return {"score": 100.0, "status": "no_data"}

        # Calculate percentage properly replicated
        properly_replicated = self._stats.total_games - self._stats.under_replicated_games
        replication_percent = (properly_replicated / self._stats.total_games) * 100

        # Penalize for single/zero copy games
        single_copy_penalty = min(self._stats.single_copy_games / 100, 20)
        zero_copy_penalty = min(self._stats.zero_copy_games * 5, 50)

        score = max(0, replication_percent - single_copy_penalty - zero_copy_penalty)

        if score >= 95:
            status = "healthy"
        elif score >= 80:
            status = "degraded"
        elif score >= 50:
            status = "at_risk"
        else:
            status = "critical"

        return {
            "score": round(score, 1),
            "status": status,
            "replication_percent": round(replication_percent, 1),
        }

    def get_alerts_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent alert history.

        Args:
            limit: Maximum alerts to return

        Returns:
            List of alert dicts, newest first
        """
        alerts = self._alerts[-limit:]
        return [
            {
                "level": a.level.value,
                "message": a.message,
                "game_count": a.game_count,
                "timestamp": a.timestamp,
                "resolved": a.resolved,
                "resolved_at": a.resolved_at if a.resolved else None,
            }
            for a in reversed(alerts)
        ]


# Module-level singleton
_replication_monitor: ReplicationMonitorDaemon | None = None


def get_replication_monitor(
    config: ReplicationMonitorConfig | None = None,
) -> ReplicationMonitorDaemon:
    """Get the singleton ReplicationMonitorDaemon instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        ReplicationMonitorDaemon instance
    """
    global _replication_monitor
    if _replication_monitor is None:
        _replication_monitor = ReplicationMonitorDaemon(config)
    return _replication_monitor


def reset_replication_monitor() -> None:
    """Reset the singleton (for testing)."""
    global _replication_monitor
    _replication_monitor = None

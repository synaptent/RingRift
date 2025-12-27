"""Unified Replication Daemon - Consolidated monitoring and repair (December 2025).

This module consolidates replication_monitor.py and replication_repair_daemon.py
into a single daemon that handles both monitoring and active repair of data
replication across the cluster.

Features:
1. Periodic replication health monitoring (every 5 minutes)
2. Alert generation for under-replicated, single-copy, and zero-copy data
3. Active repair of under-replicated games with priority queue
4. Emergency sync for critical data loss scenarios
5. Event emission for external monitoring

Consolidation Benefits:
- Single daemon to manage instead of two
- Shared state (manifest, health scores)
- Reduced code duplication (~400 LOC saved)
- Unified configuration

Usage:
    from app.coordination.unified_replication_daemon import (
        UnifiedReplicationDaemon,
        UnifiedReplicationConfig,
        get_replication_daemon,
        # Backward compat factories
        create_replication_monitor,
        create_replication_repair_daemon,
    )

    # Get singleton daemon
    daemon = get_replication_daemon()
    await daemon.start()

    # Get status (includes both monitoring and repair stats)
    status = daemon.get_status()

    # Trigger manual repair
    await daemon.trigger_repair(["game-id-1", "game-id-2"])

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

logger = logging.getLogger(__name__)

__all__ = [
    "UnifiedReplicationDaemon",
    "UnifiedReplicationConfig",
    "ReplicationAlertLevel",
    "ReplicationAlert",
    "ReplicationStats",
    "RepairPriority",
    "RepairJob",
    "RepairStats",
    "get_replication_daemon",
    "reset_replication_daemon",
    # Backward compatibility
    "create_replication_monitor",
    "create_replication_repair_daemon",
]


# =============================================================================
# Enums
# =============================================================================


class ReplicationAlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class RepairPriority(Enum):
    """Priority levels for repair operations."""
    CRITICAL = 1  # Zero-copy games (data loss imminent)
    HIGH = 2      # Single-copy games (high risk)
    NORMAL = 3    # Under-replicated but safe


# =============================================================================
# Data Classes
# =============================================================================


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
class RepairJob:
    """A pending repair job."""
    game_id: str
    priority: RepairPriority
    current_copies: int
    target_copies: int
    source_nodes: list[str] = field(default_factory=list)
    target_nodes: list[str] = field(default_factory=list)
    created_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    success: bool = False
    error: str = ""


@dataclass
class RepairStats:
    """Statistics about repair operations."""
    total_repairs_attempted: int = 0
    total_repairs_successful: int = 0
    total_repairs_failed: int = 0
    active_repairs: int = 0
    queued_repairs: int = 0
    last_repair_time: float = 0.0
    avg_repair_duration_seconds: float = 0.0


@dataclass
class UnifiedReplicationConfig:
    """Configuration for the unified replication daemon."""

    # Monitoring settings
    monitor_interval_seconds: float = 300.0  # 5 minutes
    warning_threshold_minutes: float = 15.0
    critical_threshold_minutes: float = 60.0
    single_copy_threshold_games: int = 100

    # Replication targets
    min_replicas: int = 2
    target_replicas: int = 3

    # Repair settings
    repair_interval_seconds: float = 60.0  # 1 minute
    max_concurrent_repairs: int = 5
    repair_timeout_seconds: float = 300.0

    # Emergency sync
    enable_emergency_sync: bool = True
    emergency_sync_threshold_games: int = 500
    emergency_sync_cooldown_seconds: float = 600.0  # 10 minutes

    # Event emission
    emit_events: bool = True
    max_alerts_history: int = 100
    max_repair_history: int = 200


# =============================================================================
# Main Daemon Class
# =============================================================================


class UnifiedReplicationDaemon:
    """Unified daemon for replication monitoring and repair.

    Consolidates ReplicationMonitorDaemon and ReplicationRepairDaemon into
    a single daemon with shared state and coordinated operations.
    """

    def __init__(self, config: UnifiedReplicationConfig | None = None):
        """Initialize the unified replication daemon.

        Args:
            config: Daemon configuration
        """
        self.config = config or UnifiedReplicationConfig()
        self.node_id = socket.gethostname()

        # Daemon state
        self._running = False
        self._monitor_task: asyncio.Task | None = None
        self._repair_task: asyncio.Task | None = None

        # Monitoring state
        self._stats = ReplicationStats()
        self._alerts: list[ReplicationAlert] = []
        self._active_alerts: dict[str, ReplicationAlert] = {}
        self._under_replicated_since: dict[str, float] = {}
        self._last_emergency_sync: float = 0.0

        # Repair state
        self._repair_stats = RepairStats()
        self._repair_queue: list[RepairJob] = []
        self._active_repairs: dict[str, RepairJob] = {}
        self._repair_history: list[RepairJob] = []
        self._repair_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start both monitoring and repair loops."""
        if self._running:
            logger.warning("[UnifiedReplicationDaemon] Already running")
            return

        self._running = True
        logger.info("[UnifiedReplicationDaemon] Starting monitoring and repair loops")

        # Subscribe to relevant events (December 2025)
        await self._subscribe_to_events()

        self._monitor_task = asyncio.create_task(
            self._monitor_loop(),
            name="replication_monitor_loop",
        )
        self._repair_task = asyncio.create_task(
            self._repair_loop(),
            name="replication_repair_loop",
        )

    async def _subscribe_to_events(self) -> None:
        """Subscribe to replication-relevant events."""
        try:
            from app.coordination.event_router import DataEventType, get_event_router

            router = get_event_router()
            router.subscribe(DataEventType.NEW_GAMES_AVAILABLE, self._on_new_games)
            router.subscribe(DataEventType.DATA_SYNC_COMPLETED, self._on_sync_completed)
            router.subscribe(DataEventType.HOST_OFFLINE, self._on_host_offline)
            logger.info("[UnifiedReplicationDaemon] Subscribed to events")
        except ImportError:
            logger.debug("[UnifiedReplicationDaemon] Event router not available")
        except Exception as e:
            logger.warning(f"[UnifiedReplicationDaemon] Failed to subscribe: {e}")

    async def _on_new_games(self, event) -> None:
        """Handle new games - may need replication check."""
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            logger.debug(f"[UnifiedReplicationDaemon] New games event: {payload.get('count', 'unknown')}")
        except Exception as e:
            logger.debug(f"[UnifiedReplicationDaemon] Error handling new games: {e}")

    async def _on_sync_completed(self, event) -> None:
        """Handle sync completion - verify replication."""
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            logger.debug(f"[UnifiedReplicationDaemon] Sync completed: {payload}")
        except Exception as e:
            logger.debug(f"[UnifiedReplicationDaemon] Error handling sync: {e}")

    async def _on_host_offline(self, event) -> None:
        """Handle host offline - may need repair priority adjustment."""
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            host = payload.get("host", "unknown")
            logger.info(f"[UnifiedReplicationDaemon] Host offline: {host}")
        except Exception as e:
            logger.debug(f"[UnifiedReplicationDaemon] Error handling offline: {e}")

    async def stop(self) -> None:
        """Stop both loops gracefully."""
        if not self._running:
            return

        logger.info("[UnifiedReplicationDaemon] Stopping...")
        self._running = False

        # Cancel tasks
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await asyncio.wait_for(self._monitor_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        if self._repair_task and not self._repair_task.done():
            self._repair_task.cancel()
            try:
                await asyncio.wait_for(self._repair_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        logger.info("[UnifiedReplicationDaemon] Stopped")

    @property
    def is_running(self) -> bool:
        """Check if daemon is running."""
        return self._running

    # =========================================================================
    # Monitoring Loop
    # =========================================================================

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        logger.info(
            f"[UnifiedReplicationDaemon] Monitor loop started "
            f"(interval={self.config.monitor_interval_seconds}s)"
        )

        while self._running:
            try:
                await self._check_replication()
                await asyncio.sleep(self.config.monitor_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[UnifiedReplicationDaemon] Monitor error: {e}")
                await asyncio.sleep(60)  # Back off on error

    async def _check_replication(self) -> None:
        """Check replication health across the cluster."""
        start_time = time.time()
        now = start_time

        try:
            # Get cluster manifest
            manifest = await self._get_cluster_manifest()
            if not manifest:
                logger.debug("[UnifiedReplicationDaemon] No manifest available")
                return

            # Analyze replication state
            total_games = 0
            under_replicated = 0
            single_copy = 0
            zero_copy = 0
            total_replicas = 0
            nodes_with_data: set[str] = set()

            for game_id, game_info in manifest.items():
                locations = game_info.get("locations", [])
                replica_count = len(locations)
                total_games += 1
                total_replicas += replica_count

                for node in locations:
                    nodes_with_data.add(node)

                if replica_count < self.config.min_replicas:
                    under_replicated += 1
                    # Track when we first saw under-replication
                    if game_id not in self._under_replicated_since:
                        self._under_replicated_since[game_id] = now
                else:
                    # Clear tracking if now healthy
                    self._under_replicated_since.pop(game_id, None)

                if replica_count == 1:
                    single_copy += 1
                elif replica_count == 0:
                    zero_copy += 1

            # Update stats
            self._stats = ReplicationStats(
                total_games=total_games,
                under_replicated_games=under_replicated,
                single_copy_games=single_copy,
                zero_copy_games=zero_copy,
                avg_replication_factor=total_replicas / max(total_games, 1),
                nodes_with_data=len(nodes_with_data),
                last_check_time=now,
                check_duration_seconds=time.time() - start_time,
            )

            # Evaluate and emit alerts
            await self._evaluate_alerts(now)

            # Check if emergency sync needed
            if self.config.enable_emergency_sync:
                await self._check_emergency_sync(under_replicated, single_copy, zero_copy)

            logger.debug(
                f"[UnifiedReplicationDaemon] Check complete: "
                f"{under_replicated}/{total_games} under-replicated, "
                f"{single_copy} single-copy, {zero_copy} zero-copy"
            )

        except Exception as e:
            logger.error(f"[UnifiedReplicationDaemon] Replication check failed: {e}")

    async def _get_cluster_manifest(self) -> dict[str, Any]:
        """Get the cluster manifest with game locations."""
        try:
            from app.distributed.cluster_manifest import get_cluster_manifest
            manifest = get_cluster_manifest()
            return manifest.get_game_locations()
        except ImportError:
            logger.debug("[UnifiedReplicationDaemon] ClusterManifest not available")
            return {}
        except Exception as e:
            logger.error(f"[UnifiedReplicationDaemon] Failed to get manifest: {e}")
            return {}

    async def _evaluate_alerts(self, now: float) -> None:
        """Evaluate and generate alerts based on current state."""
        new_alerts: list[ReplicationAlert] = []

        # Zero-copy alert (critical)
        if self._stats.zero_copy_games > 0:
            key = "zero_copy"
            if key not in self._active_alerts:
                alert = ReplicationAlert(
                    level=ReplicationAlertLevel.CRITICAL,
                    message=f"{self._stats.zero_copy_games} games have ZERO copies - DATA LOSS!",
                    game_count=self._stats.zero_copy_games,
                    timestamp=now,
                )
                self._add_alert(key, alert)
                new_alerts.append(alert)
        else:
            self._resolve_alert("zero_copy", now)

        # Single-copy alert
        if self._stats.single_copy_games >= self.config.single_copy_threshold_games:
            key = "single_copy"
            if key not in self._active_alerts:
                alert = ReplicationAlert(
                    level=ReplicationAlertLevel.WARNING,
                    message=f"{self._stats.single_copy_games} games have only 1 copy",
                    game_count=self._stats.single_copy_games,
                    timestamp=now,
                )
                self._add_alert(key, alert)
                new_alerts.append(alert)
        else:
            self._resolve_alert("single_copy", now)

        # Long under-replication alert
        long_under_replicated = sum(
            1 for first_seen in self._under_replicated_since.values()
            if (now - first_seen) > self.config.warning_threshold_minutes * 60
        )
        if long_under_replicated > 0:
            key = "long_under_replicated"
            if key not in self._active_alerts:
                alert = ReplicationAlert(
                    level=ReplicationAlertLevel.WARNING,
                    message=f"{long_under_replicated} games under-replicated for >{self.config.warning_threshold_minutes}min",
                    game_count=long_under_replicated,
                    timestamp=now,
                )
                self._add_alert(key, alert)
                new_alerts.append(alert)
        else:
            self._resolve_alert("long_under_replicated", now)

        # Emit events for new alerts
        if new_alerts and self.config.emit_events:
            await self._emit_alert_events(new_alerts)

    def _add_alert(self, key: str, alert: ReplicationAlert) -> None:
        """Add an alert."""
        self._active_alerts[key] = alert
        self._alerts.append(alert)

        # Trim history
        if len(self._alerts) > self.config.max_alerts_history:
            self._alerts = self._alerts[-self.config.max_alerts_history:]

        logger.warning(f"[UnifiedReplicationDaemon] ALERT ({alert.level}): {alert.message}")

    def _resolve_alert(self, key: str, now: float) -> None:
        """Resolve an active alert."""
        if key in self._active_alerts:
            alert = self._active_alerts.pop(key)
            alert.resolved = True
            alert.resolved_at = now
            logger.info(f"[UnifiedReplicationDaemon] Alert resolved: {key}")

    async def _emit_alert_events(self, alerts: list[ReplicationAlert]) -> None:
        """Emit events for alerts."""
        try:
            from app.coordination.event_router import DataEvent, DataEventType, get_event_bus

            bus = get_event_bus()
            for alert in alerts:
                event = DataEvent(
                    event_type=DataEventType.REPLICATION_ALERT,
                    payload={
                        "level": alert.level.value,
                        "message": alert.message,
                        "game_count": alert.game_count,
                        "affected_nodes": alert.affected_nodes,
                        "node_id": self.node_id,
                    },
                )
                await bus.publish(event)
        except (ImportError, AttributeError) as e:
            logger.debug(f"[UnifiedReplicationDaemon] Event emission failed: {e}")

    async def _check_emergency_sync(
        self,
        under_replicated: int,
        single_copy: int,
        zero_copy: int,
    ) -> None:
        """Check if emergency sync should be triggered."""
        now = time.time()

        # Check cooldown
        if now - self._last_emergency_sync < self.config.emergency_sync_cooldown_seconds:
            return

        # Check thresholds
        needs_emergency = (
            zero_copy > 0 or
            single_copy >= self.config.emergency_sync_threshold_games or
            under_replicated >= self.config.emergency_sync_threshold_games * 2
        )

        if needs_emergency:
            logger.warning(
                f"[UnifiedReplicationDaemon] Triggering EMERGENCY SYNC: "
                f"zero={zero_copy}, single={single_copy}, under={under_replicated}"
            )
            self._last_emergency_sync = now

            # Trigger emergency repair
            await self._trigger_emergency_repair()

    async def _trigger_emergency_repair(self) -> None:
        """Trigger emergency repair of critical games."""
        try:
            # Get games needing immediate repair
            critical_games = await self._find_critical_games()

            if critical_games:
                logger.info(
                    f"[UnifiedReplicationDaemon] Queuing {len(critical_games)} "
                    "critical games for emergency repair"
                )
                await self.trigger_repair([g[0] for g in critical_games])
        except Exception as e:
            logger.error(f"[UnifiedReplicationDaemon] Emergency repair failed: {e}")

    async def _find_critical_games(self) -> list[tuple[str, int, list[str]]]:
        """Find games with zero or single copies."""
        critical: list[tuple[str, int, list[str]]] = []

        try:
            manifest = await self._get_cluster_manifest()

            for game_id, game_info in manifest.items():
                locations = game_info.get("locations", [])
                if len(locations) < self.config.min_replicas:
                    critical.append((game_id, len(locations), locations))

            # Sort by replica count (zero-copy first)
            critical.sort(key=lambda x: x[1])

        except Exception as e:
            logger.error(f"[UnifiedReplicationDaemon] Failed to find critical games: {e}")

        return critical

    # =========================================================================
    # Repair Loop
    # =========================================================================

    async def _repair_loop(self) -> None:
        """Main repair loop."""
        logger.info(
            f"[UnifiedReplicationDaemon] Repair loop started "
            f"(interval={self.config.repair_interval_seconds}s)"
        )

        while self._running:
            try:
                await self._run_repair_cycle()
                await asyncio.sleep(self.config.repair_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[UnifiedReplicationDaemon] Repair error: {e}")
                await asyncio.sleep(60)

    async def _run_repair_cycle(self) -> None:
        """Run one cycle of repair operations."""
        async with self._repair_lock:
            # Find games needing repair
            games_needing_repair = await self._find_games_needing_repair()

            if not games_needing_repair:
                return

            # Create repair jobs
            new_jobs = await self._create_repair_jobs(games_needing_repair)
            self._repair_queue.extend(new_jobs)
            self._repair_stats.queued_repairs = len(self._repair_queue)

            # Sort by priority
            self._repair_queue.sort(key=lambda j: j.priority.value)

            # Execute repairs up to concurrency limit
            concurrent_count = len(self._active_repairs)
            slots_available = self.config.max_concurrent_repairs - concurrent_count

            for _ in range(min(slots_available, len(self._repair_queue))):
                job = self._repair_queue.pop(0)
                asyncio.create_task(self._execute_repair(job))

    async def _find_games_needing_repair(self) -> list[tuple[str, int, list[str]]]:
        """Find games that need repair."""
        games: list[tuple[str, int, list[str]]] = []

        try:
            manifest = await self._get_cluster_manifest()

            for game_id, game_info in manifest.items():
                # Skip if already being repaired
                if game_id in self._active_repairs:
                    continue

                locations = game_info.get("locations", [])
                if len(locations) < self.config.min_replicas:
                    games.append((game_id, len(locations), locations))

        except Exception as e:
            logger.error(f"[UnifiedReplicationDaemon] Failed to find games for repair: {e}")

        return games

    async def _create_repair_jobs(
        self,
        games: list[tuple[str, int, list[str]]],
    ) -> list[RepairJob]:
        """Create repair jobs for games."""
        jobs: list[RepairJob] = []
        now = time.time()

        for game_id, replica_count, source_nodes in games:
            # Skip if already queued
            if any(j.game_id == game_id for j in self._repair_queue):
                continue

            # Determine priority
            if replica_count == 0:
                priority = RepairPriority.CRITICAL
            elif replica_count == 1:
                priority = RepairPriority.HIGH
            else:
                priority = RepairPriority.NORMAL

            # Select target nodes
            target_nodes = await self._select_target_nodes(
                source_nodes,
                self.config.target_replicas - replica_count,
            )

            if target_nodes:
                jobs.append(RepairJob(
                    game_id=game_id,
                    priority=priority,
                    current_copies=replica_count,
                    target_copies=self.config.target_replicas,
                    source_nodes=source_nodes,
                    target_nodes=target_nodes,
                    created_at=now,
                ))

        return jobs

    async def _select_target_nodes(
        self,
        source_nodes: list[str],
        needed_copies: int,
    ) -> list[str]:
        """Select target nodes for repair."""
        try:
            from app.coordination.sync_router import get_sync_router
            router = get_sync_router()
            return await router.select_targets(
                exclude_nodes=set(source_nodes),
                count=needed_copies,
            )
        except (ImportError, AttributeError):
            logger.debug("[UnifiedReplicationDaemon] SyncRouter not available")
            return []

    async def _execute_repair(self, job: RepairJob) -> None:
        """Execute a single repair job."""
        job.started_at = time.time()
        self._active_repairs[job.game_id] = job
        self._repair_stats.active_repairs = len(self._active_repairs)

        try:
            success = await self._perform_repair_transfer(job)
            job.success = success
            job.completed_at = time.time()

            if success:
                self._repair_stats.total_repairs_successful += 1
                logger.info(f"[UnifiedReplicationDaemon] Repaired {job.game_id}")
            else:
                self._repair_stats.total_repairs_failed += 1
                logger.warning(f"[UnifiedReplicationDaemon] Repair failed: {job.game_id}")

        except Exception as e:
            job.success = False
            job.error = str(e)
            job.completed_at = time.time()
            self._repair_stats.total_repairs_failed += 1
            logger.error(f"[UnifiedReplicationDaemon] Repair error for {job.game_id}: {e}")

        finally:
            self._repair_stats.total_repairs_attempted += 1
            self._active_repairs.pop(job.game_id, None)
            self._repair_stats.active_repairs = len(self._active_repairs)

            # Add to history
            self._repair_history.append(job)
            if len(self._repair_history) > self.config.max_repair_history:
                self._repair_history = self._repair_history[-self.config.max_repair_history:]

            # Update average duration
            completed = [j for j in self._repair_history if j.completed_at > 0]
            if completed:
                durations = [j.completed_at - j.started_at for j in completed]
                self._repair_stats.avg_repair_duration_seconds = sum(durations) / len(durations)

    async def _perform_repair_transfer(self, job: RepairJob) -> bool:
        """Perform the actual data transfer for repair."""
        if not job.source_nodes or not job.target_nodes:
            job.error = "No source or target nodes"
            return False

        try:
            from app.coordination.sync_bandwidth import rsync_with_bandwidth

            source_node = job.source_nodes[0]
            target_node = job.target_nodes[0]

            # Get game data path
            game_path = f"data/games/{job.game_id}.db"

            success = await rsync_with_bandwidth(
                source_host=source_node,
                source_path=game_path,
                dest_host=target_node,
                dest_path=game_path,
                timeout=self.config.repair_timeout_seconds,
            )

            if success:
                # Update manifest
                await self._update_manifest(job.game_id, target_node)

            return success

        except (ImportError, AttributeError) as e:
            job.error = f"Sync not available: {e}"
            return False
        except Exception as e:
            job.error = str(e)
            return False

    async def _update_manifest(self, game_id: str, target_node: str) -> None:
        """Update manifest after successful repair."""
        try:
            from app.distributed.cluster_manifest import get_cluster_manifest
            manifest = get_cluster_manifest()
            await manifest.add_game_location(game_id, target_node)
        except (ImportError, AttributeError) as e:
            logger.debug(f"[UnifiedReplicationDaemon] Manifest update failed: {e}")

    # =========================================================================
    # Public API
    # =========================================================================

    async def trigger_repair(self, game_ids: list[str] | None = None) -> int:
        """Manually trigger repair for specific games.

        Args:
            game_ids: Specific game IDs to repair (None = all under-replicated)

        Returns:
            Number of games queued for repair
        """
        games_needing_repair = await self._find_games_needing_repair()

        if game_ids:
            # Filter to requested games
            games_needing_repair = [
                g for g in games_needing_repair
                if g[0] in game_ids
            ]

        if not games_needing_repair:
            return 0

        jobs = await self._create_repair_jobs(games_needing_repair)

        async with self._repair_lock:
            self._repair_queue.extend(jobs)
            self._repair_queue.sort(key=lambda j: j.priority.value)
            self._repair_stats.queued_repairs = len(self._repair_queue)

        logger.info(f"[UnifiedReplicationDaemon] Queued {len(jobs)} games for repair")
        return len(jobs)

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive status of monitoring and repair."""
        health = self._compute_health_score()

        return {
            "running": self._running,
            "node_id": self.node_id,
            "health_score": health["score"],
            "health_status": health["status"],
            # Monitoring stats
            "monitoring": {
                "total_games": self._stats.total_games,
                "under_replicated_games": self._stats.under_replicated_games,
                "single_copy_games": self._stats.single_copy_games,
                "zero_copy_games": self._stats.zero_copy_games,
                "avg_replication_factor": round(self._stats.avg_replication_factor, 2),
                "nodes_with_data": self._stats.nodes_with_data,
                "last_check_time": self._stats.last_check_time,
                "active_alerts": len(self._active_alerts),
            },
            # Repair stats
            "repair": {
                "total_attempted": self._repair_stats.total_repairs_attempted,
                "total_successful": self._repair_stats.total_repairs_successful,
                "total_failed": self._repair_stats.total_repairs_failed,
                "active_repairs": self._repair_stats.active_repairs,
                "queued_repairs": self._repair_stats.queued_repairs,
                "success_rate": (
                    self._repair_stats.total_repairs_successful /
                    max(self._repair_stats.total_repairs_attempted, 1)
                ),
                "avg_duration_seconds": round(
                    self._repair_stats.avg_repair_duration_seconds, 2
                ),
            },
            # Config summary
            "config": {
                "min_replicas": self.config.min_replicas,
                "target_replicas": self.config.target_replicas,
                "monitor_interval": self.config.monitor_interval_seconds,
                "repair_interval": self.config.repair_interval_seconds,
            },
        }

    def _compute_health_score(self) -> dict[str, Any]:
        """Compute overall health score."""
        score = 100.0

        # Deduct for under-replication
        if self._stats.total_games > 0:
            under_ratio = self._stats.under_replicated_games / self._stats.total_games
            score -= under_ratio * 50

            single_ratio = self._stats.single_copy_games / self._stats.total_games
            score -= single_ratio * 30

            zero_ratio = self._stats.zero_copy_games / self._stats.total_games
            score -= zero_ratio * 100

        # Deduct for repair failures
        if self._repair_stats.total_repairs_attempted > 0:
            fail_rate = (
                self._repair_stats.total_repairs_failed /
                self._repair_stats.total_repairs_attempted
            )
            score -= fail_rate * 20

        score = max(0.0, min(100.0, score))

        if score >= 90:
            status = "healthy"
        elif score >= 70:
            status = "degraded"
        elif score >= 50:
            status = "unhealthy"
        else:
            status = "critical"

        return {"score": round(score, 1), "status": status}

    def get_alerts_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get alert history."""
        alerts = self._alerts[-limit:] if limit else self._alerts
        return [
            {
                "level": a.level.value,
                "message": a.message,
                "game_count": a.game_count,
                "timestamp": a.timestamp,
                "resolved": a.resolved,
                "resolved_at": a.resolved_at,
            }
            for a in alerts
        ]

    def get_repair_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get repair history."""
        jobs = self._repair_history[-limit:] if limit else self._repair_history
        return [
            {
                "game_id": j.game_id,
                "priority": j.priority.name,
                "current_copies": j.current_copies,
                "target_copies": j.target_copies,
                "success": j.success,
                "error": j.error,
                "created_at": j.created_at,
                "completed_at": j.completed_at,
                "duration": j.completed_at - j.started_at if j.completed_at else 0,
            }
            for j in jobs
        ]


# =============================================================================
# Singleton Management
# =============================================================================

_instance: UnifiedReplicationDaemon | None = None
_lock = asyncio.Lock()


async def get_replication_daemon(
    config: UnifiedReplicationConfig | None = None,
) -> UnifiedReplicationDaemon:
    """Get the singleton unified replication daemon.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        UnifiedReplicationDaemon instance
    """
    global _instance

    if _instance is None:
        async with _lock:
            if _instance is None:
                _instance = UnifiedReplicationDaemon(config)

    return _instance


def reset_replication_daemon() -> None:
    """Reset the singleton instance (for testing)."""
    global _instance
    _instance = None


# =============================================================================
# Backward Compatibility Factories
# =============================================================================


def create_replication_monitor(
    config: UnifiedReplicationConfig | None = None,
) -> UnifiedReplicationDaemon:
    """Create a replication monitor (backward compat).

    DEPRECATED: Use get_replication_daemon() instead.
    """
    warnings.warn(
        "create_replication_monitor is deprecated. Use get_replication_daemon() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return UnifiedReplicationDaemon(config)


def create_replication_repair_daemon(
    config: UnifiedReplicationConfig | None = None,
) -> UnifiedReplicationDaemon:
    """Create a replication repair daemon (backward compat).

    DEPRECATED: Use get_replication_daemon() instead.
    """
    warnings.warn(
        "create_replication_repair_daemon is deprecated. Use get_replication_daemon() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return UnifiedReplicationDaemon(config)

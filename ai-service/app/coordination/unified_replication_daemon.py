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

# December 2025: Use consolidated daemon stats base classes
from app.config.coordination_defaults import SyncDefaults
from app.coordination.daemon_stats import DaemonStatsBase, JobDaemonStats, PerNodeSyncStats
from app.coordination.handler_base import HandlerBase
from app.coordination.contracts import CoordinatorStatus, HealthCheckResult
from app.utils.async_utils import fire_and_forget

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

# Import from canonical location (December 2025 consolidation)
from app.coordination.alert_types import ReplicationAlertLevel


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
class ReplicationStats(DaemonStatsBase):
    """Statistics about replication health.

    December 2025: Now extends DaemonStatsBase for consistent tracking.
    Inherits: operations_attempted, operations_completed, operations_failed,
              is_healthy(), to_dict(), etc.
    """

    # Replication-specific monitoring fields
    total_games: int = 0
    under_replicated_games: int = 0
    single_copy_games: int = 0
    zero_copy_games: int = 0
    avg_replication_factor: float = 0.0
    nodes_with_data: int = 0
    check_duration_seconds: float = 0.0

    # Note: last_check_time is inherited from DaemonStatsBase

    def record_check(self, duration: float, total_games: int, under_replicated: int,
                     single_copy: int, zero_copy: int, avg_factor: float, nodes: int) -> None:
        """Record a replication health check."""
        self.record_attempt()
        self.record_success(duration_seconds=duration)
        self.check_duration_seconds = duration
        self.total_games = total_games
        self.under_replicated_games = under_replicated
        self.single_copy_games = single_copy
        self.zero_copy_games = zero_copy
        self.avg_replication_factor = avg_factor
        self.nodes_with_data = nodes


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
class RepairStats(JobDaemonStats):
    """Statistics about repair operations.

    December 2025: Now extends JobDaemonStats for consistent tracking.
    Inherits: jobs_processed, jobs_succeeded, jobs_failed, jobs_timed_out,
              last_job_time, avg_operation_duration, record_job_success(), etc.
    """

    # Repair-specific fields
    active_repairs: int = 0
    queued_repairs: int = 0

    # Backward-compatible aliases
    @property
    def total_repairs_attempted(self) -> int:
        """Alias for jobs_processed."""
        return self.jobs_processed

    @property
    def total_repairs_successful(self) -> int:
        """Alias for jobs_succeeded."""
        return self.jobs_succeeded

    @property
    def total_repairs_failed(self) -> int:
        """Alias for jobs_failed."""
        return self.jobs_failed

    @property
    def last_repair_time(self) -> float:
        """Alias for last_job_time (from JobDaemonStats)."""
        return self.last_job_time

    @property
    def avg_repair_duration_seconds(self) -> float:
        """Alias for avg_operation_duration (from DaemonStatsBase)."""
        return self.avg_operation_duration

    # Convenience methods
    def record_repair_start(self) -> None:
        """Record the start of a repair operation."""
        self.active_repairs += 1

    def record_repair_success(self, duration: float) -> None:
        """Record a successful repair operation."""
        self.record_job_success(duration_seconds=duration)
        if self.active_repairs > 0:
            self.active_repairs -= 1

    def record_repair_failure(self) -> None:
        """Record a failed repair operation."""
        self.record_job_failure()
        if self.active_repairs > 0:
            self.active_repairs -= 1

    def set_queue_size(self, size: int) -> None:
        """Update the queued repairs count."""
        self.queued_repairs = size


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

    # December 2025: Rate limiting to prevent cluster spam
    # (Harvested from deprecated replication_repair_daemon.py)
    max_repairs_per_hour: int = 500  # Hourly limit on repair attempts
    repair_cooldown_per_game_seconds: float = 600.0  # 10 min cooldown per game

    # December 2025: Ephemeral target avoidance
    # Vast.ai nodes are unreliable as replication targets
    avoid_ephemeral_targets: bool = True

    # December 2025: Bandwidth limiting per repair
    bandwidth_limit_mbps: float = 50.0  # Default 50 MB/s per transfer

    # Emergency sync - use centralized defaults
    enable_emergency_sync: bool = True
    emergency_sync_threshold_games: int = 500
    emergency_sync_cooldown_seconds: float = field(
        default_factory=lambda: SyncDefaults.EMERGENCY_SYNC_COOLDOWN  # 10 minutes
    )

    # Event emission
    emit_events: bool = True
    max_alerts_history: int = 100
    max_repair_history: int = 200


# =============================================================================
# Main Daemon Class
# =============================================================================


class UnifiedReplicationDaemon(HandlerBase):
    """Unified daemon for replication monitoring and repair.

    Consolidates ReplicationMonitorDaemon and ReplicationRepairDaemon into
    a single daemon with shared state and coordinated operations.

    January 2026: Migrated to HandlerBase for unified lifecycle and singleton.
    Uses _run_cycle() for repair loop (1min) and _on_start() to spawn monitor loop (5min).
    """

    def __init__(self, config: UnifiedReplicationConfig | None = None):
        """Initialize the unified replication daemon.

        Args:
            config: Daemon configuration
        """
        self.config = config or UnifiedReplicationConfig()
        self.node_id = socket.gethostname()

        # Initialize HandlerBase - repair loop is more frequent (1 min)
        super().__init__(
            name=f"unified_replication_{self.node_id}",
            cycle_interval=self.config.repair_interval_seconds,
        )

        # Background task for monitor loop (different interval)
        self._monitor_task: asyncio.Task | None = None

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

        # December 2025: Rate limiting state
        # (Harvested from deprecated replication_repair_daemon.py)
        self._hourly_repair_count: int = 0
        self._hourly_reset_time: float = time.time()
        self._repair_cooldowns: dict[str, float] = {}  # game_id -> last_attempt_time

        # December 2025: Per-node sync reliability tracking
        # (Harvested from deprecated replication_monitor.py)
        self._node_sync_stats: dict[str, PerNodeSyncStats] = {}

    def _get_event_subscriptions(self) -> dict:
        """Get declarative event subscriptions (HandlerBase pattern).

        Returns:
            Dict mapping event names to handler methods
        """
        return {
            "NEW_GAMES_AVAILABLE": self._on_new_games,
            "DATA_SYNC_COMPLETED": self._on_sync_completed,
            "HOST_OFFLINE": self._on_host_offline,
        }

    async def _on_start(self) -> None:
        """Hook called after HandlerBase start (spawns monitor loop)."""
        # Start monitor loop (runs on different interval than repair)
        self._monitor_task = self._safe_create_task(
            self._monitor_loop(),
            context="replication_monitor_loop"
        )
        logger.info(
            f"[UnifiedReplicationDaemon] Started monitoring and repair loops"
        )

    async def _on_stop(self) -> None:
        """Hook called before HandlerBase stop (cancels monitor loop)."""
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await asyncio.wait_for(self._monitor_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._monitor_task = None
        logger.info("[UnifiedReplicationDaemon] Stopped")

    async def _run_cycle(self) -> None:
        """Run repair cycle (HandlerBase main work loop).

        Called every cycle_interval (1 min) for repair operations.
        """
        try:
            await self._run_repair_cycle()
            self._record_success()
        except Exception as e:
            logger.error(f"[UnifiedReplicationDaemon] Repair error: {e}")
            self._record_error(str(e))

    async def start(self) -> None:
        """Start both monitoring and repair loops (delegates to HandlerBase)."""
        await super().start()

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
        """Stop both loops gracefully (delegates to HandlerBase)."""
        await super().stop()

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
        """Trigger emergency repair of critical games.

        December 31, 2025: Added batch limit to prevent CPU spike from processing
        hundreds of thousands of games at once. Only the most critical games
        (zero-copy first, then lowest replica count) are queued each cycle.
        """
        # Limit batch size to prevent CPU spike
        MAX_EMERGENCY_BATCH = 1000

        try:
            # Get games needing immediate repair
            critical_games = await self._find_critical_games()

            if critical_games:
                # Limit to batch size - already sorted by criticality (zero-copy first)
                batch_games = critical_games[:MAX_EMERGENCY_BATCH]
                skipped = len(critical_games) - len(batch_games)

                if skipped > 0:
                    logger.info(
                        f"[UnifiedReplicationDaemon] Queuing {len(batch_games)} "
                        f"critical games for emergency repair (skipped {skipped} for next cycle)"
                    )
                else:
                    logger.info(
                        f"[UnifiedReplicationDaemon] Queuing {len(batch_games)} "
                        "critical games for emergency repair"
                    )
                await self.trigger_repair([g[0] for g in batch_games])
        except Exception as e:
            logger.error(f"[UnifiedReplicationDaemon] Emergency repair failed: {e}")

    async def _find_critical_games(self) -> list[tuple[str, int, list[str]]]:
        """Find games with zero or single copies.

        December 31, 2025: Added limit to prevent processing 300K+ games.
        Uses early-exit once we have enough critical games for a batch.
        """
        # Limit search - we only process 1000 per batch anyway
        MAX_SEARCH = 5000
        critical: list[tuple[str, int, list[str]]] = []
        zero_copy: list[tuple[str, int, list[str]]] = []  # Priority queue for zero-copy

        try:
            manifest = await self._get_cluster_manifest()

            for game_id, game_info in manifest.items():
                locations = game_info.get("locations", [])
                if len(locations) < self.config.min_replicas:
                    if len(locations) == 0:
                        zero_copy.append((game_id, 0, []))
                    else:
                        critical.append((game_id, len(locations), locations))

                # Early exit once we have enough
                if len(zero_copy) + len(critical) >= MAX_SEARCH:
                    logger.debug(
                        f"[UnifiedReplicationDaemon] Early exit at {MAX_SEARCH} critical games"
                    )
                    break

            # Combine: zero-copy first, then by replica count
            critical.sort(key=lambda x: x[1])
            return zero_copy + critical

        except Exception as e:
            logger.error(f"[UnifiedReplicationDaemon] Failed to find critical games: {e}")

        return critical

    # =========================================================================
    # Repair Loop
    # =========================================================================

    # =========================================================================
    # Rate Limiting (December 2025 - harvested from deprecated daemon)
    # =========================================================================

    def _check_and_reset_hourly_limit(self) -> bool:
        """Check if hourly repair limit is reached, reset if hour elapsed.

        Returns:
            True if repairs are allowed, False if hourly limit reached
        """
        now = time.time()
        elapsed = now - self._hourly_reset_time

        # Reset counter every hour (3600 seconds)
        if elapsed >= 3600.0:
            self._hourly_repair_count = 0
            self._hourly_reset_time = now
            logger.debug("[UnifiedReplicationDaemon] Hourly repair counter reset")

        if self._hourly_repair_count >= self.config.max_repairs_per_hour:
            remaining = 3600.0 - elapsed
            logger.warning(
                f"[UnifiedReplicationDaemon] Hourly repair limit reached "
                f"({self._hourly_repair_count}/{self.config.max_repairs_per_hour}), "
                f"reset in {remaining:.0f}s"
            )
            return False

        return True

    def _is_game_on_cooldown(self, game_id: str) -> bool:
        """Check if a game is on repair cooldown.

        Args:
            game_id: The game to check

        Returns:
            True if game is on cooldown, False if repair is allowed
        """
        if game_id not in self._repair_cooldowns:
            return False

        last_attempt = self._repair_cooldowns[game_id]
        elapsed = time.time() - last_attempt

        if elapsed < self.config.repair_cooldown_per_game_seconds:
            remaining = self.config.repair_cooldown_per_game_seconds - elapsed
            logger.debug(
                f"[UnifiedReplicationDaemon] Game {game_id[:8]} on cooldown, "
                f"retry in {remaining:.0f}s"
            )
            return True

        return False

    def _record_repair_attempt(self, game_id: str) -> None:
        """Record a repair attempt for rate limiting."""
        now = time.time()
        self._hourly_repair_count += 1
        self._repair_cooldowns[game_id] = now

        # Prune old cooldowns (>2x cooldown period)
        prune_threshold = now - (self.config.repair_cooldown_per_game_seconds * 2)
        self._repair_cooldowns = {
            gid: ts for gid, ts in self._repair_cooldowns.items()
            if ts > prune_threshold
        }


    async def _run_repair_cycle(self) -> None:
        """Run one cycle of repair operations."""
        # December 2025: Check hourly rate limit before processing
        if not self._check_and_reset_hourly_limit():
            return

        async with self._repair_lock:
            # Find games needing repair
            games_needing_repair = await self._find_games_needing_repair()

            if not games_needing_repair:
                return

            # Create repair jobs (filters out games on cooldown)
            new_jobs = await self._create_repair_jobs(games_needing_repair)
            self._repair_queue.extend(new_jobs)
            self._repair_stats.queued_repairs = len(self._repair_queue)

            # Sort by priority
            self._repair_queue.sort(key=lambda j: j.priority.value)

            # Execute repairs up to concurrency limit
            concurrent_count = len(self._active_repairs)
            slots_available = self.config.max_concurrent_repairs - concurrent_count

            # December 2025: Also limit by remaining hourly budget
            remaining_budget = self.config.max_repairs_per_hour - self._hourly_repair_count
            slots_available = min(slots_available, remaining_budget)

            for _ in range(min(slots_available, len(self._repair_queue))):
                job = self._repair_queue.pop(0)
                fire_and_forget(self._execute_repair(job), name=f"repair_{job.game_id}")

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

            # December 2025: Skip if game is on cooldown
            if self._is_game_on_cooldown(game_id):
                continue

            # Determine priority
            if replica_count == 0:
                priority = RepairPriority.CRITICAL
            elif replica_count == 1:
                priority = RepairPriority.HIGH
            else:
                priority = RepairPriority.NORMAL

            # Select target nodes (filters out ephemeral if configured)
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
        """Select target nodes for repair.

        December 2025: Filters out ephemeral nodes (Vast.ai) as replication
        targets if avoid_ephemeral_targets is enabled. Ephemeral nodes can
        still be used as sources, but should not be targets since they may
        terminate at any time.
        """
        try:
            from app.coordination.sync_router import get_sync_router
            router = get_sync_router()

            # December 2025: Get ephemeral nodes to exclude as targets
            exclude_nodes = set(source_nodes)
            if self.config.avoid_ephemeral_targets:
                ephemeral_nodes = self._get_ephemeral_nodes()
                exclude_nodes.update(ephemeral_nodes)
                if ephemeral_nodes:
                    logger.debug(
                        f"[UnifiedReplicationDaemon] Excluding {len(ephemeral_nodes)} "
                        "ephemeral nodes as repair targets"
                    )

            return await router.select_targets(
                exclude_nodes=exclude_nodes,
                count=needed_copies,
            )
        except (ImportError, AttributeError):
            logger.debug("[UnifiedReplicationDaemon] SyncRouter not available")
            return []

    def _get_ephemeral_nodes(self) -> set[str]:
        """Get set of ephemeral node names (Vast.ai nodes).

        December 2025: Vast.ai nodes are ephemeral and can terminate at any
        time. They should be avoided as replication targets to prevent data loss.
        """
        ephemeral: set[str] = set()
        try:
            from app.config.cluster_config import get_cluster_nodes

            nodes = get_cluster_nodes()
            for name, node in nodes.items():
                # Vast.ai nodes are identified by provider or name pattern
                if node.provider == "vast" or name.startswith("vast-"):
                    ephemeral.add(name)
        except (ImportError, AttributeError) as e:
            logger.debug(f"[UnifiedReplicationDaemon] Could not get cluster nodes: {e}")

        return ephemeral

    async def _execute_repair(self, job: RepairJob) -> None:
        """Execute a single repair job."""
        job.started_at = time.time()
        self._active_repairs[job.game_id] = job
        self._repair_stats.active_repairs = len(self._active_repairs)

        # December 2025: Record attempt for rate limiting
        self._record_repair_attempt(job.game_id)

        try:
            success = await self._perform_repair_transfer(job)
            job.success = success
            job.completed_at = time.time()
            duration = job.completed_at - job.started_at

            if success:
                self._repair_stats.total_repairs_successful += 1
                logger.info(f"[UnifiedReplicationDaemon] Repaired {job.game_id}")
                # December 2025: Emit REPAIR_COMPLETED event
                await self._emit_repair_event(job, success=True, duration=duration)
            else:
                self._repair_stats.total_repairs_failed += 1
                logger.warning(f"[UnifiedReplicationDaemon] Repair failed: {job.game_id}")
                # December 2025: Emit REPAIR_FAILED event
                await self._emit_repair_event(job, success=False, duration=duration)

        except Exception as e:
            job.success = False
            job.error = str(e)
            job.completed_at = time.time()
            self._repair_stats.total_repairs_failed += 1
            logger.error(f"[UnifiedReplicationDaemon] Repair error for {job.game_id}: {e}")
            # December 2025: Emit REPAIR_FAILED event for exceptions
            await self._emit_repair_event(
                job, success=False, duration=job.completed_at - job.started_at
            )

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

            # December 2025: Track per-node sync reliability
            if job.target_nodes:
                repair_duration = job.completed_at - job.started_at if job.completed_at else 0.0
                for target_node in job.target_nodes:
                    if target_node not in self._node_sync_stats:
                        self._node_sync_stats[target_node] = PerNodeSyncStats(node_id=target_node)
                    stats = self._node_sync_stats[target_node]
                    if job.success:
                        stats.record_success(duration=repair_duration)
                    else:
                        stats.record_failure(reason=job.error or "Unknown error")

    async def _perform_repair_transfer(self, job: RepairJob) -> bool:
        """Perform the actual data transfer for repair.

        December 2025: Uses bandwidth-coordinated rsync with per-transfer limits.
        """
        if not job.source_nodes or not job.target_nodes:
            job.error = "No source or target nodes"
            return False

        try:
            from app.coordination.sync_bandwidth import get_coordinated_rsync
            from app.config.cluster_config import get_node_bandwidth_kbs

            source_node = job.source_nodes[0]
            target_node = job.target_nodes[0]

            # Get game data path
            game_path = f"data/games/{job.game_id}.db"

            # December 2025: Get bandwidth limit from cluster_config or use config default
            try:
                bwlimit_kbs = get_node_bandwidth_kbs(target_node)
            except (KeyError, ValueError):
                # Convert config MB/s to KB/s
                bwlimit_kbs = int(self.config.bandwidth_limit_mbps * 1024)

            # Cap at configured maximum
            max_kbs = int(self.config.bandwidth_limit_mbps * 1024)
            bwlimit_kbs = min(bwlimit_kbs, max_kbs)

            rsync = get_coordinated_rsync()
            result = await rsync.sync(
                source=f"{source_node}:{game_path}",
                dest=f"{target_node}:{game_path}",
                host=target_node,
                bwlimit_kbps=bwlimit_kbs,
                timeout=int(self.config.repair_timeout_seconds),
            )

            if result.success:
                # Update manifest
                await self._update_manifest(job.game_id, target_node)
                logger.debug(
                    f"[UnifiedReplicationDaemon] Transferred {job.game_id[:8]} "
                    f"to {target_node} ({result.bytes_transferred} bytes, "
                    f"{result.duration_seconds:.1f}s)"
                )

            return result.success

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

    async def _emit_repair_event(
        self, job: RepairJob, success: bool, duration: float
    ) -> None:
        """Emit REPAIR_COMPLETED or REPAIR_FAILED event.

        December 2025: Added for pipeline coordination of repair operations.
        Events allow other coordinators (e.g., DataPipelineOrchestrator) to
        react to repair status changes.
        """
        if not self.config.emit_events:
            return

        try:
            if success:
                from app.coordination.event_emitters import emit_repair_completed

                await emit_repair_completed(
                    game_id=job.game_id,
                    source_nodes=job.source_nodes,
                    target_nodes=job.target_nodes,
                    duration_seconds=duration,
                    new_replica_count=job.target_copies,
                    priority=job.priority.name,
                )
            else:
                from app.coordination.event_emitters import emit_repair_failed

                await emit_repair_failed(
                    game_id=job.game_id,
                    source_nodes=job.source_nodes,
                    target_nodes=job.target_nodes,
                    error=job.error or "Unknown error",
                    duration_seconds=duration,
                    current_replica_count=job.current_copies,
                    priority=job.priority.name,
                )
        except ImportError:
            logger.debug("[UnifiedReplicationDaemon] Event emitters not available")
        except Exception as e:
            logger.debug(f"[UnifiedReplicationDaemon] Failed to emit repair event: {e}")

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
            # December 2025: Per-node sync reliability metrics
            "per_node_sync": {
                node_id: stats.to_dict()
                for node_id, stats in self._node_sync_stats.items()
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

    def health_check(self) -> HealthCheckResult:
        """Check daemon health (January 2026: HandlerBase pattern).

        Returns:
            HealthCheckResult with status and details
        """
        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="UnifiedReplicationDaemon not running",
            )

        # Use existing health score computation
        health = self._compute_health_score()

        if health["status"] == "critical":
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"Replication critical: {self._stats.zero_copy_games} zero-copy games",
                details=self.get_status(),
            )

        if health["status"] == "unhealthy":
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"Replication unhealthy: {self._stats.under_replicated_games} under-replicated",
                details=self.get_status(),
            )

        # December 2025: Check for unreliable nodes (success rate < 80%)
        unreliable_nodes = [
            node_id
            for node_id, stats in self._node_sync_stats.items()
            if stats.syncs_attempted >= 5 and stats.success_rate < 0.8
        ]
        if unreliable_nodes:
            return HealthCheckResult(
                healthy=True,  # Still healthy, but with warning
                status=CoordinatorStatus.RUNNING,
                message=f"Replication running but {len(unreliable_nodes)} unreliable nodes: {', '.join(unreliable_nodes[:3])}",
                details=self.get_status(),
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"Replication daemon running (score={health['score']:.0f}%)",
            details=self.get_status(),
        )

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
# Singleton Management (January 2026: Delegates to HandlerBase pattern)
# =============================================================================


def get_replication_daemon(
    config: UnifiedReplicationConfig | None = None,
) -> UnifiedReplicationDaemon:
    """Get the singleton unified replication daemon.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        UnifiedReplicationDaemon instance

    January 2026: Now delegates to HandlerBase.get_instance() singleton pattern.
    Note: This is now synchronous (no longer async) since HandlerBase handles it.
    """
    return UnifiedReplicationDaemon.get_instance(config)


def reset_replication_daemon() -> None:
    """Reset the singleton instance (for testing).

    January 2026: Now delegates to HandlerBase.reset_instance().
    """
    UnifiedReplicationDaemon.reset_instance()


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

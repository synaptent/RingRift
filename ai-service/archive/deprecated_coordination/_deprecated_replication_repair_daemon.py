"""Replication Repair Daemon - Active repair of under-replicated data.

.. deprecated:: December 2025
    This module is deprecated. Use unified_replication_daemon.py instead:

    from app.coordination.unified_replication_daemon import (
        UnifiedReplicationDaemon,
        UnifiedReplicationConfig,
        get_replication_daemon,
    )

MIGRATION GUIDE:
    Old (deprecated):
        from app.coordination.replication_repair_daemon import get_replication_repair_daemon
        daemon = get_replication_repair_daemon()

    New (recommended):
        from app.coordination.unified_replication_daemon import get_replication_daemon
        daemon = await get_replication_daemon()

This daemon works alongside ReplicationMonitorDaemon to actively repair
under-replicated games by coordinating targeted sync operations.

Features:
1. Prioritizes repair by risk level (zero-copy > single-copy > under-replicated)
2. Uses SyncRouter for intelligent node selection
3. Respects bandwidth limits and node availability
4. Tracks repair progress and success rates
5. Coordinates with ClusterManifest for accurate replication state

December 2025: Created for Phase 4 replication enforcement.

Usage (DEPRECATED - see migration guide above):
    from app.coordination.replication_repair_daemon import (
        ReplicationRepairDaemon,
        get_replication_repair_daemon,
    )

    daemon = get_replication_repair_daemon()
    await daemon.start()

    # Get repair status
    status = daemon.get_status()
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Emit deprecation warning on import
warnings.warn(
    "replication_repair_daemon is deprecated. Use unified_replication_daemon instead:\n"
    "  from app.coordination.unified_replication_daemon import get_replication_daemon",
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)


class RepairPriority(Enum):
    """Priority levels for repair operations."""
    CRITICAL = 1  # Zero-copy games (data loss imminent)
    HIGH = 2      # Single-copy games (high risk)
    NORMAL = 3    # Under-replicated but safe


@dataclass
class RepairJob:
    """A pending repair job."""
    game_id: str
    priority: RepairPriority
    current_copies: int
    target_copies: int
    source_nodes: list[str]
    target_nodes: list[str]
    created_at: float
    started_at: float = 0.0
    completed_at: float = 0.0
    success: bool = False
    error: str = ""


@dataclass
class RepairStats:
    """Statistics about repair operations."""
    total_repairs_attempted: int = 0
    total_repairs_succeeded: int = 0
    total_repairs_failed: int = 0
    critical_repairs: int = 0
    high_priority_repairs: int = 0
    normal_priority_repairs: int = 0
    bytes_transferred: int = 0
    last_repair_time: float = 0.0
    avg_repair_duration_seconds: float = 0.0


@dataclass
class ReplicationRepairConfig:
    """Configuration for the repair daemon."""
    # Repair intervals
    check_interval_seconds: float = 60.0  # How often to check for repairs
    max_concurrent_repairs: int = 3  # Max simultaneous repair operations

    # Replication targets
    min_replicas: int = 2
    target_replicas: int = 3

    # Batch settings
    max_repairs_per_cycle: int = 50  # Max repairs to start per check cycle
    repair_timeout_seconds: float = 300.0  # Timeout for individual repair

    # Cooldown to prevent thrashing
    repair_cooldown_per_game_seconds: float = 600.0  # Don't retry same game within 10 min

    # Node selection
    prefer_local_sources: bool = True  # Prefer local node as source if available
    avoid_ephemeral_targets: bool = True  # Avoid vast.ai for new replicas

    # Rate limiting
    max_repairs_per_hour: int = 500
    bandwidth_limit_mbps: float = 50.0  # Per-repair bandwidth limit


class ReplicationRepairDaemon:
    """Daemon that actively repairs under-replicated data.

    Works with ReplicationMonitorDaemon and ClusterManifest to:
    1. Identify under-replicated games
    2. Select optimal source/target nodes
    3. Coordinate repair transfers
    4. Track and report repair progress
    """

    def __init__(self, config: ReplicationRepairConfig | None = None):
        """Initialize the repair daemon.

        Args:
            config: Repair configuration
        """
        self.config = config or ReplicationRepairConfig()
        self.node_id = socket.gethostname()

        self._running = False
        self._repair_task: asyncio.Task | None = None

        # Repair tracking
        self._pending_repairs: list[RepairJob] = []
        self._active_repairs: dict[str, RepairJob] = {}  # game_id -> job
        self._completed_repairs: list[RepairJob] = []
        self._repair_cooldowns: dict[str, float] = {}  # game_id -> last_attempt_time

        # Statistics
        self._stats = RepairStats()
        self._hourly_repair_count = 0
        self._hourly_reset_time = time.time()

        # Semaphore for concurrent repairs
        self._repair_semaphore = asyncio.Semaphore(self.config.max_concurrent_repairs)

    async def start(self) -> None:
        """Start the repair daemon."""
        if self._running:
            logger.warning("ReplicationRepairDaemon already running")
            return

        self._running = True
        self._repair_task = asyncio.create_task(
            self._repair_loop(),
            name="replication_repair"
        )
        logger.info(
            f"ReplicationRepairDaemon started (interval={self.config.check_interval_seconds}s, "
            f"max_concurrent={self.config.max_concurrent_repairs})"
        )

    async def stop(self) -> None:
        """Stop the repair daemon."""
        if not self._running:
            return

        self._running = False

        # Wait for active repairs to complete (with timeout)
        if self._active_repairs:
            logger.info(f"Waiting for {len(self._active_repairs)} active repairs to complete...")
            await asyncio.sleep(5.0)

        if self._repair_task:
            self._repair_task.cancel()
            try:
                await asyncio.wait_for(self._repair_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._repair_task = None

        logger.info("ReplicationRepairDaemon stopped")

    def is_running(self) -> bool:
        """Check if daemon is running."""
        return self._running

    async def _repair_loop(self) -> None:
        """Main repair loop."""
        while self._running:
            try:
                await self._run_repair_cycle()
                await asyncio.sleep(self.config.check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in repair cycle: {e}")
                await asyncio.sleep(60)  # Back off on error

    async def _run_repair_cycle(self) -> None:
        """Execute one repair cycle."""
        # Reset hourly counter if needed
        now = time.time()
        if now - self._hourly_reset_time >= 3600:
            self._hourly_repair_count = 0
            self._hourly_reset_time = now

        # Check rate limit
        if self._hourly_repair_count >= self.config.max_repairs_per_hour:
            logger.debug("Hourly repair limit reached, skipping cycle")
            return

        # Find games that need repair
        games_to_repair = await self._find_games_needing_repair()

        if not games_to_repair:
            return

        # Create repair jobs
        repair_jobs = await self._create_repair_jobs(games_to_repair)

        if not repair_jobs:
            return

        # Execute repairs (respecting concurrency limit)
        tasks = []
        for job in repair_jobs[:self.config.max_repairs_per_cycle]:
            if self._hourly_repair_count >= self.config.max_repairs_per_hour:
                break
            tasks.append(self._execute_repair(job))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _find_games_needing_repair(self) -> list[tuple[str, int, list[str]]]:
        """Find games that need additional replicas.

        Returns:
            List of (game_id, current_copies, source_nodes)
        """
        try:
            from app.distributed.cluster_manifest import get_cluster_manifest
        except ImportError:
            logger.debug("ClusterManifest not available")
            return []

        manifest = get_cluster_manifest()
        now = time.time()

        # Get under-replicated games
        under_replicated = manifest.get_under_replicated_games(
            min_copies=self.config.min_replicas,
            limit=self.config.max_repairs_per_cycle * 2,
        )

        # Filter by cooldown and add source nodes
        result = []
        for game_id, copies in under_replicated:
            # Check cooldown
            last_attempt = self._repair_cooldowns.get(game_id, 0)
            if now - last_attempt < self.config.repair_cooldown_per_game_seconds:
                continue

            # Skip if already being repaired
            if game_id in self._active_repairs:
                continue

            # Get nodes that have this game
            source_nodes = manifest.get_game_locations(game_id)
            if not source_nodes:
                # Zero-copy - can't repair (data lost)
                continue

            result.append((game_id, copies, source_nodes))

        # Sort by priority (fewer copies = higher priority)
        result.sort(key=lambda x: x[1])

        return result

    async def _create_repair_jobs(
        self,
        games: list[tuple[str, int, list[str]]],
    ) -> list[RepairJob]:
        """Create repair jobs for games needing repair.

        Args:
            games: List of (game_id, current_copies, source_nodes)

        Returns:
            List of RepairJob objects
        """
        try:
            from app.coordination.sync_router import get_sync_router
        except ImportError:
            logger.debug("SyncRouter not available")
            return []

        now = time.time()
        jobs = []

        router = get_sync_router()

        for game_id, copies, source_nodes in games:
            # Determine priority
            if copies == 0:
                priority = RepairPriority.CRITICAL
            elif copies == 1:
                priority = RepairPriority.HIGH
            else:
                priority = RepairPriority.NORMAL

            # Find target nodes (nodes that don't have this game)
            target_nodes = await self._select_target_nodes(
                game_id, source_nodes, copies
            )

            if not target_nodes:
                continue

            job = RepairJob(
                game_id=game_id,
                priority=priority,
                current_copies=copies,
                target_copies=self.config.target_replicas,
                source_nodes=source_nodes,
                target_nodes=target_nodes,
                created_at=now,
            )
            jobs.append(job)

        # Sort by priority
        jobs.sort(key=lambda j: j.priority.value)

        return jobs

    async def _select_target_nodes(
        self,
        game_id: str,
        source_nodes: list[str],
        current_copies: int,
    ) -> list[str]:
        """Select target nodes for replication.

        Args:
            game_id: Game to replicate
            source_nodes: Nodes that have the game
            current_copies: Current replica count

        Returns:
            List of target node hostnames
        """
        try:
            from app.coordination.sync_router import get_sync_router
        except ImportError:
            return []

        router = get_sync_router()

        # Get all available nodes
        all_nodes = router.get_available_hosts(for_receiving=True)

        # Exclude source nodes
        source_set = set(source_nodes)
        candidates = [n for n in all_nodes if n not in source_set]

        if not candidates:
            return []

        # Filter out ephemeral nodes if configured
        if self.config.avoid_ephemeral_targets:
            try:
                from app.coordination.sync_router import SyncRouter
                stable_candidates = [
                    n for n in candidates
                    if not router.is_ephemeral_host(n)
                ]
                if stable_candidates:
                    candidates = stable_candidates
            except ImportError:
                pass  # SyncRouter not available, use all candidates

        # Calculate how many more copies we need
        copies_needed = self.config.target_replicas - current_copies

        # Return up to that many target nodes
        return candidates[:copies_needed]

    async def _execute_repair(self, job: RepairJob) -> None:
        """Execute a single repair job.

        Args:
            job: The repair job to execute
        """
        async with self._repair_semaphore:
            job.started_at = time.time()
            self._active_repairs[job.game_id] = job
            self._repair_cooldowns[job.game_id] = job.started_at
            self._hourly_repair_count += 1

            try:
                success = await self._perform_repair_transfer(job)
                job.success = success

                if success:
                    self._stats.total_repairs_succeeded += 1
                    logger.info(
                        f"Repair completed: {job.game_id} "
                        f"({job.current_copies} -> {job.current_copies + len(job.target_nodes)} copies)"
                    )
                    # Emit REPAIR_COMPLETED event (December 2025)
                    await self._emit_repair_event(job, success=True)
                else:
                    self._stats.total_repairs_failed += 1
                    logger.warning(f"Repair failed: {job.game_id}: {job.error}")
                    # Emit REPAIR_FAILED event (December 2025)
                    await self._emit_repair_event(job, success=False)

            except Exception as e:
                job.success = False
                job.error = str(e)
                self._stats.total_repairs_failed += 1
                logger.error(f"Repair error for {job.game_id}: {e}")
                # Emit REPAIR_FAILED event (December 2025)
                await self._emit_repair_event(job, success=False)

            finally:
                job.completed_at = time.time()
                self._stats.total_repairs_attempted += 1

                # Update priority stats
                if job.priority == RepairPriority.CRITICAL:
                    self._stats.critical_repairs += 1
                elif job.priority == RepairPriority.HIGH:
                    self._stats.high_priority_repairs += 1
                else:
                    self._stats.normal_priority_repairs += 1

                # Update timing stats
                duration = job.completed_at - job.started_at
                total_duration = (
                    self._stats.avg_repair_duration_seconds *
                    (self._stats.total_repairs_attempted - 1) + duration
                )
                self._stats.avg_repair_duration_seconds = (
                    total_duration / self._stats.total_repairs_attempted
                )
                self._stats.last_repair_time = job.completed_at

                # Move to completed
                del self._active_repairs[job.game_id]
                self._completed_repairs.append(job)

                # Trim completed history
                if len(self._completed_repairs) > 1000:
                    self._completed_repairs = self._completed_repairs[-500:]

    async def _perform_repair_transfer(self, job: RepairJob) -> bool:
        """Perform the actual data transfer for repair.

        Args:
            job: The repair job

        Returns:
            True if successful
        """
        try:
            from app.coordination.sync_bandwidth import get_bandwidth_coordinator
        except ImportError:
            job.error = "sync_bandwidth not available"
            return False

        if not job.source_nodes or not job.target_nodes:
            job.error = "No source or target nodes"
            return False

        # Select best source (prefer local if configured)
        source = job.source_nodes[0]
        if self.config.prefer_local_sources and self.node_id in job.source_nodes:
            source = self.node_id

        coordinator = get_bandwidth_coordinator()
        success_count = 0

        # Transfer to each target
        for target in job.target_nodes:
            try:
                # Build the game database path
                # This is a simplified path - actual implementation would need
                # to query the manifest for the exact path
                game_path = f"data/games/game_{job.game_id}.db"

                result = await asyncio.wait_for(
                    coordinator.sync_file(
                        source_host=source,
                        dest_host=target,
                        source_path=game_path,
                        dest_path=game_path,
                        bandwidth_limit_mbps=self.config.bandwidth_limit_mbps,
                    ),
                    timeout=self.config.repair_timeout_seconds,
                )

                if result.get("success"):
                    success_count += 1
                    self._stats.bytes_transferred += result.get("bytes_transferred", 0)

                    # Update manifest
                    await self._update_manifest_after_repair(job.game_id, target)
                else:
                    logger.warning(
                        f"Transfer failed for {job.game_id} to {target}: {result.get('error')}"
                    )

            except asyncio.TimeoutError:
                logger.warning(f"Transfer timeout for {job.game_id} to {target}")
            except Exception as e:
                logger.warning(f"Transfer error for {job.game_id} to {target}: {e}")

        if success_count == 0:
            job.error = f"All {len(job.target_nodes)} transfers failed"
            return False

        if success_count < len(job.target_nodes):
            job.error = f"Partial success: {success_count}/{len(job.target_nodes)} transfers"

        return success_count > 0

    async def _update_manifest_after_repair(self, game_id: str, target_node: str) -> None:
        """Update the cluster manifest after successful repair.

        Args:
            game_id: The repaired game
            target_node: Node that now has a copy
        """
        try:
            from app.distributed.cluster_manifest import get_cluster_manifest

            manifest = get_cluster_manifest()
            manifest.register_game(
                game_id=game_id,
                host=target_node,
                path=f"data/games/game_{game_id}.db",
            )
        except Exception as e:
            logger.debug(f"Failed to update manifest after repair: {e}")

    async def trigger_repair(self, game_ids: list[str] | None = None) -> int:
        """Manually trigger repair for specific games or all under-replicated.

        Args:
            game_ids: Specific games to repair, or None for auto-detection

        Returns:
            Number of repairs queued
        """
        if game_ids:
            # Queue specific games for repair
            games_to_repair = []
            try:
                from app.distributed.cluster_manifest import get_cluster_manifest
                manifest = get_cluster_manifest()

                for game_id in game_ids:
                    locations = manifest.get_game_locations(game_id)
                    if len(locations) < self.config.target_replicas:
                        games_to_repair.append((game_id, len(locations), locations))
            except ImportError:
                return 0
        else:
            games_to_repair = await self._find_games_needing_repair()

        jobs = await self._create_repair_jobs(games_to_repair)

        # Execute immediately (up to limit)
        count = 0
        for job in jobs[:self.config.max_repairs_per_cycle]:
            asyncio.create_task(self._execute_repair(job))
            count += 1

        return count

    def get_status(self) -> dict[str, Any]:
        """Get current repair daemon status.

        Returns:
            Status dict with stats and active repairs
        """
        return {
            "node_id": self.node_id,
            "running": self._running,
            "stats": {
                "total_attempted": self._stats.total_repairs_attempted,
                "total_succeeded": self._stats.total_repairs_succeeded,
                "total_failed": self._stats.total_repairs_failed,
                "success_rate": (
                    self._stats.total_repairs_succeeded /
                    max(1, self._stats.total_repairs_attempted)
                ),
                "critical_repairs": self._stats.critical_repairs,
                "high_priority_repairs": self._stats.high_priority_repairs,
                "normal_priority_repairs": self._stats.normal_priority_repairs,
                "bytes_transferred": self._stats.bytes_transferred,
                "last_repair_time": self._stats.last_repair_time,
                "avg_duration_seconds": round(self._stats.avg_repair_duration_seconds, 2),
                "hourly_count": self._hourly_repair_count,
                "hourly_limit": self.config.max_repairs_per_hour,
            },
            "active_repairs": [
                {
                    "game_id": job.game_id,
                    "priority": job.priority.name,
                    "current_copies": job.current_copies,
                    "target_copies": job.target_copies,
                    "started_at": job.started_at,
                    "duration": time.time() - job.started_at,
                }
                for job in self._active_repairs.values()
            ],
            "config": {
                "check_interval_seconds": self.config.check_interval_seconds,
                "max_concurrent_repairs": self.config.max_concurrent_repairs,
                "min_replicas": self.config.min_replicas,
                "target_replicas": self.config.target_replicas,
                "max_repairs_per_hour": self.config.max_repairs_per_hour,
            },
        }

    async def _emit_repair_event(self, job, success: bool) -> None:
        """Emit repair event (December 2025: Event-driven coordination).

        Args:
            job: RepairJob with repair details
            success: Whether repair succeeded
        """
        try:
            from app.coordination.event_router import get_router, DataEventType

            router = get_router()
            if not router:
                return

            event_type = DataEventType.REPAIR_COMPLETED if success else DataEventType.REPAIR_FAILED
            await router.publish(
                event_type=event_type,
                payload={
                    "game_id": job.game_id,
                    "success": success,
                    "priority": job.priority.name,
                    "current_copies": job.current_copies,
                    "target_copies": job.target_copies,
                    "source_nodes": job.source_nodes,
                    "target_nodes": job.target_nodes,
                    "error": job.error,
                    "duration_seconds": (job.completed_at - job.started_at) if job.completed_at else 0,
                },
                source="replication_repair_daemon",
            )

        except ImportError:
            pass  # Event router not available
        except Exception as e:
            logger.debug(f"Failed to emit repair event: {e}")

    def health_check(self):
        """Check daemon health (December 2025: CoordinatorProtocol compliance).

        Returns:
            HealthCheckResult with status and details
        """
        from app.coordination.protocols import HealthCheckResult, CoordinatorStatus

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="ReplicationRepair daemon not running",
            )

        # Check success rate
        total = self._stats.total_repairs_attempted
        failed = self._stats.total_repairs_failed
        if total > 20 and failed / total > 0.5:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"ReplicationRepair low success rate: {failed}/{total} failed",
                details=self.get_status(),
            )

        # Check hourly limit
        if self._hourly_repair_count >= self.config.max_repairs_per_hour:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message=f"ReplicationRepair: hourly limit reached ({self._hourly_repair_count})",
                details=self.get_status(),
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"ReplicationRepair running ({self._stats.total_repairs_succeeded} succeeded)",
            details=self.get_status(),
        )

    def get_repair_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent repair history.

        Args:
            limit: Maximum repairs to return

        Returns:
            List of repair job dicts, newest first
        """
        repairs = self._completed_repairs[-limit:]
        return [
            {
                "game_id": job.game_id,
                "priority": job.priority.name,
                "current_copies": job.current_copies,
                "target_copies": job.target_copies,
                "success": job.success,
                "error": job.error,
                "duration": job.completed_at - job.started_at,
                "completed_at": job.completed_at,
            }
            for job in reversed(repairs)
        ]


# Module-level singleton
_repair_daemon: ReplicationRepairDaemon | None = None


def get_replication_repair_daemon(
    config: ReplicationRepairConfig | None = None,
) -> ReplicationRepairDaemon:
    """Get the singleton ReplicationRepairDaemon instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        ReplicationRepairDaemon instance
    """
    global _repair_daemon
    if _repair_daemon is None:
        _repair_daemon = ReplicationRepairDaemon(config)
    return _repair_daemon


def reset_replication_repair_daemon() -> None:
    """Reset the singleton (for testing)."""
    global _repair_daemon
    _repair_daemon = None


__all__ = [
    "ReplicationRepairDaemon",
    "ReplicationRepairConfig",
    "RepairJob",
    "RepairPriority",
    "RepairStats",
    "get_replication_repair_daemon",
    "reset_replication_repair_daemon",
]

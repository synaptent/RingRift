"""Unified Data Sync Orchestrator - Central coordinator for data synchronization.

January 2026: Created as part of unified data synchronization plan.
Coordinates backup operations to S3 and OWC based on sync events.

Key responsibilities:
1. Listen to DATA_SYNC_COMPLETED events and trigger appropriate backups
2. Track backup status across all storage destinations
3. Verify replication completeness before cleanup
4. Provide unified visibility into data distribution
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from app.coordination.handler_base import HandlerBase
from app.distributed.cluster_manifest import (
    ClusterManifest,
    DataSource,
    get_cluster_manifest,
)

logger = logging.getLogger(__name__)


class ReplicationStatus(str, Enum):
    """Status of data replication."""

    COMPLETE = "complete"  # Has required number of copies
    PARTIAL = "partial"  # Has some copies but not enough
    PENDING = "pending"  # Backup in progress
    MISSING = "missing"  # No backup copies
    FAILED = "failed"  # Backup failed


@dataclass
class BackupStatus:
    """Status of a backup operation."""

    config_key: str
    db_path: str
    s3_backed_up: bool = False
    owc_backed_up: bool = False
    s3_key: str | None = None
    owc_path: str | None = None
    last_s3_backup: float = 0.0
    last_owc_backup: float = 0.0
    pending_s3: bool = False
    pending_owc: bool = False


@dataclass
class DataVisibilityReport:
    """Report of data visibility across all sources."""

    timestamp: float
    configs: dict[str, dict[str, Any]]
    total_games_local: int = 0
    total_games_p2p: int = 0
    total_games_s3: int = 0
    total_games_owc: int = 0
    under_replicated_configs: list[str] = field(default_factory=list)
    fully_replicated_configs: list[str] = field(default_factory=list)


@dataclass
class OrchestratorConfig:
    """Configuration for UnifiedDataSyncOrchestrator."""

    # Replication requirements
    min_replicas_before_cleanup: int = 2

    # Backup settings
    s3_enabled: bool = True
    owc_enabled: bool = True
    s3_bucket: str = "ringrift-models-20251214"
    owc_host: str = "mac-studio"
    owc_base_path: str = "/Volumes/RingRift-Data"

    # Timeouts
    backup_timeout: float = 600.0  # 10 minutes

    # Cycle interval
    cycle_interval: float = 300.0  # 5 minutes

    # Health check
    max_backup_age_hours: float = 24.0


class UnifiedDataSyncOrchestrator(HandlerBase):
    """Central orchestrator for all data synchronization.

    This daemon coordinates backup operations to S3 and OWC:
    1. Listens to DATA_SYNC_COMPLETED events from AutoSyncDaemon
    2. Triggers backup to S3 and/or OWC based on flags
    3. Tracks backup status for each database
    4. Verifies replication before allowing cleanup
    """

    def __init__(self, config: OrchestratorConfig | None = None):
        super().__init__(
            name="unified_data_sync_orchestrator",
            cycle_interval=config.cycle_interval if config else 300.0,
        )
        self.config = config or OrchestratorConfig()
        self._manifest = get_cluster_manifest()
        self._backup_status: dict[str, BackupStatus] = {}
        self._pending_backups: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._metrics = OrchestratorMetrics()

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Return event subscriptions."""
        return {
            "DATA_SYNC_COMPLETED": self._on_data_sync_completed,
            "BACKUP_COMPLETED": self._on_backup_completed,
        }

    async def _run_cycle(self) -> None:
        """Main cycle - process pending backups and update metrics."""
        # Process any pending backups
        await self._process_pending_backups()

        # Update metrics
        await self._update_metrics()

        # Check for under-replicated data
        await self._check_replication_status()

    async def _on_data_sync_completed(self, event: Any) -> None:
        """Handle DATA_SYNC_COMPLETED event - trigger backups based on flags.

        AutoSyncDaemon emits this after syncing data from GPU nodes.
        The event includes flags indicating which backups are needed.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event

            needs_owc_backup = payload.get("needs_owc_backup", False)
            needs_s3_backup = payload.get("needs_s3_backup", False)

            if not needs_owc_backup and not needs_s3_backup:
                return

            # Extract sync details
            db_path = payload.get("db_path", "")
            config_key = payload.get("config_key", "")
            game_count = payload.get("game_count", 0)

            if not db_path or not config_key:
                logger.debug(
                    "[UnifiedDataSyncOrchestrator] Skipping backup - missing db_path or config_key"
                )
                return

            # Queue backup tasks
            backup_request = {
                "db_path": db_path,
                "config_key": config_key,
                "game_count": game_count,
                "needs_s3": needs_s3_backup and self.config.s3_enabled,
                "needs_owc": needs_owc_backup and self.config.owc_enabled,
                "queued_at": time.time(),
            }

            await self._pending_backups.put(backup_request)
            logger.info(
                f"[UnifiedDataSyncOrchestrator] Queued backup for {config_key}: "
                f"S3={needs_s3_backup}, OWC={needs_owc_backup}"
            )

        except Exception as e:
            logger.warning(
                f"[UnifiedDataSyncOrchestrator] Error handling DATA_SYNC_COMPLETED: {e}"
            )

    async def _on_backup_completed(self, event: Any) -> None:
        """Handle BACKUP_COMPLETED event - update status tracking."""
        try:
            payload = event.payload if hasattr(event, "payload") else event

            if not payload.get("success", False):
                return

            backup_details = payload.get("backup_details", [])
            for detail in backup_details:
                config_key = detail.get("config_key", "")
                db_path = detail.get("db_path", "")

                if not config_key:
                    continue

                # Update or create backup status
                status_key = f"{config_key}:{db_path}"
                if status_key not in self._backup_status:
                    self._backup_status[status_key] = BackupStatus(
                        config_key=config_key, db_path=db_path
                    )

                status = self._backup_status[status_key]

                # Update S3 status
                if detail.get("s3_key"):
                    status.s3_backed_up = True
                    status.s3_key = detail["s3_key"]
                    status.last_s3_backup = time.time()
                    status.pending_s3 = False
                    self._metrics.s3_backups_succeeded += 1

                # Update OWC status
                if detail.get("owc_path"):
                    status.owc_backed_up = True
                    status.owc_path = detail["owc_path"]
                    status.last_owc_backup = time.time()
                    status.pending_owc = False
                    self._metrics.owc_backups_succeeded += 1

        except Exception as e:
            logger.warning(
                f"[UnifiedDataSyncOrchestrator] Error handling BACKUP_COMPLETED: {e}"
            )

    async def _process_pending_backups(self) -> None:
        """Process queued backup requests."""
        processed = 0
        max_per_cycle = 10

        while not self._pending_backups.empty() and processed < max_per_cycle:
            try:
                request = self._pending_backups.get_nowait()
                await self._execute_backup(request)
                processed += 1
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                logger.warning(
                    f"[UnifiedDataSyncOrchestrator] Error processing backup: {e}"
                )

    async def _execute_backup(self, request: dict[str, Any]) -> None:
        """Execute a backup request by triggering UnifiedBackupDaemon."""
        config_key = request["config_key"]
        db_path = request["db_path"]

        # Update tracking
        status_key = f"{config_key}:{db_path}"
        if status_key not in self._backup_status:
            self._backup_status[status_key] = BackupStatus(
                config_key=config_key, db_path=db_path
            )

        status = self._backup_status[status_key]

        if request.get("needs_s3"):
            status.pending_s3 = True
            await self._trigger_s3_backup(request)

        if request.get("needs_owc"):
            status.pending_owc = True
            await self._trigger_owc_backup(request)

    async def _trigger_s3_backup(self, request: dict[str, Any]) -> None:
        """Trigger S3 backup for a database."""
        try:
            from app.coordination.event_router import get_event_router

            router = get_event_router()
            await router.publish(
                event_type="BACKUP_REQUESTED",
                payload={
                    "db_path": request["db_path"],
                    "config_key": request["config_key"],
                    "destination": "s3",
                    "s3_bucket": self.config.s3_bucket,
                },
            )
            logger.debug(
                f"[UnifiedDataSyncOrchestrator] Triggered S3 backup for {request['config_key']}"
            )
        except Exception as e:
            logger.warning(f"[UnifiedDataSyncOrchestrator] Failed to trigger S3 backup: {e}")
            self._metrics.s3_backups_failed += 1

    async def _trigger_owc_backup(self, request: dict[str, Any]) -> None:
        """Trigger OWC backup for a database."""
        try:
            from app.coordination.event_router import get_event_router

            router = get_event_router()
            await router.publish(
                event_type="BACKUP_REQUESTED",
                payload={
                    "db_path": request["db_path"],
                    "config_key": request["config_key"],
                    "destination": "owc",
                    "owc_host": self.config.owc_host,
                    "owc_base_path": self.config.owc_base_path,
                },
            )
            logger.debug(
                f"[UnifiedDataSyncOrchestrator] Triggered OWC backup for {request['config_key']}"
            )
        except Exception as e:
            logger.warning(f"[UnifiedDataSyncOrchestrator] Failed to trigger OWC backup: {e}")
            self._metrics.owc_backups_failed += 1

    async def _check_replication_status(self) -> None:
        """Check for under-replicated data and emit alerts."""
        under_replicated = []

        for status_key, status in self._backup_status.items():
            copies = 0
            if status.s3_backed_up:
                copies += 1
            if status.owc_backed_up:
                copies += 1

            if copies < self.config.min_replicas_before_cleanup:
                under_replicated.append(status.config_key)

        self._metrics.under_replicated_count = len(under_replicated)

        if under_replicated:
            logger.warning(
                f"[UnifiedDataSyncOrchestrator] {len(under_replicated)} configs under-replicated"
            )

    async def _update_metrics(self) -> None:
        """Update internal metrics."""
        self._metrics.last_update = time.time()
        self._metrics.pending_backups = self._pending_backups.qsize()

    async def verify_replication_complete(
        self,
        file_path: str,
        min_copies: int | None = None,
    ) -> ReplicationStatus:
        """Verify file has required replicas before cleanup.

        Args:
            file_path: Path to the file to check
            min_copies: Minimum required copies (default: config value)

        Returns:
            ReplicationStatus indicating whether safe to cleanup
        """
        min_required = min_copies or self.config.min_replicas_before_cleanup

        # Check all backup statuses for this file
        copies = 0
        for status_key, status in self._backup_status.items():
            if status.db_path == file_path:
                if status.s3_backed_up:
                    copies += 1
                if status.owc_backed_up:
                    copies += 1
                if status.pending_s3 or status.pending_owc:
                    return ReplicationStatus.PENDING

        if copies >= min_required:
            return ReplicationStatus.COMPLETE
        elif copies > 0:
            return ReplicationStatus.PARTIAL
        else:
            return ReplicationStatus.MISSING

    async def get_data_visibility_report(self) -> DataVisibilityReport:
        """Get unified view of data across all sources.

        Returns comprehensive report of data distribution and replication status.
        """
        configs: dict[str, dict[str, Any]] = {}
        total_local = 0
        total_p2p = 0
        total_s3 = 0
        total_owc = 0
        under_replicated = []
        fully_replicated = []

        # Get all known configs
        known_configs = set()
        for status in self._backup_status.values():
            known_configs.add(status.config_key)

        for config_key in known_configs:
            sources = self._manifest.find_across_all_sources(config_key)

            local_games = sum(
                loc.get("game_count", 0) for loc in sources.get(DataSource.LOCAL, [])
            )
            p2p_games = sum(
                loc.get("game_count", 0) for loc in sources.get(DataSource.P2P, [])
            )
            s3_games = sum(
                loc.get("game_count", 0) for loc in sources.get(DataSource.S3, [])
            )
            owc_games = sum(
                loc.get("game_count", 0) for loc in sources.get(DataSource.OWC, [])
            )

            total_local += local_games
            total_p2p += p2p_games
            total_s3 += s3_games
            total_owc += owc_games

            # Check replication status
            backup_copies = 0
            if s3_games > 0:
                backup_copies += 1
            if owc_games > 0:
                backup_copies += 1

            if backup_copies >= self.config.min_replicas_before_cleanup:
                fully_replicated.append(config_key)
            else:
                under_replicated.append(config_key)

            configs[config_key] = {
                "local_games": local_games,
                "p2p_games": p2p_games,
                "s3_games": s3_games,
                "owc_games": owc_games,
                "backup_copies": backup_copies,
                "fully_replicated": backup_copies >= self.config.min_replicas_before_cleanup,
            }

        return DataVisibilityReport(
            timestamp=time.time(),
            configs=configs,
            total_games_local=total_local,
            total_games_p2p=total_p2p,
            total_games_s3=total_s3,
            total_games_owc=total_owc,
            under_replicated_configs=under_replicated,
            fully_replicated_configs=fully_replicated,
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get orchestrator metrics for monitoring."""
        return {
            "pending_backups": self._metrics.pending_backups,
            "s3_backups_succeeded": self._metrics.s3_backups_succeeded,
            "s3_backups_failed": self._metrics.s3_backups_failed,
            "owc_backups_succeeded": self._metrics.owc_backups_succeeded,
            "owc_backups_failed": self._metrics.owc_backups_failed,
            "under_replicated_count": self._metrics.under_replicated_count,
            "last_update": self._metrics.last_update,
        }

    def health_check(self) -> "HealthCheckResult":
        """Return health check result.

        Sprint 15.4: Fixed return type annotation and status enum usage.
        """
        from app.coordination.contracts import HealthCheckResult, CoordinatorStatus

        # Check backup health
        s3_healthy = self._metrics.s3_backups_failed < 5
        owc_healthy = self._metrics.owc_backups_failed < 5
        overall_healthy = s3_healthy and owc_healthy

        return HealthCheckResult(
            healthy=overall_healthy,
            status=CoordinatorStatus.RUNNING if overall_healthy else CoordinatorStatus.PAUSED,
            details={
                "s3_healthy": s3_healthy,
                "owc_healthy": owc_healthy,
                "pending_backups": self._metrics.pending_backups,
                "under_replicated": self._metrics.under_replicated_count,
            },
        )


@dataclass
class OrchestratorMetrics:
    """Metrics for UnifiedDataSyncOrchestrator."""

    pending_backups: int = 0
    s3_backups_succeeded: int = 0
    s3_backups_failed: int = 0
    owc_backups_succeeded: int = 0
    owc_backups_failed: int = 0
    under_replicated_count: int = 0
    last_update: float = 0.0


# Singleton instance
_orchestrator_instance: UnifiedDataSyncOrchestrator | None = None


def get_unified_data_sync_orchestrator() -> UnifiedDataSyncOrchestrator:
    """Get the singleton UnifiedDataSyncOrchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = UnifiedDataSyncOrchestrator()
    return _orchestrator_instance


def reset_unified_data_sync_orchestrator() -> None:
    """Reset the singleton instance (for testing)."""
    global _orchestrator_instance
    _orchestrator_instance = None


async def create_unified_data_sync_orchestrator_daemon() -> UnifiedDataSyncOrchestrator:
    """Factory function for daemon_runners.py."""
    return get_unified_data_sync_orchestrator()

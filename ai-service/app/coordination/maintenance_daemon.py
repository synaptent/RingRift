"""Maintenance Daemon - Automated system maintenance and cleanup.

December 2025 (Phase 3.3): Addresses long-term sustainability by automating:
1. Log rotation (100MB, 10 backups)
2. Database maintenance (VACUUM)
3. Stale data archival
4. Dead letter queue pruning

Scheduled Tasks:
    Hourly:  Log rotation check
    Daily:   Archive games older than 30 days
    Weekly:  VACUUM all databases
    Weekly:  Prune dead letter queue

Usage:
    from app.coordination.maintenance_daemon import MaintenanceDaemon

    daemon = MaintenanceDaemon()
    await daemon.start()

Integration with DaemonManager:
    DaemonType.MAINTENANCE factory creates and manages this daemon.
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import logging
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "MaintenanceConfig",
    "MaintenanceDaemon",
    "MaintenanceStats",
    "get_maintenance_daemon",
]

# December 2025: Use consolidated daemon stats base class
from app.coordination.daemon_stats import CleanupDaemonStats
from app.coordination.event_utils import parse_config_key
from app.config.thresholds import SQLITE_CONNECT_TIMEOUT
from app.config.coordination_defaults import WorkQueueCleanupDefaults

# January 2026: Import HandlerBase for standardized daemon patterns
from app.coordination.handler_base import HandlerBase


@dataclass
class MaintenanceConfig:
    """Configuration for maintenance daemon."""
    enabled: bool = True

    # Log rotation
    log_max_size_mb: float = 100.0
    log_backup_count: int = 10
    log_rotation_interval_hours: float = 1.0

    # Database maintenance
    db_vacuum_interval_hours: float = 168.0  # Weekly
    db_maintenance_enabled: bool = True
    # February 2026: Skip VACUUM on databases larger than this threshold (bytes).
    # VACUUM requires ~2x the DB size in memory+disk. On a 128 GB coordinator with
    # 11 GB databases, this causes OOM kernel panics.
    db_vacuum_max_size_bytes: int = 500 * 1024 * 1024  # 500 MB

    # Game archival
    archive_games_older_than_days: int = 30
    archive_interval_hours: float = 24.0  # Daily
    archive_enabled: bool = True
    archive_directory: str = field(
        default_factory=lambda: os.getenv(
            "RINGRIFT_ARCHIVE_DIR", str(Path("data/games/archive"))
        )
    )
    archive_compress: bool = True  # Use gzip compression for archived databases
    archive_to_s3: bool = field(
        default_factory=lambda: os.getenv("RINGRIFT_ARCHIVE_TO_S3", "false").lower() == "true"
    )
    archive_s3_bucket: str = field(
        default_factory=lambda: os.getenv("RINGRIFT_ARCHIVE_S3_BUCKET", "ringrift-archive")
    )

    # DLQ cleanup (also handled by DLQ_RETRY daemon, but backup cleanup here)
    dlq_cleanup_interval_hours: float = 168.0  # Weekly
    dlq_retention_days: int = 7

    # Work queue cleanup (Jan 2026: centralized in WorkQueueCleanupDefaults)
    queue_cleanup_interval_hours: float = 1.0  # Hourly
    queue_stale_pending_hours: float = WorkQueueCleanupDefaults.MAX_PENDING_AGE_HOURS
    queue_stale_claimed_hours: float = WorkQueueCleanupDefaults.MAX_CLAIMED_AGE_HOURS
    queue_cleanup_enabled: bool = True

    # Orphan file detection (December 2025)
    orphan_detection_interval_hours: float = 24.0  # Daily
    orphan_detection_enabled: bool = True
    orphan_auto_cleanup: bool = False  # If True, delete orphans; if False, just report
    orphan_auto_recovery: bool = True  # December 2025: If True, re-register orphan DBs to manifest

    # General
    dry_run: bool = False  # If True, log actions but don't execute


@dataclass
class MaintenanceStats(CleanupDaemonStats):
    """Statistics from maintenance operations.

    December 2025: Now extends CleanupDaemonStats for consistent tracking.
    Inherits: items_scanned, items_cleaned, items_quarantined, bytes_reclaimed,
              record_cleanup(), is_healthy(), to_dict(), etc.
    """

    # Maintenance-specific fields (not in base class)
    logs_rotated: int = 0
    bytes_reclaimed_logs: int = 0
    databases_vacuumed: int = 0
    games_archived: int = 0
    dlq_entries_cleaned: int = 0
    queue_items_cleaned: int = 0  # December 2025
    queue_items_reset: int = 0  # December 2025
    orphan_dbs_found: int = 0  # December 2025
    orphan_npz_found: int = 0  # December 2025
    orphan_models_found: int = 0  # December 2025
    orphan_dbs_recovered: int = 0  # December 2025: Re-registered to manifest
    elo_integrity_issues: int = 0  # Feb 2026: Winner_id mismatches found
    last_log_rotation: float = 0.0
    last_db_vacuum: float = 0.0
    last_archive_run: float = 0.0
    last_dlq_cleanup: float = 0.0
    last_queue_cleanup: float = 0.0  # December 2025
    last_orphan_detection: float = 0.0  # December 2025
    last_elo_integrity_check: float = 0.0  # Feb 2026

    def record_log_rotation(self, logs: int, bytes_reclaimed: int) -> None:
        """Record a log rotation operation."""
        self.logs_rotated += logs
        self.bytes_reclaimed_logs += bytes_reclaimed
        self.bytes_reclaimed += bytes_reclaimed
        self.last_log_rotation = time.time()

    def record_vacuum(self, databases: int) -> None:
        """Record database vacuum operation."""
        self.databases_vacuumed += databases
        self.last_db_vacuum = time.time()

    def record_archive(self, games: int) -> None:
        """Record game archival operation."""
        self.games_archived += games
        self.last_archive_run = time.time()


class MaintenanceDaemon(HandlerBase):
    """Daemon for automated system maintenance.

    Runs scheduled maintenance tasks:
    - Log rotation (hourly)
    - Database VACUUM (weekly)
    - Game archival (daily)
    - DLQ cleanup (weekly)

    January 2026: Migrated to HandlerBase for standardized patterns:
    - Singleton via get_instance() (backward-compat get_maintenance_daemon() still works)
    - Event subscription via _get_event_subscriptions()
    - Cycle management via _run_cycle()
    - Health check via health_check()
    """

    # Event source for safe event emission
    _event_source = "MaintenanceDaemon"

    def __init__(self, config: MaintenanceConfig | None = None):
        # Initialize HandlerBase with 10-minute cycle interval
        super().__init__(name="maintenance_daemon", config=config, cycle_interval=600.0)
        self.config = config or MaintenanceConfig()
        self._maintenance_stats = MaintenanceStats()
        self._ai_service_dir = Path(__file__).parent.parent.parent

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Return event subscriptions for HandlerBase.

        January 2026: Migrated from _subscribe_to_disk_events().
        """
        return {
            "DISK_SPACE_LOW": self._on_disk_space_low,
        }

    async def _run_cycle(self) -> None:
        """Main work loop iteration - required by HandlerBase.

        January 2026: Delegates to existing _run_maintenance_cycle().
        """
        await self._run_maintenance_cycle()

    async def _on_start(self) -> None:
        """Called before main loop starts - HandlerBase hook.

        January 2026: Initialize maintenance stats timing.
        """
        now = time.time()
        self._maintenance_stats.last_log_rotation = now
        self._maintenance_stats.last_db_vacuum = now
        self._maintenance_stats.last_archive_run = now
        self._maintenance_stats.last_dlq_cleanup = now
        self._maintenance_stats.last_queue_cleanup = now
        self._maintenance_stats.last_orphan_detection = now
        logger.info("[Maintenance] Daemon initialized timing stats")

    async def start(self) -> None:
        """Start the maintenance daemon.

        January 2026: Now calls HandlerBase.start() for standardized lifecycle.
        """
        if self._running:
            return

        # Call parent start which handles subscription and loop
        await super().start()
        logger.info("[Maintenance] Daemon started via HandlerBase")

    async def _subscribe_to_disk_events(self) -> None:
        """Subscribe to disk space events for reactive cleanup.

        December 2025: Wire DISK_SPACE_LOW event to trigger cleanup.
        Previously this event was emitted but not subscribed, meaning
        disk space warnings never triggered cleanup.
        """
        try:
            from app.coordination.event_router import DataEventType, get_router

            router = get_router()
            if router is None:
                logger.debug("[Maintenance] Event router not available")
                return

            # Subscribe to DISK_SPACE_LOW to trigger cleanup
            router.subscribe(
                DataEventType.DISK_SPACE_LOW.value,
                self._on_disk_space_low,
            )

            logger.info("[Maintenance] Subscribed to DISK_SPACE_LOW event")

        except ImportError as e:
            logger.warning(f"[Maintenance] Event router not available: {e}")
        except (RuntimeError, AttributeError) as e:
            logger.debug(f"[Maintenance] Failed to subscribe to disk events: {e}")

    async def _on_disk_space_low(self, event) -> None:
        """Handle DISK_SPACE_LOW event - trigger cleanup.

        December 2025: React to disk space warnings by running cleanup.
        This ensures maintenance runs proactively when disk space is low,
        rather than waiting for scheduled cleanup.

        Args:
            event: Event with payload containing host, usage_percent, free_gb
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            host = payload.get("host", "unknown")
            usage_percent = payload.get("usage_percent", 0)
            free_gb = payload.get("free_gb", 0)
            threshold = payload.get("threshold", 70)

            logger.warning(
                f"[Maintenance] DISK_SPACE_LOW on {host}: "
                f"{usage_percent:.1f}% used (threshold: {threshold}%), "
                f"{free_gb:.1f}GB free - triggering cleanup"
            )

            # Only respond to events for this host
            import socket
            local_hostname = socket.gethostname()
            if host not in (local_hostname, "localhost", "127.0.0.1"):
                logger.debug(f"[Maintenance] Ignoring disk event for other host: {host}")
                return

            # Run immediate cleanup cycle
            await self._rotate_logs()
            await self._cleanup_dlq()

            # If very low space, also vacuum databases to reclaim space
            if usage_percent >= 80:
                logger.warning(
                    f"[Maintenance] Critical disk usage ({usage_percent:.1f}%), "
                    f"running database vacuum"
                )
                await self._vacuum_databases()

            logger.info(
                f"[Maintenance] Cleanup completed for disk space event. "
                f"Stats: logs_rotated={self._maintenance_stats.logs_rotated}, "
                f"bytes_reclaimed={self._maintenance_stats.bytes_reclaimed}"
            )

        except (RuntimeError, OSError, AttributeError) as e:
            logger.error(f"[Maintenance] Error handling DISK_SPACE_LOW: {e}")

    async def stop(self) -> None:
        """Stop the daemon."""
        self._running = False
        logger.info("[Maintenance] Daemon stopped")

    async def run_forever(self) -> None:
        """Run maintenance loop until stopped."""
        await self.start()

        while self._running:
            try:
                await self._run_maintenance_cycle()
            except (RuntimeError, OSError, ConnectionError) as e:
                logger.error(f"[Maintenance] Cycle error: {e}")

            # Check every 10 minutes
            await asyncio.sleep(600)

    async def _run_maintenance_cycle(self) -> None:
        """Run a maintenance check cycle."""
        if not self.config.enabled:
            return

        # February 2026: Block heavy maintenance when coordinator is low on RAM/disk
        from app.utils.resource_guard import coordinator_resource_gate
        if not coordinator_resource_gate("MAINTENANCE"):
            return

        now = time.time()
        cycle_start = now
        tasks_run = []

        # Hourly: Log rotation
        hours_since_log_rotation = (now - self._maintenance_stats.last_log_rotation) / 3600
        if hours_since_log_rotation >= self.config.log_rotation_interval_hours:
            await self._rotate_logs()
            self._maintenance_stats.last_log_rotation = now
            tasks_run.append("log_rotation")

        # Daily: Archive old games
        hours_since_archive = (now - self._maintenance_stats.last_archive_run) / 3600
        if hours_since_archive >= self.config.archive_interval_hours:
            if self.config.archive_enabled:
                await self._archive_old_games()
                tasks_run.append("archive")
            self._maintenance_stats.last_archive_run = now

        # Weekly: VACUUM databases
        hours_since_vacuum = (now - self._maintenance_stats.last_db_vacuum) / 3600
        if hours_since_vacuum >= self.config.db_vacuum_interval_hours:
            if self.config.db_maintenance_enabled:
                await self._vacuum_databases()
                tasks_run.append("vacuum")
            self._maintenance_stats.last_db_vacuum = now

        # Weekly: DLQ cleanup
        hours_since_dlq = (now - self._maintenance_stats.last_dlq_cleanup) / 3600
        if hours_since_dlq >= self.config.dlq_cleanup_interval_hours:
            await self._cleanup_dlq()
            self._maintenance_stats.last_dlq_cleanup = now
            tasks_run.append("dlq_cleanup")

        # Hourly: Work queue stale item cleanup (December 2025)
        hours_since_queue = (now - self._maintenance_stats.last_queue_cleanup) / 3600
        if hours_since_queue >= self.config.queue_cleanup_interval_hours:
            if self.config.queue_cleanup_enabled:
                await self._cleanup_stale_queue_items()
                tasks_run.append("queue_cleanup")
            self._maintenance_stats.last_queue_cleanup = now

        # Daily: Orphan file detection (December 2025)
        hours_since_orphan = (now - self._maintenance_stats.last_orphan_detection) / 3600
        if hours_since_orphan >= self.config.orphan_detection_interval_hours:
            if self.config.orphan_detection_enabled:
                await self._detect_orphan_files()
                tasks_run.append("orphan_detection")
            self._maintenance_stats.last_orphan_detection = now

        # Hourly: Elo database integrity check (Feb 2026)
        # Detects winner_id mismatches caused by ID reconciliation bugs
        hours_since_elo_check = (now - self._maintenance_stats.last_elo_integrity_check) / 3600
        if hours_since_elo_check >= 1.0:
            await self._check_elo_integrity()
            self._maintenance_stats.last_elo_integrity_check = now
            tasks_run.append("elo_integrity")

        # December 2025: Emit event if any maintenance tasks ran
        if tasks_run:
            await self._emit_cleanup_event(tasks_run, cycle_start)

    async def _emit_cleanup_event(self, tasks_run: list[str], cycle_start: float) -> None:
        """Emit DISK_CLEANUP_TRIGGERED event after maintenance cycle.

        December 2025: Added to enable downstream coordination with sync daemons
        and disk space managers. The event signals that cleanup completed and
        disk space may have been freed.

        Args:
            tasks_run: List of maintenance tasks that were executed
            cycle_start: Timestamp when the maintenance cycle started
        """
        try:
            from app.coordination.event_router import publish

            duration = time.time() - cycle_start
            await publish(
                event_type="DISK_CLEANUP_TRIGGERED",
                payload={
                    "tasks": tasks_run,
                    "duration_seconds": duration,
                    "bytes_reclaimed": self._maintenance_stats.bytes_reclaimed,
                    "logs_rotated": self._maintenance_stats.logs_rotated,
                    "databases_vacuumed": self._maintenance_stats.databases_vacuumed,
                    "games_archived": self._maintenance_stats.games_archived,
                    "dlq_entries_cleaned": self._maintenance_stats.dlq_entries_cleaned,
                    "queue_items_cleaned": self._maintenance_stats.queue_items_cleaned,
                    "orphan_dbs_recovered": self._maintenance_stats.orphan_dbs_recovered,
                },
                source="maintenance_daemon",
            )
            logger.debug(
                f"[Maintenance] Emitted DISK_CLEANUP_TRIGGERED: {tasks_run}, "
                f"duration={duration:.1f}s"
            )
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.debug(f"[Maintenance] Failed to emit cleanup event: {e}")

    async def _rotate_logs(self) -> None:
        """Rotate log files that exceed max size."""
        logs_dir = self._ai_service_dir / "logs"
        if not logs_dir.exists():
            return

        max_size_bytes = int(self.config.log_max_size_mb * 1024 * 1024)
        rotated = 0
        bytes_saved = 0

        try:
            for log_file in logs_dir.glob("*.log"):
                if not log_file.is_file():
                    continue

                file_size = log_file.stat().st_size
                if file_size <= max_size_bytes:
                    continue

                if self.config.dry_run:
                    logger.info(f"[Maintenance] DRY RUN: Would rotate {log_file.name} ({file_size / 1024 / 1024:.1f} MB)")
                    continue

                # Rotate: compress current log, start fresh
                try:
                    # Shift existing backups
                    for i in range(self.config.log_backup_count - 1, 0, -1):
                        old_backup = log_file.with_suffix(f".log.{i}.gz")
                        new_backup = log_file.with_suffix(f".log.{i + 1}.gz")
                        if old_backup.exists():
                            if i + 1 >= self.config.log_backup_count:
                                old_backup.unlink()  # Delete oldest
                            else:
                                shutil.move(str(old_backup), str(new_backup))

                    # Compress current log to .1.gz (non-blocking for large log files)
                    # Dec 29, 2025: Use asyncio.to_thread to avoid event loop stalls
                    backup_path = log_file.with_suffix(".log.1.gz")

                    def _compress_and_truncate():
                        with open(log_file, "rb") as f_in, gzip.open(backup_path, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                        # Truncate original log
                        with open(log_file, "w") as f:
                            f.write(f"# Log rotated at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

                    await asyncio.to_thread(_compress_and_truncate)

                    rotated += 1
                    bytes_saved += file_size
                    logger.info(f"[Maintenance] Rotated {log_file.name} ({file_size / 1024 / 1024:.1f} MB)")

                except (OSError, RuntimeError) as e:
                    logger.warning(f"[Maintenance] Failed to rotate {log_file}: {e}")

        except (OSError, RuntimeError) as e:
            logger.error(f"[Maintenance] Log rotation error: {e}")

        if rotated:
            self._maintenance_stats.logs_rotated += rotated
            self._maintenance_stats.bytes_reclaimed_logs += bytes_saved
            logger.info(f"[Maintenance] Rotated {rotated} logs, reclaimed {bytes_saved / 1024 / 1024:.1f} MB")

    async def _vacuum_databases(self) -> None:
        """VACUUM all SQLite databases."""
        data_dir = self._ai_service_dir / "data"
        games_dir = data_dir / "games"

        db_paths = list(games_dir.glob("*.db")) if games_dir.exists() else []
        # Also check for manifest database
        manifest_db = data_dir / "data_manifest.db"
        if manifest_db.exists():
            db_paths.append(manifest_db)

        vacuumed = 0
        skipped_large = 0
        for db_path in db_paths:
            if self.config.dry_run:
                logger.info(f"[Maintenance] DRY RUN: Would VACUUM {db_path.name}")
                continue

            try:
                # Get size before
                size_before = db_path.stat().st_size

                # February 2026: Skip VACUUM on large databases to prevent OOM.
                # VACUUM requires ~2x DB size in memory+disk I/O. An 11 GB VACUUM
                # caused 34 GB of disk writes and kernel panics on the coordinator.
                if size_before > self.config.db_vacuum_max_size_bytes:
                    skipped_large += 1
                    logger.info(
                        f"[Maintenance] Skipping VACUUM for {db_path.name} "
                        f"({size_before / 1024 / 1024:.0f} MB > "
                        f"{self.config.db_vacuum_max_size_bytes / 1024 / 1024:.0f} MB limit)"
                    )
                    continue

                # December 2025: Run blocking VACUUM in thread pool to avoid blocking event loop
                def _vacuum_sync(path: str) -> None:
                    with sqlite3.connect(path) as conn:
                        conn.execute("VACUUM")

                await asyncio.to_thread(_vacuum_sync, str(db_path))

                # Get size after
                size_after = db_path.stat().st_size
                savings = size_before - size_after

                vacuumed += 1
                if savings > 1024 * 1024:  # Only log if > 1MB savings
                    logger.info(
                        f"[Maintenance] VACUUM {db_path.name}: "
                        f"{size_before / 1024 / 1024:.1f} MB → {size_after / 1024 / 1024:.1f} MB"
                    )

            except (sqlite3.Error, OSError, RuntimeError) as e:
                logger.warning(f"[Maintenance] Failed to VACUUM {db_path}: {e}")

        self._maintenance_stats.databases_vacuumed += vacuumed
        if vacuumed or skipped_large:
            logger.info(
                f"[Maintenance] VACUUM completed: {vacuumed} databases vacuumed"
                + (f", {skipped_large} skipped (too large)" if skipped_large else "")
            )

    async def _archive_old_games(self) -> None:
        """Archive games older than threshold to cold storage.

        Archives old database files by:
        1. Finding databases via GameDiscovery
        2. Compressing them with gzip (if enabled)
        3. Optionally uploading to S3
        4. Removing the original file
        """
        try:
            from app.utils.game_discovery import GameDiscovery

            discovery = GameDiscovery()
            threshold_days = self.config.archive_games_older_than_days
            archive_dir = Path(self.config.archive_directory)
            cutoff_time = time.time() - (threshold_days * 86400)

            all_dbs = discovery.find_all_databases()
            archived_count = 0

            for db_info in all_dbs:
                db_path = Path(db_info.path)

                # Skip archive directory and canonical databases
                if "archive" in str(db_path).lower():
                    continue
                if "canonical" in db_path.name:
                    continue

                # Check modification time
                try:
                    mtime = db_path.stat().st_mtime
                    if mtime > cutoff_time:
                        continue  # Not old enough
                except OSError:
                    continue

                age_days = (time.time() - mtime) / 86400

                if self.config.dry_run:
                    logger.info(
                        f"[Maintenance] DRY RUN: Would archive {db_path.name} "
                        f"({age_days:.1f} days old, {db_info.game_count} games)"
                    )
                    continue

                # Archive the database
                try:
                    success = await self._archive_single_database(
                        db_path, archive_dir, age_days
                    )
                    if success:
                        archived_count += 1
                        self._maintenance_stats.games_archived += db_info.game_count
                        logger.info(
                            f"[Maintenance] Archived {db_path.name} "
                            f"({db_info.game_count} games, {age_days:.1f} days old)"
                        )
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[Maintenance] Failed to archive {db_path}: {e}")

            if archived_count > 0:
                logger.info(
                    f"[Maintenance] Archival complete: {archived_count} databases, "
                    f"{self._maintenance_stats.games_archived} games total"
                )

        except ImportError:
            logger.debug("[Maintenance] GameDiscovery not available for archival")
        except (RuntimeError, OSError) as e:
            logger.warning(f"[Maintenance] Archive error: {e}")

    async def _archive_single_database(
        self, db_path: Path, archive_dir: Path, age_days: float
    ) -> bool:
        """Archive a single database file.

        Args:
            db_path: Path to the database to archive
            archive_dir: Directory to store archived files
            age_days: Age of the database in days (for logging)

        Returns:
            True if archived successfully
        """
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Create archive filename with timestamp
        timestamp = int(time.time())
        if self.config.archive_compress:
            archive_name = f"{db_path.stem}_{timestamp}.db.gz"
            archive_path = archive_dir / archive_name

            # Compress with gzip in thread to avoid blocking event loop
            def _compress_file() -> None:
                with open(db_path, "rb") as f_in, gzip.open(archive_path, "wb", compresslevel=6) as f_out:
                    shutil.copyfileobj(f_in, f_out)

            await asyncio.to_thread(_compress_file)
        else:
            archive_name = f"{db_path.stem}_{timestamp}.db"
            archive_path = archive_dir / archive_name
            await asyncio.to_thread(shutil.copy2, db_path, archive_path)

        logger.debug(f"[Maintenance] Created archive: {archive_path}")

        # Upload to S3 if configured
        if self.config.archive_to_s3:
            try:
                await self._upload_to_s3(archive_path)
                logger.debug(f"[Maintenance] Uploaded to S3: {archive_path.name}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[Maintenance] S3 upload failed for {archive_path}: {e}")
                # Continue anyway - local archive succeeded

        # Remove original file
        try:
            db_path.unlink()
            logger.debug(f"[Maintenance] Removed original: {db_path}")
            return True
        except OSError as e:
            logger.warning(f"[Maintenance] Failed to remove original {db_path}: {e}")
            return False

    async def _upload_to_s3(self, archive_path: Path) -> None:
        """Upload archived file to S3.

        Args:
            archive_path: Path to the archived file

        Raises:
            RuntimeError: If S3 upload fails
        """
        bucket = self.config.archive_s3_bucket
        s3_key = f"archives/{archive_path.name}"

        cmd = [
            "aws", "s3", "cp",
            str(archive_path),
            f"s3://{bucket}/{s3_key}",
            "--quiet",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"S3 upload failed: {stderr.decode().strip()}")

    async def _cleanup_dlq(self) -> None:
        """Cleanup old dead letter queue entries."""
        try:
            from app.distributed.unified_manifest import get_unified_manifest

            manifest = get_unified_manifest()
            if manifest is None:
                return

            if self.config.dry_run:
                stats = manifest.get_manifest_stats()
                logger.info(
                    f"[Maintenance] DRY RUN: Would cleanup DLQ "
                    f"({stats.dead_letter_count} unresolved entries)"
                )
                return

            cleaned = manifest.cleanup_old_dead_letters(days=self.config.dlq_retention_days)
            if cleaned:
                self._maintenance_stats.dlq_entries_cleaned += cleaned
                logger.info(f"[Maintenance] Cleaned {cleaned} old DLQ entries")

        except ImportError:
            pass
        except (RuntimeError, OSError) as e:
            logger.warning(f"[Maintenance] DLQ cleanup error: {e}")

    async def _cleanup_stale_queue_items(self) -> None:
        """Cleanup stale work queue items (December 2025).

        Removes items that were never executed:
        - PENDING items older than queue_stale_pending_hours
        - Resets CLAIMED items older than queue_stale_claimed_hours

        This prevents queue bloat from items that will never execute.
        """
        try:
            from app.coordination.work_queue import get_work_queue

            queue = get_work_queue()
            if queue is None:
                return

            if self.config.dry_run:
                status = queue.get_queue_status()
                pending_count = status.get("by_status", {}).get("pending", 0)
                claimed_count = status.get("by_status", {}).get("claimed", 0)
                logger.info(
                    f"[Maintenance] DRY RUN: Would cleanup stale queue items "
                    f"({pending_count} pending, {claimed_count} claimed)"
                )
                return

            # Cleanup stale items
            result = queue.cleanup_stale_items(
                max_pending_age_hours=self.config.queue_stale_pending_hours,
                max_claimed_age_hours=self.config.queue_stale_claimed_hours,
            )

            removed = result.get("removed_stale_pending", 0)
            reset = result.get("reset_stale_claimed", 0)

            if removed or reset:
                self._maintenance_stats.queue_items_cleaned += removed
                self._maintenance_stats.queue_items_reset += reset
                logger.info(
                    f"[Maintenance] Queue cleanup: removed {removed} stale pending, "
                    f"reset {reset} stale claimed"
                )

            # Also cleanup completed items older than 24h
            old_cleaned = queue.cleanup_old_items(max_age_seconds=86400.0)
            if old_cleaned:
                self._maintenance_stats.queue_items_cleaned += old_cleaned
                logger.info(f"[Maintenance] Cleaned {old_cleaned} old completed queue items")

        except ImportError:
            pass
        except (RuntimeError, OSError) as e:
            logger.warning(f"[Maintenance] Queue cleanup error: {e}")

    async def _check_elo_integrity(self) -> None:
        """Check Elo database for winner_id consistency (Feb 2026).

        Detects match_history rows where winner_id doesn't match any registered
        participant. This catches the class of bug where ID reconciliation scripts
        rename participant_ids but not winner_id (4,111 mismatches in one session).
        """
        try:
            from app.tournament.unified_elo_db import get_elo_database

            db = get_elo_database()
            issues = await asyncio.to_thread(db.check_winner_consistency)
            if issues:
                self._maintenance_stats.elo_integrity_issues = len(issues)
                logger.warning(
                    f"[Maintenance] Elo integrity: {len(issues)} winner_id mismatches "
                    f"(first: {issues[0].get('winner_id', '?')})"
                )
                try:
                    from app.coordination.event_router import emit_event
                    from app.distributed.data_events import DataEventType
                    emit_event(DataEventType.DATA_QUALITY_DEGRADED, {
                        "source": "elo_integrity_check",
                        "issue": "winner_id_mismatch",
                        "count": len(issues),
                    })
                except (ImportError, AttributeError):
                    pass
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"[Maintenance] Elo integrity check error: {e}")

    async def _detect_orphan_files(self) -> None:
        """Detect files on disk that aren't tracked in ClusterManifest (December 2025).

        Orphan files can accumulate when:
        - Manifest wasn't updated after file creation
        - Files were manually copied
        - Sync completed but manifest update failed

        Reports orphans for investigation; optionally cleans them up.
        """
        try:
            from app.distributed.cluster_manifest import get_cluster_manifest
        except ImportError:
            logger.debug("[Maintenance] ClusterManifest not available for orphan detection")
            return

        try:
            manifest = get_cluster_manifest()

            # Scan directories
            games_dir = self._ai_service_dir / "data" / "games"
            training_dir = self._ai_service_dir / "data" / "training"
            models_dir = self._ai_service_dir / "models"

            orphan_dbs = []
            orphan_npz = []
            orphan_models = []

            # Check game databases
            if games_dir.exists():
                tracked_dbs = set(manifest.get_all_db_paths())
                for db_file in games_dir.glob("**/*.db"):
                    if str(db_file) not in tracked_dbs and db_file.name not in tracked_dbs:
                        orphan_dbs.append(db_file)

            # Check NPZ files
            if training_dir.exists():
                tracked_npz = set(manifest.get_all_npz_paths())
                for npz_file in training_dir.glob("**/*.npz"):
                    if str(npz_file) not in tracked_npz and npz_file.name not in tracked_npz:
                        orphan_npz.append(npz_file)

            # Check model files (skip symlinks)
            if models_dir.exists():
                tracked_models = set(manifest.get_all_model_paths())
                for model_file in models_dir.glob("**/*.pth"):
                    if model_file.is_symlink():
                        continue  # Skip symlinks
                    if str(model_file) not in tracked_models and model_file.name not in tracked_models:
                        orphan_models.append(model_file)

            # Update stats
            self._maintenance_stats.orphan_dbs_found = len(orphan_dbs)
            self._maintenance_stats.orphan_npz_found = len(orphan_npz)
            self._maintenance_stats.orphan_models_found = len(orphan_models)

            total_orphans = len(orphan_dbs) + len(orphan_npz) + len(orphan_models)

            if total_orphans > 0:
                logger.warning(
                    f"[Maintenance] Found {total_orphans} orphan files: "
                    f"{len(orphan_dbs)} DBs, {len(orphan_npz)} NPZ, {len(orphan_models)} models"
                )

                # Log details for investigation
                for db in orphan_dbs[:5]:  # Limit to first 5
                    logger.info(f"[Maintenance] Orphan DB: {db}")
                for npz in orphan_npz[:5]:
                    logger.info(f"[Maintenance] Orphan NPZ: {npz}")
                for model in orphan_models[:5]:
                    logger.info(f"[Maintenance] Orphan model: {model}")

                if total_orphans > 15:
                    logger.info(f"[Maintenance] ... and {total_orphans - 15} more orphan files")

                # December 2025: Auto-recovery of orphan databases (preferred over cleanup)
                if self.config.orphan_auto_recovery and not self.config.dry_run and orphan_dbs:
                    recovered = await self._recover_orphan_databases(manifest, orphan_dbs)
                    self._maintenance_stats.orphan_dbs_recovered = recovered
                    if recovered > 0:
                        logger.info(f"[Maintenance] Recovered {recovered} orphan databases to manifest")

                # Optional cleanup (only for NPZ files, DBs should be recovered not deleted)
                if self.config.orphan_auto_cleanup and not self.config.dry_run:
                    await self._cleanup_orphan_files(orphan_dbs, orphan_npz, orphan_models)
            else:
                logger.debug("[Maintenance] No orphan files detected")

        except (RuntimeError, OSError) as e:
            logger.warning(f"[Maintenance] Orphan detection error: {e}")

    async def _cleanup_orphan_files(
        self,
        orphan_dbs: list,
        orphan_npz: list,
        orphan_models: list,
    ) -> None:
        """Cleanup orphan files (only called if orphan_auto_cleanup is True).

        Note: This is DANGEROUS - only enable if you're sure files are truly orphaned.
        Default is to just report, not delete.

        Args:
            orphan_dbs: List of orphan database files
            orphan_npz: List of orphan NPZ files
            orphan_models: List of orphan model files
        """
        cleaned = 0

        # Only cleanup NPZ files - DBs and models are more valuable
        for npz_file in orphan_npz:
            try:
                npz_file.unlink()
                cleaned += 1
                logger.info(f"[Maintenance] Deleted orphan NPZ: {npz_file}")
            except OSError as e:
                logger.warning(f"[Maintenance] Failed to delete {npz_file}: {e}")

        if cleaned:
            logger.info(f"[Maintenance] Cleaned {cleaned} orphan NPZ files")

    async def _recover_orphan_databases(self, manifest: Any, orphan_dbs: list) -> int:
        """Recover orphan databases by re-registering them to ClusterManifest (December 2025).

        This closes a critical data flow gap where games exist on disk but aren't
        tracked in the manifest, making them invisible to training pipelines.

        Args:
            manifest: ClusterManifest instance
            orphan_dbs: List of orphan database file paths

        Returns:
            Number of databases successfully recovered
        """
        recovered = 0

        for db_path in orphan_dbs:
            try:
                # Parse board_type and num_players from filename
                # Expected patterns: canonical_{board}_{n}p.db, {board}_{n}p.db, selfplay_{board}_{n}p.db
                name = db_path.stem  # Remove .db extension

                # Try to extract config from filename
                board_type = None
                num_players = None

                # Try canonical pattern first
                if name.startswith("canonical_"):
                    name = name[len("canonical_"):]

                # Try selfplay pattern
                if name.startswith("selfplay_"):
                    name = name[len("selfplay_"):]

                # Parse board_type and num_players using canonical utility
                parsed = parse_config_key(name)
                if parsed:
                    board_type = parsed.board_type
                    num_players = parsed.num_players

                # Fallback: if we can't parse, use defaults
                if not board_type:
                    board_type = "unknown"
                if not num_players:
                    num_players = 2

                # Count games in the database
                # December 30, 2025: Wrap blocking SQLite in asyncio.to_thread
                def _count_games(path: str) -> int:
                    try:
                        with sqlite3.connect(path, timeout=SQLITE_CONNECT_TIMEOUT) as conn:
                            cursor = conn.execute("SELECT COUNT(*) FROM games")
                            return cursor.fetchone()[0]
                    except (sqlite3.Error, OSError, IndexError):
                        return -1  # Signal error

                game_count = await asyncio.to_thread(_count_games, str(db_path))
                if game_count < 0:
                    # If we can't read the database, skip it
                    logger.debug(f"[Maintenance] Couldn't read orphan DB {db_path}, skipping")
                    continue

                if game_count == 0:
                    # Empty database, not worth recovering
                    continue

                # Get node_id (use local hostname)
                import socket
                node_id = os.environ.get("RINGRIFT_NODE_ID", socket.gethostname())

                # Register to manifest
                # Dec 31, 2025: Use register_database() instead of register_games_batch()
                if hasattr(manifest, 'register_database'):
                    manifest.register_database(
                        db_path=str(db_path),
                        node_id=node_id,
                        board_type=board_type,
                        num_players=num_players,
                        game_count=game_count,
                    )
                    recovered += 1
                    logger.info(
                        f"[Maintenance] Recovered orphan DB: {db_path.name} "
                        f"({game_count} games, {board_type}_{num_players}p)"
                    )

                    # Emit event for downstream consumers
                    from app.coordination.event_emission_helpers import safe_emit_event_async

                    await safe_emit_event_async(
                        "ORPHAN_GAMES_REGISTERED",
                        {
                            "db_path": str(db_path),
                            "node_id": node_id,
                            "board_type": board_type,
                            "num_players": num_players,
                            "game_count": game_count,
                        },
                        source="maintenance_daemon",
                        context="MaintenanceDaemon.recover_orphan_db",
                    )

            except (OSError, RuntimeError) as e:
                logger.warning(f"[Maintenance] Failed to recover orphan DB {db_path}: {e}")

        return recovered

    def get_status(self) -> dict[str, Any]:
        """Get daemon status."""
        return {
            "running": self._running,
            "enabled": self.config.enabled,
            "dry_run": self.config.dry_run,
            "stats": {
                "logs_rotated": self._maintenance_stats.logs_rotated,
                "bytes_reclaimed_logs_mb": self._maintenance_stats.bytes_reclaimed_logs / 1024 / 1024,
                "databases_vacuumed": self._maintenance_stats.databases_vacuumed,
                "games_archived": self._maintenance_stats.games_archived,
                "dlq_entries_cleaned": self._maintenance_stats.dlq_entries_cleaned,
                "queue_items_cleaned": self._maintenance_stats.queue_items_cleaned,  # December 2025
                "queue_items_reset": self._maintenance_stats.queue_items_reset,  # December 2025
                "orphan_dbs_found": self._maintenance_stats.orphan_dbs_found,  # December 2025
                "orphan_npz_found": self._maintenance_stats.orphan_npz_found,  # December 2025
                "orphan_models_found": self._maintenance_stats.orphan_models_found,  # December 2025
                "orphan_dbs_recovered": self._maintenance_stats.orphan_dbs_recovered,  # December 2025
            },
            "last_runs": {
                "log_rotation": self._maintenance_stats.last_log_rotation,
                "db_vacuum": self._maintenance_stats.last_db_vacuum,
                "archive": self._maintenance_stats.last_archive_run,
                "dlq_cleanup": self._maintenance_stats.last_dlq_cleanup,
                "queue_cleanup": self._maintenance_stats.last_queue_cleanup,  # December 2025
                "orphan_detection": self._maintenance_stats.last_orphan_detection,  # December 2025
            },
            "config": {
                "log_max_size_mb": self.config.log_max_size_mb,
                "log_backup_count": self.config.log_backup_count,
                "db_vacuum_interval_hours": self.config.db_vacuum_interval_hours,
                "archive_days_threshold": self.config.archive_games_older_than_days,
                "queue_cleanup_interval_hours": self.config.queue_cleanup_interval_hours,  # December 2025
                "orphan_detection_interval_hours": self.config.orphan_detection_interval_hours,  # December 2025
            },
        }

    def health_check(self) -> "HealthCheckResult":
        """Check daemon health status.

        December 2025: Added to satisfy CoordinatorProtocol for unified health monitoring.
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        if not self.is_running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="Maintenance daemon not running",
            )

        # Calculate time since last maintenance runs
        now = time.time()
        hours_since_log = (now - self._maintenance_stats.last_log_rotation) / 3600 if self._maintenance_stats.last_log_rotation else float('inf')
        hours_since_vacuum = (now - self._maintenance_stats.last_db_vacuum) / 3600 if self._maintenance_stats.last_db_vacuum else float('inf')

        # Warning if maintenance tasks are overdue
        if hours_since_log > self.config.log_rotation_interval_hours * 3:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"Log rotation overdue ({hours_since_log:.1f}h since last run)",
                details=self.get_status(),
            )

        if hours_since_vacuum > self.config.db_vacuum_interval_hours * 1.5:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"Database vacuum overdue ({hours_since_vacuum:.1f}h since last run)",
                details=self.get_status(),
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"Maintenance daemon running (logs: {self._maintenance_stats.logs_rotated}, vacuums: {self._maintenance_stats.databases_vacuumed})",
            details=self.get_status(),
        )


# January 2026: Module-level singleton now uses HandlerBase.get_instance()
# Backward-compat functions retained for existing callers


def get_maintenance_daemon() -> MaintenanceDaemon:
    """Get the singleton MaintenanceDaemon instance.

    January 2026: Now delegates to HandlerBase.get_instance() for consistency.
    """
    return MaintenanceDaemon.get_instance()


def reset_maintenance_daemon() -> None:
    """Reset the singleton (for testing).

    January 2026: Now uses HandlerBase.reset_instance() pattern.
    """
    MaintenanceDaemon.reset_instance()

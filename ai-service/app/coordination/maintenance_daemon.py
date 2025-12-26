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
import gzip
import logging
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "MaintenanceConfig",
    "MaintenanceDaemon",
    "get_maintenance_daemon",
]


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

    # Game archival
    archive_games_older_than_days: int = 30
    archive_interval_hours: float = 24.0  # Daily
    archive_enabled: bool = True

    # DLQ cleanup (also handled by DLQ_RETRY daemon, but backup cleanup here)
    dlq_cleanup_interval_hours: float = 168.0  # Weekly
    dlq_retention_days: int = 7

    # Work queue cleanup (December 2025)
    queue_cleanup_interval_hours: float = 1.0  # Hourly
    queue_stale_pending_hours: float = 24.0  # Remove PENDING items older than this
    queue_stale_claimed_hours: float = 1.0  # Reset CLAIMED items older than this
    queue_cleanup_enabled: bool = True

    # General
    dry_run: bool = False  # If True, log actions but don't execute


@dataclass
class MaintenanceStats:
    """Statistics from maintenance operations."""
    logs_rotated: int = 0
    bytes_reclaimed_logs: int = 0
    databases_vacuumed: int = 0
    games_archived: int = 0
    dlq_entries_cleaned: int = 0
    queue_items_cleaned: int = 0  # December 2025
    queue_items_reset: int = 0  # December 2025
    last_log_rotation: float = 0.0
    last_db_vacuum: float = 0.0
    last_archive_run: float = 0.0
    last_dlq_cleanup: float = 0.0
    last_queue_cleanup: float = 0.0  # December 2025


class MaintenanceDaemon:
    """Daemon for automated system maintenance.

    Runs scheduled maintenance tasks:
    - Log rotation (hourly)
    - Database VACUUM (weekly)
    - Game archival (daily)
    - DLQ cleanup (weekly)
    """

    def __init__(self, config: MaintenanceConfig | None = None):
        self.config = config or MaintenanceConfig()
        self._running = False
        self._stats = MaintenanceStats()
        self._ai_service_dir = Path(__file__).parent.parent.parent

    async def start(self) -> None:
        """Start the maintenance daemon."""
        if self._running:
            return

        self._running = True
        logger.info("[Maintenance] Daemon started")

        # Initialize last run times
        now = time.time()
        self._stats.last_log_rotation = now
        self._stats.last_db_vacuum = now
        self._stats.last_archive_run = now
        self._stats.last_dlq_cleanup = now

        # Run initial maintenance check
        await self._run_maintenance_cycle()

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
            except Exception as e:
                logger.error(f"[Maintenance] Cycle error: {e}")

            # Check every 10 minutes
            await asyncio.sleep(600)

    async def _run_maintenance_cycle(self) -> None:
        """Run a maintenance check cycle."""
        if not self.config.enabled:
            return

        now = time.time()

        # Hourly: Log rotation
        hours_since_log_rotation = (now - self._stats.last_log_rotation) / 3600
        if hours_since_log_rotation >= self.config.log_rotation_interval_hours:
            await self._rotate_logs()
            self._stats.last_log_rotation = now

        # Daily: Archive old games
        hours_since_archive = (now - self._stats.last_archive_run) / 3600
        if hours_since_archive >= self.config.archive_interval_hours:
            if self.config.archive_enabled:
                await self._archive_old_games()
            self._stats.last_archive_run = now

        # Weekly: VACUUM databases
        hours_since_vacuum = (now - self._stats.last_db_vacuum) / 3600
        if hours_since_vacuum >= self.config.db_vacuum_interval_hours:
            if self.config.db_maintenance_enabled:
                await self._vacuum_databases()
            self._stats.last_db_vacuum = now

        # Weekly: DLQ cleanup
        hours_since_dlq = (now - self._stats.last_dlq_cleanup) / 3600
        if hours_since_dlq >= self.config.dlq_cleanup_interval_hours:
            await self._cleanup_dlq()
            self._stats.last_dlq_cleanup = now

        # Hourly: Work queue stale item cleanup (December 2025)
        hours_since_queue = (now - self._stats.last_queue_cleanup) / 3600
        if hours_since_queue >= self.config.queue_cleanup_interval_hours:
            if self.config.queue_cleanup_enabled:
                await self._cleanup_stale_queue_items()
            self._stats.last_queue_cleanup = now

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

                    # Compress current log to .1.gz
                    backup_path = log_file.with_suffix(".log.1.gz")
                    with open(log_file, "rb") as f_in:
                        with gzip.open(backup_path, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)

                    # Truncate original log
                    with open(log_file, "w") as f:
                        f.write(f"# Log rotated at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

                    rotated += 1
                    bytes_saved += file_size
                    logger.info(f"[Maintenance] Rotated {log_file.name} ({file_size / 1024 / 1024:.1f} MB)")

                except Exception as e:
                    logger.warning(f"[Maintenance] Failed to rotate {log_file}: {e}")

        except Exception as e:
            logger.error(f"[Maintenance] Log rotation error: {e}")

        if rotated:
            self._stats.logs_rotated += rotated
            self._stats.bytes_reclaimed_logs += bytes_saved
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
        for db_path in db_paths:
            if self.config.dry_run:
                logger.info(f"[Maintenance] DRY RUN: Would VACUUM {db_path.name}")
                continue

            try:
                # Get size before
                size_before = db_path.stat().st_size

                conn = sqlite3.connect(str(db_path))
                conn.execute("VACUUM")
                conn.close()

                # Get size after
                size_after = db_path.stat().st_size
                savings = size_before - size_after

                vacuumed += 1
                if savings > 1024 * 1024:  # Only log if > 1MB savings
                    logger.info(
                        f"[Maintenance] VACUUM {db_path.name}: "
                        f"{size_before / 1024 / 1024:.1f} MB → {size_after / 1024 / 1024:.1f} MB"
                    )

            except Exception as e:
                logger.warning(f"[Maintenance] Failed to VACUUM {db_path}: {e}")

        self._stats.databases_vacuumed += vacuumed
        if vacuumed:
            logger.info(f"[Maintenance] VACUUM completed: {vacuumed} databases")

    async def _archive_old_games(self) -> None:
        """Archive games older than threshold to cold storage.

        Note: This is a placeholder - actual implementation depends on
        cold storage backend (S3, GCS, or just separate directory).
        """
        try:
            from app.distributed.unified_manifest import get_unified_manifest

            manifest = get_unified_manifest()
            if manifest is None:
                return

            # Get stats on games that could be archived
            stats = manifest.get_manifest_stats()
            threshold_days = self.config.archive_games_older_than_days

            if self.config.dry_run:
                logger.info(
                    f"[Maintenance] DRY RUN: Would check {stats.total_games} games "
                    f"for archival (>{threshold_days} days old)"
                )
                return

            # For now, just log that archival would happen
            # Full implementation would:
            # 1. Query games older than threshold
            # 2. Export to compressed archive
            # 3. Remove from active database
            # 4. Update manifest
            logger.debug(
                f"[Maintenance] Archive check: {stats.total_games} games tracked, "
                f"threshold={threshold_days} days"
            )

        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"[Maintenance] Archive check error: {e}")

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
                self._stats.dlq_entries_cleaned += cleaned
                logger.info(f"[Maintenance] Cleaned {cleaned} old DLQ entries")

        except ImportError:
            pass
        except Exception as e:
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
                self._stats.queue_items_cleaned += removed
                self._stats.queue_items_reset += reset
                logger.info(
                    f"[Maintenance] Queue cleanup: removed {removed} stale pending, "
                    f"reset {reset} stale claimed"
                )

            # Also cleanup completed items older than 24h
            old_cleaned = queue.cleanup_old_items(max_age_seconds=86400.0)
            if old_cleaned:
                self._stats.queue_items_cleaned += old_cleaned
                logger.info(f"[Maintenance] Cleaned {old_cleaned} old completed queue items")

        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"[Maintenance] Queue cleanup error: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get daemon status."""
        return {
            "running": self._running,
            "enabled": self.config.enabled,
            "dry_run": self.config.dry_run,
            "stats": {
                "logs_rotated": self._stats.logs_rotated,
                "bytes_reclaimed_logs_mb": self._stats.bytes_reclaimed_logs / 1024 / 1024,
                "databases_vacuumed": self._stats.databases_vacuumed,
                "games_archived": self._stats.games_archived,
                "dlq_entries_cleaned": self._stats.dlq_entries_cleaned,
                "queue_items_cleaned": self._stats.queue_items_cleaned,  # December 2025
                "queue_items_reset": self._stats.queue_items_reset,  # December 2025
            },
            "last_runs": {
                "log_rotation": self._stats.last_log_rotation,
                "db_vacuum": self._stats.last_db_vacuum,
                "archive": self._stats.last_archive_run,
                "dlq_cleanup": self._stats.last_dlq_cleanup,
                "queue_cleanup": self._stats.last_queue_cleanup,  # December 2025
            },
            "config": {
                "log_max_size_mb": self.config.log_max_size_mb,
                "log_backup_count": self.config.log_backup_count,
                "db_vacuum_interval_hours": self.config.db_vacuum_interval_hours,
                "archive_days_threshold": self.config.archive_games_older_than_days,
                "queue_cleanup_interval_hours": self.config.queue_cleanup_interval_hours,  # December 2025
            },
        }


# Module-level singleton
_maintenance_daemon: MaintenanceDaemon | None = None


def get_maintenance_daemon() -> MaintenanceDaemon:
    """Get the singleton MaintenanceDaemon instance."""
    global _maintenance_daemon
    if _maintenance_daemon is None:
        _maintenance_daemon = MaintenanceDaemon()
    return _maintenance_daemon


def reset_maintenance_daemon() -> None:
    """Reset the singleton (for testing)."""
    global _maintenance_daemon
    _maintenance_daemon = None

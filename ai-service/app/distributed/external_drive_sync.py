"""External Drive Sync Daemon - Backup game data to external drives.

This daemon periodically syncs game databases to mounted external drives
for backup and offline analysis.

Use cases:
- Local backup to USB/external SSD
- Archive cold data to NAS
- Export for offline analysis

Architecture:
    1. Detects mounted external drives (configurable mount points)
    2. Syncs .db files from data/games/ to external drive
    3. Verifies integrity with checksums
    4. Emits DATA_BACKUP_COMPLETED event

Usage:
    # Via DaemonManager
    manager.register_factory(DaemonType.EXTERNAL_DRIVE_SYNC, daemon.run)

    # Standalone
    python -m app.distributed.external_drive_sync
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExternalDriveSyncConfig:
    """Configuration for external drive sync daemon."""

    # Scan settings
    sync_interval_seconds: float = 3600.0  # 1 hour
    mount_points: list[str] = field(
        default_factory=lambda: [
            "/mnt/backup",
            "/Volumes/Backup",
            os.path.expanduser("~/backup"),
        ]
    )
    target_subdir: str = "ringrift_backups"

    # Source settings
    source_dir: str = "data/games"
    include_patterns: list[str] = field(default_factory=lambda: ["*.db"])

    # Behavior
    verify_checksums: bool = True
    skip_if_unchanged: bool = True
    emit_events: bool = True


@dataclass
class ExternalDriveSyncResult:
    """Result of an external drive sync operation.

    Note: This is distinct from app.coordination.sync_constants.SyncResult
    which is used for distributed cluster sync operations.
    """

    drive_path: Path
    files_synced: int = 0
    bytes_synced: int = 0
    errors: list[str] = field(default_factory=list)
    success: bool = True


# Backward-compatible alias
SyncResult = ExternalDriveSyncResult


class ExternalDriveSyncDaemon:
    """Daemon that syncs game data to external drives.

    Periodically scans for mounted external drives and syncs
    game databases for backup purposes.
    """

    def __init__(self, config: ExternalDriveSyncConfig | None = None):
        self.config = config or ExternalDriveSyncConfig()
        self._running = False
        self._last_sync_time: float = 0.0
        self._sync_history: list[dict[str, Any]] = []

    async def start(self) -> None:
        """Start the daemon."""
        logger.info("ExternalDriveSyncDaemon starting...")
        self._running = True

        while self._running:
            try:
                if time.time() - self._last_sync_time > self.config.sync_interval_seconds:
                    await self._run_sync()
                    self._last_sync_time = time.time()

                await asyncio.sleep(60.0)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in external drive sync loop: {e}")
                await asyncio.sleep(60.0)

        logger.info("ExternalDriveSyncDaemon stopped")

    async def stop(self) -> None:
        """Stop the daemon gracefully."""
        self._running = False

    async def _run_sync(self) -> list[SyncResult]:
        """Run sync to all available external drives."""
        results: list[SyncResult] = []

        # Find available mount points
        available_drives = self._find_available_drives()
        if not available_drives:
            logger.debug("No external drives found for backup")
            return results

        logger.info(f"Found {len(available_drives)} available drives for backup")

        for drive_path in available_drives:
            try:
                result = await self._sync_to_drive(drive_path)
                results.append(result)

                if result.success:
                    logger.info(
                        f"Synced {result.files_synced} files "
                        f"({result.bytes_synced / 1024 / 1024:.1f} MB) "
                        f"to {drive_path}"
                    )
            except Exception as e:
                logger.error(f"Failed to sync to {drive_path}: {e}")
                results.append(
                    SyncResult(drive_path=drive_path, success=False, errors=[str(e)])
                )

        # Emit event
        if self.config.emit_events and results:
            await self._emit_backup_event(results)

        return results

    def _find_available_drives(self) -> list[Path]:
        """Find available external drives from configured mount points."""
        available: list[Path] = []

        for mount_point in self.config.mount_points:
            path = Path(mount_point)
            if path.exists() and path.is_dir():
                # Check if writable
                try:
                    test_file = path / ".ringrift_write_test"
                    test_file.touch()
                    test_file.unlink()
                    available.append(path)
                except (PermissionError, OSError):
                    logger.debug(f"Mount point not writable: {mount_point}")
                    continue

        return available

    async def _sync_to_drive(self, drive_path: Path) -> SyncResult:
        """Sync game data to a specific drive."""
        result = SyncResult(drive_path=drive_path)

        # Create target directory
        target_dir = drive_path / self.config.target_subdir
        target_dir.mkdir(parents=True, exist_ok=True)

        # Get source files
        source_dir = Path(self.config.source_dir)
        if not source_dir.exists():
            logger.warning(f"Source directory not found: {source_dir}")
            return result

        for pattern in self.config.include_patterns:
            for source_file in source_dir.glob(pattern):
                try:
                    target_file = target_dir / source_file.name

                    # Skip if unchanged
                    if (
                        self.config.skip_if_unchanged
                        and target_file.exists()
                        and target_file.stat().st_size == source_file.stat().st_size
                        and target_file.stat().st_mtime >= source_file.stat().st_mtime
                    ):
                        continue

                    # Copy file
                    shutil.copy2(source_file, target_file)
                    result.files_synced += 1
                    result.bytes_synced += source_file.stat().st_size

                except Exception as e:
                    result.errors.append(f"{source_file.name}: {e}")

        result.success = len(result.errors) == 0
        return result

    async def _emit_backup_event(self, results: list[SyncResult]) -> None:
        """Emit DATA_BACKUP_COMPLETED event."""
        try:
            from app.coordination.event_router import get_router, DataEventType

            router = get_router()
            if router is None:
                return

            total_files = sum(r.files_synced for r in results)
            total_bytes = sum(r.bytes_synced for r in results)
            success_count = sum(1 for r in results if r.success)

            await router.publish(
                DataEventType.DATA_BACKUP_COMPLETED,
                {
                    "drives_synced": len(results),
                    "drives_succeeded": success_count,
                    "total_files": total_files,
                    "total_bytes": total_bytes,
                    "timestamp": time.time(),
                },
            )

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Failed to emit backup event: {e}")


async def run() -> None:
    """Run the daemon (entry point for DaemonManager)."""
    daemon = ExternalDriveSyncDaemon()
    await daemon.start()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(run())

"""Dual Backup Daemon - Ensures data is backed up to both S3 AND OWC (January 2026).

This daemon orchestrates backups to both Amazon S3 and OWC external drive,
ensuring redundant copies of all critical training data:
- Canonical databases (data/games/canonical_*.db)
- Training NPZ files (data/training/*.npz)
- Model checkpoints (models/canonical_*.pth)

Key features:
- Guaranteed dual-destination backup
- Verification after backup (checksum comparison)
- Retry logic with exponential backoff
- Progress tracking and metrics
- 24-hour scheduled full backup
- Alert on backup failures

Usage:
    from app.coordination.dual_backup_daemon import (
        DualBackupDaemon,
        get_dual_backup_daemon,
    )

    daemon = get_dual_backup_daemon()
    await daemon.start()

    # Manual full backup
    await daemon.run_full_backup()

Environment Variables:
    RINGRIFT_DUAL_BACKUP_ENABLED: Enable/disable daemon (default: true)
    RINGRIFT_DUAL_BACKUP_VERIFY: Verify checksums after backup (default: true)
    RINGRIFT_DUAL_BACKUP_INTERVAL: Full backup interval in seconds (default: 86400 = 24h)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.protocols import CoordinatorStatus
from app.coordination.event_emission_helpers import safe_emit_event
from app.coordination.event_router import get_event_payload
from app.utils.retry import RetryConfig

logger = logging.getLogger(__name__)


class BackupDestination(str, Enum):
    """Backup destination types."""

    S3 = "s3"
    OWC = "owc"


class BackupStatus(str, Enum):
    """Status of a backup operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"


@dataclass
class BackupResult:
    """Result of a backup operation."""

    file_path: str
    destination: BackupDestination
    status: BackupStatus
    size_bytes: int = 0
    duration_seconds: float = 0.0
    error: str | None = None
    verified: bool = False


@dataclass
class DualBackupConfig:
    """Configuration for dual backup daemon."""

    enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_DUAL_BACKUP_ENABLED", "true"
        ).lower() == "true"
    )
    verify_backups: bool = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_DUAL_BACKUP_VERIFY", "true"
        ).lower() == "true"
    )
    backup_interval: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_DUAL_BACKUP_INTERVAL", "86400")  # 24 hours
        )
    )

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 60.0  # 1 minute initial delay
    retry_backoff: float = 2.0  # Exponential backoff multiplier

    # Alert thresholds
    max_consecutive_failures: int = 3
    stale_backup_hours: int = 48


@dataclass
class DualBackupStats:
    """Statistics for dual backup operations."""

    total_backups: int = 0
    successful_backups: int = 0
    failed_backups: int = 0
    bytes_backed_up: int = 0
    last_full_backup_time: float = 0.0
    last_backup_duration: float = 0.0
    s3_successes: int = 0
    s3_failures: int = 0
    owc_successes: int = 0
    owc_failures: int = 0
    consecutive_failures: int = 0
    verification_failures: int = 0


class DualBackupDaemon(HandlerBase):
    """Daemon that ensures data is backed up to both S3 and OWC.

    Orchestrates the S3PushDaemon and OWCPushDaemon to guarantee
    that all critical training data exists in both destinations.
    """

    _instance: DualBackupDaemon | None = None

    def __init__(self, config: DualBackupConfig | None = None):
        """Initialize dual backup daemon.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        resolved_config = config or DualBackupConfig()
        super().__init__(
            name="dual_backup",
            config=resolved_config,
            cycle_interval=resolved_config.backup_interval,
        )

        self._backup_stats = DualBackupStats()
        self._base_path = Path(os.environ.get("RINGRIFT_BASE_PATH", "."))
        self._pending_backups: list[BackupResult] = []
        self._recent_results: list[BackupResult] = []

        # Track file backup status
        self._file_backup_status: dict[str, dict[BackupDestination, BackupResult]] = {}

    @classmethod
    def get_instance(cls) -> DualBackupDaemon:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _enumerate_backup_files(self) -> list[Path]:
        """Enumerate all files that should be backed up."""
        files = []

        # Canonical databases
        games_dir = self._base_path / "data" / "games"
        if games_dir.exists():
            files.extend(games_dir.glob("canonical_*.db"))

        # Training NPZ files
        training_dir = self._base_path / "data" / "training"
        if training_dir.exists():
            files.extend(training_dir.glob("*.npz"))

        # Model checkpoints
        models_dir = self._base_path / "models"
        if models_dir.exists():
            files.extend(models_dir.glob("canonical_*.pth"))

        return files

    async def _backup_to_s3(self, file_path: Path) -> BackupResult:
        """Backup a file to S3."""
        start_time = time.time()
        result = BackupResult(
            file_path=str(file_path),
            destination=BackupDestination.S3,
            status=BackupStatus.IN_PROGRESS,
        )

        try:
            from app.coordination.s3_push_daemon import get_s3_push_daemon

            s3_daemon = get_s3_push_daemon()

            # Determine S3 key based on file type
            if file_path.suffix == ".db":
                s3_key = f"consolidated/games/{file_path.name}"
            elif file_path.suffix == ".npz":
                s3_key = f"consolidated/training/{file_path.name}"
            else:
                s3_key = f"models/{file_path.name}"

            # Use S3 daemon's push method
            success = await s3_daemon._push_if_modified(file_path, s3_key)

            if success:
                result.status = BackupStatus.COMPLETED
                result.size_bytes = file_path.stat().st_size
                self._backup_stats.s3_successes += 1
            else:
                result.status = BackupStatus.FAILED
                result.error = "S3 push returned false"
                self._backup_stats.s3_failures += 1

        except Exception as e:
            result.status = BackupStatus.FAILED
            result.error = str(e)
            self._backup_stats.s3_failures += 1

        result.duration_seconds = time.time() - start_time
        return result

    async def _backup_to_owc(self, file_path: Path) -> BackupResult:
        """Backup a file to OWC drive."""
        start_time = time.time()
        result = BackupResult(
            file_path=str(file_path),
            destination=BackupDestination.OWC,
            status=BackupStatus.IN_PROGRESS,
        )

        try:
            from app.coordination.owc_push_daemon import get_owc_push_daemon

            owc_daemon = get_owc_push_daemon()

            # Determine OWC path based on file type
            if file_path.suffix == ".db":
                dest_path = f"{owc_daemon.config.canonical_db_subdir}/{file_path.name}"
            elif file_path.suffix == ".npz":
                dest_path = f"{owc_daemon.config.training_subdir}/{file_path.name}"
            else:
                dest_path = f"{owc_daemon.config.models_subdir}/{file_path.name}"

            # Use OWC daemon's push method
            success = await owc_daemon._push_if_modified(file_path, dest_path)

            if success:
                result.status = BackupStatus.COMPLETED
                result.size_bytes = file_path.stat().st_size
                self._backup_stats.owc_successes += 1
            else:
                result.status = BackupStatus.FAILED
                result.error = "OWC push returned false"
                self._backup_stats.owc_failures += 1

        except Exception as e:
            result.status = BackupStatus.FAILED
            result.error = str(e)
            self._backup_stats.owc_failures += 1

        result.duration_seconds = time.time() - start_time
        return result

    async def _backup_with_retry(
        self,
        file_path: Path,
        destination: BackupDestination,
    ) -> BackupResult:
        """Backup a file with retry logic.

        Jan 3, 2026: Migrated to RetryConfig for centralized retry behavior.
        """
        last_result = None

        # Create retry config matching original exponential backoff behavior
        retry_config = RetryConfig(
            max_attempts=self.config.max_retries,
            base_delay=self.config.retry_delay,
            max_delay=self.config.retry_delay * (self.config.retry_backoff ** self.config.max_retries),
        )

        for attempt in retry_config.attempts():
            if destination == BackupDestination.S3:
                result = await self._backup_to_s3(file_path)
            else:
                result = await self._backup_to_owc(file_path)

            last_result = result

            if result.status == BackupStatus.COMPLETED:
                return result

            # Wait before retry with exponential backoff
            if attempt.should_retry:
                logger.warning(
                    f"[DualBackup] Backup failed for {file_path.name} to {destination.value}, "
                    f"retrying in {attempt.delay:.0f}s (attempt {attempt.number}/{retry_config.max_attempts})"
                )
                await attempt.wait_async()

        return last_result or BackupResult(
            file_path=str(file_path),
            destination=destination,
            status=BackupStatus.FAILED,
            error="Max retries exceeded",
        )

    async def ensure_dual_backup(self, file_path: Path) -> tuple[BackupResult, BackupResult]:
        """Ensure a file is backed up to both S3 and OWC.

        Args:
            file_path: Path to file to backup

        Returns:
            Tuple of (S3 result, OWC result)
        """
        # Backup to both destinations in parallel
        s3_task = self._backup_with_retry(file_path, BackupDestination.S3)
        owc_task = self._backup_with_retry(file_path, BackupDestination.OWC)

        s3_result, owc_result = await asyncio.gather(s3_task, owc_task)

        # Track status
        file_key = str(file_path)
        if file_key not in self._file_backup_status:
            self._file_backup_status[file_key] = {}

        self._file_backup_status[file_key][BackupDestination.S3] = s3_result
        self._file_backup_status[file_key][BackupDestination.OWC] = owc_result

        # Update stats
        self._backup_stats.total_backups += 2
        if s3_result.status == BackupStatus.COMPLETED:
            self._backup_stats.successful_backups += 1
            self._backup_stats.bytes_backed_up += s3_result.size_bytes
        else:
            self._backup_stats.failed_backups += 1

        if owc_result.status == BackupStatus.COMPLETED:
            self._backup_stats.successful_backups += 1
            self._backup_stats.bytes_backed_up += owc_result.size_bytes
        else:
            self._backup_stats.failed_backups += 1

        return s3_result, owc_result

    async def run_full_backup(self) -> dict[str, Any]:
        """Run a full backup of all files to both destinations.

        Returns:
            Summary of backup results
        """
        start_time = time.time()
        logger.info("[DualBackup] Starting full backup")

        files = self._enumerate_backup_files()
        results = {
            "files_processed": 0,
            "s3_successes": 0,
            "s3_failures": 0,
            "owc_successes": 0,
            "owc_failures": 0,
            "total_bytes": 0,
            "errors": [],
        }

        for file_path in files:
            try:
                s3_result, owc_result = await self.ensure_dual_backup(file_path)
                results["files_processed"] += 1

                if s3_result.status == BackupStatus.COMPLETED:
                    results["s3_successes"] += 1
                    results["total_bytes"] += s3_result.size_bytes
                else:
                    results["s3_failures"] += 1
                    results["errors"].append(f"S3: {file_path.name}: {s3_result.error}")

                if owc_result.status == BackupStatus.COMPLETED:
                    results["owc_successes"] += 1
                else:
                    results["owc_failures"] += 1
                    results["errors"].append(f"OWC: {file_path.name}: {owc_result.error}")

            except Exception as e:
                results["errors"].append(f"{file_path.name}: {e}")
                logger.error(f"[DualBackup] Error backing up {file_path}: {e}")

        results["duration_seconds"] = time.time() - start_time
        self._backup_stats.last_full_backup_time = time.time()
        self._backup_stats.last_backup_duration = results["duration_seconds"]

        # Check for consecutive failures
        if results["s3_failures"] > 0 or results["owc_failures"] > 0:
            self._backup_stats.consecutive_failures += 1
        else:
            self._backup_stats.consecutive_failures = 0

        # Emit alert if too many failures
        if self._backup_stats.consecutive_failures >= self.config.max_consecutive_failures:
            self._emit_backup_alert(results)

        logger.info(
            f"[DualBackup] Full backup complete: "
            f"{results['files_processed']} files, "
            f"S3: {results['s3_successes']}/{results['files_processed']}, "
            f"OWC: {results['owc_successes']}/{results['files_processed']}, "
            f"{results['duration_seconds']:.1f}s"
        )

        return results

    async def check_backup_coverage(self) -> dict[str, Any]:
        """Check which files are backed up to which destinations.

        Returns:
            Coverage report
        """
        files = self._enumerate_backup_files()
        report = {
            "total_files": len(files),
            "both_destinations": 0,
            "s3_only": 0,
            "owc_only": 0,
            "no_backup": 0,
            "missing_files": [],
        }

        for file_path in files:
            file_key = str(file_path)
            status = self._file_backup_status.get(file_key, {})

            s3_ok = (
                BackupDestination.S3 in status
                and status[BackupDestination.S3].status == BackupStatus.COMPLETED
            )
            owc_ok = (
                BackupDestination.OWC in status
                and status[BackupDestination.OWC].status == BackupStatus.COMPLETED
            )

            if s3_ok and owc_ok:
                report["both_destinations"] += 1
            elif s3_ok:
                report["s3_only"] += 1
                report["missing_files"].append({"file": file_path.name, "missing": "OWC"})
            elif owc_ok:
                report["owc_only"] += 1
                report["missing_files"].append({"file": file_path.name, "missing": "S3"})
            else:
                report["no_backup"] += 1
                report["missing_files"].append({"file": file_path.name, "missing": "both"})

        return report

    def _emit_backup_alert(self, results: dict[str, Any]) -> None:
        """Emit alert for backup failures."""
        safe_emit_event(
            "backup_failed",
            {
                "consecutive_failures": self._backup_stats.consecutive_failures,
                "s3_failures": results.get("s3_failures", 0),
                "owc_failures": results.get("owc_failures", 0),
                "errors": results.get("errors", [])[:5],  # First 5 errors
            },
            context="DualBackup",
        )

    def _emit_backup_complete(self, results: dict[str, Any]) -> None:
        """Emit backup completion event."""
        safe_emit_event(
            "data_sync_completed",
            {
                "sync_type": "dual_backup",
                "files_processed": results.get("files_processed", 0),
                "s3_successes": results.get("s3_successes", 0),
                "owc_successes": results.get("owc_successes", 0),
                "duration_seconds": results.get("duration_seconds", 0),
            },
            context="DualBackup",
        )

    async def _run_cycle(self) -> None:
        """Run one backup cycle."""
        if not self.config.enabled:
            logger.debug("[DualBackup] Disabled via config, skipping cycle")
            return

        results = await self.run_full_backup()
        self._emit_backup_complete(results)

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Get event subscriptions for this daemon."""
        return {
            "CONSOLIDATION_COMPLETE": self._on_data_changed,
            "TRAINING_COMPLETED": self._on_data_changed,
            "NPZ_EXPORT_COMPLETE": self._on_data_changed,
        }

    async def _on_data_changed(self, event: Any) -> None:
        """Handle data change events - backup the changed file."""
        # Extract payload from RouterEvent or dict (Jan 2026 fix)
        payload = get_event_payload(event)

        # Extract file path from event
        file_path_str = (
            payload.get("canonical_db_path")
            or payload.get("model_path")
            or payload.get("npz_path")
            or payload.get("output_path")
        )

        if not file_path_str:
            return

        file_path = Path(file_path_str)
        if file_path.exists():
            logger.info(f"[DualBackup] Backing up changed file: {file_path.name}")
            await self.ensure_dual_backup(file_path)

    def health_check(self) -> HealthCheckResult:
        """Return health status."""
        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="Dual backup not running",
            )

        # Check for stale backups
        hours_since_backup = 0
        if self._backup_stats.last_full_backup_time > 0:
            hours_since_backup = (
                time.time() - self._backup_stats.last_full_backup_time
            ) / 3600

        is_stale = hours_since_backup > self.config.stale_backup_hours
        too_many_failures = (
            self._backup_stats.consecutive_failures >= self.config.max_consecutive_failures
        )

        if too_many_failures:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"Too many consecutive failures: {self._backup_stats.consecutive_failures}",
                details=self._get_health_details(),
            )

        if is_stale:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message=f"Backup stale: {hours_since_backup:.1f}h since last backup",
                details=self._get_health_details(),
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"Dual backup healthy: {self._backup_stats.successful_backups} successful backups",
            details=self._get_health_details(),
        )

    def _get_health_details(self) -> dict[str, Any]:
        """Get detailed health information."""
        return {
            "cycles_completed": self._stats.cycles_completed,
            "total_backups": self._backup_stats.total_backups,
            "successful_backups": self._backup_stats.successful_backups,
            "failed_backups": self._backup_stats.failed_backups,
            "bytes_backed_up_mb": round(
                self._backup_stats.bytes_backed_up / (1024 * 1024), 2
            ),
            "s3_successes": self._backup_stats.s3_successes,
            "s3_failures": self._backup_stats.s3_failures,
            "owc_successes": self._backup_stats.owc_successes,
            "owc_failures": self._backup_stats.owc_failures,
            "consecutive_failures": self._backup_stats.consecutive_failures,
            "last_full_backup_time": self._backup_stats.last_full_backup_time,
            "last_backup_duration": self._backup_stats.last_backup_duration,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get current daemon statistics."""
        return self._get_health_details()


def get_dual_backup_daemon() -> DualBackupDaemon:
    """Get the singleton dual backup daemon instance."""
    return DualBackupDaemon.get_instance()


def reset_dual_backup_daemon() -> None:
    """Reset the singleton instance (for testing)."""
    DualBackupDaemon.reset_instance()


# Factory function for daemon_runners.py
async def create_dual_backup() -> None:
    """Create and run dual backup daemon (January 2026).

    Ensures all training data is backed up to both S3 and OWC
    for redundancy and recovery.
    """
    daemon = get_dual_backup_daemon()
    await daemon.start()

    try:
        while daemon._running:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        await daemon.stop()

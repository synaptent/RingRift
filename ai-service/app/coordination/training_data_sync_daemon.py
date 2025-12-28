"""Training Data Sync Daemon - Pre-training data synchronization.

This daemon ensures training nodes have access to the best available training
data before starting training jobs. It monitors for training activity and
proactively syncs data from remote sources (OWC drive, S3).

Features:
- Pre-training data sync: Downloads best training data before training starts
- Multi-source support: OWC external drive, S3, coordinator
- Config-aware: Only syncs data for the config being trained
- Resume support: Skips already-downloaded files
- Progress tracking: Emits events for sync progress

Usage:
    from app.coordination.training_data_sync_daemon import (
        TrainingDataSyncDaemon,
        get_training_data_sync_daemon,
        sync_training_data_for_config,
    )

    # Daemon mode (background sync)
    daemon = get_training_data_sync_daemon()
    await daemon.start()

    # One-shot sync for a specific config
    result = await sync_training_data_for_config("hex8_2p")
    print(f"Synced {result.bytes_transferred} bytes from {result.source}")
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.coordination.training_data_manifest import (
    DataSource,
    OWC_BASE_PATH,
    OWC_HOST,
    OWC_SSH_KEY,
    OWC_USER,
    S3_BUCKET,
    TrainingDataEntry,
    get_training_data_manifest,
)

logger = logging.getLogger(__name__)


# Local training data directory
LOCAL_TRAINING_DIR = Path(
    os.environ.get("RINGRIFT_TRAINING_DIR", "data/training")
)

# Minimum file size improvement to trigger re-sync (10% larger)
MIN_SIZE_IMPROVEMENT_RATIO = float(
    os.environ.get("RINGRIFT_DATA_SYNC_SIZE_RATIO", "1.1")
)

# Sync timeout in seconds
SYNC_TIMEOUT = int(os.environ.get("RINGRIFT_DATA_SYNC_TIMEOUT", "1800"))  # 30 min


@dataclass
class SyncResult:
    """Result of a sync operation."""

    config_key: str
    success: bool
    source: DataSource | None = None
    source_path: str | None = None
    local_path: str | None = None
    bytes_transferred: int = 0
    duration_seconds: float = 0
    error: str | None = None
    skipped_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "config_key": self.config_key,
            "success": self.success,
            "source": self.source.value if self.source else None,
            "source_path": self.source_path,
            "local_path": self.local_path,
            "bytes_transferred": self.bytes_transferred,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "skipped_reason": self.skipped_reason,
        }


@dataclass
class TrainingDataSyncConfig:
    """Configuration for training data sync."""

    # Check interval for pending training
    check_interval_seconds: float = 60.0

    # Minimum size improvement to trigger re-sync
    min_size_improvement_ratio: float = 1.1

    # Sync timeout
    timeout_seconds: int = 1800

    # Preferred source order
    source_priority: list[DataSource] = field(
        default_factory=lambda: [DataSource.OWC, DataSource.S3, DataSource.LOCAL]
    )

    # Whether to emit events
    emit_events: bool = True

    @classmethod
    def from_env(cls) -> "TrainingDataSyncConfig":
        """Create config from environment variables."""
        return cls(
            check_interval_seconds=float(
                os.environ.get("RINGRIFT_DATA_SYNC_INTERVAL", "60")
            ),
            min_size_improvement_ratio=float(
                os.environ.get("RINGRIFT_DATA_SYNC_SIZE_RATIO", "1.1")
            ),
            timeout_seconds=int(
                os.environ.get("RINGRIFT_DATA_SYNC_TIMEOUT", "1800")
            ),
        )


async def sync_from_owc(
    entry: TrainingDataEntry,
    local_path: Path,
    timeout: int = SYNC_TIMEOUT,
) -> SyncResult:
    """Sync training data from OWC drive.

    Args:
        entry: Training data entry with OWC path
        local_path: Local destination path
        timeout: Timeout in seconds

    Returns:
        SyncResult with transfer details
    """
    start_time = time.time()
    config_key = entry.config_key

    try:
        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Build rsync command
        ssh_key_path = Path(OWC_SSH_KEY).expanduser()
        ssh_opts = f"ssh -i {ssh_key_path} -o ConnectTimeout=30"

        remote_path = f"{OWC_USER}@{OWC_HOST}:{entry.path}"
        cmd = [
            "rsync",
            "-avz",
            "--progress",
            "--partial",
            "--inplace",
            "-e", ssh_opts,
            remote_path,
            str(local_path),
        ]

        logger.info(f"Syncing {config_key} from OWC: {entry.path}")
        logger.debug(f"Command: {' '.join(cmd)}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            return SyncResult(
                config_key=config_key,
                success=False,
                source=DataSource.OWC,
                source_path=entry.path,
                error=f"Sync timed out after {timeout}s",
                duration_seconds=time.time() - start_time,
            )

        if process.returncode != 0:
            error_msg = stderr.decode().strip() if stderr else "Unknown error"
            return SyncResult(
                config_key=config_key,
                success=False,
                source=DataSource.OWC,
                source_path=entry.path,
                error=f"rsync failed: {error_msg}",
                duration_seconds=time.time() - start_time,
            )

        # Get actual file size
        bytes_transferred = local_path.stat().st_size if local_path.exists() else 0

        return SyncResult(
            config_key=config_key,
            success=True,
            source=DataSource.OWC,
            source_path=entry.path,
            local_path=str(local_path),
            bytes_transferred=bytes_transferred,
            duration_seconds=time.time() - start_time,
        )

    except (OSError, IOError, subprocess.SubprocessError, PermissionError) as e:
        # File system, subprocess, or permission errors during sync
        return SyncResult(
            config_key=config_key,
            success=False,
            source=DataSource.OWC,
            source_path=entry.path,
            error=str(e),
            duration_seconds=time.time() - start_time,
        )


async def sync_from_s3(
    entry: TrainingDataEntry,
    local_path: Path,
    timeout: int = SYNC_TIMEOUT,
) -> SyncResult:
    """Sync training data from S3.

    Args:
        entry: Training data entry with S3 URI
        local_path: Local destination path
        timeout: Timeout in seconds

    Returns:
        SyncResult with transfer details
    """
    start_time = time.time()
    config_key = entry.config_key

    try:
        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Build aws s3 cp command
        cmd = [
            "aws", "s3", "cp",
            entry.path,  # S3 URI
            str(local_path),
            "--only-show-errors",
        ]

        logger.info(f"Syncing {config_key} from S3: {entry.path}")
        logger.debug(f"Command: {' '.join(cmd)}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            return SyncResult(
                config_key=config_key,
                success=False,
                source=DataSource.S3,
                source_path=entry.path,
                error=f"S3 sync timed out after {timeout}s",
                duration_seconds=time.time() - start_time,
            )

        if process.returncode != 0:
            error_msg = stderr.decode().strip() if stderr else "Unknown error"
            return SyncResult(
                config_key=config_key,
                success=False,
                source=DataSource.S3,
                source_path=entry.path,
                error=f"aws s3 cp failed: {error_msg}",
                duration_seconds=time.time() - start_time,
            )

        # Get actual file size
        bytes_transferred = local_path.stat().st_size if local_path.exists() else 0

        return SyncResult(
            config_key=config_key,
            success=True,
            source=DataSource.S3,
            source_path=entry.path,
            local_path=str(local_path),
            bytes_transferred=bytes_transferred,
            duration_seconds=time.time() - start_time,
        )

    except (OSError, IOError, subprocess.SubprocessError, PermissionError) as e:
        # File system, subprocess, or permission errors during S3 sync
        return SyncResult(
            config_key=config_key,
            success=False,
            source=DataSource.S3,
            source_path=entry.path,
            error=str(e),
            duration_seconds=time.time() - start_time,
        )


async def sync_training_data_for_config(
    config_key: str,
    local_dir: Path | None = None,
    force: bool = False,
    config: TrainingDataSyncConfig | None = None,
) -> SyncResult:
    """Sync the best available training data for a config.

    This is the main entry point for pre-training data sync. It:
    1. Checks the manifest for available data
    2. Compares with local data
    3. Downloads better data if available

    Args:
        config_key: Config key (e.g., 'hex8_2p')
        local_dir: Local training data directory (default: data/training)
        force: Force re-download even if local file is larger
        config: Sync configuration

    Returns:
        SyncResult with transfer details
    """
    local_dir = local_dir or LOCAL_TRAINING_DIR
    config = config or TrainingDataSyncConfig.from_env()
    local_path = local_dir / f"{config_key}.npz"

    # Get manifest
    manifest = await get_training_data_manifest()

    # Get best remote data
    best_remote = manifest.get_best_data(config_key, min_size_mb=1)
    if not best_remote:
        return SyncResult(
            config_key=config_key,
            success=False,
            error="No training data found in manifest",
        )

    # Skip if local source is the best
    if best_remote.source == DataSource.LOCAL:
        return SyncResult(
            config_key=config_key,
            success=True,
            source=DataSource.LOCAL,
            source_path=best_remote.path,
            local_path=best_remote.path,
            skipped_reason="Best data is already local",
        )

    # Check if local file exists and is good enough
    if local_path.exists() and not force:
        local_size = local_path.stat().st_size
        size_ratio = best_remote.size_bytes / local_size if local_size > 0 else float("inf")

        if size_ratio < config.min_size_improvement_ratio:
            return SyncResult(
                config_key=config_key,
                success=True,
                source=DataSource.LOCAL,
                local_path=str(local_path),
                bytes_transferred=0,
                skipped_reason=(
                    f"Local file ({local_size / 1024 / 1024:.1f}MB) is "
                    f"within {config.min_size_improvement_ratio}x of remote "
                    f"({best_remote.size_bytes / 1024 / 1024:.1f}MB)"
                ),
            )

    # Sync from remote source
    logger.info(
        f"Syncing {config_key} from {best_remote.source.value}: "
        f"{best_remote.size_mb:.1f}MB"
    )

    if best_remote.source == DataSource.OWC:
        return await sync_from_owc(
            best_remote, local_path, timeout=config.timeout_seconds
        )
    elif best_remote.source == DataSource.S3:
        return await sync_from_s3(
            best_remote, local_path, timeout=config.timeout_seconds
        )
    else:
        return SyncResult(
            config_key=config_key,
            success=False,
            error=f"Unsupported source: {best_remote.source}",
        )


@dataclass
class TrainingDataSyncDaemon:
    """Daemon that proactively syncs training data.

    This daemon monitors for pending/active training jobs and ensures
    the best training data is available before training starts.
    """

    config: TrainingDataSyncConfig = field(
        default_factory=TrainingDataSyncConfig.from_env
    )
    _running: bool = field(default=False, repr=False)
    _task: asyncio.Task | None = field(default=None, repr=False)
    _stats: dict[str, Any] = field(default_factory=dict, repr=False)

    async def start(self) -> None:
        """Start the daemon."""
        if self._running:
            logger.warning("TrainingDataSyncDaemon already running")
            return

        self._running = True
        self._stats = {
            "started_at": datetime.now(tz=timezone.utc).isoformat(),
            "syncs_completed": 0,
            "syncs_failed": 0,
            "bytes_transferred": 0,
        }
        self._task = asyncio.create_task(self._run_loop())
        logger.info("TrainingDataSyncDaemon started")

    async def stop(self) -> None:
        """Stop the daemon."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("TrainingDataSyncDaemon stopped")

    async def _run_loop(self) -> None:
        """Main daemon loop."""
        while self._running:
            try:
                # Check for pending training jobs
                pending = await self._get_pending_training_configs()

                for config_key in pending:
                    if not self._running:
                        break

                    result = await sync_training_data_for_config(
                        config_key, config=self.config
                    )

                    if result.success:
                        self._stats["syncs_completed"] += 1
                        self._stats["bytes_transferred"] += result.bytes_transferred
                        if result.bytes_transferred > 0:
                            logger.info(
                                f"Synced {config_key}: "
                                f"{result.bytes_transferred / 1024 / 1024:.1f}MB "
                                f"from {result.source.value}"
                            )
                    else:
                        self._stats["syncs_failed"] += 1
                        if result.error:
                            logger.warning(f"Failed to sync {config_key}: {result.error}")

                    # Emit event if configured
                    if self.config.emit_events:
                        await self._emit_sync_event(result)

            except (OSError, IOError, asyncio.CancelledError) as e:
                # File/network errors or task cancellation in sync loop
                logger.exception(f"Error in sync loop: {e}")

            # Wait before next check
            await asyncio.sleep(self.config.check_interval_seconds)

    async def _get_pending_training_configs(self) -> list[str]:
        """Get configs with pending or active training jobs.

        This checks:
        1. Local training processes
        2. P2P work queue for training jobs
        """
        configs = set()

        # Check for local training processes
        try:
            result = subprocess.run(
                ["pgrep", "-af", "python.*train"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if "--board-type" in line:
                        # Extract board type and num players
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "--board-type" and i + 1 < len(parts):
                                board_type = parts[i + 1]
                                # Look for --num-players
                                for j, p2 in enumerate(parts):
                                    if p2 == "--num-players" and j + 1 < len(parts):
                                        num_players = parts[j + 1]
                                        configs.add(f"{board_type}_{num_players}p")
        except (OSError, subprocess.SubprocessError, FileNotFoundError):
            # pgrep may not exist or may fail on some systems
            pass

        # Check P2P work queue for training jobs
        try:
            from app.coordination.work_queue import get_work_queue

            queue = get_work_queue()
            for item in queue.get_pending_items():
                if item.job_type == "training" and item.config_key:
                    configs.add(item.config_key)
        except (ImportError, AttributeError, OSError):
            # Work queue module may not be available or DB may be inaccessible
            pass

        return list(configs)

    async def _emit_sync_event(self, result: SyncResult) -> None:
        """Emit sync event for coordination."""
        try:
            from app.distributed.data_events import (
                DataEventType,
                emit_data_event,
            )

            event_type = (
                DataEventType.DATA_SYNC_COMPLETED
                if result.success
                else DataEventType.DATA_SYNC_FAILED
            )
            emit_data_event(
                event_type,
                config_key=result.config_key,
                source=result.source.value if result.source else None,
                bytes_transferred=result.bytes_transferred,
                duration_seconds=result.duration_seconds,
                error=result.error,
            )
        except Exception as e:
            logger.debug(f"Failed to emit sync event: {e}")

    def health_check(self) -> "HealthCheckResult":
        """Return health check status.

        Returns:
            HealthCheckResult for DaemonManager integration.
        """
        from app.coordination.contracts import CoordinatorStatus, HealthCheckResult
        from app.coordination.health_check_helper import HealthCheckHelper

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="TrainingDataSyncDaemon is not running",
                details={"stats": self._stats},
            )

        # Check for recent activity
        syncs_completed = self._stats.get("syncs_completed", 0)
        syncs_failed = self._stats.get("syncs_failed", 0)

        # Check error rate using HealthCheckHelper (degraded if >50% failure rate)
        is_healthy, msg = HealthCheckHelper.check_error_rate(
            errors=syncs_failed,
            cycles=syncs_completed + syncs_failed,
            threshold=0.5,
        )
        error_rate = syncs_failed / max(syncs_completed + syncs_failed, 1)

        if not is_healthy:
            return HealthCheckResult(
                healthy=True,  # Still running but degraded
                status=CoordinatorStatus.DEGRADED,
                message=f"High sync {msg}",
                details={
                    "running": True,
                    "error_rate": error_rate,
                    "stats": self._stats,
                },
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"TrainingDataSyncDaemon healthy: {syncs_completed} syncs, {self._stats.get('bytes_transferred', 0) / 1024 / 1024:.1f}MB transferred",
            details={
                "running": True,
                "error_rate": error_rate,
                "stats": self._stats,
            },
        )


# Singleton instance
_daemon_instance: TrainingDataSyncDaemon | None = None


def get_training_data_sync_daemon() -> TrainingDataSyncDaemon:
    """Get or create the singleton daemon instance."""
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = TrainingDataSyncDaemon()
    return _daemon_instance


def reset_training_data_sync_daemon() -> None:
    """Reset the singleton instance (for testing)."""
    global _daemon_instance
    _daemon_instance = None

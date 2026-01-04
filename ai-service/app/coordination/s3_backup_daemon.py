"""S3 Backup Daemon - Automatic S3 backup after model promotion.

This daemon watches for MODEL_PROMOTED events and automatically backs up
promoted models and related data to S3 for disaster recovery.

Architecture:
    1. Subscribes to MODEL_PROMOTED events from event_router
    2. Triggers S3 backup of the promoted model
    3. Optionally backs up game databases and state files
    4. Emits S3_BACKUP_COMPLETED event when done

Usage:
    # As standalone daemon
    python -m app.coordination.s3_backup_daemon

    # Via DaemonManager
    manager.register_factory(DaemonType.S3_BACKUP, daemon.run)

Configuration:
    Set RINGRIFT_S3_BUCKET environment variable.
    Default bucket: ringrift-models-20251214

December 2025: Created as part of Phase 2 data resilience improvements.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.utils.retry import RetryConfig

logger = logging.getLogger(__name__)

# Add parent to path for imports
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class S3BackupConfig:
    """Configuration for S3 backup daemon."""

    # S3 settings
    s3_bucket: str = field(
        default_factory=lambda: os.getenv("RINGRIFT_S3_BUCKET", "ringrift-models-20251214")
    )
    aws_region: str = field(
        default_factory=lambda: os.getenv("AWS_REGION", "us-east-1")
    )

    # Backup settings
    backup_timeout_seconds: float = 600.0  # 10 minute timeout for backup
    backup_models: bool = True
    # Dec 2025: Enabled database backup for disaster recovery
    # Only backs up canonical_*.db files (not full selfplay DBs)
    backup_databases: bool = True
    backup_state: bool = True

    # Retry settings
    retry_count: int = 3
    retry_delay_seconds: float = 30.0

    # Event settings
    emit_completion_event: bool = True

    # Debounce settings (avoid backing up every promotion in rapid succession)
    debounce_seconds: float = 60.0  # Wait 60s after promotion before backup
    max_pending_before_immediate: int = 5  # Backup immediately if 5+ pending


@dataclass
class BackupResult:
    """Result of a backup operation."""

    success: bool
    uploaded_count: int
    deleted_count: int
    error_message: str = ""
    duration_seconds: float = 0.0


class S3BackupDaemon(HandlerBase):
    """Daemon that automatically backs up to S3 after model promotion.

    Watches for MODEL_PROMOTED events and syncs models (and optionally
    databases) to S3 for disaster recovery.

    This ensures that promoted models are safely stored off-cluster
    and can be recovered if all cluster nodes are lost.

    Jan 2026: Migrated to HandlerBase for unified lifecycle management.
    """

    def __init__(self, config: S3BackupConfig | None = None):
        self._daemon_config = config or S3BackupConfig()
        super().__init__(
            name="S3BackupDaemon",
            config={"s3_bucket": self._daemon_config.s3_bucket},
            cycle_interval=30.0,  # Check every 30s (matches previous sleep)
        )
        self._last_backup_time: float = 0.0
        self._pending_promotions: list[dict[str, Any]] = []
        self._backup_lock = asyncio.Lock()

        # Thread-safe lock for _pending_promotions
        self._pending_lock = threading.Lock()

        # Event-based wake-up
        self._pending_event: asyncio.Event | None = None

        # Metrics
        self._events_processed: int = 0
        self._successful_backups: int = 0
        self._failed_backups: int = 0
        self._total_files_uploaded: int = 0

    @property
    def config(self) -> S3BackupConfig:
        """Backward-compatible config access."""
        return self._daemon_config

    def health_check(self) -> HealthCheckResult:
        """Check daemon health for CoordinatorProtocol compliance.

        December 2025: Updated to return HealthCheckResult instead of bool.
        Jan 2026: Migrated to HandlerBase.
        """
        from app.coordination.protocols import CoordinatorStatus

        if not self._running:
            return HealthCheckResult(
                healthy=True,  # Stopped is not unhealthy
                status=CoordinatorStatus.STOPPED,
                message="S3BackupDaemon not running",
            )

        # Check if we have excessive pending backups (indicates processing stall)
        with self._pending_lock:
            pending_count = len(self._pending_promotions)

        # If we have > 10 pending promotions, something is stuck
        if pending_count > 10:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"Stalled: {pending_count} pending promotions",
            )

        # Check if last backup was too long ago (if we have pending items)
        if pending_count > 0 and self._last_backup_time > 0:
            time_since_backup = time.time() - self._last_backup_time
            # If we have pending items and haven't backed up in 30 minutes, unhealthy
            if time_since_backup > 1800:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"Stalled: {time_since_backup:.0f}s since backup, {pending_count} pending",
                )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"Healthy (backups: {self._successful_backups}, pending: {pending_count})",
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get daemon metrics."""
        uptime = time.time() - self._stats.started_at if self._stats.started_at > 0 else 0.0
        return {
            "name": self._name,
            "running": self._running,
            "uptime_seconds": uptime,
            "events_processed": self._events_processed,
            "pending_promotions": len(self._pending_promotions),
            "successful_backups": self._successful_backups,
            "failed_backups": self._failed_backups,
            "total_files_uploaded": self._total_files_uploaded,
            "last_backup_time": self._last_backup_time,
            "s3_bucket": self._daemon_config.s3_bucket,
        }

    async def _on_start(self) -> None:
        """Initialize daemon on startup.

        Jan 2026: Migrated to HandlerBase lifecycle hook.
        """
        logger.info(f"S3BackupDaemon starting (bucket: {self._daemon_config.s3_bucket})")

        # Initialize event for wake-up
        self._pending_event = asyncio.Event()

        # Subscribe to MODEL_PROMOTED events
        try:
            from app.coordination.event_router import subscribe
            from app.events.types import RingRiftEventType

            subscribe(RingRiftEventType.MODEL_PROMOTED, self._on_model_promoted)
            logger.info("Subscribed to MODEL_PROMOTED events via event_router")

        except ImportError as e:
            logger.warning(f"event_router not available ({e}), will poll for changes")
        except Exception as e:
            logger.error(f"Failed to subscribe to MODEL_PROMOTED: {e}")

    async def _run_cycle(self) -> None:
        """Main cycle - check for pending backups.

        Jan 2026: Migrated from start() loop to HandlerBase _run_cycle().
        """
        try:
            # Check for pending backups with debounce
            await self._check_pending_backups()

            # Wait for event or timeout (event-based wake-up optimization)
            if self._pending_event is not None:
                try:
                    await asyncio.wait_for(
                        self._pending_event.wait(),
                        timeout=self._cycle_interval
                    )
                    self._pending_event.clear()
                except asyncio.TimeoutError:
                    pass

        except asyncio.CancelledError:
            raise  # Let HandlerBase handle cancellation
        except Exception as e:
            logger.error(f"Error in S3 backup daemon cycle: {e}")
            self._record_error(f"Cycle error: {e}", e)

    async def _on_stop(self) -> None:
        """Cleanup on shutdown.

        Jan 2026: Migrated to HandlerBase lifecycle hook.
        """
        # Process any remaining pending backups
        if self._pending_promotions:
            logger.info(
                f"Processing {len(self._pending_promotions)} pending backups before shutdown"
            )
            await self._run_backup()
        logger.info("S3BackupDaemon stopped")

    async def run(self) -> None:
        """Run the daemon main loop (blocking).

        Jan 2026: Added for backward compatibility after HandlerBase migration.
        Blocks until the daemon is stopped.
        """
        await self.start()
        # Wait for the main loop task to complete
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Call cleanup hooks when done
        await self._on_stop()

    def _on_model_promoted(self, event: dict[str, Any] | Any) -> None:
        """Handle MODEL_PROMOTED event (sync callback)."""
        # Handle both RouterEvent and dict payloads
        payload = getattr(event, "payload", event) if hasattr(event, "payload") else event
        promotion_info = {
            "model_path": payload.get("model_path"),
            "model_id": payload.get("model_id"),
            "board_type": payload.get("board_type"),
            "num_players": payload.get("num_players"),
            "elo": payload.get("elo"),
            "timestamp": time.time(),
        }
        logger.info(f"Received MODEL_PROMOTED event for S3 backup: {promotion_info}")

        # Thread-safe append
        with self._pending_lock:
            self._pending_promotions.append(promotion_info)
        self._events_processed += 1

        # Wake up main loop
        if self._pending_event is not None:
            self._pending_event.set()

    async def _check_pending_backups(self) -> None:
        """Check if we should run a backup based on pending promotions."""
        with self._pending_lock:
            pending_count = len(self._pending_promotions)
            if pending_count == 0:
                return

            oldest_promotion_time = min(
                p.get("timestamp", time.time()) for p in self._pending_promotions
            )

        # Check if we should backup now
        time_since_oldest = time.time() - oldest_promotion_time
        should_backup = (
            # Debounce period elapsed
            time_since_oldest >= self.config.debounce_seconds
            # Or too many pending
            or pending_count >= self.config.max_pending_before_immediate
        )

        if should_backup:
            await self._process_pending_backups()

    async def _process_pending_backups(self) -> None:
        """Process pending model promotions with S3 backup."""
        async with self._backup_lock:
            # Thread-safe extract pending
            with self._pending_lock:
                if not self._pending_promotions:
                    return
                promotions = self._pending_promotions.copy()
                self._pending_promotions.clear()

            logger.info(f"Running S3 backup for {len(promotions)} promoted models")

            # Run backup with retry (Jan 3, 2026: Migrated to RetryConfig)
            retry_config = RetryConfig(
                max_attempts=self.config.retry_count,
                base_delay=self.config.retry_delay_seconds,
                exponential=False,  # Keep original linear delay behavior
            )

            for attempt in retry_config.attempts():
                try:
                    result = await self._run_backup()
                    if result.success:
                        self._last_backup_time = time.time()
                        self._successful_backups += 1
                        self._total_files_uploaded += result.uploaded_count
                        logger.info(
                            f"S3 backup completed: {result.uploaded_count} files uploaded "
                            f"in {result.duration_seconds:.1f}s"
                        )

                        # Emit completion event
                        if self.config.emit_completion_event:
                            await self._emit_backup_complete(promotions, result)
                        return
                    else:
                        logger.warning(f"S3 backup attempt {attempt.number} failed")

                except Exception as e:
                    logger.error(f"S3 backup attempt {attempt.number} failed: {e}")

                if attempt.should_retry:
                    await attempt.wait_async()

            # All attempts failed
            self._failed_backups += 1
            logger.error(
                f"S3 backup failed after {retry_config.max_attempts} attempts"
            )

    async def _run_backup(self) -> BackupResult:
        """Execute S3 backup using s3_backup.py script."""
        start_time = time.time()

        backup_script = ROOT / "scripts" / "s3_backup.py"
        if not backup_script.exists():
            return BackupResult(
                success=False,
                uploaded_count=0,
                deleted_count=0,
                error_message=f"Backup script not found: {backup_script}",
            )

        # Build command based on config
        cmd = [sys.executable, str(backup_script)]

        if self.config.backup_models and not self.config.backup_databases:
            cmd.append("--models-only")
        elif self.config.backup_databases and not self.config.backup_models:
            cmd.append("--databases-only")
        # else: full backup (models, databases, state)

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(ROOT),
                env={
                    **os.environ,
                    "PYTHONPATH": str(ROOT),
                    "RINGRIFT_S3_BUCKET": self.config.s3_bucket,
                    "AWS_REGION": self.config.aws_region,
                },
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.backup_timeout_seconds,
            )

            duration = time.time() - start_time
            output = stdout.decode()

            if process.returncode == 0:
                # Parse output for counts
                uploaded = output.count("upload:")
                deleted = output.count("delete:")

                return BackupResult(
                    success=True,
                    uploaded_count=uploaded,
                    deleted_count=deleted,
                    duration_seconds=duration,
                )
            else:
                return BackupResult(
                    success=False,
                    uploaded_count=0,
                    deleted_count=0,
                    error_message=stderr.decode()[-500:] if stderr else "Unknown error",
                    duration_seconds=duration,
                )

        except asyncio.TimeoutError:
            return BackupResult(
                success=False,
                uploaded_count=0,
                deleted_count=0,
                error_message=f"Backup timed out after {self.config.backup_timeout_seconds}s",
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            return BackupResult(
                success=False,
                uploaded_count=0,
                deleted_count=0,
                error_message=str(e),
                duration_seconds=time.time() - start_time,
            )

    async def _emit_backup_complete(
        self,
        promotions: list[dict[str, Any]],
        result: BackupResult,
    ) -> None:
        """Emit S3_BACKUP_COMPLETED event."""
        try:
            from app.coordination.event_router import emit

            await emit(
                event_type="S3_BACKUP_COMPLETED",
                data={
                    "promotions": promotions,
                    "uploaded_count": result.uploaded_count,
                    "deleted_count": result.deleted_count,
                    "duration_seconds": result.duration_seconds,
                    "bucket": self.config.s3_bucket,
                    "timestamp": time.time(),
                },
            )
            logger.debug("Emitted S3_BACKUP_COMPLETED event")
        except Exception as e:
            logger.warning(f"Failed to emit backup complete event: {e}")


# Daemon adapter for DaemonManager integration
class S3BackupDaemonAdapter:
    """Adapter for integrating with DaemonManager."""

    def __init__(self, config: S3BackupConfig | None = None):
        self.config = config
        self._daemon: S3BackupDaemon | None = None

    @property
    def daemon_type(self) -> str:
        return "S3_BACKUP"

    @property
    def depends_on(self) -> list[str]:
        return ["MODEL_DISTRIBUTION"]  # Run after model distribution

    async def run(self) -> None:
        """Run the daemon (DaemonManager entry point)."""
        self._daemon = S3BackupDaemon(self.config)
        await self._daemon.start()

    async def stop(self) -> None:
        """Stop the daemon."""
        if self._daemon:
            await self._daemon.stop()

    def health_check(self) -> HealthCheckResult:
        """Check adapter health for DaemonManager integration.

        December 2025: Added for CoordinatorProtocol compliance.
        Jan 2026: Uses HandlerBase's HealthCheckResult.
        """
        from app.coordination.protocols import CoordinatorStatus

        if self._daemon is None:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="S3BackupDaemonAdapter not started",
                details={"config_set": self.config is not None},
            )

        # Delegate to underlying daemon
        return self._daemon.health_check()


async def main() -> None:
    """Run daemon standalone."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    daemon = S3BackupDaemon()
    try:
        await daemon.run()  # run() blocks until stopped
    except KeyboardInterrupt:
        await daemon.stop()


# Module-level singleton accessors
_daemon_instance: S3BackupDaemon | None = None


def get_s3_backup_daemon(config: S3BackupConfig | None = None) -> S3BackupDaemon:
    """Get or create the S3BackupDaemon singleton.

    Jan 2026: Added for HandlerBase singleton pattern.
    """
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = S3BackupDaemon(config)
    return _daemon_instance


def reset_s3_backup_daemon() -> None:
    """Reset the S3BackupDaemon singleton (for testing)."""
    global _daemon_instance
    _daemon_instance = None


if __name__ == "__main__":
    asyncio.run(main())

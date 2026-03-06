"""S3 Push Daemon - Backs up training data to AWS S3 (January 2026).

This daemon periodically pushes all game databases and NPZ files to S3 for:
- Backup redundancy across cloud providers
- Cross-region data availability
- Training data preservation for cluster nodes

The daemon only pushes files that have been modified since the last push,
minimizing bandwidth and S3 costs.

Usage:
    from app.coordination.s3_push_daemon import S3PushDaemon, get_s3_push_daemon

    daemon = get_s3_push_daemon()
    await daemon.start()

Environment Variables:
    RINGRIFT_S3_BUCKET: S3 bucket name (default: ringrift-models-20251214)
    RINGRIFT_S3_REGION: AWS region (default: us-east-1)
    RINGRIFT_S3_PUSH_INTERVAL: Push interval in seconds (default: 600)
    RINGRIFT_S3_PUSH_ENABLED: Enable/disable daemon (default: true)
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.utils.retry import RetryConfig
from app.utils.paths import AWS_CLI

logger = logging.getLogger(__name__)

# January 2026: Retry configuration for S3 uploads
S3_RETRY_CONFIG = RetryConfig(
    max_attempts=5,       # Up to 5 attempts
    base_delay=2.0,       # Start with 2 second delay
    max_delay=60.0,       # Cap at 60 seconds
    exponential=True,     # Exponential backoff
    jitter=0.2,           # 20% jitter to prevent thundering herd
)


@dataclass
class S3PushConfig:
    """Configuration for S3 push daemon."""

    bucket: str = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_S3_BUCKET", "ringrift-models-20251214"
        )
    )
    region: str = field(
        default_factory=lambda: os.environ.get("RINGRIFT_S3_REGION", "us-east-1")
    )
    push_interval: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_S3_PUSH_INTERVAL", "600")
        )
    )
    enabled: bool = field(
        default_factory=lambda: os.environ.get("RINGRIFT_S3_PUSH_ENABLED", "true").lower()
        == "true"
    )

    # Storage class for S3 uploads (STANDARD_IA is cost-effective for backups)
    storage_class: str = "STANDARD_IA"

    # Paths to monitor for changes
    game_db_paths: list[str] = field(
        default_factory=lambda: [
            "data/games",
            "data/selfplay",
        ]
    )
    training_data_paths: list[str] = field(
        default_factory=lambda: [
            "data/training",
        ]
    )
    model_paths: list[str] = field(
        default_factory=lambda: [
            "models",
        ]
    )


@dataclass
class S3PushStats:
    """Statistics for S3 push operations."""

    total_files_pushed: int = 0
    total_bytes_pushed: int = 0
    last_push_time: float = 0.0
    push_errors: int = 0
    last_error: str | None = None


class S3PushDaemon(HandlerBase):
    """Daemon that pushes training data to S3 for backup.

    Monitors local game databases, NPZ training files, and models,
    pushing any modified files to S3 for redundancy and cluster access.
    """

    _instance: S3PushDaemon | None = None

    def __init__(self, config: S3PushConfig | None = None):
        """Initialize S3 push daemon.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or S3PushConfig()
        super().__init__(name="s3_push", cycle_interval=self.config.push_interval)

        self._push_stats = S3PushStats()
        self._last_push_times: dict[str, float] = {}  # path -> mtime at last push
        self._base_path = Path(os.environ.get("RINGRIFT_BASE_PATH", "."))

    @classmethod
    def get_instance(cls) -> S3PushDaemon:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    async def _run_cycle(self) -> None:
        """Run one push cycle."""
        if not self.config.enabled:
            logger.debug("[S3PushDaemon] Disabled via config, skipping cycle")
            return

        if not self._check_aws_credentials():
            logger.warning("[S3PushDaemon] AWS credentials not configured, skipping")
            return

        try:
            # Push canonical databases
            await self._push_canonical_databases()

            # Push NPZ training files
            await self._push_training_files()

            # Push models (less frequently, they're larger)
            if self._should_push_models():
                await self._push_models()

            self._push_stats.last_push_time = time.time()
            logger.info(
                f"[S3PushDaemon] Cycle complete: "
                f"{self._push_stats.total_files_pushed} files pushed"
            )

        except Exception as e:
            self._push_stats.push_errors += 1
            self._push_stats.last_error = str(e)
            logger.error(f"[S3PushDaemon] Push cycle failed: {e}")

    def _check_aws_credentials(self) -> bool:
        """Check if AWS credentials are available."""
        # Check environment variables
        if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get(
            "AWS_SECRET_ACCESS_KEY"
        ):
            return True

        # Check AWS config file
        aws_config = Path.home() / ".aws" / "credentials"
        if aws_config.exists():
            return True

        return False

    async def _push_canonical_databases(self) -> None:
        """Push canonical game databases to S3."""
        games_dir = self._base_path / "data" / "games"
        if not games_dir.exists():
            return

        for db_path in games_dir.glob("canonical_*.db"):
            await self._push_if_modified(
                db_path, f"consolidated/games/{db_path.name}"
            )

    async def _push_training_files(self) -> None:
        """Push NPZ training files to S3."""
        training_dir = self._base_path / "data" / "training"
        if not training_dir.exists():
            return

        for npz_path in training_dir.glob("*.npz"):
            await self._push_if_modified(
                npz_path, f"consolidated/training/{npz_path.name}"
            )

    async def _push_models(self) -> None:
        """Push model checkpoints to S3."""
        models_dir = self._base_path / "models"
        if not models_dir.exists():
            return

        for model_path in models_dir.glob("canonical_*.pth"):
            await self._push_if_modified(model_path, f"models/{model_path.name}")

    def _should_push_models(self) -> bool:
        """Determine if models should be pushed this cycle.

        Models are large, so we push them less frequently (every 6 cycles).
        """
        cycle_count = getattr(self, "_cycle_count", 0)
        self._cycle_count = cycle_count + 1
        return cycle_count % 6 == 0

    async def _push_if_modified(self, local_path: Path, s3_key: str) -> bool:
        """Push file to S3 if it has been modified since last push.

        January 2026: Added exponential backoff retry (up to 5 attempts).

        Args:
            local_path: Local file path
            s3_key: S3 object key

        Returns:
            True if file was pushed, False if skipped (not modified) or failed
        """
        if not local_path.exists():
            return False

        try:
            mtime = local_path.stat().st_mtime
            last_push = self._last_push_times.get(str(local_path), 0)

            if mtime <= last_push:
                logger.debug(f"[S3PushDaemon] Skipping {local_path.name} (not modified)")
                return False

            s3_uri = f"s3://{self.config.bucket}/{s3_key}"
            file_size = local_path.stat().st_size

            # Mar 6, 2026: Don't overwrite S3 files with smaller versions.
            # After daemon restart, _last_push_times is cleared, causing all
            # files to be re-pushed. If a local NPZ was partially re-exported
            # (e.g. with --min-elo filter), pushing the smaller file would
            # clobber the good training data on S3 that GPU nodes depend on.
            if s3_key.startswith("consolidated/training/") and file_size > 0:
                try:
                    s3_size = await self._get_s3_file_size(s3_uri)
                    if s3_size > 0 and file_size < s3_size * 0.8:
                        logger.warning(
                            f"[S3PushDaemon] Skipping {local_path.name}: local "
                            f"({file_size / 1e6:.1f}MB) < S3 ({s3_size / 1e6:.1f}MB). "
                            f"Won't overwrite larger training data."
                        )
                        self._last_push_times[str(local_path)] = mtime
                        return False
                except Exception:
                    pass  # Can't check S3 size, proceed with push

            # January 2026: Retry loop with exponential backoff
            last_error: str | None = None
            for attempt in S3_RETRY_CONFIG.attempts():
                try:
                    # Run aws s3 cp in a thread to not block
                    result = await asyncio.to_thread(
                        subprocess.run,
                        [
                            "aws",
                            "s3",
                            "cp",
                            str(local_path),
                            s3_uri,
                            "--storage-class",
                            self.config.storage_class,
                            "--region",
                            self.config.region,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=300,  # 5 minute timeout for large files
                    )

                    if result.returncode == 0:
                        self._last_push_times[str(local_path)] = mtime
                        self._push_stats.total_files_pushed += 1
                        self._push_stats.total_bytes_pushed += file_size
                        if attempt.number > 1:
                            logger.info(
                                f"[S3PushDaemon] Pushed {local_path.name} to {s3_uri} "
                                f"({file_size / (1024*1024):.1f} MB) - succeeded on attempt {attempt.number}"
                            )
                        else:
                            logger.info(
                                f"[S3PushDaemon] Pushed {local_path.name} to {s3_uri} "
                                f"({file_size / (1024*1024):.1f} MB)"
                            )
                        return True

                    # Non-zero return code - retryable error
                    last_error = result.stderr or f"aws s3 cp failed with code {result.returncode}"

                    # Check if error is retryable (transient errors)
                    is_retryable = any(
                        err in last_error.lower()
                        for err in [
                            "service unavailable", "timeout", "connection",
                            "throttl", "slowdown", "internal error", "503",
                            "500", "network", "socket", "reset",
                        ]
                    )

                    if not is_retryable:
                        # Permanent error (e.g., access denied, no such bucket)
                        logger.warning(
                            f"[S3PushDaemon] Permanent error pushing {local_path.name}: {last_error}"
                        )
                        self._push_stats.push_errors += 1
                        self._push_stats.last_error = last_error
                        return False

                    if attempt.should_retry:
                        delay = attempt.delay
                        logger.warning(
                            f"[S3PushDaemon] Push {local_path.name} failed "
                            f"(attempt {attempt.number}/{S3_RETRY_CONFIG.max_attempts}): {last_error[:100]}. "
                            f"Retrying in {delay:.1f}s"
                        )
                        await attempt.wait_async()
                    else:
                        # Final attempt failed
                        logger.warning(
                            f"[S3PushDaemon] Push {local_path.name} failed after "
                            f"{S3_RETRY_CONFIG.max_attempts} attempts: {last_error[:200]}"
                        )

                except subprocess.TimeoutExpired:
                    last_error = "Timeout (>5 min)"
                    if attempt.should_retry:
                        delay = attempt.delay
                        logger.warning(
                            f"[S3PushDaemon] Push {local_path.name} timed out "
                            f"(attempt {attempt.number}/{S3_RETRY_CONFIG.max_attempts}). "
                            f"Retrying in {delay:.1f}s"
                        )
                        await attempt.wait_async()
                    else:
                        logger.warning(
                            f"[S3PushDaemon] Push {local_path.name} timed out after "
                            f"{S3_RETRY_CONFIG.max_attempts} attempts"
                        )

            # All retries exhausted
            self._push_stats.push_errors += 1
            self._push_stats.last_error = last_error
            return False

        except (OSError, ValueError) as e:
            # Non-retryable local errors (file not found, invalid path, etc.)
            logger.warning(f"[S3PushDaemon] Local error for {local_path.name}: {e}")
            self._push_stats.push_errors += 1
            self._push_stats.last_error = str(e)
            return False

    async def _get_s3_file_size(self, s3_uri: str) -> int:
        """Get the size of a file on S3. Returns 0 if not found or error."""
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["aws", "s3", "ls", s3_uri, "--region", self.config.region],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Output format: "2026-03-06 03:54:48  221871742 hex8_2p.npz"
                parts = result.stdout.strip().split()
                if len(parts) >= 3:
                    return int(parts[2])
        except Exception:
            pass
        return 0

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Get event subscriptions for this daemon.

        Subscribes to data events to trigger immediate pushes.
        """
        return {
            "DATA_SYNC_COMPLETED": self._on_data_sync_completed,
            "TRAINING_COMPLETED": self._on_training_completed,
            "NPZ_EXPORT_COMPLETE": self._on_npz_export_complete,
        }

    async def _on_data_sync_completed(self, event: Any) -> None:
        """Handle data sync completion - push synced data to S3."""
        # Jan 2026: Fix RouterEvent handling - extract payload first
        payload = getattr(event, "payload", {}) or (event if isinstance(event, dict) else {})
        if payload.get("needs_s3_backup"):
            db_path_str = payload.get("db_path")
            if db_path_str:
                db_path = Path(db_path_str)
                if db_path.exists():
                    await self._push_if_modified(
                        db_path, f"consolidated/games/{db_path.name}"
                    )

    async def _on_training_completed(self, event: Any) -> None:
        """Handle training completion - push model to S3."""
        # Jan 2026: Fix RouterEvent handling - extract payload first
        payload = getattr(event, "payload", {}) or (event if isinstance(event, dict) else {})
        model_path_str = payload.get("model_path")
        if model_path_str:
            model_path = Path(model_path_str)
            if model_path.exists():
                await self._push_if_modified(model_path, f"models/{model_path.name}")

    async def _on_npz_export_complete(self, event: Any) -> None:
        """Handle NPZ export completion - push training data to S3."""
        # Jan 2026: Fix RouterEvent handling - extract payload first
        payload = getattr(event, "payload", {}) or (event if isinstance(event, dict) else {})
        npz_path_str = payload.get("npz_path") or payload.get("output_path")
        if npz_path_str:
            npz_path = Path(npz_path_str)
            if npz_path.exists():
                await self._push_if_modified(
                    npz_path, f"consolidated/training/{npz_path.name}"
                )

    def health_check(self) -> HealthCheckResult:
        """Return health check information for the S3 push daemon.

        Returns:
            HealthCheckResult with health status and S3 push metrics.
        """
        # Use base class health check and add domain-specific details
        base_result = super().health_check()

        # Determine health based on error rate and AWS credentials
        has_credentials = self._check_aws_credentials()
        error_rate = 0.0
        total_ops = self._push_stats.total_files_pushed + self._push_stats.push_errors
        if total_ops > 0:
            error_rate = self._push_stats.push_errors / total_ops

        # Mark degraded if no credentials or high error rate
        healthy = base_result.healthy
        if not has_credentials and self.config.enabled:
            healthy = False
        if error_rate > 0.5 and total_ops >= 5:
            healthy = False

        status = "healthy"
        if not healthy:
            status = "degraded"
        if not self._running:
            status = "stopped"
        if not self.config.enabled:
            status = "disabled"

        return HealthCheckResult(
            healthy=healthy,
            status=status,
            message=f"S3PushDaemon: {self._push_stats.total_files_pushed} files pushed to S3",
            details={
                "running": self._running,
                "enabled": self.config.enabled,
                "has_aws_credentials": has_credentials,
                "bucket": self.config.bucket,
                "push_interval": self.config.push_interval,
                "stats": {
                    "total_files_pushed": self._push_stats.total_files_pushed,
                    "total_bytes_pushed": self._push_stats.total_bytes_pushed,
                    "total_mb_pushed": round(
                        self._push_stats.total_bytes_pushed / (1024 * 1024), 2
                    ),
                    "push_errors": self._push_stats.push_errors,
                    "error_rate": round(error_rate, 3),
                    "tracked_files": len(self._last_push_times),
                },
                "last_push_time": self._push_stats.last_push_time,
                "last_error": self._push_stats.last_error,
                **base_result.details,
            },
        )

    def get_stats(self) -> dict[str, Any]:
        """Get current daemon statistics."""
        return {
            "total_files_pushed": self._push_stats.total_files_pushed,
            "total_bytes_pushed": self._push_stats.total_bytes_pushed,
            "total_mb_pushed": round(self._push_stats.total_bytes_pushed / (1024 * 1024), 2),
            "last_push_time": self._push_stats.last_push_time,
            "push_errors": self._push_stats.push_errors,
            "last_error": self._push_stats.last_error,
            "tracked_files": len(self._last_push_times),
        }


def get_s3_push_daemon() -> S3PushDaemon:
    """Get the singleton S3 push daemon instance."""
    return S3PushDaemon.get_instance()


# Factory function for daemon_runners.py
async def create_s3_push() -> None:
    """Create and run S3 push daemon (January 2026).

    Pushes all game databases, training NPZ files, and models to S3
    for backup and cluster-wide access.
    """
    daemon = get_s3_push_daemon()
    await daemon.start()

    # Wait for daemon to run (it will run until stopped)
    try:
        while daemon._running:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        await daemon.stop()

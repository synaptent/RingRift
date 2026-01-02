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

from app.coordination.handler_base import HandlerBase

logger = logging.getLogger(__name__)


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

        self.stats = S3PushStats()
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

            self.stats.last_push_time = time.time()
            logger.info(
                f"[S3PushDaemon] Cycle complete: "
                f"{self.stats.total_files_pushed} files pushed"
            )

        except Exception as e:
            self.stats.push_errors += 1
            self.stats.last_error = str(e)
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

        Args:
            local_path: Local file path
            s3_key: S3 object key

        Returns:
            True if file was pushed, False if skipped (not modified)
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
                file_size = local_path.stat().st_size
                self._last_push_times[str(local_path)] = mtime
                self.stats.total_files_pushed += 1
                self.stats.total_bytes_pushed += file_size
                logger.info(
                    f"[S3PushDaemon] Pushed {local_path.name} to {s3_uri} "
                    f"({file_size / (1024*1024):.1f} MB)"
                )
                return True
            else:
                logger.warning(
                    f"[S3PushDaemon] Failed to push {local_path.name}: {result.stderr}"
                )
                self.stats.push_errors += 1
                return False

        except subprocess.TimeoutExpired:
            logger.warning(f"[S3PushDaemon] Push timed out for {local_path.name}")
            self.stats.push_errors += 1
            return False
        except Exception as e:
            logger.warning(f"[S3PushDaemon] Push error for {local_path.name}: {e}")
            self.stats.push_errors += 1
            return False

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Get event subscriptions for this daemon.

        Subscribes to data events to trigger immediate pushes.
        """
        return {
            "DATA_SYNC_COMPLETED": self._on_data_sync_completed,
            "TRAINING_COMPLETED": self._on_training_completed,
            "NPZ_EXPORT_COMPLETE": self._on_npz_export_complete,
        }

    async def _on_data_sync_completed(self, event: dict[str, Any]) -> None:
        """Handle data sync completion - push synced data to S3."""
        if event.get("needs_s3_backup"):
            db_path_str = event.get("db_path")
            if db_path_str:
                db_path = Path(db_path_str)
                if db_path.exists():
                    await self._push_if_modified(
                        db_path, f"consolidated/games/{db_path.name}"
                    )

    async def _on_training_completed(self, event: dict[str, Any]) -> None:
        """Handle training completion - push model to S3."""
        model_path_str = event.get("model_path")
        if model_path_str:
            model_path = Path(model_path_str)
            if model_path.exists():
                await self._push_if_modified(model_path, f"models/{model_path.name}")

    async def _on_npz_export_complete(self, event: dict[str, Any]) -> None:
        """Handle NPZ export completion - push training data to S3."""
        npz_path_str = event.get("npz_path") or event.get("output_path")
        if npz_path_str:
            npz_path = Path(npz_path_str)
            if npz_path.exists():
                await self._push_if_modified(
                    npz_path, f"consolidated/training/{npz_path.name}"
                )

    def get_stats(self) -> dict[str, Any]:
        """Get current daemon statistics."""
        return {
            "total_files_pushed": self.stats.total_files_pushed,
            "total_bytes_pushed": self.stats.total_bytes_pushed,
            "total_mb_pushed": round(self.stats.total_bytes_pushed / (1024 * 1024), 2),
            "last_push_time": self.stats.last_push_time,
            "push_errors": self.stats.push_errors,
            "last_error": self.stats.last_error,
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

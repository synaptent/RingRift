"""Selfplay Upload Daemon - Automatically upload selfplay games to OWC and S3.

This daemon monitors local selfplay databases and uploads them to:
1. OWC External Drive (via rsync over SSH)
2. AWS S3 (via aws s3 cp)

Key features:
- Periodic batching (every 5 minutes by default)
- Queue for later retry on failure
- Checksum verification before upload
- Progress tracking and statistics
- Event integration for coordination

Configuration via environment variables:
- RINGRIFT_S3_UPLOAD_ENABLED: Enable S3 upload (default: true)
- RINGRIFT_S3_BUCKET: S3 bucket name (default: ringrift-models-20251214)
- RINGRIFT_S3_GAMES_PREFIX: S3 key prefix (default: consolidated/games/)
- RINGRIFT_OWC_UPLOAD_ENABLED: Enable OWC upload (default: true)
- RINGRIFT_OWC_HOST: OWC host (default: mac-studio)
- RINGRIFT_OWC_PATH: OWC destination path (default: /Volumes/RingRift-Data/selfplay_repository)
- RINGRIFT_UPLOAD_INTERVAL: Upload interval in seconds (default: 300)

Usage:
    from app.coordination.selfplay_upload_daemon import (
        SelfplayUploadDaemon,
        SelfplayUploadConfig,
        get_selfplay_upload_daemon,
    )

    # Get singleton instance
    daemon = SelfplayUploadDaemon.get_instance()
    await daemon.start()

    # Or create with custom config
    config = SelfplayUploadConfig(upload_interval_seconds=600)
    daemon = SelfplayUploadDaemon(config)
    await daemon.start()

January 2026: Created as part of multi-source game discovery and sync infrastructure.
January 2026 (Sprint 12.2): Migrated to HandlerBase for unified lifecycle,
event subscription, health checks, and fire-and-forget helpers.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shlex
import shutil
import socket
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.coordination.handler_base import HandlerBase, HealthCheckResult

logger = logging.getLogger(__name__)

__all__ = [
    "SelfplayUploadDaemon",
    "SelfplayUploadConfig",
    "UploadStats",
    "get_selfplay_upload_daemon",
    "reset_selfplay_upload_daemon",
]


# =============================================================================
# Configuration
# =============================================================================

# Environment variable defaults
S3_UPLOAD_ENABLED = os.getenv("RINGRIFT_S3_UPLOAD_ENABLED", "true").lower() == "true"
S3_BUCKET = os.getenv("RINGRIFT_S3_BUCKET", "ringrift-models-20251214")
S3_GAMES_PREFIX = os.getenv("RINGRIFT_S3_GAMES_PREFIX", "consolidated/games/")

OWC_UPLOAD_ENABLED = os.getenv("RINGRIFT_OWC_UPLOAD_ENABLED", "true").lower() == "true"
OWC_HOST = os.getenv("RINGRIFT_OWC_HOST", "mac-studio")
OWC_USER = os.getenv("RINGRIFT_OWC_USER", "armand")
OWC_PATH = os.getenv("RINGRIFT_OWC_PATH", "/Volumes/RingRift-Data/selfplay_repository")
OWC_SSH_KEY = os.getenv("RINGRIFT_OWC_SSH_KEY", os.path.expanduser("~/.ssh/id_ed25519"))

UPLOAD_INTERVAL = int(os.getenv("RINGRIFT_UPLOAD_INTERVAL", "300"))  # 5 minutes
MAX_RETRY_QUEUE_SIZE = int(os.getenv("RINGRIFT_UPLOAD_MAX_RETRY_QUEUE", "1000"))
MAX_RETRIES = int(os.getenv("RINGRIFT_UPLOAD_MAX_RETRIES", "5"))
RETRY_BACKOFF_BASE = float(os.getenv("RINGRIFT_UPLOAD_RETRY_BACKOFF", "60.0"))


@dataclass
class SelfplayUploadConfig:
    """Configuration for the selfplay upload daemon."""

    enabled: bool = True
    upload_interval_seconds: int = UPLOAD_INTERVAL

    # S3 configuration
    upload_to_s3: bool = S3_UPLOAD_ENABLED
    s3_bucket: str = S3_BUCKET
    s3_prefix: str = S3_GAMES_PREFIX

    # OWC configuration
    upload_to_owc: bool = OWC_UPLOAD_ENABLED
    owc_host: str = OWC_HOST
    owc_user: str = OWC_USER
    owc_path: str = OWC_PATH
    owc_ssh_key: str = OWC_SSH_KEY

    # Retry configuration
    max_retry_queue_size: int = MAX_RETRY_QUEUE_SIZE
    max_retries: int = MAX_RETRIES
    retry_backoff_base: float = RETRY_BACKOFF_BASE

    # Minimum games in database before uploading
    min_games_for_upload: int = 10

    # SSH connection timeout
    ssh_timeout: int = 30

    # AWS CLI timeout
    aws_timeout: int = 300

    # Database patterns to scan
    db_patterns: list[str] = field(default_factory=lambda: [
        "data/games/selfplay_*.db",
        "data/games/gpu_selfplay_*.db",
        "data/games/heuristic_selfplay_*.db",
    ])

    @classmethod
    def from_env(cls) -> "SelfplayUploadConfig":
        """Create config from environment variables."""
        return cls()


@dataclass
class UploadJob:
    """Represents a pending upload job."""

    db_path: Path
    config_key: str
    game_count: int
    checksum: str
    created_at: float
    retry_count: int = 0
    last_error: str | None = None
    s3_uploaded: bool = False
    owc_uploaded: bool = False


@dataclass
class UploadStats:
    """Statistics for upload operations."""

    total_uploads_attempted: int = 0
    s3_uploads_success: int = 0
    s3_uploads_failed: int = 0
    owc_uploads_success: int = 0
    owc_uploads_failed: int = 0
    total_games_uploaded: int = 0
    total_bytes_uploaded: int = 0
    retry_queue_size: int = 0
    last_upload_time: float | None = None
    last_error: str | None = None


# =============================================================================
# Selfplay Upload Daemon
# =============================================================================


class SelfplayUploadDaemon(HandlerBase):
    """Daemon that uploads selfplay databases to OWC and S3.

    This daemon periodically scans for new selfplay databases and uploads
    them to configured destinations. Failed uploads are queued for retry.

    Key features:
    - Periodic batch uploads (configurable interval)
    - Checksum verification
    - Retry queue with exponential backoff
    - Progress tracking
    - Event integration

    Inherits from HandlerBase for unified lifecycle, singleton management,
    event subscription, and health check infrastructure.
    """

    _event_source = "SelfplayUploadDaemon"

    def __init__(self, config: SelfplayUploadConfig | None = None):
        """Initialize the upload daemon.

        Args:
            config: Upload configuration (defaults to env-based config)
        """
        # Use _upload_config to avoid collision with HandlerBase._config
        self._upload_config = config or SelfplayUploadConfig.from_env()
        # Initialize HandlerBase with upload interval as cycle interval
        super().__init__(
            name="selfplay_upload",
            cycle_interval=float(self._upload_config.upload_interval_seconds),
        )
        self._node_id = socket.gethostname()
        self._stats = UploadStats()
        self._retry_task: asyncio.Task | None = None

        # Track uploaded files to avoid duplicates
        self._uploaded_files: set[str] = set()

        # Retry queue for failed uploads
        self._retry_queue: list[UploadJob] = []
        self._retry_lock = asyncio.Lock()

        # State persistence path
        self._state_path = Path("data/state/selfplay_upload_daemon.json")

    # =========================================================================
    # HandlerBase Lifecycle Hooks
    # =========================================================================

    def _get_event_subscriptions(self) -> dict[str, Callable[[dict[str, Any]], Any]]:
        """Return event subscriptions for the daemon.

        Subscribes to events indicating new games are available for upload.
        """
        return {
            "new_games_available": self._on_new_games,
            "selfplay_complete": self._on_new_games,
        }

    async def _on_start(self) -> None:
        """Initialize daemon on startup.

        Loads previous state and starts the retry loop background task.
        The main upload loop is handled by HandlerBase._run_cycle().
        """
        if not self._upload_config.enabled:
            logger.info("[SelfplayUploadDaemon] Disabled by configuration")
            return

        logger.info(
            f"[SelfplayUploadDaemon] Starting (interval={self._upload_config.upload_interval_seconds}s, "
            f"s3={self._upload_config.upload_to_s3}, owc={self._upload_config.upload_to_owc})"
        )

        # Load previous state
        await self._load_state()

        # Start retry loop as background task (main upload loop is _run_cycle)
        self._retry_task = self._safe_create_task(
            self._retry_loop(), context="retry_loop"
        )

    async def _on_stop(self) -> None:
        """Cleanup on daemon shutdown."""
        logger.info("[SelfplayUploadDaemon] Stopping...")

        # Cancel retry task
        if self._retry_task and not self._retry_task.done():
            self._retry_task.cancel()
            try:
                await asyncio.wait_for(self._retry_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Save state
        await self._save_state()

        logger.info("[SelfplayUploadDaemon] Stopped")

    async def _run_cycle(self) -> None:
        """Run a single upload cycle (called by HandlerBase every cycle_interval)."""
        if not self._upload_config.enabled:
            return
        await self._run_upload_cycle()

    def health_check(self) -> HealthCheckResult:
        """Return health check information.

        Returns:
            HealthCheckResult with health status and metrics
        """
        # Use base class health check and add domain-specific details
        base_result = super().health_check()

        # Determine health status based on error rates
        failure_rate = 0.0
        total_attempts = self._stats.total_uploads_attempted
        if total_attempts > 0:
            total_failures = (
                self._stats.s3_uploads_failed + self._stats.owc_uploads_failed
            )
            failure_rate = total_failures / total_attempts

        # Mark unhealthy if failure rate exceeds 50% (with min 5 attempts)
        healthy = base_result.healthy and (
            total_attempts < 5 or failure_rate < 0.5
        )

        status = "healthy" if healthy else "degraded"
        if not self._running:
            status = "stopped"

        return HealthCheckResult(
            healthy=healthy,
            status=status,
            message=f"SelfplayUploadDaemon: {self._stats.total_games_uploaded} games uploaded",
            details={
                "running": self._running,
                "stats": {
                    "total_uploads_attempted": self._stats.total_uploads_attempted,
                    "s3_uploads_success": self._stats.s3_uploads_success,
                    "s3_uploads_failed": self._stats.s3_uploads_failed,
                    "owc_uploads_success": self._stats.owc_uploads_success,
                    "owc_uploads_failed": self._stats.owc_uploads_failed,
                    "total_games_uploaded": self._stats.total_games_uploaded,
                    "retry_queue_size": len(self._retry_queue),
                    "failure_rate": failure_rate,
                },
                "config": {
                    "s3_enabled": self._upload_config.upload_to_s3,
                    "owc_enabled": self._upload_config.upload_to_owc,
                    "interval": self._upload_config.upload_interval_seconds,
                },
                "last_upload_time": self._stats.last_upload_time,
                "last_error": self._stats.last_error,
                **base_result.details,
            },
        )

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _on_new_games(self, event: dict[str, Any] | object) -> None:
        """Handle new games event - queue immediate upload check."""
        try:
            db_path = getattr(event, "db_path", None)
            if db_path:
                logger.debug(f"[SelfplayUploadDaemon] New games event for {db_path}")
                # Event-triggered uploads are handled in next cycle
        except Exception as e:
            logger.warning(f"[SelfplayUploadDaemon] Error handling event: {e}")

    def _emit_upload_event(
        self,
        event_name: str,
        db_path: Path,
        destination: str,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Emit upload completion event."""
        # Jan 2026: Migrated to event_router (app.coordination.data_events deprecated Q2 2026)
        try:
            from app.coordination.event_router import DataEventType, get_event_router

            router = get_event_router()

            # Map event names to DataEventType if they exist
            event_type_map = {
                "GAMES_UPLOADED_TO_S3": "GAMES_UPLOADED_TO_S3",
                "GAMES_UPLOADED_TO_OWC": "GAMES_UPLOADED_TO_OWC",
                "UPLOAD_FAILED": "UPLOAD_FAILED",
            }

            event_name_mapped = event_type_map.get(event_name, event_name)

            # Check if event type exists
            if hasattr(DataEventType, event_name_mapped):
                event_type = getattr(DataEventType, event_name_mapped)
                router.emit(
                    event_type,
                    {
                        "db_path": str(db_path),
                        "destination": destination,
                        "success": success,
                        "error": error,
                        "timestamp": time.time(),
                        "node_id": self._node_id,
                    },
                )
            else:
                logger.debug(f"[SelfplayUploadDaemon] Event type {event_name} not defined")

        except Exception as e:
            logger.debug(f"[SelfplayUploadDaemon] Failed to emit event: {e}")

    # =========================================================================
    # Background Retry Loop
    # =========================================================================

    async def _retry_loop(self) -> None:
        """Retry loop - process failed uploads with backoff."""
        logger.info("[SelfplayUploadDaemon] Retry loop started")

        while self._running:
            try:
                await self._process_retry_queue()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[SelfplayUploadDaemon] Retry loop error: {e}")

            # Check retry queue every 60 seconds
            await asyncio.sleep(60.0)

        logger.info("[SelfplayUploadDaemon] Retry loop stopped")

    async def _run_upload_cycle(self) -> None:
        """Run a single upload cycle."""
        logger.debug("[SelfplayUploadDaemon] Starting upload cycle")

        # Find databases to upload
        databases = await self._find_databases_to_upload()

        if not databases:
            logger.debug("[SelfplayUploadDaemon] No new databases to upload")
            return

        logger.info(
            f"[SelfplayUploadDaemon] Found {len(databases)} databases to upload"
        )

        # Upload each database
        for db_info in databases:
            if not self._running:
                break

            await self._upload_database(db_info)

        self._stats.last_upload_time = time.time()

    async def _process_retry_queue(self) -> None:
        """Process failed uploads in the retry queue."""
        async with self._retry_lock:
            if not self._retry_queue:
                return

            now = time.time()
            jobs_to_retry = []
            jobs_to_keep = []

            for job in self._retry_queue:
                # Calculate backoff delay
                backoff = self._upload_config.retry_backoff_base * (2 ** job.retry_count)
                next_retry = job.created_at + backoff

                if now >= next_retry and job.retry_count < self._upload_config.max_retries:
                    jobs_to_retry.append(job)
                elif job.retry_count >= self._upload_config.max_retries:
                    logger.warning(
                        f"[SelfplayUploadDaemon] Max retries exceeded for {job.db_path}"
                    )
                else:
                    jobs_to_keep.append(job)

            self._retry_queue = jobs_to_keep

        # Retry jobs outside of lock
        for job in jobs_to_retry:
            job.retry_count += 1
            logger.info(
                f"[SelfplayUploadDaemon] Retrying upload for {job.db_path} "
                f"(attempt {job.retry_count}/{self._upload_config.max_retries})"
            )
            await self._upload_database(
                {
                    "path": job.db_path,
                    "config_key": job.config_key,
                    "game_count": job.game_count,
                    "checksum": job.checksum,
                },
                is_retry=True,
                retry_job=job,
            )

    # =========================================================================
    # Database Discovery
    # =========================================================================

    async def _find_databases_to_upload(self) -> list[dict[str, Any]]:
        """Find databases that need to be uploaded.

        Returns:
            List of database info dicts with path, config_key, game_count, checksum
        """
        databases = []

        for pattern in self._upload_config.db_patterns:
            db_files = list(Path(".").glob(pattern))

            for db_path in db_files:
                # Skip if already uploaded
                if str(db_path) in self._uploaded_files:
                    continue

                # Check database has enough games
                db_info = await self._get_database_info(db_path)
                if db_info and db_info["game_count"] >= self._upload_config.min_games_for_upload:
                    databases.append(db_info)

        return databases

    async def _get_database_info(self, db_path: Path) -> dict[str, Any] | None:
        """Get information about a database.

        Args:
            db_path: Path to the database file

        Returns:
            Dict with path, config_key, game_count, checksum or None if invalid
        """
        try:
            # Run in thread to avoid blocking
            return await asyncio.to_thread(self._get_database_info_sync, db_path)
        except Exception as e:
            logger.warning(f"[SelfplayUploadDaemon] Error reading {db_path}: {e}")
            return None

    def _get_database_info_sync(self, db_path: Path) -> dict[str, Any] | None:
        """Synchronous database info retrieval."""
        try:
            with sqlite3.connect(str(db_path), timeout=10) as conn:
                cursor = conn.cursor()

                # Get game count
                cursor.execute(
                    "SELECT COUNT(*) FROM games WHERE winner IS NOT NULL"
                )
                game_count = cursor.fetchone()[0]

                # Get config key from most recent game
                cursor.execute(
                    """SELECT board_type, num_players FROM games
                       WHERE winner IS NOT NULL
                       ORDER BY created_at DESC LIMIT 1"""
                )
                row = cursor.fetchone()
                if row:
                    config_key = f"{row[0]}_{row[1]}p"
                else:
                    config_key = "unknown"

            # Calculate checksum
            checksum = self._calculate_checksum(db_path)

            return {
                "path": db_path,
                "config_key": config_key,
                "game_count": game_count,
                "checksum": checksum,
                "size_bytes": db_path.stat().st_size,
            }

        except sqlite3.Error as e:
            logger.warning(f"[SelfplayUploadDaemon] SQLite error for {db_path}: {e}")
            return None

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    # =========================================================================
    # Upload Operations
    # =========================================================================

    async def _upload_database(
        self,
        db_info: dict[str, Any],
        is_retry: bool = False,
        retry_job: UploadJob | None = None,
    ) -> None:
        """Upload a database to configured destinations.

        Args:
            db_info: Database information dict
            is_retry: Whether this is a retry attempt
            retry_job: Existing retry job if retrying
        """
        db_path = db_info["path"]
        self._stats.total_uploads_attempted += 1

        s3_success = True
        owc_success = True
        errors = []

        # Upload to S3
        if self._upload_config.upload_to_s3:
            if retry_job and retry_job.s3_uploaded:
                s3_success = True
            else:
                s3_success = await self._upload_to_s3(db_path, db_info)
                if s3_success:
                    self._stats.s3_uploads_success += 1
                    if retry_job:
                        retry_job.s3_uploaded = True
                else:
                    self._stats.s3_uploads_failed += 1
                    errors.append("S3 upload failed")

        # Upload to OWC
        if self._upload_config.upload_to_owc:
            if retry_job and retry_job.owc_uploaded:
                owc_success = True
            else:
                owc_success = await self._upload_to_owc(db_path, db_info)
                if owc_success:
                    self._stats.owc_uploads_success += 1
                    if retry_job:
                        retry_job.owc_uploaded = True
                else:
                    self._stats.owc_uploads_failed += 1
                    errors.append("OWC upload failed")

        # Check overall success
        if s3_success and owc_success:
            self._uploaded_files.add(str(db_path))
            self._stats.total_games_uploaded += db_info["game_count"]
            self._stats.total_bytes_uploaded += db_info.get("size_bytes", 0)
            logger.info(
                f"[SelfplayUploadDaemon] Uploaded {db_path} "
                f"({db_info['game_count']} games)"
            )
        else:
            # Queue for retry if not max retries
            error_msg = "; ".join(errors)
            self._stats.last_error = error_msg

            if not is_retry:
                # Create new retry job
                job = UploadJob(
                    db_path=db_path,
                    config_key=db_info["config_key"],
                    game_count=db_info["game_count"],
                    checksum=db_info["checksum"],
                    created_at=time.time(),
                    last_error=error_msg,
                    s3_uploaded=s3_success,
                    owc_uploaded=owc_success,
                )
                await self._queue_for_retry(job)
            elif retry_job:
                # Update existing job
                retry_job.last_error = error_msg
                if retry_job.retry_count < self._upload_config.max_retries:
                    await self._queue_for_retry(retry_job)

            self._emit_upload_event(
                "UPLOAD_FAILED", db_path, "s3/owc", False, error_msg
            )

    async def _upload_to_s3(
        self, db_path: Path, db_info: dict[str, Any]
    ) -> bool:
        """Upload database to S3.

        Args:
            db_path: Path to database file
            db_info: Database metadata

        Returns:
            True if upload succeeded
        """
        try:
            config_key = db_info["config_key"]
            db_name = db_path.name

            # Build S3 key with config-based organization
            s3_key = f"s3://{self._upload_config.s3_bucket}/{self._upload_config.s3_prefix}{config_key}/{db_name}"

            # Use aws s3 cp command
            cmd = [
                "aws", "s3", "cp",
                str(db_path),
                s3_key,
                "--only-show-errors",
            ]

            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=self._upload_config.aws_timeout,
            )

            if result.returncode == 0:
                logger.debug(f"[SelfplayUploadDaemon] S3 upload success: {s3_key}")
                self._emit_upload_event("GAMES_UPLOADED_TO_S3", db_path, s3_key, True)
                return True
            else:
                logger.warning(
                    f"[SelfplayUploadDaemon] S3 upload failed: {result.stderr}"
                )
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"[SelfplayUploadDaemon] S3 upload timeout for {db_path}")
            return False
        except Exception as e:
            logger.error(f"[SelfplayUploadDaemon] S3 upload error: {e}")
            return False

    async def _upload_to_owc(
        self, db_path: Path, db_info: dict[str, Any]
    ) -> bool:
        """Upload database to OWC external drive via SSH/rsync.

        Args:
            db_path: Path to database file
            db_info: Database metadata

        Returns:
            True if upload succeeded
        """
        try:
            config_key = db_info["config_key"]
            ssh_key_path = Path(self._upload_config.owc_ssh_key).expanduser()

            if not ssh_key_path.exists():
                logger.warning(
                    f"[SelfplayUploadDaemon] SSH key not found: {ssh_key_path}"
                )
                return False

            # Build destination path with config-based organization
            dest_dir = f"{self._upload_config.owc_path}/{config_key}"
            dest_path = f"{self._upload_config.owc_user}@{self._upload_config.owc_host}:{dest_dir}/"

            # Ensure destination directory exists
            # Jan 2026: Use command list instead of shell=True for security
            mkdir_cmd = [
                "ssh",
                "-i", str(ssh_key_path),
                "-o", f"ConnectTimeout={self._upload_config.ssh_timeout}",
                "-o", "BatchMode=yes",
                f"{self._upload_config.owc_user}@{self._upload_config.owc_host}",
                f"mkdir -p {shlex.quote(dest_dir)}",
            ]

            mkdir_result = await asyncio.to_thread(
                subprocess.run,
                mkdir_cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if mkdir_result.returncode != 0:
                logger.warning(
                    f"[SelfplayUploadDaemon] Failed to create OWC dir: {mkdir_result.stderr}"
                )
                # Continue anyway, directory might exist

            # Use rsync for upload with checksum verification
            rsync_cmd = [
                "rsync", "-avz", "--checksum",
                "-e", f"ssh -i {ssh_key_path} -o ConnectTimeout={self._upload_config.ssh_timeout} -o BatchMode=yes",
                str(db_path),
                dest_path,
            ]

            result = await asyncio.to_thread(
                subprocess.run,
                rsync_cmd,
                capture_output=True,
                text=True,
                timeout=self._upload_config.aws_timeout,  # Same timeout as S3
            )

            if result.returncode == 0:
                logger.debug(
                    f"[SelfplayUploadDaemon] OWC upload success: {dest_path}"
                )
                self._emit_upload_event("GAMES_UPLOADED_TO_OWC", db_path, dest_path, True)
                return True
            else:
                logger.warning(
                    f"[SelfplayUploadDaemon] OWC upload failed: {result.stderr}"
                )
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"[SelfplayUploadDaemon] OWC upload timeout for {db_path}")
            return False
        except Exception as e:
            logger.error(f"[SelfplayUploadDaemon] OWC upload error: {e}")
            return False

    async def _queue_for_retry(self, job: UploadJob) -> None:
        """Add a failed upload job to the retry queue.

        Args:
            job: Upload job to retry
        """
        async with self._retry_lock:
            # Check queue size limit
            if len(self._retry_queue) >= self._upload_config.max_retry_queue_size:
                # Remove oldest job
                self._retry_queue.pop(0)
                logger.warning(
                    "[SelfplayUploadDaemon] Retry queue full, dropped oldest job"
                )

            self._retry_queue.append(job)
            self._stats.retry_queue_size = len(self._retry_queue)

    # =========================================================================
    # State Persistence
    # =========================================================================

    async def _save_state(self) -> None:
        """Save daemon state to disk."""
        try:
            import json

            self._state_path.parent.mkdir(parents=True, exist_ok=True)

            state = {
                "uploaded_files": list(self._uploaded_files),
                "retry_queue": [
                    {
                        "db_path": str(job.db_path),
                        "config_key": job.config_key,
                        "game_count": job.game_count,
                        "checksum": job.checksum,
                        "created_at": job.created_at,
                        "retry_count": job.retry_count,
                        "last_error": job.last_error,
                        "s3_uploaded": job.s3_uploaded,
                        "owc_uploaded": job.owc_uploaded,
                    }
                    for job in self._retry_queue
                ],
                "stats": {
                    "total_uploads_attempted": self._stats.total_uploads_attempted,
                    "s3_uploads_success": self._stats.s3_uploads_success,
                    "s3_uploads_failed": self._stats.s3_uploads_failed,
                    "owc_uploads_success": self._stats.owc_uploads_success,
                    "owc_uploads_failed": self._stats.owc_uploads_failed,
                    "total_games_uploaded": self._stats.total_games_uploaded,
                    "total_bytes_uploaded": self._stats.total_bytes_uploaded,
                },
                "saved_at": time.time(),
            }

            def _write_state() -> None:
                with open(self._state_path, "w") as f:
                    json.dump(state, f, indent=2)

            await asyncio.to_thread(_write_state)

            logger.debug(f"[SelfplayUploadDaemon] State saved to {self._state_path}")

        except Exception as e:
            logger.warning(f"[SelfplayUploadDaemon] Failed to save state: {e}")

    async def _load_state(self) -> None:
        """Load daemon state from disk."""
        try:
            import json

            if not self._state_path.exists():
                return

            def _read_state() -> dict:
                with open(self._state_path) as f:
                    return json.load(f)

            state = await asyncio.to_thread(_read_state)

            self._uploaded_files = set(state.get("uploaded_files", []))

            # Restore retry queue
            for job_data in state.get("retry_queue", []):
                job = UploadJob(
                    db_path=Path(job_data["db_path"]),
                    config_key=job_data["config_key"],
                    game_count=job_data["game_count"],
                    checksum=job_data["checksum"],
                    created_at=job_data["created_at"],
                    retry_count=job_data.get("retry_count", 0),
                    last_error=job_data.get("last_error"),
                    s3_uploaded=job_data.get("s3_uploaded", False),
                    owc_uploaded=job_data.get("owc_uploaded", False),
                )
                self._retry_queue.append(job)

            # Restore stats
            stats_data = state.get("stats", {})
            self._stats.total_uploads_attempted = stats_data.get("total_uploads_attempted", 0)
            self._stats.s3_uploads_success = stats_data.get("s3_uploads_success", 0)
            self._stats.s3_uploads_failed = stats_data.get("s3_uploads_failed", 0)
            self._stats.owc_uploads_success = stats_data.get("owc_uploads_success", 0)
            self._stats.owc_uploads_failed = stats_data.get("owc_uploads_failed", 0)
            self._stats.total_games_uploaded = stats_data.get("total_games_uploaded", 0)
            self._stats.total_bytes_uploaded = stats_data.get("total_bytes_uploaded", 0)

            logger.info(
                f"[SelfplayUploadDaemon] Loaded state: "
                f"{len(self._uploaded_files)} uploaded, "
                f"{len(self._retry_queue)} in retry queue"
            )

        except Exception as e:
            logger.warning(f"[SelfplayUploadDaemon] Failed to load state: {e}")


# =============================================================================
# Singleton Management (delegates to HandlerBase)
# =============================================================================


def get_selfplay_upload_daemon(
    config: SelfplayUploadConfig | None = None,
) -> SelfplayUploadDaemon:
    """Get or create the singleton SelfplayUploadDaemon instance.

    Delegates to HandlerBase.get_instance() for thread-safe singleton management.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        SelfplayUploadDaemon instance
    """
    return SelfplayUploadDaemon.get_instance(config=config)


def reset_selfplay_upload_daemon() -> None:
    """Reset the singleton instance (for testing).

    Delegates to HandlerBase.reset_instance().
    """
    SelfplayUploadDaemon.reset_instance()


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    import argparse

    async def main():
        parser = argparse.ArgumentParser(description="Selfplay Upload Daemon")
        parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
        parser.add_argument("--dry-run", action="store_true", help="Discover but don't upload")
        parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
        args = parser.parse_args()

        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(levelname)s: %(message)s",
        )

        config = SelfplayUploadConfig()
        daemon = SelfplayUploadDaemon(config)

        if args.once or args.dry_run:
            # Just run discovery
            databases = await daemon._find_databases_to_upload()
            print(f"\nFound {len(databases)} databases to upload:")
            for db in databases:
                print(f"  {db['path']}: {db['config_key']} ({db['game_count']} games)")

            if not args.dry_run and databases:
                print("\nUploading...")
                for db in databases:
                    await daemon._upload_database(db)
                print("\nUpload complete")
                print(f"Stats: {daemon.health_check()['stats']}")
        else:
            # Run daemon
            await daemon.start()
            try:
                # Run until interrupted
                while daemon._running:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                await daemon.stop()

    asyncio.run(main())

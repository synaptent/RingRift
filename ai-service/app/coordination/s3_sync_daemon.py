"""S3 Sync Daemon - Unified S3 backup and sync (January 2026).

This daemon consolidates three previous S3 daemons:
- S3BackupDaemon: Event-driven model backup after promotion
- S3PushDaemon: Periodic incremental push of all data types
- S3NodeSyncDaemon: Node-namespaced push for cluster nodes

Features:
1. EVENT-DRIVEN: Immediate backup on MODEL_PROMOTED, TRAINING_COMPLETED
2. PERIODIC: Incremental sync every 10 minutes for changed files
3. NODE NAMESPACING: Each node pushes to s3://bucket/nodes/{node_id}/
4. MODIFICATION TRACKING: Only uploads files changed since last push
5. DEBOUNCING: Avoids rapid-fire backups during promotion bursts

Usage:
    from app.coordination.s3_sync_daemon import S3SyncDaemon, get_s3_sync_daemon

    daemon = get_s3_sync_daemon()
    await daemon.start()

Environment Variables:
    RINGRIFT_S3_BUCKET: S3 bucket (default: ringrift-models-20251214)
    RINGRIFT_S3_REGION: AWS region (default: us-east-1)
    RINGRIFT_S3_SYNC_INTERVAL: Sync interval in seconds (default: 600)
    RINGRIFT_S3_SYNC_ENABLED: Enable/disable daemon (default: true)
    RINGRIFT_S3_NODE_NAMESPACING: Use node-namespaced paths (default: true)

January 2026: Created to consolidate S3BackupDaemon, S3PushDaemon, S3NodeSyncDaemon.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult

logger = logging.getLogger(__name__)


def _get_node_id() -> str:
    """Get unique node identifier."""
    node_id = os.getenv("RINGRIFT_NODE_ID")
    if node_id:
        return node_id

    hostname = socket.gethostname()
    # Clean up common prefixes
    for prefix in ["ip-", "instance-", "node-"]:
        if hostname.startswith(prefix):
            hostname = hostname[len(prefix) :]
    return hostname


@dataclass
class S3SyncConfig:
    """Configuration for unified S3 sync daemon."""

    # S3 settings
    s3_bucket: str = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_S3_BUCKET", "ringrift-models-20251214"
        )
    )
    aws_region: str = field(
        default_factory=lambda: os.environ.get("RINGRIFT_S3_REGION", "us-east-1")
    )

    # Sync settings
    sync_interval: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_S3_SYNC_INTERVAL", "600")
        )
    )
    enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_S3_SYNC_ENABLED", "true"
        ).lower()
        == "true"
    )

    # Node namespacing (like S3NodeSyncDaemon)
    use_node_namespacing: bool = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_S3_NODE_NAMESPACING", "true"
        ).lower()
        == "true"
    )

    # Data types to sync
    sync_games: bool = True
    sync_models: bool = True
    sync_npz: bool = True

    # S3 storage class
    storage_class: str = "STANDARD_IA"

    # Debounce for event-driven backups (from S3BackupDaemon)
    debounce_seconds: float = 60.0
    max_pending_before_immediate: int = 5

    # Timeouts and retries
    upload_timeout_seconds: float = 600.0
    retry_count: int = 3
    retry_delay_seconds: float = 30.0

    # Emit completion events
    emit_completion_event: bool = True


@dataclass
class S3SyncStats:
    """Statistics for S3 sync operations."""

    total_files_pushed: int = 0
    total_bytes_pushed: int = 0
    successful_syncs: int = 0
    failed_syncs: int = 0
    events_processed: int = 0
    last_sync_time: float = 0.0
    last_error: str | None = None
    # Phase 1: Track event-driven vs periodic pushes (Jan 2026)
    event_driven_pushes: int = 0
    periodic_pushes: int = 0
    last_event_driven_push_time: float = 0.0


class S3SyncDaemon(HandlerBase):
    """Unified S3 sync daemon combining event-driven and periodic backup.

    Consolidates:
    - S3BackupDaemon: EVENT-driven model backup after MODEL_PROMOTED
    - S3PushDaemon: PERIODIC incremental sync with modification tracking
    - S3NodeSyncDaemon: Node-namespaced paths for cluster-wide backup

    January 2026: Created as part of backup daemon consolidation.
    """

    _instance: S3SyncDaemon | None = None

    def __init__(self, config: S3SyncConfig | None = None):
        """Initialize unified S3 sync daemon.

        Args:
            config: Optional configuration. Uses environment defaults if not provided.
        """
        self.config = config or S3SyncConfig()
        self.node_id = _get_node_id()

        super().__init__(
            name=f"s3_sync_{self.node_id}",
            cycle_interval=self.config.sync_interval,
        )

        self._stats = S3SyncStats()
        self._last_push_times: dict[str, float] = {}  # path -> mtime at last push
        self._base_path = Path(os.environ.get("RINGRIFT_BASE_PATH", "."))

        # Pending promotions for debounced backup (from S3BackupDaemon)
        self._pending_promotions: list[dict[str, Any]] = []
        self._pending_lock = asyncio.Lock()

    @classmethod
    def get_instance(cls, config: S3SyncConfig | None = None) -> S3SyncDaemon:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    # =========================================================================
    # Event Subscriptions
    # =========================================================================

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Subscribe to events for immediate backup."""
        return {
            "MODEL_PROMOTED": self._on_model_promoted,
            "TRAINING_COMPLETED": self._on_training_completed,
            "DATA_SYNC_COMPLETED": self._on_data_sync_completed,
            "NPZ_EXPORT_COMPLETE": self._on_npz_export_complete,
            "SELFPLAY_COMPLETE": self._on_selfplay_complete,
            # Phase 1: Event-driven S3 push for new games (Jan 2026)
            "NEW_GAMES_AVAILABLE": self._on_new_games_available,
        }

    async def _on_model_promoted(self, event: dict[str, Any]) -> None:
        """Handle MODEL_PROMOTED - queue for debounced backup."""
        payload = event.get("payload", event) if isinstance(event, dict) else {}
        promotion_info = {
            "model_path": payload.get("model_path"),
            "board_type": payload.get("board_type"),
            "num_players": payload.get("num_players"),
            "timestamp": time.time(),
        }
        logger.info(f"[S3SyncDaemon] MODEL_PROMOTED received: {promotion_info}")

        async with self._pending_lock:
            self._pending_promotions.append(promotion_info)
        self._stats.events_processed += 1

    async def _on_training_completed(self, event: Any) -> None:
        """Handle TRAINING_COMPLETED - immediate model push."""
        # Jan 2026: Fix RouterEvent handling - extract payload first
        payload = getattr(event, "payload", {}) or (event if isinstance(event, dict) else {})
        model_path_str = payload.get("model_path")
        if model_path_str:
            model_path = Path(model_path_str)
            if model_path.exists():
                await self._push_file(model_path, self._get_s3_key(model_path, "models"))
        self._stats.events_processed += 1

    async def _on_data_sync_completed(self, event: Any) -> None:
        """Handle DATA_SYNC_COMPLETED - push synced database."""
        # Jan 2026: Fix RouterEvent handling - extract payload first
        payload = getattr(event, "payload", {}) or (event if isinstance(event, dict) else {})
        if payload.get("needs_s3_backup"):
            db_path_str = payload.get("db_path")
            if db_path_str:
                db_path = Path(db_path_str)
                if db_path.exists():
                    await self._push_file(db_path, self._get_s3_key(db_path, "games"))
        self._stats.events_processed += 1

    async def _on_npz_export_complete(self, event: Any) -> None:
        """Handle NPZ_EXPORT_COMPLETE - push training data."""
        # Jan 2026: Fix RouterEvent handling - extract payload first
        payload = getattr(event, "payload", {}) or (event if isinstance(event, dict) else {})
        npz_path_str = payload.get("npz_path") or payload.get("output_path")
        if npz_path_str:
            npz_path = Path(npz_path_str)
            if npz_path.exists():
                await self._push_file(npz_path, self._get_s3_key(npz_path, "training"))
        self._stats.events_processed += 1

    async def _on_selfplay_complete(self, event: Any) -> None:
        """Handle SELFPLAY_COMPLETE - push game database."""
        # Jan 2026: Fix RouterEvent handling - extract payload first
        payload = getattr(event, "payload", {}) or (event if isinstance(event, dict) else {})
        db_path_str = payload.get("db_path")
        if db_path_str:
            db_path = Path(db_path_str)
            if db_path.exists():
                await self._push_file(db_path, self._get_s3_key(db_path, "games"))
        self._stats.events_processed += 1

    async def _on_new_games_available(self, event: Any) -> None:
        """Handle NEW_GAMES_AVAILABLE - immediate S3 push for new games.

        Phase 1 of S3-as-primary-storage: Push games to S3 immediately on
        completion, not just on 10-minute interval.

        January 2026: Added as part of S3 first-class storage tier upgrade.
        """
        # Jan 2026: Fix RouterEvent handling - extract payload first
        payload = getattr(event, "payload", {}) or (event if isinstance(event, dict) else {})
        config_key = payload.get("config") or payload.get("config_key")
        new_games = payload.get("new_games", 0)

        logger.info(
            f"[S3SyncDaemon] NEW_GAMES_AVAILABLE: {config_key} ({new_games} games)"
        )

        if config_key:
            # Push the canonical database for this config
            db_name = f"canonical_{config_key}.db"
            db_path = self._base_path / "data" / "games" / db_name
            pushed = False
            if db_path.exists():
                success = await self._push_file(db_path, self._get_s3_key(db_path, "games"))
                if success:
                    pushed = True
                    logger.info(
                        f"[S3SyncDaemon] Pushed {db_name} to S3 (event-driven)"
                    )
            else:
                # Try alternate naming patterns
                for pattern in [
                    f"{config_key}_selfplay.db",
                    f"{config_key}.db",
                ]:
                    alt_path = self._base_path / "data" / "games" / pattern
                    if alt_path.exists():
                        success = await self._push_file(alt_path, self._get_s3_key(alt_path, "games"))
                        if success:
                            pushed = True
                        break

            # Track event-driven push stats
            if pushed:
                self._stats.event_driven_pushes += 1
                self._stats.last_event_driven_push_time = time.time()
                # Emit S3_PUSH_COMPLETED event for monitoring
                await self._emit_s3_push_completed(config_key, "games")

        self._stats.events_processed += 1

    # =========================================================================
    # Main Cycle
    # =========================================================================

    async def _run_cycle(self) -> None:
        """Main sync cycle - periodic push + debounced promotion backup."""
        if not self.config.enabled:
            logger.debug("[S3SyncDaemon] Disabled via config, skipping cycle")
            return

        if not self._check_aws_credentials():
            logger.warning("[S3SyncDaemon] AWS credentials not configured, skipping")
            return

        try:
            # 1. Process pending promotions (debounced)
            await self._process_pending_promotions()

            # 2. Periodic sync of all data types
            if self.config.sync_games:
                await self._sync_games()
            if self.config.sync_npz:
                await self._sync_npz()
            if self.config.sync_models:
                await self._sync_models()

            self._stats.last_sync_time = time.time()
            self._stats.successful_syncs += 1

            logger.info(
                f"[S3SyncDaemon] Cycle complete: {self._stats.total_files_pushed} total files pushed"
            )

        except Exception as e:
            self._stats.failed_syncs += 1
            self._stats.last_error = str(e)
            logger.error(f"[S3SyncDaemon] Sync cycle failed: {e}")
            self._record_error(f"Sync cycle failed: {e}", e)

    async def _process_pending_promotions(self) -> None:
        """Process pending promotions with debouncing (from S3BackupDaemon)."""
        async with self._pending_lock:
            if not self._pending_promotions:
                return

            oldest_time = min(p.get("timestamp", time.time()) for p in self._pending_promotions)
            time_since_oldest = time.time() - oldest_time
            pending_count = len(self._pending_promotions)

            should_backup = (
                time_since_oldest >= self.config.debounce_seconds
                or pending_count >= self.config.max_pending_before_immediate
            )

            if not should_backup:
                return

            promotions = self._pending_promotions.copy()
            self._pending_promotions.clear()

        logger.info(f"[S3SyncDaemon] Processing {len(promotions)} pending promotions")

        # Push models directory
        await self._sync_models()

        # Emit completion event
        if self.config.emit_completion_event:
            await self._emit_backup_complete(promotions)

    # =========================================================================
    # Sync Operations
    # =========================================================================

    async def _sync_games(self) -> None:
        """Sync game databases to S3."""
        games_dir = self._base_path / "data" / "games"
        if not games_dir.exists():
            return

        for db_path in games_dir.glob("canonical_*.db"):
            await self._push_if_modified(db_path, self._get_s3_key(db_path, "games"))

    async def _sync_npz(self) -> None:
        """Sync NPZ training files to S3."""
        training_dir = self._base_path / "data" / "training"
        if not training_dir.exists():
            return

        for npz_path in training_dir.glob("*.npz"):
            await self._push_if_modified(npz_path, self._get_s3_key(npz_path, "training"))

    async def _sync_models(self) -> None:
        """Sync model checkpoints to S3."""
        models_dir = self._base_path / "models"
        if not models_dir.exists():
            return

        for model_path in models_dir.glob("canonical_*.pth"):
            await self._push_if_modified(model_path, self._get_s3_key(model_path, "models"))

    def _get_s3_key(self, local_path: Path, data_type: str) -> str:
        """Get S3 key for a local file.

        Args:
            local_path: Local file path
            data_type: One of "games", "training", "models"

        Returns:
            S3 key with optional node namespacing
        """
        if self.config.use_node_namespacing:
            # Node-namespaced: s3://bucket/nodes/{node_id}/{data_type}/{filename}
            return f"nodes/{self.node_id}/{data_type}/{local_path.name}"
        else:
            # Consolidated: s3://bucket/consolidated/{data_type}/{filename}
            return f"consolidated/{data_type}/{local_path.name}"

    async def _push_if_modified(self, local_path: Path, s3_key: str) -> bool:
        """Push file to S3 if modified since last push.

        Returns:
            True if file was pushed, False if skipped
        """
        if not local_path.exists():
            return False

        try:
            mtime = local_path.stat().st_mtime
            last_push = self._last_push_times.get(str(local_path), 0)

            if mtime <= last_push:
                logger.debug(f"[S3SyncDaemon] Skipping {local_path.name} (not modified)")
                return False

            return await self._push_file(local_path, s3_key)

        except Exception as e:
            logger.warning(f"[S3SyncDaemon] Error checking {local_path.name}: {e}")
            return False

    async def _push_file(self, local_path: Path, s3_key: str) -> bool:
        """Push a single file to S3.

        Returns:
            True if successful, False otherwise
        """
        if not local_path.exists():
            return False

        s3_uri = f"s3://{self.config.s3_bucket}/{s3_key}"

        try:
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
                    self.config.aws_region,
                ],
                capture_output=True,
                text=True,
                timeout=int(self.config.upload_timeout_seconds),
            )

            if result.returncode == 0:
                file_size = local_path.stat().st_size
                self._last_push_times[str(local_path)] = local_path.stat().st_mtime
                self._stats.total_files_pushed += 1
                self._stats.total_bytes_pushed += file_size
                logger.info(
                    f"[S3SyncDaemon] Pushed {local_path.name} to {s3_uri} "
                    f"({file_size / (1024*1024):.1f} MB)"
                )
                return True
            else:
                logger.warning(
                    f"[S3SyncDaemon] Failed to push {local_path.name}: {result.stderr}"
                )
                return False

        except subprocess.TimeoutExpired:
            logger.warning(f"[S3SyncDaemon] Push timed out for {local_path.name}")
            return False
        except Exception as e:
            logger.warning(f"[S3SyncDaemon] Push error for {local_path.name}: {e}")
            return False

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _check_aws_credentials(self) -> bool:
        """Check if AWS credentials are available."""
        if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
            return True

        aws_config = Path.home() / ".aws" / "credentials"
        if aws_config.exists():
            return True

        return False

    async def _emit_backup_complete(self, promotions: list[dict[str, Any]]) -> None:
        """Emit S3_BACKUP_COMPLETED event."""
        try:
            from app.coordination.event_router import emit

            await emit(
                event_type="S3_BACKUP_COMPLETED",
                data={
                    "promotions": promotions,
                    "files_pushed": self._stats.total_files_pushed,
                    "bytes_pushed": self._stats.total_bytes_pushed,
                    "bucket": self.config.s3_bucket,
                    "node_id": self.node_id,
                    "timestamp": time.time(),
                },
            )
            logger.debug("[S3SyncDaemon] Emitted S3_BACKUP_COMPLETED event")
        except Exception as e:
            logger.warning(f"[S3SyncDaemon] Failed to emit backup complete event: {e}")

    async def _emit_s3_push_completed(self, config_key: str, data_type: str) -> None:
        """Emit S3_PUSH_COMPLETED event for monitoring S3 replication lag.

        Phase 1 of S3-as-primary-storage: Enables health monitoring to track
        how quickly data reaches S3 after selfplay completion.

        January 2026: Added as part of S3 first-class storage tier upgrade.
        """
        try:
            from app.coordination.event_router import emit

            await emit(
                event_type="S3_PUSH_COMPLETED",
                data={
                    "config_key": config_key,
                    "data_type": data_type,
                    "bucket": self.config.s3_bucket,
                    "node_id": self.node_id,
                    "push_type": "event_driven",
                    "timestamp": time.time(),
                },
            )
            logger.debug(f"[S3SyncDaemon] Emitted S3_PUSH_COMPLETED for {config_key}")
        except Exception as e:
            logger.warning(f"[S3SyncDaemon] Failed to emit S3_PUSH_COMPLETED: {e}")

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """Return health check information."""
        from app.coordination.protocols import CoordinatorStatus

        if not self._running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="S3SyncDaemon not running",
            )

        has_credentials = self._check_aws_credentials()
        total_ops = self._stats.successful_syncs + self._stats.failed_syncs
        error_rate = self._stats.failed_syncs / total_ops if total_ops > 0 else 0.0

        # Determine health
        healthy = True
        if not has_credentials and self.config.enabled:
            healthy = False
        if error_rate > 0.5 and total_ops >= 5:
            healthy = False

        status = CoordinatorStatus.RUNNING
        if not healthy:
            status = CoordinatorStatus.DEGRADED
        if not self.config.enabled:
            status = CoordinatorStatus.STOPPED

        return HealthCheckResult(
            healthy=healthy,
            status=status,
            message=f"S3SyncDaemon: {self._stats.total_files_pushed} files pushed",
            details={
                "running": self._running,
                "enabled": self.config.enabled,
                "has_aws_credentials": has_credentials,
                "node_id": self.node_id,
                "bucket": self.config.s3_bucket,
                "use_node_namespacing": self.config.use_node_namespacing,
                "sync_interval": self.config.sync_interval,
                "stats": {
                    "total_files_pushed": self._stats.total_files_pushed,
                    "total_bytes_pushed": self._stats.total_bytes_pushed,
                    "total_mb_pushed": round(self._stats.total_bytes_pushed / (1024 * 1024), 2),
                    "successful_syncs": self._stats.successful_syncs,
                    "failed_syncs": self._stats.failed_syncs,
                    "events_processed": self._stats.events_processed,
                    "error_rate": round(error_rate, 3),
                    "tracked_files": len(self._last_push_times),
                    # Phase 1: Event-driven vs periodic push stats (Jan 2026)
                    "event_driven_pushes": self._stats.event_driven_pushes,
                    "periodic_pushes": self._stats.periodic_pushes,
                    "last_event_driven_push_time": self._stats.last_event_driven_push_time,
                },
                "last_sync_time": self._stats.last_sync_time,
                "last_error": self._stats.last_error,
            },
        )

    def get_stats(self) -> dict[str, Any]:
        """Get current daemon statistics."""
        return {
            "total_files_pushed": self._stats.total_files_pushed,
            "total_bytes_pushed": self._stats.total_bytes_pushed,
            "total_mb_pushed": round(self._stats.total_bytes_pushed / (1024 * 1024), 2),
            "successful_syncs": self._stats.successful_syncs,
            "failed_syncs": self._stats.failed_syncs,
            "events_processed": self._stats.events_processed,
            "last_sync_time": self._stats.last_sync_time,
            "last_error": self._stats.last_error,
            "tracked_files": len(self._last_push_times),
            # Phase 1: Event-driven vs periodic push stats (Jan 2026)
            "event_driven_pushes": self._stats.event_driven_pushes,
            "periodic_pushes": self._stats.periodic_pushes,
            "last_event_driven_push_time": self._stats.last_event_driven_push_time,
        }


# =============================================================================
# Module-level accessors
# =============================================================================


def get_s3_sync_daemon(config: S3SyncConfig | None = None) -> S3SyncDaemon:
    """Get the singleton S3 sync daemon instance."""
    return S3SyncDaemon.get_instance(config)


def reset_s3_sync_daemon() -> None:
    """Reset the singleton instance (for testing)."""
    S3SyncDaemon.reset_instance()


# Factory function for daemon_runners.py
async def create_s3_sync() -> None:
    """Create and run unified S3 sync daemon.

    January 2026: Consolidates S3BackupDaemon, S3PushDaemon, S3NodeSyncDaemon.
    """
    daemon = get_s3_sync_daemon()
    await daemon.start()

    # Wait for daemon to run
    try:
        while daemon._running:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        await daemon.stop()


# =============================================================================
# Backward Compatibility (Deprecated)
# =============================================================================


def get_s3_backup_daemon() -> S3SyncDaemon:
    """DEPRECATED: Use get_s3_sync_daemon() instead.

    Returns the unified S3SyncDaemon for backward compatibility.
    """
    import warnings

    warnings.warn(
        "get_s3_backup_daemon() is deprecated. Use get_s3_sync_daemon() instead. "
        "Removal scheduled for Q2 2026.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_s3_sync_daemon()


def get_s3_push_daemon() -> S3SyncDaemon:
    """DEPRECATED: Use get_s3_sync_daemon() instead.

    Returns the unified S3SyncDaemon for backward compatibility.
    """
    import warnings

    warnings.warn(
        "get_s3_push_daemon() is deprecated. Use get_s3_sync_daemon() instead. "
        "Removal scheduled for Q2 2026.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_s3_sync_daemon()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    async def main() -> None:
        daemon = get_s3_sync_daemon()
        try:
            await daemon.start()
            while daemon._running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await daemon.stop()

    asyncio.run(main())

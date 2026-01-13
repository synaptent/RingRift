"""Sync Push Daemon - Push-based data sync for GPU training nodes.

December 2025: Created for push-based sync with verified cleanup.

GPU nodes generate selfplay data that needs to be synced to the coordinator
(and potentially other storage nodes) before local cleanup can occur safely.

This daemon:
1. At 50% disk: Starts pushing completed games to coordinator
2. At 70% disk: Pushes urgently (higher priority)
3. At 75% disk: Cleans up files with 2+ verified copies
4. Never deletes files without verified sync receipts

Key principle: Data is only deleted after N verified copies exist elsewhere.

Usage:
    from app.coordination.sync_push_daemon import (
        SyncPushDaemon,
        SyncPushConfig,
        get_sync_push_daemon,
    )

    # Get singleton
    daemon = get_sync_push_daemon()
    await daemon.start()

    # Or create with custom config
    config = SyncPushConfig(
        push_threshold_percent=50.0,
        min_copies_before_delete=2,
    )
    daemon = SyncPushDaemon(config)
    await daemon.start()
"""

from __future__ import annotations

import asyncio
import base64
import fcntl
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.protocols import CoordinatorStatus
from app.coordination.sync_integrity import compute_file_checksum
from app.distributed.cluster_manifest import (
    ClusterManifest,
    SyncReceipt,
    get_cluster_manifest,
)

# Dec 2025: Event emission for pipeline coordination
# Jan 2026: Migrated to event_router (app.coordination.data_events deprecated Q2 2026)
try:
    from app.coordination.event_router import DataEventType, get_router

    HAS_EVENT_BUS = True
except ImportError:
    HAS_EVENT_BUS = False

logger = logging.getLogger(__name__)

__all__ = [
    "SyncPushConfig",
    "SyncPushDaemon",
    "get_sync_push_daemon",
    "reset_sync_push_daemon",
]


# =============================================================================
# Constants
# =============================================================================

# Environment variable prefix
ENV_PREFIX = "RINGRIFT_SYNC_PUSH"

# Dec 2025: Use centralized P2P port config instead of hardcoded value
from app.config.coordination_defaults import get_p2p_port

# Maximum file size to push inline (larger files use chunked transfer)
MAX_INLINE_SIZE_MB = 50


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SyncPushConfig:
    """Configuration for push-based sync daemon.

    December 2025: Simplified - no longer inherits from DaemonConfig.
    HandlerBase uses cycle_interval directly.

    Thresholds control when push and cleanup actions occur:
    - push_threshold_percent: Start pushing at this disk usage (default: 50%)
    - urgent_threshold_percent: Push aggressively at this usage (default: 70%)
    - cleanup_threshold_percent: Safe cleanup at this usage (default: 75%)
    - min_copies_before_delete: Require N verified copies before local delete

    Interval controls:
    - check_interval_seconds: How often to check disk and push (default: 300s)
    - max_files_per_cycle: Maximum files to push per cycle (default: 10)
    """

    # Check interval (passed to HandlerBase as cycle_interval)
    check_interval_seconds: int = 300

    # Daemon control
    enabled: bool = True

    # Disk thresholds (in percent)
    push_threshold_percent: float = 50.0
    urgent_threshold_percent: float = 70.0
    cleanup_threshold_percent: float = 75.0

    # Replication requirements
    min_copies_before_delete: int = 2

    # Transfer settings
    max_files_per_cycle: int = 10
    max_file_size_mb: int = 500  # Skip files larger than this
    push_timeout_seconds: float = 300.0  # Per-file push timeout

    # Coordinator settings (auto-discovered from P2P)
    coordinator_url: str = ""
    coordinator_node_id: str = ""

    # Data paths
    data_dir: str = ""  # Auto-discovered if empty

    @classmethod
    def from_env(cls, prefix: str = ENV_PREFIX) -> "SyncPushConfig":
        """Load configuration from environment variables."""
        config = cls()
        config.enabled = os.environ.get(f"{prefix}_ENABLED", "1") == "1"

        # Thresholds
        if os.environ.get(f"{prefix}_THRESHOLD"):
            config.push_threshold_percent = float(
                os.environ.get(f"{prefix}_THRESHOLD", "50")
            )
        if os.environ.get(f"{prefix}_URGENT_THRESHOLD"):
            config.urgent_threshold_percent = float(
                os.environ.get(f"{prefix}_URGENT_THRESHOLD", "70")
            )
        if os.environ.get(f"{prefix}_CLEANUP_THRESHOLD"):
            config.cleanup_threshold_percent = float(
                os.environ.get(f"{prefix}_CLEANUP_THRESHOLD", "75")
            )

        # Replication
        if os.environ.get(f"{prefix}_MIN_COPIES"):
            config.min_copies_before_delete = int(
                os.environ.get(f"{prefix}_MIN_COPIES", "2")
            )

        # Interval
        if os.environ.get(f"{prefix}_INTERVAL"):
            config.check_interval_seconds = int(
                os.environ.get(f"{prefix}_INTERVAL", "300")
            )

        # Coordinator
        if os.environ.get(f"{prefix}_COORDINATOR_URL"):
            config.coordinator_url = os.environ.get(f"{prefix}_COORDINATOR_URL", "")

        # Data directory
        if os.environ.get(f"{prefix}_DATA_DIR"):
            config.data_dir = os.environ.get(f"{prefix}_DATA_DIR", "")

        return config


# =============================================================================
# Sync Push Daemon
# =============================================================================


class SyncPushDaemon(HandlerBase):
    """Push-based sync daemon for GPU training nodes.

    Runs on GPU nodes to proactively push selfplay data to coordinator
    before disk fills up. Only deletes local files after receiving
    verified sync receipts from N+ destinations.

    December 2025: Migrated to HandlerBase pattern.
    - Uses HandlerBase singleton (get_instance/reset_instance)
    - Uses _stats for metrics tracking

    Key behaviors:
    1. At 50% disk: Start pushing completed games
    2. At 70% disk: Push urgently with higher priority
    3. At 75% disk: Clean up files with 2+ verified copies
    4. Never delete files without verified sync receipts
    """

    def __init__(self, config: SyncPushConfig | None = None):
        """Initialize the sync push daemon.

        Args:
            config: Configuration. If None, loads from environment.
        """
        self._daemon_config = config or SyncPushConfig.from_env()

        super().__init__(
            name="SyncPushDaemon",
            config=self._daemon_config,
            cycle_interval=float(self._daemon_config.check_interval_seconds),
        )

        self._manifest: ClusterManifest | None = None
        self._session: aiohttp.ClientSession | None = None

        # Statistics
        self._files_pushed = 0
        self._bytes_pushed = 0
        self._files_cleaned = 0
        self._bytes_cleaned = 0
        self._push_failures = 0

        # Coordinator discovery
        self._coordinator_url: str = ""
        self._last_coordinator_check: float = 0.0

    @property
    def config(self) -> SyncPushConfig:
        """Get daemon configuration."""
        return self._daemon_config

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def _on_start(self) -> None:
        """Initialize resources on daemon start."""
        # Get cluster manifest
        self._manifest = get_cluster_manifest()

        # Create HTTP session
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.push_timeout_seconds)
        )

        # Discover coordinator
        await self._discover_coordinator()

        logger.info(
            f"[{self.name}] Started with thresholds: "
            f"push={self.config.push_threshold_percent}%, "
            f"urgent={self.config.urgent_threshold_percent}%, "
            f"cleanup={self.config.cleanup_threshold_percent}%"
        )

    async def _on_stop(self) -> None:
        """Cleanup resources on daemon stop."""
        if self._session:
            await self._session.close()
            self._session = None

        logger.info(
            f"[{self.name}] Stats: "
            f"pushed={self._files_pushed} files ({self._bytes_pushed / 1024 / 1024:.1f} MB), "
            f"cleaned={self._files_cleaned} files ({self._bytes_cleaned / 1024 / 1024:.1f} MB), "
            f"failures={self._push_failures}"
        )

    # =========================================================================
    # Event Emission (Dec 2025)
    # =========================================================================

    async def _emit_event(
        self,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Emit event for pipeline coordination.

        Dec 2025: Added for integration with DataPipelineOrchestrator.
        Sync events enable downstream systems to trigger exports and training.
        """
        if not HAS_EVENT_BUS:
            return

        try:
            router = get_router()
            await router.publish(event_type, {
                **payload,
                "source": self.name,
                "node_id": self.node_id,
                "timestamp": time.time(),
            })
        except (RuntimeError, OSError, ConnectionError, TimeoutError) as e:
            # Narrow to event bus errors (December 2025 exception narrowing)
            logger.debug(f"[{self.name}] Failed to emit {event_type}: {e}")

    # =========================================================================
    # Main Cycle
    # =========================================================================

    async def _run_cycle(self) -> None:
        """Run one daemon cycle.

        1. Check disk usage
        2. If above cleanup threshold, run safe cleanup
        3. If above push threshold, push pending files
        4. Update coordinator discovery if stale
        """
        if not self._manifest:
            self._manifest = get_cluster_manifest()

        # Get current disk usage
        disk_usage = self._get_disk_usage()
        if disk_usage < 0:
            logger.warning(f"[{self.name}] Could not determine disk usage")
            return

        logger.debug(
            f"[{self.name}] Disk usage: {disk_usage:.1f}% "
            f"(push={self.config.push_threshold_percent}%, "
            f"cleanup={self.config.cleanup_threshold_percent}%)"
        )

        # Refresh coordinator if stale (every 5 minutes)
        if time.time() - self._last_coordinator_check > 300:
            await self._discover_coordinator()

        # Priority 1: Safe cleanup if above cleanup threshold
        if disk_usage >= self.config.cleanup_threshold_percent:
            cleaned = await self._safe_cleanup()
            if cleaned > 0:
                logger.info(
                    f"[{self.name}] Cleaned {cleaned} files "
                    f"(disk at {disk_usage:.1f}%)"
                )
                # Recheck disk after cleanup
                disk_usage = self._get_disk_usage()

        # Priority 2: Push if above push threshold
        if disk_usage >= self.config.push_threshold_percent:
            urgent = disk_usage >= self.config.urgent_threshold_percent
            pushed = await self._push_pending_files(urgent=urgent)
            if pushed > 0:
                level = "URGENT" if urgent else "normal"
                logger.info(
                    f"[{self.name}] Pushed {pushed} files ({level})"
                )

    # =========================================================================
    # Disk Management
    # =========================================================================

    def _get_disk_usage(self) -> float:
        """Get current disk usage percentage.

        Returns:
            Disk usage as percentage (0-100), or -1 on error.
        """
        try:
            # Determine path to check
            if self.config.data_dir:
                check_path = Path(self.config.data_dir)
            elif self._manifest:
                check_path = self._manifest.db_path.parent.parent / "games"
            else:
                check_path = Path("data/games")

            if not check_path.exists():
                check_path = Path.cwd()

            usage = shutil.disk_usage(check_path)
            return (usage.used / usage.total) * 100

        except Exception as e:
            logger.warning(f"[{self.name}] Error getting disk usage: {e}")
            return -1.0

    def _get_data_dir(self) -> Path:
        """Get the data directory to scan for files to push."""
        if self.config.data_dir:
            return Path(self.config.data_dir)
        elif self._manifest:
            return self._manifest.db_path.parent.parent / "games"
        else:
            return Path("data/games")

    # =========================================================================
    # File Push
    # =========================================================================

    async def _push_pending_files(self, urgent: bool = False) -> int:
        """Push files that haven't been synced yet.

        Args:
            urgent: If True, push with higher priority and skip some checks

        Returns:
            Number of files successfully pushed

        Dec 2025: Added event emission for pipeline coordination.
        """
        if not self._coordinator_url:
            logger.warning(
                f"[{self.name}] No coordinator URL - cannot push"
            )
            return 0

        if not self._manifest:
            return 0

        # Get files needing sync
        # Use shorter max_age_hours for urgent pushes
        max_age = 0.5 if urgent else 24.0  # 30 min vs 24 hours
        pending = self._manifest.get_pending_sync_files(
            max_age_hours=max_age,
            data_dir=self._get_data_dir(),
        )

        if not pending:
            logger.debug(f"[{self.name}] No pending files to push")
            return 0

        # Limit files per cycle
        files_to_push = pending[: self.config.max_files_per_cycle]
        pushed = 0
        bytes_pushed = 0
        start_time = time.time()

        # Dec 2025: Emit sync started event
        await self._emit_event(
            DataEventType.DATA_SYNC_STARTED.value if HAS_EVENT_BUS else "sync_started",
            {
                "sync_type": "push",
                "urgent": urgent,
                "file_count": len(files_to_push),
                "target": self._coordinator_url,
            },
        )

        for file_path in files_to_push:
            try:
                # Check file size
                stat = file_path.stat()
                if stat.st_size > self.config.max_file_size_mb * 1024 * 1024:
                    logger.debug(
                        f"[{self.name}] Skipping large file: {file_path}"
                    )
                    continue

                # Push the file
                success = await self._push_file(file_path)
                if success:
                    pushed += 1
                    bytes_pushed += stat.st_size
                    self._files_pushed += 1
                    self._bytes_pushed += stat.st_size
                else:
                    self._push_failures += 1

            except Exception as e:
                logger.error(
                    f"[{self.name}] Error pushing {file_path}: {e}"
                )
                self._push_failures += 1

        # Dec 2025: Emit sync completed event
        duration = time.time() - start_time
        if pushed > 0:
            await self._emit_event(
                DataEventType.DATA_SYNC_COMPLETED.value if HAS_EVENT_BUS else "sync_completed",
                {
                    "sync_type": "push",
                    "urgent": urgent,
                    "files_pushed": pushed,
                    "bytes_pushed": bytes_pushed,
                    "duration_seconds": duration,
                    "target": self._coordinator_url,
                },
            )
        elif self._push_failures > 0:
            await self._emit_event(
                DataEventType.DATA_SYNC_FAILED.value if HAS_EVENT_BUS else "sync_failed",
                {
                    "sync_type": "push",
                    "urgent": urgent,
                    "failure_count": self._push_failures,
                    "duration_seconds": duration,
                    "target": self._coordinator_url,
                },
            )

        return pushed

    async def _push_file(self, file_path: Path) -> bool:
        """Push a single file to the coordinator.

        Args:
            file_path: Path to the file to push

        Returns:
            True if push succeeded and receipt was recorded
        """
        if not self._session:
            return False

        # Compute checksum using consolidated sync_integrity module
        try:
            checksum = compute_file_checksum(file_path)
        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.warning(
                f"[{self.name}] Error computing checksum for {file_path}: {e}"
            )
            return False

        # Prepare request
        stat = file_path.stat()
        relative_path = str(file_path)

        # For small files, send inline as base64
        use_inline = stat.st_size <= MAX_INLINE_SIZE_MB * 1024 * 1024

        try:
            if use_inline:
                # Read and encode file
                content = file_path.read_bytes()
                content_b64 = base64.b64encode(content).decode("ascii")

                payload = {
                    "file_path": relative_path,
                    "checksum": checksum,
                    "file_size": stat.st_size,
                    "source_node": self.node_id,
                    "content": content_b64,
                }
            else:
                # For large files, just notify (coordinator will pull via rsync)
                payload = {
                    "file_path": relative_path,
                    "checksum": checksum,
                    "file_size": stat.st_size,
                    "source_node": self.node_id,
                    "pull_required": True,
                }

            # Send to coordinator
            url = f"{self._coordinator_url}/sync/push"
            async with self._session.post(url, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning(
                        f"[{self.name}] Push failed for {file_path}: "
                        f"{resp.status} - {text}"
                    )
                    return False

                result = await resp.json()

            # Record receipt if verified
            if result.get("checksum_verified"):
                receipt = SyncReceipt(
                    file_path=relative_path,
                    file_checksum=checksum,
                    synced_to=result.get("node_id", self.config.coordinator_node_id),
                    synced_at=time.time(),
                    verified=True,
                    file_size=stat.st_size,
                    source_node=self.node_id,
                )
                self._manifest.register_sync_receipt(receipt)

                logger.debug(
                    f"[{self.name}] Pushed and verified: {file_path}"
                )
                return True
            else:
                # Record unverified receipt (will need verification later)
                receipt = SyncReceipt(
                    file_path=relative_path,
                    file_checksum=checksum,
                    synced_to=result.get("node_id", self.config.coordinator_node_id),
                    synced_at=time.time(),
                    verified=False,
                    file_size=stat.st_size,
                    source_node=self.node_id,
                )
                self._manifest.register_sync_receipt(receipt)

                logger.debug(
                    f"[{self.name}] Pushed (unverified): {file_path}"
                )
                return True

        except asyncio.TimeoutError:
            logger.warning(
                f"[{self.name}] Timeout pushing {file_path}"
            )
            return False
        except Exception as e:
            logger.error(f"[{self.name}] Error pushing {file_path}: {e}")
            return False

    # =========================================================================
    # Safe Cleanup
    # =========================================================================

    async def _safe_cleanup(self) -> int:
        """Delete only files with verified N-copy replication.

        This is the SAFE version - only deletes if ClusterManifest confirms
        the file exists on at least min_copies_before_delete other nodes.

        Returns:
            Number of files deleted
        """
        if not self._manifest:
            return 0

        deleted = 0
        data_dir = self._get_data_dir()

        if not data_dir.exists():
            return 0

        for db_path in data_dir.glob("*.db"):
            if db_path.name.startswith("."):
                continue

            # Skip canonical databases
            if "canonical" in db_path.name.lower():
                continue

            # Check if safe to delete
            file_path = str(db_path)
            if not self._manifest.is_safe_to_delete(
                file_path,
                min_copies=self.config.min_copies_before_delete,
            ):
                continue

            # Safe to delete - file has N+ verified copies
            # Dec 29, 2025: Add file locking to prevent race with active writes (Phase 1.3)
            # Jan 7, 2026: Wrap blocking I/O in asyncio.to_thread() for async safety

            def _delete_with_lock() -> tuple[bool, int, str | None]:
                """Blocking file deletion with lock - runs in thread pool.

                Returns:
                    (success, file_size, error_message)
                """
                try:
                    stat = db_path.stat()
                    file_size = stat.st_size

                    # Try to acquire exclusive lock before deletion
                    # This ensures no other process has the file open for writing
                    try:
                        with open(db_path, "r") as f:
                            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                            # Lock acquired - file is not in use, safe to delete
                            db_path.unlink()
                    except BlockingIOError:
                        # File is in use by another process - skip this cycle
                        return (False, 0, "file in use")
                    except (OSError, PermissionError) as e:
                        # Can't open for locking - skip
                        return (False, 0, str(e))

                    # Also remove WAL/SHM files (they're no longer needed)
                    wal_path = db_path.with_suffix(".db-wal")
                    shm_path = db_path.with_suffix(".db-shm")
                    if wal_path.exists():
                        try:
                            wal_path.unlink()
                        except OSError:
                            pass  # Best effort
                    if shm_path.exists():
                        try:
                            shm_path.unlink()
                        except OSError:
                            pass  # Best effort

                    return (True, file_size, None)
                except Exception as e:
                    return (False, 0, str(e))

            success, file_size, error = await asyncio.to_thread(_delete_with_lock)

            if not success:
                if error:
                    logger.debug(f"[{self.name}] Skipping {db_path}: {error}")
                continue

            # Clean up receipts
            self._manifest.delete_sync_receipts(file_path)

            deleted += 1
            self._files_cleaned += 1
            self._bytes_cleaned += file_size

            logger.info(
                f"[{self.name}] Safe delete: {db_path} "
                f"(verified {self.config.min_copies_before_delete}+ copies)"
            )

        return deleted

    # =========================================================================
    # Utilities
    # =========================================================================

    # _compute_sha256 removed Dec 2025 - use compute_file_checksum from sync_integrity

    async def _discover_coordinator(self) -> None:
        """Discover coordinator URL from P2P status.

        Queries local P2P daemon to find the current leader.
        """
        self._last_coordinator_check = time.time()

        # Check if explicitly configured
        if self.config.coordinator_url:
            self._coordinator_url = self.config.coordinator_url
            return

        # Try to discover from P2P
        try:
            p2p_port = get_p2p_port()
            p2p_url = f"http://localhost:{p2p_port}/status"

            async with aiohttp.ClientSession() as session:
                async with session.get(p2p_url, timeout=aiohttp.ClientTimeout(total=5.0)) as resp:
                    if resp.status == 200:
                        status = await resp.json()
                        leader_id = status.get("leader_id")
                        if leader_id:
                            # Get leader's address from peers
                            peers = status.get("peers", {})
                            if leader_id in peers:
                                peer_info = peers[leader_id]
                                leader_addr = peer_info.get("address", "")
                                if leader_addr:
                                    # Parse address (might be in host:port format)
                                    if ":" in leader_addr:
                                        host = leader_addr.split(":")[0]
                                    else:
                                        host = leader_addr
                                    self._coordinator_url = f"http://{host}:{p2p_port}"
                                    self.config.coordinator_node_id = leader_id
                                    logger.debug(
                                        f"[{self.name}] Discovered coordinator: "
                                        f"{self._coordinator_url} (leader={leader_id})"
                                    )
                                    return

        except Exception as e:
            logger.debug(
                f"[{self.name}] Could not discover coordinator: {e}"
            )

        # Fallback to localhost if no coordinator found
        if not self._coordinator_url:
            logger.warning(
                f"[{self.name}] No coordinator discovered - sync disabled"
            )

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """Return health check result for DaemonManager integration."""
        is_healthy = self.is_running and self._stats.errors_count < 10

        details = {
            "files_pushed": self._files_pushed,
            "bytes_pushed_mb": self._bytes_pushed / 1024 / 1024,
            "files_cleaned": self._files_cleaned,
            "bytes_cleaned_mb": self._bytes_cleaned / 1024 / 1024,
            "push_failures": self._push_failures,
            "coordinator_url": self._coordinator_url,
            "cycles_completed": self._stats.cycles_completed,
            "uptime_seconds": self.uptime_seconds,
            "errors_count": self._stats.errors_count,
        }

        if is_healthy:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                details=details,
            )
        else:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="Too many errors",
                details=details,
            )

    def get_status(self) -> dict[str, Any]:
        """Get daemon status for monitoring."""
        return {
            "daemon": self.name,
            "running": self._running,
            "uptime_seconds": self.uptime_seconds,
            "cycles_completed": self._stats.cycles_completed,
            "errors_count": self._stats.errors_count,
            "coordinator_url": self._coordinator_url,
            "stats": {
                "files_pushed": self._files_pushed,
                "bytes_pushed": self._bytes_pushed,
                "files_cleaned": self._files_cleaned,
                "bytes_cleaned": self._bytes_cleaned,
                "push_failures": self._push_failures,
            },
            "config": {
                "push_threshold_percent": self.config.push_threshold_percent,
                "urgent_threshold_percent": self.config.urgent_threshold_percent,
                "cleanup_threshold_percent": self.config.cleanup_threshold_percent,
                "min_copies_before_delete": self.config.min_copies_before_delete,
            },
        }


# =============================================================================
# Singleton Access (using HandlerBase class methods)
# =============================================================================


def get_sync_push_daemon() -> SyncPushDaemon:
    """Get the singleton SyncPushDaemon instance.

    Uses HandlerBase.get_instance() for thread-safe singleton access.
    """
    return SyncPushDaemon.get_instance()


def reset_sync_push_daemon() -> None:
    """Reset the singleton (for testing).

    Uses HandlerBase.reset_instance() for thread-safe cleanup.
    """
    SyncPushDaemon.reset_instance()

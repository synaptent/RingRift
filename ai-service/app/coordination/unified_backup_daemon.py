"""Unified Backup Daemon - Backs up all selfplay games to OWC and S3.

This daemon ensures ALL games from ALL sources are backed up to:
1. OWC external drive on mac-studio (/Volumes/RingRift-Data)
2. AWS S3 bucket (ringrift-models-20251214)

Key Features:
- Discovers all game databases using GameDiscovery patterns
- Syncs to OWC external drive via rsync over SSH
- Pushes to S3 directly or triggers OWC→S3 mirror
- Event-driven: responds to DATA_SYNC_COMPLETED events
- Periodic backup cycle (default: 5 minutes)

January 2026 - Created for comprehensive game data capture.

Usage:
    from app.coordination.unified_backup_daemon import (
        UnifiedBackupDaemon,
        get_unified_backup_daemon,
    )

    daemon = get_unified_backup_daemon()
    await daemon.start()

    # Or trigger manual backup
    await daemon.backup_all_databases()
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
from app.coordination.contracts import HealthCheckResult, CoordinatorStatus

logger = logging.getLogger(__name__)

# Environment variable defaults
DEFAULT_OWC_HOST = "mac-studio"
DEFAULT_OWC_BASE_PATH = "/Volumes/RingRift-Data"
DEFAULT_S3_BUCKET = "ringrift-models-20251214"
DEFAULT_S3_REGION = "us-east-1"
DEFAULT_BACKUP_INTERVAL = 300.0  # 5 minutes


@dataclass
class BackupConfig:
    """Configuration for unified backup daemon."""

    # OWC external drive settings
    owc_host: str = field(default_factory=lambda: os.getenv("OWC_HOST", DEFAULT_OWC_HOST))
    owc_base_path: str = field(
        default_factory=lambda: os.getenv("OWC_BASE_PATH", DEFAULT_OWC_BASE_PATH)
    )
    owc_ssh_key: str = field(
        default_factory=lambda: os.getenv("OWC_SSH_KEY", "~/.ssh/id_ed25519")
    )
    owc_ssh_user: str = field(
        default_factory=lambda: os.getenv("OWC_SSH_USER", "armand")
    )

    # S3 settings
    s3_bucket: str = field(
        default_factory=lambda: os.getenv("RINGRIFT_S3_BUCKET", DEFAULT_S3_BUCKET)
    )
    s3_region: str = field(
        default_factory=lambda: os.getenv("RINGRIFT_S3_REGION", DEFAULT_S3_REGION)
    )
    s3_prefix: str = field(
        default_factory=lambda: os.getenv("RINGRIFT_S3_PREFIX", "consolidated")
    )

    # Backup behavior
    backup_interval: float = field(
        default_factory=lambda: float(
            os.getenv("RINGRIFT_BACKUP_INTERVAL", str(DEFAULT_BACKUP_INTERVAL))
        )
    )
    enable_owc_backup: bool = field(
        default_factory=lambda: os.getenv("RINGRIFT_OWC_BACKUP_ENABLED", "true").lower() == "true"
    )
    enable_s3_backup: bool = field(
        default_factory=lambda: os.getenv("RINGRIFT_S3_BACKUP_ENABLED", "true").lower() == "true"
    )

    # Rsync settings
    rsync_timeout: int = 300  # 5 minutes
    rsync_bwlimit: int = 0  # No limit by default (KB/s, 0 = unlimited)


@dataclass
class BackupStats:
    """Statistics for backup operations."""

    owc_syncs_completed: int = 0
    owc_syncs_failed: int = 0
    s3_pushes_completed: int = 0
    s3_pushes_failed: int = 0
    databases_backed_up: int = 0
    last_backup_time: float = 0.0
    last_error: str = ""
    last_error_time: float = 0.0
    total_bytes_synced: int = 0


@dataclass
class DatabaseBackupResult:
    """Result of backing up a single database."""

    db_path: str
    config_key: str  # e.g., "hex8_2p"
    game_count: int
    owc_path: str | None = None  # Path on OWC if backed up
    s3_key: str | None = None  # S3 key if backed up
    owc_success: bool = False
    s3_success: bool = False
    error: str | None = None


class UnifiedBackupDaemon(HandlerBase):
    """Daemon that backs up all selfplay games to OWC and S3.

    Subscribes to:
        - DATA_SYNC_COMPLETED: Push to OWC/S3 when new data synced
        - SELFPLAY_COMPLETE: Push new games to backup
        - NPZ_EXPORT_COMPLETE: Push training data to backup

    Emits:
        - BACKUP_COMPLETED: When backup cycle completes
        - BACKUP_FAILED: When backup fails
    """

    _instance: UnifiedBackupDaemon | None = None
    _lock = asyncio.Lock()
    _event_source = "UnifiedBackupDaemon"

    def __init__(self, config: BackupConfig | None = None):
        """Initialize the backup daemon.

        Args:
            config: Optional configuration. Uses defaults from environment if not provided.
        """
        self.config = config or BackupConfig()
        super().__init__(
            name="unified_backup",
            cycle_interval=self.config.backup_interval,
        )
        self._backup_stats = BackupStats()
        self._discovery: Any = None  # Lazy loaded
        self._last_backup_hashes: dict[str, str] = {}  # Track what we've backed up

    @classmethod
    async def get_instance(cls) -> UnifiedBackupDaemon:
        """Get singleton instance."""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton for testing."""
        cls._instance = None

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Subscribe to data sync and selfplay events."""
        return {
            "DATA_SYNC_COMPLETED": self._on_data_sync_completed,
            "SELFPLAY_COMPLETE": self._on_selfplay_complete,
            "NPZ_EXPORT_COMPLETE": self._on_npz_export_complete,
        }

    async def _run_cycle(self) -> None:
        """Main backup cycle - discover and backup all databases."""
        try:
            await self.backup_all_databases()
        except Exception as e:
            self._backup_stats.last_error = str(e)
            self._backup_stats.last_error_time = time.time()
            logger.error(f"[UnifiedBackup] Backup cycle failed: {e}")

    async def backup_all_databases(self) -> int:
        """Discover and backup all game databases.

        Returns:
            Number of databases backed up.
        """
        if not self.config.enable_owc_backup and not self.config.enable_s3_backup:
            logger.debug("[UnifiedBackup] Both OWC and S3 backup disabled, skipping")
            return 0

        # Lazy load GameDiscovery
        if self._discovery is None:
            from app.utils.game_discovery import GameDiscovery
            self._discovery = GameDiscovery()

        # Find all databases
        databases = self._discovery.find_all_databases()
        if not databases:
            logger.debug("[UnifiedBackup] No databases found to backup")
            return 0

        backed_up = 0
        results: list[DatabaseBackupResult] = []

        for db_info in databases:
            if not db_info.path.exists():
                continue

            # Skip empty databases
            if db_info.game_count == 0:
                continue

            # Extract config_key from db_info
            config_key = f"{db_info.board_type}_{db_info.num_players}p"

            result = DatabaseBackupResult(
                db_path=str(db_info.path),
                config_key=config_key,
                game_count=db_info.game_count,
            )

            try:
                # Backup to OWC
                if self.config.enable_owc_backup:
                    owc_path = await self._sync_to_owc(db_info.path)
                    if owc_path:
                        self._backup_stats.owc_syncs_completed += 1
                        result.owc_success = True
                        result.owc_path = owc_path
                    else:
                        self._backup_stats.owc_syncs_failed += 1

                # Backup to S3
                if self.config.enable_s3_backup:
                    s3_key = await self._push_to_s3(db_info.path)
                    if s3_key:
                        self._backup_stats.s3_pushes_completed += 1
                        result.s3_success = True
                        result.s3_key = s3_key
                    else:
                        self._backup_stats.s3_pushes_failed += 1

                backed_up += 1
                self._backup_stats.databases_backed_up += 1
                results.append(result)

            except Exception as e:
                logger.warning(f"[UnifiedBackup] Failed to backup {db_info.path.name}: {e}")
                self._backup_stats.last_error = str(e)
                self._backup_stats.last_error_time = time.time()
                result.error = str(e)
                results.append(result)

        self._backup_stats.last_backup_time = time.time()

        if backed_up > 0:
            logger.info(f"[UnifiedBackup] Backed up {backed_up} databases")
            await self._emit_backup_completed(results)

        return backed_up

    async def _sync_to_owc(self, db_path: Path) -> str | None:
        """Sync a database to OWC external drive.

        Args:
            db_path: Path to the database file.

        Returns:
            Remote path on OWC if sync succeeded, None otherwise.
        """
        try:
            # Determine remote path based on db type
            if "canonical" in db_path.name:
                remote_subdir = "games"
            elif "training" in str(db_path.parent):
                remote_subdir = "training"
            else:
                remote_subdir = "games/selfplay"

            remote_path = f"{self.config.owc_base_path}/{remote_subdir}/{db_path.name}"
            ssh_target = f"{self.config.owc_ssh_user}@{self.config.owc_host}"
            ssh_key = os.path.expanduser(self.config.owc_ssh_key)

            # Ensure remote directory exists
            mkdir_cmd = [
                "ssh", "-i", ssh_key,
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=10",
                ssh_target,
                f"mkdir -p {self.config.owc_base_path}/{remote_subdir}"
            ]
            await asyncio.to_thread(
                subprocess.run, mkdir_cmd, check=True, capture_output=True, timeout=30
            )

            # Feb 2026: Memory-aware transfer - use scp if memory is high
            use_rsync = True
            try:
                from app.coordination.rsync_command_builder import should_use_rsync
                use_rsync = should_use_rsync()
            except ImportError:
                pass

            if not use_rsync:
                # Use scp (less memory than rsync's delta algorithm)
                scp_cmd = [
                    "scp", "-i", ssh_key,
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "ConnectTimeout=10",
                    str(db_path), f"{ssh_target}:{remote_path}",
                ]
                result = await asyncio.to_thread(
                    subprocess.run,
                    scp_cmd,
                    capture_output=True,
                    timeout=self.config.rsync_timeout,
                )
                if result.returncode == 0:
                    logger.debug(f"[UnifiedBackup] Memory-aware: scp'd {db_path.name} to OWC")
                    return remote_path
                logger.info("[UnifiedBackup] scp fallback failed, trying rsync")

            # Rsync the database
            rsync_cmd = [
                "rsync", "-avz", "--progress",
                "-e", f"ssh -i {ssh_key} -o StrictHostKeyChecking=no",
            ]
            if self.config.rsync_bwlimit > 0:
                rsync_cmd.extend(["--bwlimit", str(self.config.rsync_bwlimit)])
            rsync_cmd.extend([str(db_path), f"{ssh_target}:{remote_path}"])

            result = await asyncio.to_thread(
                subprocess.run,
                rsync_cmd,
                capture_output=True,
                timeout=self.config.rsync_timeout,
            )

            if result.returncode == 0:
                logger.debug(f"[UnifiedBackup] Synced {db_path.name} to OWC")
                return remote_path
            else:
                logger.warning(
                    f"[UnifiedBackup] rsync to OWC failed: {result.stderr.decode()[:200]}"
                )
                return None

        except subprocess.TimeoutExpired:
            logger.warning(f"[UnifiedBackup] OWC sync timed out for {db_path.name}")
            return None
        except subprocess.CalledProcessError as e:
            logger.warning(f"[UnifiedBackup] OWC sync failed: {e}")
            return None
        except Exception as e:
            logger.error(f"[UnifiedBackup] OWC sync error: {e}")
            return None

    async def _push_to_s3(self, db_path: Path) -> str | None:
        """Push a database to S3.

        Args:
            db_path: Path to the database file.

        Returns:
            S3 key if push succeeded, None otherwise.
        """
        try:
            # Determine S3 key based on db type
            if "canonical" in db_path.name:
                s3_key = f"{self.config.s3_prefix}/games/{db_path.name}"
            elif "training" in str(db_path.parent) or db_path.suffix == ".npz":
                s3_key = f"{self.config.s3_prefix}/training/{db_path.name}"
            else:
                s3_key = f"{self.config.s3_prefix}/games/selfplay/{db_path.name}"

            s3_uri = f"s3://{self.config.s3_bucket}/{s3_key}"

            # Use STANDARD_IA for cost-effective infrequent access
            cmd = [
                "aws", "s3", "cp",
                str(db_path), s3_uri,
                "--storage-class", "STANDARD_IA",
                "--region", self.config.s3_region,
            ]

            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                timeout=self.config.rsync_timeout,
            )

            if result.returncode == 0:
                logger.debug(f"[UnifiedBackup] Pushed {db_path.name} to S3")
                # Track file size
                try:
                    self._backup_stats.total_bytes_synced += db_path.stat().st_size
                except OSError:
                    pass
                return s3_key
            else:
                logger.warning(
                    f"[UnifiedBackup] S3 push failed: {result.stderr.decode()[:200]}"
                )
                return None

        except subprocess.TimeoutExpired:
            logger.warning(f"[UnifiedBackup] S3 push timed out for {db_path.name}")
            return None
        except FileNotFoundError:
            logger.warning("[UnifiedBackup] AWS CLI not found - S3 backup disabled")
            return None
        except Exception as e:
            logger.error(f"[UnifiedBackup] S3 push error: {e}")
            return None

    async def trigger_owc_s3_mirror(self) -> bool:
        """Trigger the OWC→S3 mirror script on mac-studio.

        Returns:
            True if trigger succeeded.
        """
        try:
            ssh_target = f"{self.config.owc_ssh_user}@{self.config.owc_host}"
            ssh_key = os.path.expanduser(self.config.owc_ssh_key)

            cmd = [
                "ssh", "-i", ssh_key,
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=10",
                ssh_target,
                "cd ~/ringrift/ai-service && python scripts/owc_s3_mirror.py --tier 1"
            ]

            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                timeout=600,  # 10 minute timeout for full mirror
            )

            if result.returncode == 0:
                logger.info("[UnifiedBackup] Triggered OWC→S3 mirror successfully")
                return True
            else:
                logger.warning(
                    f"[UnifiedBackup] OWC→S3 mirror failed: {result.stderr.decode()[:200]}"
                )
                return False

        except Exception as e:
            logger.error(f"[UnifiedBackup] Failed to trigger OWC→S3 mirror: {e}")
            return False

    async def _emit_backup_completed(self, results: list[DatabaseBackupResult]) -> None:
        """Emit BACKUP_COMPLETED event with per-database details.

        Args:
            results: List of backup results for each database.
        """
        # Build per-database details for ClusterManifest tracking
        backup_details = []
        for result in results:
            detail = {
                "config_key": result.config_key,
                "game_count": result.game_count,
                "db_path": result.db_path,
            }
            if result.owc_success and result.owc_path:
                detail["owc_path"] = result.owc_path
                detail["owc_host"] = self.config.owc_host
            if result.s3_success and result.s3_key:
                detail["s3_key"] = result.s3_key
                detail["s3_bucket"] = self.config.s3_bucket
            if result.error:
                detail["error"] = result.error
            backup_details.append(detail)

        await self._safe_emit_event_async(
            "BACKUP_COMPLETED",
            {
                "source": "unified_backup",
                "databases_backed_up": len([r for r in results if r.owc_success or r.s3_success]),
                "timestamp": time.time(),
                "owc_enabled": self.config.enable_owc_backup,
                "s3_enabled": self.config.enable_s3_backup,
                "owc_host": self.config.owc_host,
                "owc_base_path": self.config.owc_base_path,
                "s3_bucket": self.config.s3_bucket,
                # Per-database backup details for ClusterManifest
                "backup_details": backup_details,
            }
        )

    async def _on_data_sync_completed(self, event: Any) -> None:
        """Handle DATA_SYNC_COMPLETED - push synced data to OWC/S3."""
        # Jan 2026: Fix RouterEvent handling - extract payload first
        payload = getattr(event, "payload", {}) or (event if isinstance(event, dict) else {})
        db_path_str = payload.get("db_path")
        if not db_path_str:
            return

        db_path = Path(db_path_str)
        if not db_path.exists():
            return

        needs_owc = payload.get("needs_owc_backup", True)
        needs_s3 = payload.get("needs_s3_backup", True)

        if needs_owc and self.config.enable_owc_backup:
            await self._sync_to_owc(db_path)

        if needs_s3 and self.config.enable_s3_backup:
            await self._push_to_s3(db_path)

    async def _on_selfplay_complete(self, event: Any) -> None:
        """Handle SELFPLAY_COMPLETE - backup new games."""
        # Jan 2026: Fix RouterEvent handling - extract payload first
        payload = getattr(event, "payload", {}) or (event if isinstance(event, dict) else {})
        db_path_str = payload.get("db_path")
        if not db_path_str:
            return

        db_path = Path(db_path_str)
        if db_path.exists():
            if self.config.enable_owc_backup:
                await self._sync_to_owc(db_path)
            if self.config.enable_s3_backup:
                await self._push_to_s3(db_path)

    async def _on_npz_export_complete(self, event: Any) -> None:
        """Handle NPZ_EXPORT_COMPLETE - backup training data."""
        # Jan 2026: Fix RouterEvent handling - extract payload first
        payload = getattr(event, "payload", {}) or (event if isinstance(event, dict) else {})
        npz_path_str = payload.get("output_path") or payload.get("npz_path")
        if not npz_path_str:
            return

        npz_path = Path(npz_path_str)
        if npz_path.exists():
            if self.config.enable_owc_backup:
                await self._sync_to_owc(npz_path)
            if self.config.enable_s3_backup:
                await self._push_to_s3(npz_path)

    def health_check(self) -> HealthCheckResult:
        """Return health status for daemon manager."""
        # Calculate error rate
        total_ops = (
            self._backup_stats.owc_syncs_completed + self._backup_stats.owc_syncs_failed +
            self._backup_stats.s3_pushes_completed + self._backup_stats.s3_pushes_failed
        )
        error_rate = 0.0
        if total_ops > 0:
            errors = self._backup_stats.owc_syncs_failed + self._backup_stats.s3_pushes_failed
            error_rate = errors / total_ops

        # Determine status
        if not self._running:
            status = CoordinatorStatus.STOPPED
        elif error_rate > 0.5:
            status = CoordinatorStatus.DEGRADED
        elif self._backup_stats.last_error and time.time() - self._backup_stats.last_error_time < 300:
            status = CoordinatorStatus.DEGRADED
        else:
            status = CoordinatorStatus.HEALTHY

        return HealthCheckResult(
            name="unified_backup",
            status=status,
            details={
                "owc_syncs_completed": self._backup_stats.owc_syncs_completed,
                "owc_syncs_failed": self._backup_stats.owc_syncs_failed,
                "s3_pushes_completed": self._backup_stats.s3_pushes_completed,
                "s3_pushes_failed": self._backup_stats.s3_pushes_failed,
                "databases_backed_up": self._backup_stats.databases_backed_up,
                "last_backup_time": self._backup_stats.last_backup_time,
                "total_bytes_synced": self._backup_stats.total_bytes_synced,
                "error_rate": error_rate,
                "last_error": self._backup_stats.last_error if self._backup_stats.last_error else None,
                "owc_enabled": self.config.enable_owc_backup,
                "s3_enabled": self.config.enable_s3_backup,
            },
        )


# Singleton accessor functions
_daemon_instance: UnifiedBackupDaemon | None = None
_daemon_lock = asyncio.Lock()


async def get_unified_backup_daemon() -> UnifiedBackupDaemon:
    """Get singleton instance of UnifiedBackupDaemon."""
    global _daemon_instance
    async with _daemon_lock:
        if _daemon_instance is None:
            _daemon_instance = UnifiedBackupDaemon()
        return _daemon_instance


def reset_unified_backup_daemon() -> None:
    """Reset singleton for testing."""
    global _daemon_instance
    _daemon_instance = None
    UnifiedBackupDaemon.reset_instance()


# Factory function for backward compatibility
def create_unified_backup_daemon(config: BackupConfig | None = None) -> UnifiedBackupDaemon:
    """Create a new UnifiedBackupDaemon instance.

    Args:
        config: Optional configuration.

    Returns:
        New daemon instance (not singleton).
    """
    return UnifiedBackupDaemon(config=config)

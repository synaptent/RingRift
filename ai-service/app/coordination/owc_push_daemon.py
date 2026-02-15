"""OWC Push Daemon - Backs up training data to OWC external drive (January 2026).

This daemon periodically pushes canonical databases, NPZ training files, and models
to the OWC external drive (/Volumes/RingRift-Data) on mac-studio for:
- Local backup redundancy
- Recovery from cluster disruption
- Cross-location data availability

The daemon only pushes files that have been modified since the last push,
minimizing disk I/O and network bandwidth.

Usage:
    from app.coordination.owc_push_daemon import OWCPushDaemon, get_owc_push_daemon

    daemon = get_owc_push_daemon()
    await daemon.start()

Environment Variables:
    RINGRIFT_OWC_PUSH_ENABLED: Enable/disable daemon (default: true)
    RINGRIFT_OWC_PUSH_INTERVAL: Push interval in seconds (default: 21600 = 6 hours)
    RINGRIFT_OWC_PUSH_PATH: OWC mount path (default: /Volumes/RingRift-Data)
    OWC_HOST: Remote host with OWC drive (default: mac-studio)
    OWC_USER: SSH user (default: armand)
    OWC_SSH_KEY: SSH key path (default: ~/.ssh/id_ed25519)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.protocols import CoordinatorStatus
from app.config.coordination_defaults import build_ssh_options

logger = logging.getLogger(__name__)


def _is_running_on_owc_host(owc_host: str) -> bool:
    """Check if we're running on the OWC host itself."""
    hostname = socket.gethostname().lower()
    owc_host_lower = owc_host.lower()

    local_patterns = [
        owc_host_lower,
        f"{owc_host_lower}.local",
        owc_host_lower.replace("-", ""),
        owc_host_lower.replace("-", "").replace(".", ""),
    ]

    hostname_normalized = hostname.replace("-", "").replace(".", "").replace("_", "")

    for pattern in local_patterns:
        pattern_normalized = pattern.replace("-", "").replace(".", "").replace("_", "")
        if hostname_normalized.startswith(pattern_normalized):
            return True

    if owc_host_lower in ("localhost", "127.0.0.1", "::1"):
        return True

    return False


@dataclass
class OWCPushConfig:
    """Configuration for OWC push daemon."""

    enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_OWC_PUSH_ENABLED", "true"
        ).lower() == "true"
    )
    push_interval: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_OWC_PUSH_INTERVAL", "21600")  # 6 hours
        )
    )
    owc_base_path: str = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_OWC_PUSH_PATH", "/Volumes/RingRift-Data"
        )
    )
    owc_host: str = field(
        default_factory=lambda: os.environ.get("OWC_HOST", "mac-studio")
    )
    owc_user: str = field(
        default_factory=lambda: os.environ.get("OWC_USER", "armand")
    )
    owc_ssh_key: str = field(
        default_factory=lambda: os.environ.get(
            "OWC_SSH_KEY", os.path.expanduser("~/.ssh/id_ed25519")
        )
    )

    # Timeouts
    ssh_timeout: int = 60
    rsync_timeout: int = 1800  # 30 minutes for large files

    # Subdirectories on OWC
    canonical_db_subdir: str = "consolidated/games"
    training_subdir: str = "consolidated/training"
    models_subdir: str = "models"


@dataclass
class OWCPushStats:
    """Statistics for OWC push operations."""

    total_files_pushed: int = 0
    total_bytes_pushed: int = 0
    last_push_time: float = 0.0
    push_errors: int = 0
    last_error: str | None = None
    consecutive_failures: int = 0


class OWCPushDaemon(HandlerBase):
    """Daemon that pushes training data to OWC external drive for backup.

    Monitors local canonical databases, NPZ training files, and models,
    pushing any modified files to OWC for redundancy and recovery.
    """

    _instance: OWCPushDaemon | None = None

    def __init__(self, config: OWCPushConfig | None = None):
        """Initialize OWC push daemon.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or OWCPushConfig()
        super().__init__(name="owc_push", cycle_interval=self.config.push_interval)

        self._push_stats = OWCPushStats()
        self._last_push_times: dict[str, float] = {}  # path -> mtime at last push
        self._file_checksums: dict[str, str] = {}  # path -> SHA256 for dedup
        self._base_path = Path(os.environ.get("RINGRIFT_BASE_PATH", "."))
        self._owc_available = True

        # Detect if running locally on OWC host
        self._is_local = _is_running_on_owc_host(self.config.owc_host)
        if self._is_local:
            logger.info(
                f"[OWCPush] Running on OWC host '{self.config.owc_host}', "
                f"using local file access"
            )

    @classmethod
    def get_instance(cls) -> OWCPushDaemon:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    async def _check_owc_available(self) -> bool:
        """Check if OWC drive is accessible."""
        if self._is_local:
            owc_path = Path(self.config.owc_base_path)
            return owc_path.exists() and owc_path.is_dir()

        # Check via SSH
        try:
            ssh_opts = build_ssh_options(
                key_path=self.config.owc_ssh_key,
                include_keepalive=False,
            )
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    "ssh",
                    "-o", f"ConnectTimeout={self.config.ssh_timeout}",
                    "-i", self.config.owc_ssh_key,
                    f"{self.config.owc_user}@{self.config.owc_host}",
                    f"ls -d '{self.config.owc_base_path}' 2>/dev/null",
                ],
                capture_output=True,
                text=True,
                timeout=self.config.ssh_timeout,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.debug(f"[OWCPush] OWC availability check failed: {e}")
            return False

    async def _run_cycle(self) -> None:
        """Run one push cycle."""
        if not self.config.enabled:
            logger.debug("[OWCPush] Disabled via config, skipping cycle")
            return

        # February 2026: Block when coordinator is low on RAM/disk
        from app.utils.resource_guard import coordinator_resource_gate
        if not coordinator_resource_gate("OWC_PUSH"):
            return

        # Check OWC availability
        if not await self._check_owc_available():
            if self._owc_available:
                if self._is_local:
                    logger.warning(
                        f"[OWCPush] OWC drive not available at {self.config.owc_base_path}"
                    )
                else:
                    logger.warning(
                        f"[OWCPush] OWC drive not available at "
                        f"{self.config.owc_host}:{self.config.owc_base_path}"
                    )
            self._owc_available = False
            self._push_stats.consecutive_failures += 1
            return

        if not self._owc_available:
            logger.info("[OWCPush] OWC drive is now available")
        self._owc_available = True
        self._push_stats.consecutive_failures = 0

        try:
            files_pushed = 0

            # Push canonical databases
            files_pushed += await self._push_canonical_databases()

            # Push NPZ training files
            files_pushed += await self._push_training_files()

            # Push models (less frequently, they're larger)
            if self._should_push_models():
                files_pushed += await self._push_models()

            self._push_stats.last_push_time = time.time()
            if files_pushed > 0:
                logger.info(
                    f"[OWCPush] Cycle complete: "
                    f"{files_pushed} files pushed, "
                    f"{self._push_stats.total_files_pushed} total"
                )
            else:
                logger.debug("[OWCPush] Cycle complete: no files needed pushing")

        except Exception as e:
            self._push_stats.push_errors += 1
            self._push_stats.last_error = str(e)
            self._push_stats.consecutive_failures += 1
            logger.error(f"[OWCPush] Push cycle failed: {e}")

    async def _push_canonical_databases(self) -> int:
        """Push canonical game databases to OWC."""
        games_dir = self._base_path / "data" / "games"
        if not games_dir.exists():
            return 0

        files_pushed = 0
        for db_path in games_dir.glob("canonical_*.db"):
            dest_path = f"{self.config.canonical_db_subdir}/{db_path.name}"
            if await self._push_if_modified(db_path, dest_path):
                files_pushed += 1

        return files_pushed

    async def _push_training_files(self) -> int:
        """Push NPZ training files to OWC."""
        training_dir = self._base_path / "data" / "training"
        if not training_dir.exists():
            return 0

        files_pushed = 0
        for npz_path in training_dir.glob("*.npz"):
            dest_path = f"{self.config.training_subdir}/{npz_path.name}"
            if await self._push_if_modified(npz_path, dest_path):
                files_pushed += 1

        return files_pushed

    async def _push_models(self) -> int:
        """Push model checkpoints to OWC."""
        models_dir = self._base_path / "models"
        if not models_dir.exists():
            return 0

        files_pushed = 0
        for model_path in models_dir.glob("canonical_*.pth"):
            dest_path = f"{self.config.models_subdir}/{model_path.name}"
            if await self._push_if_modified(model_path, dest_path):
                files_pushed += 1

        return files_pushed

    def _should_push_models(self) -> bool:
        """Determine if models should be pushed this cycle.

        Models are large, so we push them less frequently (every 4 cycles).
        """
        cycle_count = getattr(self, "_cycle_count", 0)
        self._cycle_count = cycle_count + 1
        return cycle_count % 4 == 0

    def _compute_file_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of file for deduplication."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def _push_if_modified(self, local_path: Path, dest_rel_path: str) -> bool:
        """Push file to OWC if it has been modified since last push.

        Args:
            local_path: Local file path
            dest_rel_path: Relative path within OWC base directory

        Returns:
            True if file was pushed, False if skipped (not modified)
        """
        if not local_path.exists():
            return False

        try:
            mtime = local_path.stat().st_mtime
            last_push = self._last_push_times.get(str(local_path), 0)

            # Skip if not modified
            if mtime <= last_push:
                logger.debug(f"[OWCPush] Skipping {local_path.name} (not modified)")
                return False

            # Compute checksum for deduplication
            checksum = await asyncio.to_thread(self._compute_file_checksum, local_path)
            if checksum == self._file_checksums.get(str(local_path)):
                logger.debug(f"[OWCPush] Skipping {local_path.name} (checksum match)")
                self._last_push_times[str(local_path)] = mtime
                return False

            # Perform the push
            dest_full_path = f"{self.config.owc_base_path}/{dest_rel_path}"

            if self._is_local:
                success = await self._push_local(local_path, dest_full_path)
            else:
                success = await self._push_remote(local_path, dest_full_path)

            if success:
                file_size = local_path.stat().st_size
                self._last_push_times[str(local_path)] = mtime
                self._file_checksums[str(local_path)] = checksum
                self._push_stats.total_files_pushed += 1
                self._push_stats.total_bytes_pushed += file_size
                logger.info(
                    f"[OWCPush] Pushed {local_path.name} "
                    f"({file_size / (1024*1024):.1f} MB)"
                )
                return True
            else:
                self._push_stats.push_errors += 1
                return False

        except (OSError, IOError) as e:
            logger.warning(f"[OWCPush] Push error for {local_path.name}: {e}")
            self._push_stats.push_errors += 1
            return False

    async def _push_local(self, local_path: Path, dest_path: str) -> bool:
        """Push file using local file copy (when on OWC host)."""
        try:
            dest = Path(dest_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(shutil.copy2, local_path, dest)
            return True
        except (OSError, IOError) as e:
            logger.warning(f"[OWCPush] Local copy failed: {e}")
            return False

    async def _push_remote(self, local_path: Path, dest_path: str) -> bool:
        """Push file using rsync over SSH."""
        try:
            # Ensure destination directory exists
            dest_dir = os.path.dirname(dest_path)
            mkdir_result = await asyncio.to_thread(
                subprocess.run,
                [
                    "ssh",
                    "-i", self.config.owc_ssh_key,
                    "-o", f"ConnectTimeout={self.config.ssh_timeout}",
                    f"{self.config.owc_user}@{self.config.owc_host}",
                    f"mkdir -p '{dest_dir}'",
                ],
                capture_output=True,
                text=True,
                timeout=self.config.ssh_timeout,
            )
            if mkdir_result.returncode != 0:
                logger.warning(f"[OWCPush] Failed to create directory: {mkdir_result.stderr}")

            # Rsync the file
            remote_dest = f"{self.config.owc_user}@{self.config.owc_host}:{dest_path}"
            ssh_opts = f"ssh -i {self.config.owc_ssh_key} -o ConnectTimeout={self.config.ssh_timeout}"

            result = await asyncio.to_thread(
                subprocess.run,
                [
                    "rsync", "-avz", "--progress",
                    "-e", ssh_opts,
                    str(local_path),
                    remote_dest,
                ],
                capture_output=True,
                text=True,
                timeout=self.config.rsync_timeout,
            )

            if result.returncode == 0:
                return True
            else:
                logger.warning(f"[OWCPush] Rsync failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.warning(f"[OWCPush] Rsync timed out for {local_path.name}")
            return False
        except (OSError, IOError) as e:
            logger.warning(f"[OWCPush] Remote push error: {e}")
            return False

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Get event subscriptions for this daemon.

        Subscribes to data events to trigger immediate pushes.
        """
        return {
            "DATA_SYNC_COMPLETED": self._on_data_sync_completed,
            "TRAINING_COMPLETED": self._on_training_completed,
            "NPZ_EXPORT_COMPLETE": self._on_npz_export_complete,
            "CONSOLIDATION_COMPLETE": self._on_consolidation_complete,
        }

    async def _on_data_sync_completed(self, event: Any) -> None:
        """Handle data sync completion - push synced data to OWC."""
        # Jan 2026: Fix RouterEvent handling - extract payload first
        payload = getattr(event, "payload", {}) or (event if isinstance(event, dict) else {})
        if payload.get("needs_owc_backup"):
            db_path_str = payload.get("db_path")
            if db_path_str:
                db_path = Path(db_path_str)
                if db_path.exists() and db_path.name.startswith("canonical_"):
                    dest = f"{self.config.canonical_db_subdir}/{db_path.name}"
                    await self._push_if_modified(db_path, dest)

    async def _on_training_completed(self, event: Any) -> None:
        """Handle training completion - push model to OWC."""
        # Jan 2026: Fix RouterEvent handling - extract payload first
        payload = getattr(event, "payload", {}) or (event if isinstance(event, dict) else {})
        model_path_str = payload.get("model_path")
        if model_path_str:
            model_path = Path(model_path_str)
            if model_path.exists():
                dest = f"{self.config.models_subdir}/{model_path.name}"
                await self._push_if_modified(model_path, dest)

    async def _on_npz_export_complete(self, event: Any) -> None:
        """Handle NPZ export completion - push training data to OWC."""
        # Jan 2026: Fix RouterEvent handling - extract payload first
        payload = getattr(event, "payload", {}) or (event if isinstance(event, dict) else {})
        npz_path_str = payload.get("npz_path") or payload.get("output_path")
        if npz_path_str:
            npz_path = Path(npz_path_str)
            if npz_path.exists():
                dest = f"{self.config.training_subdir}/{npz_path.name}"
                await self._push_if_modified(npz_path, dest)

    async def _on_consolidation_complete(self, event: Any) -> None:
        """Handle consolidation completion - push canonical DB to OWC."""
        # Jan 2026: Fix RouterEvent handling - extract payload first
        payload = getattr(event, "payload", {}) or (event if isinstance(event, dict) else {})
        db_path_str = payload.get("canonical_db_path") or payload.get("db_path")
        if db_path_str:
            db_path = Path(db_path_str)
            if db_path.exists():
                dest = f"{self.config.canonical_db_subdir}/{db_path.name}"
                await self._push_if_modified(db_path, dest)

    def health_check(self) -> HealthCheckResult:
        """Return health status."""
        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="OWCPush not running",
            )

        if not self._owc_available:
            return HealthCheckResult(
                healthy=True,  # Still healthy, just OWC unavailable
                status=CoordinatorStatus.RUNNING,
                message="OWC drive not available",
                details={
                    "owc_host": self.config.owc_host,
                    "is_local": self._is_local,
                    "consecutive_failures": self._push_stats.consecutive_failures,
                },
            )

        mode = "local" if self._is_local else "remote"
        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"OWC push active ({mode}), {self._push_stats.total_files_pushed} files pushed",
            details={
                "cycles_completed": self._stats.cycles_completed,
                "total_files_pushed": self._push_stats.total_files_pushed,
                "total_mb_pushed": round(
                    self._push_stats.total_bytes_pushed / (1024 * 1024), 2
                ),
                "owc_available": self._owc_available,
                "is_local": self._is_local,
                "push_errors": self._push_stats.push_errors,
                "last_push_time": self._push_stats.last_push_time,
            },
        )

    def get_stats(self) -> dict[str, Any]:
        """Get current daemon statistics."""
        return {
            "total_files_pushed": self._push_stats.total_files_pushed,
            "total_bytes_pushed": self._push_stats.total_bytes_pushed,
            "total_mb_pushed": round(
                self._push_stats.total_bytes_pushed / (1024 * 1024), 2
            ),
            "last_push_time": self._push_stats.last_push_time,
            "push_errors": self._push_stats.push_errors,
            "last_error": self._push_stats.last_error,
            "tracked_files": len(self._last_push_times),
            "owc_available": self._owc_available,
            "is_local": self._is_local,
        }


def get_owc_push_daemon() -> OWCPushDaemon:
    """Get the singleton OWC push daemon instance."""
    return OWCPushDaemon.get_instance()


def reset_owc_push_daemon() -> None:
    """Reset the singleton instance (for testing)."""
    OWCPushDaemon.reset_instance()


# Factory function for daemon_runners.py
async def create_owc_push() -> None:
    """Create and run OWC push daemon (January 2026).

    Pushes all canonical databases, training NPZ files, and models to OWC
    external drive for backup and recovery.
    """
    daemon = get_owc_push_daemon()
    await daemon.start()

    try:
        while daemon._running:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        await daemon.stop()

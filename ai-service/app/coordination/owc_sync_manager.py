"""OWC Sync Manager - Unified bidirectional sync with OWC external drive (January 2026).

Consolidates:
- ExternalDriveSyncDaemon (PULL from cluster/S3 to OWC)
- OWCPushDaemon (PUSH local data to OWC)
- DualBackupDaemon (orchestrate S3+OWC backups)
- UnifiedBackupDaemon (backup to both destinations)

This daemon provides bidirectional synchronization:
- PUSH: Local canonical databases, NPZ, and models → OWC drive
- PULL: Cluster nodes and S3 → OWC drive

Key features:
- Unified event subscriptions for all backup triggers
- Checksum verification for data integrity
- Multi-transport support (local copy, rsync, S3 sync)
- Automatic detection of local vs remote OWC access
- Configurable intervals per data type and direction

Usage:
    from app.coordination.owc_sync_manager import (
        OWCSyncManager,
        get_owc_sync_manager,
        OWCSyncConfig,
    )

    manager = get_owc_sync_manager()
    await manager.start()

Environment Variables:
    RINGRIFT_OWC_SYNC_ENABLED: Enable/disable manager (default: true)
    RINGRIFT_OWC_BASE_PATH: OWC mount path (default: /Volumes/RingRift-Data)
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
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.protocols import CoordinatorStatus
from app.coordination.event_emission_helpers import safe_emit_event
from app.config.coordination_defaults import build_ssh_options
from app.config.cluster_config import (
    ExternalStorageConfig,
    get_cluster_nodes,
    get_sync_routing,
    ClusterNode,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class SyncDirection(str, Enum):
    """Direction of sync operation."""

    PUSH = "push"  # Local → OWC
    PULL = "pull"  # Cluster/S3 → OWC


class DataCategory(str, Enum):
    """Category of data being synced."""

    GAMES = "games"
    NPZ = "npz"
    MODELS = "models"


DEFAULT_S3_BUCKET = "ringrift-models-20251214"


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
class OWCSyncConfig:
    """Configuration for OWC sync manager."""

    # Enable/disable
    enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_OWC_SYNC_ENABLED", "true"
        ).lower() == "true"
    )

    # OWC connection settings
    owc_base_path: str = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_OWC_BASE_PATH", "/Volumes/RingRift-Data"
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

    # PUSH intervals (local → OWC)
    push_interval: float = 21600.0  # 6 hours default push cycle
    push_models_every_n_cycles: int = 4  # Models every 4th push cycle

    # PULL intervals (cluster/S3 → OWC)
    pull_games_interval: float = 1800.0  # 30 minutes
    pull_npz_interval: float = 3600.0  # 1 hour
    pull_models_interval: float = 1800.0  # 30 minutes

    # S3 settings for pull
    s3_bucket: str = DEFAULT_S3_BUCKET
    s3_consolidated_prefix: str = "consolidated"

    # Timeouts
    ssh_timeout: int = 60
    rsync_timeout: int = 1800  # 30 minutes for large files

    # Bandwidth limits (KB/s)
    bandwidth_limit_kbps: int = 50000  # 50 MB/s

    # Concurrency
    max_concurrent_syncs: int = field(
        default_factory=lambda: int(
            os.environ.get("RINGRIFT_OWC_SYNC_MAX_CONCURRENT", "5")
        )
    )

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 60.0

    # OWC subdirectories
    canonical_db_subdir: str = "consolidated/games"
    training_subdir: str = "consolidated/training"
    models_subdir: str = "models"
    selfplay_repository_subdir: str = "selfplay_repository"

    # Feature flags
    enable_push: bool = True
    enable_pull: bool = True
    enable_s3_pull: bool = True
    verify_checksums: bool = True


@dataclass
class OWCSyncStats:
    """Statistics for OWC sync operations."""

    # Push stats
    push_files_synced: int = 0
    push_bytes_synced: int = 0
    push_errors: int = 0
    last_push_time: float = 0.0

    # Pull stats
    pull_nodes_synced: int = 0
    pull_files_synced: int = 0
    pull_bytes_synced: int = 0
    pull_errors: int = 0
    last_pull_time: float = 0.0

    # S3 pull stats
    s3_npz_synced: int = 0
    s3_models_synced: int = 0
    s3_errors: int = 0
    last_s3_sync_time: float = 0.0

    # Overall
    consecutive_failures: int = 0
    last_error: str | None = None


# =============================================================================
# Unified OWC Sync Manager
# =============================================================================


class OWCSyncManager(HandlerBase):
    """Unified manager for bidirectional OWC synchronization.

    January 2026: Consolidates ExternalDriveSyncDaemon, OWCPushDaemon,
    DualBackupDaemon, and UnifiedBackupDaemon into a single daemon.

    Handles:
    - PUSH: Local canonical databases, NPZ, models → OWC drive
    - PULL: Cluster nodes games → OWC drive
    - PULL: S3 consolidated NPZ and models → OWC drive
    """

    _instance: OWCSyncManager | None = None

    def __init__(self, config: OWCSyncConfig | None = None):
        """Initialize OWC sync manager.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or OWCSyncConfig()

        # Use the shorter interval as the cycle interval
        min_interval = min(
            self.config.push_interval,
            self.config.pull_games_interval,
            self.config.pull_npz_interval,
            self.config.pull_models_interval,
        )
        super().__init__(
            name="owc_sync_manager",
            cycle_interval=min(60.0, min_interval / 10),  # Check frequently
        )

        self._sync_stats = OWCSyncStats()
        self._base_path = Path(os.environ.get("RINGRIFT_BASE_PATH", "."))

        # Track last sync times per category/direction
        self._last_push_time = 0.0
        self._last_pull_games_time = 0.0
        self._last_pull_npz_time = 0.0
        self._last_pull_models_time = 0.0
        self._push_cycle_count = 0

        # Checksum cache for push deduplication
        self._file_checksums: dict[str, str] = {}
        self._file_mtimes: dict[str, float] = {}

        # OWC availability
        self._owc_available = True
        self._is_local = _is_running_on_owc_host(self.config.owc_host)

        # External storage config (for pull operations)
        self._storage_config: ExternalStorageConfig | None = None
        if self.config.enable_pull:
            self._storage_config = self._detect_external_storage()

        if self._is_local:
            logger.info(
                f"[OWCSyncManager] Running on OWC host '{self.config.owc_host}', "
                f"using local file access"
            )

    @classmethod
    def get_instance(cls) -> OWCSyncManager:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    def _detect_external_storage(self) -> ExternalStorageConfig | None:
        """Detect external storage configuration for this host."""
        try:
            hostname = socket.gethostname().lower()
            host_aliases = [hostname, hostname.split(".")[0]]
            if "mac-studio" in hostname.lower():
                host_aliases.append("mac-studio")

            sync_routing = get_sync_routing()
            for storage in sync_routing.allowed_external_storage:
                if storage.host.lower() in host_aliases:
                    logger.info(
                        f"[OWCSyncManager] Detected external storage for {storage.host}: "
                        f"{storage.path}"
                    )
                    return storage

            logger.info(
                f"[OWCSyncManager] No external storage configured for {hostname}"
            )
            return None

        except (socket.error, socket.gaierror) as e:
            logger.warning(f"[OWCSyncManager] Network error detecting storage: {e}")
            return None
        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.warning(f"[OWCSyncManager] Config access error: {e}")
            return None
        except (KeyError, AttributeError, TypeError) as e:
            logger.error(f"[OWCSyncManager] Invalid config structure: {e}")
            return None

    # =========================================================================
    # OWC Availability Check
    # =========================================================================

    async def _check_owc_available(self) -> bool:
        """Check if OWC drive is accessible."""
        if self._is_local:
            owc_path = Path(self.config.owc_base_path)
            return owc_path.exists() and owc_path.is_dir()

        # Check via SSH
        try:
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
            logger.debug(f"[OWCSyncManager] OWC availability check failed: {e}")
            return False

    def _has_aws_credentials(self) -> bool:
        """Check if AWS credentials are available."""
        if os.getenv("AWS_ACCESS_KEY_ID"):
            return True

        creds_file = Path.home() / ".aws" / "credentials"
        if creds_file.exists():
            import configparser
            config = configparser.ConfigParser()
            try:
                config.read(creds_file)
                for section in config.sections():
                    if config.get(section, "aws_access_key_id", fallback=None):
                        return True
            except (configparser.Error, OSError, UnicodeDecodeError):
                pass

        return False

    # =========================================================================
    # Main Cycle
    # =========================================================================

    async def _run_cycle(self) -> None:
        """Run one sync cycle."""
        if not self.config.enabled:
            logger.debug("[OWCSyncManager] Disabled via config, skipping cycle")
            return

        # Check OWC availability
        if not await self._check_owc_available():
            if self._owc_available:
                logger.warning(
                    f"[OWCSyncManager] OWC drive not available at "
                    f"{self.config.owc_base_path}"
                )
            self._owc_available = False
            self._sync_stats.consecutive_failures += 1
            return

        if not self._owc_available:
            logger.info("[OWCSyncManager] OWC drive is now available")
        self._owc_available = True

        now = time.time()
        errors_this_cycle = 0

        try:
            # PUSH operations (local → OWC)
            if (
                self.config.enable_push
                and now - self._last_push_time >= self.config.push_interval
            ):
                push_errors = await self._run_push_cycle()
                errors_this_cycle += push_errors
                self._last_push_time = now

            # PULL games from cluster
            if (
                self.config.enable_pull
                and self._storage_config
                and self._storage_config.receive_games
                and now - self._last_pull_games_time >= self.config.pull_games_interval
            ):
                pull_errors = await self._pull_games_from_cluster()
                errors_this_cycle += pull_errors
                self._last_pull_games_time = now

            # PULL NPZ from S3
            if (
                self.config.enable_s3_pull
                and self._has_aws_credentials()
                and now - self._last_pull_npz_time >= self.config.pull_npz_interval
            ):
                s3_errors = await self._pull_npz_from_s3()
                errors_this_cycle += s3_errors
                self._last_pull_npz_time = now

            # PULL models from S3
            if (
                self.config.enable_s3_pull
                and self._has_aws_credentials()
                and now - self._last_pull_models_time >= self.config.pull_models_interval
            ):
                s3_errors = await self._pull_models_from_s3()
                errors_this_cycle += s3_errors
                self._last_pull_models_time = now

            # Update stats
            if errors_this_cycle == 0:
                self._sync_stats.consecutive_failures = 0
            else:
                self._sync_stats.consecutive_failures += 1

        except Exception as e:
            self._sync_stats.consecutive_failures += 1
            self._sync_stats.last_error = str(e)
            logger.error(f"[OWCSyncManager] Cycle error: {e}")

    # =========================================================================
    # PUSH Operations (local → OWC)
    # =========================================================================

    async def _run_push_cycle(self) -> int:
        """Run a push cycle to sync local data to OWC.

        Returns:
            Number of errors encountered.
        """
        logger.debug("[OWCSyncManager] Starting push cycle")
        errors = 0
        self._push_cycle_count += 1

        # Push canonical databases
        errors += await self._push_canonical_databases()

        # Push NPZ training files
        errors += await self._push_training_files()

        # Push models (less frequently)
        if self._push_cycle_count % self.config.push_models_every_n_cycles == 0:
            errors += await self._push_models()

        self._sync_stats.last_push_time = time.time()

        if errors == 0:
            logger.debug("[OWCSyncManager] Push cycle complete - no errors")
        else:
            logger.warning(f"[OWCSyncManager] Push cycle complete with {errors} errors")

        return errors

    async def _push_canonical_databases(self) -> int:
        """Push canonical game databases to OWC."""
        games_dir = self._base_path / "data" / "games"
        if not games_dir.exists():
            return 0

        errors = 0
        for db_path in games_dir.glob("canonical_*.db"):
            dest_path = f"{self.config.canonical_db_subdir}/{db_path.name}"
            if not await self._push_if_modified(db_path, dest_path):
                errors += 1

        return errors

    async def _push_training_files(self) -> int:
        """Push NPZ training files to OWC."""
        training_dir = self._base_path / "data" / "training"
        if not training_dir.exists():
            return 0

        errors = 0
        for npz_path in training_dir.glob("*.npz"):
            dest_path = f"{self.config.training_subdir}/{npz_path.name}"
            if not await self._push_if_modified(npz_path, dest_path):
                errors += 1

        return errors

    async def _push_models(self) -> int:
        """Push model checkpoints to OWC."""
        models_dir = self._base_path / "models"
        if not models_dir.exists():
            return 0

        errors = 0
        for model_path in models_dir.glob("canonical_*.pth"):
            dest_path = f"{self.config.models_subdir}/{model_path.name}"
            if not await self._push_if_modified(model_path, dest_path):
                errors += 1

        return errors

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
            True if successful (pushed or skipped), False on error.
        """
        if not local_path.exists():
            return True  # Not an error, just nothing to do

        try:
            mtime = local_path.stat().st_mtime
            last_mtime = self._file_mtimes.get(str(local_path), 0)

            # Skip if not modified
            if mtime <= last_mtime:
                return True

            # Compute checksum for deduplication
            checksum = await asyncio.to_thread(self._compute_file_checksum, local_path)
            if checksum == self._file_checksums.get(str(local_path)):
                self._file_mtimes[str(local_path)] = mtime
                return True

            # Perform the push
            dest_full_path = f"{self.config.owc_base_path}/{dest_rel_path}"

            if self._is_local:
                success = await self._push_local(local_path, dest_full_path)
            else:
                success = await self._push_remote(local_path, dest_full_path)

            if success:
                file_size = local_path.stat().st_size
                self._file_mtimes[str(local_path)] = mtime
                self._file_checksums[str(local_path)] = checksum
                self._sync_stats.push_files_synced += 1
                self._sync_stats.push_bytes_synced += file_size
                logger.info(
                    f"[OWCSyncManager] Pushed {local_path.name} "
                    f"({file_size / (1024*1024):.1f} MB)"
                )
                return True
            else:
                self._sync_stats.push_errors += 1
                return False

        except (OSError, IOError) as e:
            logger.warning(f"[OWCSyncManager] Push error for {local_path.name}: {e}")
            self._sync_stats.push_errors += 1
            return False

    async def _push_local(self, local_path: Path, dest_path: str) -> bool:
        """Push file using local file copy (when on OWC host)."""
        try:
            dest = Path(dest_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(shutil.copy2, local_path, dest)
            return True
        except (OSError, IOError) as e:
            logger.warning(f"[OWCSyncManager] Local copy failed: {e}")
            return False

    async def _push_remote(self, local_path: Path, dest_path: str) -> bool:
        """Push file using rsync over SSH."""
        try:
            # Ensure destination directory exists
            dest_dir = os.path.dirname(dest_path)
            await asyncio.to_thread(
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

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            logger.warning(f"[OWCSyncManager] Rsync timed out for {local_path.name}")
            return False
        except (OSError, IOError) as e:
            logger.warning(f"[OWCSyncManager] Remote push error: {e}")
            return False

    # =========================================================================
    # PULL Operations (cluster/S3 → OWC)
    # =========================================================================

    async def _pull_games_from_cluster(self) -> int:
        """Pull games from cluster nodes to OWC.

        Returns:
            Number of errors encountered.
        """
        if not self._storage_config:
            return 0

        logger.info("[OWCSyncManager] Starting games pull from cluster")

        nodes = self._get_sync_source_nodes()
        if not nodes:
            logger.debug("[OWCSyncManager] No cluster nodes available for games pull")
            return 0

        # Determine destination directory
        games_subdir = self._storage_config.subdirs.get(
            "games", self.config.selfplay_repository_subdir
        )
        dest_base = Path(self._storage_config.path) / games_subdir
        dest_base.mkdir(parents=True, exist_ok=True)

        # Sync from nodes with limited concurrency
        semaphore = asyncio.Semaphore(self.config.max_concurrent_syncs)
        errors = 0

        async def sync_node(node: ClusterNode) -> int:
            async with semaphore:
                return await self._pull_games_from_node(node, dest_base)

        results = await asyncio.gather(
            *[sync_node(node) for node in nodes],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                errors += 1
                logger.error(f"[OWCSyncManager] Node sync exception: {result}")
            elif isinstance(result, int):
                errors += result

        self._sync_stats.last_pull_time = time.time()

        logger.info(
            f"[OWCSyncManager] Games pull complete - "
            f"{self._sync_stats.pull_nodes_synced} nodes processed, {errors} errors"
        )

        # Emit completion event
        await self._emit_sync_completed("games", len(nodes))

        return errors

    async def _pull_games_from_node(
        self,
        node: ClusterNode,
        dest_base: Path,
    ) -> int:
        """Pull games from a single node.

        Returns:
            Number of errors (0 or 1).
        """
        try:
            source_path = f"{node.ringrift_path}/data/games/"
            node_dest = dest_base / node.name
            node_dest.mkdir(parents=True, exist_ok=True)

            ssh_cmd = build_ssh_options(
                key_path=os.path.expanduser(node.ssh_key) if node.ssh_key else None,
                port=node.ssh_port if node.ssh_port and node.ssh_port != 22 else None,
                node_id=node.name,
                include_keepalive=False,
            )

            cmd = [
                "rsync", "-avz", "--progress",
                "--include=*.db",
                "--include=*.db-wal",
                "--include=*.db-shm",
                "--exclude=*",
                f"--timeout={self.config.rsync_timeout}",
            ]

            if self.config.bandwidth_limit_kbps > 0:
                cmd.append(f"--bwlimit={self.config.bandwidth_limit_kbps}")

            cmd.extend([
                "-e", ssh_cmd,
                f"{node.ssh_user}@{node.best_ip}:{source_path}",
                str(node_dest) + "/",
            ])

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                self._sync_stats.pull_nodes_synced += 1
                self._sync_stats.pull_files_synced += 1
                logger.info(f"[OWCSyncManager] Pulled games from {node.name}")
                return 0
            else:
                self._sync_stats.pull_errors += 1
                logger.warning(
                    f"[OWCSyncManager] Failed to pull games from {node.name}: "
                    f"{stderr.decode()[:200]}"
                )
                return 1

        except Exception as e:
            self._sync_stats.pull_errors += 1
            logger.error(f"[OWCSyncManager] Error pulling games from {node.name}: {e}")
            return 1

    def _get_sync_source_nodes(self) -> list[ClusterNode]:
        """Get list of cluster nodes to sync games from."""
        try:
            nodes = get_cluster_nodes()

            source_nodes = []
            for name, node in nodes.items():
                if not node.is_active:
                    continue
                if not node.best_ip:
                    continue
                if not node.ssh_user:
                    continue
                if node.is_coordinator:
                    continue
                source_nodes.append(node)

            return source_nodes

        except Exception as e:
            logger.error(f"[OWCSyncManager] Failed to get cluster nodes: {e}")
            return []

    async def _pull_npz_from_s3(self) -> int:
        """Pull NPZ training data from S3.

        Returns:
            Number of errors (0 or 1).
        """
        if not self._storage_config:
            return 0

        logger.info("[OWCSyncManager] Starting NPZ pull from S3")

        npz_subdir = self._storage_config.subdirs.get("npz", "canonical_data")
        dest_path = Path(self._storage_config.path) / npz_subdir
        dest_path.mkdir(parents=True, exist_ok=True)

        s3_source = (
            f"s3://{self.config.s3_bucket}/"
            f"{self.config.s3_consolidated_prefix}/training/"
        )

        cmd = ["aws", "s3", "sync", s3_source, str(dest_path)]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                self._sync_stats.s3_npz_synced += 1
                self._sync_stats.last_s3_sync_time = time.time()
                logger.info(f"[OWCSyncManager] Pulled NPZ from S3 to {dest_path}")
                await self._emit_sync_completed("npz", 1)
                return 0
            else:
                self._sync_stats.s3_errors += 1
                logger.warning(
                    f"[OWCSyncManager] Failed to pull NPZ from S3: "
                    f"{stderr.decode()[:200]}"
                )
                return 1

        except Exception as e:
            self._sync_stats.s3_errors += 1
            logger.error(f"[OWCSyncManager] Error pulling NPZ from S3: {e}")
            return 1

    async def _pull_models_from_s3(self) -> int:
        """Pull models from S3.

        Returns:
            Number of errors (0 or 1).
        """
        if not self._storage_config:
            return 0

        logger.info("[OWCSyncManager] Starting models pull from S3")

        models_subdir = self._storage_config.subdirs.get("models", "canonical_models")
        dest_path = Path(self._storage_config.path) / models_subdir
        dest_path.mkdir(parents=True, exist_ok=True)

        s3_source = (
            f"s3://{self.config.s3_bucket}/"
            f"{self.config.s3_consolidated_prefix}/models/"
        )

        cmd = [
            "aws", "s3", "sync", s3_source, str(dest_path),
            "--exclude", "*",
            "--include", "*.pth",
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                self._sync_stats.s3_models_synced += 1
                self._sync_stats.last_s3_sync_time = time.time()
                logger.info(f"[OWCSyncManager] Pulled models from S3 to {dest_path}")
                await self._emit_sync_completed("models", 1)
                return 0
            else:
                self._sync_stats.s3_errors += 1
                logger.warning(
                    f"[OWCSyncManager] Failed to pull models from S3: "
                    f"{stderr.decode()[:200]}"
                )
                return 1

        except Exception as e:
            self._sync_stats.s3_errors += 1
            logger.error(f"[OWCSyncManager] Error pulling models from S3: {e}")
            return 1

    # =========================================================================
    # Event Subscriptions
    # =========================================================================

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Get event subscriptions for immediate sync triggers."""
        return {
            "DATA_SYNC_COMPLETED": self._on_data_sync_completed,
            "TRAINING_COMPLETED": self._on_training_completed,
            "NPZ_EXPORT_COMPLETE": self._on_npz_export_complete,
            "CONSOLIDATION_COMPLETE": self._on_consolidation_complete,
            "MODEL_PROMOTED": self._on_model_promoted,
        }

    async def _on_data_sync_completed(self, event: dict[str, Any]) -> None:
        """Handle data sync completion - push synced data to OWC."""
        if event.get("needs_owc_backup"):
            db_path_str = event.get("db_path")
            if db_path_str:
                db_path = Path(db_path_str)
                if db_path.exists() and db_path.name.startswith("canonical_"):
                    dest = f"{self.config.canonical_db_subdir}/{db_path.name}"
                    await self._push_if_modified(db_path, dest)

    async def _on_training_completed(self, event: dict[str, Any]) -> None:
        """Handle training completion - push model to OWC."""
        model_path_str = event.get("model_path")
        if model_path_str:
            model_path = Path(model_path_str)
            if model_path.exists():
                dest = f"{self.config.models_subdir}/{model_path.name}"
                await self._push_if_modified(model_path, dest)

    async def _on_npz_export_complete(self, event: dict[str, Any]) -> None:
        """Handle NPZ export completion - push training data to OWC."""
        npz_path_str = event.get("npz_path") or event.get("output_path")
        if npz_path_str:
            npz_path = Path(npz_path_str)
            if npz_path.exists():
                dest = f"{self.config.training_subdir}/{npz_path.name}"
                await self._push_if_modified(npz_path, dest)

    async def _on_consolidation_complete(self, event: dict[str, Any]) -> None:
        """Handle consolidation completion - push canonical DB to OWC."""
        db_path_str = event.get("canonical_db_path") or event.get("db_path")
        if db_path_str:
            db_path = Path(db_path_str)
            if db_path.exists():
                dest = f"{self.config.canonical_db_subdir}/{db_path.name}"
                await self._push_if_modified(db_path, dest)

    async def _on_model_promoted(self, event: dict[str, Any]) -> None:
        """Handle model promotion - ensure model is backed up to OWC."""
        model_path_str = event.get("model_path") or event.get("new_model_path")
        if model_path_str:
            model_path = Path(model_path_str)
            if model_path.exists():
                dest = f"{self.config.models_subdir}/{model_path.name}"
                await self._push_if_modified(model_path, dest)

    # =========================================================================
    # Event Emission
    # =========================================================================

    async def _emit_sync_completed(self, data_type: str, items_count: int) -> None:
        """Emit DATA_BACKUP_COMPLETED event after successful sync."""
        safe_emit_event(
            "data_backup_completed",
            {
                "data_type": data_type,
                "items_count": items_count,
                "storage_path": self.config.owc_base_path,
                "sync_type": "owc_sync_manager",
                "stats": {
                    "push_files": self._sync_stats.push_files_synced,
                    "pull_nodes": self._sync_stats.pull_nodes_synced,
                    "s3_synced": self._sync_stats.s3_npz_synced + self._sync_stats.s3_models_synced,
                },
            },
            context="OWCSyncManager",
        )

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """Return health status."""
        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="OWCSyncManager not running",
            )

        if not self._owc_available:
            return HealthCheckResult(
                healthy=True,  # Still healthy, just OWC unavailable
                status=CoordinatorStatus.RUNNING,
                message="OWC drive not available",
                details=self._get_health_details(),
            )

        # Check for high error rate
        total_ops = (
            self._sync_stats.push_files_synced
            + self._sync_stats.pull_nodes_synced
            + self._sync_stats.s3_npz_synced
            + self._sync_stats.s3_models_synced
        )
        total_errors = (
            self._sync_stats.push_errors
            + self._sync_stats.pull_errors
            + self._sync_stats.s3_errors
        )

        if total_ops > 0:
            error_rate = total_errors / (total_ops + total_errors)
            if error_rate > 0.5:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"High sync error rate: {error_rate:.1%}",
                    details=self._get_health_details(),
                )

        # Check for too many consecutive failures
        if self._sync_stats.consecutive_failures >= 5:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"Too many consecutive failures: {self._sync_stats.consecutive_failures}",
                details=self._get_health_details(),
            )

        mode = "local" if self._is_local else "remote"
        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=(
                f"OWCSyncManager active ({mode}), "
                f"{self._sync_stats.push_files_synced} pushed, "
                f"{self._sync_stats.pull_nodes_synced} pulled"
            ),
            details=self._get_health_details(),
        )

    def _get_health_details(self) -> dict[str, Any]:
        """Get detailed health information."""
        return {
            "cycles_completed": self._stats.cycles_completed,
            "owc_available": self._owc_available,
            "is_local": self._is_local,
            "push_files_synced": self._sync_stats.push_files_synced,
            "push_bytes_mb": round(
                self._sync_stats.push_bytes_synced / (1024 * 1024), 2
            ),
            "push_errors": self._sync_stats.push_errors,
            "pull_nodes_synced": self._sync_stats.pull_nodes_synced,
            "pull_files_synced": self._sync_stats.pull_files_synced,
            "pull_errors": self._sync_stats.pull_errors,
            "s3_npz_synced": self._sync_stats.s3_npz_synced,
            "s3_models_synced": self._sync_stats.s3_models_synced,
            "s3_errors": self._sync_stats.s3_errors,
            "consecutive_failures": self._sync_stats.consecutive_failures,
            "last_error": self._sync_stats.last_error,
            "last_push_time": self._sync_stats.last_push_time,
            "last_pull_time": self._sync_stats.last_pull_time,
            "last_s3_sync_time": self._sync_stats.last_s3_sync_time,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get current daemon statistics."""
        return self._get_health_details()


# =============================================================================
# Singleton Access
# =============================================================================


def get_owc_sync_manager() -> OWCSyncManager:
    """Get the singleton OWC sync manager instance."""
    return OWCSyncManager.get_instance()


def reset_owc_sync_manager() -> None:
    """Reset the singleton instance (for testing)."""
    OWCSyncManager.reset_instance()


# =============================================================================
# Factory function for daemon_runners.py
# =============================================================================


async def create_owc_sync_manager() -> None:
    """Create and run OWC sync manager (January 2026).

    Unified bidirectional sync with OWC external drive, consolidating
    ExternalDriveSyncDaemon, OWCPushDaemon, DualBackupDaemon, and
    UnifiedBackupDaemon.
    """
    manager = get_owc_sync_manager()
    await manager.start()

    try:
        while manager._running:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        await manager.stop()


# =============================================================================
# Backward Compatibility - Deprecated Functions
# =============================================================================


def get_external_drive_sync_daemon() -> OWCSyncManager:
    """Get OWC sync manager (backward compatibility).

    .. deprecated:: January 2026
        Use :func:`get_owc_sync_manager` instead.
    """
    warnings.warn(
        "get_external_drive_sync_daemon() is deprecated, use get_owc_sync_manager()",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_owc_sync_manager()


def get_owc_push_daemon() -> OWCSyncManager:
    """Get OWC sync manager (backward compatibility).

    .. deprecated:: January 2026
        Use :func:`get_owc_sync_manager` instead.
    """
    warnings.warn(
        "get_owc_push_daemon() is deprecated, use get_owc_sync_manager()",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_owc_sync_manager()


def get_dual_backup_daemon() -> OWCSyncManager:
    """Get OWC sync manager (backward compatibility).

    .. deprecated:: January 2026
        Use :func:`get_owc_sync_manager` instead.
    """
    warnings.warn(
        "get_dual_backup_daemon() is deprecated, use get_owc_sync_manager()",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_owc_sync_manager()


def get_unified_backup_daemon() -> OWCSyncManager:
    """Get OWC sync manager (backward compatibility).

    .. deprecated:: January 2026
        Use :func:`get_owc_sync_manager` instead.
    """
    warnings.warn(
        "get_unified_backup_daemon() is deprecated, use get_owc_sync_manager()",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_owc_sync_manager()

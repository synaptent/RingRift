"""External Drive Sync Daemon - Pulls cluster data to external storage (OWC drive).

December 2025: Created as part of data sync infrastructure fix.

This daemon pulls valuable data from:
1. Cluster nodes -> External storage (games, databases)
2. S3 consolidated -> External storage (NPZ training data, models)

It ONLY runs on coordinator nodes that have external storage configured
in distributed_hosts.yaml under `allowed_external_storage`.

Configuration example (distributed_hosts.yaml):
    sync_routing:
      allowed_external_storage:
        - host: mac-studio
          path: /Volumes/RingRift-Data
          receive_games: true
          receive_npz: true
          receive_models: true
          subdirs:
            games: selfplay_repository
            npz: canonical_data
            models: canonical_models
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

from app.config.cluster_config import (
    ExternalStorageConfig,
    get_cluster_nodes,
    get_sync_routing,
    ClusterNode,
)
from app.coordination.protocols import HealthCheckResult, CoordinatorStatus

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SYNC_INTERVAL = 1800  # 30 minutes
DEFAULT_S3_BUCKET = "ringrift-models-20251214"


@dataclass
class ExternalDriveSyncConfig:
    """Configuration for external drive sync daemon."""

    # Sync intervals
    games_sync_interval: float = 1800.0  # Pull games every 30 min
    npz_sync_interval: float = 3600.0    # Pull NPZ every hour
    models_sync_interval: float = 1800.0  # Pull models every 30 min

    # S3 settings
    s3_bucket: str = DEFAULT_S3_BUCKET
    s3_consolidated_prefix: str = "consolidated"

    # Bandwidth limits (KB/s per node)
    bandwidth_limit_kbps: int = 50000  # 50 MB/s default

    # Rsync options
    rsync_timeout: int = 600  # 10 min timeout per node
    max_concurrent_syncs: int = 3  # Parallel node syncs

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 30.0

    # Feature flags
    sync_games: bool = True
    sync_npz: bool = True
    sync_models: bool = True

    # Dry run mode
    dry_run: bool = False


# =============================================================================
# Daemon Implementation
# =============================================================================


class ExternalDriveSyncDaemon:
    """Daemon that pulls data from cluster nodes and S3 to external storage.

    This daemon:
    1. Discovers cluster nodes with game data
    2. Rsyncs games from each node to external storage
    3. Pulls consolidated NPZ from S3 (if credentials available)
    4. Pulls canonical models from S3

    Only runs on coordinator nodes with external_storage configured.
    """

    def __init__(
        self,
        config: ExternalDriveSyncConfig | None = None,
        storage_config: ExternalStorageConfig | None = None,
    ):
        """Initialize the daemon.

        Args:
            config: Daemon configuration
            storage_config: External storage configuration (from distributed_hosts.yaml)
        """
        self.config = config or ExternalDriveSyncConfig()
        self._storage_config = storage_config

        # State tracking
        self._running = False
        self._start_time = 0.0
        self._last_games_sync = 0.0
        self._last_npz_sync = 0.0
        self._last_models_sync = 0.0
        self._cycles_completed = 0
        self._errors_count = 0

        # Sync stats
        self._stats = {
            "games_synced": 0,
            "npz_synced": 0,
            "models_synced": 0,
            "bytes_transferred": 0,
            "nodes_synced": 0,
            "sync_errors": 0,
        }

        # Get storage config if not provided
        if self._storage_config is None:
            self._storage_config = self._detect_external_storage()

    def _detect_external_storage(self) -> ExternalStorageConfig | None:
        """Detect external storage configuration for this host."""
        try:
            hostname = socket.gethostname().lower()
            # Also check common aliases
            host_aliases = [hostname, hostname.split(".")[0]]
            if "mac-studio" in hostname.lower():
                host_aliases.append("mac-studio")

            sync_routing = get_sync_routing()
            for storage in sync_routing.allowed_external_storage:
                if storage.host.lower() in host_aliases:
                    logger.info(
                        f"[ExternalDriveSync] Detected external storage for {storage.host}: "
                        f"{storage.path}"
                    )
                    return storage

            logger.info(
                f"[ExternalDriveSync] No external storage configured for {hostname}"
            )
            return None

        except Exception as e:
            logger.warning(f"[ExternalDriveSync] Failed to detect external storage: {e}")
            return None

    @property
    def uptime_seconds(self) -> float:
        """Get daemon uptime."""
        if self._start_time > 0:
            return time.time() - self._start_time
        return 0.0

    @property
    def is_enabled(self) -> bool:
        """Check if daemon should run on this host."""
        if self._storage_config is None:
            return False

        # Check if storage path exists
        storage_path = Path(self._storage_config.path)
        if not storage_path.exists():
            logger.warning(
                f"[ExternalDriveSync] Storage path does not exist: {storage_path}"
            )
            return False

        return True

    async def start(self) -> None:
        """Start the daemon."""
        if not self.is_enabled:
            logger.info(
                "[ExternalDriveSync] Daemon disabled - no external storage configured "
                "or storage path not mounted"
            )
            return

        self._running = True
        self._start_time = time.time()

        logger.info(
            f"[ExternalDriveSync] Starting daemon - syncing to "
            f"{self._storage_config.path}"
        )

        try:
            await self._run_loop()
        finally:
            self._running = False

    async def stop(self) -> None:
        """Stop the daemon."""
        self._running = False
        logger.info("[ExternalDriveSync] Stopping daemon")

    async def _run_loop(self) -> None:
        """Main daemon loop."""
        while self._running:
            try:
                now = time.time()

                # Check if games sync is due
                if (
                    self.config.sync_games
                    and self._storage_config.receive_games
                    and now - self._last_games_sync >= self.config.games_sync_interval
                ):
                    await self._sync_games_from_cluster()
                    self._last_games_sync = now

                # Check if NPZ sync is due
                if (
                    self.config.sync_npz
                    and self._storage_config.receive_npz
                    and now - self._last_npz_sync >= self.config.npz_sync_interval
                ):
                    await self._sync_npz_from_s3()
                    self._last_npz_sync = now

                # Check if models sync is due
                if (
                    self.config.sync_models
                    and self._storage_config.receive_models
                    and now - self._last_models_sync >= self.config.models_sync_interval
                ):
                    await self._sync_models_from_s3()
                    self._last_models_sync = now

                self._cycles_completed += 1

                # Sleep before next check
                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._errors_count += 1
                logger.error(f"[ExternalDriveSync] Error in sync loop: {e}")
                await asyncio.sleep(60)

    # =========================================================================
    # Games Sync (from cluster nodes)
    # =========================================================================

    async def _sync_games_from_cluster(self) -> None:
        """Sync games from all cluster nodes to external storage."""
        logger.info("[ExternalDriveSync] Starting games sync from cluster nodes")

        nodes = self._get_sync_source_nodes()
        if not nodes:
            logger.info("[ExternalDriveSync] No cluster nodes available for games sync")
            return

        # Determine destination directory
        games_subdir = self._storage_config.subdirs.get("games", "selfplay_repository")
        dest_base = Path(self._storage_config.path) / games_subdir
        dest_base.mkdir(parents=True, exist_ok=True)

        # Sync from each node in parallel (limited concurrency)
        semaphore = asyncio.Semaphore(self.config.max_concurrent_syncs)

        async def sync_node(node: ClusterNode):
            async with semaphore:
                await self._sync_games_from_node(node, dest_base)

        tasks = [sync_node(node) for node in nodes]
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(
            f"[ExternalDriveSync] Games sync complete - "
            f"{self._stats['nodes_synced']} nodes processed"
        )

    async def _sync_games_from_node(
        self,
        node: ClusterNode,
        dest_base: Path,
    ) -> bool:
        """Sync games from a single node.

        Args:
            node: Cluster node to sync from
            dest_base: Base destination directory

        Returns:
            True if sync succeeded
        """
        try:
            # Build source path
            source_path = f"{node.ringrift_path}/data/games/"

            # Build destination per-node subdirectory
            node_dest = dest_base / node.name
            node_dest.mkdir(parents=True, exist_ok=True)

            # Build rsync command
            ssh_cmd = self._build_ssh_command(node)
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

            if self.config.dry_run:
                cmd.append("--dry-run")

            cmd.extend([
                "-e", ssh_cmd,
                f"{node.ssh_user}@{node.best_ip}:{source_path}",
                str(node_dest) + "/",
            ])

            logger.debug(f"[ExternalDriveSync] Running: {' '.join(cmd[:6])}...")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                self._stats["nodes_synced"] += 1
                self._stats["games_synced"] += 1
                logger.info(f"[ExternalDriveSync] Synced games from {node.name}")
                return True
            else:
                self._stats["sync_errors"] += 1
                logger.warning(
                    f"[ExternalDriveSync] Failed to sync games from {node.name}: "
                    f"{stderr.decode()[:200]}"
                )
                return False

        except Exception as e:
            self._stats["sync_errors"] += 1
            logger.error(f"[ExternalDriveSync] Error syncing games from {node.name}: {e}")
            return False

    def _build_ssh_command(self, node: ClusterNode) -> str:
        """Build SSH command for rsync."""
        ssh_parts = ["ssh"]

        if node.ssh_port and node.ssh_port != 22:
            ssh_parts.append(f"-p {node.ssh_port}")

        if node.ssh_key:
            ssh_key = os.path.expanduser(node.ssh_key)
            ssh_parts.append(f"-i {ssh_key}")

        ssh_parts.extend(["-o", "StrictHostKeyChecking=no"])
        ssh_parts.extend(["-o", "ConnectTimeout=30"])

        return " ".join(ssh_parts)

    def _get_sync_source_nodes(self) -> list[ClusterNode]:
        """Get list of cluster nodes to sync games from."""
        try:
            nodes = get_cluster_nodes()

            # Filter to active GPU nodes with SSH access
            source_nodes = []
            for name, node in nodes.items():
                if not node.is_active:
                    continue
                if not node.best_ip:
                    continue
                if not node.ssh_user:
                    continue
                # Skip coordinators (they don't generate games)
                if node.is_coordinator:
                    continue
                source_nodes.append(node)

            logger.debug(
                f"[ExternalDriveSync] Found {len(source_nodes)} source nodes for games sync"
            )
            return source_nodes

        except Exception as e:
            logger.error(f"[ExternalDriveSync] Failed to get cluster nodes: {e}")
            return []

    # =========================================================================
    # S3 Sync (NPZ and models)
    # =========================================================================

    def _has_aws_credentials(self) -> bool:
        """Check if AWS credentials are available."""
        # Check env var
        if os.getenv("AWS_ACCESS_KEY_ID"):
            return True

        # Check credentials file
        creds_file = Path.home() / ".aws" / "credentials"
        if creds_file.exists():
            import configparser
            config = configparser.ConfigParser()
            try:
                config.read(creds_file)
                for section in config.sections():
                    if config.get(section, "aws_access_key_id", fallback=None):
                        return True
            except Exception:
                pass

        return False

    async def _sync_npz_from_s3(self) -> None:
        """Sync consolidated NPZ training data from S3."""
        if not self._has_aws_credentials():
            logger.debug("[ExternalDriveSync] Skipping S3 NPZ sync - no AWS credentials")
            return

        logger.info("[ExternalDriveSync] Starting NPZ sync from S3")

        npz_subdir = self._storage_config.subdirs.get("npz", "canonical_data")
        dest_path = Path(self._storage_config.path) / npz_subdir
        dest_path.mkdir(parents=True, exist_ok=True)

        s3_source = f"s3://{self.config.s3_bucket}/{self.config.s3_consolidated_prefix}/training/"

        cmd = ["aws", "s3", "sync", s3_source, str(dest_path)]

        if self.config.dry_run:
            cmd.append("--dryrun")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                self._stats["npz_synced"] += 1
                logger.info(f"[ExternalDriveSync] Synced NPZ from S3 to {dest_path}")
            else:
                self._stats["sync_errors"] += 1
                logger.warning(
                    f"[ExternalDriveSync] Failed to sync NPZ from S3: "
                    f"{stderr.decode()[:200]}"
                )

        except Exception as e:
            self._stats["sync_errors"] += 1
            logger.error(f"[ExternalDriveSync] Error syncing NPZ from S3: {e}")

    async def _sync_models_from_s3(self) -> None:
        """Sync canonical models from S3."""
        if not self._has_aws_credentials():
            logger.debug("[ExternalDriveSync] Skipping S3 models sync - no AWS credentials")
            return

        logger.info("[ExternalDriveSync] Starting models sync from S3")

        models_subdir = self._storage_config.subdirs.get("models", "canonical_models")
        dest_path = Path(self._storage_config.path) / models_subdir
        dest_path.mkdir(parents=True, exist_ok=True)

        s3_source = f"s3://{self.config.s3_bucket}/{self.config.s3_consolidated_prefix}/models/"

        cmd = ["aws", "s3", "sync", s3_source, str(dest_path), "--exclude", "*", "--include", "*.pth"]

        if self.config.dry_run:
            cmd.append("--dryrun")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                self._stats["models_synced"] += 1
                logger.info(f"[ExternalDriveSync] Synced models from S3 to {dest_path}")
            else:
                self._stats["sync_errors"] += 1
                logger.warning(
                    f"[ExternalDriveSync] Failed to sync models from S3: "
                    f"{stderr.decode()[:200]}"
                )

        except Exception as e:
            self._stats["sync_errors"] += 1
            logger.error(f"[ExternalDriveSync] Error syncing models from S3: {e}")

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """Return health check result for daemon protocol."""
        details = {
            "running": self._running,
            "enabled": self.is_enabled,
            "uptime_seconds": self.uptime_seconds,
            "cycles_completed": self._cycles_completed,
            "errors_count": self._errors_count,
            "storage_path": self._storage_config.path if self._storage_config else None,
            "stats": self._stats.copy(),
            "last_games_sync": self._last_games_sync,
            "last_npz_sync": self._last_npz_sync,
            "last_models_sync": self._last_models_sync,
        }

        if not self.is_enabled:
            return HealthCheckResult(
                healthy=True,  # Not unhealthy, just disabled
                status=CoordinatorStatus.STOPPED,
                message="External drive sync disabled - no external storage configured",
                details=details,
            )

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="External drive sync daemon not running",
                details=details,
            )

        # Check for high error rate
        total_ops = sum([
            self._stats["games_synced"],
            self._stats["npz_synced"],
            self._stats["models_synced"],
        ])
        if total_ops > 0:
            error_rate = self._stats["sync_errors"] / (total_ops + self._stats["sync_errors"])
            if error_rate > 0.5:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"High sync error rate: {error_rate:.1%}",
                    details=details,
                )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"External drive sync running - {self._stats['nodes_synced']} nodes synced",
            details=details,
        )

    def get_status(self) -> dict[str, Any]:
        """Get daemon status for monitoring."""
        health = self.health_check()
        return {
            "name": "ExternalDriveSyncDaemon",
            "running": self._running,
            "enabled": self.is_enabled,
            "uptime_seconds": self.uptime_seconds,
            "storage_path": self._storage_config.path if self._storage_config else None,
            "config": {
                "sync_games": self.config.sync_games,
                "sync_npz": self.config.sync_npz,
                "sync_models": self.config.sync_models,
                "games_interval": self.config.games_sync_interval,
                "npz_interval": self.config.npz_sync_interval,
                "models_interval": self.config.models_sync_interval,
            },
            "health": {
                "healthy": health.healthy,
                "status": health.status.value if hasattr(health.status, "value") else str(health.status),
                "message": health.message,
            },
            **health.details,
        }


# =============================================================================
# Singleton Access
# =============================================================================

_daemon_instance: ExternalDriveSyncDaemon | None = None


def get_external_drive_sync_daemon() -> ExternalDriveSyncDaemon:
    """Get the singleton ExternalDriveSyncDaemon instance."""
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = ExternalDriveSyncDaemon()
    return _daemon_instance


def reset_external_drive_sync_daemon() -> None:
    """Reset the singleton (for testing)."""
    global _daemon_instance
    _daemon_instance = None

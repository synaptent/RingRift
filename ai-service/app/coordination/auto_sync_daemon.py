"""Automated P2P Data Sync Daemon (December 2025).

Orchestrates data synchronization across the cluster using a hybrid approach:
- Layer 1: Push-from-generator (immediate push to neighbors on game completion)
- Layer 2: P2P gossip replication (eventual consistency across cluster)

Key features:
- Excludes coordinator nodes from receiving synced data (disk space)
- Skips sync between Lambda NFS nodes (shared storage)
- Prioritizes ephemeral nodes (Vast.ai) for urgent sync
- Integrates with existing BandwidthManager for rate limiting
- Uses ClusterManifest for disk capacity and exclusion rules
- Automatic disk cleanup when usage exceeds threshold

Usage:
    from app.coordination.auto_sync_daemon import AutoSyncDaemon

    daemon = AutoSyncDaemon()
    await daemon.start()
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from app.distributed.cluster_manifest import ClusterManifest

logger = logging.getLogger(__name__)


@dataclass
class AutoSyncConfig:
    """Configuration for automated data sync."""
    enabled: bool = True
    interval_seconds: int = 300
    gossip_interval_seconds: int = 60
    exclude_hosts: list[str] = field(default_factory=list)
    skip_nfs_sync: bool = True
    max_concurrent_syncs: int = 4
    min_games_to_sync: int = 10
    bandwidth_limit_mbps: int = 20
    # Disk usage thresholds (from sync_routing)
    max_disk_usage_percent: float = 70.0
    target_disk_usage_percent: float = 60.0
    # Enable automatic disk cleanup
    auto_cleanup_enabled: bool = True
    # Use ClusterManifest for tracking
    use_cluster_manifest: bool = True

    @classmethod
    def from_config_file(cls, config_path: Path | None = None) -> AutoSyncConfig:
        """Load configuration from distributed_hosts.yaml or unified_loop.yaml."""
        base_dir = Path(__file__).resolve().parent.parent.parent

        # Try distributed_hosts.yaml first (canonical source)
        if config_path is None:
            config_path = base_dir / "config" / "distributed_hosts.yaml"

        config = cls()

        # Load from distributed_hosts.yaml
        if config_path.exists():
            try:
                with open(config_path) as f:
                    data = yaml.safe_load(f)

                # Get sync_routing settings
                sync_routing = data.get("sync_routing", {})
                config.max_disk_usage_percent = sync_routing.get(
                    "max_disk_usage_percent", 70.0
                )
                config.target_disk_usage_percent = sync_routing.get(
                    "target_disk_usage_percent", 60.0
                )

                # Get auto_sync settings
                auto_sync = data.get("auto_sync", {})
                config.enabled = auto_sync.get("enabled", True)
                config.interval_seconds = auto_sync.get("interval_seconds", 300)
                config.gossip_interval_seconds = auto_sync.get("gossip_interval_seconds", 60)
                config.exclude_hosts = auto_sync.get("exclude_hosts", [])
                config.skip_nfs_sync = auto_sync.get("skip_nfs_sync", True)
                config.max_concurrent_syncs = auto_sync.get("max_concurrent_syncs", 4)
                config.min_games_to_sync = auto_sync.get("min_games_to_sync", 10)
                config.bandwidth_limit_mbps = auto_sync.get("bandwidth_limit_mbps", 20)

            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        # Fallback to unified_loop.yaml
        unified_config_path = base_dir / "config" / "unified_loop.yaml"
        if unified_config_path.exists():
            try:
                with open(unified_config_path) as f:
                    data = yaml.safe_load(f)

                auto_sync = data.get("auto_sync", {})
                # Only override if not already set
                if not config.exclude_hosts:
                    config.exclude_hosts = auto_sync.get("exclude_hosts", [])

                # Also check data_aggregation.excluded_nodes for compatibility
                data_agg = data.get("data_aggregation", {})
                for node in data_agg.get("excluded_nodes", []):
                    if node not in config.exclude_hosts:
                        config.exclude_hosts.append(node)

            except Exception as e:
                logger.warning(f"Failed to load unified_loop.yaml: {e}")

        return config


@dataclass
class SyncStats:
    """Statistics for sync operations."""
    total_syncs: int = 0
    successful_syncs: int = 0
    failed_syncs: int = 0
    games_synced: int = 0
    bytes_transferred: int = 0
    last_sync_time: float = 0.0
    last_error: str | None = None


class AutoSyncDaemon:
    """Daemon that orchestrates automated P2P data synchronization.

    Uses hybrid approach:
    - Gossip-based replication for eventual consistency
    - Provider-aware sync (skip NFS, prioritize ephemeral)
    - Coordinator exclusion (save disk space)
    - ClusterManifest for central tracking and disk management
    """

    def __init__(self, config: AutoSyncConfig | None = None):
        self.config = config or AutoSyncConfig.from_config_file()
        self.node_id = socket.gethostname()
        self._running = False
        self._stats = SyncStats()
        self._sync_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._gossip_daemon = None
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_syncs)

        # ClusterManifest integration
        self._cluster_manifest: ClusterManifest | None = None
        if self.config.use_cluster_manifest:
            self._init_cluster_manifest()

        # Detect provider type
        self._provider = self._detect_provider()
        self._is_nfs_node = self._check_nfs_mount()

        logger.info(
            f"AutoSyncDaemon initialized: node={self.node_id}, "
            f"provider={self._provider}, nfs={self._is_nfs_node}, "
            f"manifest={self._cluster_manifest is not None}"
        )

    def _init_cluster_manifest(self) -> None:
        """Initialize the ClusterManifest for tracking."""
        try:
            from app.distributed.cluster_manifest import get_cluster_manifest
            self._cluster_manifest = get_cluster_manifest()

            # Update local capacity
            capacity = self._cluster_manifest.update_local_capacity()
            logger.info(
                f"ClusterManifest initialized: disk usage {capacity.usage_percent:.1f}%"
            )

        except ImportError as e:
            logger.warning(f"ClusterManifest not available: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize ClusterManifest: {e}")

    def _detect_provider(self) -> str:
        """Detect the cloud provider for this node."""
        # Check Lambda (NFS mount)
        if Path("/lambda/nfs").exists():
            return "lambda"

        # Check Vast.ai (workspace directory)
        if Path("/workspace").exists():
            return "vast"

        # Check Mac
        if Path("/Volumes").exists():
            import platform
            if platform.system() == "Darwin":
                return "mac"

        # Check hostname patterns
        hostname = self.node_id.lower()
        if hostname.startswith("lambda-"):
            return "lambda"
        if hostname.startswith("vast-") or hostname.startswith("c."):
            return "vast"
        if "hetzner" in hostname:
            return "hetzner"

        return "unknown"

    def _check_nfs_mount(self) -> bool:
        """Check if NFS storage is mounted."""
        nfs_path = Path("/lambda/nfs/RingRift")
        return nfs_path.exists() and nfs_path.is_dir()

    def is_running(self) -> bool:
        """Check if the daemon is running."""
        return self._running

    async def start(self) -> None:
        """Start the auto sync daemon."""
        if not self.config.enabled:
            logger.info("AutoSyncDaemon disabled by config")
            return

        self._running = True
        logger.info(f"Starting AutoSyncDaemon on {self.node_id}")

        # Start gossip sync daemon
        await self._start_gossip_sync()

        # Start main sync loop
        self._sync_task = asyncio.create_task(self._sync_loop())

        logger.info(
            f"AutoSyncDaemon started: "
            f"interval={self.config.interval_seconds}s, "
            f"exclude={self.config.exclude_hosts}"
        )

    async def stop(self) -> None:
        """Stop the auto sync daemon."""
        logger.info("Stopping AutoSyncDaemon...")
        self._running = False

        # Stop sync task
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        # Stop gossip daemon
        if self._gossip_daemon:
            await self._gossip_daemon.stop()

        logger.info("AutoSyncDaemon stopped")

    async def _start_gossip_sync(self) -> None:
        """Initialize and start the gossip sync daemon."""
        try:
            from app.distributed.gossip_sync import (
                GossipSyncDaemon,
                load_peer_config,
            )

            # Find config
            base_dir = Path(__file__).resolve().parent.parent.parent
            config_path = base_dir / "config" / "remote_hosts.yaml"
            data_dir = base_dir / "data" / "games"

            if not config_path.exists():
                logger.warning("No remote_hosts.yaml found, gossip sync disabled")
                return

            peers = load_peer_config(config_path)

            self._gossip_daemon = GossipSyncDaemon(
                node_id=self.node_id,
                data_dir=data_dir,
                peers_config=peers,
                exclude_hosts=self.config.exclude_hosts,
            )

            await self._gossip_daemon.start()
            logger.info("Gossip sync daemon started")

        except ImportError as e:
            logger.warning(f"Gossip sync not available: {e}")
        except Exception as e:
            logger.error(f"Failed to start gossip sync: {e}")

    async def _sync_loop(self) -> None:
        """Main sync loop."""
        while self._running:
            try:
                await self._sync_cycle()
                self._stats.total_syncs += 1
                self._stats.successful_syncs += 1
                self._stats.last_sync_time = time.time()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._stats.failed_syncs += 1
                self._stats.last_error = str(e)
                logger.error(f"Sync cycle error: {e}")

            await asyncio.sleep(self.config.interval_seconds)

    async def _sync_cycle(self) -> None:
        """Execute one sync cycle."""
        # Skip if NFS node and skip_nfs_sync is enabled
        if self._is_nfs_node and self.config.skip_nfs_sync:
            logger.debug("Skipping sync cycle (NFS node)")
            return

        # Skip if this node is excluded
        if self.node_id in self.config.exclude_hosts:
            logger.debug("Skipping sync cycle (excluded host)")
            return

        # Check ClusterManifest exclusion rules
        if self._cluster_manifest:
            from app.distributed.cluster_manifest import DataType
            if not self._cluster_manifest.can_receive_data(self.node_id, DataType.GAME):
                policy = self._cluster_manifest.get_sync_policy(self.node_id)
                logger.debug(
                    f"Skipping sync cycle (manifest exclusion: {policy.exclusion_reason})"
                )
                return

        # Check disk capacity before syncing
        if not await self._check_disk_capacity():
            return

        # Check for pending data to sync
        pending = await self._get_pending_sync_data()
        if pending < self.config.min_games_to_sync:
            logger.debug(f"Skipping sync: only {pending} games pending")
            return

        logger.info(f"Sync cycle: {pending} games pending")

        # Trigger data collection from peers
        await self._collect_from_peers()

        # Register synced data to manifest
        await self._register_synced_data()

    async def _check_disk_capacity(self) -> bool:
        """Check if disk has capacity for more data.

        Returns:
            True if sync should proceed, False if disk is full
        """
        if not self._cluster_manifest:
            return True

        # Update and check capacity
        capacity = self._cluster_manifest.update_local_capacity()

        if capacity.usage_percent >= self.config.max_disk_usage_percent:
            logger.warning(
                f"Disk usage {capacity.usage_percent:.1f}% exceeds threshold "
                f"({self.config.max_disk_usage_percent}%), triggering cleanup"
            )

            # Run cleanup if enabled
            if self.config.auto_cleanup_enabled:
                await self._run_disk_cleanup()

                # Check again after cleanup
                capacity = self._cluster_manifest.update_local_capacity()
                if capacity.usage_percent >= self.config.max_disk_usage_percent:
                    logger.error(
                        f"Disk still at {capacity.usage_percent:.1f}% after cleanup, "
                        "skipping sync"
                    )
                    return False
            else:
                return False

        return True

    async def _run_disk_cleanup(self) -> None:
        """Run disk cleanup to free space."""
        if not self._cluster_manifest:
            return

        try:
            from app.distributed.cluster_manifest import DiskCleanupPolicy

            policy = DiskCleanupPolicy(
                trigger_usage_percent=self.config.max_disk_usage_percent,
                target_usage_percent=self.config.target_disk_usage_percent,
                min_age_days=7,
                min_replicas_before_delete=2,
                preserve_canonical=True,
            )

            result = self._cluster_manifest.run_disk_cleanup(policy)

            if result.triggered and result.bytes_freed > 0:
                logger.info(
                    f"Disk cleanup freed {result.bytes_freed / 1024 / 1024:.1f} MB "
                    f"({result.databases_deleted} DBs, {result.npz_deleted} NPZ files)"
                )

        except Exception as e:
            logger.error(f"Disk cleanup failed: {e}")

    async def _register_synced_data(self) -> None:
        """Register synced games to ClusterManifest."""
        if not self._cluster_manifest:
            return

        # Get data directory
        base_dir = Path(__file__).resolve().parent.parent.parent
        data_dir = base_dir / "data" / "games"

        if not data_dir.exists():
            return

        import sqlite3

        registered = 0
        for db_path in data_dir.glob("*.db"):
            if db_path.name.startswith(".") or "manifest" in db_path.name:
                continue

            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Get board type and num_players
                cursor.execute(
                    "SELECT board_type, num_players FROM games LIMIT 1"
                )
                row = cursor.fetchone()
                board_type = row[0] if row else None
                num_players = row[1] if row else None

                # Get game IDs
                cursor.execute("SELECT game_id FROM games")
                game_ids = [r[0] for r in cursor.fetchall()]
                conn.close()

                if game_ids:
                    # Register games in batch
                    games = [
                        (gid, self.node_id, str(db_path))
                        for gid in game_ids
                    ]
                    count = self._cluster_manifest.register_games_batch(
                        games,
                        board_type=board_type,
                        num_players=num_players,
                    )
                    registered += count

            except Exception as e:
                logger.debug(f"Failed to register games from {db_path}: {e}")

        if registered > 0:
            logger.info(f"Registered {registered} games to ClusterManifest")

    async def _get_pending_sync_data(self) -> int:
        """Get count of games pending sync."""
        # Check local game count vs expected
        base_dir = Path(__file__).resolve().parent.parent.parent
        data_dir = base_dir / "data" / "games"

        if not data_dir.exists():
            return 0

        import sqlite3
        total_games = 0

        for db_path in data_dir.glob("*.db"):
            if "schema" in db_path.name or "wal" in db_path.name:
                continue
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.execute("SELECT COUNT(*) FROM games")
                total_games += cursor.fetchone()[0]
                conn.close()
            except Exception as e:
                logger.debug(f"Failed to count games in {db_path}: {e}")

        return total_games

    async def _collect_from_peers(self) -> None:
        """Collect data from peers via gossip."""
        # Gossip daemon handles this automatically
        if self._gossip_daemon:
            status = self._gossip_daemon.get_status()
            self._stats.games_synced = status.get("total_pulled", 0)
            logger.debug(
                f"Gossip status: {status['known_games']} known, "
                f"{status['total_pushed']} pushed, {status['total_pulled']} pulled"
            )

    def get_status(self) -> dict[str, Any]:
        """Get daemon status."""
        gossip_status = {}
        if self._gossip_daemon:
            gossip_status = self._gossip_daemon.get_status()

        # Get manifest status
        manifest_status = {}
        if self._cluster_manifest:
            try:
                capacity = self._cluster_manifest.get_node_capacity(self.node_id)
                inventory = self._cluster_manifest.get_node_inventory(self.node_id)
                policy = self._cluster_manifest.get_sync_policy(self.node_id)

                manifest_status = {
                    "enabled": True,
                    "disk_usage_percent": capacity.usage_percent if capacity else 0,
                    "can_receive_games": policy.receive_games,
                    "exclusion_reason": policy.exclusion_reason,
                    "registered_games": inventory.game_count,
                    "registered_models": inventory.model_count,
                    "registered_npz": inventory.npz_count,
                }
            except Exception as e:
                manifest_status = {"enabled": True, "error": str(e)}
        else:
            manifest_status = {"enabled": False}

        return {
            "node_id": self.node_id,
            "running": self._running,
            "provider": self._provider,
            "is_nfs_node": self._is_nfs_node,
            "config": {
                "enabled": self.config.enabled,
                "interval_seconds": self.config.interval_seconds,
                "exclude_hosts": self.config.exclude_hosts,
                "max_disk_usage_percent": self.config.max_disk_usage_percent,
                "auto_cleanup_enabled": self.config.auto_cleanup_enabled,
            },
            "stats": {
                "total_syncs": self._stats.total_syncs,
                "successful_syncs": self._stats.successful_syncs,
                "failed_syncs": self._stats.failed_syncs,
                "games_synced": self._stats.games_synced,
                "last_sync_time": self._stats.last_sync_time,
                "last_error": self._stats.last_error,
            },
            "gossip": gossip_status,
            "manifest": manifest_status,
        }


# Module-level instance for singleton access
_auto_sync_daemon: AutoSyncDaemon | None = None


def get_auto_sync_daemon() -> AutoSyncDaemon:
    """Get the singleton AutoSyncDaemon instance."""
    global _auto_sync_daemon
    if _auto_sync_daemon is None:
        _auto_sync_daemon = AutoSyncDaemon()
    return _auto_sync_daemon


def reset_auto_sync_daemon() -> None:
    """Reset the singleton (for testing)."""
    global _auto_sync_daemon
    _auto_sync_daemon = None


__all__ = [
    "AutoSyncConfig",
    "AutoSyncDaemon",
    "SyncStats",
    "get_auto_sync_daemon",
    "reset_auto_sync_daemon",
]

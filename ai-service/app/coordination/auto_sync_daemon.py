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

from app.coordination.protocols import (
    CoordinatorStatus,
    HealthCheckResult,
    register_coordinator,
    unregister_coordinator,
)
from app.core.async_context import fire_and_forget

if TYPE_CHECKING:
    from app.distributed.cluster_manifest import ClusterManifest

logger = logging.getLogger(__name__)


@dataclass
class AutoSyncConfig:
    """Configuration for automated data sync."""
    enabled: bool = True
    interval_seconds: int = 60  # December 2025: Reduced from 300s for faster data discovery
    gossip_interval_seconds: int = 30  # December 2025: Reduced from 60s for faster replication
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

        # CoordinatorProtocol state (December 2025 - Phase 14)
        self._coordinator_status = CoordinatorStatus.INITIALIZING
        self._start_time: float = 0.0
        self._events_processed: int = 0
        self._errors_count: int = 0
        self._last_error: str = ""

        # ClusterManifest integration
        self._cluster_manifest: ClusterManifest | None = None
        if self.config.use_cluster_manifest:
            self._init_cluster_manifest()

        # Detect provider type
        self._provider = self._detect_provider()
        self._is_nfs_node = self._check_nfs_mount()

        # Phase 9: Event subscription for DATA_STALE triggers
        self._subscribed = False
        self._urgent_sync_pending: dict[str, float] = {}  # config_key -> request_time

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

    # =========================================================================
    # CoordinatorProtocol Implementation (December 2025 - Phase 14)
    # =========================================================================

    @property
    def name(self) -> str:
        """Unique name identifying this coordinator."""
        return "AutoSyncDaemon"

    @property
    def status(self) -> CoordinatorStatus:
        """Current status of the coordinator."""
        return self._coordinator_status

    @property
    def uptime_seconds(self) -> float:
        """Time since daemon started, in seconds."""
        if self._start_time <= 0:
            return 0.0
        return time.time() - self._start_time

    def is_running(self) -> bool:
        """Check if the daemon is running."""
        return self._running

    async def start(self) -> None:
        """Start the auto sync daemon."""
        if not self.config.enabled:
            self._coordinator_status = CoordinatorStatus.STOPPED
            logger.info("AutoSyncDaemon disabled by config")
            return

        if self._coordinator_status == CoordinatorStatus.RUNNING:
            return  # Already running

        self._running = True
        self._coordinator_status = CoordinatorStatus.RUNNING
        self._start_time = time.time()
        logger.info(f"Starting AutoSyncDaemon on {self.node_id}")

        # Phase 9: Subscribe to DATA_STALE events to trigger urgent sync
        self._subscribe_to_events()

        # Start gossip sync daemon
        await self._start_gossip_sync()

        # Start main sync loop
        self._sync_task = asyncio.create_task(self._sync_loop())

        # Register with coordinator registry
        register_coordinator(self)

        logger.info(
            f"AutoSyncDaemon started: "
            f"interval={self.config.interval_seconds}s, "
            f"exclude={self.config.exclude_hosts}"
        )

    def _subscribe_to_events(self) -> None:
        """Subscribe to events that trigger sync (Phase 9).

        Subscribes to:
        - DATA_STALE: Training data is stale, trigger urgent sync
        - SYNC_TRIGGERED: External sync request
        """
        if self._subscribed:
            return
        try:
            from app.coordination.event_router import DataEventType, get_event_bus

            bus = get_event_bus()

            # Subscribe to DATA_STALE to trigger urgent sync
            if hasattr(DataEventType, 'DATA_STALE'):
                bus.subscribe(DataEventType.DATA_STALE, self._on_data_stale)
                logger.info("[AutoSyncDaemon] Subscribed to DATA_STALE")

            # Subscribe to SYNC_TRIGGERED for external requests
            if hasattr(DataEventType, 'SYNC_TRIGGERED'):
                bus.subscribe(DataEventType.SYNC_TRIGGERED, self._on_sync_triggered)
                logger.info("[AutoSyncDaemon] Subscribed to SYNC_TRIGGERED")

            self._subscribed = True
        except Exception as e:
            logger.warning(f"[AutoSyncDaemon] Failed to subscribe to events: {e}")

    async def _on_data_stale(self, event) -> None:
        """Handle DATA_STALE event by triggering urgent sync (Phase 9).

        When training data is detected as stale, we trigger an immediate
        sync operation to fetch fresh data from the cluster.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            board_type = payload.get("board_type", "")
            num_players = payload.get("num_players", 0)
            data_age_hours = payload.get("data_age_hours", 0.0)

            config_key = f"{board_type}_{num_players}p" if board_type and num_players else "unknown"

            logger.warning(
                f"[AutoSyncDaemon] DATA_STALE received for {config_key}: "
                f"age={data_age_hours:.1f}h - triggering urgent sync"
            )

            # Track the urgent sync request
            self._urgent_sync_pending[config_key] = time.time()
            self._events_processed += 1

            # Trigger immediate sync (don't wait for next interval)
            fire_and_forget(self._trigger_urgent_sync(config_key))

        except Exception as e:
            self._errors_count += 1
            self._last_error = str(e)
            logger.error(f"[AutoSyncDaemon] Error handling DATA_STALE: {e}")

    async def _on_sync_triggered(self, event) -> None:
        """Handle external SYNC_TRIGGERED event (Phase 9)."""
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            reason = payload.get("reason", "unknown")
            config_key = payload.get("config_key", "")

            logger.info(
                f"[AutoSyncDaemon] SYNC_TRIGGERED received: "
                f"reason={reason}, config={config_key}"
            )

            self._events_processed += 1

            # Trigger immediate sync
            if config_key:
                fire_and_forget(self._trigger_urgent_sync(config_key))
            else:
                fire_and_forget(self._sync_all())

        except Exception as e:
            self._errors_count += 1
            self._last_error = str(e)
            logger.error(f"[AutoSyncDaemon] Error handling SYNC_TRIGGERED: {e}")

    async def _trigger_urgent_sync(self, config_key: str) -> None:
        """Trigger an urgent sync operation for a specific config (Phase 9)."""
        try:
            logger.info(f"[AutoSyncDaemon] Urgent sync starting for {config_key}")

            # Find nodes with fresh data for this config
            if self._cluster_manifest:
                # Use manifest to find data sources
                await self._sync_all()
            else:
                # Fallback to full sync
                await self._sync_all()

            # Clear the pending flag
            self._urgent_sync_pending.pop(config_key, None)

            logger.info(f"[AutoSyncDaemon] Urgent sync completed for {config_key}")

        except Exception as e:
            logger.error(f"[AutoSyncDaemon] Urgent sync failed for {config_key}: {e}")

    async def stop(self) -> None:
        """Stop the auto sync daemon."""
        if self._coordinator_status == CoordinatorStatus.STOPPED:
            return  # Already stopped

        self._coordinator_status = CoordinatorStatus.STOPPING
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

        # Unregister from coordinator registry
        unregister_coordinator(self.name)

        self._coordinator_status = CoordinatorStatus.STOPPED
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

    async def _emit_sync_failed(self, error: str) -> None:
        """Emit DATA_SYNC_FAILED event."""
        try:
            from app.distributed.data_events import emit_data_sync_failed
            await emit_data_sync_failed(
                host=self.node_id,
                error=error,
                retry_count=self._stats.failed_syncs,
                source="AutoSyncDaemon",
            )
        except Exception as e:
            logger.debug(f"Could not emit DATA_SYNC_FAILED: {e}")

    async def _emit_sync_completed(self, games_synced: int, bytes_transferred: int = 0) -> None:
        """Emit DATA_SYNC_COMPLETED event for feedback loop coupling."""
        try:
            from app.coordination.event_router import get_router
            from app.distributed.data_events import DataEventType

            router = get_router()
            if router:
                await router.publish(
                    event_type=DataEventType.DATA_SYNC_COMPLETED,
                    payload={
                        "node_id": self.node_id,
                        "games_synced": games_synced,
                        "bytes_transferred": bytes_transferred,
                        "total_syncs": self._stats.total_syncs,
                        "successful_syncs": self._stats.successful_syncs,
                    },
                    source="AutoSyncDaemon",
                )
        except Exception as e:
            logger.debug(f"Could not emit DATA_SYNC_COMPLETED: {e}")

    async def _sync_loop(self) -> None:
        """Main sync loop."""
        while self._running:
            try:
                games_synced = await self._sync_cycle()
                self._stats.total_syncs += 1
                self._stats.successful_syncs += 1
                self._stats.last_sync_time = time.time()
                # Emit DATA_SYNC_COMPLETED event for feedback loop
                if games_synced and games_synced > 0:
                    fire_and_forget(
                        self._emit_sync_completed(games_synced),
                        error_callback=lambda exc: logger.debug(f"Failed to emit sync completed: {exc}"),
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._stats.failed_syncs += 1
                self._stats.last_error = str(e)
                logger.error(f"Sync cycle error: {e}")
                # Emit DATA_SYNC_FAILED event
                fire_and_forget(
                    self._emit_sync_failed(str(e)),
                    error_callback=lambda exc: logger.debug(f"Failed to emit sync failed: {exc}"),
                )

            await asyncio.sleep(self.config.interval_seconds)

    async def _sync_cycle(self) -> int:
        """Execute one sync cycle.

        Returns:
            Number of games synced (0 if skipped or no data).
        """
        # Skip if NFS node and skip_nfs_sync is enabled
        if self._is_nfs_node and self.config.skip_nfs_sync:
            logger.debug("Skipping sync cycle (NFS node)")
            return 0

        # Skip if this node is excluded
        if self.node_id in self.config.exclude_hosts:
            logger.debug("Skipping sync cycle (excluded host)")
            return 0

        # Check ClusterManifest exclusion rules
        if self._cluster_manifest:
            from app.distributed.cluster_manifest import DataType
            if not self._cluster_manifest.can_receive_data(self.node_id, DataType.GAME):
                policy = self._cluster_manifest.get_sync_policy(self.node_id)
                logger.debug(
                    f"Skipping sync cycle (manifest exclusion: {policy.exclusion_reason})"
                )
                return 0

        # Check disk capacity before syncing
        if not await self._check_disk_capacity():
            return 0

        # Check for pending data to sync
        pending = await self._get_pending_sync_data()
        if pending < self.config.min_games_to_sync:
            logger.debug(f"Skipping sync: only {pending} games pending")
            return 0

        logger.info(f"Sync cycle: {pending} games pending")

        # Trigger data collection from peers
        await self._collect_from_peers()

        # Register synced data to manifest
        await self._register_synced_data()

        return pending

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
                with sqlite3.connect(db_path, timeout=5) as conn:
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

    def get_metrics(self) -> dict[str, Any]:
        """Get daemon metrics in protocol-compliant format.

        Returns:
            Dictionary of metrics including sync-specific stats.
        """
        return {
            "name": self.name,
            "status": self._coordinator_status.value,
            "uptime_seconds": self.uptime_seconds,
            "start_time": self._start_time,
            "events_processed": self._events_processed,
            "errors_count": self._errors_count,
            "last_error": self._last_error,
            # Sync-specific metrics
            "node_id": self.node_id,
            "provider": self._provider,
            "is_nfs_node": self._is_nfs_node,
            "total_syncs": self._stats.total_syncs,
            "successful_syncs": self._stats.successful_syncs,
            "failed_syncs": self._stats.failed_syncs,
            "games_synced": self._stats.games_synced,
            "bytes_transferred": self._stats.bytes_transferred,
            "last_sync_time": self._stats.last_sync_time,
        }

    def health_check(self) -> HealthCheckResult:
        """Check daemon health.

        Returns:
            Health check result with status and sync details.
        """
        # Check for error state
        if self._coordinator_status == CoordinatorStatus.ERROR:
            return HealthCheckResult.unhealthy(
                f"Daemon in error state: {self._last_error}"
            )

        # Check if stopped
        if self._coordinator_status == CoordinatorStatus.STOPPED:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="Daemon is stopped",
            )

        # Check if disabled by config
        if not self.config.enabled:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="Daemon disabled by configuration",
            )

        # Check sync health
        if self._stats.failed_syncs > self._stats.successful_syncs * 0.5:
            return HealthCheckResult.degraded(
                f"High failure rate: {self._stats.failed_syncs} failures, "
                f"{self._stats.successful_syncs} successes",
                failure_rate=self._stats.failed_syncs / max(self._stats.total_syncs, 1),
            )

        # Check for stale sync
        if self._stats.last_sync_time > 0:
            sync_age = time.time() - self._stats.last_sync_time
            if sync_age > self.config.interval_seconds * 3:
                return HealthCheckResult.degraded(
                    f"No sync in {sync_age:.0f}s (interval: {self.config.interval_seconds}s)",
                    seconds_since_last_sync=sync_age,
                )

        # Healthy
        return HealthCheckResult(
            healthy=True,
            status=self._coordinator_status,
            details={
                "uptime_seconds": self.uptime_seconds,
                "total_syncs": self._stats.total_syncs,
                "games_synced": self._stats.games_synced,
                "gossip_active": self._gossip_daemon is not None,
                "manifest_active": self._cluster_manifest is not None,
            },
        )


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

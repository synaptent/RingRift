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
from app.core.async_context import fire_and_forget, safe_create_task

if TYPE_CHECKING:
    from app.distributed.cluster_manifest import ClusterManifest

logger = logging.getLogger(__name__)

# Import centralized thresholds for quality filtering
try:
    from app.config.thresholds import (
        SYNC_MIN_QUALITY,
        SYNC_QUALITY_SAMPLE_SIZE,
    )
except ImportError:
    SYNC_MIN_QUALITY = 0.5
    SYNC_QUALITY_SAMPLE_SIZE = 20

# Import quality extraction utilities
try:
    from app.distributed.quality_extractor import (
        QualityExtractorConfig,
        extract_quality_from_synced_db,
        get_elo_lookup_from_service,
    )
    HAS_QUALITY_EXTRACTION = True
except ImportError:
    HAS_QUALITY_EXTRACTION = False
    QualityExtractorConfig = None
    extract_quality_from_synced_db = None
    get_elo_lookup_from_service = None


@dataclass
class AutoSyncConfig:
    """Configuration for automated data sync.

    Quality thresholds are loaded from app.config.thresholds for centralized configuration.
    """
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
    # Quality-based sync filtering - from centralized config
    quality_filter_enabled: bool = True
    min_quality_for_sync: float = SYNC_MIN_QUALITY
    quality_sample_size: int = SYNC_QUALITY_SAMPLE_SIZE
    # Quality extraction for priority-based training
    enable_quality_extraction: bool = True
    min_quality_score_for_priority: float = 0.7  # Only queue high-quality games

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
                config.interval_seconds = auto_sync.get("interval_seconds", 60)  # Dec 2025: Match dataclass
                config.gossip_interval_seconds = auto_sync.get("gossip_interval_seconds", 30)  # Dec 2025: Match
                config.exclude_hosts = auto_sync.get("exclude_hosts", [])
                config.skip_nfs_sync = auto_sync.get("skip_nfs_sync", True)
                config.max_concurrent_syncs = auto_sync.get("max_concurrent_syncs", 4)
                config.min_games_to_sync = auto_sync.get("min_games_to_sync", 10)
                config.bandwidth_limit_mbps = auto_sync.get("bandwidth_limit_mbps", 20)

            except (OSError, ValueError) as e:
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

            except (OSError, ValueError) as e:
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
    # Quality filtering stats (December 2025)
    databases_skipped_quality: int = 0
    databases_quality_checked: int = 0
    # Quality extraction stats (December 2025)
    games_quality_extracted: int = 0
    games_added_to_priority: int = 0


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

        # Quality extraction for training data prioritization (December 2025)
        self._quality_config: Any = None
        self._elo_lookup: Any = None
        if self.config.enable_quality_extraction and HAS_QUALITY_EXTRACTION:
            try:
                self._quality_config = QualityExtractorConfig()
                self._elo_lookup = get_elo_lookup_from_service()
                logger.info("Quality extraction enabled for training data prioritization")
            except Exception as e:
                logger.warning(f"Failed to initialize quality extraction: {e}")
                self.config.enable_quality_extraction = False

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
        except (RuntimeError, OSError) as e:
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

    async def sync_now(self) -> int:
        """Trigger an immediate sync cycle.

        Dec 2025: Added to expose sync functionality to sync_facade.py.

        Returns:
            Number of games synced (0 if skipped or no data).
        """
        if not self._running:
            logger.warning("[AutoSyncDaemon] sync_now() called but daemon not running")
            return 0

        return await self._sync_cycle()

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
        self._sync_task = safe_create_task(
            self._sync_loop(),
            name="auto_sync_loop",
        )

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

            # Subscribe to NEW_GAMES_AVAILABLE for push-on-generate (Dec 2025)
            # Layer 1: Immediate push to neighbors when games are generated
            if hasattr(DataEventType, 'NEW_GAMES_AVAILABLE'):
                bus.subscribe(DataEventType.NEW_GAMES_AVAILABLE, self._on_new_games_available)
                logger.info("[AutoSyncDaemon] Subscribed to NEW_GAMES_AVAILABLE (push-on-generate)")

            # Subscribe to DATA_SYNC_STARTED for sync coordination (Dec 2025)
            if hasattr(DataEventType, 'DATA_SYNC_STARTED'):
                bus.subscribe(DataEventType.DATA_SYNC_STARTED, self._on_data_sync_started)
                logger.info("[AutoSyncDaemon] Subscribed to DATA_SYNC_STARTED")

            # Subscribe to MODEL_DISTRIBUTION_COMPLETE for model sync tracking (Dec 2025)
            if hasattr(DataEventType, 'MODEL_DISTRIBUTION_COMPLETE'):
                bus.subscribe(DataEventType.MODEL_DISTRIBUTION_COMPLETE, self._on_model_distribution_complete)
                logger.info("[AutoSyncDaemon] Subscribed to MODEL_DISTRIBUTION_COMPLETE")

            self._subscribed = True
        except (ImportError, RuntimeError, AttributeError) as e:
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

        except (RuntimeError, OSError, ConnectionError) as e:
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

        except (RuntimeError, OSError, ConnectionError) as e:
            self._errors_count += 1
            self._last_error = str(e)
            logger.error(f"[AutoSyncDaemon] Error handling SYNC_TRIGGERED: {e}")

    async def _on_new_games_available(self, event) -> None:
        """Handle NEW_GAMES_AVAILABLE event - push-on-generate (Dec 2025).

        Layer 1 of the sync architecture: When new games are generated,
        immediately push to up to 3 neighbor nodes for rapid replication.
        This is especially important for Vast.ai ephemeral nodes.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            config_key = payload.get("config", "")
            new_games = payload.get("new_games", 0)
            total_games = payload.get("total_games", 0)

            # Only push if we have a meaningful batch (avoid spamming for 1-2 games)
            min_games = self.config.min_games_to_sync or 5
            if new_games < min_games:
                logger.debug(
                    f"[AutoSyncDaemon] Push-on-generate: skipping for {config_key} "
                    f"({new_games} < {min_games} min games)"
                )
                return

            logger.info(
                f"[AutoSyncDaemon] Push-on-generate: {config_key} "
                f"({new_games} new games, {total_games} total) - pushing to neighbors"
            )

            self._events_processed += 1

            # Trigger push to neighbors (Layer 1)
            fire_and_forget(self._push_to_neighbors(config_key, new_games))

        except (RuntimeError, OSError, ConnectionError) as e:
            self._errors_count += 1
            self._last_error = str(e)
            logger.error(f"[AutoSyncDaemon] Error handling NEW_GAMES_AVAILABLE: {e}")

    async def _on_data_sync_started(self, event) -> None:
        """Handle DATA_SYNC_STARTED - sync operation initiated.

        Tracks active sync operations to avoid concurrent syncs to the
        same target, which can cause conflicts and waste bandwidth.

        Added: December 2025
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            host = payload.get("host", "")
            sync_type = payload.get("sync_type", "incremental")

            logger.info(
                f"[AutoSyncDaemon] Sync started to {host} (type: {sync_type})"
            )

            # Track active sync to avoid concurrent operations
            if not hasattr(self, "_active_syncs"):
                self._active_syncs = {}
            self._active_syncs[host] = {
                "start_time": time.time(),
                "sync_type": sync_type,
            }

            self._events_processed += 1

        except (RuntimeError, AttributeError) as e:
            logger.debug(f"[AutoSyncDaemon] Error handling DATA_SYNC_STARTED: {e}")

    async def _on_model_distribution_complete(self, event) -> None:
        """Handle MODEL_DISTRIBUTION_COMPLETE - model synced to cluster.

        Logs model distribution completion and clears any pending model
        sync requests. This prevents redundant model distribution attempts.

        Added: December 2025
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            model_id = payload.get("model_id", "")
            board_type = payload.get("board_type", "")
            num_players = payload.get("num_players", 0)
            distributed_to = payload.get("distributed_to", [])

            config_key = f"{board_type}_{num_players}p" if board_type and num_players else ""

            logger.info(
                f"[AutoSyncDaemon] Model distribution complete: {model_id} "
                f"({config_key}) -> {len(distributed_to)} nodes"
            )

            # Clear any pending model sync requests
            if hasattr(self, "_pending_model_syncs"):
                self._pending_model_syncs.pop(config_key, None)

            self._events_processed += 1

        except (RuntimeError, AttributeError) as e:
            logger.debug(f"[AutoSyncDaemon] Error handling MODEL_DISTRIBUTION_COMPLETE: {e}")

    async def _push_to_neighbors(self, config_key: str, new_games: int) -> None:
        """Push data to up to 3 neighbor nodes (Layer 1: push-from-generator).

        Prefers storage nodes with large disk capacity.
        Skips coordinator nodes and nodes with low disk space.
        """
        try:
            # Get available neighbors
            neighbors = await self._get_push_neighbors(max_neighbors=3)
            if not neighbors:
                logger.debug(
                    f"[AutoSyncDaemon] Push-on-generate: no eligible neighbors for {config_key}"
                )
                return

            # Push to each neighbor
            pushed_count = 0
            for neighbor_id in neighbors[:3]:
                try:
                    success = await self._sync_to_peer(neighbor_id)
                    if success:
                        pushed_count += 1
                        logger.debug(
                            f"[AutoSyncDaemon] Pushed {config_key} to {neighbor_id}"
                        )
                except (RuntimeError, OSError, ConnectionError) as e:
                    logger.warning(
                        f"[AutoSyncDaemon] Failed to push to {neighbor_id}: {e}"
                    )

            if pushed_count > 0:
                logger.info(
                    f"[AutoSyncDaemon] Push-on-generate complete: {config_key} "
                    f"pushed to {pushed_count}/{len(neighbors)} neighbors"
                )

        except (RuntimeError, OSError, ConnectionError) as e:
            logger.error(f"[AutoSyncDaemon] Push-on-generate failed for {config_key}: {e}")

    async def _get_push_neighbors(self, max_neighbors: int = 3) -> list[str]:
        """Get list of neighbor nodes for push-on-generate.

        Returns nodes sorted by priority:
        1. Storage nodes (large disk)
        2. Non-ephemeral nodes
        3. Healthy nodes with low disk usage
        """
        try:
            neighbors = []

            # Get cluster manifest if available
            if self._cluster_manifest:
                # Get all nodes with their storage capacity
                all_nodes = self._cluster_manifest.get_all_nodes()

                for node_id, node_info in all_nodes.items():
                    # Skip self
                    if node_id == self.node_id:
                        continue

                    # Skip excluded nodes (coordinators, etc.)
                    if node_id in self.config.exclude_hosts:
                        continue

                    # Skip nodes with high disk usage
                    disk_usage = node_info.get("disk_usage_percent", 0)
                    if disk_usage > self.config.max_disk_usage_percent:
                        continue

                    # Compute priority score
                    priority = 0.0
                    # Prefer storage nodes
                    if node_info.get("is_storage_node", False):
                        priority += 10.0
                    # Prefer non-ephemeral
                    if not node_info.get("is_ephemeral", False):
                        priority += 5.0
                    # Prefer nodes with more free space
                    priority += (100 - disk_usage) / 20.0

                    neighbors.append((node_id, priority))

                # Sort by priority (descending) and return top N
                neighbors.sort(key=lambda x: x[1], reverse=True)
                return [n[0] for n in neighbors[:max_neighbors]]

            # Fallback: no manifest available, return empty list
            # (we cannot determine neighbors without cluster manifest)
            logger.debug("[AutoSyncDaemon] No cluster manifest available for push neighbors")
            return neighbors

        except (RuntimeError, AttributeError, KeyError) as e:
            logger.warning(f"[AutoSyncDaemon] Error getting push neighbors: {e}")
            return []

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

        except (RuntimeError, OSError, ConnectionError) as e:
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
        except (RuntimeError, OSError, ConnectionError) as e:
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
        except (RuntimeError, OSError, ConnectionError) as e:
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
        except (RuntimeError, OSError, ConnectionError) as e:
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
            except (RuntimeError, OSError, ConnectionError) as e:
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

        except (RuntimeError, OSError, ImportError) as e:
            logger.error(f"Disk cleanup failed: {e}")

    async def _extract_quality_from_synced_db(self, db_path: Path) -> float:
        """Extract quality scores from a synced database.

        Computes average quality score across all games in the database
        for training data prioritization.

        Args:
            db_path: Path to the synced database file

        Returns:
            Average quality score (0.0-1.0), or 0.0 if extraction fails
        """
        if not self.config.enable_quality_extraction or not HAS_QUALITY_EXTRACTION:
            return 0.0

        try:
            # Extract quality for all games in the database
            qualities = extract_quality_from_synced_db(
                local_dir=db_path.parent,
                elo_lookup=self._elo_lookup,
                config=self._quality_config or QualityExtractorConfig(),
            )

            if not qualities or db_path.name not in qualities:
                logger.debug(f"No quality scores extracted from {db_path.name}")
                return 0.0

            game_qualities = qualities[db_path.name]
            if not game_qualities:
                return 0.0

            # Compute average quality
            avg_quality = sum(q.quality_score for q in game_qualities) / len(game_qualities)

            # Update stats
            self._stats.games_quality_extracted += len(game_qualities)

            # Add high-quality games to priority queue
            high_quality_count = 0
            for quality in game_qualities:
                if quality.quality_score >= self.config.min_quality_score_for_priority:
                    await self._update_priority_queue(
                        config_key=f"{db_path.stem}",
                        quality_score=quality.quality_score,
                        game_count=1,
                    )
                    high_quality_count += 1

            self._stats.games_added_to_priority += high_quality_count

            logger.info(
                f"Extracted quality from {db_path.name}: "
                f"{len(game_qualities)} games, avg={avg_quality:.3f}, "
                f"{high_quality_count} added to priority queue"
            )

            return avg_quality

        except Exception as e:
            logger.warning(f"Quality extraction failed for {db_path.name}: {e}")
            return 0.0

    async def _update_priority_queue(
        self,
        config_key: str,
        quality_score: float,
        game_count: int,
    ) -> None:
        """Update the priority queue with high-quality game data.

        Emits QUALITY_SCORE_UPDATED event for curriculum learning integration.

        Args:
            config_key: Configuration identifier (e.g., "hex8_2p")
            quality_score: Quality score (0.0-1.0)
            game_count: Number of games at this quality level
        """
        try:
            # Emit QUALITY_SCORE_UPDATED event for curriculum learning
            from app.distributed.data_events import emit_quality_score_updated

            # Determine quality category
            if quality_score >= 0.8:
                category = "excellent"
                weight = 2.0
            elif quality_score >= 0.7:
                category = "good"
                weight = 1.5
            elif quality_score >= 0.6:
                category = "adequate"
                weight = 1.0
            else:
                category = "poor"
                weight = 0.5

            await emit_quality_score_updated(
                game_id=config_key,
                quality_score=quality_score,
                quality_category=category,
                training_weight=weight,
                game_length=0,  # Not tracked at this level
                is_decisive=True,  # Assume high-quality games are decisive
                source="AutoSyncDaemon",
            )

            logger.debug(
                f"Priority queue updated: {config_key} quality={quality_score:.3f} "
                f"({category}), {game_count} games"
            )

        except Exception as e:
            logger.debug(f"Failed to update priority queue for {config_key}: {e}")

    def _should_sync_database(self, db_path: Path) -> tuple[bool, str]:
        """Check if database meets minimum quality for sync.

        Samples recent games and computes average quality score.
        Databases with avg quality below threshold are skipped to save bandwidth.

        Args:
            db_path: Path to the database file

        Returns:
            Tuple of (should_sync, reason_message)
        """
        if not self.config.quality_filter_enabled:
            return True, "Quality filter disabled"

        import sqlite3

        try:
            from app.quality.unified_quality import compute_game_quality_from_params

            conn = sqlite3.connect(str(db_path), timeout=5)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT game_id, game_status, winner, termination_reason,
                       total_moves, board_type
                FROM games
                ORDER BY created_at DESC
                LIMIT ?
            """, (self.config.quality_sample_size,))

            games = cursor.fetchall()
            conn.close()

            if len(games) < 5:
                # Too few games - sync anyway, small DBs aren't worth filtering
                return True, f"Small DB ({len(games)} games), sync"

            # Compute quality scores for sampled games
            qualities = []
            for g in games:
                try:
                    q = compute_game_quality_from_params(
                        game_id=g["game_id"],
                        game_status=g["game_status"],
                        winner=g["winner"],
                        termination_reason=g["termination_reason"],
                        total_moves=g["total_moves"],
                        board_type=g["board_type"] or "square8",
                    )
                    qualities.append(q.quality_score)
                except (KeyError, TypeError, ValueError) as e:
                    logger.debug(f"Quality check error for game: {e}")
                    qualities.append(0.3)  # Assume poor quality on error

            if not qualities:
                # All quality computations failed - skip sync (likely bad data)
                return False, "No quality scores computed, skip sync"

            avg_quality = sum(qualities) / len(qualities)
            self._stats.databases_quality_checked += 1

            if avg_quality < self.config.min_quality_for_sync:
                self._stats.databases_skipped_quality += 1
                return False, f"Low quality: {avg_quality:.2f} < {self.config.min_quality_for_sync}"

            return True, f"Quality OK: {avg_quality:.2f}"

        except sqlite3.OperationalError as e:
            # Table doesn't exist or schema mismatch - skip sync (don't sync broken DBs)
            if "no such column" in str(e) or "no such table" in str(e):
                logger.debug(f"Quality check skipped (schema issue) for {db_path.name}: {e}")
                return False, f"Schema issue, skip sync: {e}"
            logger.warning(f"Quality check DB error for {db_path.name}: {e}")
            return False, f"DB error, skip sync: {e}"
        except ImportError as e:
            # Quality module unavailable - conservative: skip sync
            logger.warning(f"Quality module not available, skipping sync: {e}")
            return False, "Quality module unavailable, skip sync"
        except (RuntimeError, OSError, ConnectionError) as e:
            # Transient error - skip this sync attempt, will retry later
            logger.warning(f"Quality check failed for {db_path.name}: {e}")
            return False, f"Check failed, skip sync: {e}"

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
        skipped_quality = 0
        for db_path in data_dir.glob("*.db"):
            if db_path.name.startswith(".") or "manifest" in db_path.name:
                continue

            # Quality check before registering
            should_register, reason = self._should_sync_database(db_path)
            if not should_register:
                logger.info(f"Skipping registration for {db_path.name}: {reason}")
                skipped_quality += 1
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

                    # Extract quality scores after successful registration
                    if self.config.enable_quality_extraction:
                        fire_and_forget(self._extract_quality_from_synced_db(db_path))

            except (OSError, RuntimeError) as e:
                logger.debug(f"Failed to register games from {db_path}: {e}")

        if registered > 0 or skipped_quality > 0:
            logger.info(
                f"Registered {registered} games to ClusterManifest "
                f"(skipped {skipped_quality} low-quality databases)"
            )

    async def _get_pending_sync_data(self) -> int:
        """Get count of games pending sync (from quality-passing databases)."""
        # Check local game count vs expected
        base_dir = Path(__file__).resolve().parent.parent.parent
        data_dir = base_dir / "data" / "games"

        if not data_dir.exists():
            return 0

        import sqlite3
        total_games = 0
        skipped_dbs = 0

        for db_path in data_dir.glob("*.db"):
            if "schema" in db_path.name or "wal" in db_path.name:
                continue

            # Quality filter - skip low quality databases
            should_sync, reason = self._should_sync_database(db_path)
            if not should_sync:
                logger.debug(f"Excluding {db_path.name} from pending count: {reason}")
                skipped_dbs += 1
                continue

            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.execute("SELECT COUNT(*) FROM games")
                total_games += cursor.fetchone()[0]
                conn.close()
            except (OSError, RuntimeError) as e:
                logger.debug(f"Failed to count games in {db_path}: {e}")

        if skipped_dbs > 0:
            logger.debug(f"Excluded {skipped_dbs} low-quality databases from sync count")

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
            except (RuntimeError, OSError, AttributeError) as e:
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
                "databases_quality_checked": self._stats.databases_quality_checked,
                "databases_skipped_quality": self._stats.databases_skipped_quality,
                "games_quality_extracted": self._stats.games_quality_extracted,
                "games_added_to_priority": self._stats.games_added_to_priority,
            },
            "quality_filter": {
                "enabled": self.config.quality_filter_enabled,
                "min_quality": self.config.min_quality_for_sync,
                "sample_size": self.config.quality_sample_size,
            },
            "quality_extraction": {
                "enabled": self.config.enable_quality_extraction,
                "min_quality_for_priority": self.config.min_quality_score_for_priority,
                "games_extracted": self._stats.games_quality_extracted,
                "games_prioritized": self._stats.games_added_to_priority,
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

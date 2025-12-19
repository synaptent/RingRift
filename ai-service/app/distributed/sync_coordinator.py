"""Sync Coordinator - Unified data synchronization orchestrator.

This module provides a single entry point for all data synchronization operations
across the RingRift distributed training cluster. It integrates:

1. aria2 transport for high-performance multi-source downloads
2. SSH/rsync transport for reliable file transfer
3. P2P HTTP transport as fallback
4. Gossip sync for eventually-consistent P2P replication
5. NFS optimization (skip sync when storage is shared)

The coordinator automatically selects the best transport based on:
- Storage provider capabilities (NFS, ephemeral, local)
- Transport availability (aria2c installed, SSH keys configured)
- Historical success rates (circuit breaker integration)

Usage:
    coordinator = SyncCoordinator.get_instance()

    # Sync all training data from cluster
    await coordinator.sync_training_data()

    # Sync specific models
    await coordinator.sync_models(model_ids=["ringrift_best_sq8_2p"])

    # Full cluster sync
    stats = await coordinator.full_cluster_sync()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .storage_provider import (
    StorageProvider,
    TransportConfig,
    get_storage_provider,
    get_optimal_transport_config,
    get_aria2_sources,
)

logger = logging.getLogger(__name__)

# Import transports with graceful fallbacks
try:
    from .aria2_transport import Aria2Transport, check_aria2_available, Aria2Config
    HAS_ARIA2 = True
except ImportError:
    HAS_ARIA2 = False
    check_aria2_available = lambda: False

try:
    from .p2p_sync_client import P2PSyncClient, P2PFallbackSync
    HAS_P2P = True
except ImportError:
    HAS_P2P = False

try:
    from .gossip_sync import GossipSyncDaemon, GossipPeer
    HAS_GOSSIP = True
except ImportError:
    HAS_GOSSIP = False

try:
    from .ssh_transport import SSHTransport
    HAS_SSH = True
except ImportError:
    HAS_SSH = False


class SyncCategory(Enum):
    """Categories of data to sync."""
    GAMES = "games"
    MODELS = "models"
    TRAINING = "training"
    ELO = "elo"
    ALL = "all"


@dataclass
class SyncStats:
    """Statistics from a sync operation."""
    category: str
    files_synced: int = 0
    files_failed: int = 0
    bytes_transferred: int = 0
    duration_seconds: float = 0.0
    transport_used: str = ""
    sources_tried: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        total = self.files_synced + self.files_failed
        return self.files_synced / total if total > 0 else 1.0


@dataclass
class ClusterSyncStats:
    """Statistics from a full cluster sync."""
    total_files_synced: int = 0
    total_bytes_transferred: int = 0
    duration_seconds: float = 0.0
    categories: Dict[str, SyncStats] = field(default_factory=dict)
    transport_distribution: Dict[str, int] = field(default_factory=dict)
    nodes_synced: int = 0
    nodes_failed: int = 0


class SyncCoordinator:
    """Unified coordinator for all data synchronization operations.

    This class provides a single entry point for syncing data across the
    distributed training cluster, automatically selecting the best transport
    and optimizing for the current storage provider.
    """

    _instance: Optional["SyncCoordinator"] = None

    def __init__(
        self,
        provider: Optional[StorageProvider] = None,
        config: Optional[TransportConfig] = None,
    ):
        self._provider = provider or get_storage_provider()
        self._config = config or get_optimal_transport_config(self._provider)

        # Transport instances (lazily initialized)
        self._aria2: Optional[Aria2Transport] = None
        self._p2p: Optional[P2PSyncClient] = None
        self._gossip: Optional[GossipSyncDaemon] = None
        self._ssh: Optional[SSHTransport] = None

        # State tracking
        self._running = False
        self._last_sync_times: Dict[str, float] = {}
        self._sync_stats: Dict[str, SyncStats] = {}

        # Source discovery cache
        self._aria2_sources: List[str] = []
        self._source_discovery_time: float = 0

        logger.info(
            f"SyncCoordinator initialized: provider={self._provider.provider_type.value}, "
            f"aria2={HAS_ARIA2 and check_aria2_available()}, "
            f"shared_storage={self._provider.has_shared_storage}"
        )

    @classmethod
    def get_instance(
        cls,
        provider: Optional[StorageProvider] = None,
        config: Optional[TransportConfig] = None,
    ) -> "SyncCoordinator":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls(provider, config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        if cls._instance is not None:
            asyncio.create_task(cls._instance.shutdown())
        cls._instance = None

    # =========================================================================
    # Transport Initialization
    # =========================================================================

    def _init_aria2(self) -> Optional[Aria2Transport]:
        """Initialize aria2 transport if available."""
        if not HAS_ARIA2 or not self._config.enable_aria2:
            return None
        if not check_aria2_available():
            logger.warning("aria2c not installed, aria2 transport disabled")
            return None
        if self._aria2 is None:
            config = Aria2Config(
                connections_per_server=self._config.aria2_connections_per_server,
                split=self._config.aria2_split,
                data_server_port=self._config.aria2_data_server_port,
            )
            self._aria2 = Aria2Transport(config)
            logger.debug("aria2 transport initialized")
        return self._aria2

    def _init_p2p(self) -> Optional[P2PSyncClient]:
        """Initialize P2P transport if available."""
        if not HAS_P2P or not self._config.enable_p2p:
            return None
        if self._p2p is None:
            self._p2p = P2PSyncClient()
            logger.debug("P2P transport initialized")
        return self._p2p

    async def _init_gossip(self, peers: Optional[List[Dict[str, Any]]] = None) -> Optional[GossipSyncDaemon]:
        """Initialize gossip sync daemon if available."""
        if not HAS_GOSSIP or not self._config.enable_gossip:
            return None
        if self._provider.has_shared_storage:
            logger.debug("Gossip sync disabled for shared storage provider")
            return None
        if self._gossip is None:
            gossip_peers = []
            if peers:
                for p in peers:
                    gossip_peers.append(GossipPeer(
                        host=p.get("host", ""),
                        port=p.get("port", self._config.gossip_port),
                    ))
            self._gossip = GossipSyncDaemon(
                port=self._config.gossip_port,
                peers=gossip_peers,
                data_dir=self._provider.selfplay_dir,
            )
            logger.debug("Gossip sync daemon initialized")
        return self._gossip

    # =========================================================================
    # Source Discovery
    # =========================================================================

    async def discover_sources(self, force_refresh: bool = False) -> List[str]:
        """Discover available aria2 data sources in the cluster.

        Args:
            force_refresh: Force re-discovery even if cache is fresh

        Returns:
            List of aria2 data server URLs
        """
        cache_ttl = 300  # 5 minutes
        now = time.time()

        if not force_refresh and self._aria2_sources and (now - self._source_discovery_time) < cache_ttl:
            return self._aria2_sources

        # Get sources from cluster config
        sources = get_aria2_sources(exclude_self=True)

        # Validate sources are reachable
        if HAS_ARIA2 and self._config.enable_aria2:
            aria2 = self._init_aria2()
            if aria2:
                valid_sources = []
                for source in sources:
                    try:
                        inventory = await aria2.fetch_inventory(source, timeout=5)
                        if inventory and inventory.reachable:
                            valid_sources.append(source)
                    except Exception as e:
                        logger.debug(f"Source {source} not reachable: {e}")
                sources = valid_sources

        self._aria2_sources = sources
        self._source_discovery_time = now
        logger.info(f"Discovered {len(sources)} aria2 data sources")
        return sources

    # =========================================================================
    # Sync Operations
    # =========================================================================

    async def sync_training_data(
        self,
        sources: Optional[List[str]] = None,
        max_age_hours: float = 168,  # 1 week
    ) -> SyncStats:
        """Sync training data from cluster sources.

        Args:
            sources: Specific sources to sync from (auto-discovers if None)
            max_age_hours: Only sync files newer than this

        Returns:
            SyncStats with operation results
        """
        start_time = time.time()
        stats = SyncStats(category="training")

        # Skip if we have shared storage
        if self._provider.has_shared_storage:
            logger.info("Skipping training data sync - using shared NFS storage")
            stats.transport_used = "nfs_shared"
            return stats

        # Discover sources
        if sources is None:
            sources = await self.discover_sources()
        stats.sources_tried = len(sources)

        if not sources:
            logger.warning("No sources available for training data sync")
            return stats

        # Try aria2 first (best performance)
        aria2 = self._init_aria2()
        if aria2:
            try:
                result = await aria2.sync_training_data(
                    sources,
                    self._provider.training_dir,
                    max_age_hours=max_age_hours,
                )
                stats.files_synced = result.files_synced
                stats.bytes_transferred = result.bytes_transferred
                stats.transport_used = "aria2"
                stats.duration_seconds = time.time() - start_time
                if result.success:
                    logger.info(
                        f"Training data sync complete: {stats.files_synced} files, "
                        f"{stats.bytes_transferred / (1024*1024):.1f}MB via aria2"
                    )
                    return stats
                stats.errors.extend(result.errors)
            except Exception as e:
                stats.errors.append(f"aria2 sync failed: {e}")
                logger.warning(f"aria2 training sync failed, trying fallback: {e}")

        # Fallback to P2P HTTP
        p2p = self._init_p2p()
        if p2p:
            try:
                for source in sources[:3]:  # Limit fallback attempts
                    result = await p2p.sync_directory(
                        source,
                        self._provider.training_dir,
                        patterns=["*.npz", "*.h5"],
                    )
                    stats.files_synced += result.get("files_synced", 0)
                    stats.bytes_transferred += result.get("bytes_transferred", 0)
                stats.transport_used = "p2p_http"
            except Exception as e:
                stats.errors.append(f"P2P sync failed: {e}")
                logger.warning(f"P2P training sync failed: {e}")

        stats.duration_seconds = time.time() - start_time
        return stats

    async def sync_models(
        self,
        model_ids: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
    ) -> SyncStats:
        """Sync model checkpoints from cluster sources.

        Args:
            model_ids: Specific model IDs to sync (syncs all best models if None)
            sources: Specific sources to sync from (auto-discovers if None)

        Returns:
            SyncStats with operation results
        """
        start_time = time.time()
        stats = SyncStats(category="models")

        # Skip if we have shared storage
        if self._provider.has_shared_storage:
            logger.info("Skipping model sync - using shared NFS storage")
            stats.transport_used = "nfs_shared"
            return stats

        # Discover sources
        if sources is None:
            sources = await self.discover_sources()
        stats.sources_tried = len(sources)

        if not sources:
            logger.warning("No sources available for model sync")
            return stats

        # Use aria2 for model sync
        aria2 = self._init_aria2()
        if aria2:
            try:
                patterns = None
                if model_ids:
                    patterns = [f"*{mid}*" for mid in model_ids]

                result = await aria2.sync_models(
                    sources,
                    self._provider.models_dir,
                    patterns=patterns,
                )
                stats.files_synced = result.files_synced
                stats.bytes_transferred = result.bytes_transferred
                stats.transport_used = "aria2"
                stats.duration_seconds = time.time() - start_time
                if result.success:
                    logger.info(
                        f"Model sync complete: {stats.files_synced} models, "
                        f"{stats.bytes_transferred / (1024*1024):.1f}MB"
                    )
                    return stats
                stats.errors.extend(result.errors)
            except Exception as e:
                stats.errors.append(f"aria2 model sync failed: {e}")
                logger.warning(f"aria2 model sync failed: {e}")

        stats.duration_seconds = time.time() - start_time
        return stats

    async def sync_games(
        self,
        sources: Optional[List[str]] = None,
        board_types: Optional[List[str]] = None,
    ) -> SyncStats:
        """Sync selfplay game databases from cluster sources.

        Args:
            sources: Specific sources to sync from (auto-discovers if None)
            board_types: Only sync specific board types

        Returns:
            SyncStats with operation results
        """
        start_time = time.time()
        stats = SyncStats(category="games")

        # Skip if we have shared storage
        if self._provider.has_shared_storage:
            logger.info("Skipping game sync - using shared NFS storage")
            stats.transport_used = "nfs_shared"
            return stats

        # Discover sources
        if sources is None:
            sources = await self.discover_sources()
        stats.sources_tried = len(sources)

        if not sources:
            logger.warning("No sources available for game sync")
            return stats

        # Use aria2 for game sync
        aria2 = self._init_aria2()
        if aria2:
            try:
                result = await aria2.sync_games(
                    sources,
                    self._provider.selfplay_dir,
                )
                stats.files_synced = result.files_synced
                stats.bytes_transferred = result.bytes_transferred
                stats.transport_used = "aria2"
                stats.duration_seconds = time.time() - start_time
                if result.success:
                    logger.info(
                        f"Game sync complete: {stats.files_synced} databases, "
                        f"{stats.bytes_transferred / (1024*1024):.1f}MB"
                    )
                    return stats
                stats.errors.extend(result.errors)
            except Exception as e:
                stats.errors.append(f"aria2 game sync failed: {e}")
                logger.warning(f"aria2 game sync failed: {e}")

        stats.duration_seconds = time.time() - start_time
        return stats

    async def full_cluster_sync(
        self,
        categories: Optional[List[SyncCategory]] = None,
    ) -> ClusterSyncStats:
        """Perform a full sync of all data categories from the cluster.

        Args:
            categories: Categories to sync (all if None)

        Returns:
            ClusterSyncStats with complete sync results
        """
        start_time = time.time()
        stats = ClusterSyncStats()

        if categories is None:
            categories = [SyncCategory.GAMES, SyncCategory.MODELS, SyncCategory.TRAINING]

        # Skip if we have shared storage
        if self._provider.has_shared_storage:
            logger.info("Full cluster sync skipped - using shared NFS storage")
            stats.duration_seconds = time.time() - start_time
            return stats

        # Discover sources once
        sources = await self.discover_sources(force_refresh=True)
        if not sources:
            logger.warning("No sources available for cluster sync")
            stats.duration_seconds = time.time() - start_time
            return stats

        # Sync each category
        for category in categories:
            if category == SyncCategory.GAMES:
                cat_stats = await self.sync_games(sources)
            elif category == SyncCategory.MODELS:
                cat_stats = await self.sync_models(sources=sources)
            elif category == SyncCategory.TRAINING:
                cat_stats = await self.sync_training_data(sources)
            else:
                continue

            stats.categories[category.value] = cat_stats
            stats.total_files_synced += cat_stats.files_synced
            stats.total_bytes_transferred += cat_stats.bytes_transferred

            if cat_stats.transport_used:
                stats.transport_distribution[cat_stats.transport_used] = (
                    stats.transport_distribution.get(cat_stats.transport_used, 0) + 1
                )

        stats.nodes_synced = len(sources)
        stats.duration_seconds = time.time() - start_time

        logger.info(
            f"Full cluster sync complete: {stats.total_files_synced} files, "
            f"{stats.total_bytes_transferred / (1024*1024):.1f}MB in {stats.duration_seconds:.1f}s"
        )

        return stats

    # =========================================================================
    # Background Sync
    # =========================================================================

    async def start_background_sync(
        self,
        interval_seconds: Optional[int] = None,
        categories: Optional[List[SyncCategory]] = None,
    ) -> None:
        """Start background sync daemon.

        Args:
            interval_seconds: Sync interval (uses provider default if None)
            categories: Categories to sync
        """
        if self._running:
            logger.warning("Background sync already running")
            return

        if interval_seconds is None:
            interval_seconds = self._provider.capabilities.max_sync_interval_seconds

        self._running = True
        logger.info(f"Starting background sync with {interval_seconds}s interval")

        # Start gossip daemon if enabled
        if self._config.enable_gossip and not self._provider.has_shared_storage:
            gossip = await self._init_gossip()
            if gossip:
                await gossip.start()

        # Run sync loop
        while self._running:
            try:
                await self.full_cluster_sync(categories)
            except Exception as e:
                logger.error(f"Background sync failed: {e}")

            await asyncio.sleep(interval_seconds)

    def stop_background_sync(self) -> None:
        """Stop background sync daemon."""
        self._running = False
        logger.info("Background sync stopped")

    async def shutdown(self) -> None:
        """Shutdown the coordinator and all transports."""
        self._running = False

        if self._aria2:
            await self._aria2.close()
            self._aria2 = None

        if self._gossip:
            await self._gossip.stop()
            self._gossip = None

        logger.info("SyncCoordinator shutdown complete")

    # =========================================================================
    # Status and Monitoring
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get current sync status and statistics."""
        return {
            "provider": self._provider.provider_type.value,
            "shared_storage": self._provider.has_shared_storage,
            "running": self._running,
            "transports": {
                "aria2": HAS_ARIA2 and check_aria2_available() and self._config.enable_aria2,
                "p2p": HAS_P2P and self._config.enable_p2p,
                "gossip": HAS_GOSSIP and self._config.enable_gossip and not self._provider.has_shared_storage,
                "ssh": HAS_SSH and self._config.enable_ssh,
            },
            "sources_discovered": len(self._aria2_sources),
            "last_sync_times": self._last_sync_times,
            "config": {
                "aria2_connections": self._config.aria2_connections_per_server,
                "gossip_port": self._config.gossip_port,
                "fallback_chain": self._config.fallback_chain,
            },
        }


# =============================================================================
# Module-level convenience functions
# =============================================================================

async def sync_training_data(**kwargs) -> SyncStats:
    """Convenience function to sync training data."""
    return await SyncCoordinator.get_instance().sync_training_data(**kwargs)


async def sync_models(**kwargs) -> SyncStats:
    """Convenience function to sync models."""
    return await SyncCoordinator.get_instance().sync_models(**kwargs)


async def sync_games(**kwargs) -> SyncStats:
    """Convenience function to sync games."""
    return await SyncCoordinator.get_instance().sync_games(**kwargs)


async def full_cluster_sync(**kwargs) -> ClusterSyncStats:
    """Convenience function for full cluster sync."""
    return await SyncCoordinator.get_instance().full_cluster_sync(**kwargs)

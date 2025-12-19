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
import socket
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .storage_provider import (
    StorageProvider,
    TransportConfig,
    get_storage_provider,
    get_optimal_transport_config,
    get_aria2_sources,
)
from .unified_manifest import (
    DataManifest,
    GameQualityMetadata,
    PriorityQueueEntry,
)

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"

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

try:
    from .sync_utils import rsync_directory
    HAS_RSYNC = True
except ImportError:
    HAS_RSYNC = False
    rsync_directory = None

try:
    from app.metrics.orchestrator import (
        record_sync_coordinator_op,
        record_nfs_skip,
        update_data_server_status,
        update_sync_sources_count,
    )
    HAS_SYNC_METRICS = True
except ImportError:
    HAS_SYNC_METRICS = False

    def record_sync_coordinator_op(*args, **kwargs):
        return None

    def record_nfs_skip(*args, **kwargs):
        return None

    def update_data_server_status(*args, **kwargs):
        return None

    def update_sync_sources_count(*args, **kwargs):
        return None


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
    # Quality-aware sync stats
    high_quality_games_synced: int = 0
    avg_quality_score: float = 0.0
    avg_elo: float = 0.0

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
    # Quality-aware stats
    total_high_quality_games: int = 0
    avg_quality_score: float = 0.0
    quality_distribution: Dict[str, Any] = field(default_factory=dict)


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
        manifest_path: Optional[Path] = None,
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

        # Quality-aware sync: manifest integration
        self._manifest: Optional[DataManifest] = None
        self._manifest_path = manifest_path
        self._quality_lookup: Dict[str, float] = {}
        self._elo_lookup: Dict[str, float] = {}
        self._quality_lookup_time: float = 0
        self._init_manifest()

        logger.info(
            f"SyncCoordinator initialized: provider={self._provider.provider_type.value}, "
            f"aria2={HAS_ARIA2 and check_aria2_available()}, "
            f"shared_storage={self._provider.has_shared_storage}, "
            f"manifest={'enabled' if self._manifest else 'disabled'}"
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

    def _resolve_games_dir(self) -> Path:
        """Resolve the preferred local directory for game DBs."""
        base = self._provider.selfplay_dir
        games_dir = base.parent / "games"
        if base.name == "selfplay" and games_dir.exists():
            return games_dir
        return base

    @staticmethod
    def _snapshot_files(base_dir: Path, patterns: List[str]) -> Dict[str, int]:
        snapshot: Dict[str, int] = {}
        if not base_dir.exists():
            return snapshot
        for pattern in patterns:
            for path in base_dir.rglob(pattern):
                if not path.is_file():
                    continue
                try:
                    rel_path = str(path.relative_to(base_dir))
                    snapshot[rel_path] = path.stat().st_size
                except Exception:
                    continue
        return snapshot

    @staticmethod
    def _diff_snapshot(before: Dict[str, int], after: Dict[str, int]) -> Tuple[int, int]:
        new_files = {k: v for k, v in after.items() if k not in before}
        return len(new_files), sum(new_files.values())

    async def _sync_with_rsync(
        self,
        local_dir: Path,
        remote_subdir: str,
        include_patterns: List[str],
    ) -> Tuple[int, int, List[str]]:
        if not HAS_RSYNC or rsync_directory is None:
            return 0, 0, ["rsync not available"]

        try:
            from app.distributed.hosts import load_remote_hosts
            hosts = list(load_remote_hosts().values())
        except Exception as e:
            return 0, 0, [f"failed to load hosts: {e}"]

        hostname = socket.gethostname().lower()
        files_synced = 0
        bytes_transferred = 0
        errors: List[str] = []

        pre_snapshot = self._snapshot_files(local_dir, include_patterns)
        loop = asyncio.get_event_loop()

        for host in hosts[:3]:
            if host.name.lower() == hostname:
                continue
            if self._provider.should_skip_rsync_to(host.name):
                continue

            remote_dir = f"{host.work_directory.rstrip('/')}/{remote_subdir.lstrip('/')}"
            try:
                success = await loop.run_in_executor(
                    None,
                    lambda: rsync_directory(
                        host,
                        remote_dir,
                        local_dir,
                        include_patterns=include_patterns,
                        exclude_patterns=["*"],
                        timeout=self._config.ssh_timeout,
                    ),
                )
                if not success:
                    errors.append(f"rsync failed for {host.name}")
                    continue
            except Exception as e:
                errors.append(f"rsync error for {host.name}: {e}")
                continue

            post_snapshot = self._snapshot_files(local_dir, include_patterns)
            new_files, new_bytes = self._diff_snapshot(pre_snapshot, post_snapshot)
            pre_snapshot = post_snapshot
            files_synced += new_files
            bytes_transferred += new_bytes

        return files_synced, bytes_transferred, errors

    async def _sync_with_p2p(
        self,
        local_dir: Path,
        pattern: str,
    ) -> Tuple[int, int, List[str]]:
        p2p = self._init_p2p()
        if not p2p:
            return 0, 0, ["p2p not available"]

        try:
            from app.distributed.hosts import load_remote_hosts
            hosts = list(load_remote_hosts().values())
        except Exception as e:
            return 0, 0, [f"failed to load hosts: {e}"]

        try:
            from app.config.unified_config import get_config
            p2p_port = get_config().distributed.p2p_port
        except Exception:
            p2p_port = 8770

        hostname = socket.gethostname().lower()
        files_synced = 0
        bytes_transferred = 0
        errors: List[str] = []

        for host in hosts[:3]:
            if host.name.lower() == hostname:
                continue
            peer_host = host.tailscale_ip or host.ssh_host
            if not peer_host:
                continue
            peer_host = peer_host.split("@", 1)[1] if "@" in peer_host else peer_host
            try:
                result = await p2p.sync_from_peer(
                    peer_host=peer_host,
                    peer_port=p2p_port,
                    pattern=pattern,
                    local_dir=local_dir,
                )
                if result.success:
                    files_synced += result.files_synced
                    bytes_transferred += result.bytes_transferred
                else:
                    errors.extend(result.errors)
            except Exception as e:
                errors.append(f"p2p error for {host.name}: {e}")

        return files_synced, bytes_transferred, errors

    # =========================================================================
    # Manifest & Quality Integration
    # =========================================================================

    def _init_manifest(self) -> None:
        """Initialize the data manifest for quality-aware sync."""
        if self._manifest is not None:
            return

        # Try specified path first
        manifest_paths = []
        if self._manifest_path:
            manifest_paths.append(self._manifest_path)

        # Then try standard locations
        manifest_paths.extend([
            DEFAULT_DATA_DIR / "data_manifest.db",
            self._provider.data_dir / "data_manifest.db" if hasattr(self._provider, 'data_dir') else None,
            Path.home() / "ringrift" / "ai-service" / "data" / "data_manifest.db",
        ])

        for path in manifest_paths:
            if path and path.exists():
                try:
                    self._manifest = DataManifest(path)
                    logger.info(f"Loaded manifest from {path}")
                    self._refresh_quality_lookup()
                    return
                except Exception as e:
                    logger.warning(f"Failed to load manifest from {path}: {e}")

        # Create new manifest if none found
        default_path = DEFAULT_DATA_DIR / "data_manifest.db"
        try:
            default_path.parent.mkdir(parents=True, exist_ok=True)
            self._manifest = DataManifest(default_path)
            logger.info(f"Created new manifest at {default_path}")
        except Exception as e:
            logger.warning(f"Failed to create manifest: {e}")
            self._manifest = None

    def _refresh_quality_lookup(self, limit: int = 50000) -> int:
        """Refresh the quality lookup tables from manifest.

        Args:
            limit: Maximum number of games to load into lookup

        Returns:
            Number of games loaded into lookup
        """
        if not self._manifest:
            return 0

        cache_ttl = 300  # 5 minutes
        now = time.time()

        if self._quality_lookup and (now - self._quality_lookup_time) < cache_ttl:
            return len(self._quality_lookup)

        try:
            high_quality_games = self._manifest.get_high_quality_games(
                min_quality_score=0.0,  # Get all with scores
                limit=limit,
            )

            self._quality_lookup = {}
            self._elo_lookup = {}

            for game in high_quality_games:
                self._quality_lookup[game.game_id] = game.quality_score
                self._elo_lookup[game.game_id] = game.avg_player_elo

            self._quality_lookup_time = now
            logger.debug(f"Refreshed quality lookup: {len(self._quality_lookup)} games")
            return len(self._quality_lookup)

        except Exception as e:
            logger.warning(f"Failed to refresh quality lookup: {e}")
            return 0

    def get_quality_lookup(self, force_refresh: bool = False) -> Dict[str, float]:
        """Get quality lookup dictionary for training integration.

        Args:
            force_refresh: Force refresh from manifest

        Returns:
            Dict mapping game_id to quality_score
        """
        if force_refresh or not self._quality_lookup:
            self._quality_lookup_time = 0  # Force refresh
            self._refresh_quality_lookup()
        return self._quality_lookup.copy()

    def get_elo_lookup(self, force_refresh: bool = False) -> Dict[str, float]:
        """Get Elo lookup dictionary for training integration.

        Args:
            force_refresh: Force refresh from manifest

        Returns:
            Dict mapping game_id to avg_player_elo
        """
        if force_refresh or not self._elo_lookup:
            self._quality_lookup_time = 0  # Force refresh
            self._refresh_quality_lookup()
        return self._elo_lookup.copy()

    def get_manifest(self) -> Optional[DataManifest]:
        """Get the data manifest instance.

        Returns:
            DataManifest instance or None if not initialized
        """
        return self._manifest

    def get_high_quality_game_ids(
        self,
        min_quality: float = 0.7,
        min_elo: Optional[float] = None,
        limit: int = 10000,
    ) -> List[str]:
        """Get list of high-quality game IDs for training.

        Args:
            min_quality: Minimum quality score threshold
            min_elo: Optional minimum average Elo threshold
            limit: Maximum number of games to return

        Returns:
            List of game IDs meeting quality criteria
        """
        if not self._manifest:
            return []

        try:
            games = self._manifest.get_high_quality_games(
                min_quality_score=min_quality,
                limit=limit,
            )

            # Filter by Elo if specified
            if min_elo is not None:
                games = [g for g in games if g.avg_player_elo >= min_elo]

            return [g.game_id for g in games]

        except Exception as e:
            logger.warning(f"Failed to get high quality game IDs: {e}")
            return []

    # =========================================================================
    # Data Server (for aria2 clients to download from this node)
    # =========================================================================

    _data_server_process: Optional[asyncio.subprocess.Process] = None
    _data_server_port: int = 8766

    async def start_data_server(self, port: int = 8766) -> bool:
        """Start the aria2 data server for serving files to other nodes.

        This allows other nodes to download files from this node using aria2.

        Args:
            port: Port to serve on (default: 8766)

        Returns:
            True if server started successfully
        """
        if self._data_server_process is not None:
            logger.warning("Data server already running")
            return True

        self._data_server_port = port

        try:
            # Start the data server as a subprocess
            script_path = Path(__file__).parent.parent.parent / "scripts" / "aria2_data_sync.py"
            if not script_path.exists():
                logger.error(f"Data server script not found: {script_path}")
                update_data_server_status(port, False)
                return False

            self._data_server_process = await asyncio.create_subprocess_exec(
                "python", str(script_path), "serve", "--port", str(port),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Give it a moment to start
            await asyncio.sleep(0.5)

            if self._data_server_process.returncode is not None:
                logger.error("Data server failed to start")
                self._data_server_process = None
                update_data_server_status(port, False)
                return False

            logger.info(f"Data server started on port {port}")
            update_data_server_status(port, True)
            return True

        except Exception as e:
            logger.error(f"Failed to start data server: {e}")
            self._data_server_process = None
            update_data_server_status(port, False)
            return False

    async def stop_data_server(self) -> None:
        """Stop the aria2 data server."""
        if self._data_server_process is None:
            return

        try:
            self._data_server_process.terminate()
            await asyncio.wait_for(self._data_server_process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            self._data_server_process.kill()
        finally:
            self._data_server_process = None
            update_data_server_status(self._data_server_port, False)
            logger.info("Data server stopped")

    def is_data_server_running(self) -> bool:
        """Check if data server is running."""
        return (
            self._data_server_process is not None
            and self._data_server_process.returncode is None
        )

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
        update_sync_sources_count(len(sources))
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
            record_nfs_skip("training")
            return stats

        # Discover sources
        if sources is None:
            sources = await self.discover_sources()
        stats.sources_tried = len(sources)

        if not sources:
            logger.warning("No sources available for training data sync")
            return stats

        for transport in self._config.fallback_chain:
            if transport == "aria2" and self._config.enable_aria2:
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
                        stats.errors.extend(result.errors)
                        record_sync_coordinator_op(
                            "training",
                            "aria2",
                            result.files_synced,
                            result.bytes_transferred,
                            stats.duration_seconds,
                            success=result.success,
                            error_type="aria2_error" if result.errors else None,
                        )
                        if result.success or result.files_synced > 0:
                            logger.info(
                                f"Training data sync complete: {stats.files_synced} files, "
                                f"{stats.bytes_transferred / (1024*1024):.1f}MB via aria2"
                            )
                            return stats
                    except Exception as e:
                        stats.errors.append(f"aria2 sync failed: {e}")
                        logger.warning(f"aria2 training sync failed, trying fallback: {e}")
                        record_sync_coordinator_op(
                            "training",
                            "aria2",
                            0,
                            0,
                            time.time() - start_time,
                            success=False,
                            error_type=type(e).__name__,
                        )

            if transport == "ssh" and self._config.enable_ssh:
                files, bytes_sent, errors = await self._sync_with_rsync(
                    self._provider.training_dir,
                    "data/training",
                    ["*.npz", "*.h5"],
                )
                stats.files_synced = files
                stats.bytes_transferred = bytes_sent
                stats.transport_used = "ssh"
                stats.errors.extend(errors)
                stats.duration_seconds = time.time() - start_time
                record_sync_coordinator_op(
                    "training",
                    "ssh",
                    files,
                    bytes_sent,
                    stats.duration_seconds,
                    success=len(errors) == 0,
                    error_type="rsync_error" if errors else None,
                )
                if files > 0 or not errors:
                    return stats

            if transport == "p2p" and self._config.enable_p2p:
                files_npz, bytes_npz, errors_npz = await self._sync_with_p2p(
                    self._provider.training_dir,
                    "data/training/*.npz",
                )
                files_h5, bytes_h5, errors_h5 = await self._sync_with_p2p(
                    self._provider.training_dir,
                    "data/training/*.h5",
                )
                files = files_npz + files_h5
                bytes_sent = bytes_npz + bytes_h5
                errors = errors_npz + errors_h5
                stats.files_synced = files
                stats.bytes_transferred = bytes_sent
                stats.transport_used = "p2p"
                stats.errors.extend(errors)
                stats.duration_seconds = time.time() - start_time
                record_sync_coordinator_op(
                    "training",
                    "p2p",
                    files,
                    bytes_sent,
                    stats.duration_seconds,
                    success=len(errors) == 0,
                    error_type="p2p_error" if errors else None,
                )
                if files > 0 or not errors:
                    return stats

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
            record_nfs_skip("models")
            return stats

        # Discover sources
        if sources is None:
            sources = await self.discover_sources()
        stats.sources_tried = len(sources)

        if not sources:
            logger.warning("No sources available for model sync")
            return stats

        for transport in self._config.fallback_chain:
            if transport == "aria2" and self._config.enable_aria2:
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
                        stats.errors.extend(result.errors)
                        record_sync_coordinator_op(
                            "models",
                            "aria2",
                            result.files_synced,
                            result.bytes_transferred,
                            stats.duration_seconds,
                            success=result.success,
                            error_type="aria2_error" if result.errors else None,
                        )
                        if result.success or result.files_synced > 0:
                            logger.info(
                                f"Model sync complete: {stats.files_synced} models, "
                                f"{stats.bytes_transferred / (1024*1024):.1f}MB"
                            )
                            return stats
                    except Exception as e:
                        stats.errors.append(f"aria2 model sync failed: {e}")
                        logger.warning(f"aria2 model sync failed: {e}")
                        record_sync_coordinator_op(
                            "models",
                            "aria2",
                            0,
                            0,
                            time.time() - start_time,
                            success=False,
                            error_type=type(e).__name__,
                        )

            if transport == "ssh" and self._config.enable_ssh:
                include_patterns = ["*.pth", "*.onnx"]
                if model_ids:
                    include_patterns = [f"*{mid}*.pth" for mid in model_ids] + [
                        f"*{mid}*.onnx" for mid in model_ids
                    ]
                files, bytes_sent, errors = await self._sync_with_rsync(
                    self._provider.models_dir,
                    "models",
                    include_patterns,
                )
                stats.files_synced = files
                stats.bytes_transferred = bytes_sent
                stats.transport_used = "ssh"
                stats.errors.extend(errors)
                stats.duration_seconds = time.time() - start_time
                record_sync_coordinator_op(
                    "models",
                    "ssh",
                    files,
                    bytes_sent,
                    stats.duration_seconds,
                    success=len(errors) == 0,
                    error_type="rsync_error" if errors else None,
                )
                if files > 0 or not errors:
                    return stats

            if transport == "p2p" and self._config.enable_p2p:
                patterns = ["models/*.pth", "models/*.onnx"]
                if model_ids:
                    patterns = [f"models/*{mid}*.pth" for mid in model_ids] + [
                        f"models/*{mid}*.onnx" for mid in model_ids
                    ]
                total_files = 0
                total_bytes = 0
                total_errors: List[str] = []
                for pattern in patterns:
                    files, bytes_sent, errors = await self._sync_with_p2p(
                        self._provider.models_dir,
                        pattern,
                    )
                    total_files += files
                    total_bytes += bytes_sent
                    total_errors.extend(errors)
                stats.files_synced = total_files
                stats.bytes_transferred = total_bytes
                stats.transport_used = "p2p"
                stats.errors.extend(total_errors)
                stats.duration_seconds = time.time() - start_time
                record_sync_coordinator_op(
                    "models",
                    "p2p",
                    total_files,
                    total_bytes,
                    stats.duration_seconds,
                    success=len(total_errors) == 0,
                    error_type="p2p_error" if total_errors else None,
                )
                if total_files > 0 or not total_errors:
                    return stats

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
        local_games_dir = self._resolve_games_dir()

        # Skip if we have shared storage
        if self._provider.has_shared_storage:
            logger.info("Skipping game sync - using shared NFS storage")
            stats.transport_used = "nfs_shared"
            record_nfs_skip("games")
            return stats

        # Discover sources
        if sources is None:
            sources = await self.discover_sources()
        stats.sources_tried = len(sources)

        if not sources:
            logger.warning("No sources available for game sync")
            return stats

        for transport in self._config.fallback_chain:
            if transport == "aria2" and self._config.enable_aria2:
                aria2 = self._init_aria2()
                if aria2:
                    try:
                        result = await aria2.sync_games(
                            sources,
                            local_games_dir,
                        )
                        stats.files_synced = result.files_synced
                        stats.bytes_transferred = result.bytes_transferred
                        stats.transport_used = "aria2"
                        stats.duration_seconds = time.time() - start_time
                        stats.errors.extend(result.errors)
                        record_sync_coordinator_op(
                            "games",
                            "aria2",
                            result.files_synced,
                            result.bytes_transferred,
                            stats.duration_seconds,
                            success=result.success,
                            error_type="aria2_error" if result.errors else None,
                        )
                        if result.success or result.files_synced > 0:
                            logger.info(
                                f"Game sync complete: {stats.files_synced} databases, "
                                f"{stats.bytes_transferred / (1024*1024):.1f}MB"
                            )
                            return stats
                    except Exception as e:
                        stats.errors.append(f"aria2 game sync failed: {e}")
                        logger.warning(f"aria2 game sync failed: {e}")
                        record_sync_coordinator_op(
                            "games",
                            "aria2",
                            0,
                            0,
                            time.time() - start_time,
                            success=False,
                            error_type=type(e).__name__,
                        )

            if transport == "ssh" and self._config.enable_ssh:
                include_patterns = ["*.db"]
                files, bytes_sent, errors = await self._sync_with_rsync(
                    local_games_dir,
                    "data/games",
                    include_patterns,
                )
                stats.files_synced = files
                stats.bytes_transferred = bytes_sent
                stats.transport_used = "ssh"
                stats.errors.extend(errors)
                stats.duration_seconds = time.time() - start_time
                record_sync_coordinator_op(
                    "games",
                    "ssh",
                    files,
                    bytes_sent,
                    stats.duration_seconds,
                    success=len(errors) == 0,
                    error_type="rsync_error" if errors else None,
                )
                if files > 0 or not errors:
                    return stats

            if transport == "p2p" and self._config.enable_p2p:
                files, bytes_sent, errors = await self._sync_with_p2p(
                    local_games_dir,
                    "data/games/*.db",
                )
                stats.files_synced = files
                stats.bytes_transferred = bytes_sent
                stats.transport_used = "p2p"
                stats.errors.extend(errors)
                stats.duration_seconds = time.time() - start_time
                record_sync_coordinator_op(
                    "games",
                    "p2p",
                    files,
                    bytes_sent,
                    stats.duration_seconds,
                    success=len(errors) == 0,
                    error_type="p2p_error" if errors else None,
                )
                if files > 0 or not errors:
                    return stats

        stats.duration_seconds = time.time() - start_time
        return stats

    async def sync_high_quality_games(
        self,
        min_quality_score: float = 0.7,
        min_elo: Optional[float] = None,
        limit: int = 1000,
        sources: Optional[List[str]] = None,
    ) -> SyncStats:
        """Sync high-quality games with priority from the cluster.

        This method prioritizes syncing games with high quality scores (based on
        Elo, game length, and decisiveness) before bulk sync operations. This
        ensures that training nodes always have access to the best training data.

        Args:
            min_quality_score: Minimum quality score threshold (0.0-1.0)
            min_elo: Optional minimum average Elo threshold
            limit: Maximum number of high-quality games to sync
            sources: Specific sources to sync from (auto-discovers if None)

        Returns:
            SyncStats with operation results including quality metrics
        """
        start_time = time.time()
        stats = SyncStats(category="high_quality_games")

        # Skip if we have shared storage (no sync needed)
        if self._provider.has_shared_storage:
            logger.info("Skipping high-quality game sync - using shared NFS storage")
            stats.transport_used = "nfs_shared"
            return stats

        # Check manifest availability
        if not self._manifest:
            logger.warning("Cannot sync high-quality games - no manifest available")
            stats.errors.append("No manifest available for quality-based sync")
            return stats

        # Get priority queue entries from manifest
        priority_entries = self._manifest.get_priority_queue_batch(
            limit=limit,
            min_priority=min_quality_score,
        )

        if not priority_entries:
            logger.debug("No high-quality games pending in priority queue")
            # Try direct query from synced_games
            high_quality_games = self._manifest.get_high_quality_games(
                min_quality_score=min_quality_score,
                limit=limit,
            )
            if min_elo:
                high_quality_games = [g for g in high_quality_games if g.avg_player_elo >= min_elo]

            if not high_quality_games:
                logger.info("No high-quality games to sync")
                return stats

            stats.high_quality_games_synced = len(high_quality_games)
            if high_quality_games:
                stats.avg_quality_score = sum(g.quality_score for g in high_quality_games) / len(high_quality_games)
                stats.avg_elo = sum(g.avg_player_elo for g in high_quality_games) / len(high_quality_games)

            logger.info(
                f"Found {len(high_quality_games)} high-quality games "
                f"(avg quality: {stats.avg_quality_score:.3f}, avg Elo: {stats.avg_elo:.0f})"
            )
            stats.duration_seconds = time.time() - start_time
            return stats

        # Group entries by source host for efficient sync
        entries_by_host: Dict[str, List[PriorityQueueEntry]] = {}
        for entry in priority_entries:
            if entry.source_host not in entries_by_host:
                entries_by_host[entry.source_host] = []
            entries_by_host[entry.source_host].append(entry)

        # Discover sources if not provided
        if sources is None:
            sources = await self.discover_sources()
        stats.sources_tried = len(sources)

        if not sources:
            logger.warning("No sources available for high-quality game sync")
            return stats

        # Calculate quality stats from priority entries
        quality_scores = [e.priority_score for e in priority_entries]
        elo_scores = [e.avg_player_elo for e in priority_entries if e.avg_player_elo]
        stats.avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        stats.avg_elo = sum(elo_scores) / len(elo_scores) if elo_scores else 0.0

        # Sync high-quality games from each host
        synced_entry_ids: List[int] = []
        aria2 = self._init_aria2()

        for host, entries in entries_by_host.items():
            # Find matching source URL for this host
            host_sources = [s for s in sources if host in s]
            if not host_sources:
                logger.debug(f"No source URL found for host {host}")
                continue

            game_ids = [e.game_id for e in entries]
            logger.info(f"Syncing {len(game_ids)} high-quality games from {host}")

            if aria2:
                try:
                    # Use aria2 for priority sync
                    result = await aria2.sync_games(
                        host_sources,
                        self._resolve_games_dir(),
                        # Note: aria2 sync doesn't support game_id filtering yet
                        # This syncs all games from the source, but we track which ones
                        # were high-quality for metrics
                    )
                    stats.files_synced += result.files_synced
                    stats.bytes_transferred += result.bytes_transferred
                    stats.transport_used = "aria2"

                    if result.success:
                        # Mark priority queue entries as synced
                        synced_entry_ids.extend([e.id for e in entries])
                        stats.high_quality_games_synced += len(entries)

                except Exception as e:
                    stats.errors.append(f"aria2 sync from {host} failed: {e}")
                    logger.warning(f"High-quality sync from {host} failed: {e}")

        # Mark synced entries in manifest
        if synced_entry_ids:
            self._manifest.mark_queue_entries_synced(synced_entry_ids)
            logger.info(f"Marked {len(synced_entry_ids)} priority queue entries as synced")

        # Refresh quality lookup after sync
        self._quality_lookup_time = 0  # Force refresh
        self._refresh_quality_lookup()

        stats.duration_seconds = time.time() - start_time
        logger.info(
            f"High-quality game sync complete: {stats.high_quality_games_synced} games synced, "
            f"avg quality: {stats.avg_quality_score:.3f}, avg Elo: {stats.avg_elo:.0f}"
        )

        return stats

    async def full_cluster_sync(
        self,
        categories: Optional[List[SyncCategory]] = None,
        sync_high_quality_first: bool = True,
    ) -> ClusterSyncStats:
        """Perform a full sync of all data categories from the cluster.

        Args:
            categories: Categories to sync (all if None)
            sync_high_quality_first: If True, sync high-quality games with priority
                before bulk category syncs. This ensures training nodes have access
                to the best training data as quickly as possible.

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

        # Sync high-quality games FIRST (priority sync)
        if sync_high_quality_first:
            hq_stats = await self.sync_high_quality_games(
                min_quality_score=0.7,
                limit=1000,
                sources=sources,
            )
            stats.categories["high_quality_games"] = hq_stats
            stats.total_high_quality_games = hq_stats.high_quality_games_synced
            stats.avg_quality_score = hq_stats.avg_quality_score
            stats.total_files_synced += hq_stats.files_synced
            stats.total_bytes_transferred += hq_stats.bytes_transferred

            if hq_stats.transport_used:
                stats.transport_distribution[hq_stats.transport_used] = (
                    stats.transport_distribution.get(hq_stats.transport_used, 0) + 1
                )

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

        # Get quality distribution from manifest
        if self._manifest:
            stats.quality_distribution = self._manifest.get_quality_distribution()

        stats.nodes_synced = len(sources)
        stats.duration_seconds = time.time() - start_time

        logger.info(
            f"Full cluster sync complete: {stats.total_files_synced} files, "
            f"{stats.total_bytes_transferred / (1024*1024):.1f}MB in {stats.duration_seconds:.1f}s "
            f"(high-quality: {stats.total_high_quality_games} games)"
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

        # Stop data server
        await self.stop_data_server()

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
        status = {
            "provider": self._provider.provider_type.value,
            "shared_storage": self._provider.has_shared_storage,
            "running": self._running,
            "data_server": {
                "running": self.is_data_server_running(),
                "port": self._data_server_port if self.is_data_server_running() else None,
            },
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
                "data_server_port": self._config.aria2_data_server_port,
            },
            "quality": {
                "manifest_enabled": self._manifest is not None,
                "quality_lookup_size": len(self._quality_lookup),
                "elo_lookup_size": len(self._elo_lookup),
                "quality_lookup_age_seconds": time.time() - self._quality_lookup_time if self._quality_lookup_time else 0,
            },
        }

        # Add quality distribution if manifest is available
        if self._manifest:
            try:
                status["quality"]["distribution"] = self._manifest.get_quality_distribution()
                status["quality"]["priority_queue"] = self._manifest.get_priority_queue_stats()
            except Exception as e:
                logger.debug(f"Failed to get quality stats: {e}")

        return status


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


async def sync_high_quality_games(**kwargs) -> SyncStats:
    """Convenience function to sync high-quality games with priority."""
    return await SyncCoordinator.get_instance().sync_high_quality_games(**kwargs)


async def full_cluster_sync(**kwargs) -> ClusterSyncStats:
    """Convenience function for full cluster sync."""
    return await SyncCoordinator.get_instance().full_cluster_sync(**kwargs)


def get_quality_lookup() -> Dict[str, float]:
    """Get quality lookup dictionary for training integration."""
    return SyncCoordinator.get_instance().get_quality_lookup()


def get_elo_lookup() -> Dict[str, float]:
    """Get Elo lookup dictionary for training integration."""
    return SyncCoordinator.get_instance().get_elo_lookup()


# =============================================================================
# HIGH_QUALITY_DATA_AVAILABLE → Priority Sync Integration (December 2025)
# =============================================================================

class HighQualityDataSyncWatcher:
    """Watches for high-quality data events and triggers priority sync.

    Subscribes to HIGH_QUALITY_DATA_AVAILABLE events from the event bus and
    triggers immediate priority sync of high-quality games for training.

    Usage:
        from app.distributed.sync_coordinator import wire_high_quality_to_sync

        # Wire high-quality data events to priority sync
        watcher = wire_high_quality_to_sync()
    """

    def __init__(
        self,
        sync_cooldown_seconds: float = 60.0,
        min_quality_score: float = 0.7,
        max_games_per_sync: int = 500,
    ):
        """Initialize the high-quality data sync watcher.

        Args:
            sync_cooldown_seconds: Minimum time between syncs
            min_quality_score: Minimum quality score to consider "high quality"
            max_games_per_sync: Maximum games to sync per trigger
        """
        self.sync_cooldown_seconds = sync_cooldown_seconds
        self.min_quality_score = min_quality_score
        self.max_games_per_sync = max_games_per_sync

        self._last_sync_time: float = 0.0
        self._pending_hosts: Set[str] = set()
        self._subscribed = False
        self._sync_in_progress = False

    def subscribe_to_high_quality_events(self) -> bool:
        """Subscribe to HIGH_QUALITY_DATA_AVAILABLE events from the event bus.

        Returns:
            True if successfully subscribed
        """
        try:
            from app.distributed.data_events import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()
            bus.subscribe(DataEventType.HIGH_QUALITY_DATA_AVAILABLE, self._on_high_quality_data)
            self._subscribed = True
            logger.info("[HighQualityDataSyncWatcher] Subscribed to HIGH_QUALITY_DATA_AVAILABLE events")
            return True
        except Exception as e:
            logger.warning(f"[HighQualityDataSyncWatcher] Failed to subscribe: {e}")
            return False

    def unsubscribe(self) -> None:
        """Unsubscribe from high-quality data events."""
        if not self._subscribed:
            return

        try:
            from app.distributed.data_events import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()
            bus.unsubscribe(DataEventType.HIGH_QUALITY_DATA_AVAILABLE, self._on_high_quality_data)
            self._subscribed = False
        except Exception:
            pass

    def _on_high_quality_data(self, event) -> None:
        """Handle HIGH_QUALITY_DATA_AVAILABLE event.

        Triggers priority sync of high-quality games from the source host.
        """
        payload = event.payload if hasattr(event, 'payload') else {}

        source_host = payload.get("host", payload.get("source_host", ""))
        game_count = payload.get("game_count", payload.get("games", 0))
        avg_quality = payload.get("avg_quality", payload.get("quality_score", 0.7))

        logger.info(
            f"[HighQualityDataSyncWatcher] High-quality data available from {source_host}: "
            f"{game_count} games (avg quality: {avg_quality:.2f})"
        )

        # Track pending host
        if source_host:
            self._pending_hosts.add(source_host)

        # Try to trigger sync
        self._maybe_trigger_sync()

    def _maybe_trigger_sync(self) -> bool:
        """Potentially trigger priority sync.

        Returns:
            True if sync was triggered
        """
        now = time.time()

        # Check cooldown
        if now - self._last_sync_time < self.sync_cooldown_seconds:
            logger.debug("[HighQualityDataSyncWatcher] Sync cooldown active, queuing")
            return False

        # Check if sync already in progress
        if self._sync_in_progress:
            logger.debug("[HighQualityDataSyncWatcher] Sync already in progress")
            return False

        # Trigger async sync
        self._sync_in_progress = True
        self._last_sync_time = now

        # Get pending hosts and clear
        hosts = list(self._pending_hosts)
        self._pending_hosts.clear()

        # Schedule the sync (fire-and-forget, with error handling)
        try:
            asyncio.create_task(self._execute_priority_sync(hosts))
        except RuntimeError:
            # No running event loop - try to run synchronously
            try:
                asyncio.run(self._execute_priority_sync(hosts))
            except Exception as e:
                logger.warning(f"[HighQualityDataSyncWatcher] Failed to execute sync: {e}")
                self._sync_in_progress = False
                return False

        return True

    async def _execute_priority_sync(self, source_hosts: List[str]) -> None:
        """Execute priority sync of high-quality games.

        Args:
            source_hosts: Hosts to prioritize for sync
        """
        try:
            coordinator = SyncCoordinator.get_instance()

            logger.info(
                f"[HighQualityDataSyncWatcher] Starting priority sync "
                f"(quality>={self.min_quality_score}, max={self.max_games_per_sync} games)"
            )

            stats = await coordinator.sync_high_quality_games(
                min_quality_score=self.min_quality_score,
                limit=self.max_games_per_sync,
            )

            logger.info(
                f"[HighQualityDataSyncWatcher] Priority sync complete: "
                f"{stats.high_quality_games_synced} high-quality games synced "
                f"(avg quality: {stats.avg_quality_score:.2f})"
            )

            # Emit sync completed event
            self._emit_sync_completed(stats, source_hosts)

        except Exception as e:
            logger.error(f"[HighQualityDataSyncWatcher] Priority sync failed: {e}")
        finally:
            self._sync_in_progress = False

    def _emit_sync_completed(self, stats: SyncStats, source_hosts: List[str]) -> None:
        """Emit sync completed event."""
        try:
            from app.distributed.data_events import (
                DataEvent,
                DataEventType,
                get_event_bus,
            )

            event = DataEvent(
                event_type=DataEventType.DATA_SYNC_COMPLETED,
                payload={
                    "sync_type": "high_quality_priority",
                    "games_synced": stats.high_quality_games_synced,
                    "files_synced": stats.files_synced,
                    "bytes_transferred": stats.bytes_transferred,
                    "avg_quality_score": stats.avg_quality_score,
                    "source_hosts": source_hosts,
                    "duration_seconds": stats.duration_seconds,
                },
                source="high_quality_sync_watcher",
            )

            bus = get_event_bus()
            asyncio.create_task(bus.publish(event))

        except Exception as e:
            logger.debug(f"Failed to emit sync completed event: {e}")

    def force_sync(self) -> bool:
        """Force an immediate priority sync.

        Returns:
            True if sync was triggered
        """
        self._last_sync_time = 0  # Reset cooldown
        return self._maybe_trigger_sync()


# Singleton high-quality sync watcher
_hq_sync_watcher: Optional[HighQualityDataSyncWatcher] = None


def wire_high_quality_to_sync(
    sync_cooldown_seconds: float = 60.0,
    min_quality_score: float = 0.7,
    max_games_per_sync: int = 500,
) -> HighQualityDataSyncWatcher:
    """Wire HIGH_QUALITY_DATA_AVAILABLE events to priority sync.

    This connects high-quality data detection to immediate priority sync,
    ensuring that valuable training data is synced as soon as it's available.

    Args:
        sync_cooldown_seconds: Minimum time between syncs
        min_quality_score: Minimum quality score to consider "high quality"
        max_games_per_sync: Maximum games to sync per trigger

    Returns:
        HighQualityDataSyncWatcher instance
    """
    global _hq_sync_watcher

    _hq_sync_watcher = HighQualityDataSyncWatcher(
        sync_cooldown_seconds=sync_cooldown_seconds,
        min_quality_score=min_quality_score,
        max_games_per_sync=max_games_per_sync,
    )
    _hq_sync_watcher.subscribe_to_high_quality_events()

    logger.info(
        f"[wire_high_quality_to_sync] HIGH_QUALITY_DATA_AVAILABLE events wired to priority sync "
        f"(cooldown={sync_cooldown_seconds}s, min_quality={min_quality_score})"
    )

    return _hq_sync_watcher


def get_high_quality_sync_watcher() -> Optional[HighQualityDataSyncWatcher]:
    """Get the global high-quality sync watcher if configured."""
    return _hq_sync_watcher

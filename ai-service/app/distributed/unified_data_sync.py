"""Unified Data Sync Service - Consolidated data synchronization.

This module provides a single, unified entry point for all data synchronization
functionality, consolidating:
- streaming_data_collector.py (continuous incremental sync)
- manifest_replication.py (distributed manifest for fault tolerance)
- p2p_sync_client.py (HTTP fallback when SSH fails)
- content_deduplication.py (content-hash deduplication)
- ingestion_wal.py (crash-safe game ingestion)
- collector_watchdog.py (health monitoring and auto-restart)
- gossip_sync.py (P2P gossip-based data replication)

Features:
1. Continuous polling with configurable intervals
2. Multi-transport sync (SSH/rsync primary, P2P HTTP fallback)
3. Crash-safe ingestion via write-ahead log
4. Content-based deduplication
5. Distributed manifest replication
6. Self-healing with watchdog monitoring
7. Event-driven coordination
8. P2P gossip sync for resilient, eventually-consistent replication

Usage:
    # Programmatic
    service = UnifiedDataSyncService.from_config(config_path)
    await service.run()

    # CLI
    python -m app.distributed.unified_data_sync --config config/unified_loop.yaml

    # Via scripts (backward compatible)
    python scripts/unified_data_sync.py
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import signal
import socket
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import yaml

logger = logging.getLogger(__name__)

# Use centralized path constants
from app.utils.paths import AI_SERVICE_ROOT

# Import sub-components with graceful fallback
try:
    from app.distributed.manifest_replication import (
        ManifestReplicator,
        ReplicaHost,
        create_replicator_from_config,
    )
    HAS_MANIFEST_REPLICATION = True
except ImportError:
    HAS_MANIFEST_REPLICATION = False
    ManifestReplicator = None

try:
    from app.distributed.p2p_sync_client import (
        P2PFallbackSync,
        P2PSyncClient,
    )
    HAS_P2P_FALLBACK = True
except ImportError:
    HAS_P2P_FALLBACK = False
    P2PFallbackSync = None

try:
    from app.distributed.content_deduplication import (
        ContentDeduplicator,
        create_deduplicator,
    )
    HAS_CONTENT_DEDUP = True
except ImportError:
    HAS_CONTENT_DEDUP = False
    ContentDeduplicator = None

# Quality extraction for priority-based sync
try:
    from app.distributed.quality_extractor import (
        extract_batch_quality,
        extract_quality_from_synced_db,
        get_elo_lookup_from_service,
        compute_priority_score,
        QualityExtractorConfig,
    )
    from app.distributed.unified_manifest import GameQualityMetadata
    HAS_QUALITY_EXTRACTION = True
except ImportError:
    HAS_QUALITY_EXTRACTION = False
    extract_batch_quality = None
    GameQualityMetadata = None

# Storage provider for NFS detection and provider-specific paths
try:
    from app.distributed.storage_provider import (
        StorageProvider,
        get_storage_provider,
        should_sync_to_node,
        is_nfs_available,
    )
    HAS_STORAGE_PROVIDER = True
except ImportError:
    HAS_STORAGE_PROVIDER = False
    get_storage_provider = None
    should_sync_to_node = lambda x: True  # Always sync if no provider
    is_nfs_available = lambda: False

try:
    from app.distributed.ingestion_wal import (
        IngestionWAL,
        WALEntry,
        create_ingestion_wal,
    )
    HAS_INGESTION_WAL = True
except ImportError:
    HAS_INGESTION_WAL = False
    IngestionWAL = None

try:
    from app.distributed.data_events import (
        DataEventType,
        DataEvent,
        get_event_bus,
        emit_new_games,
    )
    HAS_EVENT_BUS = True
except ImportError:
    HAS_EVENT_BUS = False

# Import coordination helpers (consolidated imports)
from app.coordination.helpers import (
    # Sync lock
    has_sync_lock,
    acquire_sync_lock_safe,
    release_sync_lock_safe,
    # Bandwidth
    has_bandwidth_manager,
    get_transfer_priorities,
    request_bandwidth_safe,
    release_bandwidth_safe,
    # Orchestrator
    has_coordination,
    get_orchestrator_roles,
    get_registry_safe,
    # Cross-process events
    has_cross_process_events,
    publish_event_safe,
)

HAS_SYNC_LOCK = has_sync_lock()
HAS_BANDWIDTH_MANAGER = has_bandwidth_manager()
HAS_ORCHESTRATOR_REGISTRY = has_coordination()
HAS_CROSS_PROCESS_EVENTS = has_cross_process_events()

# Wrapper functions for backwards compatibility
def acquire_sync_lock(host: str, timeout: float = 120.0) -> bool:
    return acquire_sync_lock_safe(host, timeout)

def release_sync_lock(host: str) -> None:
    release_sync_lock_safe(host)

def request_bandwidth(host: str, mbps: float = 100.0, priority=None):
    return request_bandwidth_safe(host, mbps, priority)

def release_bandwidth(host: str) -> None:
    release_bandwidth_safe(host)

TransferPriority = get_transfer_priorities()
OrchestratorRole = get_orchestrator_roles()
get_registry = get_registry_safe

def publish_cross_process_event(event_type: str, payload: dict = None):
    publish_event_safe(event_type, payload)

# Circuit breaker still has its own import (not in helpers)
try:
    from app.distributed.circuit_breaker import (
        get_host_breaker,
        get_circuit_registry,
        FallbackChain,
        CircuitOpenError,
        CircuitState,
    )
    HAS_CIRCUIT_BREAKER = True
except ImportError:
    HAS_CIRCUIT_BREAKER = False
    FallbackChain = None

# Retry policy for network operations (2025-12)
try:
    from app.training.fault_tolerance import RetryPolicy
    HAS_RETRY_POLICY = True
except ImportError:
    HAS_RETRY_POLICY = False
    RetryPolicy = None

# Try to import unified manifest (consolidated implementation)
try:
    from app.distributed.unified_manifest import (
        DataManifest as UnifiedDataManifest,
        HostSyncState as UnifiedHostSyncState,
        create_manifest,
    )
    HAS_UNIFIED_MANIFEST = True
except ImportError:
    HAS_UNIFIED_MANIFEST = False
    UnifiedDataManifest = None
    UnifiedHostSyncState = None

# Gossip sync for P2P data replication (optional backend)
try:
    from app.distributed.gossip_sync import (
        GossipSyncDaemon,
        load_peer_config,
        GOSSIP_PORT,
    )
    HAS_GOSSIP_SYNC = True
except ImportError:
    HAS_GOSSIP_SYNC = False
    GossipSyncDaemon = None
    GOSSIP_PORT = 8771

# Aria2 transport for high-performance multi-connection downloads
try:
    from app.distributed.aria2_transport import (
        Aria2Transport,
        Aria2Config,
        create_aria2_transport,
        check_aria2_available,
    )
    HAS_ARIA2_TRANSPORT = True
except ImportError:
    HAS_ARIA2_TRANSPORT = False
    Aria2Transport = None
    check_aria2_available = lambda: False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SyncHostConfig:
    """Configuration for a data sync source host.

    This is the sync-specific host configuration with fields for remote data
    paths, sync roles, and ephemeral status. For general-purpose distributed
    worker configuration (with Cloudflare tunnels, worker ports, etc.), use:

        from app.distributed.hosts import HostConfig
    """

    name: str
    ssh_host: str
    ssh_user: str = "ubuntu"
    ssh_port: int = 22
    ssh_key: Optional[str] = None
    remote_db_path: str = "~/ringrift/ai-service/data/games"
    enabled: bool = True
    role: str = "selfplay"
    is_ephemeral: bool = False  # Vast.ai instances


# Backwards-compatible alias
HostConfig = SyncHostConfig


@dataclass
class SyncConfig:
    """Unified sync configuration."""
    # Polling intervals
    poll_interval_seconds: int = 60
    ephemeral_poll_interval_seconds: int = 15  # Faster for Vast.ai

    # Sync method
    sync_method: str = "incremental"  # "incremental" or "full"
    enable_p2p_fallback: bool = True
    p2p_port: int = 8770

    # Deduplication
    deduplication: bool = True
    content_deduplication: bool = True  # Hash-based content dedup

    # WAL settings
    enable_wal: bool = True
    wal_max_unprocessed: int = 10000

    # Manifest replication
    enable_manifest_replication: bool = True
    manifest_replication_interval: int = 300
    min_replicas: int = 2

    # Thresholds
    min_games_per_sync: int = 5
    training_threshold: int = 300

    # Timeouts
    ssh_timeout: int = 30
    rsync_timeout: int = 300

    # Retry settings
    max_consecutive_failures: int = 5
    retry_max_attempts: int = 3
    retry_base_delay_seconds: float = 5.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: int = 600

    # Watchdog settings
    enable_watchdog: bool = True
    watchdog_check_interval: int = 30
    watchdog_unhealthy_threshold: int = 3

    # Paths
    local_sync_dir: str = "data/games/synced"
    manifest_db_path: str = "data/data_manifest.db"
    wal_dir: str = "data/ingestion_wal"

    # Checksum validation
    checksum_validation: bool = True

    # Dead letter queue
    dead_letter_enabled: bool = True

    # Gossip sync (P2P data replication)
    enable_gossip_sync: bool = False  # Disabled by default, enable for P2P replication
    gossip_port: int = 8771

    # Aria2 transport (high-performance multi-connection downloads)
    enable_aria2_transport: bool = True  # Enable aria2 when available
    aria2_data_server_port: int = 8766  # Port for aria2 data servers
    aria2_connections_per_server: int = 16
    aria2_split: int = 16
    aria2_source_urls: List[str] = field(default_factory=list)  # type: ignore[misc]

    # Quality extraction for priority-based sync
    enable_quality_extraction: bool = True  # Extract quality scores during sync
    quality_elo_weight: float = 0.4
    quality_length_weight: float = 0.3
    quality_decisive_weight: float = 0.3
    min_quality_score_for_priority: float = 0.5  # Minimum quality for priority queue


# HostSyncState - use unified implementation if available
if HAS_UNIFIED_MANIFEST:
    HostSyncState = UnifiedHostSyncState
else:
    @dataclass
    class HostSyncState:
        """Sync state for a host (legacy fallback)."""
        name: str
        last_sync_time: float = 0.0
        last_game_count: int = 0
        total_games_synced: int = 0
        consecutive_failures: int = 0
        last_error: str = ""
        last_error_time: float = 0.0


# =============================================================================
# Data Manifest - Legacy implementation (kept for fallback)
# For new code, use: from app.distributed.unified_manifest import DataManifest
# =============================================================================

class _LegacyDataManifest:
    """Tracks synced game IDs for deduplication (legacy fallback)."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the manifest database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS synced_games (
                game_id TEXT PRIMARY KEY,
                source_host TEXT NOT NULL,
                source_db TEXT NOT NULL,
                synced_at REAL NOT NULL,
                board_type TEXT,
                num_players INTEGER,
                content_hash TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_synced_games_host
            ON synced_games(source_host);

            CREATE INDEX IF NOT EXISTS idx_synced_games_time
            ON synced_games(synced_at);

            CREATE INDEX IF NOT EXISTS idx_synced_games_content
            ON synced_games(content_hash);

            CREATE TABLE IF NOT EXISTS host_states (
                host_name TEXT PRIMARY KEY,
                last_sync_time REAL,
                last_game_count INTEGER,
                total_games_synced INTEGER,
                consecutive_failures INTEGER,
                last_error TEXT,
                last_error_time REAL
            );

            CREATE TABLE IF NOT EXISTS sync_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                host_name TEXT NOT NULL,
                sync_time REAL NOT NULL,
                games_synced INTEGER NOT NULL,
                duration_seconds REAL,
                success INTEGER NOT NULL,
                sync_method TEXT
            );

            CREATE TABLE IF NOT EXISTS dead_letter_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                source_host TEXT NOT NULL,
                source_db TEXT NOT NULL,
                error_message TEXT NOT NULL,
                error_type TEXT NOT NULL,
                added_at REAL NOT NULL,
                retry_count INTEGER DEFAULT 0,
                last_retry_at REAL,
                resolved INTEGER DEFAULT 0
            );
        """)
        conn.commit()
        conn.close()

    def is_game_synced(self, game_id: str) -> bool:
        """Check if a game has already been synced."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM synced_games WHERE game_id = ?", (game_id,))
        result = cursor.fetchone() is not None
        conn.close()
        return result

    def is_content_synced(self, content_hash: str) -> bool:
        """Check if content with this hash has been synced."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM synced_games WHERE content_hash = ?", (content_hash,))
        result = cursor.fetchone() is not None
        conn.close()
        return result

    def mark_games_synced(
        self,
        game_ids: List[str],
        source_host: str,
        source_db: str,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
        content_hashes: Optional[List[str]] = None,
    ):
        """Mark games as synced."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = time.time()

        for i, game_id in enumerate(game_ids):
            content_hash = content_hashes[i] if content_hashes and i < len(content_hashes) else None
            cursor.execute("""
                INSERT OR IGNORE INTO synced_games
                (game_id, source_host, source_db, synced_at, board_type, num_players, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (game_id, source_host, source_db, now, board_type, num_players, content_hash))

        conn.commit()
        conn.close()

    def get_synced_count(self) -> int:
        """Get total number of synced games."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM synced_games")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def save_host_state(self, state: HostSyncState):
        """Save host sync state."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO host_states
            (host_name, last_sync_time, last_game_count, total_games_synced,
             consecutive_failures, last_error, last_error_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            state.name, state.last_sync_time, state.last_game_count,
            state.total_games_synced, state.consecutive_failures,
            state.last_error, state.last_error_time
        ))
        conn.commit()
        conn.close()

    def load_host_state(self, host_name: str) -> Optional[HostSyncState]:
        """Load host sync state."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT host_name, last_sync_time, last_game_count, total_games_synced,
                   consecutive_failures, last_error, last_error_time
            FROM host_states WHERE host_name = ?
        """, (host_name,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return HostSyncState(
                name=row[0],
                last_sync_time=row[1] or 0.0,
                last_game_count=row[2] or 0,
                total_games_synced=row[3] or 0,
                consecutive_failures=row[4] or 0,
                last_error=row[5] or "",
                last_error_time=row[6] or 0.0,
            )
        return None

    def record_sync(
        self,
        host_name: str,
        games_synced: int,
        duration: float,
        success: bool,
        sync_method: str = "ssh",
    ):
        """Record a sync event to history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sync_history
            (host_name, sync_time, games_synced, duration_seconds, success, sync_method)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (host_name, time.time(), games_synced, duration, int(success), sync_method))
        conn.commit()
        conn.close()

    def add_to_dead_letter(
        self,
        game_id: str,
        source_host: str,
        source_db: str,
        error_message: str,
        error_type: str,
    ):
        """Add a failed game to the dead-letter queue."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO dead_letter_queue
            (game_id, source_host, source_db, error_message, error_type, added_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (game_id, source_host, source_db, error_message, error_type, time.time()))
        conn.commit()
        conn.close()

    def get_dead_letter_count(self) -> int:
        """Get count of unresolved dead-letter entries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM dead_letter_queue WHERE resolved = 0")
        count = cursor.fetchone()[0]
        conn.close()
        return count


# DataManifest - use unified implementation if available
if HAS_UNIFIED_MANIFEST:
    DataManifest = UnifiedDataManifest
else:
    DataManifest = _LegacyDataManifest


# =============================================================================
# Unified Data Sync Service
# =============================================================================

class UnifiedDataSyncService:
    """Unified service for all data synchronization needs.

    Consolidates:
    - Streaming data collection (rsync-based)
    - P2P HTTP fallback sync
    - Manifest replication
    - Content deduplication
    - WAL for crash safety
    - Health monitoring
    """

    def __init__(
        self,
        config: SyncConfig,
        hosts: List[HostConfig],
        manifest: DataManifest,
        http_port: int = 8772,
    ):
        self.config = config
        self.hosts = {h.name: h for h in hosts}
        self.manifest = manifest
        self.host_states: Dict[str, HostSyncState] = {}
        self.http_port = http_port

        # Runtime state
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._last_cycle_time: float = 0.0
        self._last_cycle_games: int = 0

        # Statistics
        self._sync_stats = {
            "ssh": 0,
            "p2p_http": 0,
            "aria2": 0,
            "failed": 0,
            "deduplicated": 0,
        }

        # Categorize hosts
        self._ephemeral_hosts: Set[str] = set()
        self._persistent_hosts: Set[str] = set()
        for host in hosts:
            if host.is_ephemeral:
                self._ephemeral_hosts.add(host.name)
            else:
                self._persistent_hosts.add(host.name)

        # Initialize sub-components
        self._manifest_replicator: Optional[ManifestReplicator] = None
        self._p2p_fallback: Optional[P2PFallbackSync] = None
        self._content_deduplicator: Optional[ContentDeduplicator] = None
        self._ingestion_wal: Optional[IngestionWAL] = None
        self._gossip_daemon: Optional[GossipSyncDaemon] = None
        self._aria2_transport: Optional[Aria2Transport] = None

        self._init_components()

        # Load previous host states
        for host in hosts:
            state = manifest.load_host_state(host.name)
            if state:
                self.host_states[host.name] = state
            else:
                self.host_states[host.name] = HostSyncState(name=host.name)

        # HTTP server state
        self._app = None
        self._http_runner = None

        # Track intervals
        self._last_manifest_replication: float = 0.0
        self._last_ephemeral_sync: float = 0.0
        self._last_persistent_sync: float = 0.0

    def _init_components(self):
        """Initialize sub-components based on config."""
        # Manifest replication
        if self.config.enable_manifest_replication and HAS_MANIFEST_REPLICATION:
            try:
                hosts_config = AI_SERVICE_ROOT / "config" / "remote_hosts.yaml"
                self._manifest_replicator = create_replicator_from_config(
                    manifest_path=AI_SERVICE_ROOT / self.config.manifest_db_path,
                    hosts_config_path=hosts_config,
                    min_replicas=self.config.min_replicas,
                )
                logger.info(f"Manifest replication enabled ({len(self._manifest_replicator.replica_hosts)} replicas)")
            except Exception as e:
                logger.warning(f"Could not initialize manifest replication: {e}")

        # P2P fallback
        if self.config.enable_p2p_fallback and HAS_P2P_FALLBACK:
            try:
                self._p2p_fallback = P2PFallbackSync(p2p_port=self.config.p2p_port)
                logger.info("P2P HTTP fallback enabled")
            except Exception as e:
                logger.warning(f"Could not initialize P2P fallback: {e}")

        # Content deduplication
        if self.config.content_deduplication and HAS_CONTENT_DEDUP:
            try:
                self._content_deduplicator = create_deduplicator(
                    manifest_db_path=AI_SERVICE_ROOT / self.config.manifest_db_path,
                )
                logger.info("Content-hash deduplication enabled")
            except Exception as e:
                logger.warning(f"Could not initialize content deduplication: {e}")

        # WAL for crash safety
        if self.config.enable_wal and HAS_INGESTION_WAL:
            try:
                self._ingestion_wal = create_ingestion_wal(
                    data_dir=AI_SERVICE_ROOT / "data",
                    max_unprocessed=self.config.wal_max_unprocessed,
                )
                logger.info("Write-ahead log enabled")
            except Exception as e:
                logger.warning(f"Could not initialize WAL: {e}")

        # Gossip sync for P2P data replication
        if self.config.enable_gossip_sync and HAS_GOSSIP_SYNC:
            try:
                hosts_config = AI_SERVICE_ROOT / "config" / "remote_hosts.yaml"
                peers_config = load_peer_config(hosts_config)
                node_id = socket.gethostname()
                self._gossip_daemon = GossipSyncDaemon(
                    node_id=node_id,
                    data_dir=AI_SERVICE_ROOT / "data" / "games",
                    peers_config=peers_config,
                    listen_port=self.config.gossip_port,
                )
                logger.info(f"Gossip sync enabled ({len(peers_config)} peers)")
            except Exception as e:
                logger.warning(f"Could not initialize gossip sync: {e}")

        # Aria2 transport for high-performance multi-connection downloads
        if self.config.enable_aria2_transport and HAS_ARIA2_TRANSPORT:
            try:
                if check_aria2_available():
                    aria2_config = {
                        "connections_per_server": self.config.aria2_connections_per_server,
                        "split": self.config.aria2_split,
                        "data_server_port": self.config.aria2_data_server_port,
                    }
                    self._aria2_transport = create_aria2_transport(aria2_config)
                    logger.info("Aria2 transport enabled (high-performance multi-connection downloads)")
                else:
                    logger.info("Aria2 transport disabled (aria2c not found)")
            except Exception as e:
                logger.warning(f"Could not initialize aria2 transport: {e}")

        # Quality extraction for priority-based sync
        self._elo_lookup = None
        self._quality_config = None
        if self.config.enable_quality_extraction and HAS_QUALITY_EXTRACTION:
            try:
                self._elo_lookup = get_elo_lookup_from_service()
                self._quality_config = QualityExtractorConfig(
                    elo_weight=self.config.quality_elo_weight,
                    length_weight=self.config.quality_length_weight,
                    decisive_weight=self.config.quality_decisive_weight,
                )
                logger.info("Quality extraction enabled for priority-based sync")
            except Exception as e:
                logger.warning(f"Could not initialize quality extraction: {e}")

    def _extract_and_store_quality(self, local_dir: Path, host_name: str) -> int:
        """Extract quality scores from synced databases and store in manifest.

        Args:
            local_dir: Directory containing synced .db files
            host_name: Name of the source host

        Returns:
            Number of games with quality scores extracted
        """
        if not HAS_QUALITY_EXTRACTION or not self.config.enable_quality_extraction:
            return 0

        try:
            # Extract quality from all synced databases
            quality_results = extract_quality_from_synced_db(
                local_dir=local_dir,
                elo_lookup=self._elo_lookup,
                config=self._quality_config or QualityExtractorConfig(),
            )

            total_extracted = 0
            for db_name, qualities in quality_results.items():
                # Store quality metadata in manifest
                marked = self.manifest.mark_games_synced_with_quality(
                    games=qualities,
                    source_host=host_name,
                    source_db=db_name,
                )
                total_extracted += marked

                # Add high-quality games to priority queue for future reference
                for quality in qualities:
                    if quality.quality_score >= self.config.min_quality_score_for_priority:
                        try:
                            self.manifest.add_to_priority_queue(
                                game_id=quality.game_id,
                                source_host=host_name,
                                source_db=db_name,
                                priority_score=quality.quality_score,
                                avg_player_elo=quality.avg_player_elo,
                                game_length=quality.game_length,
                                is_decisive=quality.is_decisive,
                            )
                        except Exception as e:
                            logger.debug(f"Failed to add {quality.game_id} to priority queue: {e}")

            if total_extracted > 0:
                logger.info(f"{host_name}: Extracted quality for {total_extracted} games")

            return total_extracted

        except Exception as e:
            logger.warning(f"{host_name}: Quality extraction failed: {e}")
            return 0

    def _build_ssh_args(self, host: HostConfig) -> str:
        """Build SSH arguments string.

        Uses secure TOFU (Trust On First Use) model:
        - StrictHostKeyChecking=accept-new: Accept new keys, reject changed keys
        - BatchMode=yes: Never prompt for passwords (fail fast instead)
        - ServerAliveInterval: Detect dead connections

        For new hosts, run: scripts/cluster_ssh_init.py --scan
        """
        args = [
            f"-o ConnectTimeout={self.config.ssh_timeout}",
            "-o StrictHostKeyChecking=accept-new",  # Secure: accept new, reject changed
            "-o BatchMode=yes",
            "-o ServerAliveInterval=15",
            "-o ServerAliveCountMax=3",
        ]
        if host.ssh_port != 22:
            args.append(f"-p {host.ssh_port}")
        if host.ssh_key:
            # Only add key if file exists
            key_path = os.path.expanduser(host.ssh_key)
            if os.path.exists(key_path):
                args.append(f"-i {key_path}")
        return " ".join(args)

    def _release_resources(
        self,
        bandwidth_allocated: bool,
        sync_lock_acquired: bool,
        host: HostConfig,
    ) -> None:
        """Release bandwidth and sync lock resources."""
        if bandwidth_allocated and HAS_BANDWIDTH_MANAGER:
            try:
                release_bandwidth(host.ssh_host)
            except Exception as e:
                logger.debug(f"{host.name}: bandwidth release error: {e}")

        if sync_lock_acquired and HAS_SYNC_LOCK:
            try:
                release_sync_lock(host.name)
            except Exception as e:
                logger.debug(f"{host.name}: sync lock release error: {e}")

    async def _get_remote_game_count(self, host: HostConfig) -> int:
        """Get the current game count on a remote host."""
        ssh_args = self._build_ssh_args(host)
        cmd = f"""ssh {ssh_args} {host.ssh_user}@{host.ssh_host} "
            cd {host.remote_db_path} 2>/dev/null && \\
            for db in *.db; do \\
                [ -f \\"\\$db\\" ] && sqlite3 \\"\\$db\\" 'SELECT COUNT(*) FROM games' 2>/dev/null || true; \\
            done | awk '{{s+=\\$1}} END {{print s+0}}'
        " """

        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.ssh_timeout
            )
            count = int(stdout.decode().strip() or "0")
            return count
        except Exception as e:
            raise RuntimeError(f"Failed to get game count: {e}")

    def _compute_file_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of a file.

        Args:
            file_path: Path to the file

        Returns:
            Hex-encoded SHA256 hash
        """
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def _validate_synced_files(
        self,
        local_dir: Path,
        host_name: str,
    ) -> Dict[str, Any]:
        """Validate checksums of synced database files.

        This validates:
        1. SQLite database integrity (PRAGMA integrity_check)
        2. Computes and records file checksums for future validation

        Args:
            local_dir: Directory containing synced files
            host_name: Name of source host

        Returns:
            Dict with validation results: {"valid": bool, "errors": list, "checksums": dict}
        """
        errors = []
        checksums = {}
        files_validated = 0

        try:
            for db_file in local_dir.glob("*.db"):
                try:
                    # Compute file checksum
                    file_checksum = await asyncio.get_event_loop().run_in_executor(
                        None, self._compute_file_checksum, db_file
                    )
                    checksums[db_file.name] = file_checksum

                    # Verify SQLite integrity
                    conn = sqlite3.connect(db_file)
                    cursor = conn.cursor()
                    cursor.execute("PRAGMA integrity_check")
                    result = cursor.fetchone()
                    conn.close()

                    if result[0] != "ok":
                        errors.append(f"{db_file.name}: SQLite integrity check failed: {result[0]}")
                        continue

                    files_validated += 1

                except sqlite3.DatabaseError as e:
                    errors.append(f"{db_file.name}: Database error: {e}")
                except Exception as e:
                    errors.append(f"{db_file.name}: Validation error: {e}")

            # Store checksums in manifest for future validation
            if checksums and hasattr(self.manifest, 'record_checksums'):
                self.manifest.record_checksums(host_name, checksums)

            return {
                "valid": len(errors) == 0,
                "files_validated": files_validated,
                "errors": errors,
                "checksums": checksums,
            }

        except Exception as e:
            return {
                "valid": False,
                "files_validated": 0,
                "errors": [f"Validation failed: {e}"],
                "checksums": {},
            }

    async def _sync_host_ssh(self, host: HostConfig, local_dir: Path) -> Tuple[int, str]:
        """Sync from host using SSH/rsync.

        Returns (games_synced, error_message).
        """
        ssh_args = self._build_ssh_args(host)
        rsync_cmd = f'rsync -avz --checksum -e "ssh {ssh_args}" {host.ssh_user}@{host.ssh_host}:{host.remote_db_path}/*.db {local_dir}/'

        try:
            process = await asyncio.create_subprocess_shell(
                rsync_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.rsync_timeout
            )

            if process.returncode != 0:
                return 0, stderr.decode()[:200]

            # Count games in synced DBs
            total = 0
            for db_file in local_dir.glob("*.db"):
                try:
                    conn = sqlite3.connect(db_file)
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM games")
                    total += cursor.fetchone()[0]
                    conn.close()
                except sqlite3.Error as e:
                    logger.debug(f"Could not count games in {db_file.name}: {e}")
                except Exception as e:
                    logger.warning(f"Unexpected error counting games in {db_file.name}: {e}")

            return total, ""

        except asyncio.TimeoutError:
            return 0, "timeout"
        except Exception as e:
            return 0, str(e)[:200]

    async def _sync_host_p2p(self, host: HostConfig, local_dir: Path) -> Tuple[int, str]:
        """Sync from host using P2P HTTP fallback.

        Returns (games_synced, error_message).
        """
        if not self._p2p_fallback:
            return 0, "P2P fallback not available"

        try:
            success, games, method = await self._p2p_fallback.sync_with_fallback(
                host=host.name,
                ssh_host=host.ssh_host,
                ssh_user=host.ssh_user,
                ssh_port=host.ssh_port,
                remote_db_path=host.remote_db_path,
                local_dir=local_dir,
            )

            if success:
                return games, ""
            else:
                return 0, "P2P sync failed"

        except Exception as e:
            return 0, str(e)[:200]

    async def _sync_host_aria2(self, host: HostConfig, local_dir: Path) -> Tuple[int, str]:
        """Sync from host using aria2 high-performance transport.

        Returns (games_synced, error_message).
        """
        if not self._aria2_transport:
            return 0, "Aria2 transport not available"

        try:
            # Build source URL from host info
            source_url = f"http://{host.ssh_host}:{self.config.aria2_data_server_port}"

            # Use aria2 to sync games
            result = await self._aria2_transport.sync_games(
                source_urls=[source_url],
                local_dir=local_dir,
                max_age_hours=168,  # 1 week
            )

            if result.success or result.files_synced > 0:
                return result.files_synced, ""
            elif result.errors:
                return 0, "; ".join(result.errors[:3])
            else:
                return 0, "Aria2 sync returned no files"

        except Exception as e:
            return 0, str(e)[:200]

    async def _sync_cluster_aria2(self) -> int:
        """Sync from all known cluster nodes using aria2.

        This method syncs from all configured source URLs simultaneously,
        using aria2's multi-source download capability for maximum throughput.

        Returns total games synced.
        """
        if not self._aria2_transport:
            return 0

        if not self.config.aria2_source_urls:
            # Build source URLs from hosts
            source_urls = [
                f"http://{h.ssh_host}:{self.config.aria2_data_server_port}"
                for h in self.hosts.values()
                if h.enabled
            ]
        else:
            source_urls = self.config.aria2_source_urls

        if not source_urls:
            return 0

        try:
            local_dir = AI_SERVICE_ROOT / self.config.local_sync_dir
            result = await self._aria2_transport.sync_games(
                source_urls=source_urls,
                local_dir=local_dir,
                max_age_hours=168,
            )

            if result.files_synced > 0:
                logger.info(
                    f"Aria2 cluster sync: {result.files_synced} files, "
                    f"{result.bytes_transferred / (1024*1024):.1f}MB in {result.duration_seconds:.1f}s"
                )

            return result.files_synced

        except Exception as e:
            logger.warning(f"Aria2 cluster sync failed: {e}")
            return 0

    async def _sync_host(self, host: HostConfig) -> int:
        """Sync games from a single host. Returns count of new games."""
        state = self.host_states[host.name]

        # NFS optimization: Skip sync if both nodes have shared NFS storage
        # This is a major performance win for Lambda Labs where all nodes
        # have access to the same 14PB shared filesystem
        if HAS_STORAGE_PROVIDER and not should_sync_to_node(host.name):
            if is_nfs_available():
                logger.debug(
                    f"{host.name}: Skipping sync - both nodes have shared NFS storage"
                )
                # Still update state to reflect availability via shared storage.
                state.last_sync_time = time.time()
                state.consecutive_failures = 0
                state.last_error = ""
                self.manifest.save_host_state(state)
                return 0

        # Circuit breaker check
        if HAS_CIRCUIT_BREAKER:
            breaker = get_host_breaker()
            if not breaker.can_execute(host.ssh_host):
                logger.debug(f"{host.name}: Circuit OPEN, skipping")
                return 0

        # Check backoff for failed hosts
        if state.consecutive_failures > 0:
            backoff = min(
                self.config.max_backoff_seconds,
                self.config.poll_interval_seconds * (self.config.backoff_multiplier ** state.consecutive_failures)
            )
            if time.time() - state.last_error_time < backoff:
                return 0

        start_time = time.time()
        local_dir = AI_SERVICE_ROOT / self.config.local_sync_dir / host.name
        local_dir.mkdir(parents=True, exist_ok=True)

        # Acquire resources
        sync_lock_acquired = False
        bandwidth_allocated = False

        if HAS_SYNC_LOCK:
            try:
                sync_lock_acquired = acquire_sync_lock(host.name, "unified-sync", wait=True, wait_timeout=60.0)
                if not sync_lock_acquired:
                    logger.debug(f"{host.name}: could not acquire sync lock")
                    return 0
            except Exception as e:
                logger.debug(f"{host.name}: sync lock error: {e}")

        if HAS_BANDWIDTH_MANAGER:
            try:
                bandwidth_allocated = request_bandwidth(
                    host.ssh_host, estimated_mb=100, priority=TransferPriority.NORMAL, timeout=30.0
                )
            except Exception as e:
                logger.debug(f"{host.name}: bandwidth request error: {e}")

        try:
            # Total timeout budget for all fallbacks (prevent hour-long stalls)
            # SSH (300s max) + P2P (300s max) + aria2 (300s max) = 15 min max total
            total_budget = 900.0  # 15 minutes max for all fallback attempts
            budget_start = time.time()

            def remaining_budget() -> float:
                return max(0, total_budget - (time.time() - budget_start))

            # Try SSH/rsync first
            games_synced, error = await self._sync_host_ssh(host, local_dir)
            sync_method = "ssh"

            # Fallback to P2P if SSH failed (check budget first)
            if games_synced == 0 and error and self.config.enable_p2p_fallback:
                if remaining_budget() < 10:
                    logger.warning(f"{host.name}: Timeout budget exhausted, skipping P2P fallback")
                else:
                    logger.info(f"{host.name}: SSH failed ({error}), trying P2P fallback")
                    games_synced, error = await self._sync_host_p2p(host, local_dir)
                    sync_method = "p2p_http" if games_synced > 0 else "failed"

            # Fallback to aria2 if P2P also failed (check budget first)
            if games_synced == 0 and error and self._aria2_transport:
                if remaining_budget() < 10:
                    logger.warning(f"{host.name}: Timeout budget exhausted, skipping aria2 fallback")
                else:
                    logger.info(f"{host.name}: P2P failed ({error}), trying aria2 transport")
                    games_synced, error = await self._sync_host_aria2(host, local_dir)
                    sync_method = "aria2" if games_synced > 0 else "failed"

            if games_synced == 0 and error:
                raise RuntimeError(error)

            # Write to WAL before processing (crash safety)
            if self._ingestion_wal and games_synced > 0:
                # WAL entry would be created here for each game
                pass

            # Validate checksums if enabled
            if self.config.checksum_validation and games_synced > 0:
                validation_result = await self._validate_synced_files(local_dir, host.name)
                if not validation_result["valid"]:
                    logger.warning(f"{host.name}: Checksum validation failed: {validation_result['errors']}")
                    # Continue anyway but log the warning - don't fail the sync

            # Extract and store quality scores for priority-based training
            if games_synced > 0:
                quality_count = self._extract_and_store_quality(local_dir, host.name)
                if quality_count > 0:
                    self._sync_stats["quality_extracted"] = self._sync_stats.get("quality_extracted", 0) + quality_count

            # Update statistics
            self._sync_stats[sync_method] += 1

            # Update state
            duration = time.time() - start_time
            state.last_sync_time = time.time()
            state.total_games_synced += games_synced
            state.consecutive_failures = 0
            state.last_error = ""

            self.manifest.record_sync(host.name, games_synced, duration, True, sync_method)
            self.manifest.save_host_state(state)

            # Circuit breaker success
            if HAS_CIRCUIT_BREAKER:
                get_host_breaker().record_success(host.ssh_host)

            # Emit events
            if HAS_EVENT_BUS and games_synced > 0:
                await emit_new_games(host.name, games_synced, state.total_games_synced, "unified_data_sync")

            if HAS_CROSS_PROCESS_EVENTS and games_synced > 0:
                publish_cross_process_event(
                    event_type="new_games",
                    payload={"host": host.name, "new_games": games_synced},
                    source="unified_data_sync",
                )

            return games_synced

        except Exception as e:
            state.consecutive_failures += 1
            state.last_error = str(e)[:200]
            state.last_error_time = time.time()

            duration = time.time() - start_time
            self.manifest.record_sync(host.name, 0, duration, False, "failed")
            self.manifest.save_host_state(state)

            self._sync_stats["failed"] += 1

            if HAS_CIRCUIT_BREAKER:
                get_host_breaker().record_failure(host.ssh_host, e)

            logger.warning(f"{host.name}: Sync failed ({state.consecutive_failures}): {e}")
            return 0

        finally:
            self._release_resources(bandwidth_allocated, sync_lock_acquired, host)

    async def run_collection_cycle(self) -> int:
        """Run one data collection cycle. Returns total new games."""
        total_new = 0
        tasks = []

        now = time.time()

        for host in self.hosts.values():
            if not host.enabled:
                continue

            state = self.host_states.get(host.name)
            if state and state.consecutive_failures >= self.config.max_consecutive_failures:
                continue

            # Determine if this host should sync this cycle
            if host.is_ephemeral:
                if now - self._last_ephemeral_sync < self.config.ephemeral_poll_interval_seconds:
                    continue
            else:
                if now - self._last_persistent_sync < self.config.poll_interval_seconds:
                    continue

            tasks.append(self._sync_host(host))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, int):
                    total_new += result

        # Update last sync times
        if self._ephemeral_hosts:
            self._last_ephemeral_sync = now
        if self._persistent_hosts:
            self._last_persistent_sync = now

        return total_new

    async def _replicate_manifest(self) -> None:
        """Replicate manifest to other hosts."""
        if not self._manifest_replicator:
            return

        if time.time() - self._last_manifest_replication < self.config.manifest_replication_interval:
            return

        try:
            replicas = await self._manifest_replicator.replicate_async()
            if replicas > 0:
                logger.info(f"Manifest replicated to {replicas} hosts")
            self._last_manifest_replication = time.time()
        except Exception as e:
            logger.warning(f"Manifest replication failed: {e}")

    async def _recover_from_wal(self) -> None:
        """Recover any unprocessed WAL entries on startup."""
        if not self._ingestion_wal:
            return

        try:
            stats = self._ingestion_wal.get_statistics()
            unprocessed = stats.get("unprocessed", 0)

            if unprocessed > 0:
                logger.info(f"Recovering {unprocessed} unprocessed WAL entries...")

                def process_entry(entry: WALEntry) -> bool:
                    # Mark game as synced in manifest
                    self.manifest.mark_games_synced(
                        [entry.game_id],
                        source_host=entry.source_host,
                        source_db="wal_recovery",
                    )
                    return True

                processed, failed = self._ingestion_wal.recover(process_entry)
                logger.info(f"WAL recovery: {processed} processed, {failed} failed")
        except Exception as e:
            logger.warning(f"WAL recovery failed: {e}")

    async def _recover_manifest_from_replicas(self) -> None:
        """Recover manifest from replicas if needed."""
        if not self._manifest_replicator:
            return

        try:
            recovered = await self._manifest_replicator.recover_if_needed()
            if recovered:
                logger.info("Recovered manifest from replica")
                # Reload host states after recovery
                for host in self.hosts.values():
                    state = self.manifest.load_host_state(host.name)
                    if state:
                        self.host_states[host.name] = state
        except Exception as e:
            logger.warning(f"Manifest recovery failed: {e}")

    async def _setup_http(self):
        """Set up HTTP API server."""
        try:
            from aiohttp import web
        except ImportError:
            logger.warning("aiohttp not installed, HTTP API disabled")
            return

        async def handle_health(request):
            return web.json_response({
                "status": "healthy",
                "running": self._running,
            })

        async def handle_status(request):
            return web.json_response({
                "running": self._running,
                "poll_interval": self.config.poll_interval_seconds,
                "total_synced": self.manifest.get_synced_count(),
                "dead_letter_count": self.manifest.get_dead_letter_count(),
                "hosts_count": len(self.hosts),
                "last_cycle_time": self._last_cycle_time,
                "last_cycle_games": self._last_cycle_games,
                "sync_stats": self._sync_stats,
                "ephemeral_hosts": list(self._ephemeral_hosts),
                "components": {
                    "manifest_replication": self._manifest_replicator is not None,
                    "p2p_fallback": self._p2p_fallback is not None,
                    "aria2_transport": self._aria2_transport is not None,
                    "content_deduplication": self._content_deduplicator is not None,
                    "wal": self._ingestion_wal is not None,
                    "gossip_sync": self._gossip_daemon is not None,
                },
                "gossip_status": self._gossip_daemon.get_status() if self._gossip_daemon else None,
            })

        async def handle_hosts(request):
            hosts_status = []
            for name, state in self.host_states.items():
                host = self.hosts.get(name)
                hosts_status.append({
                    "name": name,
                    "enabled": host.enabled if host else False,
                    "role": host.role if host else "unknown",
                    "is_ephemeral": host.is_ephemeral if host else False,
                    "last_sync_time": state.last_sync_time,
                    "total_games_synced": state.total_games_synced,
                    "consecutive_failures": state.consecutive_failures,
                    "healthy": state.consecutive_failures < self.config.max_consecutive_failures,
                })
            return web.json_response(hosts_status)

        async def handle_trigger(request):
            asyncio.create_task(self.run_collection_cycle())
            return web.json_response({"triggered": "all", "status": "started"})

        self._app = web.Application()
        self._app.router.add_get('/health', handle_health)
        self._app.router.add_get('/status', handle_status)
        self._app.router.add_get('/hosts', handle_hosts)
        self._app.router.add_post('/trigger', handle_trigger)

        self._http_runner = web.AppRunner(self._app)
        await self._http_runner.setup()
        site = web.TCPSite(self._http_runner, '0.0.0.0', self.http_port)
        await site.start()
        logger.info(f"HTTP API listening on port {self.http_port}")

    async def _cleanup_http(self):
        """Clean up HTTP server."""
        if self._http_runner:
            await self._http_runner.cleanup()

    async def run(self):
        """Main collection loop."""
        self._running = True
        logger.info(f"Starting unified data sync with {len(self.hosts)} hosts")

        # Recovery on startup
        await self._recover_manifest_from_replicas()
        await self._recover_from_wal()

        # Acquire orchestrator role
        has_role = False
        if HAS_ORCHESTRATOR_REGISTRY:
            try:
                registry = get_registry()
                node_id = socket.gethostname()
                has_role = registry.try_acquire(OrchestratorRole.DATA_SYNC, node_id)
                if has_role:
                    logger.info("Acquired DATA_SYNC orchestrator role")
                else:
                    logger.warning("Could not acquire DATA_SYNC role")
            except Exception as e:
                logger.warning(f"OrchestratorRegistry error: {e}")

        # Start HTTP API
        await self._setup_http()

        # Start gossip daemon if enabled
        if self._gossip_daemon:
            try:
                await self._gossip_daemon.start()
                logger.info("Gossip sync daemon started")
            except Exception as e:
                logger.warning(f"Failed to start gossip daemon: {e}")

        heartbeat_interval = 30
        last_heartbeat = time.time()

        try:
            while self._running:
                try:
                    cycle_start = time.time()
                    new_games = await self.run_collection_cycle()

                    self._last_cycle_time = time.time()
                    self._last_cycle_games = new_games

                    if new_games > 0:
                        total = self.manifest.get_synced_count()
                        logger.info(f"Cycle complete: {new_games} new games (total: {total})")

                        # Replicate manifest after successful sync
                        await self._replicate_manifest()

                    # Heartbeat
                    if HAS_ORCHESTRATOR_REGISTRY and has_role and (time.time() - last_heartbeat) >= heartbeat_interval:
                        try:
                            registry.heartbeat(OrchestratorRole.DATA_SYNC)
                            last_heartbeat = time.time()
                        except Exception as e:
                            logger.warning(f"Heartbeat error: {e}")

                except Exception as e:
                    logger.error(f"Cycle error: {e}")

                # Wait for next cycle
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self.config.poll_interval_seconds - elapsed)

                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=sleep_time)
                    break
                except asyncio.TimeoutError:
                    pass

        finally:
            if HAS_ORCHESTRATOR_REGISTRY and has_role:
                try:
                    registry.release(OrchestratorRole.DATA_SYNC)
                    logger.info("Released DATA_SYNC role")
                except Exception as e:
                    logger.debug(f"Error releasing DATA_SYNC role during shutdown: {e}")

            if self._p2p_fallback and hasattr(self._p2p_fallback, 'close'):
                try:
                    await self._p2p_fallback.close()
                    logger.info("P2P fallback sync closed")
                except Exception as e:
                    logger.warning(f"Error closing P2P fallback sync: {e}")

            # Stop gossip daemon
            if self._gossip_daemon:
                try:
                    await self._gossip_daemon.stop()
                    logger.info("Gossip sync daemon stopped")
                except Exception as e:
                    logger.warning(f"Error stopping gossip daemon: {e}")

            await self._cleanup_http()

        logger.info("Unified data sync stopped")

    def stop(self):
        """Request graceful shutdown."""
        self._running = False
        self._shutdown_event.set()

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {
            "running": self._running,
            "total_synced": self.manifest.get_synced_count(),
            "dead_letter_count": self.manifest.get_dead_letter_count(),
            "sync_stats": self._sync_stats.copy(),
            "host_count": len(self.hosts),
            "ephemeral_host_count": len(self._ephemeral_hosts),
            "last_cycle_time": self._last_cycle_time,
            "last_cycle_games": self._last_cycle_games,
        }

        if self._content_deduplicator:
            stats["deduplication"] = self._content_deduplicator.get_statistics()

        if self._ingestion_wal:
            stats["wal"] = self._ingestion_wal.get_statistics()

        if self._manifest_replicator:
            stats["manifest_replication"] = self._manifest_replicator.get_status()

        if self._gossip_daemon:
            stats["gossip_sync"] = self._gossip_daemon.get_status()

        # Quality extraction stats
        if self.config.enable_quality_extraction and HAS_QUALITY_EXTRACTION:
            try:
                stats["quality_distribution"] = self.manifest.get_quality_distribution()
                stats["priority_queue"] = self.manifest.get_priority_queue_stats()
            except Exception as e:
                logger.debug(f"Failed to get quality stats: {e}")

        return stats

    @classmethod
    def from_config(
        cls,
        config_path: Path,
        hosts_config_path: Optional[Path] = None,
    ) -> "UnifiedDataSyncService":
        """Create service from configuration files."""
        # Load main config
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        # Build sync config
        di = data.get("data_ingestion", {})
        aria2_cfg = di.get("aria2", {})
        cluster_cfg = data.get("cluster", {})

        # Gossip config can be in cluster section (gossip_sync_enabled) or data_ingestion (enable_gossip_sync)
        enable_gossip = cluster_cfg.get("gossip_sync_enabled", di.get("enable_gossip_sync", False))
        gossip_port = cluster_cfg.get("gossip_port", di.get("gossip_port", GOSSIP_PORT))

        config = SyncConfig(
            poll_interval_seconds=di.get("poll_interval_seconds", 60),
            ephemeral_poll_interval_seconds=di.get("ephemeral_poll_interval_seconds", 15),
            sync_method=di.get("sync_method", "incremental"),
            deduplication=di.get("deduplication", True),
            min_games_per_sync=di.get("min_games_per_sync", 5),
            checksum_validation=di.get("checksum_validation", True),
            retry_max_attempts=di.get("retry_max_attempts", 3),
            retry_base_delay_seconds=di.get("retry_base_delay_seconds", 5.0),
            dead_letter_enabled=di.get("dead_letter_enabled", True),
            training_threshold=data.get("training", {}).get("trigger_threshold_games", 300),
            enable_gossip_sync=enable_gossip,
            gossip_port=gossip_port,
            # Aria2 transport configuration
            enable_aria2_transport=aria2_cfg.get("enabled", True),
            aria2_data_server_port=aria2_cfg.get("data_server_port", 8766),
            aria2_connections_per_server=aria2_cfg.get("connections_per_server", 16),
            aria2_split=aria2_cfg.get("split", 16),
            aria2_source_urls=aria2_cfg.get("source_urls", []),
            # Quality extraction configuration
            enable_quality_extraction=di.get("enable_quality_extraction", True),
            quality_elo_weight=di.get("quality_elo_weight", 0.4),
            quality_length_weight=di.get("quality_length_weight", 0.3),
            quality_decisive_weight=di.get("quality_decisive_weight", 0.3),
            min_quality_score_for_priority=di.get("min_quality_score_for_priority", 0.5),
        )

        # Determine hosts config path
        if hosts_config_path is None:
            hosts_config_path = config_path.parent / data.get("hosts_config_path", "remote_hosts.yaml")

        # Load hosts
        hosts = load_hosts_from_yaml(hosts_config_path)

        # Initialize manifest
        manifest_path = config_path.parent.parent / config.manifest_db_path
        manifest = DataManifest(manifest_path)

        return cls(config, hosts, manifest)


# =============================================================================
# Host Loading
# =============================================================================

def load_hosts_from_yaml(path: Path) -> List[HostConfig]:
    """Load host configurations from YAML file."""
    if not path.exists():
        return []

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    hosts = []

    # Load standard hosts
    for name, host_data in data.get("standard_hosts", {}).items():
        hosts.append(HostConfig(
            name=name,
            ssh_host=host_data.get("ssh_host", ""),
            ssh_user=host_data.get("ssh_user", "ubuntu"),
            ssh_port=host_data.get("ssh_port", 22),
            remote_db_path=host_data.get("remote_path", "~/ringrift/ai-service/data/games"),
            role=host_data.get("role", "selfplay"),
            is_ephemeral=False,
        ))

    # Load vast hosts (ephemeral)
    for name, host_data in data.get("vast_hosts", {}).items():
        hosts.append(HostConfig(
            name=name,
            ssh_host=host_data.get("host", ""),
            ssh_user=host_data.get("user", "root"),
            ssh_port=host_data.get("port", 22),
            remote_db_path=host_data.get("remote_path", "/dev/shm/games"),
            role=host_data.get("role", "selfplay"),
            is_ephemeral=True,  # Vast.ai hosts are ephemeral
        ))

    return hosts


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for unified data sync service."""
    parser = argparse.ArgumentParser(
        description="Unified Data Sync Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default config
    python -m app.distributed.unified_data_sync

    # Run with custom config
    python -m app.distributed.unified_data_sync --config config/unified_loop.yaml

    # One-shot sync
    python -m app.distributed.unified_data_sync --once

    # Dry run (check what would sync)
    python -m app.distributed.unified_data_sync --dry-run
        """
    )
    parser.add_argument("--config", type=str, default="config/unified_loop.yaml", help="Config file")
    parser.add_argument("--hosts", type=str, default="config/remote_hosts.yaml", help="Hosts file")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--dry-run", action="store_true", help="Check what would sync")
    parser.add_argument("--interval", type=int, help="Override poll interval")
    parser.add_argument("--http-port", type=int, default=8772, help="HTTP API port")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [UnifiedSync] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    config_path = AI_SERVICE_ROOT / args.config
    hosts_path = AI_SERVICE_ROOT / args.hosts

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    # Create service
    service = UnifiedDataSyncService.from_config(config_path, hosts_path)

    if args.interval:
        service.config.poll_interval_seconds = args.interval

    service.http_port = args.http_port

    if args.dry_run:
        logger.info("Dry run - checking hosts...")
        for host in service.hosts.values():
            status = "ephemeral" if host.is_ephemeral else "persistent"
            logger.info(f"  {host.name}: {host.ssh_user}@{host.ssh_host}:{host.ssh_port} ({status})")
        logger.info(f"Components enabled:")
        logger.info(f"  Manifest replication: {service._manifest_replicator is not None}")
        logger.info(f"  P2P fallback: {service._p2p_fallback is not None}")
        logger.info(f"  Aria2 transport: {service._aria2_transport is not None}")
        logger.info(f"  Content dedup: {service._content_deduplicator is not None}")
        logger.info(f"  WAL: {service._ingestion_wal is not None}")
        logger.info(f"  Gossip sync: {service._gossip_daemon is not None}")
        return

    # Signal handlers
    def signal_handler(sig, frame):
        logger.info("Shutdown requested")
        service.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    if args.once:
        asyncio.run(service.run_collection_cycle())
        logger.info(f"One-shot complete: {service._last_cycle_games} games synced")
    else:
        asyncio.run(service.run())


if __name__ == "__main__":
    main()

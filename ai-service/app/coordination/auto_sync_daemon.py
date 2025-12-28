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

Module Structure
----------------
Classes:
    SyncStrategy          - Enum-like class for sync mode selection (lines 98-113)
    AutoSyncConfig        - Configuration dataclass with all sync settings (lines 127-260)
    SyncStats             - Statistics tracking (extends SyncDaemonStats) (lines 262-330)
    AutoSyncDaemon        - Main daemon class (lines 332-3524)

Factory Functions (lines 3526-3665):
    get_auto_sync_daemon()           - Singleton accessor
    reset_auto_sync_daemon()         - Reset singleton for testing
    create_ephemeral_sync_daemon()   - Factory for Vast.ai/spot nodes
    create_cluster_data_sync_daemon() - Factory for broadcast strategy
    create_training_sync_daemon()    - Factory for training-priority sync
    get_ephemeral_sync_daemon()      - Deprecated: Use AutoSyncDaemon(strategy=EPHEMERAL)
    get_cluster_data_sync_daemon()   - Deprecated: Use AutoSyncDaemon(strategy=BROADCAST)
    is_ephemeral_host()              - Detect if running on ephemeral node

AutoSyncDaemon Key Methods
--------------------------
Lifecycle:
    start()                  - Start background sync loops (lines ~450-550)
    stop()                   - Graceful shutdown with final sync (lines ~550-650)
    health_check()           - Return HealthCheckResult for monitoring (lines ~650-750)

Sync Operations:
    sync_to_node()           - Sync databases to a specific node
    sync_from_node()         - Pull databases from a specific node
    broadcast_sync()         - Push to all eligible nodes
    trigger_priority_sync()  - Immediate sync for urgent data

Background Loops:
    _sync_loop()             - Main sync cycle (respects interval)
    _gossip_loop()           - Gossip-based replication cycle
    _cleanup_loop()          - Disk cleanup when usage high

Node Selection:
    _get_eligible_targets()  - Filter nodes by disk/NFS/exclusion rules
    _prioritize_targets()    - Sort by ephemeral, training activity

Event Integration:
    - Subscribes to: NEW_GAMES_AVAILABLE, TRAINING_STARTED, NODE_RECOVERED
    - Emits: DATA_SYNC_STARTED, DATA_SYNC_COMPLETED, DATA_SYNC_FAILED

Usage:
    from app.coordination.auto_sync_daemon import AutoSyncDaemon

    daemon = AutoSyncDaemon()
    await daemon.start()

    # Or with specific strategy
    daemon = AutoSyncDaemon(strategy=SyncStrategy.EPHEMERAL)
    await daemon.start()
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import socket
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

# December 2025: Use consolidated daemon stats base class
from app.coordination.daemon_stats import SyncDaemonStats
from app.db.write_lock import is_database_safe_to_sync
from app.coordination.protocols import (
    CoordinatorStatus,
    HealthCheckResult,
    register_coordinator,
    unregister_coordinator,
)
from app.coordination.sync_integrity import check_sqlite_integrity
from app.coordination.sync_mutex import acquire_sync_lock, release_sync_lock
from app.core.async_context import fire_and_forget, safe_create_task

# Circuit breaker for fault-tolerant sync operations (December 2025)
try:
    from app.distributed.circuit_breaker import CircuitBreaker, CircuitState
    HAS_CIRCUIT_BREAKER = True
except ImportError:
    HAS_CIRCUIT_BREAKER = False
    CircuitBreaker = None
    CircuitState = None

# Resilient handler wrapper for fault-tolerant event handling (December 2025)
try:
    from app.coordination.handler_resilience import resilient_handler
    HAS_RESILIENT_HANDLER = True
except ImportError:
    HAS_RESILIENT_HANDLER = False
    resilient_handler = None

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


class SyncStrategy:
    """Sync strategy enum for AutoSyncDaemon (December 2025 consolidation).

    Strategies:
    - HYBRID: Default. Push-from-generator + gossip replication (persistent hosts)
    - EPHEMERAL: Aggressive 5-second sync for Vast.ai/spot instances
    - BROADCAST: Leader-only push to all eligible nodes
    - PULL: Coordinator pulls data from cluster (for recovery/backup)
    - AUTO: Auto-detect based on node type (ephemeral detection, leader status)
    """
    HYBRID = "hybrid"
    EPHEMERAL = "ephemeral"
    BROADCAST = "broadcast"
    PULL = "pull"  # December 2025: Coordinator-side reverse sync
    AUTO = "auto"


# Minimum moves per game for completeness validation (December 2025)
# Games with fewer moves are considered incomplete and skipped during sync
MIN_MOVES_PER_GAME: dict[tuple[str, int], int] = {
    ("hex8", 2): 20, ("hex8", 3): 30, ("hex8", 4): 40,
    ("square8", 2): 20, ("square8", 3): 30, ("square8", 4): 40,
    ("square19", 2): 50, ("square19", 3): 80, ("square19", 4): 100,
    ("hexagonal", 2): 50, ("hexagonal", 3): 80, ("hexagonal", 4): 100,
}
DEFAULT_MIN_MOVES = 5  # Fallback for unknown configurations


@dataclass
class AutoSyncConfig:
    """Configuration for automated data sync.

    Quality thresholds are loaded from app.config.thresholds for centralized configuration.

    December 2025: Added strategy parameter for consolidated sync modes:
    - hybrid: Push-from-generator + gossip (default)
    - ephemeral: Aggressive 5s sync for Vast.ai/spot
    - broadcast: Leader-only push to all nodes
    - auto: Detect based on node type
    """
    enabled: bool = True
    # Strategy selection (December 2025 consolidation)
    strategy: str = SyncStrategy.AUTO  # auto, hybrid, ephemeral, broadcast
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
    # Ephemeral-specific settings (December 2025 consolidation)
    ephemeral_poll_seconds: int = 5  # Aggressive polling for ephemeral hosts
    ephemeral_write_through: bool = True  # Wait for push confirmation
    ephemeral_write_through_timeout: int = 60  # Max wait for confirmation
    ephemeral_wal_enabled: bool = True  # Write-ahead log for durability
    # Broadcast-specific settings (December 2025 consolidation)
    broadcast_high_priority_configs: list[str] = field(
        default_factory=lambda: ["square8_2p", "hex8_2p", "hex8_3p", "hex8_4p"]
    )

    @classmethod
    def from_config_file(cls, config_path: Path | None = None) -> AutoSyncConfig:
        """Load configuration from distributed_hosts.yaml or unified_loop.yaml."""
        from app.config.cluster_config import load_cluster_config

        config = cls()

        # Load from distributed_hosts.yaml via cluster_config helper
        try:
            cluster_cfg = load_cluster_config(config_path)

            # Get sync_routing settings
            config.max_disk_usage_percent = cluster_cfg.sync_routing.max_disk_usage_percent
            config.target_disk_usage_percent = cluster_cfg.sync_routing.target_disk_usage_percent

            # Get auto_sync settings
            auto_sync = cluster_cfg.auto_sync
            config.enabled = auto_sync.enabled
            config.interval_seconds = auto_sync.interval_seconds
            config.gossip_interval_seconds = auto_sync.gossip_interval_seconds
            config.exclude_hosts = list(auto_sync.exclude_hosts)
            config.skip_nfs_sync = auto_sync.skip_nfs_sync
            config.max_concurrent_syncs = auto_sync.max_concurrent_syncs
            config.min_games_to_sync = auto_sync.min_games_to_sync
            config.bandwidth_limit_mbps = auto_sync.bandwidth_limit_mbps

            # December 27, 2025: Auto-exclude coordinator nodes and nodes with skip_sync_receive
            # This is Layer 2 of the multi-layer coordinator disk protection plan.
            # Previously, role: coordinator and skip_sync_receive: true in config were never
            # enforced. Now we automatically add these nodes to exclude_hosts.
            #
            # EXCEPTION: Nodes with use_external_storage: true are allowed to receive sync
            # because data is routed to external storage (e.g., mac-studio with OWC drive).
            for host_name, host_config in cluster_cfg.hosts_raw.items():
                if host_name in config.exclude_hosts:
                    continue  # Already excluded

                # Check skip_sync_receive flag - always exclude these
                if host_config.get("skip_sync_receive", False):
                    config.exclude_hosts.append(host_name)
                    logger.debug(
                        f"[AutoSyncConfig] Auto-excluding node with skip_sync_receive: {host_name}"
                    )
                    continue

                # Check role - coordinators should not receive synced data
                # UNLESS they have external storage configured
                #
                # December 2025: Added hostname pattern fallback for coordinator detection.
                # Some coordinator nodes may not have role: coordinator in config,
                # so we also check common coordinator hostname patterns.
                is_coordinator_by_role = host_config.get("role") == "coordinator"
                is_coordinator_by_flag = host_config.get("is_coordinator", False)
                is_coordinator_by_hostname = any(
                    pattern in host_name.lower()
                    for pattern in ["mac-studio", "localhost", "local-mac", "coordinator"]
                )
                is_coordinator = is_coordinator_by_role or is_coordinator_by_flag or is_coordinator_by_hostname

                if is_coordinator:
                    # Allow if external storage is configured
                    if host_config.get("use_external_storage", False):
                        logger.debug(
                            f"[AutoSyncConfig] Allowing coordinator with external storage: {host_name}"
                        )
                        continue
                    # Exclude coordinators without external storage
                    config.exclude_hosts.append(host_name)
                    detection_method = (
                        "role" if is_coordinator_by_role else
                        "flag" if is_coordinator_by_flag else
                        "hostname"
                    )
                    logger.debug(
                        f"[AutoSyncConfig] Auto-excluding coordinator node: {host_name} "
                        f"(detected via {detection_method})"
                    )

        except (OSError, ValueError, KeyError, AttributeError) as e:
            logger.warning(f"Failed to load cluster config: {e}")

        base_dir = Path(__file__).resolve().parent.parent.parent

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
class SyncStats(SyncDaemonStats):
    """Statistics for sync operations.

    December 2025: Now extends SyncDaemonStats for consistent tracking.
    Inherits: syncs_completed, syncs_failed, bytes_synced, last_sync_duration,
              errors_count, last_error, consecutive_failures, is_healthy(), etc.
    """

    # AutoSync-specific fields (not in base class)
    games_synced: int = 0
    # Quality filtering stats (December 2025)
    databases_skipped_quality: int = 0
    databases_quality_checked: int = 0
    # Quality extraction stats (December 2025)
    games_quality_extracted: int = 0
    games_added_to_priority: int = 0
    # Verification stats (December 2025 - Gap 4 fix)
    databases_verified: int = 0
    databases_verification_failed: int = 0
    last_verification_time: float = 0.0

    # Backward compatibility aliases
    @property
    def total_syncs(self) -> int:
        """Alias for operations_attempted (backward compatibility)."""
        return self.operations_attempted

    @property
    def successful_syncs(self) -> int:
        """Alias for syncs_completed (backward compatibility)."""
        return self.syncs_completed

    @property
    def failed_syncs(self) -> int:
        """Alias for syncs_failed (backward compatibility)."""
        return self.syncs_failed

    @property
    def bytes_transferred(self) -> int:
        """Alias for bytes_synced (backward compatibility)."""
        return self.bytes_synced

    @property
    def last_sync_time(self) -> float:
        """Alias for last_check_time (backward compatibility)."""
        return self.last_check_time

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary with AutoSync-specific fields."""
        base = super().to_dict()
        base.update({
            # AutoSync-specific
            "games_synced": self.games_synced,
            "databases_skipped_quality": self.databases_skipped_quality,
            "databases_quality_checked": self.databases_quality_checked,
            "games_quality_extracted": self.games_quality_extracted,
            "games_added_to_priority": self.games_added_to_priority,
            "databases_verified": self.databases_verified,
            "databases_verification_failed": self.databases_verification_failed,
            "last_verification_time": self.last_verification_time,
            # Backward compat aliases
            "total_syncs": self.total_syncs,
            "successful_syncs": self.successful_syncs,
            "failed_syncs": self.failed_syncs,
            "bytes_transferred": self.bytes_transferred,
            "last_sync_time": self.last_sync_time,
        })
        return base


class AutoSyncDaemon:
    """Daemon that orchestrates automated P2P data synchronization.

    December 2025 Consolidation: Unified daemon supporting multiple strategies:
    - HYBRID: Push-from-generator + gossip replication (default for persistent hosts)
    - EPHEMERAL: Aggressive 5s sync for Vast.ai/spot instances (from ephemeral_sync.py)
    - BROADCAST: Leader-only push to all nodes (from cluster_data_sync.py)
    - AUTO: Auto-detect based on node type

    Key features:
    - Gossip-based replication for eventual consistency
    - Provider-aware sync (skip NFS, prioritize ephemeral)
    - Coordinator exclusion (save disk space)
    - ClusterManifest for central tracking and disk management
    - Write-through mode for ephemeral hosts (zero data loss)
    - WAL (write-ahead log) for durability
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

        # December 2025: Resolve strategy based on auto-detection or explicit config
        self._resolved_strategy = self._resolve_strategy()
        self._is_ephemeral = self._resolved_strategy == SyncStrategy.EPHEMERAL
        self._is_broadcast = self._resolved_strategy == SyncStrategy.BROADCAST

        # Ephemeral-specific state (December 2025 consolidation)
        self._pending_games: list[dict[str, Any]] = []  # For ephemeral mode
        self._push_lock = asyncio.Lock()
        self._wal_initialized = False
        if self._is_ephemeral and self.config.ephemeral_wal_enabled:
            self._init_ephemeral_wal()

        # December 2025: Retry queue for failed write-through pushes
        self._pending_writes_file = Path("data/ephemeral_pending_writes.jsonl")
        self._pending_writes_task: asyncio.Task | None = None
        self._init_pending_writes_file()

        # Phase 9: Event subscription for DATA_STALE triggers
        self._subscribed = False
        self._urgent_sync_pending: dict[str, float] = {}  # config_key -> request_time

        # Quality extraction for training data prioritization (December 2025)
        self._quality_config: Any = None
        self._elo_lookup: Any = None
        if self.config.enable_quality_extraction:
            if HAS_QUALITY_EXTRACTION:
                try:
                    self._quality_config = QualityExtractorConfig()
                    self._elo_lookup = get_elo_lookup_from_service()
                    logger.info("Quality extraction enabled for training data prioritization")
                except (RuntimeError, OSError, ValueError, KeyError) as e:
                    logger.warning(f"Failed to initialize quality extraction: {e}")
                    self.config.enable_quality_extraction = False
            else:
                # December 2025: Log warning when quality extraction is requested but unavailable
                logger.warning(
                    "[AutoSyncDaemon] Quality extraction requested but module unavailable. "
                    "Install quality_extractor dependencies or set enable_quality_extraction=False"
                )
                self.config.enable_quality_extraction = False

        # Circuit breaker for node-level fault tolerance (December 2025)
        self._circuit_breaker: CircuitBreaker | None = None
        if HAS_CIRCUIT_BREAKER and CircuitBreaker:
            self._circuit_breaker = CircuitBreaker(
                failure_threshold=3,  # Open circuit after 3 consecutive failures
                recovery_timeout=60.0,  # Wait 60s before testing recovery
                half_open_max_calls=2,  # Allow 2 test calls in half-open state
                operation_type="sync",
                on_state_change=self._on_circuit_state_change,
            )
            logger.info("Circuit breaker enabled for sync fault tolerance")

        logger.info(
            f"AutoSyncDaemon initialized: node={self.node_id}, "
            f"provider={self._provider}, nfs={self._is_nfs_node}, "
            f"strategy={self._resolved_strategy}, "
            f"manifest={self._cluster_manifest is not None}"
        )

    def _resolve_strategy(self) -> str:
        """Resolve sync strategy based on config or auto-detection.

        December 2025 consolidation: Determines the best strategy for this node.

        Returns:
            One of SyncStrategy.HYBRID, EPHEMERAL, or BROADCAST
        """
        strategy = self.config.strategy

        # If explicit strategy specified, use it
        if strategy != SyncStrategy.AUTO:
            logger.info(f"Using explicit sync strategy: {strategy}")
            return strategy

        # Auto-detect based on node characteristics
        # Check for ephemeral host (Vast.ai, spot instances)
        if self._detect_ephemeral_host():
            logger.info("Auto-detected ephemeral host, using EPHEMERAL strategy")
            return SyncStrategy.EPHEMERAL

        # Check if we're the cluster leader (for broadcast mode)
        if self._is_cluster_leader():
            logger.info("Auto-detected cluster leader, using BROADCAST strategy")
            return SyncStrategy.BROADCAST

        # Default to hybrid
        logger.info("Using default HYBRID strategy")
        return SyncStrategy.HYBRID

    def _on_circuit_state_change(
        self, target: str, old_state: CircuitState, new_state: CircuitState
    ) -> None:
        """Handle circuit breaker state changes (December 2025).

        Logs state transitions and emits events for monitoring.
        """
        if old_state == new_state:
            return

        logger.warning(
            f"[AutoSyncDaemon] Circuit breaker for {target}: {old_state.value} -> {new_state.value}"
        )

        # Emit event for monitoring dashboards
        try:
            from app.distributed.data_events import DataEventType, emit_data_event

            emit_data_event(
                event_type=DataEventType.SYNC_NODE_UNREACHABLE
                if new_state.value == "open"
                else DataEventType.DATA_SYNC_COMPLETED,
                source="AutoSyncDaemon",
                metadata={
                    "target_node": target,
                    "old_state": old_state.value,
                    "new_state": new_state.value,
                    "event": "circuit_state_change",
                },
            )
        except (ImportError, RuntimeError, AttributeError):
            pass  # Event emission is best-effort

    def _detect_ephemeral_host(self) -> bool:
        """Detect if running on an ephemeral host.

        December 2025: Consolidated from ephemeral_sync.py
        """
        # Check Vast.ai
        if Path("/workspace").exists():
            return True

        # Check provider via canonical detection
        from app.config.cluster_config import get_host_provider
        provider = get_host_provider(self.node_id)
        if provider == "vast":
            return True

        # Check for spot instance markers
        import os
        if os.environ.get("AWS_SPOT_INSTANCE"):
            return True

        # Check RAM disk (Vast.ai uses /dev/shm for temp storage)
        return Path("/dev/shm/ringrift").exists()

    def _is_cluster_leader(self) -> bool:
        """Check if this node is the cluster leader.

        December 2025: Used for broadcast strategy auto-detection.
        """
        try:
            from urllib.request import Request, urlopen

            from app.config.ports import get_p2p_status_url

            url = get_p2p_status_url()
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=5) as resp:
                status = json.loads(resp.read().decode())
                leader_id = status.get("leader_id", "")
                return leader_id == self.node_id
        except (OSError, ValueError, json.JSONDecodeError, TimeoutError):
            return False

    def _init_ephemeral_wal(self) -> None:
        """Initialize write-ahead log for ephemeral mode durability.

        December 2025: Consolidated from ephemeral_sync.py
        """

        try:
            wal_path = Path("data/ephemeral_sync_wal.jsonl")
            wal_path.parent.mkdir(parents=True, exist_ok=True)

            if not wal_path.exists():
                wal_path.touch()

            self._wal_initialized = True
            self._wal_path = wal_path
            logger.debug(f"[AutoSyncDaemon] Ephemeral WAL initialized: {wal_path}")

            # Recover pending games from WAL
            self._load_ephemeral_wal()

        except OSError as e:
            logger.error(f"[AutoSyncDaemon] Failed to initialize ephemeral WAL: {e}")
            self._wal_initialized = False

    def _load_ephemeral_wal(self) -> None:
        """Load pending games from WAL on startup.

        December 2025: Consolidated from ephemeral_sync.py
        """

        if not self._wal_initialized or not hasattr(self, "_wal_path"):
            return

        try:
            if not self._wal_path.exists() or self._wal_path.stat().st_size == 0:
                return

            loaded_count = 0
            with open(self._wal_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if not entry.get("synced", False):
                            self._pending_games.append(entry)
                            loaded_count += 1
                    except json.JSONDecodeError:
                        continue

            if loaded_count > 0:
                logger.info(
                    f"[AutoSyncDaemon] Recovered {loaded_count} pending games from WAL"
                )

        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"[AutoSyncDaemon] Failed to load WAL: {e}")

    def _append_to_wal(self, game_entry: dict[str, Any]) -> None:
        """Append pending game to WAL for durability.

        December 2025: Consolidated from ephemeral_sync.py
        Called when a game is added to pending list, before sync attempt.
        """
        import os as os_module

        if not self._wal_initialized or not hasattr(self, "_wal_path"):
            return

        try:
            with open(self._wal_path, 'a') as f:
                f.write(json.dumps(game_entry) + '\n')
                f.flush()
                os_module.fsync(f.fileno())  # Force to disk

        except OSError as e:
            logger.debug(f"[AutoSyncDaemon] Failed to append to WAL: {e}")

    def _clear_wal(self) -> None:
        """Clear WAL after successful sync of all pending games.

        December 2025: Consolidated from ephemeral_sync.py
        Called when all pending games have been confirmed synced.
        """
        if not self._wal_initialized or not hasattr(self, "_wal_path"):
            return

        try:
            self._wal_path.write_text('')
            logger.debug("[AutoSyncDaemon] WAL cleared after successful sync")

        except OSError as e:
            logger.debug(f"[AutoSyncDaemon] Failed to clear WAL: {e}")

    # =========================================================================
    # December 2025: Retry queue for failed write-through pushes
    # =========================================================================

    def _init_pending_writes_file(self) -> None:
        """Initialize the pending writes retry queue file.

        December 2025: Added to prevent data loss from write-through timeouts.
        """
        try:
            self._pending_writes_file.parent.mkdir(parents=True, exist_ok=True)
            if not self._pending_writes_file.exists():
                self._pending_writes_file.touch()
            logger.debug(
                f"[AutoSyncDaemon] Pending writes file initialized: {self._pending_writes_file}"
            )
        except OSError as e:
            logger.error(f"[AutoSyncDaemon] Failed to initialize pending writes file: {e}")

    def _persist_failed_write(self, game_entry: dict[str, Any]) -> None:
        """Persist a failed write to the retry queue file.

        December 2025: Called when all retry attempts fail for write-through.

        Args:
            game_entry: The game entry that failed to sync
        """
        try:
            entry = {
                **game_entry,
                "failed_at": time.time(),
                "retry_count": 0,
            }
            with open(self._pending_writes_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
            logger.info(
                f"[AutoSyncDaemon] Persisted failed write for game {game_entry.get('game_id')} "
                "to retry queue"
            )
        except OSError as e:
            logger.error(f"[AutoSyncDaemon] Failed to persist write to retry queue: {e}")

    async def _push_with_retry(
        self,
        game_entry: dict[str, Any],
        max_attempts: int = 3,
        base_delay: float = 2.0,
    ) -> bool:
        """Push a game with exponential backoff retry.

        December 2025: Added to prevent data loss from transient network failures.

        Args:
            game_entry: The game entry to push
            max_attempts: Maximum retry attempts (default: 3)
            base_delay: Base delay in seconds (delays: 2s, 4s, 8s)

        Returns:
            True if push succeeded, False if all retries failed
        """
        db_path = game_entry.get("db_path")
        if not db_path:
            logger.warning("[AutoSyncDaemon] No db_path in game entry, cannot push")
            return False

        targets = await self._get_ephemeral_sync_targets()
        if not targets:
            logger.warning("[AutoSyncDaemon] No sync targets available for retry")
            return False

        for attempt in range(max_attempts):
            delay = base_delay * (2 ** attempt)  # 2s, 4s, 8s

            for target in targets[:3]:  # Try up to 3 targets per attempt
                try:
                    success = await asyncio.wait_for(
                        self._rsync_to_target(db_path, target),
                        timeout=self.config.ephemeral_write_through_timeout,
                    )
                    if success:
                        logger.debug(
                            f"[AutoSyncDaemon] Retry push succeeded on attempt {attempt + 1}"
                        )
                        return True
                except asyncio.TimeoutError:
                    logger.debug(
                        f"[AutoSyncDaemon] Retry push timeout to {target} "
                        f"(attempt {attempt + 1}/{max_attempts})"
                    )
                except (RuntimeError, OSError) as e:
                    logger.debug(
                        f"[AutoSyncDaemon] Retry push failed to {target}: {e} "
                        f"(attempt {attempt + 1}/{max_attempts})"
                    )

            # Wait before next retry attempt (except on last attempt)
            if attempt < max_attempts - 1:
                logger.debug(
                    f"[AutoSyncDaemon] Waiting {delay}s before retry attempt {attempt + 2}"
                )
                await asyncio.sleep(delay)

        # All retries failed
        logger.warning(
            f"[AutoSyncDaemon] All {max_attempts} retry attempts failed for "
            f"game {game_entry.get('game_id')}"
        )
        return False

    async def _process_pending_writes(self) -> None:
        """Background task to periodically retry failed writes.

        December 2025: Runs every 60 seconds to retry persisted failed writes.
        """
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute

                if not self._pending_writes_file.exists():
                    continue

                # Read pending writes
                pending_writes: list[dict[str, Any]] = []
                remaining_writes: list[dict[str, Any]] = []

                try:
                    with open(self._pending_writes_file) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                entry = json.loads(line)
                                pending_writes.append(entry)
                            except json.JSONDecodeError:
                                continue
                except OSError as e:
                    logger.debug(f"[AutoSyncDaemon] Failed to read pending writes: {e}")
                    continue

                if not pending_writes:
                    continue

                logger.info(
                    f"[AutoSyncDaemon] Processing {len(pending_writes)} pending writes"
                )

                for entry in pending_writes:
                    # Check if entry is too old (>24 hours)
                    failed_at = entry.get("failed_at", 0)
                    if time.time() - failed_at > 86400:  # 24 hours
                        logger.warning(
                            f"[AutoSyncDaemon] Abandoning stale pending write "
                            f"(age > 24h): {entry.get('game_id')}"
                        )
                        continue

                    # Try to push
                    success = await self._push_with_retry(entry, max_attempts=2)
                    if not success:
                        # Increment retry count and keep in queue
                        entry["retry_count"] = entry.get("retry_count", 0) + 1
                        if entry["retry_count"] < 5:  # Max 5 retry cycles
                            remaining_writes.append(entry)
                        else:
                            logger.error(
                                f"[AutoSyncDaemon] Permanently failed to sync game "
                                f"{entry.get('game_id')} after 5 retry cycles"
                            )
                            # Emit event for alerting
                            fire_and_forget(
                                self._emit_sync_failed(
                                    f"Permanent failure for game {entry.get('game_id')}"
                                ),
                                on_error=lambda exc: logger.debug(
                                    f"Failed to emit sync failed: {exc}"
                                ),
                            )

                # Rewrite the file with remaining writes
                try:
                    with open(self._pending_writes_file, "w") as f:
                        for entry in remaining_writes:
                            f.write(json.dumps(entry) + "\n")
                except OSError as e:
                    logger.error(
                        f"[AutoSyncDaemon] Failed to update pending writes file: {e}"
                    )

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[AutoSyncDaemon] Error in pending writes processor: {e}")

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

        # Use canonical provider detection
        from app.config.cluster_config import get_host_provider
        return get_host_provider(self.node_id)

    def _check_nfs_mount(self) -> bool:
        """Check if NFS storage is mounted."""
        nfs_path = Path("/lambda/nfs/RingRift")
        return nfs_path.exists() and nfs_path.is_dir()

    def _validate_database_completeness(self, db_path: str | Path) -> tuple[bool, str]:
        """Validate that games have complete move data (December 2025).

        Checks that games in the database have at least the minimum number of moves
        expected for their board type and player count. Games with incomplete move
        data (e.g., only 2 moves when 50+ expected) are a sign of the race condition
        where sync captured the database mid-write.

        Args:
            db_path: Path to the database file

        Returns:
            Tuple of (is_valid: bool, message: str)
            - (True, "OK") if database is valid
            - (False, "<reason>") if validation fails
        """
        db_path = Path(db_path)

        if not db_path.exists():
            return False, f"Database does not exist: {db_path}"

        try:
            # December 27, 2025: Use context manager to prevent connection leaks
            with sqlite3.connect(str(db_path), timeout=10.0) as conn:
                cursor = conn.cursor()

                # Get games with their move counts
                cursor.execute("""
                    SELECT g.game_id, g.board_type, g.num_players,
                           COUNT(m.move_number) as move_count
                    FROM games g
                    LEFT JOIN game_moves m ON g.game_id = m.game_id
                    GROUP BY g.game_id
                """)

                incomplete_games = []
                for row in cursor.fetchall():
                    game_id, board_type, num_players, move_count = row

                    # Get minimum moves for this configuration
                    min_moves = MIN_MOVES_PER_GAME.get(
                        (board_type, num_players),
                        DEFAULT_MIN_MOVES
                    )

                    if move_count < min_moves:
                        incomplete_games.append(
                            f"{game_id[:8]}... ({board_type}_{num_players}p): {move_count}/{min_moves} moves"
                        )

            if incomplete_games:
                # Limit message to first 5 incomplete games
                sample = incomplete_games[:5]
                if len(incomplete_games) > 5:
                    sample.append(f"... and {len(incomplete_games) - 5} more")
                return False, f"{len(incomplete_games)} incomplete games: {', '.join(sample)}"

            return True, "OK"

        except sqlite3.Error as e:
            return False, f"SQLite error: {e}"
        except OSError as e:
            return False, f"OS error: {e}"

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
        Dec 27, 2025: Added DATA_SYNC_COMPLETED event emission for pipeline coordination.

        Returns:
            Number of games synced (0 if skipped or no data).
        """
        if not self._running:
            logger.warning("[AutoSyncDaemon] sync_now() called but daemon not running")
            return 0

        try:
            games_synced = await self._sync_cycle()
            # December 27, 2025: Emit DATA_SYNC_COMPLETED for pipeline coordination
            # This ensures DataPipelineOrchestrator knows when sync finishes
            if games_synced and games_synced > 0:
                fire_and_forget(
                    self._emit_sync_completed(games_synced),
                    on_error=lambda exc: logger.debug(
                        f"Failed to emit sync completed from sync_now(): {exc}"
                    ),
                )
            return games_synced
        except (RuntimeError, OSError, ConnectionError) as e:
            logger.error(f"[AutoSyncDaemon] sync_now() error: {e}")
            fire_and_forget(
                self._emit_sync_failed(str(e)),
                on_error=lambda exc: logger.debug(
                    f"Failed to emit sync failed from sync_now(): {exc}"
                ),
            )
            return 0

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

        # December 2025: Setup termination handlers for ephemeral mode
        if self._is_ephemeral:
            self._setup_termination_handlers()

        # Phase 9: Subscribe to DATA_STALE events to trigger urgent sync
        self._subscribe_to_events()

        # Start gossip sync daemon
        await self._start_gossip_sync()

        # Start main sync loop
        self._sync_task = safe_create_task(
            self._sync_loop(),
            name="auto_sync_loop",
        )

        # December 2025: Start pending writes retry processor
        self._pending_writes_task = safe_create_task(
            self._process_pending_writes(),
            name="pending_writes_processor",
        )

        # Register with coordinator registry
        register_coordinator(self)

        logger.info(
            f"AutoSyncDaemon started: "
            f"interval={self.config.interval_seconds}s, "
            f"exclude={self.config.exclude_hosts}"
        )

    def _setup_termination_handlers(self) -> None:
        """Setup signal handlers for termination (ephemeral mode).

        December 2025: Consolidated from ephemeral_sync.py
        """
        import signal

        def handle_termination(sig, frame):
            logger.warning(f"[AutoSyncDaemon] Received termination signal {sig}")
            try:
                asyncio.get_running_loop()
                safe_create_task(
                    self._handle_termination(),
                    name="auto_sync_termination",
                )
            except RuntimeError:
                try:
                    asyncio.run(self._handle_termination())
                except (RuntimeError, OSError, asyncio.CancelledError) as e:
                    logger.error(f"[AutoSyncDaemon] Cannot run final sync: {e}")

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, handle_termination)
            except (OSError, ValueError) as e:
                logger.debug(f"[AutoSyncDaemon] Could not set handler for {sig}: {e}")

    async def _handle_termination(self) -> None:
        """Handle termination signal - do final sync.

        December 2025: Consolidated from ephemeral_sync.py
        """
        logger.warning("[AutoSyncDaemon] Handling termination - starting final sync")

        # Emit termination event for work migration
        await self._emit_termination_event()

        # Do final sync
        await self._final_sync()

    async def _emit_termination_event(self) -> None:
        """Emit HOST_OFFLINE event to notify coordinator of termination.

        December 2025: Consolidated from ephemeral_sync.py
        Enables coordinator to migrate work before ephemeral node terminates.
        """
        try:
            from app.coordination.event_router import emit_host_offline

            await emit_host_offline(
                host=self.node_id,
                reason=f"ephemeral_termination:pending_games={len(self._pending_games)}",
                last_seen=time.time(),
                source="AutoSyncDaemon",
            )
            logger.info(f"[AutoSyncDaemon] Emitted HOST_OFFLINE event for {self.node_id}")

        except ImportError:
            logger.debug("[AutoSyncDaemon] emit_host_offline not available")
        except (RuntimeError, OSError, asyncio.CancelledError) as e:
            logger.warning(f"[AutoSyncDaemon] Failed to emit termination event: {e}")

    async def _final_sync(self) -> None:
        """Perform final sync before shutdown (ephemeral mode).

        December 2025: Consolidated from ephemeral_sync.py
        """
        if not self._pending_games:
            logger.debug("[AutoSyncDaemon] No pending games for final sync")
            return

        logger.info(f"[AutoSyncDaemon] Final sync: {len(self._pending_games)} games pending")

        try:
            await self._push_pending_games(force=True)
        except (RuntimeError, OSError, asyncio.CancelledError, asyncio.TimeoutError) as e:
            logger.error(f"[AutoSyncDaemon] Final sync failed: {e}")
            # Dec 2025: Emit DATA_SYNC_FAILED for final sync failures
            fire_and_forget(
                self._emit_sync_failed(f"Final sync failed: {e}"),
                on_error=lambda exc: logger.debug(f"Failed to emit sync failed: {exc}"),
            )

    async def on_game_complete(
        self,
        game_result: dict[str, Any],
        db_path: Path | str | None = None,
    ) -> bool:
        """Handle game completion - queue for immediate push (ephemeral mode).

        December 2025: Consolidated from ephemeral_sync.py
        When write_through_enabled=True, waits for push confirmation before
        returning, ensuring the game is safely synced to a persistent node.

        Args:
            game_result: Game result dict with game_id, moves, etc.
            db_path: Path to database containing the game

        Returns:
            True if game was successfully synced (write-through mode) or queued,
            False if write-through failed (data at risk)
        """
        if not self._is_ephemeral:
            # Non-ephemeral mode: just track the game, normal sync will handle it
            return True

        game_id = game_result.get("game_id")

        # Add to pending
        game_entry = {
            "game_id": game_id,
            "db_path": str(db_path) if db_path else None,
            "timestamp": time.time(),
            "synced": False,
        }
        self._pending_games.append(game_entry)

        # Persist to WAL for durability
        self._append_to_wal(game_entry)

        # Immediate push if we have pending games
        if len(self._pending_games) >= 1:
            self._events_processed += 1

            # Write-through mode - wait for confirmation
            if self.config.ephemeral_write_through:
                try:
                    success = await asyncio.wait_for(
                        self._push_pending_games_with_confirmation(),
                        timeout=self.config.ephemeral_write_through_timeout,
                    )
                    if success:
                        logger.debug(f"[AutoSyncDaemon] Write-through success for game {game_id}")
                        return True
                    else:
                        logger.warning(f"[AutoSyncDaemon] Write-through push failed for game {game_id}")
                        # Dec 2025: Emit DATA_SYNC_FAILED for write-through failures (critical for ephemeral nodes)
                        fire_and_forget(
                            self._emit_sync_failed(f"Write-through push failed for game {game_id}"),
                            on_error=lambda exc: logger.debug(f"Failed to emit sync failed: {exc}"),
                        )
                        return False
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[AutoSyncDaemon] Write-through timeout for game {game_id} "
                        f"(timeout={self.config.ephemeral_write_through_timeout}s)"
                    )
                    # Dec 2025: Retry with exponential backoff before giving up
                    retry_success = await self._push_with_retry(
                        game_entry, max_attempts=3, base_delay=2.0
                    )
                    if retry_success:
                        logger.info(
                            f"[AutoSyncDaemon] Write-through succeeded after retry for game {game_id}"
                        )
                        return True

                    # All retries failed - persist to retry queue to prevent data loss
                    self._persist_failed_write(game_entry)

                    # Emit SYNC_STALLED for failover routing (Dec 2025)
                    fire_and_forget(
                        self._emit_sync_stalled(
                            target_node="write_through_target",
                            timeout_seconds=self.config.ephemeral_write_through_timeout,
                            data_type="game",
                        )
                    )
                    # Fall back to async push (for any remaining pending games)
                    fire_and_forget(self._push_pending_games())
                    return False
            else:
                # Legacy async push (fire-and-forget)
                await self._push_pending_games()
                return True

        return True

    async def _push_pending_games_with_confirmation(self) -> bool:
        """Push pending games and return True if at least one target succeeds.

        December 2025: Consolidated from ephemeral_sync.py
        Write-through variant that returns sync status.
        """
        if not self._pending_games:
            return True

        async with self._push_lock:
            games_to_push = self._pending_games.copy()
            self._pending_games.clear()

            logger.info(f"[AutoSyncDaemon] Write-through: pushing {len(games_to_push)} games")

            # Get unique DB paths
            db_paths = set()
            for game in games_to_push:
                if game.get("db_path"):
                    db_paths.add(game["db_path"])

            if not db_paths:
                logger.warning("[AutoSyncDaemon] No database paths to push")
                return False

            # Get sync targets
            targets = await self._get_ephemeral_sync_targets()
            if not targets:
                logger.warning("[AutoSyncDaemon] No sync targets available")
                self._pending_games.extend(games_to_push)  # Put back
                return False

            # Push each DB to at least one target
            any_success = False
            for db_path in db_paths:
                for target in targets[:3]:  # Try up to 3 targets
                    try:
                        success = await self._rsync_to_target(db_path, target)
                        if success:
                            any_success = True
                            # Mark games as synced
                            for game in games_to_push:
                                game["synced"] = True
                            break
                    except (RuntimeError, OSError, asyncio.TimeoutError) as e:
                        logger.debug(f"[AutoSyncDaemon] Push to {target} failed: {e}")

            if any_success:
                await self._emit_game_synced(
                    games_pushed=len(games_to_push),
                    target_nodes=targets[:1],
                    db_paths=list(db_paths),
                )
                self._clear_wal()

            return any_success

    async def _push_pending_games(self, force: bool = False) -> None:
        """Push pending games to sync targets.

        December 2025: Consolidated from ephemeral_sync.py
        """
        if not self._pending_games:
            return

        async with self._push_lock:
            games_to_push = self._pending_games.copy()
            self._pending_games.clear()

            logger.info(f"[AutoSyncDaemon] Pushing {len(games_to_push)} games")

            # Get unique DB paths
            db_paths = set()
            for game in games_to_push:
                if game.get("db_path"):
                    db_paths.add(game["db_path"])

            if not db_paths:
                logger.warning("[AutoSyncDaemon] No database paths to push")
                return

            # Get sync targets
            targets = await self._get_ephemeral_sync_targets()
            if not targets:
                logger.warning("[AutoSyncDaemon] No sync targets available")
                self._pending_games.extend(games_to_push)  # Put back
                return

            # Push each DB to targets
            successful_targets = []
            for db_path in db_paths:
                for target in targets[:3]:
                    try:
                        success = await self._rsync_to_target(db_path, target)
                        if success:
                            self._stats.games_synced += len(games_to_push)
                            successful_targets.append(target)
                            break
                    except (RuntimeError, OSError, asyncio.TimeoutError) as e:
                        logger.debug(f"[AutoSyncDaemon] Push to {target} failed: {e}")

            if successful_targets:
                await self._emit_game_synced(
                    games_pushed=len(games_to_push),
                    target_nodes=successful_targets,
                    db_paths=list(db_paths),
                )
                self._clear_wal()

    async def _get_ephemeral_sync_targets(self) -> list[str]:
        """Get sync targets for ephemeral mode.

        December 2025: Consolidated from ephemeral_sync.py
        """
        try:
            from app.coordination.sync_router import get_sync_router

            router = get_sync_router()
            targets = router.get_sync_targets(
                data_type="game",
                max_targets=3,
            )
            return [t.node_id for t in targets]

        except ImportError:
            logger.warning("[AutoSyncDaemon] SyncRouter not available")
            return []
        except (RuntimeError, OSError, ValueError, KeyError) as e:
            logger.error(f"[AutoSyncDaemon] Failed to get sync targets: {e}")
            return []

    async def _rsync_to_target(self, db_path: str, target_node: str) -> bool:
        """Rsync a database to a target node.

        December 2025: Consolidated from ephemeral_sync.py
        December 2025: Added sync mutex to prevent concurrent syncs to same target
        December 2025: Added circuit breaker for fault tolerance
        December 2025: Added write lock check to prevent syncing incomplete data

        Args:
            db_path: Local database path
            target_node: Target node ID

        Returns:
            True if successful
        """
        # Check write lock - don't sync if database is being written to
        if not is_database_safe_to_sync(db_path):
            logger.debug(
                f"[AutoSyncDaemon] Database {db_path} has active write lock, skipping sync"
            )
            return False

        # Dec 2025: CRITICAL - Checkpoint WAL before transfer to prevent corruption
        # Without this, WAL mode databases may transfer without their -wal files,
        # resulting in missing transactions and data corruption
        try:
            from app.coordination.sync_integrity import prepare_database_for_transfer
            success, msg = prepare_database_for_transfer(Path(db_path))
            if not success:
                logger.warning(
                    f"[AutoSyncDaemon] Failed to prepare {db_path} for transfer: {msg}"
                )
                # Continue anyway - may still work for non-WAL databases
        except ImportError:
            logger.debug("[AutoSyncDaemon] sync_integrity not available, skipping prepare step")
        except (OSError, sqlite3.Error) as e:
            logger.warning(f"[AutoSyncDaemon] Error preparing database: {e}")
            # Continue anyway - database may still be transferable

        # Check circuit breaker before attempting sync (December 2025)
        if self._circuit_breaker and not self._circuit_breaker.allow_request(target_node):
            logger.debug(
                f"[AutoSyncDaemon] Circuit open for {target_node}, skipping sync"
            )
            return False

        # Create lock key: target_node + filename to prevent concurrent writes
        db_name = Path(db_path).name if db_path else "unknown"
        lock_key = f"{target_node}:{db_name}"

        # Acquire sync lock to prevent race conditions
        if not acquire_sync_lock(lock_key, operation="rsync", timeout=60):
            logger.debug(f"[AutoSyncDaemon] Could not acquire lock for {lock_key}, skipping")
            return False

        success = False
        try:
            from app.coordination.sync_bandwidth import rsync_with_bandwidth_limit
            from app.coordination.sync_router import get_sync_router

            router = get_sync_router()
            cap = router.get_node_capability(target_node)

            if not cap:
                return False

            # Use centralized timeout (Dec 2025)
            from app.config.thresholds import RSYNC_TIMEOUT
            result = rsync_with_bandwidth_limit(
                source=db_path,
                target_host=target_node,
                timeout=RSYNC_TIMEOUT,
            )

            success = result.success
            return success

        except ImportError:
            success = await self._direct_rsync(db_path, target_node)
            return success
        except (RuntimeError, OSError, asyncio.TimeoutError) as e:
            logger.debug(f"[AutoSyncDaemon] Rsync error: {e}")
            success = False
            # Emit sync failure event (Dec 2025)
            await self._emit_sync_failure(target_node, db_path, str(e))
            return False
        finally:
            # Always release the lock
            release_sync_lock(lock_key)
            # Record success/failure with circuit breaker (December 2025)
            if self._circuit_breaker:
                if success:
                    self._circuit_breaker.record_success(target_node)
                else:
                    self._circuit_breaker.record_failure(target_node)

    async def _direct_rsync(self, db_path: str, target_node: str) -> bool:
        """Direct rsync without bandwidth management.

        December 2025: Consolidated from ephemeral_sync.py
        December 2025: Updated to use cluster_config helpers instead of inline YAML
        """
        import os
        import subprocess

        try:
            # December 2025: Use cluster_config helpers instead of inline YAML parsing
            from app.config.cluster_config import get_cluster_nodes, get_node_bandwidth_kbs

            nodes = get_cluster_nodes()
            node = nodes.get(target_node)

            if not node:
                logger.debug(f"[AutoSyncDaemon] Node {target_node} not found in cluster config")
                return False

            ssh_host = node.best_ip
            ssh_user = node.ssh_user or "ubuntu"
            ssh_key = node.ssh_key or "~/.ssh/id_cluster"

            # Dec 2025: Use storage path from node config (supports OWC routing)
            remote_games_path = node.get_storage_path("games")

            # Get bandwidth limit for this node
            bwlimit_args = []
            try:
                bwlimit_kbs = get_node_bandwidth_kbs(target_node)
                if bwlimit_kbs > 0:
                    bwlimit_args = [f"--bwlimit={bwlimit_kbs}"]
            except (KeyError, ValueError):
                pass

            if not ssh_host:
                return False

            ssh_key = os.path.expanduser(ssh_key)
            remote_full = f"{ssh_user}@{ssh_host}:{remote_games_path}/"

            rsync_cmd = [
                "rsync",
                "-avz",
                "--compress",
                *bwlimit_args,
                "-e", f"ssh -i {ssh_key} -o StrictHostKeyChecking=no -o ConnectTimeout=10",
                db_path,
                remote_full,
            ]

            # Use centralized timeout (Dec 2025)
            from app.config.thresholds import RSYNC_TIMEOUT
            result = await asyncio.to_thread(
                subprocess.run,
                rsync_cmd,
                capture_output=True,
                text=True,
                timeout=RSYNC_TIMEOUT,
            )
            return result.returncode == 0

        except subprocess.TimeoutExpired:
            logger.warning(f"[AutoSyncDaemon] Rsync timeout to {target_node}")
            # Emit sync stalled event for failover routing (Dec 2025)
            from app.config.thresholds import RSYNC_TIMEOUT
            await self._emit_sync_stalled(
                target_node=target_node,
                timeout_seconds=RSYNC_TIMEOUT,
                data_type="game",
            )
            # Also emit failure for general error tracking
            await self._emit_sync_failure(target_node, db_path, f"Rsync timeout after {RSYNC_TIMEOUT}s")
            return False
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            logger.debug(f"[AutoSyncDaemon] Rsync error: {e}")
            # Emit sync failure event (Dec 2025)
            await self._emit_sync_failure(target_node, db_path, str(e))
            return False

    async def _emit_game_synced(
        self,
        games_pushed: int,
        target_nodes: list[str],
        db_paths: list[str],
    ) -> None:
        """Emit GAME_SYNCED event for feedback loop coupling.

        December 2025: Consolidated from ephemeral_sync.py
        """
        try:
            from app.coordination.event_router import DataEventType, get_router

            router = get_router()
            if router:
                await router.publish(
                    event_type=DataEventType.GAME_SYNCED,
                    payload={
                        "node_id": self.node_id,
                        "games_pushed": games_pushed,
                        "target_nodes": target_nodes,
                        "db_paths": db_paths,
                        "is_ephemeral": self._is_ephemeral,
                        "timestamp": time.time(),
                    },
                    source="AutoSyncDaemon",
                )
        except (RuntimeError, AttributeError, ImportError) as e:
            logger.debug(f"[AutoSyncDaemon] Could not emit GAME_SYNCED event: {e}")

    async def _emit_sync_failure(
        self,
        target_node: str,
        db_path: str,
        error: str,
    ) -> None:
        """Emit DATA_SYNC_FAILED event when rsync fails.

        December 2025: Added for sync failure visibility and feedback loops.
        """
        try:
            from app.distributed.data_events import emit_data_sync_failed

            await emit_data_sync_failed(
                host=target_node,
                error=error,
                retry_count=0,
                source="AutoSyncDaemon",
            )
        except (RuntimeError, AttributeError, ImportError) as e:
            logger.debug(f"[AutoSyncDaemon] Could not emit DATA_SYNC_FAILED event: {e}")

    async def _emit_sync_stalled(
        self,
        target_node: str,
        timeout_seconds: float,
        data_type: str = "game",
        retry_count: int = 0,
    ) -> None:
        """Emit SYNC_STALLED event when sync operation times out.

        December 2025: Added to trigger failover to alternative sync sources.
        SYNC_STALLED is distinct from DATA_SYNC_FAILED - it specifically indicates
        a timeout condition that the SyncRouter can use to route around slow nodes.
        """
        try:
            from app.distributed.data_events import emit_sync_stalled

            await emit_sync_stalled(
                source_host=self.node_id,
                target_host=target_node,
                data_type=data_type,
                timeout_seconds=timeout_seconds,
                retry_count=retry_count,
                source="AutoSyncDaemon",
            )
        except (RuntimeError, AttributeError, ImportError) as e:
            logger.debug(f"[AutoSyncDaemon] Could not emit SYNC_STALLED event: {e}")

    # =========================================================================
    # Broadcast Mode Methods (December 2025 - from cluster_data_sync.py)
    # =========================================================================

    def discover_local_databases(self) -> list[Path]:
        """Find all game databases on this node that should be synced.

        December 2025: Consolidated from cluster_data_sync.py
        """
        base_dir = Path(__file__).resolve().parent.parent.parent
        data_dir = base_dir / "data" / "games"

        if not data_dir.exists():
            return []

        # Database patterns to sync
        sync_patterns = [
            "canonical_*.db",
            "gumbel_*.db",
            "selfplay_*.db",
            "synced/*.db",
        ]

        databases = []
        for pattern in sync_patterns:
            databases.extend(data_dir.glob(pattern))

        # Filter out empty databases
        databases = [db for db in databases if db.stat().st_size > 1024]

        # Sort by priority (high-priority configs first)
        high_priority_configs = frozenset(self.config.broadcast_high_priority_configs)

        def priority_key(path: Path) -> tuple[int, str]:
            name = path.stem
            for config in high_priority_configs:
                if config in name:
                    return (0, name)
            return (1, name)

        databases.sort(key=priority_key)

        logger.info(f"[AutoSyncDaemon] Found {len(databases)} databases to sync")
        return databases

    def get_bandwidth_for_node(self, node_id: str, provider: str = "default") -> int:
        """Get bandwidth limit in KB/s for a specific node.

        December 2025: Consolidated to use cluster_config.get_node_bandwidth_kbs()
        which provides a unified source of truth for bandwidth limits.

        Args:
            node_id: Target node ID
            provider: Provider name (unused, kept for backward compatibility)

        Returns:
            Bandwidth limit in KB/s
        """
        try:
            from app.config.cluster_config import get_node_bandwidth_kbs

            bw = get_node_bandwidth_kbs(node_id)
            logger.debug(f"[AutoSyncDaemon] Using bandwidth {bw}KB/s for {node_id}")
            return bw
        except ImportError:
            # Fallback if cluster_config not available
            logger.warning("[AutoSyncDaemon] cluster_config not available, using defaults")
            return 20_000  # Conservative default

    async def get_broadcast_targets(self) -> list[dict[str, Any]]:
        """Get nodes eligible to receive broadcast sync data.

        December 2025: Consolidated from cluster_data_sync.py

        Filters:
        - Not excluded by policy
        - Has sufficient free disk space
        - Not retired
        - Is reachable (recent heartbeat)
        """
        from urllib.request import Request, urlopen

        try:
            from app.config.ports import get_p2p_status_url
            from app.coordination.coordinator_config import get_exclusion_policy

            url = get_p2p_status_url()
            req = Request(url, headers={"Accept": "application/json"})

            with urlopen(req, timeout=10) as resp:
                status = json.loads(resp.read().decode())

        except (OSError, ValueError, json.JSONDecodeError, TimeoutError) as e:
            logger.warning(f"[AutoSyncDaemon] Failed to get P2P status: {e}")
            return []

        if not status:
            return []

        targets = []

        try:
            exclusion_policy = get_exclusion_policy()
        except ImportError:
            exclusion_policy = None

        peers = status.get("peers", {})
        for node_id, info in peers.items():
            # Skip excluded nodes
            if exclusion_policy and exclusion_policy.should_exclude(node_id):
                continue

            # Skip retired nodes
            if info.get("retired", False):
                continue

            # Check disk space
            disk_free = info.get("disk_free_gb", 0)
            min_disk = 50  # Default
            if exclusion_policy:
                min_disk = getattr(exclusion_policy, 'min_disk_free_gb', 50)
            if disk_free < min_disk:
                continue

            # Check for stale heartbeat (>5 min old)
            last_heartbeat = info.get("last_heartbeat", 0)
            if time.time() - last_heartbeat > 300:
                continue

            # Get host address
            host = info.get("host", "")
            if not host:
                continue

            # Detect provider for bandwidth hints
            provider = info.get("provider", "default")
            if not provider or provider == "default":
                node_lower = node_id.lower()
                if "lambda" in node_lower:
                    provider = "lambda"
                elif "runpod" in node_lower:
                    provider = "runpod"
                elif "nebius" in node_lower:
                    provider = "nebius"
                elif "vast" in node_lower:
                    provider = "vast"
                elif "vultr" in node_lower:
                    provider = "vultr"
                elif "hetzner" in node_lower:
                    provider = "hetzner"

            targets.append({
                "node_id": node_id,
                "host": host,
                "disk_free_gb": disk_free,
                "is_nfs": info.get("nfs_accessible", False),
                "provider": provider,
            })

        # Sort by disk space (push to nodes with most space first)
        targets.sort(key=lambda t: t["disk_free_gb"], reverse=True)

        logger.info(f"[AutoSyncDaemon] Found {len(targets)} broadcast targets")
        return targets

    async def sync_to_target_with_retry(
        self,
        source: Path,
        target: dict[str, Any],
        max_retries: int | None = None,
    ) -> dict[str, Any]:
        """Sync database to target with exponential backoff retry.

        December 2025: Added for improved sync reliability.

        Args:
            source: Source database path
            target: Target node info dict
            max_retries: Max retry attempts (default: SYNC_MAX_RETRIES)

        Returns:
            Sync result dict with success, bytes_transferred, duration, error
        """
        # Get retry defaults
        try:
            from app.config.coordination_defaults import RetryDefaults
            max_retries = max_retries or RetryDefaults.SYNC_MAX_RETRIES
            base_delay = RetryDefaults.SYNC_BASE_DELAY
            max_delay = RetryDefaults.SYNC_MAX_DELAY
            backoff_multiplier = RetryDefaults.BACKOFF_MULTIPLIER
        except ImportError:
            max_retries = max_retries or 3
            base_delay = 2.0
            max_delay = 30.0
            backoff_multiplier = 2.0

        last_result = None
        target_id = target.get("node_id", "unknown")

        for attempt in range(max_retries):
            # Try the sync
            result = await self.broadcast_sync_to_target(source, target)

            if result.get("success"):
                if attempt > 0:
                    logger.info(
                        f"[AutoSyncDaemon] Sync to {target_id} succeeded on attempt {attempt + 1}"
                    )
                return result

            last_result = result
            error = result.get("error", "Unknown")

            # Check if we should retry (not for all errors)
            if "Connection refused" in str(error) or "No route to host" in str(error):
                logger.debug(f"[AutoSyncDaemon] Not retrying {target_id}: host unreachable")
                break

            if attempt < max_retries - 1:
                # Calculate delay with exponential backoff
                delay = min(base_delay * (backoff_multiplier ** attempt), max_delay)
                # Add jitter (±10%)
                import random
                jitter = delay * 0.1 * (random.random() * 2 - 1)
                delay = delay + jitter

                logger.debug(
                    f"[AutoSyncDaemon] Sync to {target_id} failed (attempt {attempt + 1}), "
                    f"retrying in {delay:.1f}s: {error}"
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        if last_result and max_retries > 1:
            logger.warning(
                f"[AutoSyncDaemon] Sync to {target_id} failed after {max_retries} attempts: "
                f"{last_result.get('error', 'Unknown')}"
            )

        return last_result or {
            "source": str(source),
            "target": target_id,
            "success": False,
            "error": "Max retries exhausted",
        }

    async def broadcast_sync_to_target(
        self,
        source: Path,
        target: dict[str, Any],
    ) -> dict[str, Any]:
        """Push a database to a target node using rsync (broadcast mode).

        December 2025: Consolidated from cluster_data_sync.py

        Args:
            source: Source database path
            target: Target node info dict

        Returns:
            Sync result dict with success, bytes_transferred, duration, error
        """
        start_time = time.time()

        # NFS optimization: Lambda nodes share storage, no sync needed
        if target.get("is_nfs", False):
            logger.debug(f"[AutoSyncDaemon] Skipping sync to {target['node_id']}: NFS-connected")
            return {
                "source": str(source),
                "target": target["node_id"],
                "success": True,
                "bytes_transferred": 0,
                "duration_seconds": 0,
            }

        # Dec 2025: CRITICAL - Checkpoint WAL before transfer to prevent corruption
        try:
            from app.coordination.sync_integrity import prepare_database_for_transfer
            prep_success, prep_msg = prepare_database_for_transfer(source)
            if not prep_success:
                logger.warning(
                    f"[AutoSyncDaemon] Failed to prepare {source} for broadcast: {prep_msg}"
                )
        except ImportError:
            pass  # sync_integrity not available
        except (OSError, sqlite3.Error) as e:
            logger.warning(f"[AutoSyncDaemon] Error preparing database for broadcast: {e}")

        # Get provider-specific bandwidth limit
        bandwidth_kbps = self.get_bandwidth_for_node(
            target["node_id"],
            target.get("provider", "default"),
        )

        # Dec 2025: Get storage path from cluster config (supports OWC routing)
        from app.config.cluster_config import get_cluster_nodes
        cluster_nodes = get_cluster_nodes()
        node_config = cluster_nodes.get(target["node_id"])
        if node_config:
            games_path = node_config.get_storage_path("games")
        else:
            games_path = "~/ringrift/ai-service/data/games"

        # Build rsync command
        ssh_user = node_config.ssh_user if node_config else "ubuntu"
        target_path = f"{ssh_user}@{target['host']}:{games_path}/synced/"
        ssh_opts = (
            "ssh -i ~/.ssh/id_cluster "
            "-o StrictHostKeyChecking=no "
            "-o ConnectTimeout=10 "
            "-o TCPKeepAlive=yes "
            "-o ServerAliveInterval=30 "
            "-o ServerAliveCountMax=3"
        )
        # December 2025: Removed --partial to prevent corruption from stitched segments
        # on connection resets. Fresh transfers are safer than resumed partial ones.
        # December 27, 2025: Removed --inplace as it conflicts with --delay-updates
        # --delay-updates provides atomic file updates (safer for databases)
        # --inplace writes directly to file (faster but not atomic)
        # These are mutually exclusive - choose safety over speed
        cmd = [
            "rsync",
            "-avz",
            "--progress",
            f"--bwlimit={bandwidth_kbps}",
            "--timeout=60",
            "--delay-updates",
            "--checksum",
            "-e", ssh_opts,
            str(source),
            target_path,
        ]

        # Dynamic timeout: 2 seconds per MB, minimum 120s, maximum 1800s
        file_size_mb = source.stat().st_size / (1024 * 1024) if source.exists() else 100
        dynamic_timeout = max(120, min(1800, int(60 + file_size_mb * 2)))

        try:
            logger.info(f"[AutoSyncDaemon] Syncing {source.name} to {target['node_id']}")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            _stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=dynamic_timeout,
            )

            duration = time.time() - start_time

            if proc.returncode == 0:
                bytes_transferred = source.stat().st_size

                logger.info(
                    f"[AutoSyncDaemon] Synced {source.name} to {target['node_id']} in {duration:.1f}s"
                )
                return {
                    "source": str(source),
                    "target": target["node_id"],
                    "success": True,
                    "bytes_transferred": bytes_transferred,
                    "duration_seconds": duration,
                }
            else:
                error = stderr.decode().strip() if stderr else "Unknown error"
                logger.warning(f"[AutoSyncDaemon] Sync failed to {target['node_id']}: {error}")
                # Dec 2025: Emit DATA_SYNC_FAILED for individual sync failures
                fire_and_forget(
                    self._emit_sync_failure(target["node_id"], str(source), error),
                    on_error=lambda e: logger.debug(f"Failed to emit sync failure: {e}"),
                )
                return {
                    "source": str(source),
                    "target": target["node_id"],
                    "success": False,
                    "duration_seconds": duration,
                    "error": error,
                }

        except asyncio.TimeoutError:
            logger.error(f"[AutoSyncDaemon] Sync to {target['node_id']} timed out")
            # Emit SYNC_STALLED for failover routing (Dec 2025)
            fire_and_forget(
                self._emit_sync_stalled(
                    target_node=target["node_id"],
                    timeout_seconds=dynamic_timeout,
                    data_type="game",
                ),
                on_error=lambda e: logger.debug(f"Failed to emit SYNC_STALLED: {e}"),
            )
            return {
                "source": str(source),
                "target": target["node_id"],
                "success": False,
                "duration_seconds": time.time() - start_time,
                "error": "Timeout",
            }
        except (OSError, asyncio.CancelledError, subprocess.SubprocessError) as e:
            logger.error(f"[AutoSyncDaemon] Sync to {target['node_id']} error: {e}")
            # Dec 2025: Emit DATA_SYNC_FAILED for sync exceptions
            fire_and_forget(
                self._emit_sync_failure(target["node_id"], str(source), str(e)),
                on_error=lambda exc: logger.debug(f"Failed to emit sync failure: {exc}"),
            )
            return {
                "source": str(source),
                "target": target["node_id"],
                "success": False,
                "duration_seconds": time.time() - start_time,
                "error": str(e),
            }

    async def cleanup_stale_partials(self, max_age_hours: int = 24) -> int:
        """Remove stale .rsync-partial directories to prevent disk bloat.

        December 2025: Consolidated from cluster_data_sync.py

        Args:
            max_age_hours: Delete partial dirs older than this

        Returns:
            Number of files cleaned up
        """
        import datetime

        base_dir = Path(__file__).resolve().parent.parent.parent
        data_dir = base_dir / "data" / "games"
        partial_dir = data_dir / ".rsync-partial"

        cleaned = 0

        if partial_dir.exists():
            cutoff = datetime.datetime.now() - datetime.timedelta(hours=max_age_hours)

            for item in partial_dir.iterdir():
                try:
                    mtime = datetime.datetime.fromtimestamp(item.stat().st_mtime)
                    if mtime < cutoff and item.is_file():
                        item.unlink()
                        cleaned += 1
                        logger.debug(f"[AutoSyncDaemon] Cleaned stale partial: {item}")
                except OSError as e:
                    logger.debug(f"[AutoSyncDaemon] Error cleaning {item}: {e}")

        return cleaned

    async def broadcast_sync_cycle(self) -> int:
        """Execute one broadcast sync cycle (leader-only).

        December 2025: Consolidated from cluster_data_sync.py

        Returns:
            Number of successful syncs
        """
        if not self._is_broadcast:
            return 0

        logger.info("[AutoSyncDaemon] Starting broadcast sync cycle")

        # Clean up stale partial transfers periodically
        if self._stats.total_syncs % 10 == 0:
            try:
                cleaned = await self.cleanup_stale_partials()
                if cleaned > 0:
                    logger.info(f"[AutoSyncDaemon] Cleaned {cleaned} stale partial files")
            except OSError as e:
                logger.debug(f"[AutoSyncDaemon] Partial cleanup error: {e}")

        # Get eligible targets
        targets = await self.get_broadcast_targets()
        if not targets:
            logger.info("[AutoSyncDaemon] No broadcast targets available")
            return 0

        # Get databases to sync
        databases = self.discover_local_databases()
        if not databases:
            logger.info("[AutoSyncDaemon] No databases to sync")
            return 0

        # Sync each database to each target (with concurrency limit and retry)
        results: list[dict[str, Any]] = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent_syncs)

        async def sync_with_limit(db: Path, target: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                # December 2025: Use retry wrapper for improved reliability
                return await self.sync_to_target_with_retry(db, target)

        # Create all sync tasks
        tasks = []
        for db in databases:
            for target in targets:
                tasks.append(sync_with_limit(db, target))

        # Execute concurrently
        if tasks:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions
            results = [
                r for r in task_results
                if isinstance(r, dict)
            ]

            # Log summary
            successful = sum(1 for r in results if r.get("success", False))
            failed = len(results) - successful
            logger.info(
                f"[AutoSyncDaemon] Broadcast sync complete: {successful} successful, {failed} failed"
            )

            return successful

        return 0

    def _wrap_handler(self, handler):
        """Wrap handler with resilient_handler for fault tolerance (December 2025).

        Args:
            handler: The async event handler to wrap

        Returns:
            Wrapped handler with exception boundary and metrics, or original if unavailable
        """
        if HAS_RESILIENT_HANDLER and resilient_handler:
            return resilient_handler(handler, coordinator="AutoSyncDaemon")
        return handler

    def _subscribe_to_events(self) -> None:
        """Subscribe to events that trigger sync (Phase 9).

        December 2025: Handlers wrapped with resilient_handler for fault tolerance.

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
                bus.subscribe(DataEventType.DATA_STALE, self._wrap_handler(self._on_data_stale))
                logger.info("[AutoSyncDaemon] Subscribed to DATA_STALE")

            # Subscribe to SYNC_TRIGGERED for external requests
            if hasattr(DataEventType, 'SYNC_TRIGGERED'):
                bus.subscribe(DataEventType.SYNC_TRIGGERED, self._wrap_handler(self._on_sync_triggered))
                logger.info("[AutoSyncDaemon] Subscribed to SYNC_TRIGGERED")

            # Subscribe to NEW_GAMES_AVAILABLE for push-on-generate (Dec 2025)
            # Layer 1: Immediate push to neighbors when games are generated
            if hasattr(DataEventType, 'NEW_GAMES_AVAILABLE'):
                bus.subscribe(DataEventType.NEW_GAMES_AVAILABLE, self._wrap_handler(self._on_new_games_available))
                logger.info("[AutoSyncDaemon] Subscribed to NEW_GAMES_AVAILABLE (push-on-generate)")

            # Subscribe to DATA_SYNC_STARTED for sync coordination (Dec 2025)
            if hasattr(DataEventType, 'DATA_SYNC_STARTED'):
                bus.subscribe(DataEventType.DATA_SYNC_STARTED, self._wrap_handler(self._on_data_sync_started))
                logger.info("[AutoSyncDaemon] Subscribed to DATA_SYNC_STARTED")

            # Subscribe to MODEL_DISTRIBUTION_COMPLETE for model sync tracking (Dec 2025)
            if hasattr(DataEventType, 'MODEL_DISTRIBUTION_COMPLETE'):
                bus.subscribe(DataEventType.MODEL_DISTRIBUTION_COMPLETE, self._wrap_handler(self._on_model_distribution_complete))
                logger.info("[AutoSyncDaemon] Subscribed to MODEL_DISTRIBUTION_COMPLETE")

            # Subscribe to SELFPLAY_COMPLETE for immediate sync on selfplay completion (Dec 2025)
            # Phase F: Trigger sync immediately when selfplay batch finishes
            if hasattr(DataEventType, 'SELFPLAY_COMPLETE'):
                bus.subscribe(DataEventType.SELFPLAY_COMPLETE, self._wrap_handler(self._on_selfplay_complete))
                logger.info("[AutoSyncDaemon] Subscribed to SELFPLAY_COMPLETE (immediate sync)")

            # Subscribe to TRAINING_STARTED for priority sync to training nodes (Dec 2025)
            if hasattr(DataEventType, 'TRAINING_STARTED'):
                bus.subscribe(DataEventType.TRAINING_STARTED, self._wrap_handler(self._on_training_started))
                logger.info("[AutoSyncDaemon] Subscribed to TRAINING_STARTED (priority sync)")

            # Subscribe to NODE_RECOVERED to clear exclusion state for recovered nodes (Dec 2025)
            if hasattr(DataEventType, 'NODE_RECOVERED'):
                bus.subscribe(DataEventType.NODE_RECOVERED, self._wrap_handler(self._on_node_recovered))
                logger.info("[AutoSyncDaemon] Subscribed to NODE_RECOVERED (exclusion reset)")

            self._subscribed = True
            if HAS_RESILIENT_HANDLER:
                logger.info("[AutoSyncDaemon] Event handlers wrapped with resilient_handler")
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
            # Dec 27, 2025: Handle both "config_key" and "config" for compatibility
            config_key = payload.get("config_key") or payload.get("config", "")
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
            logger.warning(f"[AutoSyncDaemon] Error handling DATA_SYNC_STARTED: {e}")

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
            logger.warning(f"[AutoSyncDaemon] Error handling MODEL_DISTRIBUTION_COMPLETE: {e}")

    async def _on_selfplay_complete(self, event) -> None:
        """Handle SELFPLAY_COMPLETE event - immediate sync on selfplay finish.

        Phase F (December 2025): When a selfplay batch completes, immediately
        trigger game data sync to propagate fresh training data across the cluster.
        This closes the loop from selfplay -> sync -> training for faster iteration.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            config_key = payload.get("config_key", "")
            games_played = payload.get("games_played", 0)

            # Only sync if we have a meaningful batch
            min_games = self.config.min_games_to_sync or 5
            if games_played < min_games:
                logger.debug(
                    f"[AutoSyncDaemon] Selfplay sync skipped: {config_key} "
                    f"({games_played} < {min_games} min games)"
                )
                return

            logger.info(
                f"[AutoSyncDaemon] SELFPLAY_COMPLETE: {config_key} "
                f"({games_played} games) - triggering immediate cluster sync"
            )

            self._events_processed += 1

            # Trigger immediate sync for this config
            fire_and_forget(self._trigger_urgent_sync(config_key))

            # Also push to neighbors for Layer 1 replication
            fire_and_forget(self._push_to_neighbors(config_key, games_played))

        except (RuntimeError, OSError, ConnectionError) as e:
            self._errors_count += 1
            self._last_error = str(e)
            logger.error(f"[AutoSyncDaemon] Error handling SELFPLAY_COMPLETE: {e}")

    async def _on_training_started(self, event) -> None:
        """Handle TRAINING_STARTED event - priority sync to training node.

        December 2025: When training starts on a node, we should immediately
        sync fresh data to that node to ensure it has the latest training samples.
        This reduces latency in the training feedback loop.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            config_key = payload.get("config_key", "")
            node_id = payload.get("node_id", "")

            if not node_id:
                logger.debug("[AutoSyncDaemon] TRAINING_STARTED: no node_id, skipping")
                return

            logger.info(
                f"[AutoSyncDaemon] TRAINING_STARTED: {config_key} on {node_id} "
                "- triggering priority sync to training node"
            )

            self._events_processed += 1

            # Trigger priority sync to the training node
            try:
                success = await self._sync_to_peer(node_id)
                if success:
                    logger.info(
                        f"[AutoSyncDaemon] Priority sync to training node {node_id} complete"
                    )
            except (RuntimeError, OSError, ConnectionError) as sync_err:
                logger.warning(
                    f"[AutoSyncDaemon] Priority sync to {node_id} failed: {sync_err}"
                )

        except (RuntimeError, OSError, ConnectionError) as e:
            self._errors_count += 1
            self._last_error = str(e)
            logger.error(f"[AutoSyncDaemon] Error handling TRAINING_STARTED: {e}")

    async def _on_node_recovered(self, event) -> None:
        """Handle NODE_RECOVERED event - clear exclusion state for recovered nodes.

        December 2025: When a node recovers from being offline or unhealthy,
        we should clear any temporary exclusion state so it can participate
        in sync operations again.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            node_id = payload.get("node_id", "")

            if not node_id:
                logger.debug("[AutoSyncDaemon] NODE_RECOVERED: no node_id, skipping")
                return

            logger.info(
                f"[AutoSyncDaemon] NODE_RECOVERED: {node_id} "
                "- clearing exclusion state"
            )

            self._events_processed += 1

            # Clear any temporary exclusion for this node
            if hasattr(self, "_excluded_nodes") and node_id in self._excluded_nodes:
                self._excluded_nodes.discard(node_id)
                logger.info(f"[AutoSyncDaemon] Cleared exclusion for {node_id}")

            # Reset failure counter if we track per-node failures
            if hasattr(self, "_node_failure_counts") and node_id in self._node_failure_counts:
                self._node_failure_counts[node_id] = 0
                logger.debug(f"[AutoSyncDaemon] Reset failure count for {node_id}")

        except (RuntimeError, OSError, ConnectionError) as e:
            self._errors_count += 1
            self._last_error = str(e)
            logger.error(f"[AutoSyncDaemon] Error handling NODE_RECOVERED: {e}")

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
            with contextlib.suppress(asyncio.CancelledError):
                await self._sync_task

        # December 2025: Stop pending writes processor
        if self._pending_writes_task:
            self._pending_writes_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pending_writes_task

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

            # Find config - use canonical distributed_hosts.yaml (Dec 2025)
            base_dir = Path(__file__).resolve().parent.parent.parent
            config_path = base_dir / "config" / "distributed_hosts.yaml"
            data_dir = base_dir / "data" / "games"

            if not config_path.exists():
                logger.warning("No distributed_hosts.yaml found, gossip sync disabled")
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
            from app.coordination.event_router import emit_data_sync_failed
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
            from app.coordination.event_router import DataEventType, get_router

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
                        on_error=lambda exc: logger.debug(f"Failed to emit sync completed: {exc}"),
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
                    on_error=lambda exc: logger.debug(f"Failed to emit sync failed: {exc}"),
                )

            await asyncio.sleep(self.config.interval_seconds)

    async def _sync_cycle(self) -> int:
        """Execute one sync cycle.

        December 2025: Unified sync cycle supporting multiple strategies:
        - BROADCAST: Leader pushes to all eligible nodes
        - EPHEMERAL: Aggressive polling handled by on_game_complete()
        - HYBRID: Gossip-based replication (default)
        - PULL: Coordinator pulls data from cluster nodes (reverse sync)

        Returns:
            Number of games synced (0 if skipped or no data).
        """
        # December 2025: Use broadcast sync cycle for BROADCAST strategy
        if self._is_broadcast:
            return await self.broadcast_sync_cycle()

        # December 2025: Use pull sync cycle for PULL strategy (coordinator recovery)
        if self._resolved_strategy == SyncStrategy.PULL:
            return await self._pull_from_cluster_nodes()

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

        # December 2025 - Gap 4 fix: Verify synced databases after collection
        verification_passed = await self._verify_synced_databases()
        if not verification_passed:
            logger.warning("[AutoSyncDaemon] Some databases failed verification")
            # Continue anyway - partial data is better than no data

        # Register synced data to manifest
        await self._register_synced_data()

        return pending

    async def _pull_from_cluster_nodes(self) -> int:
        """Pull data FROM cluster nodes TO coordinator (reverse sync).

        December 2025: Implements PULL strategy for coordinator recovery.
        This is the inverse of normal sync - coordinator pulls data from
        generating nodes rather than pushing to receivers.

        Used for:
        - Coordinator data recovery after restart
        - Backfilling missing data from cluster
        - Consolidating distributed game databases

        Returns:
            Number of games pulled and validated.
        """
        # Only coordinators should use PULL strategy
        # Check env.is_coordinator from centralized config
        try:
            from app.config.env import env
            is_coordinator = env.is_coordinator
        except ImportError:
            # Fallback to checking hostname
            is_coordinator = "mac-studio" in socket.gethostname().lower() or "coordinator" in socket.gethostname().lower()

        if not is_coordinator:
            logger.debug("[AutoSyncDaemon] PULL strategy requires coordinator role")
            return 0

        # Get sync sources from SyncRouter
        try:
            from app.coordination.sync_router import get_sync_router, DataType

            sync_router = get_sync_router()
            if not sync_router:
                logger.warning("[AutoSyncDaemon] SyncRouter not available for PULL")
                return 0

            # Refresh node capabilities before routing
            sync_router.refresh_from_cluster_config()

            sources = sync_router.get_sync_sources(
                data_type=DataType.GAME,
                target_node=self.node_id,
                max_sources=5,  # Limit to top 5 sources per cycle
            )

            if not sources:
                logger.debug("[AutoSyncDaemon] No sync sources available for PULL")
                return 0

            logger.info(
                f"[AutoSyncDaemon] PULL: Found {len(sources)} sources: "
                f"{[s.node_id for s in sources]}"
            )

        except ImportError as e:
            logger.warning(f"[AutoSyncDaemon] SyncRouter import failed: {e}")
            return 0

        # Pull from each source
        total_pulled = 0
        for source in sources:
            try:
                pulled = await self._pull_from_node(source.node_id)
                total_pulled += pulled
                if pulled > 0:
                    logger.info(
                        f"[AutoSyncDaemon] Pulled {pulled} games from {source.node_id}"
                    )
            except Exception as e:
                logger.warning(
                    f"[AutoSyncDaemon] Failed to pull from {source.node_id}: {e}"
                )
                # Record failure for circuit breaker
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure(source.node_id)

        # Update stats
        self._stats.sync_cycles += 1
        self._stats.games_synced += total_pulled

        # Emit sync completion event
        if total_pulled > 0:
            await self._emit_pull_sync_completed(total_pulled, len(sources))

        return total_pulled

    async def _pull_from_node(self, source_node: str) -> int:
        """Pull game databases from a specific cluster node.

        Args:
            source_node: Node ID to pull from

        Returns:
            Number of games successfully pulled and validated.
        """
        # Get node SSH config
        try:
            from app.config.cluster_config import get_cluster_nodes

            nodes = get_cluster_nodes()
            node = nodes.get(source_node)
            if not node:
                logger.warning(f"[AutoSyncDaemon] Node {source_node} not found in config")
                return 0

            # Get SSH connection info
            ssh_host = node.best_ip
            ssh_user = node.ssh_user or "ubuntu"
            ssh_key = node.ssh_key or "~/.ssh/id_cluster"

            if not ssh_host:
                logger.warning(f"[AutoSyncDaemon] No SSH host for {source_node}")
                return 0

        except ImportError as e:
            logger.warning(f"[AutoSyncDaemon] cluster_config import failed: {e}")
            return 0

        import os
        ssh_key = os.path.expanduser(ssh_key)

        # List remote databases
        remote_games_path = self._get_remote_games_path(source_node)
        remote_dbs = await self._list_remote_databases(
            ssh_host, ssh_user, ssh_key, remote_games_path
        )

        if not remote_dbs:
            logger.debug(f"[AutoSyncDaemon] No databases found on {source_node}")
            return 0

        logger.debug(
            f"[AutoSyncDaemon] Found {len(remote_dbs)} databases on {source_node}"
        )

        # Prepare local temp directory for pulled databases
        base_dir = Path(__file__).resolve().parent.parent.parent
        pull_dir = base_dir / "data" / "games" / "pulled"
        pull_dir.mkdir(parents=True, exist_ok=True)

        games_pulled = 0
        for remote_db in remote_dbs:
            try:
                # Pull database
                local_path = await self._rsync_pull(
                    ssh_host, ssh_user, ssh_key,
                    remote_games_path, remote_db, pull_dir
                )

                if not local_path or not local_path.exists():
                    continue

                # Validate completeness
                is_valid, msg = self._validate_database_completeness(local_path)
                if not is_valid:
                    logger.warning(
                        f"[AutoSyncDaemon] Skipping {remote_db} from {source_node}: {msg}"
                    )
                    local_path.unlink(missing_ok=True)
                    continue

                # Check integrity
                is_intact, errors = check_sqlite_integrity(local_path)
                if not is_intact:
                    logger.warning(
                        f"[AutoSyncDaemon] {remote_db} from {source_node} failed integrity: {errors}"
                    )
                    local_path.unlink(missing_ok=True)
                    continue

                # Count games for stats (Dec 27, 2025: Use context manager to prevent leaks)
                try:
                    with sqlite3.connect(str(local_path), timeout=10.0) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT COUNT(*) FROM games")
                        game_count = cursor.fetchone()[0]
                        games_pulled += game_count
                except sqlite3.Error:
                    pass

                # Merge into canonical database
                await self._merge_into_canonical(local_path, source_node)

                # Clean up pulled file after merge
                local_path.unlink(missing_ok=True)

            except Exception as e:
                logger.warning(
                    f"[AutoSyncDaemon] Error pulling {remote_db} from {source_node}: {e}"
                )

        return games_pulled

    async def _list_remote_databases(
        self,
        ssh_host: str,
        ssh_user: str,
        ssh_key: str,
        remote_path: str,
    ) -> list[str]:
        """List database files on a remote node.

        Returns:
            List of database filenames (not full paths).
        """
        cmd = [
            "ssh",
            "-i", ssh_key,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            f"{ssh_user}@{ssh_host}",
            f"ls -1 {remote_path}/*.db 2>/dev/null || echo ''"
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30.0)

            output = stdout.decode().strip()
            if not output:
                return []

            # Extract just filenames
            dbs = []
            for line in output.split("\n"):
                line = line.strip()
                if line.endswith(".db"):
                    dbs.append(Path(line).name)

            return dbs

        except asyncio.TimeoutError:
            logger.warning(f"[AutoSyncDaemon] Timeout listing databases on {ssh_host}")
            return []
        except Exception as e:
            logger.debug(f"[AutoSyncDaemon] Error listing remote dbs: {e}")
            return []

    async def _rsync_pull(
        self,
        ssh_host: str,
        ssh_user: str,
        ssh_key: str,
        remote_path: str,
        db_name: str,
        local_dir: Path,
    ) -> Path | None:
        """Pull a single database file from a remote node.

        Returns:
            Local path to the pulled file, or None if failed.
        """
        local_path = local_dir / db_name
        remote_full = f"{ssh_user}@{ssh_host}:{remote_path}/{db_name}"

        cmd = [
            "rsync",
            "-az",
            "--timeout=60",
            "-e", f"ssh -i {ssh_key} -o StrictHostKeyChecking=no -o ConnectTimeout=10",
            remote_full,
            str(local_path),
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120.0)

            if proc.returncode == 0 and local_path.exists():
                return local_path
            else:
                if stderr:
                    logger.debug(f"[AutoSyncDaemon] Rsync pull error: {stderr.decode()}")
                return None

        except asyncio.TimeoutError:
            logger.warning(f"[AutoSyncDaemon] Rsync pull timeout for {db_name}")
            return None
        except Exception as e:
            logger.debug(f"[AutoSyncDaemon] Rsync pull error: {e}")
            return None

    async def _merge_into_canonical(self, pulled_db: Path, source_node: str) -> None:
        """Merge a pulled database into the appropriate canonical database.

        Uses ATTACH DATABASE to copy games that don't already exist locally.

        Args:
            pulled_db: Path to the pulled database file
            source_node: Source node for logging
        """
        # Determine canonical database from pulled db name
        # e.g., "selfplay_hex8_2p.db" -> "canonical_hex8_2p.db"
        db_name = pulled_db.name
        canonical_name = self._get_canonical_name(db_name)

        base_dir = Path(__file__).resolve().parent.parent.parent
        canonical_path = base_dir / "data" / "games" / canonical_name

        # If no canonical exists, just rename the pulled file
        if not canonical_path.exists():
            pulled_db.rename(canonical_path)
            logger.info(
                f"[AutoSyncDaemon] Created canonical {canonical_name} from {source_node}"
            )
            return

        # Merge games from pulled into canonical (Dec 27, 2025: Use context manager)
        conn = None
        try:
            conn = sqlite3.connect(str(canonical_path), timeout=30.0)
            cursor = conn.cursor()

            # Attach pulled database
            cursor.execute(f"ATTACH DATABASE ? AS pulled", (str(pulled_db),))

            # Get count before merge
            cursor.execute("SELECT COUNT(*) FROM games")
            before_count = cursor.fetchone()[0]

            # Copy games that don't exist
            cursor.execute("""
                INSERT OR IGNORE INTO games
                SELECT * FROM pulled.games
                WHERE game_id NOT IN (SELECT game_id FROM games)
            """)

            # Copy moves for new games
            cursor.execute("""
                INSERT OR IGNORE INTO game_moves
                SELECT * FROM pulled.game_moves
                WHERE game_id NOT IN (SELECT DISTINCT game_id FROM game_moves)
            """)

            conn.commit()

            # Get count after merge
            cursor.execute("SELECT COUNT(*) FROM games")
            after_count = cursor.fetchone()[0]

            try:
                cursor.execute("DETACH DATABASE pulled")
            except sqlite3.Error:
                pass  # May already be detached

            new_games = after_count - before_count
            if new_games > 0:
                logger.info(
                    f"[AutoSyncDaemon] Merged {new_games} games from {source_node} "
                    f"into {canonical_name}"
                )

        except sqlite3.Error as e:
            logger.warning(f"[AutoSyncDaemon] Merge failed for {db_name}: {e}")
        finally:
            if conn:
                conn.close()

    def _get_canonical_name(self, db_name: str) -> str:
        """Convert a database name to its canonical form.

        Examples:
            selfplay_hex8_2p.db -> canonical_hex8_2p.db
            games_square8_4p.db -> canonical_square8_4p.db
            hex8_2p_selfplay.db -> canonical_hex8_2p.db
        """
        # Extract board type and player count from name
        import re

        # Try common patterns
        patterns = [
            r"(hex8|square8|square19|hexagonal)_(\d)p",  # hex8_2p
            r"(\d)p_(hex8|square8|square19|hexagonal)",  # 2p_hex8
        ]

        for pattern in patterns:
            match = re.search(pattern, db_name)
            if match:
                groups = match.groups()
                if groups[0].isdigit():
                    # Pattern 2: 2p_hex8
                    board_type = groups[1]
                    num_players = groups[0]
                else:
                    # Pattern 1: hex8_2p
                    board_type = groups[0]
                    num_players = groups[1]
                return f"canonical_{board_type}_{num_players}p.db"

        # Fallback: just prefix with canonical_
        if db_name.startswith("canonical_"):
            return db_name
        return f"canonical_{db_name}"

    def _get_remote_games_path(self, node_id: str) -> str:
        """Get the remote games directory path for a node.

        Different providers have different paths.
        """
        # Check provider from config
        try:
            from app.config.cluster_config import get_host_provider

            provider = get_host_provider(node_id)
        except ImportError:
            provider = None

        # Common paths by provider
        if provider == "runpod":
            return "/workspace/ringrift/ai-service/data/games"
        elif provider == "vast":
            return "~/ringrift/ai-service/data/games"
        elif provider == "nebius":
            return "~/ringrift/ai-service/data/games"
        elif provider == "vultr":
            return "/root/ringrift/ai-service/data/games"
        else:
            return "~/ringrift/ai-service/data/games"

    async def _emit_pull_sync_completed(self, games_pulled: int, sources_count: int) -> None:
        """Emit event when PULL sync completes successfully."""
        try:
            from app.coordination.data_events import (
                DataEventType,
                emit_data_event,
            )

            await emit_data_event(
                DataEventType.DATA_SYNC_COMPLETED,
                {
                    "sync_type": "pull",
                    "games_synced": games_pulled,
                    "sources_count": sources_count,
                    "node_id": self.node_id,
                    "timestamp": time.time(),
                },
            )
        except ImportError:
            pass  # Events not available

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

        except (RuntimeError, OSError, KeyError, ValueError) as e:
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
            from app.coordination.event_router import emit_quality_score_updated

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

        except (RuntimeError, ImportError, AttributeError) as e:
            logger.warning(f"Failed to update priority queue for {config_key}: {e}")

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

            # Dec 2025: Use context manager to ensure connection is closed
            with sqlite3.connect(str(db_path), timeout=5) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT game_id, game_status, winner, termination_reason,
                           total_moves, board_type
                    FROM games
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (self.config.quality_sample_size,))
                games = cursor.fetchall()

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
                logger.warning(f"Failed to register games from {db_path}: {e}")

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
                # Dec 2025: Use context manager to ensure connection is closed
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM games")
                    total_games += cursor.fetchone()[0]
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

    async def _verify_synced_databases(self) -> bool:
        """Verify integrity of synced databases (December 2025 - Gap 4 fix).

        Runs SQLite PRAGMA integrity_check on all recently synced databases
        to detect corruption from incomplete transfers, network errors, etc.

        Returns:
            True if all databases pass verification, False if any fail.
        """
        base_dir = Path(__file__).resolve().parent.parent.parent
        data_dir = base_dir / "data" / "games"

        if not data_dir.exists():
            return True

        start_time = time.time()
        verified_count = 0
        failed_count = 0
        failed_dbs: list[str] = []

        for db_path in data_dir.glob("*.db"):
            # Skip manifest and schema files
            if "manifest" in db_path.name or "schema" in db_path.name:
                continue

            # Skip WAL files (only check main database)
            if db_path.suffix in ["-wal", "-shm"]:
                continue

            try:
                is_valid, errors = check_sqlite_integrity(db_path)

                if is_valid:
                    verified_count += 1
                else:
                    failed_count += 1
                    failed_dbs.append(db_path.name)
                    logger.error(
                        f"[AutoSyncDaemon] Database {db_path.name} failed integrity check: {errors}"
                    )
                    # Emit verification failed event
                    fire_and_forget(
                        self._emit_sync_verification_failed(
                            db_path.name,
                            f"Integrity check failed: {errors[:2]}",
                        )
                    )

            except (OSError, sqlite3.Error) as e:
                # Database may be locked or in use - not necessarily corrupted
                logger.debug(f"[AutoSyncDaemon] Could not verify {db_path.name}: {e}")

        # Update stats
        self._stats.databases_verified += verified_count
        self._stats.databases_verification_failed += failed_count
        self._stats.last_verification_time = time.time()

        elapsed = time.time() - start_time
        if verified_count > 0 or failed_count > 0:
            logger.info(
                f"[AutoSyncDaemon] Verification complete: {verified_count} passed, "
                f"{failed_count} failed ({elapsed:.2f}s)"
            )

        return failed_count == 0

    async def _emit_sync_verification_failed(self, db_name: str, error: str) -> None:
        """Emit SYNC_VERIFICATION_FAILED event for feedback loop."""
        try:
            from app.coordination.event_router import DataEventType, get_router

            router = get_router()
            if router:
                await router.publish(
                    event_type=DataEventType.DATA_SYNC_FAILED,
                    payload={
                        "node_id": self.node_id,
                        "db_name": db_name,
                        "error": error,
                        "verification_failed": True,
                        "total_verified": self._stats.databases_verified,
                        "total_failed": self._stats.databases_verification_failed,
                    },
                    source="AutoSyncDaemon:verification",
                )
        except (RuntimeError, OSError, ConnectionError) as e:
            logger.debug(f"Could not emit SYNC_VERIFICATION_FAILED: {e}")

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
                # December 2025 - Gap 4 fix: Verification stats
                "databases_verified": self._stats.databases_verified,
                "databases_verification_failed": self._stats.databases_verification_failed,
                "last_verification_time": self._stats.last_verification_time,
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
            # December 2025 - Gap 4 fix: Verification metrics
            "databases_verified": self._stats.databases_verified,
            "databases_verification_failed": self._stats.databases_verification_failed,
            "last_verification_time": self._stats.last_verification_time,
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

        # December 2025 - Gap 4 fix: Check verification health
        if self._stats.databases_verified > 0:
            verification_failure_rate = (
                self._stats.databases_verification_failed /
                max(self._stats.databases_verified + self._stats.databases_verification_failed, 1)
            )
            if verification_failure_rate > 0.1:  # More than 10% failure rate
                return HealthCheckResult.degraded(
                    f"High verification failure rate: {self._stats.databases_verification_failed} failed, "
                    f"{self._stats.databases_verified} passed ({verification_failure_rate*100:.1f}%)",
                    verification_failure_rate=verification_failure_rate,
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
                # December 2025 - Gap 4 fix: Verification stats
                "databases_verified": self._stats.databases_verified,
                "databases_verification_failed": self._stats.databases_verification_failed,
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


def create_ephemeral_sync_daemon(
    on_termination: Any = None,
) -> AutoSyncDaemon:
    """Factory function for ephemeral sync (backward compatibility).

    December 2025: Creates an AutoSyncDaemon with EPHEMERAL strategy.
    This replaces the standalone EphemeralSyncDaemon.

    Args:
        on_termination: Optional callback for termination handling (ignored)

    Returns:
        AutoSyncDaemon configured for ephemeral mode
    """
    import warnings
    warnings.warn(
        "create_ephemeral_sync_daemon() is deprecated. "
        "Use AutoSyncDaemon(config=AutoSyncConfig(strategy='ephemeral')) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    config = AutoSyncConfig.from_config_file()
    config.strategy = SyncStrategy.EPHEMERAL
    return AutoSyncDaemon(config=config)


def create_cluster_data_sync_daemon() -> AutoSyncDaemon:
    """Factory function for cluster data sync (backward compatibility).

    December 2025: Creates an AutoSyncDaemon with BROADCAST strategy.
    This replaces the standalone ClusterDataSyncDaemon.

    Returns:
        AutoSyncDaemon configured for broadcast mode
    """
    import warnings
    warnings.warn(
        "create_cluster_data_sync_daemon() is deprecated. "
        "Use AutoSyncDaemon(config=AutoSyncConfig(strategy='broadcast')) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    config = AutoSyncConfig.from_config_file()
    config.strategy = SyncStrategy.BROADCAST
    return AutoSyncDaemon(config=config)


def create_training_sync_daemon() -> AutoSyncDaemon:
    """Factory function for training node sync daemon.

    December 2025: Creates an AutoSyncDaemon configured for training node
    synchronization with BROADCAST strategy and reduced sync interval (30s)
    to ensure training nodes have fresh data.

    Returns:
        AutoSyncDaemon configured for training sync mode
    """
    config = AutoSyncConfig.from_config_file()
    config.strategy = SyncStrategy.BROADCAST
    config.sync_interval = 30.0  # Faster sync for training freshness
    return AutoSyncDaemon(config=config)


# Backward compatibility aliases (December 2025)
# These will be removed in Q2 2026
EphemeralSyncDaemon = AutoSyncDaemon  # Deprecated alias
ClusterDataSyncDaemon = AutoSyncDaemon  # Deprecated alias


def get_ephemeral_sync_daemon() -> AutoSyncDaemon:
    """Get ephemeral sync daemon (backward compatibility).

    December 2025: Returns AutoSyncDaemon with EPHEMERAL strategy.
    """
    import warnings
    warnings.warn(
        "get_ephemeral_sync_daemon() is deprecated. "
        "Use get_auto_sync_daemon() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_auto_sync_daemon()


def get_cluster_data_sync_daemon() -> AutoSyncDaemon:
    """Get cluster data sync daemon (backward compatibility).

    December 2025: Returns AutoSyncDaemon with BROADCAST strategy if leader.
    """
    import warnings
    warnings.warn(
        "get_cluster_data_sync_daemon() is deprecated. "
        "Use get_auto_sync_daemon() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_auto_sync_daemon()


__all__ = [  # noqa: RUF022
    # Core classes
    "AutoSyncConfig",
    "AutoSyncDaemon",
    "SyncStats",
    "SyncStrategy",
    # Singleton accessors
    "get_auto_sync_daemon",
    "reset_auto_sync_daemon",
    # Factory functions (December 2025 consolidation)
    "create_ephemeral_sync_daemon",
    "create_cluster_data_sync_daemon",
    # Utility functions (December 2025 consolidation from ephemeral_sync.py)
    "is_ephemeral_host",
    # Backward compatibility (deprecated)
    "get_ephemeral_sync_daemon",
    "get_cluster_data_sync_daemon",
    "EphemeralSyncDaemon",
    "ClusterDataSyncDaemon",
]


def is_ephemeral_host() -> bool:
    """Check if current host is ephemeral.

    December 2025: Consolidated from ephemeral_sync.py
    Convenience function for checking ephemeral host status.
    """
    daemon = get_auto_sync_daemon()
    return getattr(daemon, "_is_ephemeral", False)

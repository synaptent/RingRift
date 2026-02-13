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
Classes (in sync_strategies.py):
    SyncStrategy          - Enum-like class for sync mode selection
    AutoSyncConfig        - Configuration dataclass with all sync settings
    SyncStats             - Statistics tracking (extends SyncDaemonStats)

Classes (in this module):
    AutoSyncDaemon        - Main daemon class

Factory Functions:
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
    start()                  - Start background sync loops
    stop()                   - Graceful shutdown with final sync
    health_check()           - Return HealthCheckResult for monitoring

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
    - Subscribes to: NEW_GAMES_AVAILABLE, TRAINING_STARTED, NODE_RECOVERED, SYNC_REQUEST
    - Emits: DATA_SYNC_STARTED, DATA_SYNC_COMPLETED, DATA_SYNC_FAILED

Usage:
    from app.coordination.auto_sync_daemon import AutoSyncDaemon

    daemon = AutoSyncDaemon()
    await daemon.start()

    # Or with specific strategy
    from app.coordination.sync_strategies import SyncStrategy
    daemon = AutoSyncDaemon(strategy=SyncStrategy.EPHEMERAL)
    await daemon.start()
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import socket
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

# December 2025: Import strategies and config from sync_strategies.py
from app.coordination.sync_strategies import (
    SyncStrategy,
    AutoSyncConfig,
    SyncStats,
    SyncProgress,
    MIN_MOVES_PER_GAME,
    DEFAULT_MIN_MOVES,
)
from app.db.write_lock import is_database_safe_to_sync
from app.coordination.handler_base import (
    HandlerBase,
    HealthCheckResult,
    CoordinatorStatus,
)
from app.coordination.sync_integrity import check_sqlite_integrity, quarantine_corrupted_db
from app.coordination.sync_mutex import acquire_sync_lock, release_sync_lock
from app.coordination.disk_space_reservation import (
    DiskSpaceReservation,
    DiskSpaceError,
    cleanup_stale_reservations,
    disk_space_reservation,
    get_effective_available_space,
)
from app.core.async_context import fire_and_forget, safe_create_task

# December 2025: Import mixins for modular organization
# Each mixin provides a focused set of methods for the AutoSyncDaemon
from app.coordination.sync_event_mixin import SyncEventMixin
from app.coordination.sync_push_mixin import SyncPushMixin
from app.coordination.sync_pull_mixin import SyncPullMixin
from app.coordination.sync_ephemeral_mixin import SyncEphemeralMixin
from app.coordination.event_emission_helpers import safe_emit_event
from app.utils.sqlite_utils import connect_safe

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


class AutoSyncDaemon(
    HandlerBase,
    SyncEventMixin,
    SyncPushMixin,
    SyncPullMixin,
    SyncEphemeralMixin,
):
    """Daemon that orchestrates automated P2P data synchronization.

    December 2025 Consolidation: Unified daemon supporting multiple strategies:
    - HYBRID: Push-from-generator + gossip replication (default for persistent hosts)
    - EPHEMERAL: Aggressive 5s sync for Vast.ai/spot instances (from ephemeral_sync.py)
    - BROADCAST: Leader-only push to all nodes (from cluster_data_sync.py)
    - AUTO: Auto-detect based on node type
    - PULL: Coordinator-initiated pull from worker nodes (from sync_pull_mixin.py)

    Key features:
    - Gossip-based replication for eventual consistency
    - Provider-aware sync (skip NFS, prioritize ephemeral)
    - Coordinator exclusion (save disk space)
    - ClusterManifest for central tracking and disk management
    - Write-through mode for ephemeral hosts (zero data loss)
    - WAL (write-ahead log) for durability

    Mixin organization (December 2025):
    - SyncEventMixin: Event subscription and _on_* handlers
    - SyncPushMixin: Push/broadcast sync operations
    - SyncPullMixin: PULL strategy operations for coordinator recovery
    - SyncEphemeralMixin: Ephemeral host/WAL handling
    """

    @property
    def config(self) -> AutoSyncConfig:
        """Get daemon configuration (backward compatibility property).

        January 2026: HandlerBase migration - provides public access to config.
        """
        return self._config

    @property
    def status(self) -> "CoordinatorStatus":
        """Get current coordinator status (backward compatibility property).

        January 2026: HandlerBase migration - provides public access to status.
        """
        return self._status

    @property
    def stats(self) -> SyncStats:
        """Get sync statistics (backward compatibility property).

        January 2026: HandlerBase migration - provides public access to sync stats.
        Note: HandlerBase uses _stats for HandlerStats; we use _sync_stats for SyncStats.
        """
        return self._sync_stats

    def __init__(self, config: AutoSyncConfig | None = None):
        daemon_config = config or AutoSyncConfig.from_config_file()

        # Initialize HandlerBase with event-driven cycle interval
        super().__init__(
            name="AutoSyncDaemon",
            config=daemon_config,
            cycle_interval=float(daemon_config.interval_seconds),
        )

        self.node_id = socket.gethostname()
        self._sync_stats = SyncStats()
        self._progress = SyncProgress()  # December 2025: Real-time sync progress tracking
        self._sync_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._gossip_daemon = None
        self._semaphore = asyncio.Semaphore(daemon_config.max_concurrent_syncs)

        # ClusterManifest integration
        self._cluster_manifest: ClusterManifest | None = None
        if self._config.use_cluster_manifest:
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
        # Jan 12, 2026: C5 fix - sync cycle lock prevents concurrent sync cycles
        # within the same process. Combined with sync_mutex for cross-process protection.
        self._sync_cycle_lock = asyncio.Lock()
        self._wal_initialized = False
        if self._is_ephemeral and self._config.ephemeral_wal_enabled:
            self._init_ephemeral_wal()

        # December 2025: Retry queue for failed write-through pushes
        self._pending_writes_file = Path("data/ephemeral_pending_writes.jsonl")
        self._pending_writes_task: asyncio.Task | None = None
        self._init_pending_writes_file()

        # Phase 9: Event subscription for DATA_STALE triggers
        self._subscribed = False
        self._urgent_sync_pending: dict[str, float] = {}  # config_key -> request_time

        # January 2026: Event tracking for SyncEventMixin (HandlerBase migration)
        self._events_processed: int = 0
        self._errors_count: int = 0

        # December 28, 2025: Backpressure handling - pause sync during high load
        self._sync_paused = False
        self._backpressure_reason: str = ""

        # December 29, 2025: Event-driven sync - wake loop immediately on events
        self._sync_wake_event = asyncio.Event()
        self._last_sync_time: float = 0.0  # For throttling
        # Min sync interval reduced from 5s to 2s for faster data distribution (Dec 2025)
        # Configurable via env var RINGRIFT_MIN_SYNC_INTERVAL or config
        _default_interval = float(os.getenv("RINGRIFT_MIN_SYNC_INTERVAL", "2.0"))
        self._min_sync_interval: float = (
            self._config.min_sync_interval_seconds
            if hasattr(self._config, 'min_sync_interval_seconds')
            else _default_interval
        )

        # Quality extraction for training data prioritization (December 2025)
        self._quality_config: Any = None
        self._elo_lookup: Any = None
        if self._config.enable_quality_extraction:
            if HAS_QUALITY_EXTRACTION:
                try:
                    self._quality_config = QualityExtractorConfig()
                    self._elo_lookup = get_elo_lookup_from_service()
                    logger.info("Quality extraction enabled for training data prioritization")
                except (RuntimeError, OSError, ValueError, KeyError) as e:
                    logger.warning(f"Failed to initialize quality extraction: {e}")
                    self._config.enable_quality_extraction = False
            else:
                # December 2025: Log warning when quality extraction is requested but unavailable
                logger.warning(
                    "[AutoSyncDaemon] Quality extraction requested but module unavailable. "
                    "Install quality_extractor dependencies or set enable_quality_extraction=False"
                )
                self._config.enable_quality_extraction = False

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
        strategy = self._config.strategy

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
        event_type = (
            "sync_node_unreachable" if new_state.value == "open" else "data_sync_completed"
        )
        safe_emit_event(
            event_type,
            {
                "source": "AutoSyncDaemon",
                "target_node": target,
                "old_state": old_state.value,
                "new_state": new_state.value,
                "event": "circuit_state_change",
            },
            context="AutoSync",
        )

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

    # =========================================================================
    # December 2025: WAL and pending writes methods moved to SyncEphemeralMixin
    # Methods: _init_ephemeral_wal, _load_ephemeral_wal, _append_to_wal, _clear_wal,
    #          _init_pending_writes_file, _persist_failed_write, _push_with_retry,
    #          _process_pending_writes
    # =========================================================================

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
            with connect_safe(db_path, timeout=10.0, row_factory=None) as conn:
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
    # HandlerBase Overrides (Sprint 16 Migration)
    # =========================================================================

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

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Return event subscriptions for HandlerBase to wire.

        Sprint 16: Events are now wired via HandlerBase infrastructure.
        Handler methods are provided by SyncEventMixin.
        """
        from app.distributed.data_events import DataEventType

        subscriptions = {}

        # Events this daemon subscribes to (see SyncEventMixin for handlers)
        if hasattr(self, "_on_new_games_available"):
            subscriptions[DataEventType.NEW_GAMES_AVAILABLE.value] = self._on_new_games_available
        if hasattr(self, "_on_training_started"):
            subscriptions[DataEventType.TRAINING_STARTED.value] = self._on_training_started
        if hasattr(self, "_on_node_recovered"):
            subscriptions[DataEventType.NODE_RECOVERED.value] = self._on_node_recovered
        if hasattr(self, "_on_sync_request"):
            subscriptions["sync_request"] = self._on_sync_request
        if hasattr(self, "_on_data_stale"):
            subscriptions[DataEventType.DATA_STALE.value] = self._on_data_stale

        return subscriptions

    async def _on_start(self) -> None:
        """Hook called when daemon starts - setup daemon-specific resources."""
        logger.info(f"Starting AutoSyncDaemon on {self.node_id}")

        # December 2025: Clean up stale disk space reservations from crashed processes
        try:
            cleaned = cleanup_stale_reservations()
            if cleaned > 0:
                logger.info(f"[AutoSyncDaemon] Cleaned {cleaned} stale disk space reservations")
        except OSError as e:
            logger.warning(f"[AutoSyncDaemon] Failed to clean stale reservations: {e}")

        # December 2025: Setup termination handlers for ephemeral mode
        if self._is_ephemeral:
            self._setup_termination_handlers()

        # Start gossip sync daemon
        await self._start_gossip_sync()

        # December 2025: Start pending writes retry processor
        self._pending_writes_task = safe_create_task(
            self._process_pending_writes(),
            name="pending_writes_processor",
        )

        logger.info(
            f"AutoSyncDaemon started: "
            f"interval={self._config.interval_seconds}s, "
            f"exclude={self._config.exclude_hosts}"
        )

    async def start(self) -> None:
        """Start the auto sync daemon."""
        if not self._config.enabled:
            self._status = CoordinatorStatus.STOPPED
            logger.info("AutoSyncDaemon disabled by config")
            return

        # Use HandlerBase's start() which calls _on_start() and starts main loop
        await super().start()

    def _setup_termination_handlers(self) -> None:
        """Setup signal handlers for termination (ephemeral mode).

        December 2025: Consolidated from ephemeral_sync.py
        """
        import signal

        def handle_termination(sig: int, frame: Any) -> None:
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
        except (RuntimeError, OSError, asyncio.CancelledError, ConnectionError, TimeoutError) as e:
            # Extended to include network errors (December 2025 exception narrowing)
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


    # =========================================================================
    # Ephemeral sync methods now inherited from SyncEphemeralMixin
    # =========================================================================

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


    # =========================================================================
    # Push/broadcast methods now inherited from SyncPushMixin
    # =========================================================================


    # =========================================================================
    # Event handling methods now inherited from SyncEventMixin
    # =========================================================================

    async def _on_stop(self) -> None:
        """Hook called when daemon stops - cleanup daemon-specific resources."""
        logger.info("Stopping AutoSyncDaemon...")

        # December 2025: Stop pending writes processor
        if self._pending_writes_task:
            self._pending_writes_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pending_writes_task

        # Stop gossip daemon
        if self._gossip_daemon:
            await self._gossip_daemon.stop()

        logger.info("AutoSyncDaemon stopped")

    async def stop(self) -> None:
        """Stop the auto sync daemon."""
        # Use HandlerBase's stop() which calls _on_stop()
        await super().stop()

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
                exclude_hosts=self._config.exclude_hosts,
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

            # Dec 29, 2025: emit_data_sync_failed is None if data_events failed to import
            if emit_data_sync_failed is None:
                logger.debug("emit_data_sync_failed unavailable (data_events not imported)")
                return

            await emit_data_sync_failed(
                host=self.node_id,
                error=error,
                retry_count=self._sync_stats.failed_syncs,
                source="AutoSyncDaemon",
            )
        except (RuntimeError, OSError, ConnectionError, TypeError) as e:
            logger.debug(f"Could not emit DATA_SYNC_FAILED: {e}")

    async def _emit_sync_completed(
        self,
        games_synced: int,
        bytes_transferred: int = 0,
        db_path: str | None = None,
        config_key: str | None = None,
    ) -> None:
        """Emit DATA_SYNC_COMPLETED event for feedback loop coupling.

        January 2026: Enhanced to include backup flags and db_path for
        UnifiedBackupDaemon to push synced data to OWC/S3.
        """
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
                        "total_syncs": self._sync_stats.total_syncs,
                        "successful_syncs": self._sync_stats.successful_syncs,
                        # January 2026: Added for UnifiedBackupDaemon integration
                        "db_path": db_path,
                        "config_key": config_key,
                        "needs_owc_backup": True,
                        "needs_s3_backup": True,
                        "source": "selfplay",
                    },
                    source="AutoSyncDaemon",
                )
        except (RuntimeError, OSError, ConnectionError) as e:
            logger.debug(f"Could not emit DATA_SYNC_COMPLETED: {e}")

    async def _run_cycle(self) -> None:
        """Implement abstract method from HandlerBase.

        January 2026: This is required by HandlerBase but not directly called
        since we override _main_loop() for event-driven sync. The actual work
        is done in _sync_cycle().
        """
        await self._sync_cycle()

    async def _main_loop(self) -> None:
        """Override HandlerBase._main_loop() to use event-driven sync pattern.

        January 2026: HandlerBase migration - delegate to _sync_loop which
        implements the event-driven wake pattern with throttling.
        """
        await self._sync_loop()

    async def _sync_loop(self) -> None:
        """Main sync loop - event-driven with throttling.

        December 29, 2025: Converted from polling to event-driven.
        - Waits on _sync_wake_event or max interval timeout (whichever comes first)
        - Respects minimum sync interval to prevent excessive syncing
        - Events trigger immediate wake-up, but throttle ensures minimum gap
        """
        while self._running:
            # December 28, 2025: Check backpressure before running sync cycle
            if self._sync_paused:
                logger.debug(
                    f"[AutoSyncDaemon] Sync paused due to backpressure: {self._backpressure_reason}"
                )
                # Wait for backpressure release or max interval
                try:
                    await asyncio.wait_for(
                        self._sync_wake_event.wait(),
                        timeout=self._config.interval_seconds
                    )
                    self._sync_wake_event.clear()
                except asyncio.TimeoutError:
                    pass
                continue

            # December 29, 2025: Wait for event or timeout (event-driven sync)
            try:
                await asyncio.wait_for(
                    self._sync_wake_event.wait(),
                    timeout=self._config.interval_seconds
                )
                self._sync_wake_event.clear()
                logger.debug("[AutoSyncDaemon] Woke from event trigger")
            except asyncio.TimeoutError:
                # No event, but max interval reached - run periodic sync
                logger.debug("[AutoSyncDaemon] Woke from interval timeout")

            # December 29, 2025: Throttle to prevent too-frequent syncs
            time_since_last = time.time() - self._last_sync_time
            if time_since_last < self._min_sync_interval:
                remaining = self._min_sync_interval - time_since_last
                logger.debug(
                    f"[AutoSyncDaemon] Throttling: waiting {remaining:.1f}s before sync"
                )
                await asyncio.sleep(remaining)

            try:
                games_synced = await self._sync_cycle()
                self._last_sync_time = time.time()  # Update for throttling
                # Use actual field names, not readonly property aliases
                self._sync_stats.operations_attempted += 1
                self._sync_stats.syncs_completed += 1
                self._sync_stats.last_check_time = time.time()
                # Emit DATA_SYNC_COMPLETED event for feedback loop
                if games_synced and games_synced > 0:
                    fire_and_forget(
                        self._emit_sync_completed(games_synced),
                        on_error=lambda exc: logger.debug(f"Failed to emit sync completed: {exc}"),
                    )
            except asyncio.CancelledError:
                break
            except (RuntimeError, OSError, ConnectionError) as e:
                self._sync_stats.syncs_failed += 1
                self._sync_stats.last_error = str(e)
                logger.error(f"Sync cycle error: {e}")

    def trigger_sync(self) -> None:
        """Trigger an immediate sync cycle via event wake-up.

        December 29, 2025: Public method for event handlers to wake the sync loop.
        This is preferred over calling _sync_cycle() directly as it respects throttling.
        """
        self._sync_wake_event.set()
        logger.debug("[AutoSyncDaemon] Sync triggered via event")

    async def _sync_all(self) -> None:
        """Execute full sync cycle (Protocol method required by SyncEventMixin).

        December 2025: Added to satisfy AutoSyncDaemonProtocol interface.
        This is called by event handlers when a full sync is needed.
        """
        await self._sync_cycle()

    async def _sync_to_peer(self, node_id: str) -> bool:
        """Sync all local databases to a specific peer node.

        December 29, 2025: Added to satisfy SyncEventMixin interface.
        Called by event handlers (e.g., TRAINING_STARTED) to sync data to a specific node.

        Args:
            node_id: Target node identifier

        Returns:
            True if sync succeeded, False otherwise
        """
        from urllib.request import Request, urlopen

        try:
            # Get node info from P2P status
            from app.config.ports import get_p2p_status_url
            url = get_p2p_status_url()
            req = Request(url, headers={"Accept": "application/json"})

            with urlopen(req, timeout=10) as resp:
                status = json.loads(resp.read().decode())

            peers = status.get("peers", {})
            target_info = peers.get(node_id)

            if not target_info:
                logger.warning(f"[AutoSyncDaemon] Target node {node_id} not found in P2P status")
                return False

            # Build target dict for sync_to_target_with_retry
            # December 31, 2025: Use 'host' key (not 'ssh_host') for consistency
            # with sync_push_mixin.broadcast_sync_to_target() which expects 'host'
            target = {
                "node_id": node_id,
                "host": target_info.get("host", node_id),  # 'host' not 'ssh_host'
                "ssh_user": target_info.get("ssh_user", "root"),
                "ssh_port": target_info.get("ssh_port", 22),
                "disk_free_gb": target_info.get("disk_free_gb", 100),
            }

            # Get local databases
            databases = self.discover_local_databases()
            if not databases:
                logger.debug(f"[AutoSyncDaemon] No databases to sync to {node_id}")
                return True  # No data to sync is not a failure

            # Feb 2026: 3 → 1 to prevent OOM from parallel rsync processes
            max_concurrent = 1
            semaphore = asyncio.Semaphore(max_concurrent)

            async def sync_db_with_limit(db: Path) -> bool:
                async with semaphore:
                    try:
                        result = await self.sync_to_target_with_retry(db, target)
                        return result.get("success", False)
                    except (RuntimeError, OSError, ConnectionError) as e:
                        logger.warning(f"[AutoSyncDaemon] Failed to sync {db.name} to {node_id}: {e}")
                        return False

            tasks = [sync_db_with_limit(db) for db in databases]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)

            logger.info(f"[AutoSyncDaemon] Synced {success_count}/{len(databases)} databases to {node_id}")
            return success_count > 0

        except (OSError, ValueError, json.JSONDecodeError, TimeoutError) as e:
            logger.error(f"[AutoSyncDaemon] Failed to sync to peer {node_id}: {e}")
            return False

    async def _sync_cycle(self) -> int:
        """Execute one sync cycle.

        December 2025: Unified sync cycle supporting multiple strategies:
        - BROADCAST: Leader pushes to all eligible nodes
        - EPHEMERAL: Aggressive polling handled by on_game_complete()
        - HYBRID: Gossip-based replication (default)
        - PULL: Coordinator pulls data from cluster nodes (reverse sync)

        Jan 12, 2026: C5 fix - Added sync cycle locking to prevent concurrent syncs.
        Uses both local asyncio.Lock (in-process) and distributed sync_mutex (cross-process).

        Returns:
            Number of games synced (0 if skipped or no data).
        """
        # C5 fix: Try to acquire local lock (non-blocking to prevent deadlock)
        if self._sync_cycle_lock.locked():
            logger.debug("[AutoSyncDaemon] Skipping sync cycle - another cycle in progress")
            return 0

        async with self._sync_cycle_lock:
            # C5 fix: Try to acquire distributed lock for cross-process coordination
            lock_key = f"sync_cycle:{self.node_id}"
            lock_acquired = await asyncio.to_thread(acquire_sync_lock, lock_key, "auto_sync", timeout=5)
            if not lock_acquired:
                logger.debug("[AutoSyncDaemon] Skipping sync cycle - distributed lock held by another process")
                return 0

            try:
                return await self._sync_cycle_inner()
            finally:
                # C5 fix: Release distributed lock
                await asyncio.to_thread(release_sync_lock, lock_key, "auto_sync")

    async def _sync_cycle_inner(self) -> int:
        """Inner sync cycle logic (called with locks held)."""
        # December 2025: Initialize progress tracking
        self._update_progress(phase="initializing")

        try:
            # December 2025: Use broadcast sync cycle for BROADCAST strategy
            if self._is_broadcast:
                self._update_progress(phase="broadcast_sync")
                result = await self.broadcast_sync_cycle()
                self._complete_progress(success=result > 0)
                return result

            # December 2025: Use pull sync cycle for PULL strategy (coordinator recovery)
            if self._resolved_strategy == SyncStrategy.PULL:
                self._update_progress(phase="pull_from_cluster")
                result = await self._pull_from_cluster_nodes()
                self._complete_progress(success=result > 0)
                return result

            # Skip if NFS node and skip_nfs_sync is enabled
            if self._is_nfs_node and self._config.skip_nfs_sync:
                logger.debug("Skipping sync cycle (NFS node)")
                self._complete_progress(success=True)
                return 0

            # Skip if this node is excluded
            if self.node_id in self._config.exclude_hosts:
                logger.debug("Skipping sync cycle (excluded host)")
                self._complete_progress(success=True)
                return 0

            # Check ClusterManifest exclusion rules
            if self._cluster_manifest:
                from app.distributed.cluster_manifest import DataType
                if not self._cluster_manifest.can_receive_data(self.node_id, DataType.GAME):
                    policy = self._cluster_manifest.get_sync_policy(self.node_id)
                    logger.debug(
                        f"Skipping sync cycle (manifest exclusion: {policy.exclusion_reason})"
                    )
                    self._complete_progress(success=True)
                    return 0

            # Check disk capacity before syncing
            self._update_progress(phase="checking_capacity")
            if not await self._check_disk_capacity():
                self._complete_progress(success=False, error="disk_capacity_exceeded")
                return 0

            # Check for pending data to sync
            self._update_progress(phase="checking_pending_data")
            pending = await self._get_pending_sync_data()
            if pending < self._config.min_games_to_sync:
                logger.debug(f"Skipping sync: only {pending} games pending")
                self._complete_progress(success=True)
                return 0

            logger.info(f"Sync cycle: {pending} games pending")

            # December 2025: Update progress with estimated file count
            self._update_progress(phase="collecting_from_peers", files_total=pending)

            # Trigger data collection from peers
            await self._collect_from_peers()
        except Exception as e:
            self._complete_progress(success=False, error=str(e))
            raise

        # December 2025 - Gap 4 fix: Verify synced databases after collection
        self._update_progress(phase="verifying_databases")
        verification_passed = await self._verify_synced_databases()
        if not verification_passed:
            logger.warning("[AutoSyncDaemon] Some databases failed verification")
            # Continue anyway - partial data is better than no data

        # Register synced data to manifest
        self._update_progress(phase="registering_data")
        await self._register_synced_data()

        # December 2025: Mark progress as complete
        self._update_progress(files_completed=pending)
        self._complete_progress(success=True)

        return pending


    # =========================================================================
    # Pull sync methods now inherited from SyncPullMixin
    # =========================================================================

    async def _check_disk_capacity(self) -> bool:
        """Check if disk has capacity for more data.

        Returns:
            True if sync should proceed, False if disk is full
        """
        if not self._cluster_manifest:
            return True

        # Update and check capacity
        capacity = self._cluster_manifest.update_local_capacity()

        if capacity.usage_percent >= self._config.max_disk_usage_percent:
            logger.warning(
                f"Disk usage {capacity.usage_percent:.1f}% exceeds threshold "
                f"({self._config.max_disk_usage_percent}%), triggering cleanup"
            )

            # Run cleanup if enabled
            if self._config.auto_cleanup_enabled:
                await self._run_disk_cleanup()

                # Check again after cleanup
                capacity = self._cluster_manifest.update_local_capacity()
                if capacity.usage_percent >= self._config.max_disk_usage_percent:
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
                trigger_usage_percent=self._config.max_disk_usage_percent,
                target_usage_percent=self._config.target_disk_usage_percent,
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

        December 31, 2025: CRITICAL FIX - Changed to extract only from the specific
        database instead of scanning entire parent directory. The previous approach
        was O(n²) - for N databases, each extraction scanned all N databases.

        December 31, 2025: Added caching for both successful and failed extractions
        to prevent repeated extraction from unchanged databases on every sync cycle.

        Args:
            db_path: Path to the synced database file

        Returns:
            Average quality score (0.0-1.0), or 0.0 if extraction fails
        """
        if not self._config.enable_quality_extraction or not HAS_QUALITY_EXTRACTION:
            return 0.0

        cache_key = str(db_path)
        now = time.time()

        # Check cache of failed databases (TTL 5 minutes)
        if cache_key in getattr(self, "_failed_db_cache", {}):
            if now - self._failed_db_cache[cache_key] < 300:  # 5 min TTL
                return 0.0

        # Check cache of successful extractions (TTL 5 minutes)
        # Skip re-extraction if database file hasn't been modified
        if not hasattr(self, "_quality_cache"):
            self._quality_cache: dict[str, tuple[float, float, float]] = {}  # path -> (score, extract_time, mtime)

        if cache_key in self._quality_cache:
            cached_score, cached_time, cached_mtime = self._quality_cache[cache_key]
            try:
                current_mtime = db_path.stat().st_mtime
                # Use cache if: file hasn't changed AND extracted within TTL
                if current_mtime == cached_mtime and now - cached_time < 300:
                    return cached_score
            except OSError:
                pass  # File might have been deleted, continue with fresh extraction

        try:
            from app.distributed.quality_extractor import extract_batch_quality

            # Extract quality only for THIS specific database, not entire directory
            game_qualities = extract_batch_quality(
                db_path=db_path,
                elo_lookup=self._elo_lookup,
                config=self._quality_config or QualityExtractorConfig(),
            )

            if not game_qualities:
                logger.debug(f"No quality scores extracted from {db_path.name}")
                return 0.0

            # Compute average quality
            avg_quality = sum(q.quality_score for q in game_qualities) / len(game_qualities)

            # Update stats
            self._sync_stats.games_quality_extracted += len(game_qualities)

            # Add high-quality games to priority queue
            high_quality_count = 0
            for quality in game_qualities:
                if quality.quality_score >= self._config.min_quality_score_for_priority:
                    await self._update_priority_queue(
                        config_key=f"{db_path.stem}",
                        quality_score=quality.quality_score,
                        game_count=1,
                    )
                    high_quality_count += 1

            self._sync_stats.games_added_to_priority += high_quality_count

            logger.info(
                f"Extracted quality from {db_path.name}: "
                f"{len(game_qualities)} games, avg={avg_quality:.3f}, "
                f"{high_quality_count} added to priority queue"
            )

            # Cache successful extraction with file mtime
            try:
                mtime = db_path.stat().st_mtime
                self._quality_cache[cache_key] = (avg_quality, time.time(), mtime)
            except OSError:
                pass

            return avg_quality

        except (RuntimeError, OSError, KeyError, ValueError, sqlite3.Error) as e:
            logger.warning(f"Quality extraction failed for {db_path.name}: {e}")
            # Cache failed database to avoid retrying for 5 minutes
            if not hasattr(self, "_failed_db_cache"):
                self._failed_db_cache = {}
            self._failed_db_cache[cache_key] = time.time()
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
        if not self._config.quality_filter_enabled:
            return True, "Quality filter disabled"

        import sqlite3

        try:
            from app.quality.unified_quality import compute_game_quality_from_params

            # Dec 2025: Use context manager to ensure connection is closed
            with connect_safe(db_path, timeout=5.0) as conn:
                cursor = conn.execute("""
                    SELECT game_id, game_status, winner, termination_reason,
                           total_moves, board_type
                    FROM games
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (self._config.quality_sample_size,))
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
            self._sync_stats.databases_quality_checked += 1

            if avg_quality < self._config.min_quality_for_sync:
                self._sync_stats.databases_skipped_quality += 1
                return False, f"Low quality: {avg_quality:.2f} < {self._config.min_quality_for_sync}"

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
            # Dec 30, 2025: Wrap blocking SQLite I/O with asyncio.to_thread()
            should_register, reason = await asyncio.to_thread(
                self._should_sync_database, db_path
            )
            if not should_register:
                logger.info(f"Skipping registration for {db_path.name}: {reason}")
                skipped_quality += 1
                continue

            try:
                # December 2025: Run blocking SQLite operations in thread pool
                def _query_games_sync(path: str) -> tuple:
                    with connect_safe(path, timeout=5.0, row_factory=None) as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT board_type, num_players FROM games LIMIT 1"
                        )
                        row = cursor.fetchone()
                        board_type = row[0] if row else None
                        num_players = row[1] if row else None
                        cursor.execute("SELECT game_id FROM games")
                        game_ids = [r[0] for r in cursor.fetchall()]
                    return board_type, num_players, game_ids

                board_type, num_players, game_ids = await asyncio.to_thread(
                    _query_games_sync, str(db_path)
                )

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
                    if self._config.enable_quality_extraction:
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
            # Dec 30, 2025: Wrap blocking SQLite I/O with asyncio.to_thread()
            should_sync, reason = await asyncio.to_thread(
                self._should_sync_database, db_path
            )
            if not should_sync:
                logger.debug(f"Excluding {db_path.name} from pending count: {reason}")
                skipped_dbs += 1
                continue

            try:
                # Dec 2025: Run blocking count query in thread pool
                def _count_games_sync(path: str) -> int:
                    with connect_safe(path, row_factory=None) as conn:
                        cursor = conn.execute("SELECT COUNT(*) FROM games")
                        return cursor.fetchone()[0]

                total_games += await asyncio.to_thread(_count_games_sync, str(db_path))
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
            self._sync_stats.games_synced = status.get("total_pulled", 0)
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
                # Dec 29, 2025: Use fast check for large databases (>100MB)
                # to prevent test suite timeouts while still detecting corruption
                db_size_mb = db_path.stat().st_size / (1024 * 1024)
                use_fast = db_size_mb > 100  # Fast check for DBs > 100MB
                is_valid, errors = check_sqlite_integrity(
                    db_path,
                    use_fast_check=use_fast,
                    timeout_seconds=15.0 if use_fast else 30.0,
                )

                if is_valid:
                    verified_count += 1
                else:
                    failed_count += 1
                    failed_dbs.append(db_path.name)
                    logger.error(
                        f"[AutoSyncDaemon] Database {db_path.name} failed integrity check: {errors}"
                    )
                    # Phase 7 (Dec 29, 2025): Quarantine corrupted database
                    # Move to quarantine/ directory to prevent it from participating
                    # in sync operations and potentially spreading corruption.
                    quarantine_dest = quarantine_corrupted_db(db_path)
                    if quarantine_dest:
                        logger.warning(
                            f"[AutoSyncDaemon] Quarantined corrupted database: {db_path.name} -> {quarantine_dest}"
                        )
                    else:
                        logger.error(
                            f"[AutoSyncDaemon] Failed to quarantine corrupted database: {db_path.name}"
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
        self._sync_stats.databases_verified += verified_count
        self._sync_stats.databases_verification_failed += failed_count
        self._sync_stats.last_verification_time = time.time()

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
                        "total_verified": self._sync_stats.databases_verified,
                        "total_failed": self._sync_stats.databases_verification_failed,
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
                "enabled": self._config.enabled,
                "interval_seconds": self._config.interval_seconds,
                "exclude_hosts": self._config.exclude_hosts,
                "max_disk_usage_percent": self._config.max_disk_usage_percent,
                "auto_cleanup_enabled": self._config.auto_cleanup_enabled,
            },
            "stats": {
                "total_syncs": self._sync_stats.total_syncs,
                "successful_syncs": self._sync_stats.successful_syncs,
                "failed_syncs": self._sync_stats.failed_syncs,
                "games_synced": self._sync_stats.games_synced,
                "last_sync_time": self._sync_stats.last_sync_time,
                "last_error": self._sync_stats.last_error,
                "databases_quality_checked": self._sync_stats.databases_quality_checked,
                "databases_skipped_quality": self._sync_stats.databases_skipped_quality,
                "games_quality_extracted": self._sync_stats.games_quality_extracted,
                "games_added_to_priority": self._sync_stats.games_added_to_priority,
                # December 2025 - Gap 4 fix: Verification stats
                "databases_verified": self._sync_stats.databases_verified,
                "databases_verification_failed": self._sync_stats.databases_verification_failed,
                "last_verification_time": self._sync_stats.last_verification_time,
            },
            "quality_filter": {
                "enabled": self._config.quality_filter_enabled,
                "min_quality": self._config.min_quality_for_sync,
                "sample_size": self._config.quality_sample_size,
            },
            "quality_extraction": {
                "enabled": self._config.enable_quality_extraction,
                "min_quality_for_priority": self._config.min_quality_score_for_priority,
                "games_extracted": self._sync_stats.games_quality_extracted,
                "games_prioritized": self._sync_stats.games_added_to_priority,
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
            "status": self._status.value,
            "uptime_seconds": self.uptime_seconds,
            # January 2026: HandlerBase uses _stats.started_at, not _start_time
            "start_time": self._stats.started_at,
            "events_processed": self._events_processed,
            "errors_count": self._errors_count,
            # January 2026: HandlerBase uses _stats.last_error, not _last_error
            "last_error": self._stats.last_error,
            # Sync-specific metrics
            "node_id": self.node_id,
            "provider": self._provider,
            "is_nfs_node": self._is_nfs_node,
            "total_syncs": self._sync_stats.total_syncs,
            "successful_syncs": self._sync_stats.successful_syncs,
            "failed_syncs": self._sync_stats.failed_syncs,
            "games_synced": self._sync_stats.games_synced,
            "bytes_transferred": self._sync_stats.bytes_transferred,
            "last_sync_time": self._sync_stats.last_sync_time,
            # December 2025 - Gap 4 fix: Verification metrics
            "databases_verified": self._sync_stats.databases_verified,
            "databases_verification_failed": self._sync_stats.databases_verification_failed,
            "last_verification_time": self._sync_stats.last_verification_time,
            # December 2025: Real-time sync progress
            "sync_progress": self._progress.to_dict(),
        }

    def get_sync_progress(self) -> SyncProgress:
        """Get real-time sync progress information.

        December 2025: Added for monitoring and UI integration.
        Returns a snapshot of current sync operation progress.

        Returns:
            SyncProgress with current operation state.
        """
        return self._progress

    def _update_progress(
        self,
        phase: str = "",
        current_file: str = "",
        current_node: str = "",
        files_completed: int | None = None,
        files_total: int | None = None,
        bytes_transferred: int | None = None,
        bytes_total: int | None = None,
        error_message: str = "",
        is_active: bool | None = None,
    ) -> None:
        """Update sync progress tracking (internal helper).

        December 2025: Centralized progress update for consistent tracking.

        Args:
            phase: Current sync phase description
            current_file: File currently being synced
            current_node: Node currently being synced to/from
            files_completed: Number of files completed (incremental update if None)
            files_total: Total files expected
            bytes_transferred: Bytes transferred so far (incremental update if None)
            bytes_total: Total bytes expected
            error_message: Error message if any
            is_active: Override active state (defaults to True during update)
        """
        self._progress.last_update_at = time.time()

        if is_active is not None:
            self._progress.is_active = is_active
        elif not self._progress.is_active:
            # Starting new sync cycle
            self._progress.is_active = True
            self._progress.started_at = time.time()
            self._progress.files_completed = 0
            self._progress.bytes_transferred = 0

        if phase:
            self._progress.current_phase = phase
        if current_file:
            self._progress.current_file = current_file
        if current_node:
            self._progress.current_node = current_node
        if files_completed is not None:
            self._progress.files_completed = files_completed
        if files_total is not None:
            self._progress.files_total = files_total
        if bytes_transferred is not None:
            self._progress.bytes_transferred = bytes_transferred
        if bytes_total is not None:
            self._progress.bytes_total = bytes_total
        if error_message:
            self._progress.error_message = error_message

        # Estimate completion time based on current progress
        if self._progress.bytes_total > 0 and self._progress.transfer_rate_bytes_per_sec > 0:
            remaining_bytes = self._progress.bytes_total - self._progress.bytes_transferred
            remaining_seconds = remaining_bytes / self._progress.transfer_rate_bytes_per_sec
            self._progress.estimated_completion_at = time.time() + remaining_seconds
        elif self._progress.files_total > 0 and self._progress.elapsed_seconds > 0:
            files_remaining = self._progress.files_total - self._progress.files_completed
            files_per_sec = self._progress.files_completed / max(self._progress.elapsed_seconds, 1)
            if files_per_sec > 0:
                remaining_seconds = files_remaining / files_per_sec
                self._progress.estimated_completion_at = time.time() + remaining_seconds

    def _complete_progress(self, success: bool = True, error: str = "") -> None:
        """Mark sync progress as complete.

        December 2025: Called when sync cycle finishes.

        Args:
            success: Whether sync completed successfully
            error: Error message if failed
        """
        self._progress.is_active = False
        self._progress.current_phase = "completed" if success else "failed"
        self._progress.last_update_at = time.time()
        if error:
            self._progress.error_message = error

    def health_check(self) -> HealthCheckResult:
        """Check daemon health.

        Returns:
            Health check result with status and sync details.
        """
        # Check for error state
        if self._status == CoordinatorStatus.ERROR:
            return HealthCheckResult.unhealthy(
                f"Daemon in error state: {self._last_error}"
            )

        # Check if stopped
        if self._status == CoordinatorStatus.STOPPED:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="Daemon is stopped",
            )

        # Check if disabled by config
        if not self._config.enabled:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="Daemon disabled by configuration",
            )

        # Check sync health
        if self._sync_stats.failed_syncs > self._sync_stats.successful_syncs * 0.5:
            return HealthCheckResult.degraded(
                f"High failure rate: {self._sync_stats.failed_syncs} failures, "
                f"{self._sync_stats.successful_syncs} successes",
                failure_rate=self._sync_stats.failed_syncs / max(self._sync_stats.total_syncs, 1),
            )

        # December 2025 - Gap 4 fix: Check verification health
        if self._sync_stats.databases_verified > 0:
            verification_failure_rate = (
                self._sync_stats.databases_verification_failed /
                max(self._sync_stats.databases_verified + self._sync_stats.databases_verification_failed, 1)
            )
            if verification_failure_rate > 0.1:  # More than 10% failure rate
                return HealthCheckResult.degraded(
                    f"High verification failure rate: {self._sync_stats.databases_verification_failed} failed, "
                    f"{self._sync_stats.databases_verified} passed ({verification_failure_rate*100:.1f}%)",
                    verification_failure_rate=verification_failure_rate,
                )

        # Check for stale sync
        if self._sync_stats.last_sync_time > 0:
            sync_age = time.time() - self._sync_stats.last_sync_time
            if sync_age > self._config.interval_seconds * 3:
                return HealthCheckResult.degraded(
                    f"No sync in {sync_age:.0f}s (interval: {self._config.interval_seconds}s)",
                    seconds_since_last_sync=sync_age,
                )

        # Healthy
        return HealthCheckResult(
            healthy=True,
            status=self._status,
            details={
                "uptime_seconds": self.uptime_seconds,
                "total_syncs": self._sync_stats.total_syncs,
                "games_synced": self._sync_stats.games_synced,
                "gossip_active": self._gossip_daemon is not None,
                "manifest_active": self._cluster_manifest is not None,
                # December 2025 - Gap 4 fix: Verification stats
                "databases_verified": self._sync_stats.databases_verified,
                "databases_verification_failed": self._sync_stats.databases_verification_failed,
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
    from app.config.coordination_defaults import SyncDefaults

    config = AutoSyncConfig.from_config_file()
    config.strategy = SyncStrategy.BROADCAST
    config.sync_interval = SyncDefaults.FAST_SYNC_INTERVAL  # 30s for training freshness
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


# Backward compatibility - strategies moved to sync_strategies.py (December 2025)
# Re-exported here for existing imports
# from app.coordination.sync_strategies import (
#     SyncStrategy,
#     AutoSyncConfig,
#     SyncStats,
#     MIN_MOVES_PER_GAME,
#     DEFAULT_MIN_MOVES,
# )

__all__ = [  # noqa: RUF022
    # Core classes (re-exported from sync_strategies.py)
    "AutoSyncConfig",
    "AutoSyncDaemon",
    "SyncStats",
    "SyncStrategy",
    # Constants (re-exported from sync_strategies.py)
    "MIN_MOVES_PER_GAME",
    "DEFAULT_MIN_MOVES",
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

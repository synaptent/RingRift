"""Automated NPZ Export Daemon (December 2025).

Automatically exports training data (NPZ files) when game counts exceed thresholds.
This closes the gap between selfplay and training by eliminating the manual export step.

Key features:
- Subscribes to SELFPLAY_COMPLETE events
- Tracks accumulated games per configuration
- Triggers export when game threshold reached (default: 100 games)
- Emits NPZ_EXPORT_STARTED event when export begins
- Emits NPZ_EXPORT_COMPLETE event after successful export
- Integrates with GameDiscovery for finding databases
- Supports cooldown to prevent export spam

Usage:
    from app.coordination.auto_export_daemon import AutoExportDaemon

    daemon = AutoExportDaemon()
    await daemon.start()

December 2025: Created as part of Phase 1 automation improvements.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.core.async_context import safe_create_task
from app.coordination.event_handler_utils import extract_config_key
from app.coordination.event_utils import make_config_key, parse_config_key
from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.utils.sqlite_utils import connect_safe

# Sprint 4 (Jan 2, 2026): Export validation defaults
try:
    from app.config.coordination_defaults import ExportValidationDefaults
except ImportError:
    ExportValidationDefaults = None  # Fallback for standalone usage

logger = logging.getLogger(__name__)


@dataclass
class AutoExportConfig:
    """Configuration for automated NPZ export."""

    enabled: bool = True
    # Minimum games before triggering export
    # Lowered from 500→100→50 (Dec 2025) for faster training iteration
    min_games_threshold: int = 50
    # Cooldown between exports for same config (seconds)
    # Reduced from 30min→5min→1min (Dec 2025) to minimize training data lag
    export_cooldown_seconds: int = 60  # 1 minute
    # Maximum concurrent exports - coordinator=1 to prevent P2P event loop I/O contention
    # Feb 2026: Made env-configurable via RINGRIFT_MAX_CONCURRENT_EXPORTS
    max_concurrent_exports: int = int(os.environ.get("RINGRIFT_MAX_CONCURRENT_EXPORTS",
        "1" if os.environ.get("RINGRIFT_IS_COORDINATOR", "").lower() == "true" else "4"))
    # Maximum workers per export subprocess (None = use export_replay_dataset default)
    # Feb 2026: On coordinator nodes, cap workers to prevent multiprocessing fan-out OOM
    max_export_workers: int | None = None
    # Timeout for export subprocess (seconds)
    export_timeout_seconds: int = 3600  # 1 hour
    # Whether to use incremental export (--use-cache)
    use_incremental_export: bool = True
    # Quality filtering options
    require_completed_games: bool = True
    min_moves: int = 10
    # Data source options (December 30, 2025)
    # Include gauntlet/tournament games by default for higher quality training data
    include_gauntlet: bool = True  # Include evaluation gauntlet games
    include_tournaments: bool = True  # Include tournament games
    # Output directory for NPZ files
    output_dir: Path = field(default_factory=lambda: Path("data/training"))
    # State persistence (Phase 8 Dec 2025)
    state_db_path: Path = field(default_factory=lambda: Path("data/export_daemon_state.db"))
    persist_state: bool = True  # Enable state persistence to recover on crash
    # Event-driven batch export settings (December 2025)
    # Part of 48-hour autonomous operation optimization.
    # Implements "N games OR M seconds, whichever first" semantics.
    event_driven: bool = True  # Enable event-driven mode (vs pure timer)
    batch_accumulation_timeout_seconds: int = 30  # Max wait before export
    immediate_threshold_multiplier: float = 2.0  # Export immediately if games >= threshold * 2
    # January 3, 2026: Sync-gating option - when disabled, exports proceed immediately
    # with local data instead of waiting for DATA_SYNC_COMPLETED from cluster.
    # Pipeline analysis showed sync-gating added 15-20 Elo worth of latency by
    # delaying exports until remote data arrived. Disabled by default for faster iteration.
    gate_export_on_sync: bool = False


@dataclass
class ConfigExportState:
    """Tracks export state for a single configuration."""

    config_key: str
    board_type: str
    num_players: int
    games_since_last_export: int = 0
    last_export_time: float = 0.0
    last_export_games: int = 0
    total_exported_samples: int = 0
    export_in_progress: bool = False
    consecutive_failures: int = 0


@dataclass
class BatchAccumulator:
    """Tracks accumulated games for batch export triggering.

    December 2025: Part of event-driven export optimization.
    Implements "N games OR M seconds, whichever first" semantics.
    """

    config_key: str
    accumulated_games: int = 0
    accumulation_started: float = 0.0  # Timestamp when accumulation began
    timer_task: asyncio.Task | None = None

    def reset(self) -> None:
        """Reset accumulator after export."""
        self.accumulated_games = 0
        self.accumulation_started = 0.0
        if self.timer_task and not self.timer_task.done():
            self.timer_task.cancel()
        self.timer_task = None


class AutoExportDaemon(HandlerBase):
    """Daemon that automatically exports training data when thresholds are met.

    Inherits from HandlerBase (December 2025 migration) providing:
    - Automatic event subscription via _get_event_subscriptions()
    - Singleton pattern via get_instance()
    - Standardized health check format
    - Lifecycle management (start/stop)
    """

    def __init__(self, config: AutoExportConfig | None = None):
        self._daemon_config = config or AutoExportConfig()
        super().__init__(
            name="auto_export",
            config=self._daemon_config,
            # Jan 5, 2026 (Task 8.6): Reduced from 300s (5 min) to 30s
            # Faster scanning improves data flow latency to training pipeline.
            # Rate limiting still enforced via min_games_threshold (50 games).
            cycle_interval=30.0,  # 30 seconds scan interval
        )
        self._export_states: dict[str, ConfigExportState] = {}
        # Feb 2026: Semaphore must be created inside the event loop (_on_start),
        # not in __init__. Creating it here binds to a different/no event loop,
        # causing 8,000-14,000+ waiters and export failures.
        self._export_semaphore: asyncio.Semaphore | None = None
        self._state_db_initialized = False
        # December 2025: Deduplication guard - when StageEvent subscriptions are active,
        # skip DataEventType handlers to prevent double-counting games
        self._stage_events_active = False
        # December 2025 Phase 3: Track configs pending sync completion.
        # Export is gated on sync completion to prevent race conditions where
        # export starts before data from other nodes has arrived.
        self._pending_sync_configs: set[str] = set()
        # Dec 31, 2025: Track when each config was marked pending (for stale cleanup)
        self._pending_sync_times: dict[str, float] = {}
        self._max_pending_time = 300.0  # 5 minutes max wait for sync
        # Track whether we should skip due to coordinator mode
        self._coordinator_skip = False
        # Event-driven batch accumulation (December 2025)
        # Part of 48-hour autonomous operation optimization.
        self._batch_accumulators: dict[str, BatchAccumulator] = {}
        self._pending_timer_exports: set[str] = set()  # Configs with active timers

    @property
    def config(self) -> AutoExportConfig:
        """Get the daemon configuration."""
        return self._daemon_config

    def _get_event_subscriptions(self) -> dict[str, Callable]:
        """Return event subscriptions for HandlerBase.

        Subscribes to:
        - SELFPLAY_COMPLETE: Track new games and trigger export
        - SYNC_COMPLETE: Cross-node data sync completion
        - NEW_GAMES_AVAILABLE: New games from local selfplay
        - DATA_SYNC_COMPLETED: Clear sync pending flag
        """
        return {
            "selfplay_complete": self._on_selfplay_complete,
            "sync_complete": self._on_sync_complete,
            "new_games_available": self._on_new_games,
            "data_sync_completed": self._on_data_sync_completed,
        }

    async def _on_start(self) -> None:
        """Hook called before main loop - check coordinator mode and init state DB."""
        # Feb 2026: Create semaphore inside the running event loop to avoid
        # cross-loop binding that caused 8,000-14,000+ waiters and export failures.
        # Mar 2026: Guard against Semaphore(0) which silently deadlocks all exports.
        # This happens when RINGRIFT_MAX_CONCURRENT_EXPORTS=0 is set in the environment.
        sem_value = max(1, self._daemon_config.max_concurrent_exports)
        if self._daemon_config.max_concurrent_exports <= 0:
            logger.warning(
                f"[AutoExportDaemon] max_concurrent_exports={self._daemon_config.max_concurrent_exports} "
                f"would deadlock Semaphore. Clamping to 1."
            )
        self._export_semaphore = asyncio.Semaphore(sem_value)

        from app.config.env import env

        # Initialize state persistence first for visibility, even on coordinators
        # Jan 3, 2026: State DB should be initialized for observability regardless
        # of whether exports are actually run on this node
        if self._daemon_config.persist_state:
            await asyncio.to_thread(self._init_state_db)
            await asyncio.to_thread(self._load_state)

        # Jan 26, 2026: Changed from `is_coordinator or not export_enabled` to just
        # `not export_enabled`. The coordinator CAN export consolidated game data if
        # RINGRIFT_EXPORT_ENABLED=true is set. Previously, coordinator was always
        # skipped even when export_enabled=True, breaking the training data pipeline.
        if not env.export_enabled:
            logger.info(
                f"[AutoExportDaemon] Export operations disabled on node: {env.node_id} "
                f"(is_coordinator={env.is_coordinator}, export_enabled={env.export_enabled}). "
                f"Set RINGRIFT_EXPORT_ENABLED=true to enable exports on this node. "
                f"State tracking still active for observability."
            )
            self._coordinator_skip = True
            return

    # ========== State Persistence (Phase 8 Dec 2025) ==========

    def _init_state_db(self) -> None:
        """Initialize SQLite database for state persistence.

        Phase 8 Dec 2025: Persists export state to survive daemon restarts.
        This prevents data loss when pending export counts are lost on crash.
        """
        import sqlite3

        try:
            db_path = self.config.state_db_path
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Use context manager to ensure connection is always closed
            with connect_safe(db_path, row_factory=None) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS export_state (
                        config_key TEXT PRIMARY KEY,
                        board_type TEXT NOT NULL,
                        num_players INTEGER NOT NULL,
                        games_since_last_export INTEGER DEFAULT 0,
                        last_export_time REAL DEFAULT 0,
                        last_export_games INTEGER DEFAULT 0,
                        total_exported_samples INTEGER DEFAULT 0,
                        consecutive_failures INTEGER DEFAULT 0,
                        updated_at REAL DEFAULT 0
                    )
                """)

                conn.commit()
            self._state_db_initialized = True

            logger.info(f"[AutoExportDaemon] State database initialized: {db_path}")

        except (sqlite3.Error, OSError) as e:
            logger.error(f"[AutoExportDaemon] Failed to initialize state DB: {e}")
            self._state_db_initialized = False

    def _load_state(self) -> None:
        """Load persisted state from SQLite on startup.

        Recovers pending export counts and last export times that would
        otherwise be lost on daemon restart.
        """
        if not self._state_db_initialized:
            return

        import sqlite3

        try:
            # Use context manager to ensure connection is always closed
            with connect_safe(self.config.state_db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT * FROM export_state")
                rows = cursor.fetchall()

                loaded_count = 0
                for row in rows:
                    config_key = row["config_key"]
                    self._export_states[config_key] = ConfigExportState(
                        config_key=config_key,
                        board_type=row["board_type"],
                        num_players=row["num_players"],
                        games_since_last_export=row["games_since_last_export"],
                        last_export_time=row["last_export_time"],
                        last_export_games=row["last_export_games"],
                        total_exported_samples=row["total_exported_samples"],
                        consecutive_failures=row["consecutive_failures"],
                    )
                    loaded_count += 1

            if loaded_count > 0:
                logger.info(
                    f"[AutoExportDaemon] Loaded {loaded_count} config states from persistence"
                )

        except (sqlite3.Error, OSError) as e:
            logger.error(f"[AutoExportDaemon] Failed to load state: {e}")

    def _save_state(self, config_key: str) -> None:
        """Persist state for a single config to SQLite.

        Called after any state change to ensure durability.
        """
        if not self._state_db_initialized or not self.config.persist_state:
            return

        state = self._export_states.get(config_key)
        if not state:
            return

        import sqlite3

        try:
            # Use context manager to ensure connection is always closed
            with connect_safe(self.config.state_db_path, row_factory=None) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO export_state
                    (config_key, board_type, num_players, games_since_last_export,
                     last_export_time, last_export_games, total_exported_samples,
                     consecutive_failures, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    config_key,
                    state.board_type,
                    state.num_players,
                    state.games_since_last_export,
                    state.last_export_time,
                    state.last_export_games,
                    state.total_exported_samples,
                    state.consecutive_failures,
                    time.time(),
                ))

                conn.commit()

        except (sqlite3.Error, OSError) as e:
            logger.debug(f"[AutoExportDaemon] Failed to save state for {config_key}: {e}")

    # ========== Event Handlers ==========

    async def _on_selfplay_complete(self, result: Any) -> None:
        """Handle selfplay completion event.

        December 2025 Phase 3: Mark config as pending sync before recording games.
        This prevents export from starting before sync has completed, avoiding
        race conditions where export uses incomplete data.
        """
        try:
            # Extract config info from result
            board_type = getattr(result, "board_type", None)
            num_players = getattr(result, "num_players", None)
            games_generated = getattr(result, "games_generated", 0)

            if not board_type or not num_players:
                # Try to extract from metadata
                metadata = getattr(result, "metadata", {})
                board_type = board_type or metadata.get("board_type")
                num_players = num_players or metadata.get("num_players")

            if not board_type or not num_players:
                logger.debug("[AutoExportDaemon] Missing config info in selfplay result")
                return

            config_key = make_config_key(board_type, num_players)

            # Phase 3: Mark config as pending sync - export will wait for DATA_SYNC_COMPLETED
            # January 3, 2026: Only apply sync-gating if gate_export_on_sync is enabled.
            # When disabled, exports proceed immediately with local data.
            if self.config.gate_export_on_sync:
                self._pending_sync_configs.add(config_key)
                self._pending_sync_times[config_key] = time.time()  # Dec 31, 2025: Track pending time
                logger.debug(
                    f"[AutoExportDaemon] {config_key}: Marked as pending sync "
                    "(export gated on DATA_SYNC_COMPLETED)"
                )

            await self._record_games(config_key, board_type, num_players, games_generated)

        except Exception as e:  # noqa: BLE001
            logger.error(f"[AutoExportDaemon] Error handling selfplay complete: {e}")

    async def _on_sync_complete(self, result: Any) -> None:
        """Handle sync completion event (Phase 3A.3: Dec 2025).

        When games are synced from other nodes, check if export is needed.
        This enables cross-node data to trigger exports just like local selfplay.
        """
        try:
            # Extract sync info from result
            metadata = getattr(result, "metadata", {})
            games_synced = metadata.get("games_synced", 0) or metadata.get("files_synced", 0)
            config_key = metadata.get("config_key", "")

            if not games_synced or not config_key:
                return

            board_type, num_players = self._parse_config_key(config_key)
            if not board_type or not num_players:
                return

            logger.info(
                f"[AutoExportDaemon] Sync complete: {games_synced} games for {config_key}"
            )
            await self._record_games(config_key, board_type, num_players, games_synced)

        except Exception as e:  # noqa: BLE001
            logger.error(f"[AutoExportDaemon] Error handling sync complete: {e}")

    async def _on_new_games(self, event: Any) -> None:
        """Handle new games available event."""
        try:
            payload = getattr(event, "payload", {})
            new_games = payload.get("new_games", 0)
            board_type = payload.get("board_type")
            num_players = payload.get("num_players")

            if board_type and num_players:
                config_key = make_config_key(board_type, num_players)
                await self._record_games(config_key, board_type, num_players, new_games)

        except Exception as e:  # noqa: BLE001
            logger.error(f"[AutoExportDaemon] Error handling new games event: {e}")

    async def _on_selfplay_complete_event(self, event: Any) -> None:
        """Handle SELFPLAY_COMPLETE data events.

        December 2025: Added deduplication guard. If StageEvent subscription is active,
        skip this handler to prevent double-counting games (the StageEvent handler
        already processed this event).
        """
        # Deduplication guard: skip if StageEvent already handled this
        if self._stage_events_active:
            logger.debug(
                "[AutoExportDaemon] Skipping DataEventType.SELFPLAY_COMPLETE "
                "(StageEvent already active)"
            )
            return

        try:
            payload = getattr(event, "payload", {}) or {}
            config_key = extract_config_key(payload)
            games_generated = payload.get("games_played", payload.get("games_generated", 0))
            if not config_key or not games_generated:
                return

            board_type, num_players = self._parse_config_key(config_key)
            if not board_type or not num_players:
                return

            await self._record_games(config_key, board_type, num_players, games_generated)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[AutoExportDaemon] Error handling SELFPLAY_COMPLETE event: {e}")

    async def _on_data_sync_completed(self, event: Any) -> None:
        """Handle DATA_SYNC_COMPLETED data events.

        December 2025 Phase 3: Clear pending sync flag and allow export to proceed.
        This completes the sync-gating mechanism that prevents export race conditions.

        Dec 30, 2025 Phase 4.1: Also scans synced databases for export after sync.
        This enables training on data synced from other cluster nodes.

        Dec 31, 2025: Fix sync-gating deadlock - clear ALL pending configs when ANY
        sync completes. Sync is a global operation, not per-config. Previously, exports
        waited forever because DATA_SYNC_COMPLETED events don't include config_key.
        """
        try:
            payload = getattr(event, "payload", {}) or {}
            config_key = extract_config_key(payload)
            games_synced = payload.get("games_synced", 0) or payload.get("files_synced", 0)
            source_host = payload.get("source_host")

            # Dec 31, 2025: Clear ALL pending configs when ANY sync completes
            # This fixes the deadlock where exports wait forever for config-specific
            # sync events that never arrive (sync is global, not per-config).
            pending = list(self._pending_sync_configs)
            if pending:
                logger.info(
                    f"[AutoExportDaemon] Sync completed with {games_synced} games, "
                    f"clearing {len(pending)} pending configs: {pending}"
                )
                self._pending_sync_configs.clear()
                self._pending_sync_times.clear()  # Also clear timing data

                # Trigger export check for all previously-pending configs
                for pending_config_key in pending:
                    await self._maybe_trigger_export(pending_config_key)

            # Phase 3 (legacy): If specific config_key present, also record games
            if config_key:
                board_type, num_players = self._parse_config_key(config_key)
                if board_type and num_players:
                    await self._record_games(config_key, board_type, num_players, games_synced)

            # Phase 4.1 (Dec 30, 2025): Scan synced databases for export
            if games_synced > 0 and source_host:
                await self._scan_synced_databases(source_host, games_synced)

        except Exception as e:  # noqa: BLE001
            logger.error(f"[AutoExportDaemon] Error handling DATA_SYNC_COMPLETED event: {e}")

    async def _scan_synced_databases(self, source_host: str, games_synced: int) -> None:
        """Scan synced databases from a specific source host for export.

        Dec 30, 2025: Phase 4.1 of distributed data pipeline architecture.

        After data is synced from another node, this scans the synced directory
        to find databases that contain games ready for export. This enables
        training on data generated by remote cluster nodes.

        Args:
            source_host: Source host that provided the synced data
            games_synced: Number of games that were synced
        """
        from pathlib import Path

        try:
            # Determine synced directory
            synced_base = self._get_data_path("games") / "synced"

            # Check source-specific subdirectory first
            synced_dir = synced_base / source_host
            if not synced_dir.exists():
                synced_dir = synced_base  # Fall back to base synced dir

            if not synced_dir.exists():
                return

            # Scan for databases
            # Dec 30, 2025: Wrap blocking SQLite in asyncio.to_thread()
            import asyncio
            import sqlite3

            def _scan_databases_sync(scan_dir: Path) -> list[tuple[str, str, int, int]]:
                """Sync helper to scan databases without blocking event loop."""
                results: list[tuple[str, str, int, int]] = []
                for db_path in scan_dir.glob("*.db"):
                    if not db_path.is_file():
                        continue
                    try:
                        with connect_safe(db_path, row_factory=None) as conn:
                            cursor = conn.execute("""
                                SELECT board_type, num_players, COUNT(DISTINCT game_id) as game_count
                                FROM games
                                WHERE board_type IS NOT NULL
                                GROUP BY board_type, num_players
                            """)
                            for row in cursor:
                                board_type, num_players, game_count = row
                                if board_type and num_players and game_count > 0:
                                    results.append((board_type, num_players, game_count, 1))
                    except sqlite3.Error:
                        continue
                return results

            scan_results = await asyncio.to_thread(_scan_databases_sync, synced_dir)

            databases_scanned = 0
            games_found = 0
            for board_type, num_players, game_count, db_count in scan_results:
                config_key = make_config_key(board_type, num_players)
                await self._record_games(config_key, board_type, num_players, game_count)
                games_found += game_count
                databases_scanned += db_count

            if databases_scanned > 0:
                logger.info(
                    f"[AutoExportDaemon] Scanned {databases_scanned} synced databases "
                    f"from {source_host}, found {games_found} games"
                )

        except Exception as e:
            logger.warning(f"[AutoExportDaemon] Error scanning synced databases: {e}")

    def _parse_config_key(self, config_key: str) -> tuple[str | None, int | None]:
        """Parse a config key like "square8_2p" into (board_type, num_players).

        December 2025: Now delegates to event_utils.parse_config_key() for
        consistent parsing across the codebase.
        """
        from app.coordination.event_utils import parse_config_key

        parsed = parse_config_key(config_key)
        if parsed is None:
            return None, None
        return parsed.board_type, parsed.num_players

    def _get_min_elo_for_config(self, config_key: str) -> float | None:
        """Get minimum Elo for export filtering (current_best - 300, or None if < 1400)."""
        try:
            from app.coordination.elo_progress_tracker import get_elo_progress_tracker
            tracker = get_elo_progress_tracker()
            snapshot = tracker.get_latest_snapshot(config_key)
            if snapshot is None or snapshot.best_elo < 1400:
                return None
            return snapshot.best_elo - 300
        except Exception:
            return None

    # ========== Event-Driven Batch Export (December 2025) ==========

    async def _evaluate_batch_trigger(self, config_key: str) -> None:
        """Evaluate whether to trigger export based on batch accumulation.

        December 2025: Part of 48-hour autonomous operation optimization.
        Implements "N games OR M seconds, whichever first" semantics:

        1. Immediate trigger: games >= threshold * 2.0 → export NOW
        2. Threshold trigger: games >= threshold → export NOW
        3. Timer trigger: games > 0, start timer → export when timer fires

        This reduces export latency from ~150s average to ~15s average.
        """
        if not self._daemon_config.event_driven:
            return

        state = self._export_states.get(config_key)
        if not state or state.export_in_progress:
            return

        # Get or create batch accumulator
        if config_key not in self._batch_accumulators:
            self._batch_accumulators[config_key] = BatchAccumulator(config_key=config_key)

        accumulator = self._batch_accumulators[config_key]
        games = state.games_since_last_export

        # Calculate thresholds
        threshold = self._daemon_config.min_games_threshold
        immediate_threshold = int(threshold * self._daemon_config.immediate_threshold_multiplier)

        # Case 1: Immediate trigger - very large batch
        if games >= immediate_threshold:
            logger.info(
                f"[AutoExportDaemon] {config_key}: Immediate export triggered "
                f"({games} >= {immediate_threshold} games)"
            )
            accumulator.reset()
            self._pending_timer_exports.discard(config_key)
            safe_create_task(
                self._run_export(config_key),
                name=f"batch_export_{config_key}",
            )
            return

        # Case 2: Threshold trigger - normal threshold reached
        if games >= threshold:
            logger.info(
                f"[AutoExportDaemon] {config_key}: Threshold export triggered "
                f"({games} >= {threshold} games)"
            )
            accumulator.reset()
            self._pending_timer_exports.discard(config_key)
            safe_create_task(
                self._run_export(config_key),
                name=f"batch_export_{config_key}",
            )
            return

        # Case 3: Timer trigger - start timer if not already running
        if games > 0 and config_key not in self._pending_timer_exports:
            if accumulator.accumulation_started == 0.0:
                accumulator.accumulation_started = time.time()

            # Start timer for batch accumulation timeout
            timeout = self._daemon_config.batch_accumulation_timeout_seconds
            logger.debug(
                f"[AutoExportDaemon] {config_key}: Starting {timeout}s batch timer "
                f"({games} games accumulated)"
            )
            self._pending_timer_exports.add(config_key)
            accumulator.timer_task = safe_create_task(
                self._batch_timer_callback(config_key, timeout),
                name=f"batch_timer_{config_key}",
            )

    async def _batch_timer_callback(self, config_key: str, timeout: float) -> None:
        """Timer callback that triggers export when accumulation timeout fires.

        December 2025: Part of 48-hour autonomous operation optimization.
        This ensures exports happen within `batch_accumulation_timeout_seconds`
        even if the game threshold isn't reached.
        """
        try:
            await asyncio.sleep(timeout)

            # Check if still relevant (not already exported)
            if config_key not in self._pending_timer_exports:
                return

            state = self._export_states.get(config_key)
            if not state or state.export_in_progress:
                self._pending_timer_exports.discard(config_key)
                return

            games = state.games_since_last_export
            if games > 0:
                logger.info(
                    f"[AutoExportDaemon] {config_key}: Timer export triggered "
                    f"after {timeout}s ({games} games)"
                )
                # Reset accumulator
                if config_key in self._batch_accumulators:
                    self._batch_accumulators[config_key].reset()
                self._pending_timer_exports.discard(config_key)
                await self._run_export(config_key)
            else:
                self._pending_timer_exports.discard(config_key)

        except asyncio.CancelledError:
            # Timer was cancelled (export triggered by threshold)
            self._pending_timer_exports.discard(config_key)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[AutoExportDaemon] Timer callback error for {config_key}: {e}")
            self._pending_timer_exports.discard(config_key)

    async def _record_games(
        self, config_key: str, board_type: str, num_players: int, games: int
    ) -> None:
        """Record new games and check if export should be triggered."""
        if config_key not in self._export_states:
            self._export_states[config_key] = ConfigExportState(
                config_key=config_key,
                board_type=board_type,
                num_players=num_players,
            )

        state = self._export_states[config_key]
        state.games_since_last_export += games

        # Persist state to survive daemon restarts (Phase 8)
        # Dec 30, 2025: Wrap blocking SQLite I/O with asyncio.to_thread()
        await asyncio.to_thread(self._save_state, config_key)

        logger.debug(
            f"[AutoExportDaemon] {config_key}: +{games} games, "
            f"total pending: {state.games_since_last_export}"
        )

        # Check if we should trigger export
        # December 2025: Use event-driven batch accumulation if enabled
        if self._daemon_config.event_driven:
            await self._evaluate_batch_trigger(config_key)
        else:
            # Fallback to original threshold-based triggering
            await self._maybe_trigger_export(config_key)

    async def _validate_export_readiness(
        self,
        config_key: str,
        state: ConfigExportState,
    ) -> tuple[bool, str]:
        """Validate that export data meets quality thresholds.

        Sprint 4 (Jan 2, 2026): Pre-export validation to prevent low-quality
        training data from entering the pipeline.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            state: Current export state for this config

        Returns:
            Tuple of (is_ready, reason)
        """
        if ExportValidationDefaults is None or not ExportValidationDefaults.ENABLED:
            return True, "Validation disabled"

        game_count = state.games_since_last_export

        # Bootstrap mode: allow export with lower thresholds for initial training
        total_exported = state.total_exported_samples
        if total_exported < ExportValidationDefaults.BOOTSTRAP_MODE_THRESHOLD:
            logger.debug(
                f"[AutoExportDaemon] {config_key}: Bootstrap mode "
                f"({total_exported} < {ExportValidationDefaults.BOOTSTRAP_MODE_THRESHOLD}), "
                f"skipping quality validation"
            )
            return True, "Bootstrap mode - validation skipped"

        # Check minimum games
        meets_game_threshold = game_count >= ExportValidationDefaults.MIN_GAMES

        # Check quality (attempt to get quality score from databases)
        avg_quality = await self._estimate_data_quality(config_key)
        meets_quality_threshold = avg_quality >= ExportValidationDefaults.MIN_AVG_QUALITY

        if ExportValidationDefaults.REQUIRE_BOTH:
            # Require both thresholds
            if not meets_game_threshold or not meets_quality_threshold:
                reason = (
                    f"Validation failed: games={game_count} "
                    f"(need {ExportValidationDefaults.MIN_GAMES}), "
                    f"quality={avg_quality:.2f} "
                    f"(need {ExportValidationDefaults.MIN_AVG_QUALITY})"
                )
                logger.warning(f"[AutoExportDaemon] {config_key}: {reason}")
                return False, reason
        else:
            # Require either threshold
            if not meets_game_threshold and not meets_quality_threshold:
                reason = (
                    f"Validation failed: need games>={ExportValidationDefaults.MIN_GAMES} "
                    f"OR quality>={ExportValidationDefaults.MIN_AVG_QUALITY}, "
                    f"got games={game_count}, quality={avg_quality:.2f}"
                )
                logger.warning(f"[AutoExportDaemon] {config_key}: {reason}")
                return False, reason

        return True, f"Validation passed: games={game_count}, quality={avg_quality:.2f}"

    async def _estimate_data_quality(self, config_key: str) -> float:
        """Estimate data quality for a configuration.

        Sprint 4 (Jan 2, 2026): Calculate average quality score from databases.

        Args:
            config_key: Configuration key

        Returns:
            Estimated quality score (0.0-1.0)
        """
        try:
            from app.utils.game_discovery import GameDiscovery

            discovery = GameDiscovery()
            parsed = parse_config_key(config_key)
            if not parsed:
                return 0.5  # Default if can't parse

            # Find databases for this config
            databases = discovery.find_databases_for_config(
                board_type=parsed.board_type,
                num_players=parsed.num_players,
            )

            if not databases:
                return 0.5  # Default if no databases

            # Estimate quality based on available metrics
            total_games = sum(db.game_count for db in databases)
            total_samples = sum(db.sample_count for db in databases if db.sample_count)

            if total_games == 0:
                return 0.5

            # Quality heuristics:
            # 1. Samples per game (higher is better - more moves per game)
            samples_per_game = total_samples / total_games if total_samples else 20
            quality_from_length = min(samples_per_game / 50.0, 1.0)  # 50 moves = max quality

            # 2. Assume gauntlet/tournament games have higher quality
            # This would need actual metadata tracking in the future
            quality_estimate = quality_from_length

            return quality_estimate

        except ImportError:
            return 0.5  # Default if GameDiscovery not available
        except Exception as e:
            logger.debug(f"[AutoExportDaemon] Quality estimation error: {e}")
            return 0.5

    async def _maybe_trigger_export(self, config_key: str) -> None:
        """Check conditions and trigger export if appropriate."""
        state = self._export_states.get(config_key)
        if not state:
            return

        # Phase 3: Check if sync is still pending for this config
        # Export is gated on DATA_SYNC_COMPLETED to prevent race conditions
        # January 3, 2026: Only apply sync-gating if gate_export_on_sync is enabled.
        if self.config.gate_export_on_sync and config_key in self._pending_sync_configs:
            logger.debug(
                f"[AutoExportDaemon] {config_key}: Export deferred - waiting for sync completion "
                f"(games pending: {state.games_since_last_export})"
            )
            return

        # Check if already exporting
        if state.export_in_progress:
            logger.debug(f"[AutoExportDaemon] {config_key}: Export already in progress")
            return

        # Check game threshold
        if state.games_since_last_export < self.config.min_games_threshold:
            return

        # Check cooldown
        time_since_last = time.time() - state.last_export_time
        if time_since_last < self.config.export_cooldown_seconds:
            remaining = self.config.export_cooldown_seconds - time_since_last
            logger.debug(
                f"[AutoExportDaemon] {config_key}: Cooldown active, "
                f"{remaining:.0f}s remaining"
            )
            return

        # Sprint 4 (Jan 2, 2026): Validate export readiness
        is_ready, reason = await self._validate_export_readiness(config_key, state)
        if not is_ready:
            logger.debug(f"[AutoExportDaemon] {config_key}: {reason}")
            # Emit validation failed event for monitoring
            self._emit_validation_failed_event(config_key, reason)
            return

        # Trigger export
        safe_create_task(
            self._run_export(config_key),
            name=f"export_{config_key}",
        )

    async def _run_export(self, config_key: str) -> bool:
        """Run the export subprocess for a configuration."""
        state = self._export_states.get(config_key)
        if not state:
            return False

        async with self._export_semaphore:
            # Feb 2026: Cross-process export coordination via SQLite.
            # The semaphore above is in-process only (doesn't limit master_loop/P2P).
            # ExportCoordinator uses shared SQLite to limit across all processes.
            try:
                from app.coordination.export_coordinator import get_export_coordinator
                _coord = get_export_coordinator()
                if not _coord.try_acquire(config_key):
                    logger.info(
                        f"[AutoExportDaemon] Skipping export for {config_key}: "
                        "cross-process export slot unavailable"
                    )
                    return False
                _release_export_slot = True
            except Exception:
                _release_export_slot = False

            state.export_in_progress = True

            try:
                # February 2026: Block export when coordinator is low on RAM/disk
                from app.utils.resource_guard import coordinator_resource_gate
                if not coordinator_resource_gate("NPZ_EXPORT"):
                    logger.info(
                        f"[AutoExportDaemon] Skipping export for {config_key}: "
                        "coordinator resource gate blocked (low RAM or disk)"
                    )
                    return False

                # Sprint 8 (Jan 2, 2026): Validate export readiness before starting
                valid, validation_msg = await self._validate_export_readiness(config_key, state)
                if not valid:
                    logger.info(
                        f"[AutoExportDaemon] Skipping export for {config_key}: {validation_msg}"
                    )
                    return False

                logger.info(
                    f"[AutoExportDaemon] Starting export for {config_key} "
                    f"({state.games_since_last_export} games pending)"
                )

                # Emit NPZ_EXPORT_STARTED event
                await self._emit_export_started(config_key, state.games_since_last_export)

                # Build export command
                base_dir = Path(__file__).resolve().parent.parent.parent
                script_path = base_dir / "scripts" / "export_replay_dataset.py"
                output_path = self.config.output_dir / f"{config_key}.npz"

                cmd = [
                    sys.executable,
                    str(script_path),
                    "--use-discovery",
                    "--board-type", state.board_type,
                    "--num-players", str(state.num_players),
                    "--output", str(output_path),
                    "--allow-noncanonical",  # Allow any database, not just registry
                    "--allow-pending-gate",  # Feb 2026: Allow pending_gate DBs (cluster nodes lack Node.js)
                    "--no-strict",  # Feb 28, 2026: Don't block entire export for a few bad games.
                    # --min-moves already filters short games during extraction.
                    # Strict mode was blocking exports when gauntlet DBs had 3/58
                    # games with <5 moves, preventing 97% of valid data from exporting.
                ]

                min_elo = self._get_min_elo_for_config(config_key)
                if min_elo is not None:
                    cmd.extend(["--min-elo", str(min_elo)])

                if self.config.use_incremental_export:
                    cmd.append("--use-cache")

                if self.config.require_completed_games:
                    cmd.append("--require-completed")

                if self.config.min_moves > 0:
                    cmd.extend(["--min-moves", str(self.config.min_moves)])

                # December 30, 2025: Include gauntlet/tournament games by default
                # These games have higher quality (longer time controls, stronger opponents)
                if self.config.include_gauntlet:
                    cmd.append("--include-gauntlet")

                if self.config.include_tournaments:
                    cmd.append("--include-tournaments")

                # Feb 26, 2026: Let export script auto-detect encoder version from
                # the canonical model on disk. Previously forced --encoder-version v3
                # based on PREFERRED_ARCHITECTURE, but this breaks configs where the
                # actual canonical model is still v2 (e.g., hexagonal). The export
                # script's auto-detection is correct and sufficient.
                # --include-heuristics is safe to add unconditionally since the script
                # handles it based on the actual encoder version.
                cmd.append("--include-heuristics")

                # February 2026: Cap per-export workers on coordinator to prevent
                # multiprocessing fan-out OOM. Default is os.cpu_count()-1 which
                # spawns 15 workers per export on a 16-core Mac Studio.
                workers = self.config.max_export_workers
                if workers is None:
                    from app.config.env import env
                    if env.is_coordinator:
                        workers = 2  # Coordinator: minimal workers (I/O-bound anyway)
                if workers is not None:
                    cmd.extend(["--workers", str(workers)])

                # Run export subprocess
                start_time = time.time()
                # December 29, 2025: Add RINGRIFT_ALLOW_PENDING_GATE to bypass parity
                # validation on cluster nodes that lack Node.js/npx
                export_env = {
                    **os.environ,
                    "PYTHONPATH": str(base_dir),
                    "RINGRIFT_ALLOW_PENDING_GATE": "true",
                }
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(base_dir),
                    env=export_env,
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.config.export_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    logger.error(f"[AutoExportDaemon] Export timed out for {config_key}")
                    state.consecutive_failures += 1
                    # Dec 30, 2025: Wrap blocking SQLite I/O with asyncio.to_thread()
                    await asyncio.to_thread(self._save_state, config_key)  # Persist failure count (Phase 8)
                    return False

                duration = time.time() - start_time

                if process.returncode == 0:
                    # Feb 2026: Validate NPZ after export subprocess succeeds.
                    # The subprocess may return 0 but write a corrupt file (e.g., disk full
                    # during np.savez_compressed before atomic rename was added).
                    try:
                        from app.coordination.npz_validation import quick_npz_check
                        _ok, _err = quick_npz_check(output_path)
                        if not _ok:
                            logger.error(
                                f"[AutoExportDaemon] Export produced corrupt NPZ for "
                                f"{config_key}: {_err}"
                            )
                            Path(output_path).unlink(missing_ok=True)
                            state.consecutive_failures += 1
                            await asyncio.to_thread(self._save_state, config_key)
                            return False
                    except ImportError:
                        pass  # Validation module not available

                    # Success
                    state.last_export_time = time.time()
                    state.last_export_games = state.games_since_last_export
                    state.games_since_last_export = 0
                    state.consecutive_failures = 0

                    # Parse sample count from output if available
                    samples = self._parse_sample_count(stdout.decode())
                    if samples:
                        state.total_exported_samples = samples

                    logger.info(
                        f"[AutoExportDaemon] Export complete for {config_key}: "
                        f"{state.last_export_games} games, {samples or '?'} samples, "
                        f"{duration:.1f}s"
                    )

                    # Emit completion event
                    await self._emit_export_complete(config_key, output_path, samples)

                    # December 2025: Emit NEW_GAMES_AVAILABLE for pipeline coordination
                    # This triggers downstream consumers (training, evaluation) immediately
                    await self._emit_new_games_available(config_key, samples)
                    return True

                else:
                    # Failure
                    state.consecutive_failures += 1
                    logger.error(
                        f"[AutoExportDaemon] Export failed for {config_key}: "
                        f"exit code {process.returncode}\n"
                        f"stderr: {stderr.decode()[:500]}"
                    )
                    return False

            except Exception as e:  # noqa: BLE001
                state.consecutive_failures += 1
                import traceback
                logger.error(
                    f"[AutoExportDaemon] Export error for {config_key}: {e}\n"
                    f"{traceback.format_exc()}"
                )
                return False

            finally:
                state.export_in_progress = False
                # Feb 2026: Release cross-process export slot
                if _release_export_slot:
                    try:
                        _coord.release(config_key)
                    except Exception:
                        pass
                # Persist updated state (Phase 8)
                # Dec 30, 2025: Wrap blocking SQLite I/O with asyncio.to_thread()
                await asyncio.to_thread(self._save_state, config_key)

    def _parse_sample_count(self, output: str) -> int | None:
        """Parse sample count from export script output."""
        import re

        # Look for patterns like "Exported 12345 samples" or "samples: 12345"
        patterns = [
            r"Exported\s+(\d+)\s+samples",
            r"samples:\s*(\d+)",
            r"Total samples:\s*(\d+)",
            r"(\d+)\s+samples",
        ]

        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    async def _validate_export_readiness(
        self, config_key: str, state: ConfigExportState
    ) -> tuple[bool, str]:
        """Validate that export data meets quality thresholds.

        Sprint 8 (Jan 2, 2026): Pre-export validation to prevent low-quality
        training data from entering the pipeline.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            state: Export state for this configuration

        Returns:
            Tuple of (valid, message). If not valid, message explains why.
        """
        # Check if validation is enabled
        if ExportValidationDefaults is None:
            return True, "Validation not available (fallback mode)"

        if not ExportValidationDefaults.ENABLED:
            return True, "Validation disabled"

        games_pending = state.games_since_last_export

        # Bootstrap mode: Skip validation for configs with few total games
        if state.total_exported_samples < ExportValidationDefaults.BOOTSTRAP_MODE_THRESHOLD:
            logger.debug(
                f"[AutoExportDaemon] Bootstrap mode for {config_key}: "
                f"{state.total_exported_samples} < {ExportValidationDefaults.BOOTSTRAP_MODE_THRESHOLD}"
            )
            return True, f"Bootstrap mode (samples: {state.total_exported_samples})"

        # Check minimum games
        if games_pending < ExportValidationDefaults.MIN_GAMES:
            msg = (
                f"Insufficient games: {games_pending} < {ExportValidationDefaults.MIN_GAMES}"
            )
            logger.warning(f"[AutoExportDaemon] Export validation failed for {config_key}: {msg}")
            await self._emit_export_validation_failed(config_key, msg)
            return False, msg

        # Note: Quality scoring would require scanning games - expensive.
        # For now, trust that completed games meeting min_moves threshold are sufficient.
        # Future: Add lightweight quality sampling here.

        return True, "Validation passed"

    async def _emit_export_validation_failed(
        self, config_key: str, reason: str
    ) -> None:
        """Emit EXPORT_VALIDATION_FAILED event.

        Sprint 8 (Jan 2, 2026): Notify subscribers that export was blocked.
        """
        from app.coordination.event_emission_helpers import safe_emit_event

        safe_emit_event(
            "EXPORT_VALIDATION_FAILED",
            {
                "config_key": config_key,
                "reason": reason,
                "timestamp": time.time(),
            },
            context="AutoExportDaemon",
        )

    async def _emit_export_started(
        self, config_key: str, games_pending: int
    ) -> None:
        """Emit NPZ_EXPORT_STARTED event."""
        try:
            from app.coordination.event_router import (
                StageCompletionResult,
                StageEvent,
                get_stage_event_bus,
            )

            # Dec 30, 2025: Use consolidated parse_config_key utility
            parsed = parse_config_key(config_key)
            board_type = parsed.board_type if parsed else config_key
            num_players = parsed.num_players if parsed else 2

            bus = get_stage_event_bus()
            await bus.emit(
                StageCompletionResult(
                    event=StageEvent.NPZ_EXPORT_STARTED,
                    success=True,
                    iteration=0,  # Export is not iteration-based
                    timestamp=datetime.datetime.now().isoformat(),
                    board_type=board_type,
                    num_players=num_players,
                    metadata={
                        "config": config_key,
                        "games_pending": games_pending,
                    },
                )
            )
            logger.debug(f"[AutoExportDaemon] Emitted NPZ_EXPORT_STARTED for {config_key}")

        except Exception as e:  # noqa: BLE001
            logger.warning(f"[AutoExportDaemon] Failed to emit export started event: {e}")

    async def _emit_export_complete(
        self, config_key: str, output_path: Path, samples: int | None
    ) -> None:
        """Emit NPZ_EXPORT_COMPLETE event."""
        try:
            from app.coordination.event_router import (
                StageCompletionResult,
                StageEvent,
                get_stage_event_bus,
            )

            state = self._export_states.get(config_key)
            # Dec 30, 2025: Use consolidated parse_config_key utility
            parsed = parse_config_key(config_key)
            board_type = parsed.board_type if parsed else config_key
            num_players = parsed.num_players if parsed else 2

            bus = get_stage_event_bus()
            await bus.emit(
                StageCompletionResult(
                    event=StageEvent.NPZ_EXPORT_COMPLETE,
                    success=True,
                    iteration=0,  # Export is not iteration-based
                    timestamp=datetime.datetime.now().isoformat(),
                    board_type=board_type,
                    num_players=num_players,
                    metadata={
                        "config": config_key,
                        "output_path": str(output_path),
                        "samples": samples,
                        "games_exported": state.last_export_games if state else 0,
                    },
                )
            )
            logger.debug(f"[AutoExportDaemon] Emitted NPZ_EXPORT_COMPLETE for {config_key}")

        except Exception as e:  # noqa: BLE001
            logger.warning(f"[AutoExportDaemon] Failed to emit export complete event: {e}")

    async def _emit_new_games_available(
        self, config_key: str, samples: int | None
    ) -> None:
        """Emit NEW_GAMES_AVAILABLE event for pipeline coordination.

        December 2025: Part of 48-hour autonomous operation optimization.
        This event signals to downstream consumers (training, evaluation)
        that new training data is immediately available, enabling reactive
        dispatch instead of polling-based detection.
        """
        try:
            from app.distributed.data_events import DataEvent, DataEventType
            from app.coordination.event_router import get_event_bus

            state = self._export_states.get(config_key)
            # Dec 30, 2025: Use consolidated parse_config_key utility
            parsed = parse_config_key(config_key)
            board_type = parsed.board_type if parsed else config_key
            num_players = parsed.num_players if parsed else 2

            event = DataEvent(
                event_type=DataEventType.NEW_GAMES_AVAILABLE,
                payload={
                    "config_key": config_key,
                    "board_type": board_type,
                    "num_players": num_players,
                    "samples": samples,
                    "games_exported": state.last_export_games if state else 0,
                    "source": "auto_export_daemon",
                    "trigger": "event_driven_batch",
                },
                source="AutoExportDaemon",
            )

            bus = get_event_bus()
            await bus.publish(event)
            logger.debug(
                f"[AutoExportDaemon] Emitted NEW_GAMES_AVAILABLE for {config_key} "
                f"({samples or 0} samples)"
            )

        except Exception as e:  # noqa: BLE001
            logger.warning(f"[AutoExportDaemon] Failed to emit new games available event: {e}")

    def _emit_validation_failed_event(self, config_key: str, reason: str) -> None:
        """Emit EXPORT_VALIDATION_FAILED event for monitoring.

        Sprint 4 (Jan 2, 2026): Track when exports are blocked by validation.
        """
        from app.coordination.event_emission_helpers import safe_emit_event

        parsed = parse_config_key(config_key)
        board_type = parsed.board_type if parsed else config_key
        num_players = parsed.num_players if parsed else 2

        safe_emit_event(
            "EXPORT_VALIDATION_FAILED",
            {
                "config_key": config_key,
                "board_type": board_type,
                "num_players": num_players,
                "reason": reason,
                "timestamp": time.time(),
                "source": "auto_export_daemon",
            },
            context="AutoExportDaemon",
            log_after=f"Emitted EXPORT_VALIDATION_FAILED for {config_key}",
        )

    async def _run_cycle(self) -> None:
        """Main work loop iteration - called by HandlerBase at 5 minute intervals."""
        # Skip if we're on a coordinator node
        if self._coordinator_skip:
            return

        # Dec 31, 2025: Clear stale pending configs (fallback for missed sync events)
        await self._clear_stale_pending_configs()

        # Scan for databases that may need export
        await self._scan_for_pending_exports()

    async def _clear_stale_pending_configs(self) -> None:
        """Clear pending sync configs that have waited too long.

        Dec 31, 2025: Fallback mechanism for 48-hour autonomous operation.
        If a sync event is missed or delayed beyond max_pending_time, the config
        is cleared from pending and export can proceed. This prevents permanent
        deadlocks while still allowing sync coordination when it works.
        """
        now = time.time()
        stale_configs = []

        for config_key in list(self._pending_sync_configs):
            pending_since = self._pending_sync_times.get(config_key, 0)
            if pending_since > 0 and (now - pending_since) > self._max_pending_time:
                stale_configs.append(config_key)

        if stale_configs:
            logger.warning(
                f"[AutoExportDaemon] Clearing {len(stale_configs)} stale pending configs "
                f"(waited >{self._max_pending_time}s): {stale_configs}"
            )
            for config_key in stale_configs:
                self._pending_sync_configs.discard(config_key)
                self._pending_sync_times.pop(config_key, None)
                # Trigger export check
                await self._maybe_trigger_export(config_key)

    async def _scan_for_pending_exports(self) -> None:
        """Scan databases to find configs needing export."""
        try:
            from app.utils.game_discovery import GameDiscovery

            discovery = GameDiscovery()
            databases = discovery.find_all_databases()

            for db_info in databases:
                if not db_info.board_type or not db_info.num_players:
                    continue

                config_key = f"{db_info.board_type}_{db_info.num_players}p"

                if config_key not in self._export_states:
                    self._export_states[config_key] = ConfigExportState(
                        config_key=config_key,
                        board_type=db_info.board_type,
                        num_players=db_info.num_players,
                        games_since_last_export=db_info.game_count,
                    )
                else:
                    # Update game count from discovery
                    state = self._export_states[config_key]
                    # Only update if discovery shows more games
                    if db_info.game_count > state.games_since_last_export:
                        state.games_since_last_export = db_info.game_count

                # Check if export needed
                await self._maybe_trigger_export(config_key)

        except ImportError:
            logger.debug("[AutoExportDaemon] GameDiscovery not available")
        except Exception as e:  # noqa: BLE001
            logger.error(f"[AutoExportDaemon] Error scanning databases: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get current daemon status."""
        return {
            "running": self._running,
            "configs_tracked": len(self._export_states),
            "states": {
                key: {
                    "games_pending": state.games_since_last_export,
                    "last_export": state.last_export_time,
                    "last_export_games": state.last_export_games,
                    "total_samples": state.total_exported_samples,
                    "in_progress": state.export_in_progress,
                    "failures": state.consecutive_failures,
                }
                for key, state in self._export_states.items()
            },
        }

    def health_check(self) -> HealthCheckResult:
        """Check daemon health (December 2025: CoordinatorProtocol compliance).

        Overrides HandlerBase.health_check() with export-specific logic.

        Returns:
            HealthCheckResult with status and details
        """
        from app.coordination.contracts import CoordinatorStatus

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="AutoExport daemon not running",
            )

        # Check for high failure rate
        total_failures = sum(s.consecutive_failures for s in self._export_states.values())
        if total_failures > 5:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"AutoExport daemon has {total_failures} consecutive failures",
                details=self.get_status(),
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"AutoExport daemon running ({len(self._export_states)} configs tracked)",
            details=self.get_status(),
        )


# December 2025: Using HandlerBase singleton pattern
def get_auto_export_daemon() -> AutoExportDaemon:
    """Get or create the singleton auto export daemon.

    December 2025: Now uses HandlerBase.get_instance() singleton pattern.
    """
    return AutoExportDaemon.get_instance()


def reset_auto_export_daemon() -> None:
    """Reset the singleton instance (for testing).

    December 2025: Added for test isolation.
    """
    AutoExportDaemon.reset_instance()


async def start_auto_export_daemon() -> AutoExportDaemon:
    """Start the auto export daemon (convenience function)."""
    daemon = get_auto_export_daemon()
    await daemon.start()
    return daemon


__all__ = [
    "AutoExportConfig",
    "AutoExportDaemon",
    "BatchAccumulator",
    "ConfigExportState",
    "get_auto_export_daemon",
    "reset_auto_export_daemon",
    "start_auto_export_daemon",
]

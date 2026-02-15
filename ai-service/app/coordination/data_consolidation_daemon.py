"""DataConsolidationDaemon - Consolidates scattered selfplay games into canonical databases.

This daemon solves the critical gap in the training pipeline where selfplay games
are generated across 30+ cluster nodes but never consolidated into canonical databases.

Event-Driven Flow:
    SELFPLAY_COMPLETE → DataConsolidationDaemon._on_selfplay_complete()
        ↓ (triggers consolidation for that config)
    Merge games into canonical_{board}_{n}p.db
        ↓ (deduplicate by game_id)
    CONSOLIDATION_COMPLETE → DataPipelineOrchestrator._on_consolidation_complete()
        ↓ (now safe to export from consolidated database)
    Downstream NPZ export, training, etc.

December 2025: Created as part of training data pipeline remediation.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.event_utils import make_config_key, parse_config_key
from app.coordination.event_emission_helpers import safe_emit_event
from app.coordination.health_check_helper import HealthCheckHelper
from app.coordination.contracts import CoordinatorStatus
from app.config.thresholds import SQLITE_TIMEOUT, SQLITE_CONNECT_TIMEOUT
from app.utils.sqlite_utils import connect_safe
from app.db.game_replay import SCHEMA_VERSION

logger = logging.getLogger(__name__)

__all__ = [
    "DataConsolidationDaemon",
    "ConsolidationConfig",
    "ConsolidationStats",
    "get_consolidation_daemon",
    "reset_consolidation_daemon",
]

@dataclass
class ConsolidationConfig:
    """Configuration for the consolidation daemon."""

    # Daemon control
    enabled: bool = True
    check_interval_seconds: int = 300

    # Base paths
    data_dir: Path = field(default_factory=lambda: Path("data/games"))
    canonical_dir: Path = field(default_factory=lambda: Path("data/games"))

    # Consolidation thresholds
    min_games_for_consolidation: int = 50  # Minimum new games before consolidating

    # Database settings - use central validator's constant for consistency
    @property
    def min_moves_for_valid(self) -> int:
        """Minimum moves for a valid game (from central MoveDataValidator)."""
        from app.db.move_data_validator import MIN_MOVES_REQUIRED
        return MIN_MOVES_REQUIRED

    # Note: min_moves_for_valid is now a property, default was 5
    batch_size: int = 100  # Games per batch during merge

    # Safety
    deduplicate: bool = True  # Deduplicate by game_id
    validate_before_merge: bool = True  # Check game validity before merging

    # Concurrency (December 2025: Parallelization optimization)
    max_concurrent_consolidations: int = 3  # Max parallel config consolidations

    @classmethod
    def from_env(cls) -> "ConsolidationConfig":
        """Load configuration from environment variables."""
        import os

        return cls(
            enabled=os.getenv("RINGRIFT_CONSOLIDATION_ENABLED", "true").lower() == "true",
            check_interval_seconds=int(os.getenv("RINGRIFT_CONSOLIDATION_INTERVAL", "300")),
            data_dir=Path(os.getenv("RINGRIFT_DATA_DIR", "data/games")),
            canonical_dir=Path(os.getenv("RINGRIFT_CANONICAL_DIR", "data/games")),
            min_games_for_consolidation=int(os.getenv("RINGRIFT_CONSOLIDATION_MIN_GAMES", "50")),
            max_concurrent_consolidations=int(os.getenv("RINGRIFT_CONSOLIDATION_MAX_CONCURRENT", "3")),
        )


@dataclass
class ConsolidationStats:
    """Statistics for a consolidation operation."""
    config_key: str = ""
    source_dbs_scanned: int = 0
    games_scanned: int = 0
    games_valid: int = 0
    games_merged: int = 0
    games_duplicate: int = 0
    games_invalid: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    success: bool = False
    error: str | None = None

    @property
    def duration_seconds(self) -> float:
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time


# All supported configurations
ALL_CONFIGS = [
    ("hex8", 2), ("hex8", 3), ("hex8", 4),
    ("square8", 2), ("square8", 3), ("square8", 4),
    ("square19", 2), ("square19", 3), ("square19", 4),
    ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
]


class DataConsolidationDaemon(HandlerBase):
    """Daemon that consolidates scattered selfplay games into canonical databases.

    Subscribes to:
    - NEW_GAMES_AVAILABLE: New games generated (from selfplay)
    - SELFPLAY_COMPLETE: Selfplay batch finished

    Emits:
    - CONSOLIDATION_STARTED: Beginning consolidation for a config
    - CONSOLIDATION_COMPLETE: Consolidation finished (triggers export)
    """

    def __init__(self, config: ConsolidationConfig | None = None):
        """Initialize the consolidation daemon.

        Args:
            config: Configuration for consolidation behavior
        """
        self._daemon_config = config or ConsolidationConfig.from_env()
        super().__init__(
            name="DataConsolidationDaemon",
            config=self._daemon_config,
            cycle_interval=float(self._daemon_config.check_interval_seconds),
        )

        # State tracking
        self._pending_configs: set[str] = set()  # Configs needing consolidation
        self._last_consolidation: dict[str, float] = {}  # config_key -> timestamp
        self._stats_history: list[ConsolidationStats] = []
        self._subscribed = False

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        # Concurrency control (December 2025: Parallelization optimization)
        self._consolidation_semaphore = asyncio.Semaphore(
            self._daemon_config.max_concurrent_consolidations
        )

    @property
    def config(self) -> ConsolidationConfig:
        """Return daemon configuration."""
        return self._daemon_config

    async def _on_start(self) -> None:
        """Called after daemon starts - subscribe to events and trigger initial scan.

        CRITICAL: The initial consolidation scan ensures that games generated
        before this daemon started are consolidated. Without this, canonical
        databases remain empty until NEW_GAMES_AVAILABLE or SELFPLAY_COMPLETE
        events arrive, which may never happen for historical data.

        Dec 30, 2025: Added eligibility check (Phase 3.5 of distributed data pipeline).
        Consolidation only runs on eligible nodes based on disk space, role, etc.
        """
        # Phase 3.5: Check eligibility before running consolidation
        if not await self._check_node_eligibility():
            logger.info("[DataConsolidationDaemon] Node not eligible for consolidation, skipping")
            return

        await self._subscribe_to_events()

        # CRITICAL: Trigger initial consolidation for all configs
        # This ensures historical data is consolidated on daemon startup
        logger.info("[DataConsolidationDaemon] Starting initial consolidation scan...")
        try:
            results = await self.trigger_all_consolidations()
            total_merged = sum(s.games_merged for s in results.values())
            logger.info(
                f"[DataConsolidationDaemon] Initial consolidation complete: "
                f"{total_merged} games merged across {len(results)} configs"
            )
        except Exception as e:
            logger.warning(f"[DataConsolidationDaemon] Initial consolidation failed: {e}")

    async def _on_stop(self) -> None:
        """Called before daemon stops - unsubscribe from events."""
        await self._unsubscribe_from_events()

    async def _run_cycle(self) -> None:
        """Run one consolidation cycle.

        This is called by BaseDaemon's _protected_loop() with proper
        error handling and interval sleeping.
        """
        # Feb 2026: Skip consolidation on coordinator nodes to prevent
        # local disk from filling with canonical_*.db files (100GB+)
        try:
            from app.config.env import env
            if not env.consolidation_enabled:
                logger.debug("Consolidation disabled on this node (coordinator mode)")
                return
        except ImportError:
            pass

        # February 2026: Block when coordinator is low on RAM/disk
        from app.utils.resource_guard import coordinator_resource_gate
        if not coordinator_resource_gate("DATA_CONSOLIDATION"):
            return

        await self._process_pending_consolidations()

    async def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events."""
        if self._subscribed:
            return

        try:
            from app.coordination.event_router import get_event_bus
            from app.distributed.data_events import DataEventType

            bus = get_event_bus()

            # Subscribe to events that indicate new games are available
            bus.subscribe(DataEventType.NEW_GAMES_AVAILABLE, self._on_new_games_available)
            bus.subscribe(DataEventType.SELFPLAY_COMPLETE, self._on_selfplay_complete)

            self._subscribed = True
            logger.info("[DataConsolidationDaemon] Subscribed to NEW_GAMES_AVAILABLE, SELFPLAY_COMPLETE")

        except ImportError as e:
            logger.warning(f"[DataConsolidationDaemon] Could not subscribe to events: {e}")

    async def _unsubscribe_from_events(self) -> None:
        """Unsubscribe from events on shutdown."""
        if not self._subscribed:
            return

        try:
            from app.coordination.event_router import get_event_bus
            from app.distributed.data_events import DataEventType

            bus = get_event_bus()
            bus.unsubscribe(DataEventType.NEW_GAMES_AVAILABLE, self._on_new_games_available)
            bus.unsubscribe(DataEventType.SELFPLAY_COMPLETE, self._on_selfplay_complete)

            self._subscribed = False

        except Exception as e:
            logger.warning(f"[DataConsolidationDaemon] Error unsubscribing from events: {e}")

    async def _check_node_eligibility(self) -> bool:
        """Check if this node is eligible to run consolidation.

        Dec 30, 2025: Phase 3.5 of distributed data pipeline architecture.

        Eligibility is based on:
        - Node role (coordinator, gpu_training, backbone, etc.)
        - Available disk space
        - Node-specific consolidation_enabled setting

        Returns:
            True if this node should run consolidation, False otherwise
        """
        try:
            from app.coordination.consolidation_eligibility import get_eligibility_manager
            from app.config.env import env

            manager = get_eligibility_manager()
            node_id = env.node_id

            is_eligible, reason = manager.is_node_eligible(node_id)

            if is_eligible:
                logger.info(f"[DataConsolidationDaemon] Node {node_id} eligible: {reason}")
            else:
                logger.info(f"[DataConsolidationDaemon] Node {node_id} not eligible: {reason}")

            return is_eligible

        except ImportError as e:
            # ConsolidationEligibilityManager not available - allow consolidation
            logger.debug(f"[DataConsolidationDaemon] Eligibility check unavailable: {e}")
            return True
        except Exception as e:
            # Error checking eligibility - allow consolidation as fallback
            logger.warning(f"[DataConsolidationDaemon] Eligibility check error: {e}")
            return True

    def _on_new_games_available(self, event: Any) -> None:
        """Handle NEW_GAMES_AVAILABLE event.

        Payload schema (standardized December 2025):
        - new_games: int - Number of new games available (canonical key)
        - config_key: str - Configuration key (e.g., "hex8_2p")
        - host/source: str - Source of the games

        Backward-compatible aliases: games_added, games_count, count, game_count
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", "")
            # Read with fallback chain for backward compatibility (canonical: new_games)
            games_added = payload.get(
                "new_games",
                payload.get("games_added", payload.get("games_count", payload.get("count", 0)))
            )

            if config_key and games_added > 0:
                self._pending_configs.add(config_key)
                logger.debug(
                    f"[DataConsolidationDaemon] Queued {config_key} for consolidation "
                    f"({games_added} new games)"
                )
        except Exception as e:
            logger.debug(f"[DataConsolidationDaemon] Error handling NEW_GAMES_AVAILABLE: {e}")

    def _on_selfplay_complete(self, event: Any) -> None:
        """Handle SELFPLAY_COMPLETE event."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", payload.get("config", ""))
            board_type = payload.get("board_type", "")
            num_players = payload.get("num_players", 0)

            if not config_key and board_type and num_players:
                config_key = make_config_key(board_type, num_players)

            if config_key:
                self._pending_configs.add(config_key)
                logger.debug(f"[DataConsolidationDaemon] Queued {config_key} for consolidation")

        except Exception as e:
            logger.debug(f"[DataConsolidationDaemon] Error handling SELFPLAY_COMPLETE: {e}")

    async def _process_pending_consolidations(self) -> None:
        """Process all pending consolidations in parallel.

        December 2025: Parallelized for 2-3x speedup on multi-config consolidation.
        Uses semaphore to limit concurrent consolidations (default: 3).
        """
        # February 2026: Block when coordinator is low on RAM/disk
        from app.utils.resource_guard import coordinator_resource_gate
        if not coordinator_resource_gate("DATA_CONSOLIDATION"):
            return

        async with self._lock:
            if not self._pending_configs:
                return

            # Copy and clear pending set
            configs_to_process = list(self._pending_configs)
            self._pending_configs.clear()

        # Filter configs that pass validation and cooldown check
        valid_configs: list[tuple[str, str, int]] = []  # (config_key, board_type, num_players)
        for config_key in configs_to_process:
            # Parse config key using canonical utility
            parsed = parse_config_key(config_key)
            if not parsed:
                continue
            board_type = parsed.board_type
            num_players = parsed.num_players

            # Check if enough time has passed since last consolidation
            last_time = self._last_consolidation.get(config_key, 0)
            if time.time() - last_time < 60.0:  # Minimum 1 minute between consolidations
                continue

            valid_configs.append((config_key, board_type, num_players))

        if not valid_configs:
            return

        # Process configs in parallel with semaphore limiting concurrency
        async def consolidate_with_semaphore(
            config_key: str, board_type: str, num_players: int
        ) -> ConsolidationStats | None:
            """Consolidate a single config with semaphore protection."""
            async with self._consolidation_semaphore:
                try:
                    stats = await self._consolidate_config(board_type, num_players)

                    if stats.success:
                        self._last_consolidation[config_key] = time.time()
                        self._stats_history.append(stats)

                        # Keep only recent history
                        if len(self._stats_history) > 100:
                            self._stats_history = self._stats_history[-100:]

                        logger.info(
                            f"[DataConsolidationDaemon] Consolidated {config_key}: "
                            f"{stats.games_merged} games merged, "
                            f"{stats.games_duplicate} duplicates, "
                            f"duration={stats.duration_seconds:.1f}s"
                        )
                    else:
                        logger.warning(
                            f"[DataConsolidationDaemon] Failed to consolidate {config_key}: {stats.error}"
                        )
                    return stats

                except Exception as e:
                    logger.error(f"[DataConsolidationDaemon] Error consolidating {config_key}: {e}")
                    return None

        # Run consolidations in parallel
        tasks = [
            consolidate_with_semaphore(config_key, board_type, num_players)
            for config_key, board_type, num_players in valid_configs
        ]

        if tasks:
            logger.debug(
                f"[DataConsolidationDaemon] Processing {len(tasks)} configs in parallel "
                f"(max concurrent: {self.config.max_concurrent_consolidations})"
            )
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _consolidate_config(
        self,
        board_type: str,
        num_players: int,
    ) -> ConsolidationStats:
        """Consolidate all games for a specific config into the canonical database.

        Args:
            board_type: Board type (hex8, square8, etc.)
            num_players: Number of players (2, 3, 4)

        Returns:
            ConsolidationStats with results
        """
        config_key = make_config_key(board_type, num_players)
        stats = ConsolidationStats(config_key=config_key, start_time=time.time())

        try:
            # Emit start event
            await self._emit_consolidation_started(config_key, board_type, num_players)

            # Find all source databases (via thread pool to avoid blocking)
            source_dbs = await asyncio.to_thread(
                self._find_source_databases, board_type, num_players
            )
            stats.source_dbs_scanned = len(source_dbs)

            if not source_dbs:
                stats.success = True
                stats.end_time = time.time()
                logger.debug(f"[DataConsolidationDaemon] No source databases found for {config_key}")
                return stats

            # Get or create canonical database (via thread pool to avoid blocking)
            canonical_db = self._get_canonical_db_path(board_type, num_players)
            await asyncio.to_thread(self._ensure_canonical_schema, canonical_db)

            # Get existing game IDs to avoid duplicates (via thread pool)
            existing_ids = await asyncio.to_thread(
                self._get_existing_game_ids, canonical_db
            )

            # Merge from each source database
            for source_db in source_dbs:
                try:
                    merge_stats = await self._merge_database(
                        source_db=source_db,
                        target_db=canonical_db,
                        board_type=board_type,
                        num_players=num_players,
                        existing_ids=existing_ids,
                    )

                    stats.games_scanned += merge_stats.get("scanned", 0)
                    stats.games_valid += merge_stats.get("valid", 0)
                    stats.games_merged += merge_stats.get("merged", 0)
                    stats.games_duplicate += merge_stats.get("duplicate", 0)
                    stats.games_invalid += merge_stats.get("invalid", 0)

                    # Update existing_ids with newly merged games
                    existing_ids.update(merge_stats.get("new_ids", set()))

                except Exception as e:
                    logger.warning(f"[DataConsolidationDaemon] Error merging {source_db}: {e}")

            stats.success = True
            stats.end_time = time.time()

            # Emit completion event
            await self._emit_consolidation_complete(
                config_key=config_key,
                board_type=board_type,
                num_players=num_players,
                canonical_db=str(canonical_db),
                games_merged=stats.games_merged,
                total_games=len(existing_ids),
            )

        except Exception as e:
            stats.error = str(e)
            stats.end_time = time.time()
            logger.error(f"[DataConsolidationDaemon] Consolidation failed for {config_key}: {e}")

        return stats

    def _find_source_databases(self, board_type: str, num_players: int) -> list[Path]:
        """Find all source databases containing games for a config."""
        source_dbs: set[Path] = set()  # Use set to avoid duplicates

        # Search patterns for selfplay, gauntlet, and tournament databases
        # NOTE: owc_imports and synced are critical - they contain 176K+ games (68GB)
        # that would otherwise never reach canonical databases for training
        # Jan 2026: Added gauntlet/tournament/evaluation directories for training data integration
        search_dirs = [
            self.config.data_dir,
            self.config.data_dir / "selfplay",
            self.config.data_dir / "p2p_gpu",
            self.config.data_dir / "owc_imports",  # OWC external drive archives
            self.config.data_dir / "synced",       # P2P synced databases
            self.config.data_dir / "gauntlet",     # Gauntlet evaluation games
            self.config.data_dir / "tournament",   # Tournament games
            self.config.data_dir / "evaluation",   # Model evaluation games
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            # Find all .db files
            for db_path in search_dir.glob("**/*.db"):
                # Skip canonical databases (we don't want to merge canonical into itself)
                if "canonical" in db_path.name.lower():
                    continue

                # Resolve to absolute path for deduplication
                resolved_path = db_path.resolve()

                # Check if this DB has games for our config
                if resolved_path not in source_dbs and self._has_games_for_config(db_path, board_type, num_players):
                    source_dbs.add(resolved_path)

        return list(source_dbs)

    def _has_games_for_config(self, db_path: Path, board_type: str, num_players: int) -> bool:
        """Check if a database has games for the specified config."""
        # December 27, 2025: Use context manager to prevent connection leaks
        try:
            with connect_safe(db_path, timeout=SQLITE_CONNECT_TIMEOUT, row_factory=None) as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM games
                    WHERE board_type = ? AND num_players = ?
                    LIMIT 1
                """, (board_type, num_players))
                count = cursor.fetchone()[0]
                return count > 0
        except (sqlite3.Error, OSError):
            return False

    def _get_canonical_db_path(self, board_type: str, num_players: int) -> Path:
        """Get the path to the canonical database for a config."""
        return self.config.canonical_dir / f"canonical_{board_type}_{num_players}p.db"

    def _get_existing_game_ids(self, db_path: Path) -> set[str]:
        """Get set of existing game IDs in a database."""
        if not db_path.exists():
            return set()

        # December 27, 2025: Use context manager to prevent connection leaks
        try:
            with connect_safe(db_path, timeout=SQLITE_TIMEOUT, row_factory=None) as conn:
                cursor = conn.execute("SELECT game_id FROM games")
                game_ids = {row[0] for row in cursor.fetchall()}
                return game_ids
        except sqlite3.Error as e:
            logger.warning(f"[DataConsolidationDaemon] Could not read game IDs from {db_path}: {e}")
            return set()

    def _ensure_canonical_schema(self, db_path: Path) -> None:
        """Ensure the canonical database has the correct schema."""
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # December 27, 2025: Use context manager to prevent connection leaks
        with connect_safe(db_path, timeout=SQLITE_TIMEOUT, row_factory=None) as conn:
            # Main games table - January 2026: Use canonical SCHEMA_VERSION
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS games (
                    game_id TEXT PRIMARY KEY,
                    board_type TEXT NOT NULL,
                    num_players INTEGER NOT NULL,
                    rng_seed INTEGER,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    game_status TEXT NOT NULL,
                    winner INTEGER,
                    termination_reason TEXT,
                    total_moves INTEGER NOT NULL,
                    total_turns INTEGER NOT NULL,
                    duration_ms INTEGER,
                    source TEXT,
                    schema_version INTEGER NOT NULL DEFAULT {SCHEMA_VERSION},
                    time_control_type TEXT DEFAULT 'none',
                    initial_time_ms INTEGER,
                    time_increment_ms INTEGER,
                    metadata_json TEXT,
                    quality_score REAL,
                    quality_category TEXT,
                    consolidated_at REAL
                )
            """)

            # Moves table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS game_moves (
                    game_id TEXT NOT NULL,
                    move_number INTEGER NOT NULL,
                    player INTEGER NOT NULL,
                    position_q INTEGER,
                    position_r INTEGER,
                    move_type TEXT,
                    move_probs TEXT,
                    PRIMARY KEY (game_id, move_number),
                    FOREIGN KEY (game_id) REFERENCES games(game_id)
                )
            """)

            # Initial state table (must use initial_state_json to match TypeScript)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS game_initial_state (
                    game_id TEXT PRIMARY KEY,
                    initial_state_json TEXT NOT NULL,
                    compressed INTEGER DEFAULT 0,
                    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
                )
            """)

            # State snapshots table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS game_state_snapshots (
                    game_id TEXT NOT NULL,
                    move_number INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    PRIMARY KEY (game_id, move_number),
                    FOREIGN KEY (game_id) REFERENCES games(game_id)
                )
            """)

            # Players table (must match TypeScript schema in SelfPlayGameService)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS game_players (
                    game_id TEXT NOT NULL,
                    player_number INTEGER NOT NULL,
                    player_type TEXT,
                    ai_type TEXT,
                    ai_difficulty INTEGER,
                    ai_profile_id TEXT,
                    final_eliminated_rings INTEGER,
                    final_territory_spaces INTEGER,
                    final_rings_in_hand INTEGER,
                    model_version TEXT,
                    PRIMARY KEY (game_id, player_number),
                    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
                )
            """)

            # Add consolidated_at column if not exists
            try:
                conn.execute("ALTER TABLE games ADD COLUMN consolidated_at REAL")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # December 29, 2025: Add indexes for query optimization
            # These indexes improve common query patterns:
            # - Filtering games by board config
            # - Looking up moves by game_id (FK performance)
            # - Querying by consolidation time
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_games_board_players
                ON games(board_type, num_players)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_games_consolidated_at
                ON games(consolidated_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_moves_game_id
                ON game_moves(game_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_game_id
                ON game_state_snapshots(game_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_players_game_id
                ON game_players(game_id)
            """)

            conn.commit()
            # Note: conn.close() not needed - context manager handles it

    def _merge_database_sync(
        self,
        source_db: Path,
        target_db: Path,
        board_type: str,
        num_players: int,
        existing_ids: set[str],
    ) -> dict[str, Any]:
        """Synchronous helper to merge games from source database into target.

        December 29, 2025: Extracted from async _merge_database() to run via
        asyncio.to_thread() and avoid blocking the event loop.

        Args:
            source_db: Source database path
            target_db: Target (canonical) database path
            board_type: Board type filter
            num_players: Player count filter
            existing_ids: Set of game IDs already in target

        Returns:
            Dict with merge statistics
        """
        stats = {
            "scanned": 0,
            "valid": 0,
            "merged": 0,
            "duplicate": 0,
            "invalid": 0,
            "new_ids": set(),
        }

        # December 27, 2025: Use try/finally to prevent connection leaks
        source_conn = None
        target_conn = None
        try:
            source_conn = connect_safe(source_db, timeout=SQLITE_TIMEOUT)
            target_conn = connect_safe(target_db, timeout=SQLITE_TIMEOUT, row_factory=None)

            # January 2026: Temporarily disable enforce_moves_on_insert trigger
            # The trigger requires moves to exist before game insert, but we need
            # to insert game first (FK constraint) then copy moves.
            target_conn.execute("DROP TRIGGER IF EXISTS enforce_moves_on_insert")

            # December 29, 2025: Get target database columns to filter source columns
            # This prevents INSERT errors when source has columns target doesn't have
            target_cursor = target_conn.execute("PRAGMA table_info(games)")
            target_columns = {row[1] for row in target_cursor.fetchall()}
            logger.debug(f"[DataConsolidationDaemon] Target columns: {target_columns}")

            # Query games for this config
            cursor = source_conn.execute("""
                SELECT * FROM games
                WHERE board_type = ? AND num_players = ?
                AND game_status IN ('completed', 'finished', 'victory')
            """, (board_type, num_players))

            # December 29, 2025: Fetch all rows first for batch move counting
            all_rows = cursor.fetchall()
            source_columns = [desc[0] for desc in cursor.description]

            # December 29, 2025: Filter to only columns that exist in target
            # Build mapping from source column index to value
            columns = [col for col in source_columns if col in target_columns]
            column_indices = [source_columns.index(col) for col in columns]
            logger.debug(f"[DataConsolidationDaemon] Using {len(columns)}/{len(source_columns)} columns")

            # December 29, 2025: Batch pre-fetch move counts to avoid N+1 queries
            # This reduces O(n) database round trips to O(1)
            move_counts: dict[str, int] = {}
            if self.config.validate_before_merge:
                candidate_ids = [
                    row["game_id"]
                    for row in all_rows
                    if row["game_id"] not in existing_ids
                    and (row["total_moves"] or 0) >= self.config.min_moves_for_valid
                ]
                move_counts = self._batch_count_moves(source_conn, candidate_ids)

            now = time.time()

            for row in all_rows:
                stats["scanned"] += 1
                game_id = row["game_id"]

                # Skip duplicates
                if game_id in existing_ids:
                    stats["duplicate"] += 1
                    continue

                # Validate game - check both reported and actual move counts
                total_moves = row["total_moves"] or 0
                if total_moves < self.config.min_moves_for_valid:
                    stats["invalid"] += 1
                    continue

                # December 29, 2025: Use pre-fetched move counts (O(1) lookup vs O(n) queries)
                if self.config.validate_before_merge:
                    actual_moves = move_counts.get(game_id, 0)
                    if actual_moves < self.config.min_moves_for_valid:
                        logger.debug(
                            f"[DataConsolidationDaemon] Skipping {game_id}: "
                            f"actual moves ({actual_moves}) < min ({self.config.min_moves_for_valid})"
                        )
                        stats["invalid"] += 1
                        continue

                stats["valid"] += 1

                # Insert game into target
                try:
                    # December 29, 2025: Create copies for each row to avoid shared list mutation
                    # The columns list is defined once from cursor.description, but we may
                    # need to add consolidated_at. Using copies prevents the bug where
                    # len(cols) != len(values) on subsequent iterations.
                    cols = columns.copy()

                    # December 29, 2025: Extract only values for columns that exist in target
                    # Use column_indices to get values in the same order as filtered columns
                    raw_row = list(row)
                    values = [raw_row[i] for i in column_indices]

                    # Add/update consolidated_at
                    if "consolidated_at" in cols:
                        idx = cols.index("consolidated_at")
                        values[idx] = now
                    else:
                        cols.append("consolidated_at")
                        values.append(now)

                    placeholders = ",".join("?" * len(cols))
                    target_conn.execute(
                        f"INSERT OR IGNORE INTO games ({','.join(cols)}) VALUES ({placeholders})",
                        values
                    )

                    # Copy related tables (moves, initial_state, snapshots, players)
                    self._copy_game_data_sync(source_conn, target_conn, game_id)

                    # January 2026: Post-insert validation to prevent orphan games
                    # Verify that the game has at least MIN_MOVES_REQUIRED moves
                    MIN_MOVES_REQUIRED = 5
                    cursor = target_conn.execute(
                        "SELECT COUNT(*) FROM game_moves WHERE game_id = ?",
                        (game_id,)
                    )
                    move_count = cursor.fetchone()[0]

                    if move_count < MIN_MOVES_REQUIRED:
                        # Delete orphan game - it has insufficient move data
                        target_conn.execute("DELETE FROM games WHERE game_id = ?", (game_id,))
                        stats["orphans_prevented"] = stats.get("orphans_prevented", 0) + 1
                        logger.debug(
                            f"[DataConsolidationDaemon] Prevented orphan game {game_id}: "
                            f"only {move_count} moves (need {MIN_MOVES_REQUIRED})"
                        )
                        # Emit event for monitoring
                        safe_emit_event(
                            "orphan_game_prevented",
                            {
                                "game_id": game_id,
                                "move_count": move_count,
                                "min_required": MIN_MOVES_REQUIRED,
                                "source": "data_consolidation_daemon",
                            },
                            context="DataConsolidation",
                        )
                        continue  # Skip counting this as merged

                    stats["merged"] += 1
                    stats["new_ids"].add(game_id)

                except sqlite3.Error as e:
                    logger.debug(f"[DataConsolidationDaemon] Error inserting game {game_id}: {e}")
                    stats["invalid"] += 1

            target_conn.commit()

            # January 2026: Recreate the enforce_moves_on_insert trigger after consolidation
            # This maintains database integrity for other operations that insert games
            target_conn.execute("""
                CREATE TRIGGER IF NOT EXISTS enforce_moves_on_insert
                AFTER INSERT ON games
                WHEN NEW.game_status IN ('completed', 'finished')
                BEGIN
                    SELECT CASE
                        WHEN (SELECT COUNT(*) FROM game_moves WHERE game_id = NEW.game_id) < 5
                        THEN RAISE(ABORT, 'Cannot insert completed game with fewer than 5 moves. Move data must be inserted before completing the game.')
                    END;
                END
            """)
            target_conn.commit()

        except sqlite3.Error as e:
            logger.warning(f"[DataConsolidationDaemon] Database error during merge: {e}")
        finally:
            # December 27, 2025: Ensure connections are always closed
            if source_conn:
                source_conn.close()
            if target_conn:
                target_conn.close()

        return stats

    async def _merge_database(
        self,
        source_db: Path,
        target_db: Path,
        board_type: str,
        num_players: int,
        existing_ids: set[str],
    ) -> dict[str, Any]:
        """Merge games from source database into target database.

        December 29, 2025: Refactored to use asyncio.to_thread() to avoid
        blocking the event loop during SQLite operations.

        Args:
            source_db: Source database path
            target_db: Target (canonical) database path
            board_type: Board type filter
            num_players: Player count filter
            existing_ids: Set of game IDs already in target

        Returns:
            Dict with merge statistics
        """
        return await asyncio.to_thread(
            self._merge_database_sync,
            source_db,
            target_db,
            board_type,
            num_players,
            existing_ids,
        )

    def _batch_count_moves(
        self,
        conn: sqlite3.Connection,
        game_ids: list[str],
    ) -> dict[str, int]:
        """Count actual moves for multiple games in a single query.

        December 29, 2025: Optimized batch query to replace N+1 pattern.
        Uses GROUP BY to fetch all move counts in O(1) round trips.

        Args:
            conn: SQLite connection to query
            game_ids: List of game IDs to check

        Returns:
            Dict mapping game_id -> move count (0 if not found)
        """
        if not game_ids:
            return {}

        try:
            # Batch query with GROUP BY - O(1) round trips vs O(n)
            placeholders = ",".join("?" * len(game_ids))
            cursor = conn.execute(
                f"""SELECT game_id, COUNT(*) as move_count
                    FROM game_moves
                    WHERE game_id IN ({placeholders})
                    GROUP BY game_id""",
                game_ids,
            )
            return {row[0]: row[1] for row in cursor.fetchall()}
        except sqlite3.Error as e:
            logger.warning(f"[DataConsolidationDaemon] Error counting moves: {e}")
            return {}

    def _count_actual_moves(
        self,
        conn: sqlite3.Connection,
        game_id: str,
    ) -> int:
        """Count actual moves in game_moves table for a game.

        December 2025: Added to catch race conditions where games record has
        total_moves set but game_moves table is incomplete (sync captured
        database mid-write).

        Note: For batch operations, use _batch_count_moves() instead.

        Args:
            conn: SQLite connection to query
            game_id: Game ID to check

        Returns:
            Number of moves in game_moves table (0 if not found)
        """
        try:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM game_moves WHERE game_id = ?",
                (game_id,)
            )
            result = cursor.fetchone()
            return result[0] if result else 0
        except sqlite3.Error:
            return 0

    def _copy_game_data_sync(
        self,
        source_conn: sqlite3.Connection,
        target_conn: sqlite3.Connection,
        game_id: str,
    ) -> None:
        """Copy related game data (moves, states, players) from source to target.

        December 29, 2025: Fixed to use SELECT * and dynamically get columns,
        then filter to only columns that exist in target table. This handles
        schema evolution where source and target may have different columns.

        Note: This is a sync method to be called via asyncio.to_thread().
        """
        tables = ["game_moves", "game_initial_state", "game_state_snapshots", "game_players"]

        for table in tables:
            try:
                # December 29, 2025: Get target columns for this table
                target_cursor = target_conn.execute(f"PRAGMA table_info({table})")
                target_cols = {row[1] for row in target_cursor.fetchall()}
                if not target_cols:
                    continue  # Target table doesn't exist

                # Query all data from source for this game
                cursor = source_conn.execute(
                    f"SELECT * FROM {table} WHERE game_id = ?",
                    (game_id,)
                )
                rows = cursor.fetchall()

                if not rows:
                    continue

                # Get source columns and filter to those in target
                source_cols = [desc[0] for desc in cursor.description]
                common_cols = [c for c in source_cols if c in target_cols]
                col_indices = [source_cols.index(c) for c in common_cols]

                if not common_cols:
                    continue

                # Build filtered rows with only common columns
                filtered_rows = [
                    tuple(row[i] for i in col_indices)
                    for row in rows
                ]

                placeholders = ",".join("?" * len(common_cols))
                target_conn.executemany(
                    f"INSERT OR IGNORE INTO {table} ({','.join(common_cols)}) VALUES ({placeholders})",
                    filtered_rows
                )
            except sqlite3.Error as e:
                logger.debug(f"[DataConsolidationDaemon] Error copying {table}: {e}")

    async def _emit_consolidation_started(
        self,
        config_key: str,
        board_type: str,
        num_players: int,
    ) -> None:
        """Emit CONSOLIDATION_STARTED event."""
        safe_emit_event(
            "consolidation_started",
            {
                "config_key": config_key,
                "board_type": board_type,
                "num_players": num_players,
                "timestamp": time.time(),
                "source": "data_consolidation_daemon",
            },
            context="DataConsolidation",
        )

    async def _emit_consolidation_complete(
        self,
        config_key: str,
        board_type: str,
        num_players: int,
        canonical_db: str,
        games_merged: int,
        total_games: int,
    ) -> None:
        """Emit CONSOLIDATION_COMPLETE event."""
        safe_emit_event(
            "consolidation_complete",
            {
                "config_key": config_key,
                "board_type": board_type,
                "num_players": num_players,
                "canonical_db": canonical_db,
                "games_merged": games_merged,
                "total_games": total_games,
                "timestamp": time.time(),
                "source": "data_consolidation_daemon",
            },
            context="DataConsolidation",
        )
        logger.debug(
            f"[DataConsolidationDaemon] Emitted CONSOLIDATION_COMPLETE for {config_key}: "
            f"merged={games_merged}, total={total_games}"
        )

    async def trigger_consolidation(self, config_key: str) -> ConsolidationStats | None:
        """Manually trigger consolidation for a specific config.

        Args:
            config_key: Config key (e.g., "hex8_2p")

        Returns:
            ConsolidationStats if successful, None otherwise
        """
        parsed = parse_config_key(config_key)
        if not parsed:
            return None

        return await self._consolidate_config(parsed.board_type, parsed.num_players)

    async def trigger_all_consolidations(self) -> dict[str, ConsolidationStats]:
        """Trigger consolidation for all configs.

        Returns:
            Dict mapping config_key to ConsolidationStats
        """
        results = {}

        for board_type, num_players in ALL_CONFIGS:
            config_key = make_config_key(board_type, num_players)
            stats = await self._consolidate_config(board_type, num_players)
            results[config_key] = stats

        return results

    def get_status(self) -> dict[str, Any]:
        """Get daemon status."""
        return {
            "running": self._running,
            "subscribed": self._subscribed,
            "pending_configs": list(self._pending_configs),
            "last_consolidation": dict(self._last_consolidation),
            "recent_stats": [
                {
                    "config_key": s.config_key,
                    "games_merged": s.games_merged,
                    "success": s.success,
                    "duration": s.duration_seconds,
                }
                for s in self._stats_history[-10:]
            ],
        }

    def health_check(self) -> HealthCheckResult:
        """Return health check result for DaemonManager integration.

        December 2025: Updated to use CoordinatorStatus enum and HandlerStats.
        """
        details = {
            "running": self._running,
            "subscribed": self._subscribed,
            "pending_count": len(self._pending_configs),
            "recent_consolidations": len(self._stats_history),
            "files_consolidated": sum(s.games_merged for s in self._stats_history),
            "last_consolidation": self._last_consolidation,
            "cycles_completed": self._stats.cycles_completed,
            "errors_count": self._stats.errors_count,
        }

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="DataConsolidationDaemon is not running",
                details=details,
            )

        if not self._subscribed:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message="DataConsolidationDaemon not subscribed to events",
                details=details,
            )

        # Check error rate in recent consolidations using HealthCheckHelper
        recent = self._stats_history[-20:] if self._stats_history else []
        if recent:
            error_count = sum(1 for s in recent if not s.success)
            is_healthy, msg = HealthCheckHelper.check_error_rate(
                errors=error_count,
                cycles=len(recent),
                threshold=0.5,
            )
            if not is_healthy:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"DataConsolidationDaemon {msg}",
                    details=details,
                )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"DataConsolidationDaemon healthy ({len(self._stats_history)} consolidations)",
            details=details,
        )


def get_consolidation_daemon() -> DataConsolidationDaemon:
    """Get the singleton DataConsolidationDaemon instance."""
    return DataConsolidationDaemon.get_instance()


def reset_consolidation_daemon() -> None:
    """Reset the singleton instance (for testing)."""
    DataConsolidationDaemon.reset_instance()

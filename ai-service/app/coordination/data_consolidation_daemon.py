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

from app.coordination.base_daemon import BaseDaemon, DaemonConfig
from app.coordination.protocols import HealthCheckResult

logger = logging.getLogger(__name__)

__all__ = [
    "DataConsolidationDaemon",
    "ConsolidationConfig",
    "ConsolidationStats",
    "get_consolidation_daemon",
    "reset_consolidation_daemon",
]

# Singleton instance
_consolidation_daemon: "DataConsolidationDaemon | None" = None


@dataclass
class ConsolidationConfig(DaemonConfig):
    """Configuration for the consolidation daemon.

    Inherits from DaemonConfig:
    - enabled: Whether the daemon should run
    - check_interval_seconds: How often to run consolidation cycle (default: 300s)
    - handle_signals: Whether to register SIGTERM/SIGINT handlers
    """

    # Base paths
    data_dir: Path = field(default_factory=lambda: Path("data/games"))
    canonical_dir: Path = field(default_factory=lambda: Path("data/games"))

    # Consolidation thresholds
    min_games_for_consolidation: int = 50  # Minimum new games before consolidating

    # Database settings
    min_moves_for_valid: int = 5  # Minimum moves for a valid game
    batch_size: int = 100  # Games per batch during merge

    # Safety
    deduplicate: bool = True  # Deduplicate by game_id
    validate_before_merge: bool = True  # Check game validity before merging

    @classmethod
    def from_env(cls) -> "ConsolidationConfig":
        """Load configuration from environment variables."""
        import os

        return cls(
            # DaemonConfig fields
            check_interval_seconds=int(os.getenv("RINGRIFT_CONSOLIDATION_INTERVAL", "300")),
            # ConsolidationConfig fields
            data_dir=Path(os.getenv("RINGRIFT_DATA_DIR", "data/games")),
            canonical_dir=Path(os.getenv("RINGRIFT_CANONICAL_DIR", "data/games")),
            min_games_for_consolidation=int(os.getenv("RINGRIFT_CONSOLIDATION_MIN_GAMES", "50")),
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


class DataConsolidationDaemon(BaseDaemon[ConsolidationConfig]):
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
        # Pass config to BaseDaemon (will use _get_default_config if None)
        super().__init__(config=config)

        # State tracking
        self._pending_configs: set[str] = set()  # Configs needing consolidation
        self._last_consolidation: dict[str, float] = {}  # config_key -> timestamp
        self._stats_history: list[ConsolidationStats] = []
        self._subscribed = False

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    @staticmethod
    def _get_default_config() -> ConsolidationConfig:
        """Return default configuration from environment."""
        return ConsolidationConfig.from_env()

    async def _on_start(self) -> None:
        """Called after daemon starts - subscribe to events and trigger initial scan.

        CRITICAL: The initial consolidation scan ensures that games generated
        before this daemon started are consolidated. Without this, canonical
        databases remain empty until NEW_GAMES_AVAILABLE or SELFPLAY_COMPLETE
        events arrive, which may never happen for historical data.
        """
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
            logger.debug(f"[DataConsolidationDaemon] Error unsubscribing: {e}")

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
                config_key = f"{board_type}_{num_players}p"

            if config_key:
                self._pending_configs.add(config_key)
                logger.debug(f"[DataConsolidationDaemon] Queued {config_key} for consolidation")

        except Exception as e:
            logger.debug(f"[DataConsolidationDaemon] Error handling SELFPLAY_COMPLETE: {e}")

    async def _process_pending_consolidations(self) -> None:
        """Process all pending consolidations."""
        async with self._lock:
            if not self._pending_configs:
                return

            # Copy and clear pending set
            configs_to_process = list(self._pending_configs)
            self._pending_configs.clear()

        for config_key in configs_to_process:
            try:
                # Parse config key
                parts = config_key.rsplit("_", 1)
                if len(parts) != 2:
                    continue

                board_type = parts[0]
                num_players = int(parts[1].replace("p", ""))

                # Check if enough time has passed since last consolidation
                last_time = self._last_consolidation.get(config_key, 0)
                if time.time() - last_time < 60.0:  # Minimum 1 minute between consolidations
                    continue

                # Run consolidation
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

            except Exception as e:
                logger.error(f"[DataConsolidationDaemon] Error consolidating {config_key}: {e}")

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
        config_key = f"{board_type}_{num_players}p"
        stats = ConsolidationStats(config_key=config_key, start_time=time.time())

        try:
            # Emit start event
            await self._emit_consolidation_started(config_key, board_type, num_players)

            # Find all source databases
            source_dbs = self._find_source_databases(board_type, num_players)
            stats.source_dbs_scanned = len(source_dbs)

            if not source_dbs:
                stats.success = True
                stats.end_time = time.time()
                logger.debug(f"[DataConsolidationDaemon] No source databases found for {config_key}")
                return stats

            # Get or create canonical database
            canonical_db = self._get_canonical_db_path(board_type, num_players)
            self._ensure_canonical_schema(canonical_db)

            # Get existing game IDs to avoid duplicates
            existing_ids = self._get_existing_game_ids(canonical_db)

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

        # Search patterns for selfplay databases
        search_dirs = [
            self.config.data_dir,
            self.config.data_dir / "selfplay",
            self.config.data_dir / "p2p_gpu",
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
            with sqlite3.connect(str(db_path), timeout=5.0) as conn:
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
            with sqlite3.connect(str(db_path), timeout=30.0) as conn:
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
        with sqlite3.connect(str(db_path), timeout=30.0) as conn:
            # Main games table
            conn.execute("""
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
                    schema_version INTEGER NOT NULL DEFAULT 5,
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

            # Initial state table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS game_initial_state (
                    game_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    FOREIGN KEY (game_id) REFERENCES games(game_id)
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

            # Players table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS game_players (
                    game_id TEXT NOT NULL,
                    player_index INTEGER NOT NULL,
                    player_type TEXT,
                    model_version TEXT,
                    PRIMARY KEY (game_id, player_index),
                    FOREIGN KEY (game_id) REFERENCES games(game_id)
                )
            """)

            # Add consolidated_at column if not exists
            try:
                conn.execute("ALTER TABLE games ADD COLUMN consolidated_at REAL")
            except sqlite3.OperationalError:
                pass  # Column already exists

            conn.commit()
            # Note: conn.close() not needed - context manager handles it

    async def _merge_database(
        self,
        source_db: Path,
        target_db: Path,
        board_type: str,
        num_players: int,
        existing_ids: set[str],
    ) -> dict[str, Any]:
        """Merge games from source database into target database.

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

        try:
            source_conn = sqlite3.connect(str(source_db), timeout=30.0)
            source_conn.row_factory = sqlite3.Row
            target_conn = sqlite3.connect(str(target_db), timeout=30.0)

            # Query games for this config
            cursor = source_conn.execute("""
                SELECT * FROM games
                WHERE board_type = ? AND num_players = ?
                AND game_status IN ('completed', 'finished', 'victory')
            """, (board_type, num_players))

            now = time.time()

            for row in cursor:
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

                # December 2025: Verify actual move count in game_moves table
                # This catches race conditions where games record has moves but
                # game_moves table is incomplete
                if self.config.validate_before_merge:
                    actual_moves = self._count_actual_moves(source_conn, game_id)
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
                    columns = [desc[0] for desc in cursor.description]
                    values = list(row)

                    # Add/update consolidated_at
                    if "consolidated_at" in columns:
                        idx = columns.index("consolidated_at")
                        values[idx] = now
                    else:
                        columns.append("consolidated_at")
                        values.append(now)

                    placeholders = ",".join("?" * len(columns))
                    target_conn.execute(
                        f"INSERT OR IGNORE INTO games ({','.join(columns)}) VALUES ({placeholders})",
                        values
                    )

                    # Copy related tables (moves, initial_state, snapshots, players)
                    await self._copy_game_data(source_conn, target_conn, game_id)

                    stats["merged"] += 1
                    stats["new_ids"].add(game_id)

                except sqlite3.Error as e:
                    logger.debug(f"[DataConsolidationDaemon] Error inserting game {game_id}: {e}")
                    stats["invalid"] += 1

            target_conn.commit()
            source_conn.close()
            target_conn.close()

        except sqlite3.Error as e:
            logger.warning(f"[DataConsolidationDaemon] Database error during merge: {e}")

        return stats

    def _count_actual_moves(
        self,
        conn: sqlite3.Connection,
        game_id: str,
    ) -> int:
        """Count actual moves in game_moves table for a game.

        December 2025: Added to catch race conditions where games record has
        total_moves set but game_moves table is incomplete (sync captured
        database mid-write).

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

    async def _copy_game_data(
        self,
        source_conn: sqlite3.Connection,
        target_conn: sqlite3.Connection,
        game_id: str,
    ) -> None:
        """Copy related game data (moves, states, players) from source to target."""
        tables_and_columns = [
            ("game_moves", "game_id, move_number, player, position_q, position_r, move_type, move_probs"),
            ("game_initial_state", "game_id, state_json"),
            ("game_state_snapshots", "game_id, move_number, state_json"),
            ("game_players", "game_id, player_index, player_type, model_version"),
        ]

        for table, columns in tables_and_columns:
            try:
                cursor = source_conn.execute(
                    f"SELECT {columns} FROM {table} WHERE game_id = ?",
                    (game_id,)
                )
                rows = cursor.fetchall()

                if rows:
                    placeholders = ",".join("?" * len(columns.split(", ")))
                    target_conn.executemany(
                        f"INSERT OR IGNORE INTO {table} ({columns}) VALUES ({placeholders})",
                        rows
                    )
            except sqlite3.Error:
                pass  # Table might not exist in source

    async def _emit_consolidation_started(
        self,
        config_key: str,
        board_type: str,
        num_players: int,
    ) -> None:
        """Emit CONSOLIDATION_STARTED event."""
        try:
            from app.distributed.data_events import DataEventType, emit_data_event

            await emit_data_event(
                event_type=DataEventType.CONSOLIDATION_STARTED,
                payload={
                    "config_key": config_key,
                    "board_type": board_type,
                    "num_players": num_players,
                    "timestamp": time.time(),
                },
                source="data_consolidation_daemon",
            )
        except (ImportError, AttributeError):
            pass  # Event type not available yet
        except Exception as e:
            logger.debug(f"[DataConsolidationDaemon] Error emitting CONSOLIDATION_STARTED: {e}")

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
        try:
            from app.distributed.data_events import DataEventType, emit_data_event

            await emit_data_event(
                event_type=DataEventType.CONSOLIDATION_COMPLETE,
                payload={
                    "config_key": config_key,
                    "board_type": board_type,
                    "num_players": num_players,
                    "canonical_db": canonical_db,
                    "games_merged": games_merged,
                    "total_games": total_games,
                    "timestamp": time.time(),
                },
                source="data_consolidation_daemon",
            )

            logger.debug(
                f"[DataConsolidationDaemon] Emitted CONSOLIDATION_COMPLETE for {config_key}: "
                f"merged={games_merged}, total={total_games}"
            )
        except (ImportError, AttributeError):
            pass  # Event type not available yet
        except Exception as e:
            logger.debug(f"[DataConsolidationDaemon] Error emitting CONSOLIDATION_COMPLETE: {e}")

    async def trigger_consolidation(self, config_key: str) -> ConsolidationStats | None:
        """Manually trigger consolidation for a specific config.

        Args:
            config_key: Config key (e.g., "hex8_2p")

        Returns:
            ConsolidationStats if successful, None otherwise
        """
        parts = config_key.rsplit("_", 1)
        if len(parts) != 2:
            return None

        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))

        return await self._consolidate_config(board_type, num_players)

    async def trigger_all_consolidations(self) -> dict[str, ConsolidationStats]:
        """Trigger consolidation for all configs.

        Returns:
            Dict mapping config_key to ConsolidationStats
        """
        results = {}

        for board_type, num_players in ALL_CONFIGS:
            config_key = f"{board_type}_{num_players}p"
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
        """Return health check result."""
        is_healthy = self._running and self._subscribed

        return HealthCheckResult(
            healthy=is_healthy,
            status="running" if is_healthy else "degraded",
            details={
                "running": self._running,
                "subscribed": self._subscribed,
                "pending_count": len(self._pending_configs),
                "recent_consolidations": len(self._stats_history),
            },
        )


def get_consolidation_daemon() -> DataConsolidationDaemon:
    """Get the singleton DataConsolidationDaemon instance."""
    global _consolidation_daemon
    if _consolidation_daemon is None:
        _consolidation_daemon = DataConsolidationDaemon()
    return _consolidation_daemon


def reset_consolidation_daemon() -> None:
    """Reset the singleton instance (for testing)."""
    global _consolidation_daemon
    _consolidation_daemon = None

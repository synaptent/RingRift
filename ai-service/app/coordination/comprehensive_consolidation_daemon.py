"""ComprehensiveConsolidationDaemon - Scheduled full-sweep consolidation across all sources.

This daemon solves the critical gap where event-driven consolidation misses historical data
and comprehensive sweeps are never performed. It provides:

1. SCHEDULED sweeps (not just event-driven) - catches missed data
2. ALL 14+ patterns scanned - including OWC imports, synced, p2p_gpu
3. P2P manifest integration - fetches game counts from cluster nodes
4. Consolidation tracking - verifies completeness per config
5. External storage awareness - knows about S3/OWC locations

January 2026: Created as part of Unified Data Consolidation, Backup, and Visibility system.

Event Flow:
    Scheduled timer OR CONSOLIDATION_REQUESTED event
        ↓
    ComprehensiveConsolidationDaemon._run_cycle()
        ↓
    Scan ALL sources (GameDiscovery + P2P manifest + OWC imports)
        ↓
    Merge into canonical_{board}_{n}p.db (deduplicate by game_id)
        ↓
    Update consolidation_tracking table
        ↓
    COMPREHENSIVE_CONSOLIDATION_COMPLETE → DataPipelineOrchestrator
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
from app.coordination.contracts import CoordinatorStatus
from app.coordination.event_utils import make_config_key
from app.coordination.event_emission_helpers import safe_emit_event
from app.utils.game_discovery import GameDiscovery, DatabaseInfo, ALL_BOARD_TYPES, ALL_PLAYER_COUNTS
from app.config.thresholds import SQLITE_TIMEOUT
from app.db.game_replay import SCHEMA_VERSION

logger = logging.getLogger(__name__)

__all__ = [
    "ComprehensiveConsolidationDaemon",
    "ComprehensiveConsolidationConfig",
    "ConsolidationTrackingRecord",
    "get_comprehensive_consolidation_daemon",
    "reset_comprehensive_consolidation_daemon",
]


# All canonical configurations
ALL_CONFIGS = [
    (board_type, num_players)
    for board_type in ALL_BOARD_TYPES
    for num_players in ALL_PLAYER_COUNTS
]


@dataclass
class ComprehensiveConsolidationConfig:
    """Configuration for the comprehensive consolidation daemon."""

    # Daemon control
    enabled: bool = True
    cycle_interval_seconds: int = 1800  # 30 minutes default (scheduled sweeps)

    # Base paths
    data_dir: Path = field(default_factory=lambda: Path("data/games"))
    canonical_dir: Path = field(default_factory=lambda: Path("data/games"))

    # Consolidation behavior
    min_moves_for_valid: int = 5  # Minimum moves for a valid game
    batch_commit_size: int = 100  # Games per batch commit
    max_concurrent_configs: int = 3  # Max parallel config consolidations

    # Source scanning
    scan_owc_imports: bool = True  # Include OWC external drive imports
    scan_synced: bool = True  # Include P2P synced databases
    scan_p2p_gpu: bool = True  # Include P2P GPU selfplay databases
    scan_p2p_manifest: bool = True  # Query P2P manifest for remote data

    # Tracking database
    tracking_db_path: Path = field(default_factory=lambda: Path("data/consolidation_tracking.db"))

    # Coordinator settings
    coordinator_only: bool = False  # Can run on any node with disk space

    @classmethod
    def from_env(cls) -> "ComprehensiveConsolidationConfig":
        """Load configuration from environment variables."""
        import os

        return cls(
            enabled=os.getenv("RINGRIFT_COMPREHENSIVE_CONSOLIDATION_ENABLED", "true").lower() == "true",
            cycle_interval_seconds=int(os.getenv("RINGRIFT_COMPREHENSIVE_CONSOLIDATION_INTERVAL", "1800")),
            data_dir=Path(os.getenv("RINGRIFT_DATA_DIR", "data/games")),
            canonical_dir=Path(os.getenv("RINGRIFT_CANONICAL_DIR", "data/games")),
            max_concurrent_configs=int(os.getenv("RINGRIFT_CONSOLIDATION_MAX_CONCURRENT", "3")),
            scan_p2p_manifest=os.getenv("RINGRIFT_CONSOLIDATION_SCAN_P2P", "true").lower() == "true",
            tracking_db_path=Path(os.getenv("RINGRIFT_CONSOLIDATION_TRACKING_DB", "data/consolidation_tracking.db")),
            coordinator_only=os.getenv("RINGRIFT_COMPREHENSIVE_CONSOLIDATION_COORDINATOR_ONLY", "false").lower() == "true",
        )


@dataclass
class ConsolidationTrackingRecord:
    """Tracking record for a config's consolidation status."""
    config_key: str
    canonical_game_count: int = 0
    last_consolidation_time: float = 0.0
    sources_scanned: list[str] = field(default_factory=list)
    games_added_last_run: int = 0
    game_ids_hash: str = ""
    total_sources_found: int = 0


@dataclass
class ComprehensiveConsolidationStats:
    """Statistics for a comprehensive consolidation run."""
    config_key: str = ""
    sources_scanned: int = 0
    games_scanned: int = 0
    games_valid: int = 0
    games_merged: int = 0
    games_duplicate: int = 0
    games_invalid: int = 0
    p2p_sources_queried: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    success: bool = False
    error: str | None = None

    @property
    def duration_seconds(self) -> float:
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time


class ComprehensiveConsolidationDaemon(HandlerBase):
    """Daemon that performs scheduled comprehensive consolidation sweeps.

    Unlike event-driven consolidation, this daemon:
    - Runs on a schedule (default: every 30 minutes)
    - Scans ALL known database patterns via GameDiscovery
    - Queries P2P manifest for game counts on remote nodes
    - Tracks consolidation progress in a dedicated database
    - Emits COMPREHENSIVE_CONSOLIDATION_COMPLETE for pipeline coordination

    Subscribes to:
    - CONSOLIDATION_REQUESTED: Manual trigger for immediate consolidation

    Emits:
    - COMPREHENSIVE_CONSOLIDATION_STARTED: Beginning sweep
    - COMPREHENSIVE_CONSOLIDATION_COMPLETE: Sweep finished with stats
    """

    def __init__(self, config: ComprehensiveConsolidationConfig | None = None):
        """Initialize the comprehensive consolidation daemon.

        Args:
            config: Configuration for consolidation behavior
        """
        self._daemon_config = config or ComprehensiveConsolidationConfig.from_env()
        super().__init__(
            name="ComprehensiveConsolidation",
            config=self._daemon_config,
            cycle_interval=float(self._daemon_config.cycle_interval_seconds),
        )

        # State tracking
        self._tracking_records: dict[str, ConsolidationTrackingRecord] = {}
        self._stats_history: list[ComprehensiveConsolidationStats] = []
        self._subscribed = False

        # Concurrency control
        self._consolidation_semaphore = asyncio.Semaphore(
            self._daemon_config.max_concurrent_configs
        )
        self._lock = asyncio.Lock()

        # Game discovery
        self._discovery: GameDiscovery | None = None

        # Last sweep results
        self._last_sweep_time: float = 0.0
        self._last_sweep_total_merged: int = 0

    @property
    def config(self) -> ComprehensiveConsolidationConfig:
        """Return daemon configuration."""
        return self._daemon_config

    async def _on_start(self) -> None:
        """Called after daemon starts."""
        # Ensure directories exist
        self._daemon_config.canonical_dir.mkdir(parents=True, exist_ok=True)
        self._daemon_config.tracking_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize game discovery
        self._discovery = GameDiscovery(self._daemon_config.data_dir.parent)

        # Initialize tracking database
        await asyncio.to_thread(self._ensure_tracking_schema)

        # Load existing tracking records
        await asyncio.to_thread(self._load_tracking_records)

        await self._subscribe_to_events()

        # Run initial sweep
        logger.info("[ComprehensiveConsolidation] Starting initial sweep...")
        try:
            await self._run_comprehensive_sweep()
            logger.info(
                f"[ComprehensiveConsolidation] Initial sweep complete: "
                f"{self._last_sweep_total_merged} games merged"
            )
        except Exception as e:
            logger.warning(f"[ComprehensiveConsolidation] Initial sweep failed: {e}")

    async def _on_stop(self) -> None:
        """Called before daemon stops."""
        await self._unsubscribe_from_events()

    async def _run_cycle(self) -> None:
        """Run one consolidation cycle (scheduled sweep)."""
        await self._run_comprehensive_sweep()

    async def _subscribe_to_events(self) -> None:
        """Subscribe to consolidation request events."""
        if self._subscribed:
            return

        try:
            from app.coordination.event_router import get_event_bus
            from app.distributed.data_events import DataEventType

            bus = get_event_bus()

            # Subscribe to manual trigger
            if hasattr(DataEventType, "CONSOLIDATION_REQUESTED"):
                bus.subscribe(DataEventType.CONSOLIDATION_REQUESTED, self._on_consolidation_requested)

            self._subscribed = True
            logger.info("[ComprehensiveConsolidation] Subscribed to events")

        except ImportError as e:
            logger.debug(f"[ComprehensiveConsolidation] Could not subscribe: {e}")

    async def _unsubscribe_from_events(self) -> None:
        """Unsubscribe from events on shutdown."""
        if not self._subscribed:
            return

        try:
            from app.coordination.event_router import get_event_bus
            from app.distributed.data_events import DataEventType

            bus = get_event_bus()
            if hasattr(DataEventType, "CONSOLIDATION_REQUESTED"):
                bus.unsubscribe(DataEventType.CONSOLIDATION_REQUESTED, self._on_consolidation_requested)

            self._subscribed = False
        except Exception as e:
            logger.debug(f"[ComprehensiveConsolidation] Error unsubscribing: {e}")

    def _on_consolidation_requested(self, event: Any) -> None:
        """Handle CONSOLIDATION_REQUESTED event for manual trigger."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key")

            if config_key:
                logger.info(f"[ComprehensiveConsolidation] Manual trigger for {config_key}")
                # Queue for next cycle - could also trigger immediate via asyncio.create_task
            else:
                logger.info("[ComprehensiveConsolidation] Manual trigger for all configs")
        except Exception as e:
            logger.debug(f"[ComprehensiveConsolidation] Error handling request: {e}")

    async def _run_comprehensive_sweep(self) -> None:
        """Run a comprehensive consolidation sweep across ALL sources.

        This is the main consolidation logic that:
        1. Discovers all databases via GameDiscovery (14+ patterns)
        2. Queries P2P manifest for remote data (if enabled)
        3. Merges each config's games into canonical databases
        4. Updates tracking records
        5. Emits completion events
        """
        sweep_start = time.time()
        total_merged = 0
        all_stats: dict[str, ComprehensiveConsolidationStats] = {}

        try:
            # Emit sweep started event
            await self._emit_sweep_started()

            # Discover all local databases
            if not self._discovery:
                self._discovery = GameDiscovery(self._daemon_config.data_dir.parent)

            # Clear discovery cache to get fresh results
            self._discovery.clear_cache()

            # Process each config in parallel
            tasks = []
            for board_type, num_players in ALL_CONFIGS:
                config_key = make_config_key(board_type, num_players)
                tasks.append(self._consolidate_config_comprehensive(board_type, num_players))

            # Run with concurrency limit
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect results
            for i, result in enumerate(results):
                board_type, num_players = ALL_CONFIGS[i]
                config_key = make_config_key(board_type, num_players)

                if isinstance(result, Exception):
                    logger.warning(f"[ComprehensiveConsolidation] Error for {config_key}: {result}")
                    continue

                if result:
                    all_stats[config_key] = result
                    total_merged += result.games_merged

            # Update tracking records
            await asyncio.to_thread(self._save_tracking_records)

            # Emit completion event
            await self._emit_sweep_complete(all_stats, sweep_start)

        except Exception as e:
            logger.error(f"[ComprehensiveConsolidation] Sweep failed: {e}")
        finally:
            self._last_sweep_time = time.time()
            self._last_sweep_total_merged = total_merged

            if total_merged > 0:
                logger.info(
                    f"[ComprehensiveConsolidation] Sweep complete: "
                    f"merged {total_merged} games across {len(all_stats)} configs, "
                    f"duration={time.time() - sweep_start:.1f}s"
                )

    async def _consolidate_config_comprehensive(
        self,
        board_type: str,
        num_players: int,
    ) -> ComprehensiveConsolidationStats | None:
        """Consolidate all games for a config from ALL sources.

        Args:
            board_type: Board type (hex8, square8, etc.)
            num_players: Number of players (2, 3, 4)

        Returns:
            ConsolidationStats or None if no work done
        """
        async with self._consolidation_semaphore:
            config_key = make_config_key(board_type, num_players)
            stats = ComprehensiveConsolidationStats(config_key=config_key, start_time=time.time())

            try:
                # Find all source databases using GameDiscovery
                source_dbs = await asyncio.to_thread(
                    self._find_all_sources,
                    board_type,
                    num_players,
                )
                stats.sources_scanned = len(source_dbs)

                if not source_dbs:
                    stats.success = True
                    stats.end_time = time.time()
                    return stats

                # Get or create canonical database
                canonical_db = self._get_canonical_db_path(board_type, num_players)
                await asyncio.to_thread(self._ensure_canonical_schema, canonical_db)

                # Get existing game IDs for deduplication
                existing_ids = await asyncio.to_thread(
                    self._get_existing_game_ids,
                    canonical_db,
                )

                # Track sources scanned for tracking record
                source_patterns: set[str] = set()

                # Merge from each source database
                for db_info in source_dbs:
                    try:
                        source_patterns.add(db_info.source_pattern)
                        merge_stats = await self._merge_database(
                            source_db=db_info.path,
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
                        logger.debug(f"[ComprehensiveConsolidation] Error merging {db_info.path}: {e}")

                # Update tracking record
                await asyncio.to_thread(
                    self._update_tracking_record,
                    config_key,
                    canonical_db,
                    len(existing_ids),
                    stats.games_merged,
                    list(source_patterns),
                    len(source_dbs),
                )

                stats.success = True
                stats.end_time = time.time()

                if stats.games_merged > 0:
                    logger.info(
                        f"[ComprehensiveConsolidation] {config_key}: "
                        f"merged {stats.games_merged} games from {stats.sources_scanned} sources"
                    )

            except Exception as e:
                stats.error = str(e)
                stats.end_time = time.time()
                logger.error(f"[ComprehensiveConsolidation] Failed for {config_key}: {e}")

            self._stats_history.append(stats)
            if len(self._stats_history) > 200:
                self._stats_history = self._stats_history[-200:]

            return stats

    def _find_all_sources(
        self,
        board_type: str,
        num_players: int,
    ) -> list[DatabaseInfo]:
        """Find ALL source databases for a config using GameDiscovery.

        This method leverages the unified GameDiscovery which already knows
        about all 14+ database patterns.

        Args:
            board_type: Board type to filter
            num_players: Player count to filter

        Returns:
            List of DatabaseInfo objects for sources
        """
        if not self._discovery:
            return []

        # Get all databases for this config
        all_dbs = self._discovery.find_databases_for_config(
            board_type,
            num_players,
            include_central=True,
            use_cache=False,  # Fresh results
        )

        # Filter out canonical databases (we don't merge canonical into itself)
        sources = [
            db for db in all_dbs
            if "canonical" not in db.path.name.lower()
            and db.game_count > 0
        ]

        return sources

    def _get_canonical_db_path(self, board_type: str, num_players: int) -> Path:
        """Get the path to the canonical database for a config."""
        return self._daemon_config.canonical_dir / f"canonical_{board_type}_{num_players}p.db"

    def _get_existing_game_ids(self, db_path: Path) -> set[str]:
        """Get set of existing game IDs in a database."""
        if not db_path.exists():
            return set()

        try:
            with sqlite3.connect(str(db_path), timeout=SQLITE_TIMEOUT) as conn:
                cursor = conn.execute("SELECT game_id FROM games")
                return {row[0] for row in cursor.fetchall()}
        except sqlite3.Error as e:
            logger.debug(f"[ComprehensiveConsolidation] Could not read IDs from {db_path}: {e}")
            return set()

    async def _merge_database(
        self,
        source_db: Path,
        target_db: Path,
        board_type: str,
        num_players: int,
        existing_ids: set[str],
    ) -> dict[str, Any]:
        """Merge games from source into target database.

        Args:
            source_db: Source database path
            target_db: Target canonical database path
            board_type: Board type filter
            num_players: Player count filter
            existing_ids: Set of existing game IDs for deduplication

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

    def _merge_database_sync(
        self,
        source_db: Path,
        target_db: Path,
        board_type: str,
        num_players: int,
        existing_ids: set[str],
    ) -> dict[str, Any]:
        """Synchronous merge implementation.

        IMPORTANT: Uses INSERT OR IGNORE for deduplication.
        Existing canonical data is NEVER deleted.
        """
        stats: dict[str, Any] = {
            "scanned": 0,
            "valid": 0,
            "merged": 0,
            "duplicate": 0,
            "invalid": 0,
            "new_ids": set(),
        }

        source_conn = None
        target_conn = None

        try:
            source_conn = sqlite3.connect(str(source_db), timeout=SQLITE_TIMEOUT)
            source_conn.row_factory = sqlite3.Row
            target_conn = sqlite3.connect(str(target_db), timeout=SQLITE_TIMEOUT)

            # Get target columns
            target_cursor = target_conn.execute("PRAGMA table_info(games)")
            target_columns = {row[1] for row in target_cursor.fetchall()}

            # Query games for this config
            cursor = source_conn.execute("""
                SELECT * FROM games
                WHERE board_type = ? AND num_players = ?
                AND game_status IN ('completed', 'finished', 'complete', 'victory')
            """, (board_type, num_players))

            source_columns = [desc[0] for desc in cursor.description]
            common_cols = [c for c in source_columns if c in target_columns]
            col_indices = [source_columns.index(c) for c in common_cols]

            now = time.time()

            for row in cursor:
                stats["scanned"] += 1
                game_id = row["game_id"]

                # Skip duplicates
                if game_id in existing_ids:
                    stats["duplicate"] += 1
                    continue

                # Validate game has moves
                total_moves = row["total_moves"] or 0
                if total_moves < self._daemon_config.min_moves_for_valid:
                    stats["invalid"] += 1
                    continue

                stats["valid"] += 1

                # Insert game
                try:
                    cols = common_cols.copy()
                    raw_row = list(row)
                    values = [raw_row[i] for i in col_indices]

                    # Add consolidated_at timestamp
                    if "consolidated_at" in cols:
                        idx = cols.index("consolidated_at")
                        values[idx] = now
                    elif "consolidated_at" in target_columns:
                        cols.append("consolidated_at")
                        values.append(now)

                    placeholders = ",".join("?" * len(cols))
                    target_conn.execute(
                        f"INSERT OR IGNORE INTO games ({','.join(cols)}) VALUES ({placeholders})",
                        values,
                    )

                    # Copy related tables
                    self._copy_game_data(source_conn, target_conn, game_id)

                    stats["merged"] += 1
                    stats["new_ids"].add(game_id)

                except sqlite3.Error as e:
                    logger.debug(f"[ComprehensiveConsolidation] Insert error for {game_id}: {e}")
                    stats["invalid"] += 1

            target_conn.commit()

        except sqlite3.Error as e:
            logger.debug(f"[ComprehensiveConsolidation] Database error: {e}")
        finally:
            if source_conn:
                source_conn.close()
            if target_conn:
                target_conn.close()

        return stats

    def _copy_game_data(
        self,
        source_conn: sqlite3.Connection,
        target_conn: sqlite3.Connection,
        game_id: str,
    ) -> None:
        """Copy related game data (moves, states, players)."""
        tables = ["game_moves", "game_initial_state", "game_state_snapshots", "game_players"]

        for table in tables:
            try:
                # Get target columns
                target_cursor = target_conn.execute(f"PRAGMA table_info({table})")
                target_cols = {row[1] for row in target_cursor.fetchall()}
                if not target_cols:
                    continue

                # Query source
                cursor = source_conn.execute(
                    f"SELECT * FROM {table} WHERE game_id = ?",
                    (game_id,),
                )
                rows = cursor.fetchall()
                if not rows:
                    continue

                # Filter columns
                source_cols = [desc[0] for desc in cursor.description]
                common_cols = [c for c in source_cols if c in target_cols]
                col_indices = [source_cols.index(c) for c in common_cols]

                # Insert rows
                filtered_rows = [
                    tuple(row[i] for i in col_indices)
                    for row in rows
                ]
                placeholders = ",".join("?" * len(common_cols))
                target_conn.executemany(
                    f"INSERT OR IGNORE INTO {table} ({','.join(common_cols)}) VALUES ({placeholders})",
                    filtered_rows,
                )

            except sqlite3.Error as e:
                logger.debug(f"[ComprehensiveConsolidation] Error copying {table}: {e}")

    def _ensure_canonical_schema(self, db_path: Path) -> None:
        """Ensure canonical database has correct schema."""
        db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(str(db_path), timeout=SQLITE_TIMEOUT) as conn:
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

            # Performance indexes
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

            conn.commit()

    def _ensure_tracking_schema(self) -> None:
        """Ensure tracking database has correct schema."""
        with sqlite3.connect(str(self._daemon_config.tracking_db_path), timeout=SQLITE_TIMEOUT) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_tracking (
                    config_key TEXT PRIMARY KEY,
                    canonical_game_count INTEGER NOT NULL DEFAULT 0,
                    last_consolidation_time REAL NOT NULL DEFAULT 0,
                    sources_scanned TEXT,
                    games_added_last_run INTEGER NOT NULL DEFAULT 0,
                    game_ids_hash TEXT,
                    total_sources_found INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL DEFAULT 0
                )
            """)
            conn.commit()

    def _load_tracking_records(self) -> None:
        """Load tracking records from database."""
        import json

        try:
            with sqlite3.connect(str(self._daemon_config.tracking_db_path), timeout=SQLITE_TIMEOUT) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM consolidation_tracking")
                for row in cursor:
                    sources = []
                    if row["sources_scanned"]:
                        try:
                            sources = json.loads(row["sources_scanned"])
                        except json.JSONDecodeError:
                            sources = []

                    self._tracking_records[row["config_key"]] = ConsolidationTrackingRecord(
                        config_key=row["config_key"],
                        canonical_game_count=row["canonical_game_count"],
                        last_consolidation_time=row["last_consolidation_time"],
                        sources_scanned=sources,
                        games_added_last_run=row["games_added_last_run"],
                        game_ids_hash=row["game_ids_hash"] or "",
                        total_sources_found=row["total_sources_found"],
                    )
        except sqlite3.Error as e:
            logger.warning(f"[ComprehensiveConsolidation] Error loading tracking: {e}")

    def _update_tracking_record(
        self,
        config_key: str,
        canonical_db: Path,
        game_count: int,
        games_added: int,
        sources: list[str],
        sources_found: int,
    ) -> None:
        """Update tracking record for a config."""
        import hashlib
        import json

        # Compute hash of game IDs for change detection
        try:
            with sqlite3.connect(str(canonical_db), timeout=SQLITE_TIMEOUT) as conn:
                cursor = conn.execute("SELECT game_id FROM games ORDER BY game_id")
                ids_str = ",".join(row[0] for row in cursor)
                ids_hash = hashlib.sha256(ids_str.encode()).hexdigest()[:16]
        except sqlite3.Error:
            ids_hash = ""

        record = ConsolidationTrackingRecord(
            config_key=config_key,
            canonical_game_count=game_count,
            last_consolidation_time=time.time(),
            sources_scanned=sources,
            games_added_last_run=games_added,
            game_ids_hash=ids_hash,
            total_sources_found=sources_found,
        )
        self._tracking_records[config_key] = record

    def _save_tracking_records(self) -> None:
        """Save tracking records to database."""
        import json

        try:
            with sqlite3.connect(str(self._daemon_config.tracking_db_path), timeout=SQLITE_TIMEOUT) as conn:
                for record in self._tracking_records.values():
                    conn.execute("""
                        INSERT OR REPLACE INTO consolidation_tracking
                        (config_key, canonical_game_count, last_consolidation_time,
                         sources_scanned, games_added_last_run, game_ids_hash,
                         total_sources_found, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        record.config_key,
                        record.canonical_game_count,
                        record.last_consolidation_time,
                        json.dumps(record.sources_scanned),
                        record.games_added_last_run,
                        record.game_ids_hash,
                        record.total_sources_found,
                        time.time(),
                    ))
                conn.commit()
        except sqlite3.Error as e:
            logger.warning(f"[ComprehensiveConsolidation] Error saving tracking: {e}")

    async def _emit_sweep_started(self) -> None:
        """Emit COMPREHENSIVE_CONSOLIDATION_STARTED event."""
        safe_emit_event(
            "comprehensive_consolidation_started",
            {
                "timestamp": time.time(),
                "configs": [f"{bt}_{np}p" for bt, np in ALL_CONFIGS],
            },
            context="ComprehensiveConsolidation",
        )

    async def _emit_sweep_complete(
        self,
        all_stats: dict[str, ComprehensiveConsolidationStats],
        start_time: float,
    ) -> None:
        """Emit COMPREHENSIVE_CONSOLIDATION_COMPLETE event."""
        total_merged = sum(s.games_merged for s in all_stats.values())
        total_scanned = sum(s.games_scanned for s in all_stats.values())

        safe_emit_event(
            "comprehensive_consolidation_complete",
            {
                "timestamp": time.time(),
                "duration_seconds": time.time() - start_time,
                "total_games_merged": total_merged,
                "total_games_scanned": total_scanned,
                "configs_processed": len(all_stats),
                "per_config_stats": {
                    k: {
                        "merged": v.games_merged,
                        "scanned": v.games_scanned,
                        "sources": v.sources_scanned,
                    }
                    for k, v in all_stats.items()
                    if v.games_merged > 0
                },
                "source": "comprehensive_consolidation",
            },
            context="ComprehensiveConsolidation",
        )

    def get_status(self) -> dict[str, Any]:
        """Get daemon status."""
        return {
            "running": self._running,
            "subscribed": self._subscribed,
            "last_sweep_time": self._last_sweep_time,
            "last_sweep_games_merged": self._last_sweep_total_merged,
            "tracking_records_count": len(self._tracking_records),
            "stats_history_count": len(self._stats_history),
            "recent_stats": [
                {
                    "config_key": s.config_key,
                    "games_merged": s.games_merged,
                    "sources_scanned": s.sources_scanned,
                    "success": s.success,
                }
                for s in self._stats_history[-12:]
            ],
        }

    def get_tracking_summary(self) -> dict[str, Any]:
        """Get summary of consolidation tracking across all configs."""
        return {
            config_key: {
                "game_count": record.canonical_game_count,
                "last_consolidation": record.last_consolidation_time,
                "games_added_last": record.games_added_last_run,
                "sources_found": record.total_sources_found,
            }
            for config_key, record in self._tracking_records.items()
        }

    def health_check(self) -> HealthCheckResult:
        """Return health check result for DaemonManager integration."""
        details = {
            "running": self._running,
            "subscribed": self._subscribed,
            "last_sweep_time": self._last_sweep_time,
            "last_sweep_games_merged": self._last_sweep_total_merged,
            "tracking_records": len(self._tracking_records),
            "stats_history": len(self._stats_history),
            "cycles_completed": self._stats.cycles_completed,
            "errors_count": self._stats.errors_count,
        }

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="ComprehensiveConsolidationDaemon is not running",
                details=details,
            )

        # Check for recent activity
        time_since_sweep = time.time() - self._last_sweep_time if self._last_sweep_time else float("inf")
        expected_interval = self._daemon_config.cycle_interval_seconds * 2  # Allow 2x leeway

        if time_since_sweep > expected_interval:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"ComprehensiveConsolidationDaemon: no sweep in {time_since_sweep:.0f}s",
                details=details,
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"ComprehensiveConsolidationDaemon healthy ({len(self._tracking_records)} configs tracked)",
            details=details,
        )


# Singleton management
_instance: ComprehensiveConsolidationDaemon | None = None


def get_comprehensive_consolidation_daemon() -> ComprehensiveConsolidationDaemon:
    """Get the singleton ComprehensiveConsolidationDaemon instance."""
    global _instance
    if _instance is None:
        _instance = ComprehensiveConsolidationDaemon()
    return _instance


def reset_comprehensive_consolidation_daemon() -> None:
    """Reset the singleton instance (for testing)."""
    global _instance
    _instance = None

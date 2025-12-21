"""Game replay database for storing and retrieving complete RingRift games.

This module provides SQLite-based storage for complete games from self-play,
supporting:
- Full game storage with initial state, moves, and player choices
- State reconstruction at any move number for replay
- Metadata queries for filtering and analysis
- Integration with training and sandbox UI

See docs/GAME_REPLAY_DATABASE_SPEC.md for full specification.
"""

from __future__ import annotations

import gzip
import json
import logging
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.models import BoardType, GameState, Move

if TYPE_CHECKING:
    import numpy as np

from app.rules.history_contract import validate_canonical_move

logger = logging.getLogger(__name__)

# Schema version for forward compatibility
# Version history:
# - v1: Initial schema (games, players, initial_state, moves, snapshots, choices)
# - v2: Added time control fields and engine evaluation fields
# - v3: Added game_history_entries table for GameTrace validation, state hash fields
# - v4: Added state_before_json and state_after_json to game_history_entries for full state replay
# - v5: Added metadata_json column to games to persist full recording metadata
# - v6: Added available_moves_json and available_moves_count to game_history_entries for parity debugging
# - v7: Added fsm_valid and fsm_error_code to game_history_entries for FSM validation tracking (Phase 7)
# - v8: Added game_nnue_features table for pre-computed NNUE training features
# - v9: Added quality_score and quality_category columns for training data prioritization
SCHEMA_VERSION = 9

# Default snapshot interval (every N moves)
DEFAULT_SNAPSHOT_INTERVAL = 20

# SQL schema creation statements (v2)
SCHEMA_SQL = """
-- Metadata table for schema versioning
CREATE TABLE IF NOT EXISTS schema_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Main games table
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
    schema_version INTEGER NOT NULL,
    -- v2 additions: time control
    time_control_type TEXT DEFAULT 'none',
    initial_time_ms INTEGER,
    time_increment_ms INTEGER,
    -- v5 additions: full recording metadata as JSON
    metadata_json TEXT,
    -- v9 additions: quality scoring for training data prioritization
    quality_score REAL,
    quality_category TEXT
);

-- Indexes on games
CREATE INDEX IF NOT EXISTS idx_games_board_type ON games(board_type);
CREATE INDEX IF NOT EXISTS idx_games_winner ON games(winner);
CREATE INDEX IF NOT EXISTS idx_games_termination ON games(termination_reason);
CREATE INDEX IF NOT EXISTS idx_games_created ON games(created_at);
CREATE INDEX IF NOT EXISTS idx_games_source ON games(source);
-- Composite indexes for common query patterns (training data filtering)
CREATE INDEX IF NOT EXISTS idx_games_board_players ON games(board_type, num_players);
CREATE INDEX IF NOT EXISTS idx_games_created_board ON games(created_at, board_type);
CREATE INDEX IF NOT EXISTS idx_games_board_created ON games(board_type, created_at);
-- Quality scoring indexes for training data prioritization (v9)
CREATE INDEX IF NOT EXISTS idx_games_quality ON games(quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_games_quality_board ON games(board_type, quality_score DESC);

-- Per-player metadata
CREATE TABLE IF NOT EXISTS game_players (
    game_id TEXT NOT NULL,
    player_number INTEGER NOT NULL,
    player_type TEXT NOT NULL,
    ai_type TEXT,
    ai_difficulty INTEGER,
    ai_profile_id TEXT,
    final_eliminated_rings INTEGER,
    final_territory_spaces INTEGER,
    final_rings_in_hand INTEGER,
    PRIMARY KEY (game_id, player_number),
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

-- Initial game state (for reconstruction)
CREATE TABLE IF NOT EXISTS game_initial_state (
    game_id TEXT PRIMARY KEY,
    initial_state_json TEXT NOT NULL,
    compressed INTEGER DEFAULT 0,
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

-- Move history
CREATE TABLE IF NOT EXISTS game_moves (
    game_id TEXT NOT NULL,
    move_number INTEGER NOT NULL,
    turn_number INTEGER NOT NULL,
    player INTEGER NOT NULL,
    phase TEXT NOT NULL,
    move_type TEXT NOT NULL,
    move_json TEXT NOT NULL,
    timestamp TEXT,
    think_time_ms INTEGER,
    -- v2 additions: time remaining and engine evaluation
    time_remaining_ms INTEGER,
    engine_eval REAL,
    engine_eval_type TEXT,
    engine_depth INTEGER,
    engine_nodes INTEGER,
    engine_pv TEXT,
    engine_time_ms INTEGER,
    PRIMARY KEY (game_id, move_number),
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_moves_game_turn ON game_moves(game_id, turn_number);

-- State snapshots for fast seeking
CREATE TABLE IF NOT EXISTS game_state_snapshots (
    game_id TEXT NOT NULL,
    move_number INTEGER NOT NULL,
    state_json TEXT NOT NULL,
    compressed INTEGER DEFAULT 0,
    state_hash TEXT,
    PRIMARY KEY (game_id, move_number),
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

-- Player choices during decision phases
CREATE TABLE IF NOT EXISTS game_choices (
    game_id TEXT NOT NULL,
    move_number INTEGER NOT NULL,
    choice_type TEXT NOT NULL,
    player INTEGER NOT NULL,
    options_json TEXT NOT NULL,
    selected_option_json TEXT NOT NULL,
    ai_reasoning TEXT,
    PRIMARY KEY (game_id, move_number, choice_type),
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

-- v3: History entries for GameTrace-style storage with cross-validation
-- Stores metadata for each move matching TypeScript GameHistoryEntry interface
-- v4: Added state_before_json and state_after_json for full state replay
-- v6: Added available_moves_json, available_moves_count, engine_eval, engine_depth
-- v7: Added fsm_valid, fsm_error_code for FSM validation tracking (Phase 7)
CREATE TABLE IF NOT EXISTS game_history_entries (
    game_id TEXT NOT NULL,
    move_number INTEGER NOT NULL,
    player INTEGER NOT NULL,
    phase_before TEXT NOT NULL,
    phase_after TEXT NOT NULL,
    status_before TEXT NOT NULL,
    status_after TEXT NOT NULL,
    progress_before_json TEXT NOT NULL,
    progress_after_json TEXT NOT NULL,
    state_hash_before TEXT,
    state_hash_after TEXT,
    board_summary_before_json TEXT,
    board_summary_after_json TEXT,
    -- v4 additions: full state snapshots before and after each move
    state_before_json TEXT,
    state_after_json TEXT,
    compressed_states INTEGER DEFAULT 0,
    -- v6 additions: available moves enumeration and engine diagnostics
    available_moves_json TEXT,
    available_moves_count INTEGER,
    engine_eval REAL,
    engine_depth INTEGER,
    -- v7 additions: FSM validation status (Phase 7: Data Pipeline)
    fsm_valid INTEGER,  -- 1 = valid, 0 = invalid, NULL = not checked
    fsm_error_code TEXT,  -- Error code if fsm_valid = 0
    PRIMARY KEY (game_id, move_number),
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_history_game ON game_history_entries(game_id);
-- Phase 7: Additional indexes for training data queries
CREATE INDEX IF NOT EXISTS idx_history_game_phase ON game_history_entries(game_id, phase_before);
CREATE INDEX IF NOT EXISTS idx_history_game_player ON game_history_entries(game_id, player);
CREATE INDEX IF NOT EXISTS idx_moves_player ON game_moves(game_id, player);

-- v3: Add state_hash to snapshots for validation
-- (Added via migration for existing DBs)

-- v8: NNUE features cache for instant training data extraction
-- Pre-computed features eliminate the need for game replay during training
CREATE TABLE IF NOT EXISTS game_nnue_features (
    game_id TEXT NOT NULL,
    move_number INTEGER NOT NULL,
    player_perspective INTEGER NOT NULL,  -- Player number for perspective rotation
    features BLOB NOT NULL,               -- Compressed float32 feature vector
    value REAL NOT NULL,                  -- Win/loss label (-1, 0, +1)
    board_type TEXT NOT NULL,             -- Board type for feature dimension validation
    feature_dim INTEGER NOT NULL,         -- Feature dimension for validation
    PRIMARY KEY (game_id, move_number, player_perspective),
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_nnue_game ON game_nnue_features(game_id);
CREATE INDEX IF NOT EXISTS idx_nnue_board_type ON game_nnue_features(board_type);
"""


def _compress_json(data: str) -> bytes:
    """Compress JSON string using gzip."""
    return gzip.compress(data.encode("utf-8"))


def _decompress_json(data: bytes) -> str:
    """Decompress gzip-compressed JSON."""
    return gzip.decompress(data).decode("utf-8")


def _serialize_state(state: GameState) -> str:
    """Serialize GameState to JSON string."""
    return state.model_dump_json(by_alias=True)


def _deserialize_state(json_str: str) -> GameState:
    """Deserialize JSON string to GameState."""
    # Normalize legacy status values before validation. Older recordings may
    # use "finished" as the terminal status; the canonical value is now
    # "completed" (mirroring the shared TS engine and canonical rules doc).
    try:
        data = json.loads(json_str)
    except Exception:
        # Fall back to the raw path if JSON is unexpectedly malformed; this
        # preserves previous behaviour for debugging while surfacing an error.
        return GameState.model_validate_json(json_str)

    status = data.get("gameStatus")
    if status == "finished":
        data["gameStatus"] = "completed"

    return GameState.model_validate(data)


def _serialize_move(move: Move) -> str:
    """Serialize Move to JSON string."""
    return move.model_dump_json(by_alias=True)


def _deserialize_move(json_str: str) -> Move:
    """Deserialize JSON string to Move."""
    return Move.model_validate_json(json_str)


def _compute_state_hash(state: GameState) -> str:
    """Compute a deterministic hash of game state for validation.

    Uses the canonical fingerprint format (matching TypeScript fingerprintGameState)
    hashed with a simple non-cryptographic hash for consistent cross-engine comparison.

    The fingerprint format is:
      meta#players#stacks#markers#collapsed
    where each component is sorted for determinism.
    """
    from app.rules.core import hash_game_state

    # Get the fingerprint (readable canonical format)
    fingerprint = hash_game_state(state)

    # Apply simple hash for compact storage (matches TS simpleHash)
    return _simple_hash(fingerprint)


def _simple_hash(s: str) -> str:
    """Simple string hash matching TypeScript simpleHash for cross-engine parity.

    Uses a modified FNV-1a-like hash. NOT cryptographically secure.
    """
    h1 = 0xDEADBEEF
    h2 = 0x41C6CE57

    for ch in s:
        code = ord(ch)
        # Python's integers are arbitrary precision, so we need to mask to 32 bits
        h1 = ((h1 ^ code) * 2654435761) & 0xFFFFFFFF
        h2 = ((h2 ^ code) * 1597334677) & 0xFFFFFFFF

    h1 = (((h1 ^ (h1 >> 16)) * 2246822507) ^ ((h2 ^ (h2 >> 13)) * 3266489909)) & 0xFFFFFFFF
    h2 = (((h2 ^ (h2 >> 16)) * 2246822507) ^ ((h1 ^ (h1 >> 13)) * 3266489909)) & 0xFFFFFFFF

    # Combine into 16 hex chars
    combined = (h2 << 32) | h1
    return format(combined, '016x')[:16]


def _fingerprint_state(state: GameState) -> str:
    """Get the readable fingerprint of a game state.

    This is useful for debugging when hashes don't match - you can compare
    the fingerprints to see exactly which component differs.
    """
    from app.rules.core import hash_game_state
    return hash_game_state(state)


class GameWriter:
    """Incremental game writer for live games.

    Use this when storing games incrementally during play rather than
    all at once after completion.

    Args:
        db: GameReplayDB instance
        game_id: Unique game identifier
        initial_state: Initial game state
        snapshot_interval: Create snapshots every N moves (default 20)
        all_snapshots: If True, store snapshot after EVERY move (for validation)
        store_history_entries: If True, store GameTrace-style history entries
    """

    def __init__(
        self,
        db: GameReplayDB,
        game_id: str,
        initial_state: GameState,
        snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
        all_snapshots: bool = False,
        store_history_entries: bool = False,
    ):
        self._db = db
        self._game_id = game_id
        self._initial_state = initial_state
        self._snapshot_interval = snapshot_interval
        self._all_snapshots = all_snapshots
        self._store_history_entries = store_history_entries
        self._move_count = 0
        self._turn_count = 0
        self._current_player = initial_state.current_player
        self._finalized = False

        # Track previous state for history entries
        self._prev_state: GameState | None = initial_state
        self._prev_state_hash: str | None = _compute_state_hash(initial_state) if store_history_entries else None

        # Create placeholder games record first (for FK constraint)
        self._db._create_placeholder_game(game_id, initial_state)

        # Store initial state
        self._db._store_initial_state(game_id, initial_state)

    def add_move(
        self,
        move: Move,
        state_after: GameState | None = None,
        state_before: GameState | None = None,
        available_moves: list[Move] | None = None,
        available_moves_count: int | None = None,
        engine_eval: float | None = None,
        engine_depth: int | None = None,
        fsm_valid: bool | None = None,
        fsm_error_code: str | None = None,
    ) -> None:
        """Add a move to the game.

        Args:
            move: The move that was played
            state_after: Optional state after the move (for snapshotting)
            state_before: Optional state before the move (for history entries)
                          If not provided but store_history_entries is True,
                          uses the tracked previous state.
            available_moves: Optional list of valid moves at state_before (for parity debugging)
            available_moves_count: Optional count of valid moves (lightweight alternative)
            engine_eval: Optional evaluation score from AI engine
            engine_depth: Optional search depth from AI engine
            fsm_valid: Optional FSM validation result (True = valid, False = invalid)
            fsm_error_code: Optional FSM error code if validation failed
        """
        if self._finalized:
            raise RuntimeError("GameWriter has been finalized")

        # Track turn changes
        if move.player != self._current_player:
            self._turn_count += 1
            self._current_player = move.player

        # Phase derivation: prefer the actual phase-at-move-time when available.
        # This preserves phase/move mismatches in recordings for canonical
        # validation, while still allowing legacy derivation from move type.
        phase_hint: str | None = None
        phase_source = state_before if state_before is not None else self._prev_state
        if phase_source is not None:
            current_phase = getattr(phase_source, "current_phase", None)
            if current_phase is not None:
                phase_hint = (
                    current_phase.value
                    if hasattr(current_phase, "value")
                    else str(current_phase)
                )

        self._db._store_move(
            game_id=self._game_id,
            move_number=self._move_count,
            turn_number=self._turn_count,
            move=move,
            phase=phase_hint,
        )

        # Handle snapshots
        should_snapshot = False
        if state_after is not None:
            if self._all_snapshots:
                # All-snapshots mode: store after every move
                should_snapshot = True
            elif self._move_count > 0 and self._move_count % self._snapshot_interval == 0:
                # Interval mode: store at configured intervals
                should_snapshot = True

        if should_snapshot and state_after is not None:
            state_hash = _compute_state_hash(state_after) if self._all_snapshots else None
            self._db._store_snapshot(
                game_id=self._game_id,
                move_number=self._move_count,
                state=state_after,
                state_hash=state_hash,
            )

        # Store history entry for GameTrace-style recording
        if self._store_history_entries and state_after is not None:
            before = state_before if state_before is not None else self._prev_state
            if before is not None:
                after_hash = _compute_state_hash(state_after)
                self._db._store_history_entry(
                    game_id=self._game_id,
                    move_number=self._move_count,
                    move=move,
                    state_before=before,
                    state_after=state_after,
                    state_hash_before=self._prev_state_hash,
                    state_hash_after=after_hash,
                    available_moves=available_moves,
                    available_moves_count=available_moves_count,
                    engine_eval=engine_eval,
                    engine_depth=engine_depth,
                    fsm_valid=fsm_valid,
                    fsm_error_code=fsm_error_code,
                )
                # Update tracked state for next move
                self._prev_state = state_after
                self._prev_state_hash = after_hash
        elif state_after is not None:
            # Keep phase tracking up to date even when history entries are skipped.
            self._prev_state = state_after

        self._move_count += 1

    def add_choice(
        self,
        move_number: int,
        choice_type: str,
        player: int,
        options: list[dict],
        selected: dict,
        reasoning: str | None = None,
    ) -> None:
        """Record a player choice."""
        if self._finalized:
            raise RuntimeError("GameWriter has been finalized")

        self._db._store_choice(
            game_id=self._game_id,
            move_number=move_number,
            choice_type=choice_type,
            player=player,
            options=options,
            selected=selected,
            reasoning=reasoning,
        )

    def finalize(
        self,
        final_state: GameState,
        metadata: dict | None = None,
    ) -> None:
        """Finalize and close the game record."""
        if self._finalized:
            raise RuntimeError("GameWriter already finalized")

        metadata = metadata or {}

        # Store final snapshot
        self._db._store_snapshot(
            game_id=self._game_id,
            move_number=self._move_count - 1,
            state=final_state,
        )

        # Store game metadata
        self._db._finalize_game(
            game_id=self._game_id,
            initial_state=self._initial_state,
            final_state=final_state,
            total_moves=self._move_count,
            total_turns=self._turn_count + 1,
            metadata=metadata,
        )

        self._finalized = True

    def abort(self) -> None:
        """Abort an incomplete game."""
        if self._finalized:
            return
        self._db._delete_game(self._game_id)
        self._finalized = True


class GameReplayDB:
    """Database interface for game storage and replay.

    Example usage:

        db = GameReplayDB("data/games/ringrift_games.db")

        # Store a complete game
        db.store_game(
            game_id="game-123",
            initial_state=initial_state,
            final_state=final_state,
            moves=move_list,
            choices=choice_list,
            metadata={"source": "self_play"},
        )

        # Query games
        games = db.query_games(board_type=BoardType.SQUARE8, limit=10)

        # Reconstruct state at move 25
        state = db.get_state_at_move("game-123", 25)

        # Get moves for replay
        moves = db.get_moves("game-123")
    """

    def __init__(
        self,
        db_path: str,
        snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
        enforce_canonical_history: bool = True,
    ):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
            snapshot_interval: Create snapshots every N moves
        """
        self._db_path = Path(db_path)
        self._snapshot_interval = snapshot_interval
        # When True, every recorded move must satisfy the canonical
        # phase↔MoveType contract encoded in app.rules.history_contract.
        # This is the default for new DBs so that self‑play/training
        # pipelines cannot silently write non‑canonical histories.
        self._enforce_canonical_history = enforce_canonical_history

        # Ensure parent directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema
        self._init_schema()

    @contextmanager
    def _get_conn(self):
        """Get a database connection with proper cleanup.

        Uses WAL mode and busy timeout for better concurrency with multiple workers.
        """
        conn = sqlite3.connect(str(self._db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        # WAL mode allows concurrent reads during writes
        conn.execute("PRAGMA journal_mode=WAL")
        # Wait up to 30 seconds for locks instead of failing immediately
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Initialize database schema with migration support."""
        with self._get_conn() as conn:
            # Check if schema_metadata table exists (indicates v2+ schema)
            has_metadata = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_metadata'"
            ).fetchone() is not None

            if not has_metadata:
                # Check if games table exists (indicates v1 schema needing migration)
                has_games = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='games'"
                ).fetchone() is not None

                if has_games:
                    # Existing v1 database - needs migration from v1
                    logger.info("Detected v1 schema, running migration to v2")
                    self._migrate_v1_to_v2(conn)
                    # Continue with further migrations if needed (v2->v3, v3->v4, etc.)
                    current_version = self._get_schema_version(conn)
                    if current_version < SCHEMA_VERSION:
                        logger.info(
                            f"Continuing migrations from v{current_version} to v{SCHEMA_VERSION}"
                        )
                        self._run_migrations(conn, current_version, SCHEMA_VERSION)
                else:
                    # Fresh database - create current schema directly
                    logger.info(f"Creating fresh v{SCHEMA_VERSION} schema")
                    conn.executescript(SCHEMA_SQL)
                    self._set_schema_version(conn, SCHEMA_VERSION)
            else:
                # Has metadata table - check version
                current_version = self._get_schema_version(conn)
                if current_version < SCHEMA_VERSION:
                    logger.info(
                        f"Schema version {current_version} < {SCHEMA_VERSION}, running migrations"
                    )
                    self._run_migrations(conn, current_version, SCHEMA_VERSION)
                elif current_version > SCHEMA_VERSION:
                    raise ValueError(
                        f"Database schema version {current_version} is newer than "
                        f"supported version {SCHEMA_VERSION}. Please upgrade the application."
                    )

    def _get_schema_version(self, conn: sqlite3.Connection) -> int:
        """Get current schema version from database."""
        try:
            row = conn.execute(
                "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
            ).fetchone()
            return int(row["value"]) if row else 1
        except sqlite3.OperationalError:
            # No schema_metadata table means v1
            return 1

    def _set_schema_version(self, conn: sqlite3.Connection, version: int) -> None:
        """Set schema version in database."""
        conn.execute(
            "INSERT OR REPLACE INTO schema_metadata (key, value) VALUES ('schema_version', ?)",
            (str(version),),
        )

    def _run_migrations(
        self, conn: sqlite3.Connection, from_version: int, to_version: int
    ) -> None:
        """Run incremental migrations from from_version to to_version."""
        for version in range(from_version + 1, to_version + 1):
            migration_method = getattr(self, f"_migrate_v{version - 1}_to_v{version}", None)
            if migration_method is None:
                raise ValueError(f"Missing migration method for v{version - 1} to v{version}")

            logger.info(f"Running migration from v{version - 1} to v{version}")
            migration_method(conn)
            self._set_schema_version(conn, version)

        logger.info(f"Schema migration complete: v{from_version} → v{to_version}")

    def _migrate_v1_to_v2(self, conn: sqlite3.Connection) -> None:
        """Migrate from schema v1 to v2.

        Adds:
        - schema_metadata table for version tracking
        - time_control_type, initial_time_ms, time_increment_ms to games
        - time_remaining_ms, engine_eval, engine_eval_type, engine_depth,
          engine_nodes, engine_pv, engine_time_ms to game_moves
        """
        logger.info("Migrating schema from v1 to v2")

        # Create schema_metadata table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # Add new columns to games table
        # SQLite requires individual ALTER TABLE statements for each column
        try:
            conn.execute(
                "ALTER TABLE games ADD COLUMN time_control_type TEXT DEFAULT 'none'"
            )
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise
            logger.debug("time_control_type column already exists")

        try:
            conn.execute("ALTER TABLE games ADD COLUMN initial_time_ms INTEGER")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise
            logger.debug("initial_time_ms column already exists")

        try:
            conn.execute("ALTER TABLE games ADD COLUMN time_increment_ms INTEGER")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise
            logger.debug("time_increment_ms column already exists")

        # Add new columns to game_moves table (if it exists)
        # Some databases (e.g., JSONL-converted) may not have game_moves table
        has_game_moves = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'"
        ).fetchone() is not None

        if has_game_moves:
            new_move_columns = [
                ("time_remaining_ms", "INTEGER"),
                ("engine_eval", "REAL"),
                ("engine_eval_type", "TEXT"),
                ("engine_depth", "INTEGER"),
                ("engine_nodes", "INTEGER"),
                ("engine_pv", "TEXT"),
                ("engine_time_ms", "INTEGER"),
            ]

            for col_name, col_type in new_move_columns:
                try:
                    conn.execute(f"ALTER TABLE game_moves ADD COLUMN {col_name} {col_type}")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        raise
                    logger.debug(f"{col_name} column already exists")
        else:
            logger.debug("game_moves table does not exist, skipping move column migration")

        # Set schema version
        self._set_schema_version(conn, 2)
        logger.info("Migration to v2 complete")

    def _migrate_v2_to_v3(self, conn: sqlite3.Connection) -> None:
        """Migrate from schema v2 to v3.

        Adds:
        - game_history_entries table for GameTrace-style storage
        - state_hash column to game_state_snapshots for validation
        """
        logger.info("Migrating schema from v2 to v3")

        # Create game_history_entries table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS game_history_entries (
                game_id TEXT NOT NULL,
                move_number INTEGER NOT NULL,
                player INTEGER NOT NULL,
                phase_before TEXT NOT NULL,
                phase_after TEXT NOT NULL,
                status_before TEXT NOT NULL,
                status_after TEXT NOT NULL,
                progress_before_json TEXT NOT NULL,
                progress_after_json TEXT NOT NULL,
                state_hash_before TEXT,
                state_hash_after TEXT,
                board_summary_before_json TEXT,
                board_summary_after_json TEXT,
                PRIMARY KEY (game_id, move_number),
                FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
            )
        """)

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_game ON game_history_entries(game_id)"
        )

        # Add state_hash column to snapshots (if table exists)
        # Check if game_state_snapshots table exists first
        has_snapshots = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='game_state_snapshots'"
        ).fetchone() is not None

        if has_snapshots:
            try:
                conn.execute(
                    "ALTER TABLE game_state_snapshots ADD COLUMN state_hash TEXT"
                )
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise
                logger.debug("state_hash column already exists")
        else:
            # Create the snapshots table if missing (partial database scenario)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS game_state_snapshots (
                    game_id TEXT NOT NULL,
                    move_number INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    compressed INTEGER DEFAULT 0,
                    state_hash TEXT,
                    PRIMARY KEY (game_id, move_number),
                    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
                )
            """)
            logger.debug("Created game_state_snapshots table (was missing)")

        self._set_schema_version(conn, 3)
        logger.info("Migration to v3 complete")

    def _migrate_v3_to_v4(self, conn: sqlite3.Connection) -> None:
        """Migrate from schema v3 to v4.

        Adds:
        - state_before_json column to game_history_entries for full state before move
        - state_after_json column to game_history_entries for full state after move
        - compressed_states column to indicate if state JSON is gzip compressed
        """
        logger.info("Migrating schema from v3 to v4")

        # Add state_before_json column
        try:
            conn.execute(
                "ALTER TABLE game_history_entries ADD COLUMN state_before_json TEXT"
            )
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise
            logger.debug("state_before_json column already exists")

        # Add state_after_json column
        try:
            conn.execute(
                "ALTER TABLE game_history_entries ADD COLUMN state_after_json TEXT"
            )
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise
            logger.debug("state_after_json column already exists")

        # Add compressed_states column (0 = uncompressed, 1 = gzip compressed)
        try:
            conn.execute(
                "ALTER TABLE game_history_entries ADD COLUMN compressed_states INTEGER DEFAULT 0"
            )
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise
            logger.debug("compressed_states column already exists")

        self._set_schema_version(conn, 4)
        logger.info("Migration to v4 complete")

    def _migrate_v4_to_v5(self, conn: sqlite3.Connection) -> None:
        """Migrate from schema v4 to v5.

        Adds:
        - metadata_json column to games to store full recording metadata as JSON.
        """
        logger.info("Migrating schema from v4 to v5")

        try:
            conn.execute("ALTER TABLE games ADD COLUMN metadata_json TEXT")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise
            logger.debug("metadata_json column already exists")

        self._set_schema_version(conn, 5)
        logger.info("Migration to v5 complete")

    def _migrate_v5_to_v6(self, conn: sqlite3.Connection) -> None:
        """Migrate from schema v5 to v6.

        Adds:
        - available_moves_json column to game_history_entries for storing enumerated
          legal moves at each state (useful for cross-engine parity debugging)
        - available_moves_count column for lightweight move counting without full
          enumeration
        - engine_eval, engine_depth columns to game_history_entries for storing
          evaluation diagnostics alongside state snapshots
        """
        logger.info("Migrating schema from v5 to v6")

        new_history_columns = [
            ("available_moves_json", "TEXT"),
            ("available_moves_count", "INTEGER"),
            ("engine_eval", "REAL"),
            ("engine_depth", "INTEGER"),
        ]

        for col_name, col_type in new_history_columns:
            try:
                conn.execute(
                    f"ALTER TABLE game_history_entries ADD COLUMN {col_name} {col_type}"
                )
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise
                logger.debug(f"{col_name} column already exists")

        self._set_schema_version(conn, 6)
        logger.info("Migration to v6 complete")

    def _migrate_v6_to_v7(self, conn: sqlite3.Connection) -> None:
        """Migrate from schema v6 to v7.

        Adds:
        - fsm_valid column to game_history_entries for FSM validation status
          (1 = valid, 0 = invalid, NULL = not checked)
        - fsm_error_code column for storing error code when fsm_valid = 0

        This supports Phase 7 (Data Pipeline) of the FSM Extension Strategy,
        enabling training data filtering based on FSM validation.
        """
        logger.info("Migrating schema from v6 to v7")

        new_history_columns = [
            ("fsm_valid", "INTEGER"),
            ("fsm_error_code", "TEXT"),
        ]

        for col_name, col_type in new_history_columns:
            try:
                conn.execute(
                    f"ALTER TABLE game_history_entries ADD COLUMN {col_name} {col_type}"
                )
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise
                logger.debug(f"{col_name} column already exists")

        self._set_schema_version(conn, 7)
        logger.info("Migration to v7 complete")

    def _migrate_v7_to_v8(self, conn: sqlite3.Connection) -> None:
        """Migrate from schema v7 to v8.

        Adds:
        - game_nnue_features table for pre-computed NNUE training features
          This enables instant training data extraction without game replay.
        """
        logger.info("Migrating schema from v7 to v8")

        # Create the NNUE features table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS game_nnue_features (
                game_id TEXT NOT NULL,
                move_number INTEGER NOT NULL,
                player_perspective INTEGER NOT NULL,
                features BLOB NOT NULL,
                value REAL NOT NULL,
                board_type TEXT NOT NULL,
                feature_dim INTEGER NOT NULL,
                PRIMARY KEY (game_id, move_number, player_perspective),
                FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
            )
        """)

        # Create indexes for efficient queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_nnue_game ON game_nnue_features(game_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_nnue_board_type ON game_nnue_features(board_type)"
        )

        self._set_schema_version(conn, 8)
        logger.info("Migration to v8 complete")

    def _migrate_v8_to_v9(self, conn: sqlite3.Connection) -> None:
        """Migrate from schema v8 to v9.

        Adds:
        - quality_score column to games table for training data prioritization
        - quality_category column to games table for quality classification
        - Indexes for quality-based sorting
        """
        logger.info("Migrating schema from v8 to v9")

        # Add quality_score column
        try:
            conn.execute("ALTER TABLE games ADD COLUMN quality_score REAL")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise

        # Add quality_category column
        try:
            conn.execute("ALTER TABLE games ADD COLUMN quality_category TEXT")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise

        # Create indexes for quality-based queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_games_quality ON games(quality_score DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_games_quality_board ON games(board_type, quality_score DESC)"
        )

        self._set_schema_version(conn, 9)
        logger.info("Migration to v9 complete")

    # =========================================================================
    # Write Operations
    # =========================================================================

    def store_game(
        self,
        game_id: str,
        initial_state: GameState,
        final_state: GameState,
        moves: list[Move],
        choices: list[dict] | None = None,
        metadata: dict | None = None,
        store_history_entries: bool = True,
        compress_states: bool = False,
        snapshot_interval: int = 20,
    ) -> None:
        """Store a complete game with all associated data.

        Args:
            game_id: Unique game identifier
            initial_state: Initial game state
            final_state: Final game state
            moves: List of all moves in order
            choices: Optional list of player choices
            metadata: Optional game metadata (source, termination_reason, etc.)
            store_history_entries: If True (default), store history entries with
                                   full before/after state snapshots for each move
            compress_states: If True, gzip compress state JSON in history entries
            snapshot_interval: Store snapshots every N moves for NNUE training (default 20)
        """
        # Import here to avoid circular imports
        from app.game_engine import GameEngine

        metadata = metadata or {}
        choices = choices or []

        with self._get_conn() as conn:
            # First insert the games record (FK parent) before child records
            self._finalize_game_conn(
                conn,
                game_id=game_id,
                initial_state=initial_state,
                final_state=final_state,
                total_moves=len(moves),
                total_turns=0,  # Will be updated below
                metadata=metadata,
            )

            # Store initial state
            self._store_initial_state_conn(conn, game_id, initial_state)

            # Store moves and create snapshots
            # Also compute and store history entries with before/after states
            turn_number = 0
            current_player = initial_state.current_player
            prev_state = initial_state
            prev_state_hash = _compute_state_hash(initial_state) if store_history_entries else None

            # Track state for snapshots even if not storing full history entries
            # This enables NNUE training from lean-db games
            need_state_tracking = store_history_entries or snapshot_interval > 0

            for i, move in enumerate(moves):
                if move.player != current_player:
                    turn_number += 1
                    current_player = move.player

                # Phase derivation: record the actual phase-at-move-time.
                # This lets canonical history checks detect phase/move mismatches.
                # Priority: use move.phase if provided (from JSONL import), else derive from state.
                phase_hint: str | None = None
                if hasattr(move, "phase") and move.phase is not None:
                    # Phase explicitly provided on the move (e.g., from canonical JSONL export)
                    phase_hint = move.phase if isinstance(move.phase, str) else str(move.phase)
                else:
                    # Fall back to deriving from current state
                    current_phase = getattr(prev_state, "current_phase", None)
                    if current_phase is not None:
                        phase_hint = (
                            current_phase.value
                            if hasattr(current_phase, "value")
                            else str(current_phase)
                        )

                self._store_move_conn(
                    conn,
                    game_id=game_id,
                    move_number=i,
                    turn_number=turn_number,
                    move=move,
                    phase=phase_hint,
                )

                # Compute state after applying this move (needed for history OR snapshots)
                state_after = None
                if need_state_tracking:
                    state_after = GameEngine.apply_move(prev_state, move)

                # Store history entry with before/after states (v4 feature)
                if store_history_entries and state_after is not None:
                    state_hash_after = _compute_state_hash(state_after)

                    self._store_history_entry_conn(
                        conn,
                        game_id=game_id,
                        move_number=i,
                        move=move,
                        state_before=prev_state,
                        state_after=state_after,
                        state_hash_before=prev_state_hash,
                        state_hash_after=state_hash_after,
                        store_full_states=True,
                        compress_states=compress_states,
                    )

                    prev_state_hash = state_hash_after

                # Store periodic snapshots for NNUE training (even in lean-db mode)
                if snapshot_interval > 0 and state_after is not None:
                    move_num = i + 1  # 1-indexed for interval check
                    if move_num % snapshot_interval == 0:
                        self._store_snapshot_conn(
                            conn,
                            game_id=game_id,
                            move_number=i,
                            state=state_after,
                        )

                # Update prev_state for next iteration
                if state_after is not None:
                    prev_state = state_after

            # Store choices
            for choice in choices:
                self._store_choice_conn(
                    conn,
                    game_id=game_id,
                    move_number=choice.get("move_number", 0),
                    choice_type=choice.get("choice_type", "unknown"),
                    player=choice.get("player", 0),
                    options=choice.get("options", []),
                    selected=choice.get("selected", {}),
                    reasoning=choice.get("reasoning"),
                )

            # Store final state as snapshot
            self._store_snapshot_conn(
                conn,
                game_id=game_id,
                move_number=len(moves) - 1 if moves else 0,
                state=final_state,
            )

            # Update games record with final turn count
            conn.execute(
                "UPDATE games SET total_turns = ? WHERE game_id = ?",
                (turn_number + 1, game_id),
            )

    def store_game_incremental(
        self,
        game_id: str | None = None,
        initial_state: GameState | None = None,
        all_snapshots: bool = False,
        store_history_entries: bool = True,
    ) -> GameWriter:
        """Begin incremental game storage.

        Args:
            game_id: Optional game ID (generated if not provided)
            initial_state: Initial game state (required)
            all_snapshots: If True, store snapshot after EVERY move for validation
            store_history_entries: If True (default), store history entries with
                                   full before/after state snapshots for each move

        Returns:
            GameWriter for incremental storage
        """
        if initial_state is None:
            raise ValueError("initial_state is required")

        if game_id is None:
            game_id = str(uuid.uuid4())

        return GameWriter(
            db=self,
            game_id=game_id,
            initial_state=initial_state,
            snapshot_interval=self._snapshot_interval,
            all_snapshots=all_snapshots,
            store_history_entries=store_history_entries,
        )

    # =========================================================================
    # Read Operations
    # =========================================================================

    def get_game_metadata(self, game_id: str) -> dict | None:
        """Get game metadata without loading full state."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM games WHERE game_id = ?",
                (game_id,),
            ).fetchone()

            if row is None:
                return None

            return dict(row)

    def get_initial_state(self, game_id: str) -> GameState | None:
        """Get the initial game state.

        Falls back to generating a default initial state from game metadata
        if the game_initial_state table doesn't exist (e.g., for consolidated DBs).
        """
        with self._get_conn() as conn:
            # Check if game_initial_state table exists
            has_table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='game_initial_state'"
            ).fetchone() is not None

            if has_table:
                row = conn.execute(
                    "SELECT initial_state_json, compressed FROM game_initial_state WHERE game_id = ?",
                    (game_id,),
                ).fetchone()

                if row is not None:
                    json_str = row["initial_state_json"]
                    if row["compressed"]:
                        json_str = _decompress_json(json_str)
                    return _deserialize_state(json_str)

            # Fallback: generate initial state from game metadata
            game_meta = conn.execute(
                "SELECT board_type, num_players FROM games WHERE game_id = ?",
                (game_id,),
            ).fetchone()

            if game_meta is None:
                return None

            # Import here to avoid circular dependency
            from app.training.generate_data import create_initial_state

            board_type_str = game_meta["board_type"]
            num_players = game_meta["num_players"]

            # Convert string to BoardType enum
            board_type = BoardType(board_type_str) if board_type_str else BoardType.SQUARE8

            return create_initial_state(board_type=board_type, num_players=num_players)

    def get_moves(
        self,
        game_id: str,
        start: int = 0,
        end: int | None = None,
    ) -> list[Move]:
        """Get moves in a range.

        Args:
            game_id: Game identifier
            start: Start move number (inclusive)
            end: End move number (exclusive), or None for all

        Returns:
            List of Move objects
        """
        with self._get_conn() as conn:
            if end is None:
                rows = conn.execute(
                    """
                    SELECT move_json FROM game_moves
                    WHERE game_id = ? AND move_number >= ?
                    ORDER BY move_number
                    """,
                    (game_id, start),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT move_json FROM game_moves
                    WHERE game_id = ? AND move_number >= ? AND move_number < ?
                    ORDER BY move_number
                    """,
                    (game_id, start, end),
                ).fetchall()

            return [_deserialize_move(row["move_json"]) for row in rows]

    def get_initial_states_batch(
        self,
        game_ids: list[str],
    ) -> dict[str, GameState | None]:
        """Get initial states for multiple games in a single query.

        This is more efficient than calling get_initial_state() for each game
        when processing many games (avoids N+1 query pattern).

        Args:
            game_ids: List of game identifiers

        Returns:
            Dict mapping game_id to GameState (or None if not found)
        """
        if not game_ids:
            return {}

        results: dict[str, GameState | None] = dict.fromkeys(game_ids)

        with self._get_conn() as conn:
            # Check if game_initial_state table exists
            has_table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='game_initial_state'"
            ).fetchone() is not None

            if has_table:
                # Batch query for initial states
                placeholders = ",".join("?" * len(game_ids))
                rows = conn.execute(
                    f"""
                    SELECT game_id, initial_state_json, compressed
                    FROM game_initial_state
                    WHERE game_id IN ({placeholders})
                    """,
                    game_ids,
                ).fetchall()

                for row in rows:
                    json_str = row["initial_state_json"]
                    if row["compressed"]:
                        json_str = _decompress_json(json_str)
                    results[row["game_id"]] = _deserialize_state(json_str)

            # For games without stored initial state, generate from metadata
            missing = [gid for gid, state in results.items() if state is None]
            if missing:
                placeholders = ",".join("?" * len(missing))
                meta_rows = conn.execute(
                    f"""
                    SELECT game_id, board_type, num_players
                    FROM games
                    WHERE game_id IN ({placeholders})
                    """,
                    missing,
                ).fetchall()

                from app.training.generate_data import create_initial_state

                for row in meta_rows:
                    board_type_str = row["board_type"]
                    board_type = BoardType(board_type_str) if board_type_str else BoardType.SQUARE8
                    results[row["game_id"]] = create_initial_state(
                        board_type=board_type,
                        num_players=row["num_players"],
                    )

        return results

    def get_moves_batch(
        self,
        game_ids: list[str],
    ) -> dict[str, list[Move]]:
        """Get moves for multiple games in a single query.

        This is more efficient than calling get_moves() for each game
        when processing many games (avoids N+1 query pattern).

        Args:
            game_ids: List of game identifiers

        Returns:
            Dict mapping game_id to list of Move objects
        """
        if not game_ids:
            return {}

        results: dict[str, list[Move]] = {gid: [] for gid in game_ids}

        with self._get_conn() as conn:
            placeholders = ",".join("?" * len(game_ids))
            rows = conn.execute(
                f"""
                SELECT game_id, move_json
                FROM game_moves
                WHERE game_id IN ({placeholders})
                ORDER BY game_id, move_number
                """,
                game_ids,
            ).fetchall()

            for row in rows:
                move = _deserialize_move(row["move_json"])
                results[row["game_id"]].append(move)

        return results

    def get_move_records(
        self,
        game_id: str,
        start: int = 0,
        end: int | None = None,
    ) -> list[dict]:
        """Get move records with full metadata including v2 fields.

        Args:
            game_id: Game identifier
            start: Start move number (inclusive)
            end: End move number (exclusive), or None for all

        Returns:
            List of move record dictionaries with all fields
        """
        with self._get_conn() as conn:
            if end is None:
                rows = conn.execute(
                    """
                    SELECT move_number, turn_number, player, phase, move_type, move_json,
                           timestamp, think_time_ms, time_remaining_ms, engine_eval,
                           engine_eval_type, engine_depth, engine_nodes, engine_pv,
                           engine_time_ms
                    FROM game_moves
                    WHERE game_id = ? AND move_number >= ?
                    ORDER BY move_number
                    """,
                    (game_id, start),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT move_number, turn_number, player, phase, move_type, move_json,
                           timestamp, think_time_ms, time_remaining_ms, engine_eval,
                           engine_eval_type, engine_depth, engine_nodes, engine_pv,
                           engine_time_ms
                    FROM game_moves
                    WHERE game_id = ? AND move_number >= ? AND move_number < ?
                    ORDER BY move_number
                    """,
                    (game_id, start, end),
                ).fetchall()

            result = []
            for row in rows:
                record = {
                    "moveNumber": row["move_number"],
                    "turnNumber": row["turn_number"],
                    "player": row["player"],
                    "phase": row["phase"],
                    "moveType": row["move_type"],
                    "move": json.loads(row["move_json"]),
                    "timestamp": row["timestamp"],
                    "thinkTimeMs": row["think_time_ms"],
                    "timeRemainingMs": row["time_remaining_ms"],
                    "engineEval": row["engine_eval"],
                    "engineEvalType": row["engine_eval_type"],
                    "engineDepth": row["engine_depth"],
                    "engineNodes": row["engine_nodes"],
                    "enginePV": json.loads(row["engine_pv"]) if row["engine_pv"] else None,
                    "engineTimeMs": row["engine_time_ms"],
                }
                result.append(record)

            return result

    def get_state_at_move(
        self,
        game_id: str,
        move_number: int,
        auto_inject: bool = True,
    ) -> GameState | None:
        """Reconstruct state at a specific move number.

        This method replays moves from the initial state using the current
        GameEngine implementation.

        All intermediate states are derived by applying the recorded move
        sequence with ``trace_mode=True`` so that:

        - Automatic forced elimination between turns is disabled, and
        - All eliminations/decisions must be represented as explicit
          moves in the history, matching TS ``traceMode`` semantics.

        Args:
            game_id: Game identifier
            move_number: The move number to reconstruct state after
            auto_inject: If True (default), automatically inject missing
                bookkeeping moves (no_territory_action, etc.) when needed
                to handle non-canonical recordings. If False, requires
                canonical recordings and will raise RuntimeError on gaps.

        Returns:
            GameState after the specified move, or None if not found.

        Raises:
            RuntimeError: If auto_inject=False and recorded moves have phase gaps.
        """
        # Import here to avoid circular imports
        from app.game_engine import GameEngine

        # Always start from the recorded initial state so that reconstructed
        # trajectories reflect the current canonical rules implementation.
        state = self.get_initial_state(game_id)
        if state is None:
            return None

        if move_number < 0:
            return state

        moves = self.get_moves(game_id, start=0, end=move_number + 1)
        for move in moves:
            # Auto-inject missing bookkeeping moves before applying this move
            if auto_inject:
                state = self._auto_inject_before_move(state, move)
            state = GameEngine.apply_move(state, move, trace_mode=True)

        return state

    def _get_game_move_count(self, game_id: str) -> int:
        """Get total number of moves recorded for a game.

        Used by get_state_at_move() to determine if we're at the
        final recorded position and should auto-inject bookkeeping moves.

        Args:
            game_id: Game identifier

        Returns:
            Total number of moves stored for this game
        """
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as count FROM game_moves WHERE game_id = ?",
                (game_id,),
            ).fetchone()
            return row["count"] if row else 0

    def _is_move_redundant_for_phase(self, state: GameState, move: Move) -> bool:
        """Check if a move is redundant (invalid) for the current phase.

        This matches TS's ts-replay-skip-redundant behavior where certain
        bookkeeping moves are skipped if they don't apply to the current phase.

        For example:
        - no_movement_action is only valid in movement/capture phases
        - no_line_action is only valid in line_processing phase
        - no_territory_action is only valid in territory_processing phase
        - no_placement_action is only valid in ring_placement phase

        If the current phase doesn't match, the move is considered redundant
        and should be skipped to match TS replay behavior.

        Args:
            state: Current game state
            move: The move to check

        Returns:
            True if the move should be skipped, False if it should be applied
        """
        current_phase = (
            state.current_phase.value
            if hasattr(state.current_phase, "value")
            else str(state.current_phase)
        )

        move_type = (
            move.type.value
            if hasattr(move.type, "value")
            else str(move.type)
        )

        # Define which phases each bookkeeping move type is valid in
        # (matching TS's skip-redundant logic)
        valid_phases = {
            "no_placement_action": ("ring_placement",),
            "no_movement_action": ("movement", "capture", "chain_capture"),
            "no_line_action": ("line_processing",),
            "no_territory_action": ("territory_processing",),
        }

        # If this move type has phase constraints, check if current phase matches
        if move_type in valid_phases and current_phase not in valid_phases[move_type]:
            # Move is not valid for current phase - skip it
            return True

        return False

    def _auto_inject_before_move(self, state: GameState, next_move: Move) -> GameState:
        """Auto-inject bookkeeping moves BEFORE applying a recorded move.

        This handles the case where the database recording is missing
        intermediate no-action moves. For example, after NO_LINE_ACTION
        the state is in territory_processing, but the next recorded move
        might be PLACE_RING for a different player. We need to inject
        NO_TERRITORY_ACTION to advance through the territory phase first.

        This is different from _auto_inject_no_action_moves which runs
        AFTER all moves have been applied (for end-of-game cleanup).

        Args:
            state: Current game state before the next recorded move.
            next_move: The next move from the database that we're about to apply.

        Returns:
            Updated game state with any necessary bookkeeping moves applied.
        """
        from app.game_engine import GameEngine
        from app.models import Move, MoveType, Position

        # Limit iterations to prevent infinite loops
        max_iterations = 10
        iterations = 0

        while iterations < max_iterations:
            iterations += 1

            current_phase = (
                state.current_phase.value
                if hasattr(state.current_phase, "value")
                else str(state.current_phase)
            )

            # Get the next move type for phase coercion checks
            next_type = (
                next_move.type.value
                if hasattr(next_move.type, "value")
                else str(next_move.type)
            )

            # swap_sides is a ring_placement-only meta-move; avoid injecting
            # no-op phase transitions around it and let phase validation handle
            # any out-of-phase records.
            if next_type == "swap_sides":
                break

            # RR-PARITY-FIX: When in ring_placement/movement but the next move
            # is forced_elimination, coerce the phase. This happens when Python's
            # phase machine recorded a forced_elimination mid-turn but the replay
            # engine has already advanced to the next turn's start phase.
            # Note: eliminate_rings_from_stack is TERRITORY_PROCESSING, not FORCED_ELIMINATION
            if current_phase in ("ring_placement", "movement") and next_type == "forced_elimination":
                from app.models import GamePhase
                state.current_phase = GamePhase.FORCED_ELIMINATION
                break  # Phase is now correct, exit loop

            # When in ring_placement/movement but the next move is a territory action
            # (eliminate_rings_from_stack, choose_territory_option, etc.), we need to
            # bridge through earlier phases first. This handles non-canonical recordings
            # that skip intermediate no-action phases.
            territory_moves = (
                "eliminate_rings_from_stack", "choose_territory_option",
                "process_territory_region", "skip_territory_processing", "no_territory_action"
            )
            if current_phase in ("ring_placement", "movement") and next_type in territory_moves:
                # Need to advance through ring_placement -> movement -> line_processing -> territory_processing
                # Continue the loop and let the specific phase handlers inject the appropriate no-action moves
                pass  # Fall through to specific phase handlers below

            # RR-PARITY-FIX-2025-12-11: When in ring_placement but the next move is NOT
            # a placement move, inject NO_PLACEMENT_ACTION to advance through
            # ring_placement. This handles both:
            # - Canonical recordings where placement was skipped (rings_in_hand == 0)
            # - Non-canonical recordings that skip multiple phases
            # Note: skip_placement is also a valid ring_placement move (voluntary skip to recovery)
            placement_moves = ("place_ring", "no_placement_action", "skip_placement")
            if current_phase == "ring_placement":
                if next_type in placement_moves:
                    # Next move is a placement move - no injection needed
                    break
                else:
                    # Next move is from a later phase - inject NO_PLACEMENT_ACTION to bridge
                    no_placement_move = Move(
                        id="auto-inject-no-placement",
                        type=MoveType.NO_PLACEMENT_ACTION,
                        player=state.current_player,
                        to=Position(x=0, y=0),
                        timestamp=datetime.now(),
                        thinkTime=0,
                        moveNumber=0,
                    )
                    state = GameEngine.apply_move(state, no_placement_move, trace_mode=True)
                    continue  # Re-check phase after injection

            # RR-PARITY-FIX-2025-12-11: When in movement phase but the next move is
            # from a later phase (line_processing, territory_processing), inject
            # NO_MOVEMENT_ACTION to advance. This bridges the gap when recordings
            # skip the movement phase (e.g., player has no stacks to move).
            # Note: skip_recovery is also a valid movement phase move
            movement_moves = (
                "move_stack", "no_movement_action", "recovery_slide",
                "overtaking_capture", "continue_capture_segment", "skip_capture",
                "skip_recovery"
            )
            if current_phase == "movement":
                if next_type not in movement_moves:
                    # Next move is from a later phase - need to bridge
                    no_movement_move = Move(
                        id="auto-inject-no-movement",
                        type=MoveType.NO_MOVEMENT_ACTION,
                        player=state.current_player,
                        to=Position(x=0, y=0),
                        timestamp=datetime.now(),
                        thinkTime=0,
                        moveNumber=0,
                    )
                    state = GameEngine.apply_move(state, no_movement_move, trace_mode=True)
                    continue  # Re-check phase after injection
                else:
                    # Next move is a movement/capture move, no bridging needed
                    break

            # RR-PARITY-FIX-2025-12-20: CAPTURE phase handling
            # When in capture phase but next move is from a later phase (line_processing,
            # territory_processing), inject SKIP_CAPTURE to advance. This bridges the gap
            # when TS selfplay recordings skip the skip_capture move (no captures available).
            capture_moves = ("overtaking_capture", "continue_capture_segment", "skip_capture")
            if current_phase == "capture":
                if next_type not in capture_moves:
                    # Next move is from a later phase - need to bridge with SKIP_CAPTURE
                    skip_capture_move = Move(
                        id="auto-inject-skip-capture",
                        type=MoveType.SKIP_CAPTURE,
                        player=state.current_player,
                        to=Position(x=0, y=0),
                        timestamp=datetime.now(),
                        thinkTime=0,
                        moveNumber=0,
                    )
                    state = GameEngine.apply_move(state, skip_capture_move, trace_mode=True)
                    continue  # Re-check phase after injection
                else:
                    # Next move is a capture move, no bridging needed
                    break

            # Check if we're in a no-action phase that needs auto-advancing
            if current_phase == "territory_processing":
                # Check if the next move is forced_elimination - if so, coerce phase
                if next_type == "forced_elimination":
                    from app.models import GamePhase
                    state.current_phase = GamePhase.FORCED_ELIMINATION
                    break
                # Check if the next move is already a territory action
                # Only auto-inject if the next move ISN'T a territory action
                # Note: eliminate_rings_from_stack and skip_territory_processing are also territory actions
                if next_type not in ("no_territory_action", "process_territory_region", "choose_territory_option", "eliminate_rings_from_stack", "skip_territory_processing"):
                    # Need to inject NO_TERRITORY_ACTION to advance
                    no_territory_move = Move(
                        id="auto-inject-no-territory",
                        type=MoveType.NO_TERRITORY_ACTION,
                        player=state.current_player,
                        to=Position(x=0, y=0),
                        timestamp=datetime.now(),
                        thinkTime=0,
                        moveNumber=0,
                    )
                    state = GameEngine.apply_move(state, no_territory_move, trace_mode=True)
                else:
                    # The next move is a territory action, don't auto-inject
                    break
            elif current_phase == "line_processing":
                # Check if we need to inject NO_LINE_ACTION
                # Only if the next move isn't already a line action
                # (next_type already computed above at loop start)
                if next_type not in ("no_line_action", "process_line", "choose_line_option", "choose_line_reward"):
                    no_line_move = Move(
                        id="auto-inject-no-line",
                        type=MoveType.NO_LINE_ACTION,
                        player=state.current_player,
                        to=Position(x=0, y=0),
                        timestamp=datetime.now(),
                        thinkTime=0,
                        moveNumber=0,
                    )
                    state = GameEngine.apply_move(state, no_line_move, trace_mode=True)
                else:
                    # The next move is a line action, don't auto-inject
                    break
            else:
                # Not in a no-action phase, we're good
                break

        return state

    def _auto_inject_no_action_moves(self, state: GameState) -> GameState:
        """Auto-inject NO_LINE_ACTION and NO_TERRITORY_ACTION bookkeeping moves.

        This helper matches TS's replay behavior where the orchestrator
        auto-generates these moves to complete turn traversal through
        phases that have no interactive options.

        Args:
            state: Current game state after applying a recorded move.

        Returns:
            Updated game state with bookkeeping moves auto-applied.
        """
        from app.game_engine import GameEngine, PhaseRequirementType

        # Limit iterations to prevent infinite loops in case of bugs
        max_iterations = 10
        iterations = 0

        while iterations < max_iterations:
            iterations += 1

            # Exit if game is not active
            status_value = (
                state.game_status.value
                if hasattr(state.game_status, "value")
                else str(state.game_status)
            )
            if status_value != "active":
                break

            # Check if there's a phase requirement for the current player
            requirement = GameEngine.get_phase_requirement(
                state, state.current_player
            )

            if requirement is None:
                # Interactive moves exist, stop auto-advancing
                break

            # Only auto-inject for line and territory no-action phases
            # These are the phases where TS auto-advances during replay
            if requirement.type == PhaseRequirementType.NO_LINE_ACTION_REQUIRED or requirement.type == PhaseRequirementType.NO_TERRITORY_ACTION_REQUIRED:
                bookkeeping = GameEngine.synthesize_bookkeeping_move(
                    requirement, state
                )
                state = GameEngine.apply_move(state, bookkeeping, trace_mode=True)
            else:
                # Other requirements (NO_PLACEMENT, NO_MOVEMENT, FORCED_ELIMINATION)
                # are not auto-injected during replay - they should be explicit
                # moves in the database if required, or handled differently
                break

        return state

    def get_choices_at_move(
        self,
        game_id: str,
        move_number: int,
    ) -> list[dict]:
        """Get player choices made at a specific move."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT choice_type, player, options_json, selected_option_json, ai_reasoning
                FROM game_choices
                WHERE game_id = ? AND move_number = ?
                """,
                (game_id, move_number),
            ).fetchall()

            return [
                {
                    "choice_type": row["choice_type"],
                    "player": row["player"],
                    "options": json.loads(row["options_json"]),
                    "selected": json.loads(row["selected_option_json"]),
                    "reasoning": row["ai_reasoning"],
                }
                for row in rows
            ]

    def get_history_entry(
        self,
        game_id: str,
        move_number: int,
    ) -> dict | None:
        """Get a history entry with before/after states for a specific move.

        Returns a dictionary with:
        - move_number: The move number
        - player: The player who made the move
        - phase_before/phase_after: Game phase before/after the move
        - status_before/status_after: Game status before/after the move
        - progress_before/progress_after: Player progress snapshots
        - state_hash_before/state_hash_after: State hashes for validation
        - state_before/state_after: Full GameState objects (if stored, v4+)
        - available_moves: List of valid moves at state_before (if stored, v6+)
        - available_moves_count: Count of valid moves (if stored, v6+)
        - engine_eval: Engine evaluation score (if stored, v6+)
        - engine_depth: Engine search depth (if stored, v6+)

        Returns None if no history entry exists for this move.
        """
        with self._get_conn() as conn:
            row = conn.execute(
                """
                SELECT move_number, player, phase_before, phase_after,
                       status_before, status_after, progress_before_json,
                       progress_after_json, state_hash_before, state_hash_after,
                       state_before_json, state_after_json, compressed_states,
                       available_moves_json, available_moves_count,
                       engine_eval, engine_depth
                FROM game_history_entries
                WHERE game_id = ? AND move_number = ?
                """,
                (game_id, move_number),
            ).fetchone()

            if row is None:
                return None

            result = {
                "move_number": row["move_number"],
                "player": row["player"],
                "phase_before": row["phase_before"],
                "phase_after": row["phase_after"],
                "status_before": row["status_before"],
                "status_after": row["status_after"],
                "progress_before": json.loads(row["progress_before_json"]),
                "progress_after": json.loads(row["progress_after_json"]),
                "state_hash_before": row["state_hash_before"],
                "state_hash_after": row["state_hash_after"],
                "state_before": None,
                "state_after": None,
                "available_moves": None,
                "available_moves_count": row["available_moves_count"],
                "engine_eval": row["engine_eval"],
                "engine_depth": row["engine_depth"],
            }

            # Deserialize full states if present (v4+ feature)
            if row["state_before_json"]:
                state_json = row["state_before_json"]
                if row["compressed_states"]:
                    import base64
                    state_json = _decompress_json(base64.b64decode(state_json))
                result["state_before"] = _deserialize_state(state_json)

            if row["state_after_json"]:
                state_json = row["state_after_json"]
                if row["compressed_states"]:
                    import base64
                    state_json = _decompress_json(base64.b64decode(state_json))
                result["state_after"] = _deserialize_state(state_json)

            # Deserialize available moves if present (v6+ feature)
            if row["available_moves_json"]:
                moves_data = json.loads(row["available_moves_json"])
                result["available_moves"] = [
                    _deserialize_move(m) if isinstance(m, str) else Move.model_validate(m)
                    for m in moves_data
                ]

            return result

    def get_all_history_entries(
        self,
        game_id: str,
        include_full_states: bool = True,
        include_available_moves: bool = False,
    ) -> list[dict]:
        """Get all history entries for a game.

        Args:
            game_id: Game identifier
            include_full_states: If True, deserialize and include full GameState
                                objects (may be memory-intensive for large games)
            include_available_moves: If True, deserialize and include available
                                     moves lists (v6+ feature)

        Returns a list of history entry dictionaries, ordered by move number.
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT move_number, player, phase_before, phase_after,
                       status_before, status_after, progress_before_json,
                       progress_after_json, state_hash_before, state_hash_after,
                       state_before_json, state_after_json, compressed_states,
                       available_moves_json, available_moves_count,
                       engine_eval, engine_depth
                FROM game_history_entries
                WHERE game_id = ?
                ORDER BY move_number
                """,
                (game_id,),
            ).fetchall()

            results = []
            for row in rows:
                entry = {
                    "move_number": row["move_number"],
                    "player": row["player"],
                    "phase_before": row["phase_before"],
                    "phase_after": row["phase_after"],
                    "status_before": row["status_before"],
                    "status_after": row["status_after"],
                    "progress_before": json.loads(row["progress_before_json"]),
                    "progress_after": json.loads(row["progress_after_json"]),
                    "state_hash_before": row["state_hash_before"],
                    "state_hash_after": row["state_hash_after"],
                    "state_before": None,
                    "state_after": None,
                    "available_moves": None,
                    "available_moves_count": row["available_moves_count"],
                    "engine_eval": row["engine_eval"],
                    "engine_depth": row["engine_depth"],
                }

                if include_full_states:
                    if row["state_before_json"]:
                        state_json = row["state_before_json"]
                        if row["compressed_states"]:
                            import base64
                            state_json = _decompress_json(base64.b64decode(state_json))
                        entry["state_before"] = _deserialize_state(state_json)

                    if row["state_after_json"]:
                        state_json = row["state_after_json"]
                        if row["compressed_states"]:
                            import base64
                            state_json = _decompress_json(base64.b64decode(state_json))
                        entry["state_after"] = _deserialize_state(state_json)

                if include_available_moves and row["available_moves_json"]:
                    moves_data = json.loads(row["available_moves_json"])
                    entry["available_moves"] = [
                        _deserialize_move(m) if isinstance(m, str) else Move.model_validate(m)
                        for m in moves_data
                    ]

                results.append(entry)

            return results

    # =========================================================================
    # Query Operations
    # =========================================================================

    def query_games(
        self,
        board_type: BoardType | None = None,
        num_players: int | None = None,
        winner: int | None = None,
        termination_reason: str | None = None,
        source: str | None = None,
        min_moves: int | None = None,
        max_moves: int | None = None,
        limit: int = 100,
        offset: int = 0,
        exclude_training_excluded: bool = True,
        require_moves: bool = False,
    ) -> list[dict]:
        """Query games by metadata filters.

        Returns list of game metadata dictionaries matching filters.

        Args:
            exclude_training_excluded: If True, exclude games marked with
                excluded_from_training=1 (e.g., timeout games). Default True.
            require_moves: If True, only return games that have at least one
                move in the game_moves table. Useful for consolidated DBs
                where some games may have metadata but no move data.
        """
        conditions = []
        params = []

        if board_type is not None:
            conditions.append("board_type = ?")
            # Handle both BoardType enum and string values
            params.append(board_type.value if hasattr(board_type, 'value') else str(board_type))

        if num_players is not None:
            conditions.append("num_players = ?")
            params.append(num_players)

        if winner is not None:
            conditions.append("winner = ?")
            params.append(winner)

        if termination_reason is not None:
            conditions.append("termination_reason = ?")
            params.append(termination_reason)

        if source is not None:
            conditions.append("source = ?")
            params.append(source)

        if min_moves is not None:
            conditions.append("total_moves >= ?")
            params.append(min_moves)

        if max_moves is not None:
            conditions.append("total_moves <= ?")
            params.append(max_moves)

        if require_moves:
            conditions.append(
                "EXISTS (SELECT 1 FROM game_moves m WHERE m.game_id = games.game_id)"
            )

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self._get_conn() as conn:
            # Check if excluded_from_training column exists before using it
            if exclude_training_excluded:
                cursor = conn.execute("PRAGMA table_info(games)")
                columns = {row[1] for row in cursor.fetchall()}
                if "excluded_from_training" in columns:
                    if where_clause == "1=1":
                        where_clause = "COALESCE(excluded_from_training, 0) = 0"
                    else:
                        where_clause += " AND COALESCE(excluded_from_training, 0) = 0"
            rows = conn.execute(
                f"""
                SELECT * FROM games
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()

            return [dict(row) for row in rows]

    def iterate_games(
        self,
        batch_size: int = 100,
        **filters,
    ) -> Iterator[tuple[dict, GameState, list[Move]]]:
        """Iterate over games matching filters.

        Uses batch loading to avoid N+1 query pattern. Games are loaded
        in batches of `batch_size` for efficient database access.

        Args:
            batch_size: Number of games to load per batch (default 100)
            **filters: Query filters (board_type, num_players, etc.)

        Yields:
            (metadata, initial_state, moves) tuples for each game.
        """
        # Remove limit from filters if present, use default of 10000
        filters_copy = dict(filters)
        limit = filters_copy.pop("limit", 10000)
        games = self.query_games(**filters_copy, limit=limit)

        # Process games in batches to avoid N+1 queries
        for i in range(0, len(games), batch_size):
            batch = games[i:i + batch_size]
            game_ids = [g["game_id"] for g in batch]

            # Batch load initial states and moves (3 queries per batch instead of 2N)
            initial_states = self.get_initial_states_batch(game_ids)
            moves_map = self.get_moves_batch(game_ids)

            for game_meta in batch:
                game_id = game_meta["game_id"]
                initial_state = initial_states.get(game_id)
                if initial_state is None:
                    continue
                moves = moves_map.get(game_id, [])
                yield game_meta, initial_state, moves

    def get_game_count(
        self,
        board_type: BoardType | None = None,
        num_players: int | None = None,
        winner: int | None = None,
        termination_reason: str | None = None,
        source: str | None = None,
        min_moves: int | None = None,
        max_moves: int | None = None,
        exclude_training_excluded: bool = True,
    ) -> int:
        """Get count of games matching filters.

        Args:
            exclude_training_excluded: If True, exclude games marked with
                excluded_from_training=1 (e.g., timeout games). Default True.
        """
        conditions = []
        params: list[Any] = []

        if board_type is not None:
            conditions.append("board_type = ?")
            params.append(board_type.value if isinstance(board_type, BoardType) else board_type)

        if num_players is not None:
            conditions.append("num_players = ?")
            params.append(num_players)

        if winner is not None:
            conditions.append("winner = ?")
            params.append(winner)

        if termination_reason is not None:
            conditions.append("termination_reason = ?")
            params.append(termination_reason)

        if source is not None:
            conditions.append("source = ?")
            params.append(source)

        if min_moves is not None:
            conditions.append("total_moves >= ?")
            params.append(min_moves)

        if max_moves is not None:
            conditions.append("total_moves <= ?")
            params.append(max_moves)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self._get_conn() as conn:
            # Check if excluded_from_training column exists before using it
            if exclude_training_excluded:
                cursor = conn.execute("PRAGMA table_info(games)")
                columns = {row[1] for row in cursor.fetchall()}
                if "excluded_from_training" in columns:
                    if where_clause == "1=1":
                        where_clause = "COALESCE(excluded_from_training, 0) = 0"
                    else:
                        where_clause += " AND COALESCE(excluded_from_training, 0) = 0"

            row = conn.execute(
                f"SELECT COUNT(*) as count FROM games WHERE {where_clause}",
                params,
            ).fetchone()
            return row["count"] if row else 0

    def get_game_with_players(self, game_id: str) -> dict | None:
        """Get game metadata including player details."""
        with self._get_conn() as conn:
            game_row = conn.execute(
                "SELECT * FROM games WHERE game_id = ?",
                (game_id,),
            ).fetchone()

            if game_row is None:
                return None

            player_rows = conn.execute(
                """
                SELECT player_number, player_type, ai_type, ai_difficulty,
                       final_eliminated_rings, final_territory_spaces, final_rings_in_hand
                FROM game_players
                WHERE game_id = ?
                ORDER BY player_number
                """,
                (game_id,),
            ).fetchall()

            game_dict = dict(game_row)
            game_dict["players"] = [
                {
                    "playerNumber": row["player_number"],
                    "playerType": row["player_type"],
                    "aiType": row["ai_type"],
                    "aiDifficulty": row["ai_difficulty"],
                    "finalEliminatedRings": row["final_eliminated_rings"],
                    "finalTerritorySpaces": row["final_territory_spaces"],
                    "finalRingsInHand": row["final_rings_in_hand"],
                }
                for row in player_rows
            ]

            return game_dict

    def get_stats(self) -> dict:
        """Get database statistics."""
        with self._get_conn() as conn:
            total_games = conn.execute(
                "SELECT COUNT(*) FROM games"
            ).fetchone()[0]

            games_by_board = conn.execute(
                "SELECT board_type, COUNT(*) as count FROM games GROUP BY board_type"
            ).fetchall()

            games_by_status = conn.execute(
                "SELECT game_status, COUNT(*) as count FROM games GROUP BY game_status"
            ).fetchall()

            games_by_termination = conn.execute(
                "SELECT termination_reason, COUNT(*) as count FROM games GROUP BY termination_reason"
            ).fetchall()

            total_moves = conn.execute(
                "SELECT COUNT(*) FROM game_moves"
            ).fetchone()[0]

            # Get schema version
            schema_version = self._get_schema_version(conn)

            return {
                "total_games": total_games,
                "games_by_board_type": {
                    row["board_type"]: row["count"] for row in games_by_board
                },
                "games_by_status": {
                    row["game_status"]: row["count"] for row in games_by_status
                },
                "games_by_termination": {
                    row["termination_reason"]: row["count"]
                    for row in games_by_termination
                    if row["termination_reason"]
                },
                "total_moves": total_moves,
                "schema_version": schema_version,
            }

    def vacuum(self) -> None:
        """Optimize database storage."""
        with self._get_conn() as conn:
            conn.execute("VACUUM")

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _create_placeholder_game(self, game_id: str, initial_state: GameState) -> None:
        """Create a placeholder games record for incremental writing."""
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO games
                (game_id, board_type, num_players, rng_seed, created_at, game_status,
                 total_moves, total_turns, source, schema_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game_id,
                    initial_state.board_type.value,
                    len(initial_state.players),
                    initial_state.rng_seed,
                    initial_state.created_at.isoformat(),
                    "active",  # Placeholder status
                    0,
                    0,
                    "unknown",
                    SCHEMA_VERSION,
                ),
            )

    def _store_initial_state(self, game_id: str, state: GameState) -> None:
        """Store initial state (standalone transaction)."""
        with self._get_conn() as conn:
            self._store_initial_state_conn(conn, game_id, state)

    def _store_initial_state_conn(
        self,
        conn: sqlite3.Connection,
        game_id: str,
        state: GameState,
    ) -> None:
        """Store initial state (within existing transaction)."""
        json_str = _serialize_state(state)
        conn.execute(
            """
            INSERT OR REPLACE INTO game_initial_state (game_id, initial_state_json, compressed)
            VALUES (?, ?, 0)
            """,
            (game_id, json_str),
        )

    def _store_move(
        self,
        game_id: str,
        move_number: int,
        turn_number: int,
        move: Move,
        phase: str | None = None,
        *,
        time_remaining_ms: int | None = None,
        engine_eval: float | None = None,
        engine_eval_type: str | None = None,
        engine_depth: int | None = None,
        engine_nodes: int | None = None,
        engine_pv: list[str] | None = None,
        engine_time_ms: int | None = None,
    ) -> None:
        """Store a single move (standalone transaction)."""
        with self._get_conn() as conn:
            self._store_move_conn(
                conn,
                game_id,
                move_number,
                turn_number,
                move,
                phase=phase,
                time_remaining_ms=time_remaining_ms,
                engine_eval=engine_eval,
                engine_eval_type=engine_eval_type,
                engine_depth=engine_depth,
                engine_nodes=engine_nodes,
                engine_pv=engine_pv,
                engine_time_ms=engine_time_ms,
            )

    def _store_move_conn(
        self,
        conn: sqlite3.Connection,
        game_id: str,
        move_number: int,
        turn_number: int,
        move: Move,
        phase: str | None = None,
        *,
        time_remaining_ms: int | None = None,
        engine_eval: float | None = None,
        engine_eval_type: str | None = None,
        engine_depth: int | None = None,
        engine_nodes: int | None = None,
        engine_pv: list[str] | None = None,
        engine_time_ms: int | None = None,
    ) -> None:
        """Store a single move (within existing transaction).

        Args:
            conn: Database connection
            game_id: Game identifier
            move_number: Move sequence number (0-indexed)
            turn_number: Turn number this move belongs to
            move: The Move object
            phase: Canonical phase string for this move (phase *during* which
                the move occurs). When None/empty, phase is inferred from
                move.type for legacy recordings.
            time_remaining_ms: Clock time remaining after this move (v2)
            engine_eval: Engine evaluation score (v2)
            engine_eval_type: Type of evaluation ('heuristic', 'neural', 'mcts_winrate') (v2)
            engine_depth: Search depth (v2)
            engine_nodes: Nodes searched (v2)
            engine_pv: Principal variation as list of move strings (v2)
            engine_time_ms: Time spent computing this move in ms (v2)
        """
        # Enforce canonical (phase, move_type) contract at write time for all
        # new recordings. We thread the *actual* phase-at-move-time through
        # from the engine/recorder when available so that territory moves can
        # only be written during TERRITORY_PROCESSING and forced_elimination
        # moves only during FORCED_ELIMINATION, mirroring the TS FSM.
        phase_hint = (phase or "").strip()
        check = validate_canonical_move(phase_hint, move.type.value)

        if self._enforce_canonical_history and not check.ok and check.reason:
            raise ValueError(
                f"Attempted to record non-canonical move "
                f"in GameReplayDB {self._db_path}: {check.reason}"
            )

        # Store the effective canonical phase derived by the contract helper.
        phase_to_store = check.effective_phase

        conn.execute(
            """
            INSERT INTO game_moves
            (game_id, move_number, turn_number, player, phase, move_type, move_json,
             timestamp, think_time_ms, time_remaining_ms, engine_eval, engine_eval_type,
             engine_depth, engine_nodes, engine_pv, engine_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                game_id,
                move_number,
                turn_number,
                move.player,
                phase_to_store,
                move.type.value,
                _serialize_move(move),
                move.timestamp.isoformat() if move.timestamp else None,
                move.think_time,
                time_remaining_ms,
                engine_eval,
                engine_eval_type,
                engine_depth,
                engine_nodes,
                json.dumps(engine_pv) if engine_pv else None,
                engine_time_ms,
            ),
        )

    def _store_snapshot(
        self,
        game_id: str,
        move_number: int,
        state: GameState,
        state_hash: str | None = None,
    ) -> None:
        """Store a state snapshot (standalone transaction)."""
        with self._get_conn() as conn:
            self._store_snapshot_conn(conn, game_id, move_number, state, state_hash)

    def _store_snapshot_conn(
        self,
        conn: sqlite3.Connection,
        game_id: str,
        move_number: int,
        state: GameState,
        state_hash: str | None = None,
    ) -> None:
        """Store a state snapshot (within existing transaction)."""
        json_str = _serialize_state(state)
        # Ensure state_hash is populated so that all snapshots (not just
        # all-snapshots mode) can participate in cross-engine validation.
        hash_value = state_hash if state_hash is not None else _compute_state_hash(state)
        conn.execute(
            """
            INSERT OR REPLACE INTO game_state_snapshots
            (game_id, move_number, state_json, compressed, state_hash)
            VALUES (?, ?, ?, 0, ?)
            """,
            (game_id, move_number, json_str, hash_value),
        )

    def _store_history_entry(
        self,
        game_id: str,
        move_number: int,
        move: Move,
        state_before: GameState,
        state_after: GameState,
        state_hash_before: str | None = None,
        state_hash_after: str | None = None,
        available_moves: list[Move] | None = None,
        available_moves_count: int | None = None,
        engine_eval: float | None = None,
        engine_depth: int | None = None,
        fsm_valid: bool | None = None,
        fsm_error_code: str | None = None,
    ) -> None:
        """Store a history entry for GameTrace-style recording."""
        with self._get_conn() as conn:
            self._store_history_entry_conn(
                conn, game_id, move_number, move,
                state_before, state_after, state_hash_before, state_hash_after,
                available_moves=available_moves,
                available_moves_count=available_moves_count,
                engine_eval=engine_eval,
                engine_depth=engine_depth,
                fsm_valid=fsm_valid,
                fsm_error_code=fsm_error_code,
            )

    def _store_history_entry_conn(
        self,
        conn: sqlite3.Connection,
        game_id: str,
        move_number: int,
        move: Move,
        state_before: GameState,
        state_after: GameState,
        state_hash_before: str | None = None,
        state_hash_after: str | None = None,
        store_full_states: bool = True,
        compress_states: bool = False,
        available_moves: list[Move] | None = None,
        available_moves_count: int | None = None,
        engine_eval: float | None = None,
        engine_depth: int | None = None,
        fsm_valid: bool | None = None,
        fsm_error_code: str | None = None,
    ) -> None:
        """Store a history entry (within existing transaction).

        Args:
            conn: Database connection
            game_id: Game identifier
            move_number: Move sequence number (0-indexed)
            move: The Move object
            state_before: Full GameState before the move
            state_after: Full GameState after the move
            state_hash_before: Optional pre-computed hash of state_before
            state_hash_after: Optional pre-computed hash of state_after
            store_full_states: If True (default), store full state JSON before/after
            compress_states: If True, gzip compress state JSON (default: False)
            available_moves: Optional list of valid moves at state_before (for parity debugging)
            available_moves_count: Optional count of valid moves (lightweight alternative)
            engine_eval: Optional evaluation score from AI engine
            engine_depth: Optional search depth from AI engine
            fsm_valid: Optional FSM validation result (True = valid, False = invalid)
            fsm_error_code: Optional FSM error code if validation failed
        """
        # Build progress snapshots
        progress_before = {
            "players": [
                {
                    "playerNumber": p.player_number,
                    "eliminatedRings": p.eliminated_rings,
                    "territorySpaces": p.territory_spaces,
                    "ringsInHand": p.rings_in_hand,
                }
                for p in state_before.players
            ]
        }
        progress_after = {
            "players": [
                {
                    "playerNumber": p.player_number,
                    "eliminatedRings": p.eliminated_rings,
                    "territorySpaces": p.territory_spaces,
                    "ringsInHand": p.rings_in_hand,
                }
                for p in state_after.players
            ]
        }

        # Serialize full states if requested (v4 feature)
        state_before_json: str | None = None
        state_after_json: str | None = None
        compressed_flag = 0

        if store_full_states:
            before_str = _serialize_state(state_before)
            after_str = _serialize_state(state_after)
            if compress_states:
                # Store as base64-encoded gzip for text column compatibility
                import base64
                state_before_json = base64.b64encode(_compress_json(before_str)).decode("ascii")
                state_after_json = base64.b64encode(_compress_json(after_str)).decode("ascii")
                compressed_flag = 1
            else:
                state_before_json = before_str
                state_after_json = after_str

        # Serialize available moves if provided
        available_moves_json: str | None = None
        if available_moves is not None:
            available_moves_json = json.dumps([
                _serialize_move(m) for m in available_moves
            ])

        # Convert fsm_valid bool to SQLite integer (1/0/NULL)
        fsm_valid_int: int | None = None
        if fsm_valid is not None:
            fsm_valid_int = 1 if fsm_valid else 0

        conn.execute(
            """
            INSERT OR REPLACE INTO game_history_entries
            (game_id, move_number, player, phase_before, phase_after,
             status_before, status_after, progress_before_json, progress_after_json,
             state_hash_before, state_hash_after, state_before_json, state_after_json,
             compressed_states, available_moves_json, available_moves_count,
             engine_eval, engine_depth, fsm_valid, fsm_error_code)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                game_id,
                move_number,
                move.player,
                state_before.current_phase.value,
                state_after.current_phase.value,
                state_before.game_status.value,
                state_after.game_status.value,
                json.dumps(progress_before),
                json.dumps(progress_after),
                state_hash_before,
                state_hash_after,
                state_before_json,
                state_after_json,
                compressed_flag,
                available_moves_json,
                available_moves_count,
                engine_eval,
                engine_depth,
                fsm_valid_int,
                fsm_error_code,
            ),
        )

    def _store_choice(
        self,
        game_id: str,
        move_number: int,
        choice_type: str,
        player: int,
        options: list[dict],
        selected: dict,
        reasoning: str | None = None,
    ) -> None:
        """Store a player choice (standalone transaction)."""
        with self._get_conn() as conn:
            self._store_choice_conn(
                conn, game_id, move_number, choice_type, player, options, selected, reasoning
            )

    def _store_choice_conn(
        self,
        conn: sqlite3.Connection,
        game_id: str,
        move_number: int,
        choice_type: str,
        player: int,
        options: list[dict],
        selected: dict,
        reasoning: str | None = None,
    ) -> None:
        """Store a player choice (within existing transaction)."""
        conn.execute(
            """
            INSERT OR REPLACE INTO game_choices
            (game_id, move_number, choice_type, player, options_json, selected_option_json, ai_reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                game_id,
                move_number,
                choice_type,
                player,
                json.dumps(options),
                json.dumps(selected),
                reasoning,
            ),
        )

    def _finalize_game(
        self,
        game_id: str,
        initial_state: GameState,
        final_state: GameState,
        total_moves: int,
        total_turns: int,
        metadata: dict,
    ) -> None:
        """Finalize game metadata (standalone transaction)."""
        with self._get_conn() as conn:
            self._finalize_game_conn(
                conn, game_id, initial_state, final_state, total_moves, total_turns, metadata
            )

    def _finalize_game_conn(
        self,
        conn: sqlite3.Connection,
        game_id: str,
        initial_state: GameState,
        final_state: GameState,
        total_moves: int,
        total_turns: int,
        metadata: dict,
    ) -> None:
        """Finalize game metadata (within existing transaction)."""
        metadata = metadata or {}

        # Calculate duration if timestamps available
        duration_ms = None
        if initial_state.created_at and final_state.last_move_at:
            # Normalize both datetimes to UTC to handle mixed timezone-aware/naive
            created = initial_state.created_at
            ended = final_state.last_move_at
            # If created_at is naive, assume UTC
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            # If last_move_at is naive, assume UTC
            if ended.tzinfo is None:
                ended = ended.replace(tzinfo=timezone.utc)
            duration = ended - created
            duration_ms = int(duration.total_seconds() * 1000)

        # Serialize full metadata dict for long-term debugging/analytics.
        metadata_json = json.dumps(metadata, sort_keys=True)

        # Determine termination reason
        termination_reason = metadata.get("termination_reason")
        if termination_reason is None and final_state.winner is not None:
            # Infer from final state using canonical victory condition ladder:
            # 1. Ring elimination (victory_threshold reached)
            # 2. Territory (territory_victory_threshold reached)
            # 3. LPS - Last Player Standing (lps_exclusive_player_for_completed_round)
            # 4. Structural (no stacks remaining)
            winner = final_state.winner

            # Check ring elimination victory
            elim_rings = final_state.board.eliminated_rings
            eliminated_for_winner = elim_rings.get(str(winner), 0)
            if eliminated_for_winner >= final_state.victory_threshold:
                termination_reason = "ring_elimination"
            else:
                # Check territory victory via collapsed_spaces count
                territory_counts: dict[int, int] = {}
                for p_id in final_state.board.collapsed_spaces.values():
                    territory_counts[p_id] = territory_counts.get(p_id, 0) + 1
                if territory_counts.get(winner, 0) >= final_state.territory_victory_threshold:
                    termination_reason = "territory"
                # Check LPS victory (R172 - sole player with real actions)
                elif final_state.lps_exclusive_player_for_completed_round == winner:
                    termination_reason = "last_player_standing"
                # Structural termination (no stacks remaining)
                elif not final_state.board.stacks:
                    termination_reason = "structural"

        # Compute quality score for training data prioritization
        quality_score = None
        quality_category = None
        try:
            from app.quality.unified_quality import (
                compute_game_quality_from_params,
                get_quality_category,
            )
            quality = compute_game_quality_from_params(
                game_id=game_id,
                game_status=final_state.game_status.value,
                winner=final_state.winner,
                termination_reason=termination_reason,
                total_moves=total_moves,
                board_type=initial_state.board_type.value,
                source=metadata.get("source"),
            )
            quality_score = quality.quality_score
            quality_category = get_quality_category(quality.quality_score).value
        except ImportError:
            pass  # Quality scorer not available
        except Exception as e:
            logger.warning(f"Failed to compute game quality: {e}")

        # Check if game already exists (incremental write case)
        existing = conn.execute(
            "SELECT game_id FROM games WHERE game_id = ?", (game_id,)
        ).fetchone()

        if existing:
            # Update existing record (preserves FK relationships)
            conn.execute(
                """
                UPDATE games SET
                    completed_at = ?,
                    game_status = ?,
                    winner = ?,
                    termination_reason = ?,
                    total_moves = ?,
                    total_turns = ?,
                    duration_ms = ?,
                    source = ?,
                    metadata_json = ?,
                    quality_score = ?,
                    quality_category = ?
                WHERE game_id = ?
                """,
                (
                    final_state.last_move_at.isoformat() if final_state.last_move_at else None,
                    final_state.game_status.value,
                    final_state.winner,
                    termination_reason,
                    total_moves,
                    total_turns,
                    duration_ms,
                    metadata.get("source", "unknown"),
                    metadata_json,
                    quality_score,
                    quality_category,
                    game_id,
                ),
            )
        else:
            # Insert new game record
            conn.execute(
                """
                INSERT INTO games
                (game_id, board_type, num_players, rng_seed, created_at, completed_at,
                 game_status, winner, termination_reason, total_moves, total_turns,
                 duration_ms, source, schema_version, metadata_json,
                 quality_score, quality_category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game_id,
                    initial_state.board_type.value,
                    len(initial_state.players),
                    initial_state.rng_seed,
                    initial_state.created_at.isoformat(),
                    final_state.last_move_at.isoformat() if final_state.last_move_at else None,
                    final_state.game_status.value,
                    final_state.winner,
                    termination_reason,
                    total_moves,
                    total_turns,
                    duration_ms,
                    metadata.get("source", "unknown"),
                    SCHEMA_VERSION,
                    metadata_json,
                    quality_score,
                    quality_category,
                ),
            )

        # Insert player records
        for player in final_state.players:
            ai_type = None
            ai_difficulty = None
            ai_profile_id = None

            if player.type == "ai":
                ai_type = metadata.get(f"player_{player.player_number}_ai_type", "heuristic")
                ai_difficulty = player.ai_difficulty
                ai_profile_id = metadata.get(f"player_{player.player_number}_profile_id")

            conn.execute(
                """
                INSERT OR REPLACE INTO game_players
                (game_id, player_number, player_type, ai_type, ai_difficulty, ai_profile_id,
                 final_eliminated_rings, final_territory_spaces, final_rings_in_hand)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game_id,
                    player.player_number,
                    player.type,
                    ai_type,
                    ai_difficulty,
                    ai_profile_id,
                    player.eliminated_rings,
                    player.territory_spaces,
                    player.rings_in_hand,
                ),
            )

    def _delete_game(self, game_id: str) -> None:
        """Delete a game and all associated data."""
        with self._get_conn() as conn:
            # Foreign key cascade handles related tables
            conn.execute("DELETE FROM games WHERE game_id = ?", (game_id,))

    # =========================================================================
    # NNUE Features Cache Operations
    # =========================================================================

    def store_nnue_features(
        self,
        game_id: str,
        move_number: int,
        player_perspective: int,
        features: np.ndarray,
        value: float,
        board_type: str,
    ) -> None:
        """Store pre-computed NNUE features for a game position.

        Args:
            game_id: Game identifier
            move_number: Move number (0-indexed)
            player_perspective: Player number for perspective rotation
            features: Float32 feature vector (will be compressed)
            value: Win/loss label (-1, 0, +1)
            board_type: Board type for validation
        """
        import numpy as np

        # Compress features using gzip
        features_bytes = gzip.compress(features.astype(np.float32).tobytes())

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO game_nnue_features
                (game_id, move_number, player_perspective, features, value, board_type, feature_dim)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game_id,
                    move_number,
                    player_perspective,
                    features_bytes,
                    value,
                    board_type,
                    len(features),
                ),
            )

    def store_nnue_features_batch(
        self,
        records: list[tuple[str, int, int, np.ndarray, float, str]],
    ) -> int:
        """Store multiple NNUE feature records efficiently.

        Args:
            records: List of (game_id, move_number, player_perspective, features, value, board_type)

        Returns:
            Number of records stored
        """
        import numpy as np

        if not records:
            return 0

        with self._get_conn() as conn:
            params = []
            for game_id, move_number, player_perspective, features, value, board_type in records:
                features_bytes = gzip.compress(features.astype(np.float32).tobytes())
                params.append((
                    game_id,
                    move_number,
                    player_perspective,
                    features_bytes,
                    value,
                    board_type,
                    len(features),
                ))

            conn.executemany(
                """
                INSERT OR REPLACE INTO game_nnue_features
                (game_id, move_number, player_perspective, features, value, board_type, feature_dim)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                params,
            )

        return len(records)

    def get_nnue_features(
        self,
        game_id: str,
        move_number: int | None = None,
        player_perspective: int | None = None,
    ) -> list[tuple[int, int, np.ndarray, float]]:
        """Retrieve pre-computed NNUE features for a game.

        Args:
            game_id: Game identifier
            move_number: Optional specific move number
            player_perspective: Optional specific player perspective

        Returns:
            List of (move_number, player_perspective, features, value) tuples
        """
        import numpy as np

        with self._get_conn() as conn:
            if move_number is not None and player_perspective is not None:
                cursor = conn.execute(
                    """
                    SELECT move_number, player_perspective, features, value, feature_dim
                    FROM game_nnue_features
                    WHERE game_id = ? AND move_number = ? AND player_perspective = ?
                    """,
                    (game_id, move_number, player_perspective),
                )
            elif move_number is not None:
                cursor = conn.execute(
                    """
                    SELECT move_number, player_perspective, features, value, feature_dim
                    FROM game_nnue_features
                    WHERE game_id = ? AND move_number = ?
                    ORDER BY player_perspective
                    """,
                    (game_id, move_number),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT move_number, player_perspective, features, value, feature_dim
                    FROM game_nnue_features
                    WHERE game_id = ?
                    ORDER BY move_number, player_perspective
                    """,
                    (game_id,),
                )

            results = []
            for row in cursor:
                move_num, player_persp, features_bytes, value, feature_dim = row
                # Decompress and reconstruct numpy array
                features = np.frombuffer(
                    gzip.decompress(features_bytes), dtype=np.float32
                ).copy()
                assert len(features) == feature_dim, (
                    f"Feature dimension mismatch: expected {feature_dim}, got {len(features)}"
                )
                results.append((move_num, player_persp, features, value))

            return results

    def get_nnue_features_for_training(
        self,
        board_type: str,
        num_players: int,
        limit: int | None = None,
        offset: int = 0,
    ) -> Iterator[tuple[str, int, int, np.ndarray, float]]:
        """Iterate over cached NNUE features for training.

        Yields features for all games matching board_type/num_players criteria.

        Args:
            board_type: Board type filter (e.g., "hex", "square8")
            num_players: Number of players filter
            limit: Optional max records to return
            offset: Number of records to skip

        Yields:
            (game_id, move_number, player_perspective, features, value) tuples
        """
        import numpy as np

        with self._get_conn() as conn:
            # Join with games table to filter by num_players
            query = """
                SELECT f.game_id, f.move_number, f.player_perspective, f.features, f.value, f.feature_dim
                FROM game_nnue_features f
                JOIN games g ON f.game_id = g.game_id
                WHERE f.board_type = ? AND g.num_players = ?
                ORDER BY f.game_id, f.move_number, f.player_perspective
            """
            params: list[Any] = [board_type, num_players]

            if limit is not None:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])

            cursor = conn.execute(query, params)

            for row in cursor:
                game_id, move_num, player_persp, features_bytes, value, _feature_dim = row
                features = np.frombuffer(
                    gzip.decompress(features_bytes), dtype=np.float32
                ).copy()
                yield game_id, move_num, player_persp, features, value

    def count_nnue_features(
        self,
        board_type: str | None = None,
        num_players: int | None = None,
    ) -> int:
        """Count cached NNUE feature records.

        Args:
            board_type: Optional board type filter
            num_players: Optional player count filter

        Returns:
            Number of cached feature records
        """
        with self._get_conn() as conn:
            if board_type is not None and num_players is not None:
                cursor = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM game_nnue_features f
                    JOIN games g ON f.game_id = g.game_id
                    WHERE f.board_type = ? AND g.num_players = ?
                    """,
                    (board_type, num_players),
                )
            elif board_type is not None:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM game_nnue_features WHERE board_type = ?",
                    (board_type,),
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM game_nnue_features")

            return cursor.fetchone()[0]

    def has_nnue_features(self, game_id: str) -> bool:
        """Check if a game has cached NNUE features.

        Args:
            game_id: Game identifier

        Returns:
            True if the game has cached features
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM game_nnue_features WHERE game_id = ? LIMIT 1",
                (game_id,),
            )
            return cursor.fetchone() is not None

    def delete_nnue_features(self, game_id: str) -> int:
        """Delete cached NNUE features for a game.

        Args:
            game_id: Game identifier

        Returns:
            Number of records deleted
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM game_nnue_features WHERE game_id = ?",
                (game_id,),
            )
            return cursor.rowcount

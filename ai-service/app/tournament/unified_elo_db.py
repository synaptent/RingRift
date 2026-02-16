"""Unified Elo database for all tournament types.

.. deprecated::
    This module is maintained for backward compatibility with existing tournament
    scripts. For NEW code, prefer using `app.training.elo_service.EloService` which
    provides additional features like:
    - Training feedback signals
    - Single-writer cluster coordination
    - Better integration with the training pipeline

    Migration example:
        # Old way (still works)
        from app.tournament import get_elo_database
        db = get_elo_database()
        db.record_match_and_update(...)

        # New way (preferred for new code)
        from app.training.elo_service import get_elo_service
        elo = get_elo_service()
        elo.record_match(...)

Provides a single, shared Elo rating system across all tournament scripts.
Uses composite primary key (participant_id, board_type, num_players) to track
ratings per game configuration.

Usage:
    from app.tournament import get_elo_database, EloDatabase

    # Get singleton database instance
    db = get_elo_database()

    # Record a match and update ratings
    db.record_match_and_update(
        participant_ids=["model_a", "model_b"],
        rankings=[0, 1],  # model_a won (rank 0 = 1st place)
        board_type="square8",
        num_players=2,
        tournament_id="my_tournament_123",
    )

    # Query leaderboard
    leaders = db.get_leaderboard(board_type="square8", num_players=2)
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Emit deprecation warning on import (December 2025)
warnings.warn(
    "unified_elo_db is deprecated since 2025-12. "
    "Use app.training.elo_service.EloService instead, which provides "
    "training feedback signals and better cluster coordination. "
    "This module will be removed in 2025-Q2.",
    DeprecationWarning,
    stacklevel=2,
)

from .elo import EloCalculator

logger = logging.getLogger(__name__)

# Bridge to canonical EloService for unified rating tracking
try:
    from app.training.elo_service import EloService, get_elo_service
    HAS_ELO_SERVICE = True
except ImportError:
    HAS_ELO_SERVICE = False
    get_elo_service = None
    EloService = None

# Import centralized timeout thresholds
try:
    from app.config.thresholds import SQLITE_BUSY_TIMEOUT_MS, SQLITE_TIMEOUT
except ImportError:
    SQLITE_BUSY_TIMEOUT_MS = 10000
    SQLITE_TIMEOUT = 30

# Database location - canonical Elo database for all trained models
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "unified_elo.db"

# Global singleton
_elo_db_instance: EloDatabase | None = None
_elo_db_lock = threading.RLock()


@dataclass
class UnifiedEloRating:
    """Elo rating for a participant in a specific game configuration."""

    participant_id: str
    board_type: str
    num_players: int
    rating: float = 1500.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    rating_deviation: float = 350.0  # Initial RD (high uncertainty)
    last_update: float | None = None

    # Glicko-style constants
    INITIAL_RD: float = 350.0
    MIN_RD: float = 50.0
    RD_DECAY_GAMES: int = 100

    @property
    def calculated_rd(self) -> float:
        """Calculate rating deviation based on games played."""
        import math

        if self.games_played == 0:
            return self.INITIAL_RD
        decay_factor = math.exp(-self.games_played / self.RD_DECAY_GAMES)
        return self.MIN_RD + (self.INITIAL_RD - self.MIN_RD) * decay_factor

    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played

    def to_dict(self) -> dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "rating": round(self.rating, 1),
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": round(self.win_rate, 3),
            "rating_deviation": round(self.calculated_rd, 1),
        }


@dataclass
class MatchRecord:
    """Record of a completed match."""

    match_id: str
    participant_ids: list[str]
    rankings: list[int]  # Position in final standings (0=1st, 1=2nd, etc.)
    winner_id: str | None
    board_type: str
    num_players: int
    game_length: int
    duration_sec: float
    timestamp: str
    tournament_id: str
    worker: str | None = None


class EloDatabase:
    """SQLite database for unified Elo tracking across all tournaments.

    Schema uses composite primary key (participant_id, board_type, num_players)
    to track separate ratings for each game configuration.
    """

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()
        logger.info(f"EloDatabase initialized at {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection with optimized settings.

        Uses WAL mode for better concurrent read/write performance and
        configures appropriate timeouts for multi-process access.
        """
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), timeout=float(SQLITE_TIMEOUT))
            self._local.conn.row_factory = sqlite3.Row

            # Enable WAL mode for better concurrency (multiple readers, one writer)
            # This is safe to call even if already enabled - SQLite handles idempotently
            self._local.conn.execute('PRAGMA journal_mode=WAL')

            # Increase busy timeout to handle contention from multiple processes
            self._local.conn.execute(f'PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}')

            # Synchronous NORMAL is safe with WAL mode and faster than FULL
            self._local.conn.execute('PRAGMA synchronous=NORMAL')

        return self._local.conn

    @property
    def id_column(self) -> str:
        """Get the correct ID column name for elo_ratings table.

        Returns 'model_id' for elo_leaderboard.db, 'participant_id' for unified_elo.db.
        """
        return "model_id" if getattr(self, "_uses_model_id_schema", False) else "participant_id"

    @property
    def match_columns(self) -> tuple:
        """Get the correct column names for match_history table.

        Returns ('model_a', 'model_b') for elo_leaderboard.db,
        ('participant_a', 'participant_b') for unified_elo.db.
        """
        if getattr(self, "_uses_model_id_schema", False):
            return ("model_a", "model_b")
        return ("participant_a", "participant_b")

    def _init_db(self):
        """Initialize database schema.

        This schema is backwards-compatible with existing unified_elo.db while
        supporting new features. The schema will be migrated if needed.

        Also supports elo_leaderboard.db which uses model_id instead of participant_id.
        """
        conn = self._get_connection()

        # Check if we need to migrate an existing database
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='participants'")
        participants_exists = cursor.fetchone() is not None

        # Check if elo_ratings exists with different schema (model_id vs participant_id)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='elo_ratings'")
        elo_ratings_exists = cursor.fetchone() is not None
        self._uses_model_id_schema = False

        if elo_ratings_exists:
            cursor = conn.execute("PRAGMA table_info(elo_ratings)")
            elo_columns = {row[1] for row in cursor.fetchall()}
            # elo_leaderboard.db uses model_id, unified_elo.db uses participant_id
            self._uses_model_id_schema = "model_id" in elo_columns and "participant_id" not in elo_columns
            if self._uses_model_id_schema:
                # Don't modify elo_leaderboard.db schema - it's a different format
                self._old_participant_schema = False
                self._old_match_schema = False
                return  # Skip schema creation for model_id databases

        if participants_exists:
            # Check if it's old schema (participant_id as PK) vs new (id as PK)
            cursor = conn.execute("PRAGMA table_info(participants)")
            columns = {row[1] for row in cursor.fetchall()}
            self._old_participant_schema = "participant_id" in columns and "id" not in columns

            # Check match_history schema
            cursor = conn.execute("PRAGMA table_info(match_history)")
            match_columns = {row[1] for row in cursor.fetchall()}
            self._old_match_schema = "participant_a" in match_columns
        else:
            self._old_participant_schema = False
            self._old_match_schema = False

        # Create tables with backwards-compatible schema
        conn.executescript("""
            -- Participants table: all known participants (models, AI types, etc.)
            -- Uses participant_id as PK for backwards compatibility
            CREATE TABLE IF NOT EXISTS participants (
                participant_id TEXT PRIMARY KEY,
                participant_type TEXT NOT NULL DEFAULT 'model',
                ai_type TEXT,
                difficulty INTEGER,
                use_neural_net INTEGER,
                model_path TEXT,
                model_version TEXT,
                metadata TEXT,
                created_at REAL,
                last_seen REAL
            );

            -- Elo ratings: per-configuration ratings
            CREATE TABLE IF NOT EXISTS elo_ratings (
                participant_id TEXT NOT NULL,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                rating REAL DEFAULT 1500.0,
                games_played INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                draws INTEGER DEFAULT 0,
                rating_deviation REAL DEFAULT 350.0,
                last_update REAL,
                -- Jan 2026: Harness tracking for composite Elo
                harness_type TEXT,          -- e.g., "gumbel_mcts", "minimax", "policy_only"
                simulation_count INTEGER,   -- e.g., 64, 200, 800, 1600
                PRIMARY KEY (participant_id, board_type, num_players)
            );

            -- Match history: supports both old and new formats
            -- Old format: participant_a, participant_b columns
            -- New format: participant_ids JSON array, rankings JSON array
            CREATE TABLE IF NOT EXISTS match_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                participant_a TEXT,
                participant_b TEXT,
                participant_ids TEXT,   -- JSON array (new format)
                rankings TEXT,          -- JSON array of final positions (new format)
                winner TEXT,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                game_length INTEGER,
                duration_sec REAL,
                timestamp REAL NOT NULL,
                tournament_id TEXT,
                game_id TEXT,
                worker TEXT,
                metadata TEXT,
                elo_before TEXT,        -- JSON dict of participant_id -> elo before match
                elo_after TEXT,         -- JSON dict of participant_id -> elo after match
                -- Dec 2025: Harness abstraction columns for Phase 1
                harness_type TEXT,           -- e.g. "gumbel_mcts", "minimax", "maxn"
                architecture_version TEXT,   -- e.g. "v4", "v5", "v6"
                evaluation_metadata TEXT     -- JSON blob with full EvaluationMetadata
            );

            -- Rating history: track Elo changes over time
            CREATE TABLE IF NOT EXISTS rating_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                participant_id TEXT NOT NULL,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                rating REAL NOT NULL,
                rating_change REAL,
                games_played INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                match_id TEXT,
                tournament_id TEXT
            );

        """)
        conn.commit()

        # Add missing columns if upgrading schema - MUST run before creating indexes
        # that reference new columns like game_id
        self._upgrade_schema_if_needed(conn)

        # Create indices after schema upgrade to ensure all columns exist
        conn.executescript("""
            -- Indices for common queries
            CREATE INDEX IF NOT EXISTS idx_elo_config
            ON elo_ratings(board_type, num_players, rating DESC);

            CREATE INDEX IF NOT EXISTS idx_elo_participant
            ON elo_ratings(participant_id);

            CREATE INDEX IF NOT EXISTS idx_match_timestamp
            ON match_history(timestamp DESC);

            CREATE INDEX IF NOT EXISTS idx_match_tournament
            ON match_history(tournament_id);

            CREATE INDEX IF NOT EXISTS idx_rating_history_participant
            ON rating_history(participant_id, board_type, num_players, timestamp DESC);

            -- Unique index on game_id for deduplication (2025-12-16)
            CREATE UNIQUE INDEX IF NOT EXISTS idx_match_game_id
            ON match_history(game_id) WHERE game_id IS NOT NULL;

            -- Index for composite key deduplication fallback (only if old schema columns exist)
        """)
        conn.commit()

        # Conditionally create index on old schema columns if they exist
        cursor = conn.execute("PRAGMA table_info(match_history)")
        match_cols = {row[1] for row in cursor.fetchall()}
        if "participant_a" in match_cols and "participant_b" in match_cols:
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_match_dedupe_key
                ON match_history(participant_a, participant_b, board_type, num_players, timestamp)
            """)
            conn.commit()

    def _upgrade_schema_if_needed(self, conn: sqlite3.Connection):
        """Add missing columns to existing tables for schema upgrades."""
        # Get current columns for each table
        def get_columns(table: str) -> set:
            cursor = conn.execute(f"PRAGMA table_info({table})")
            return {row[1] for row in cursor.fetchall()}

        # Upgrade participants table
        participant_cols = get_columns("participants")
        if "ai_type" not in participant_cols:
            conn.execute("ALTER TABLE participants ADD COLUMN ai_type TEXT")
        if "difficulty" not in participant_cols:
            conn.execute("ALTER TABLE participants ADD COLUMN difficulty INTEGER")
        if "use_neural_net" not in participant_cols:
            conn.execute("ALTER TABLE participants ADD COLUMN use_neural_net INTEGER")
        if "model_version" not in participant_cols:
            conn.execute("ALTER TABLE participants ADD COLUMN model_version TEXT")

        # Upgrade match_history table
        match_cols = get_columns("match_history")
        if "participant_ids" not in match_cols:
            conn.execute("ALTER TABLE match_history ADD COLUMN participant_ids TEXT")
        if "rankings" not in match_cols:
            conn.execute("ALTER TABLE match_history ADD COLUMN rankings TEXT")
        if "worker" not in match_cols:
            conn.execute("ALTER TABLE match_history ADD COLUMN worker TEXT")
        if "game_id" not in match_cols:
            conn.execute("ALTER TABLE match_history ADD COLUMN game_id TEXT")
        # Dec 2025: Harness abstraction columns for Phase 1
        if "harness_type" not in match_cols:
            conn.execute("ALTER TABLE match_history ADD COLUMN harness_type TEXT")
        if "architecture_version" not in match_cols:
            conn.execute("ALTER TABLE match_history ADD COLUMN architecture_version TEXT")
        if "evaluation_metadata" not in match_cols:
            conn.execute("ALTER TABLE match_history ADD COLUMN evaluation_metadata TEXT")

        # Upgrade rating_history table
        rating_cols = get_columns("rating_history")
        if "rating_change" not in rating_cols:
            conn.execute("ALTER TABLE rating_history ADD COLUMN rating_change REAL")
        if "match_id" not in rating_cols:
            conn.execute("ALTER TABLE rating_history ADD COLUMN match_id TEXT")

        # Upgrade elo_ratings table - add archive tracking columns
        elo_cols = get_columns("elo_ratings")
        if "archived_at" not in elo_cols:
            conn.execute("ALTER TABLE elo_ratings ADD COLUMN archived_at REAL")
        if "archive_reason" not in elo_cols:
            conn.execute("ALTER TABLE elo_ratings ADD COLUMN archive_reason TEXT")

        # Jan 2026: Add harness tracking columns for composite Elo
        if "harness_type" not in elo_cols:
            conn.execute("ALTER TABLE elo_ratings ADD COLUMN harness_type TEXT")
            conn.execute("ALTER TABLE elo_ratings ADD COLUMN simulation_count INTEGER")
            # Backfill from composite participant IDs (e.g., "model:gumbel_mcts:b800")
            conn.execute("""
                UPDATE elo_ratings
                SET harness_type = CASE
                    WHEN participant_id LIKE '%:gumbel_mcts:%' THEN 'gumbel_mcts'
                    WHEN participant_id LIKE '%:minimax:%' THEN 'minimax'
                    WHEN participant_id LIKE '%:maxn:%' THEN 'maxn'
                    WHEN participant_id LIKE '%:policy_only:%' THEN 'policy_only'
                    ELSE NULL
                END,
                simulation_count = CASE
                    WHEN participant_id LIKE '%:b64' THEN 64
                    WHEN participant_id LIKE '%:b150' THEN 150
                    WHEN participant_id LIKE '%:b200' THEN 200
                    WHEN participant_id LIKE '%:b800' THEN 800
                    WHEN participant_id LIKE '%:b1600' THEN 1600
                    ELSE NULL
                END
                WHERE participant_id LIKE '%:%'
            """)

        conn.commit()

        # Create gauntlet tracking tables if they don't exist
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS gauntlet_runs (
                run_id TEXT PRIMARY KEY,
                config_key TEXT NOT NULL,
                started_at REAL NOT NULL,
                completed_at REAL,
                models_evaluated INTEGER DEFAULT 0,
                total_games INTEGER DEFAULT 0,
                status TEXT DEFAULT 'running'
            );

            CREATE TABLE IF NOT EXISTS gauntlet_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                baseline_id TEXT NOT NULL,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                draws INTEGER DEFAULT 0,
                elo_before REAL,
                elo_after REAL,
                FOREIGN KEY (run_id) REFERENCES gauntlet_runs(run_id)
            );

            CREATE INDEX IF NOT EXISTS idx_gauntlet_results_model
                ON gauntlet_results(model_id, run_id);

            CREATE INDEX IF NOT EXISTS idx_gauntlet_runs_config
                ON gauntlet_runs(config_key, started_at DESC);

            -- Model identity tracking (Dec 2025)
            -- Tracks model files by SHA256 hash for deduplication and alias resolution
            CREATE TABLE IF NOT EXISTS model_identities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_path TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                file_size INTEGER,
                first_seen_at REAL DEFAULT (strftime('%s', 'now')),
                last_verified_at REAL DEFAULT (strftime('%s', 'now')),
                UNIQUE(model_path, content_sha256)
            );

            -- Participant aliases for same model content
            -- Links different participant IDs that refer to the same model file
            CREATE TABLE IF NOT EXISTS participant_aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                primary_participant_id TEXT NOT NULL,
                alias_participant_id TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                UNIQUE(primary_participant_id, alias_participant_id)
            );

            CREATE INDEX IF NOT EXISTS idx_model_identities_hash
                ON model_identities(content_sha256);

            CREATE INDEX IF NOT EXISTS idx_model_identities_path
                ON model_identities(model_path);

            CREATE INDEX IF NOT EXISTS idx_participant_aliases_primary
                ON participant_aliases(primary_participant_id);

            CREATE INDEX IF NOT EXISTS idx_participant_aliases_alias
                ON participant_aliases(alias_participant_id);

            CREATE INDEX IF NOT EXISTS idx_participant_aliases_hash
                ON participant_aliases(content_sha256);

            -- Sprint 15 (Jan 3, 2026): Model evaluation status tracking
            -- Tracks evaluation status for all models (local, OWC, S3) to enable
            -- comprehensive backlog evaluation automation
            CREATE TABLE IF NOT EXISTS model_evaluation_status (
                id INTEGER PRIMARY KEY,
                model_sha256 TEXT NOT NULL,           -- Content hash for deduplication
                model_path TEXT NOT NULL,             -- Canonical path (relative or absolute)
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                harness_type TEXT DEFAULT 'default',  -- Multi-harness support
                status TEXT DEFAULT 'pending',        -- pending/queued/running/evaluated/failed/stale
                elo_rating REAL,
                games_evaluated INTEGER DEFAULT 0,
                first_seen_at REAL NOT NULL,
                last_evaluated_at REAL,
                evaluation_error TEXT,                -- Last error message if failed
                source TEXT DEFAULT 'local',          -- local/owc/s3/cluster
                priority INTEGER DEFAULT 100,         -- Lower = higher priority (0-200)
                UNIQUE(model_sha256, board_type, num_players, harness_type)
            );

            CREATE INDEX IF NOT EXISTS idx_model_eval_status_pending
                ON model_evaluation_status(status, priority) WHERE status = 'pending';

            CREATE INDEX IF NOT EXISTS idx_model_eval_status_config
                ON model_evaluation_status(board_type, num_players, status);

            CREATE INDEX IF NOT EXISTS idx_model_eval_status_source
                ON model_evaluation_status(source, status);

            CREATE INDEX IF NOT EXISTS idx_model_eval_status_stale
                ON model_evaluation_status(last_evaluated_at) WHERE status = 'evaluated';
        """)
        conn.commit()

    def close(self):
        """Close thread-local connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # =========================================================================
    # Participant Management
    # =========================================================================

    def register_participant(
        self,
        participant_id: str,
        name: str | None = None,
        participant_type: str = "model",
        ai_type: str | None = None,
        difficulty: int | None = None,
        use_neural_net: bool = False,
        model_id: str | None = None,
        model_path: str | None = None,
        model_version: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Register a participant (model, baseline, or AI type)."""
        conn = self._get_connection()
        now = time.time()
        conn.execute("""
            INSERT INTO participants
            (participant_id, participant_type, ai_type, difficulty, use_neural_net,
             model_path, model_version, metadata, created_at, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(participant_id) DO UPDATE SET
                participant_type = COALESCE(excluded.participant_type, participant_type),
                ai_type = COALESCE(excluded.ai_type, ai_type),
                difficulty = COALESCE(excluded.difficulty, difficulty),
                use_neural_net = COALESCE(excluded.use_neural_net, use_neural_net),
                model_path = COALESCE(excluded.model_path, model_path),
                model_version = COALESCE(excluded.model_version, model_version),
                metadata = COALESCE(excluded.metadata, metadata),
                last_seen = excluded.last_seen
        """, (
            participant_id,
            participant_type,
            ai_type,
            difficulty,
            int(use_neural_net) if use_neural_net else None,
            model_path,
            model_version,
            json.dumps(metadata) if metadata else None,
            now,
            now,
        ))
        conn.commit()

    def get_participant(self, participant_id: str) -> dict[str, Any] | None:
        """Get participant info."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM participants WHERE participant_id = ?",
            (participant_id,)
        ).fetchone()
        if row:
            result = dict(row)
            if result.get("metadata"):
                result["metadata"] = json.loads(result["metadata"])
            return result
        return None

    def ensure_participant(self, participant_id: str, **kwargs) -> None:
        """Ensure participant exists, creating with defaults if not."""
        if not self.get_participant(participant_id):
            self.register_participant(participant_id, **kwargs)

    # =========================================================================
    # Rating Management
    # =========================================================================

    def get_rating(
        self,
        participant_id: str,
        board_type: str,
        num_players: int,
    ) -> UnifiedEloRating:
        """Get rating for a participant in a specific configuration."""
        conn = self._get_connection()
        row = conn.execute("""
            SELECT rating, games_played, wins, losses, draws, rating_deviation, last_update
            FROM elo_ratings
            WHERE participant_id = ? AND board_type = ? AND num_players = ?
        """, (participant_id, board_type, num_players)).fetchone()

        if row:
            return UnifiedEloRating(
                participant_id=participant_id,
                board_type=board_type,
                num_players=num_players,
                rating=row["rating"],
                games_played=row["games_played"],
                wins=row["wins"],
                losses=row["losses"],
                draws=row["draws"],
                rating_deviation=row["rating_deviation"] or 350.0,
                last_update=row["last_update"],
            )
        return UnifiedEloRating(participant_id, board_type, num_players)

    def get_ratings_batch(
        self,
        participant_ids: list[str],
        board_type: str,
        num_players: int,
    ) -> dict[str, UnifiedEloRating]:
        """Get ratings for multiple participants at once using batch query.

        Optimized to use a single SQL query with IN clause instead of N queries.
        For participants without ratings, returns default UnifiedEloRating.
        """
        if not participant_ids:
            return {}

        conn = self._get_connection()
        result: dict[str, UnifiedEloRating] = {}

        # Process in chunks to avoid SQLite SQLITE_MAX_VARIABLE_NUMBER limit
        chunk_size = 500
        for i in range(0, len(participant_ids), chunk_size):
            chunk = participant_ids[i:i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(f"""
                SELECT participant_id, rating, games_played, wins, losses, draws,
                       rating_deviation, last_update
                FROM elo_ratings
                WHERE participant_id IN ({placeholders})
                  AND board_type = ? AND num_players = ?
            """, (*chunk, board_type, num_players)).fetchall()

            for row in rows:
                result[row["participant_id"]] = UnifiedEloRating(
                    participant_id=row["participant_id"],
                    board_type=board_type,
                    num_players=num_players,
                    rating=row["rating"],
                    games_played=row["games_played"],
                    wins=row["wins"],
                    losses=row["losses"],
                    draws=row["draws"],
                    rating_deviation=row["rating_deviation"],
                    last_update=row["last_update"],
                )

        # Fill in defaults for missing participants
        for pid in participant_ids:
            if pid not in result:
                result[pid] = UnifiedEloRating(pid, board_type, num_players)

        return result

    # Pinned baselines that should not have their ELO updated (anchor points)
    # These use prefix matching - any participant_id starting with these is pinned
    # Random at 400 Elo is the anchor that prevents rating inflation across the system
    PINNED_BASELINES = {
        "baseline_random": 400.0,  # Random player pinned at 400 ELO as anchor
        "none:random": 400.0,      # Composite format prefix (matches none:random:d1, etc.)
        "none:random:d1": 400.0,   # Explicit composite format for D1 random
        "random": 400.0,           # Simple random prefix
        "tier1_random": 400.0,     # Tier 1 difficulty = random, pinned at 400
        "d1": 400.0,               # D1 tier = random baseline, pinned at 400
    }

    def _is_pinned_baseline(self, participant_id: str) -> float | None:
        """Check if participant is a pinned baseline and return pinned ELO if so.

        Random AI must ALWAYS be pinned at 400 ELO to serve as the anchor point
        for the entire rating system. This is critical for ELO calibration.
        """
        pid_lower = participant_id.lower()
        for prefix, pinned_elo in self.PINNED_BASELINES.items():
            if pid_lower.startswith(prefix.lower()):
                return pinned_elo
        # Also check for 'random' anywhere in the name for robustness
        if 'random' in pid_lower and 'heuristic' not in pid_lower:
            return 400.0
        return None

    def update_rating(self, rating: UnifiedEloRating) -> None:
        """Update a single rating."""
        # Check if this is a pinned baseline - if so, force the pinned rating
        pinned_elo = self._is_pinned_baseline(rating.participant_id)
        if pinned_elo is not None:
            rating.rating = pinned_elo

        conn = self._get_connection()
        conn.execute("""
            INSERT INTO elo_ratings
            (participant_id, board_type, num_players, rating, games_played,
             wins, losses, draws, rating_deviation, peak_rating, last_update)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(participant_id, board_type, num_players) DO UPDATE SET
                rating = excluded.rating,
                games_played = excluded.games_played,
                wins = excluded.wins,
                losses = excluded.losses,
                draws = excluded.draws,
                rating_deviation = excluded.rating_deviation,
                peak_rating = MAX(COALESCE(elo_ratings.peak_rating, 0), excluded.peak_rating),
                last_update = excluded.last_update
        """, (
            rating.participant_id,
            rating.board_type,
            rating.num_players,
            rating.rating,
            rating.games_played,
            rating.wins,
            rating.losses,
            rating.draws,
            rating.calculated_rd,
            rating.rating,
            time.time(),
        ))
        conn.commit()

    def update_ratings_batch(self, ratings: list[UnifiedEloRating]) -> None:
        """Update multiple ratings in a single transaction."""
        conn = self._get_connection()
        now = time.time()
        for r in ratings:
            # Check if this is a pinned baseline - if so, force the pinned rating
            pinned_elo = self._is_pinned_baseline(r.participant_id)
            if pinned_elo is not None:
                r.rating = pinned_elo
            conn.execute("""
                INSERT INTO elo_ratings
                (participant_id, board_type, num_players, rating, games_played,
                 wins, losses, draws, rating_deviation, peak_rating, last_update)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(participant_id, board_type, num_players) DO UPDATE SET
                    rating = excluded.rating,
                    games_played = excluded.games_played,
                    wins = excluded.wins,
                    losses = excluded.losses,
                    draws = excluded.draws,
                    rating_deviation = excluded.rating_deviation,
                    peak_rating = MAX(COALESCE(elo_ratings.peak_rating, 0), excluded.peak_rating),
                    last_update = excluded.last_update
            """, (
                r.participant_id, r.board_type, r.num_players, r.rating,
                r.games_played, r.wins, r.losses, r.draws, r.calculated_rd, r.rating, now,
            ))
        conn.commit()

    def reset_pinned_baselines(self) -> int:
        """Reset all pinned baselines to their anchor ELO values.

        This should be called periodically to ensure baseline ratings
        haven't drifted due to bugs or manual updates.

        Returns:
            Number of baselines reset
        """
        conn = self._get_connection()
        reset_count = 0

        for prefix, pinned_elo in self.PINNED_BASELINES.items():
            # Find all participants matching this prefix
            cursor = conn.execute("""
                SELECT participant_id, board_type, num_players, rating
                FROM elo_ratings
                WHERE participant_id LIKE ?
            """, (f"{prefix}%",))

            for row in cursor.fetchall():
                pid, board_type, num_players, current_rating = row
                if current_rating != pinned_elo:
                    conn.execute("""
                        UPDATE elo_ratings
                        SET rating = ?, last_update = ?
                        WHERE participant_id = ? AND board_type = ? AND num_players = ?
                    """, (pinned_elo, time.time(), pid, board_type, num_players))
                    reset_count += 1
                    logger.info(
                        f"Reset {pid} from {current_rating:.0f} to {pinned_elo:.0f}"
                    )

        conn.commit()
        return reset_count

    # =========================================================================
    # Match Recording and Elo Updates
    # =========================================================================

    def record_match(
        self,
        participant_ids: list[str],
        rankings: list[int],
        board_type: str,
        num_players: int,
        tournament_id: str,
        game_length: int = 0,
        duration_sec: float = 0.0,
        worker: str | None = None,
        metadata: dict | None = None,
        game_id: str | None = None,
    ) -> int:
        """Record a match without updating Elo ratings.

        Args:
            participant_ids: List of participant IDs in the match
            rankings: Final positions (0=1st, 1=2nd, etc.) for each participant
            board_type: Board type (square8, square19, hexagonal)
            num_players: Number of players (2, 3, or 4)
            tournament_id: ID of the tournament
            game_length: Number of moves in the game
            duration_sec: Duration of the game in seconds
            worker: Worker that ran the game (optional)
            metadata: Additional match metadata (optional)
            game_id: Optional game UUID for tracking

        Returns:
            Generated match ID (integer)
        """
        timestamp = time.time()

        # Always generate game_id if not provided (2025-12-16 fix for deduplication)
        if game_id is None:
            game_id = str(uuid.uuid4())

        # Determine winner (participant with ranking 0)
        winner_id = None
        for pid, rank in zip(participant_ids, rankings, strict=False):
            if rank == 0:
                winner_id = pid
                break

        # For backwards compatibility, also populate model_a/model_b or participant_a/participant_b
        col_a, col_b = self.match_columns
        model_a = participant_ids[0] if len(participant_ids) > 0 else None
        model_b = participant_ids[1] if len(participant_ids) > 1 else None

        conn = self._get_connection()

        # Check if this is the simple schema (elo_leaderboard.db) without participant_ids column
        if self._uses_model_id_schema:
            cursor = conn.execute(f"""
                INSERT INTO match_history
                ({col_a}, {col_b}, winner, board_type, num_players,
                 game_length, duration_sec, timestamp, tournament_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_a,
                model_b,
                winner_id,
                board_type,
                num_players,
                game_length,
                duration_sec,
                timestamp,
                tournament_id,
            ))
        else:
            cursor = conn.execute(f"""
                INSERT INTO match_history
                ({col_a}, {col_b}, participant_ids, rankings, winner,
                 board_type, num_players, game_length, duration_sec, timestamp,
                 tournament_id, game_id, worker, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_a,
                model_b,
                json.dumps(participant_ids),
                json.dumps(rankings),
                winner_id,
                board_type,
                num_players,
                game_length,
                duration_sec,
                timestamp,
                tournament_id,
                game_id,
                worker,
                json.dumps(metadata) if metadata else None,
            ))
        match_id = cursor.lastrowid
        conn.commit()
        return match_id

    def record_match_and_update(
        self,
        participant_ids: list[str],
        rankings: list[int],
        board_type: str,
        num_players: int,
        tournament_id: str,
        game_length: int = 0,
        duration_sec: float = 0.0,
        worker: str | None = None,
        metadata: dict | None = None,
        k_factor: float = 32.0,
        game_id: str | None = None,
    ) -> tuple[int, dict[str, float]]:
        """Record a match and update Elo ratings atomically.

        Uses IMMEDIATE transaction to prevent race conditions between
        reading current ratings and writing new ones. (2025-12-16 fix)

        Args:
            participant_ids: List of participant IDs in the match
            rankings: Final positions (0=1st, 1=2nd, etc.) for each participant
            board_type: Board type (square8, square19, hexagonal)
            num_players: Number of players (2, 3, or 4)
            tournament_id: ID of the tournament
            game_length: Number of moves in the game
            duration_sec: Duration of the game in seconds
            worker: Worker that ran the game (optional)
            metadata: Additional match metadata (optional)
            k_factor: K-factor for Elo calculation
            game_id: Optional game UUID for tracking

        Returns:
            Tuple of (match_id, dict of participant_id -> new_rating)
        """
        # Generate game_id for deduplication if not provided
        if game_id is None:
            game_id = str(uuid.uuid4())

        # Ensure all participants are registered (outside main transaction)
        for pid in participant_ids:
            self.ensure_participant(pid)

        conn = self._get_connection()
        now = time.time()

        try:
            # Use IMMEDIATE transaction to acquire write lock early
            # This prevents TOCTOU race conditions
            conn.execute("BEGIN IMMEDIATE")

            # Get current ratings INSIDE the transaction
            ratings = {}
            for pid in participant_ids:
                cursor = conn.execute("""
                    SELECT participant_id, rating, games_played, wins, losses, draws, rating_deviation
                    FROM elo_ratings
                    WHERE participant_id = ? AND board_type = ? AND num_players = ?
                """, (pid, board_type, num_players))
                row = cursor.fetchone()
                if row:
                    ratings[pid] = UnifiedEloRating(
                        participant_id=row[0],
                        board_type=board_type,
                        num_players=num_players,
                        rating=row[1],
                        games_played=row[2],
                        wins=row[3],
                        losses=row[4],
                        draws=row[5],
                        rating_deviation=row[6] if row[6] else 350.0,
                    )
                else:
                    # Create default rating
                    ratings[pid] = UnifiedEloRating(
                        participant_id=pid,
                        board_type=board_type,
                        num_players=num_players,
                    )

            # Calculate new ratings using EloCalculator
            calculator = EloCalculator(k_factor=k_factor)

            # Load current ratings into calculator
            for pid, rating in ratings.items():
                elo_rating = calculator.get_rating(pid)
                elo_rating.rating = rating.rating
                elo_rating.games_played = rating.games_played
                elo_rating.wins = rating.wins
                elo_rating.losses = rating.losses
                elo_rating.draws = rating.draws

            # Sort participants by ranking for the multiplayer update
            sorted_participants = sorted(
                zip(participant_ids, rankings, strict=False),
                key=lambda x: x[1]
            )
            ordered_ids = [pid for pid, _ in sorted_participants]

            # Update ratings in calculator
            if len(participant_ids) == 2:
                # Two-player match
                result = 1.0 if rankings[0] < rankings[1] else (0.5 if rankings[0] == rankings[1] else 0.0)
                calculator.update_ratings(participant_ids[0], participant_ids[1], result)
            else:
                # Multiplayer match
                calculator.update_multiplayer_ratings(ordered_ids)

            # Determine winner (participant with ranking 0)
            winner_id = None
            for pid, rank in zip(participant_ids, rankings, strict=False):
                if rank == 0:
                    winner_id = pid
                    break

            # Insert match record
            col_a, col_b = self.match_columns
            model_a = participant_ids[0] if len(participant_ids) > 0 else None
            model_b = participant_ids[1] if len(participant_ids) > 1 else None

            cursor = conn.execute(f"""
                INSERT INTO match_history
                ({col_a}, {col_b}, participant_ids, rankings, winner,
                 board_type, num_players, game_length, duration_sec, timestamp,
                 tournament_id, game_id, worker, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_a,
                model_b,
                json.dumps(participant_ids),
                json.dumps(rankings),
                winner_id,
                board_type,
                num_players,
                game_length,
                duration_sec,
                now,
                tournament_id,
                game_id,
                worker,
                json.dumps(metadata) if metadata else None,
            ))
            match_id = cursor.lastrowid

            # Update ratings and record history
            new_ratings = {}
            for pid in participant_ids:
                old_rating = ratings[pid].rating
                elo_rating = calculator.get_rating(pid)
                new_rating = elo_rating.rating

                # Check if this is a pinned baseline - override rating if so
                pinned_elo = self._is_pinned_baseline(pid)
                if pinned_elo is not None:
                    new_rating = pinned_elo

                new_ratings[pid] = new_rating

                # Update or insert elo_ratings
                conn.execute("""
                    INSERT INTO elo_ratings
                    (participant_id, board_type, num_players, rating, games_played,
                     wins, losses, draws, rating_deviation, last_update)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(participant_id, board_type, num_players)
                    DO UPDATE SET
                        rating = excluded.rating,
                        games_played = excluded.games_played,
                        wins = excluded.wins,
                        losses = excluded.losses,
                        draws = excluded.draws,
                        last_update = excluded.last_update
                """, (
                    pid, board_type, num_players, new_rating,
                    elo_rating.games_played, elo_rating.wins, elo_rating.losses,
                    elo_rating.draws, ratings[pid].calculated_rd, now,
                ))

                # Record rating history
                conn.execute("""
                    INSERT INTO rating_history
                    (participant_id, board_type, num_players, rating, rating_change,
                     games_played, timestamp, match_id, tournament_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pid, board_type, num_players, new_rating, new_rating - old_rating,
                    elo_rating.games_played, now, match_id, tournament_id,
                ))

            conn.execute("COMMIT")

            # Emit ELO_UPDATED events for training pipeline integration
            # NOTE: We do NOT call EloService.record_match() here as that would cause
            # dual-write and double rating updates. Instead, we emit events directly.
            # The EloDatabase is the SSoT for tournament ratings; EloService is for
            # training pipeline integration and event emission only.
            # Feb 2026: Use sync version as fallback for gauntlet/tournament evaluation
            try:
                from app.distributed.data_events import emit_elo_updated, emit_elo_updated_sync
                import asyncio

                config_key = f"{board_type}_{num_players}p"
                for pid in participant_ids:
                    old_rating = ratings[pid].rating
                    try:
                        asyncio.get_running_loop()
                        asyncio.ensure_future(emit_elo_updated(
                            config=config_key,
                            model_id=pid,
                            new_elo=new_ratings[pid],
                            old_elo=old_rating,
                            games_played=1,
                            source="unified_elo_db",
                        ))
                    except RuntimeError:
                        # No event loop - use sync version
                        emit_elo_updated_sync(
                            config=config_key,
                            model_id=pid,
                            new_elo=new_ratings[pid],
                            old_elo=old_rating,
                            games_played=1,
                            source="unified_elo_db",
                        )
            except ImportError:
                pass  # Event system not available
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"Event emission failed (non-fatal): {e}")

            return match_id, new_ratings

        except Exception as e:
            conn.execute("ROLLBACK")
            logger.error(f"Failed to record match: {e}")
            raise

    def record_two_player_result(
        self,
        winner_id: str,
        loser_id: str,
        board_type: str,
        num_players: int,
        tournament_id: str,
        is_draw: bool = False,
        game_length: int = 0,
        duration_sec: float = 0.0,
        worker: str | None = None,
    ) -> tuple[int, dict[str, float]]:
        """Convenience method for recording a two-player match result.

        Args:
            winner_id: ID of the winner (or first player if draw)
            loser_id: ID of the loser (or second player if draw)
            board_type: Board type
            num_players: Number of players (usually 2)
            tournament_id: Tournament ID
            is_draw: Whether the game was a draw
            game_length: Number of moves
            duration_sec: Duration in seconds
            worker: Worker that ran the game

        Returns:
            Tuple of (match_id, dict of participant_id -> new_rating)
        """
        if is_draw:
            rankings = [0, 0]  # Tie
        else:
            rankings = [0, 1]  # Winner first, loser second

        return self.record_match_and_update(
            participant_ids=[winner_id, loser_id],
            rankings=rankings,
            board_type=board_type,
            num_players=num_players,
            tournament_id=tournament_id,
            game_length=game_length,
            duration_sec=duration_sec,
            worker=worker,
        )

    # =========================================================================
    # Queries
    # =========================================================================

    def get_leaderboard(
        self,
        board_type: str | None = None,
        num_players: int | None = None,
        min_games: int = 1,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get leaderboard, optionally filtered by configuration."""
        conn = self._get_connection()

        # Check if participants table exists and get the id column name
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='participants'")
        has_participants = cursor.fetchone() is not None
        id_col = self.id_column  # model_id or participant_id

        if has_participants and not self._uses_model_id_schema:
            # Full schema with participants table
            query = f"""
                SELECT e.*, p.participant_type, p.ai_type, p.difficulty,
                       p.use_neural_net, p.model_path, p.model_version
                FROM elo_ratings e
                LEFT JOIN participants p ON e.{id_col} = p.participant_id
                WHERE e.games_played >= ?
            """
        else:
            # Simple schema (elo_leaderboard.db) - no participants table
            query = f"""
                SELECT {id_col} as participant_id, board_type, num_players,
                       rating, games_played, wins, losses, draws, last_update
                FROM elo_ratings
                WHERE games_played >= ?
            """
        params: list[Any] = [min_games]

        if board_type:
            query += " AND board_type = ?"
            params.append(board_type)
        if num_players:
            query += " AND num_players = ?"
            params.append(num_players)

        query += " ORDER BY rating DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_match_history(
        self,
        participant_id: str | None = None,
        tournament_id: str | None = None,
        board_type: str | None = None,
        num_players: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get match history with optional filters."""
        conn = self._get_connection()

        query = "SELECT * FROM match_history WHERE 1=1"
        params: list[Any] = []

        if participant_id:
            query += " AND participant_ids LIKE ?"
            params.append(f'%"{participant_id}"%')
        if tournament_id:
            query += " AND tournament_id = ?"
            params.append(tournament_id)
        if board_type:
            query += " AND board_type = ?"
            params.append(board_type)
        if num_players:
            query += " AND num_players = ?"
            params.append(num_players)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        results = []
        for row in rows:
            r = dict(row)
            r["participant_ids"] = json.loads(r["participant_ids"])
            r["rankings"] = json.loads(r["rankings"]) if r["rankings"] else []
            if r.get("metadata"):
                r["metadata"] = json.loads(r["metadata"])
            results.append(r)
        return results

    def get_rating_history(
        self,
        participant_id: str,
        board_type: str,
        num_players: int,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get rating history for a participant in a specific config."""
        conn = self._get_connection()
        rows = conn.execute("""
            SELECT * FROM rating_history
            WHERE participant_id = ? AND board_type = ? AND num_players = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (participant_id, board_type, num_players, limit)).fetchall()
        return [dict(row) for row in rows]

    def get_head_to_head(
        self,
        participant_a: str,
        participant_b: str,
        board_type: str | None = None,
        num_players: int | None = None,
    ) -> dict[str, Any]:
        """Get head-to-head stats between two participants."""
        conn = self._get_connection()

        # Build query to find matches with both participants
        query = """
            SELECT * FROM match_history
            WHERE participant_ids LIKE ? AND participant_ids LIKE ?
        """
        params: list[Any] = [f'%"{participant_a}"%', f'%"{participant_b}"%']

        if board_type:
            query += " AND board_type = ?"
            params.append(board_type)
        if num_players:
            query += " AND num_players = ?"
            params.append(num_players)

        rows = conn.execute(query, params).fetchall()

        a_wins = 0
        b_wins = 0
        draws = 0

        for row in rows:
            winner = row["winner"]
            if winner == participant_a:
                a_wins += 1
            elif winner == participant_b:
                b_wins += 1
            else:
                draws += 1

        total = a_wins + b_wins + draws
        return {
            "participant_a": participant_a,
            "participant_b": participant_b,
            "total_games": total,
            "a_wins": a_wins,
            "b_wins": b_wins,
            "draws": draws,
            "a_win_rate": a_wins / total if total > 0 else 0.0,
            "b_win_rate": b_wins / total if total > 0 else 0.0,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get overall database statistics."""
        conn = self._get_connection()

        participant_count = conn.execute(
            "SELECT COUNT(*) FROM participants"
        ).fetchone()[0]

        rating_count = conn.execute(
            "SELECT COUNT(*) FROM elo_ratings WHERE games_played > 0"
        ).fetchone()[0]

        match_count = conn.execute(
            "SELECT COUNT(*) FROM match_history"
        ).fetchone()[0]

        configs = conn.execute("""
            SELECT board_type, num_players, COUNT(*) as count
            FROM elo_ratings
            WHERE games_played > 0
            GROUP BY board_type, num_players
        """).fetchall()

        return {
            "total_participants": participant_count,
            "rated_participants": rating_count,
            "total_matches": match_count,
            "configurations": [dict(c) for c in configs],
        }

    def check_win_loss_invariant(
        self,
        board_type: str | None = None,
        raise_on_violation: bool = False,
    ) -> dict[str, Any]:
        """Check win/loss conservation invariant for 2-player games.

        In 2-player games, every win corresponds to exactly one loss.
        Therefore: SUM(wins) == SUM(losses) for all active ratings.

        Args:
            board_type: Optional filter by board type
            raise_on_violation: If True, raises ValueError on imbalance

        Returns:
            Dict with total_wins, total_losses, imbalance, and is_valid fields
        """
        conn = self._get_connection()

        query = """
            SELECT SUM(wins) as total_wins, SUM(losses) as total_losses
            FROM elo_ratings
            WHERE num_players = 2 AND archived_at IS NULL
        """
        params: list[Any] = []

        if board_type:
            query = """
                SELECT SUM(wins) as total_wins, SUM(losses) as total_losses
                FROM elo_ratings
                WHERE num_players = 2 AND archived_at IS NULL AND board_type = ?
            """
            params.append(board_type)

        row = conn.execute(query, params).fetchone()
        total_wins = row["total_wins"] or 0
        total_losses = row["total_losses"] or 0
        imbalance = abs(total_wins - total_losses)
        is_valid = imbalance == 0

        result = {
            "total_wins": total_wins,
            "total_losses": total_losses,
            "imbalance": imbalance,
            "is_valid": is_valid,
            "board_type": board_type or "all",
        }

        if not is_valid:
            logger.warning(
                f"Win/loss invariant VIOLATED: {total_wins} wins vs {total_losses} losses "
                f"(imbalance: {imbalance}) for board_type={board_type or 'all'}"
            )
            if raise_on_violation:
                raise ValueError(
                    f"Win/loss invariant violated: {imbalance} imbalance"
                )
        else:
            logger.debug(
                f"Win/loss invariant OK: {total_wins} wins == {total_losses} losses"
            )

        return result

    def verify_database_integrity(self) -> dict[str, Any]:
        """Run comprehensive database integrity checks.

        Returns dict with results of all checks:
        - win_loss_invariant: Win/loss conservation for 2-player games
        - duplicate_game_ids: Count of duplicate game_ids
        - null_game_ids: Count of matches without game_id
        - pinned_baselines: Status of pinned baseline ratings
        """
        conn = self._get_connection()
        results = {}

        # Check 1: Win/loss invariant
        results["win_loss_invariant"] = self.check_win_loss_invariant()

        # Check 2: Duplicate game_ids
        dup_row = conn.execute("""
            SELECT COUNT(*) as dup_count FROM (
                SELECT game_id, COUNT(*) as cnt
                FROM match_history
                WHERE game_id IS NOT NULL AND game_id != ''
                GROUP BY game_id
                HAVING cnt > 1
            )
        """).fetchone()
        results["duplicate_game_ids"] = {
            "count": dup_row["dup_count"] if dup_row else 0,
            "is_valid": (dup_row["dup_count"] if dup_row else 0) == 0,
        }

        # Check 3: Null game_ids
        null_row = conn.execute("""
            SELECT COUNT(*) as null_count
            FROM match_history
            WHERE game_id IS NULL OR game_id = ''
        """).fetchone()
        results["null_game_ids"] = {
            "count": null_row["null_count"] if null_row else 0,
        }

        # Check 4: Pinned baselines - ALL random AIs must be at 400 ELO
        baseline_rows = conn.execute("""
            SELECT participant_id, rating
            FROM elo_ratings
            WHERE (
                participant_id LIKE 'baseline_random%'
                OR participant_id LIKE 'none:random%'
                OR participant_id LIKE 'random%'
                OR LOWER(participant_id) LIKE '%random%'
            )
            AND LOWER(participant_id) NOT LIKE '%heuristic%'
            AND archived_at IS NULL
        """).fetchall()

        unpinned_baselines = [
            {"id": row["participant_id"], "rating": row["rating"]}
            for row in baseline_rows
            if abs(row["rating"] - 400.0) > 0.01
        ]
        results["pinned_baselines"] = {
            "total": len(baseline_rows),
            "unpinned": unpinned_baselines,
            "is_valid": len(unpinned_baselines) == 0,
        }

        # Overall validity
        results["all_valid"] = (
            results["win_loss_invariant"]["is_valid"]
            and results["duplicate_game_ids"]["is_valid"]
            and results["pinned_baselines"]["is_valid"]
        )

        return results

    def check_phantom_models(self) -> dict[str, Any]:
        """Check for phantom models - entries in DB where model file doesn't exist.

        Returns dict with:
        - count: Number of phantom models found
        - phantoms: List of phantom model info dicts
        - is_valid: True if no phantoms found
        """
        conn = self._get_connection()

        # Get all participants with model paths (exclude baselines)
        rows = conn.execute("""
            SELECT p.participant_id, p.model_path, r.rating, r.games_played
            FROM participants p
            LEFT JOIN elo_ratings r ON p.participant_id = r.participant_id
            WHERE p.model_path IS NOT NULL AND p.model_path != ''
              AND LOWER(p.participant_id) NOT LIKE '%random%'
              AND LOWER(p.participant_id) NOT LIKE '%heuristic%'
              AND LOWER(p.participant_id) NOT LIKE 'baseline_%'
              AND LOWER(p.participant_id) NOT LIKE 'd1%'
              AND LOWER(p.participant_id) NOT LIKE 'd2%'
              AND LOWER(p.participant_id) NOT LIKE 'd3%'
              AND LOWER(p.participant_id) NOT LIKE 'tier%'
        """).fetchall()

        phantoms = []
        for row in rows:
            model_path = row["model_path"]
            if model_path:
                path = Path(model_path)
                # Check various locations
                exists = (
                    path.exists()
                    or (Path("models") / path).exists()
                    or (Path("models") / path.name).exists()
                    or (Path("models_essential") / path.name).exists()
                )
                if not exists:
                    phantoms.append({
                        "participant_id": row["participant_id"],
                        "model_path": model_path,
                        "rating": row["rating"],
                        "games_played": row["games_played"],
                    })

        return {
            "count": len(phantoms),
            "phantoms": phantoms[:50],  # Limit to 50 for readability
            "total_if_truncated": len(phantoms) if len(phantoms) > 50 else None,
            "is_valid": len(phantoms) == 0,
        }


# =============================================================================
# Singleton Access
# =============================================================================

def get_elo_database(db_path: Path | None = None) -> EloDatabase:
    """Get or create the singleton EloDatabase instance.

    Args:
        db_path: Optional custom database path. If provided on first call,
                 will be used for the singleton. Subsequent calls ignore this.

    Returns:
        The singleton EloDatabase instance.
    """
    global _elo_db_instance

    with _elo_db_lock:
        if _elo_db_instance is None:
            _elo_db_instance = EloDatabase(db_path)
        return _elo_db_instance


def reset_elo_database() -> None:
    """Reset the singleton (for testing)."""
    global _elo_db_instance

    with _elo_db_lock:
        if _elo_db_instance is not None:
            _elo_db_instance.close()
            _elo_db_instance = None

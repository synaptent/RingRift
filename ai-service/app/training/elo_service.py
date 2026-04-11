#!/usr/bin/env python3
"""Centralized Elo Rating Service - Single source of truth for all model ratings.

This module provides THE authoritative interface for all Elo operations across
the RingRift AI improvement infrastructure. All scripts should import from here
rather than implementing their own Elo access.

Features:
- Persistent SQLite storage with unified_elo.db
- Thread-safe operations with connection pooling
- Automatic schema migrations
- Feedback hooks for training parameter adaptation
- Integration with model lifecycle management

Usage:
    from app.training.elo_service import EloService, get_elo_service

    # Get singleton instance
    elo = get_elo_service()

    # Register and rate models
    elo.register_model("model_v1", board_type="square8", num_players=2)
    elo.record_match("model_v1", "model_v2", winner="model_v1", board_type="square8", num_players=2)

    # Get feedback signals for training
    feedback = elo.get_training_feedback("square8", 2)
    if feedback.elo_stagnating:
        # Adjust training parameters
        pass
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Path setup
from app.training.elo_algorithms import compute_elo_update, is_random_participant
from app.training.elo_api import (
    get_database_stats,
    get_elo_trend,
    get_elo_trend_for_config,
    get_head_to_head,
    get_leaderboard,
    get_match_history,
    get_rating_history,
    register_models,
    update_elo_after_match,
)
from app.training.elo_backend import (
    EloBackendType,
    check_raft_elo_store_available,
    get_raft_elo_store,
    reset_raft_elo_store_cache,
)
from app.training.elo_reporting_mixin import EloReportingMixin
from app.training.elo_types import (
    EloRating,
    LeaderboardEntry,
    MatchResult,
    TrainingFeedback,
)
from app.utils.paths import UNIFIED_ELO_DB
from app.utils.torch_utils import safe_load_checkpoint

DEFAULT_ELO_DB_PATH = UNIFIED_ELO_DB

# Import canonical thresholds
try:
    from app.config.thresholds import (
        BASELINE_ELO_RANDOM,
        ELO_K_FACTOR,
        INITIAL_ELO_RATING,
        MIN_GAMES_FOR_ELO,
        get_pinned_baseline_rating,
    )
except ImportError:
    # Fallback defaults if thresholds not available
    INITIAL_ELO_RATING = 1500.0
    ELO_K_FACTOR = 32
    MIN_GAMES_FOR_ELO = 30
    BASELINE_ELO_RANDOM = 400

    def get_pinned_baseline_rating(participant_id: str) -> float | None:
        """Fallback baseline pinning when thresholds.py is unavailable."""
        return float(BASELINE_ELO_RANDOM) if is_random_participant(participant_id) else None

# Import coordination for single-writer enforcement
# Using the new coordination module (cluster_coordinator is deprecated)
try:
    from app.coordination.helpers import (
        get_orchestrator_roles,
        get_role_holder,
        has_coordination as _has_coordination,
        has_role,
    )
    HAS_COORDINATION = _has_coordination()
    OrchestratorRole = get_orchestrator_roles()
except ImportError:
    HAS_COORDINATION = False
    OrchestratorRole = None
    has_role = None
    get_role_holder = None

# Singleton instance
_elo_service_instance: EloService | None = None
_elo_service_lock = threading.RLock()

class EloService(EloReportingMixin):
    """Centralized Elo rating service with feedback integration and single-writer enforcement."""

    # Use canonical thresholds from app.config.thresholds
    K_FACTOR = float(ELO_K_FACTOR)
    INITIAL_ELO = float(INITIAL_ELO_RATING)
    CONFIDENCE_GAMES = MIN_GAMES_FOR_ELO  # Games needed for high confidence

    def __init__(
        self,
        db_path: Path | None = None,
        enforce_single_writer: bool = True,
        use_raft: bool = True,
    ):
        """Initialize the Elo service.

        Args:
            db_path: Path to SQLite database
            enforce_single_writer: If True, check cluster coordination before writes
            use_raft: If True, use Raft backend when available for strong consistency
        """
        self.db_path = db_path or DEFAULT_ELO_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._feedback_callbacks: list[Callable[[TrainingFeedback], None]] = []
        self._enforce_single_writer = enforce_single_writer and HAS_COORDINATION
        self._use_raft = use_raft

        # Determine backend type (Dec 30, 2025 - P5.2)
        self._backend: EloBackendType = EloBackendType.SQLITE
        if self._use_raft and check_raft_elo_store_available():
            self._backend = EloBackendType.RAFT
            logger.info("EloService using Raft backend for cluster-wide consistency")
        else:
            logger.debug("EloService using SQLite backend")

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection.

        Mar 2026: Also recreate if the cached connection was closed externally.
        This fixes "Cannot operate on a closed database" errors when the P2P
        EloSyncManager closes the DB during a restart and the EloProgressTracker
        tries to use the stale cached connection.
        """
        conn = getattr(self._local, 'connection', None)
        if conn is not None:
            try:
                conn.execute("SELECT 1")
            except Exception:
                # Connection is stale/closed — recreate
                conn = None
                self._local.connection = None
        if conn is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    @contextmanager
    def _transaction(self, is_write: bool = True):
        """Context manager for database transactions.

        Args:
            is_write: If True, check single-writer enforcement before proceeding
        """
        # Check single-writer enforcement for write operations
        if (is_write and self._enforce_single_writer and OrchestratorRole is not None
                and has_role is not None and has_role(OrchestratorRole.TOURNAMENT_RUNNER)):
            # Check if tournament role is held (tournaments write to Elo DB)
            holder_info = get_role_holder(OrchestratorRole.TOURNAMENT_RUNNER) if get_role_holder is not None else None
            if holder_info and hasattr(holder_info, 'pid') and holder_info.pid != os.getpid():
                raise RuntimeError(
                    f"Elo write blocked: TOURNAMENT_RUNNER role held by PID {holder_info.pid}. "
                    "Only one process should write to Elo DB at a time."
                )

        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise

    def check_write_permission(self) -> tuple[bool, str]:
        """Check if this process can write to the Elo database.

        Returns:
            (can_write, reason) tuple
        """
        if not self._enforce_single_writer:
            return True, "Single-writer enforcement disabled"

        if OrchestratorRole is None or has_role is None:
            return True, "No coordinator available"

        if has_role(OrchestratorRole.TOURNAMENT_RUNNER):
            holder_info = get_role_holder(OrchestratorRole.TOURNAMENT_RUNNER) if get_role_holder is not None else None
            if holder_info and hasattr(holder_info, 'pid'):
                if holder_info.pid == os.getpid():
                    return True, "This process holds TOURNAMENT_RUNNER role"
                return False, f"TOURNAMENT_RUNNER role held by PID {holder_info.pid}"

        return True, "No conflicting role held"

    def execute_query(
        self,
        query: str,
        params: tuple = ()
    ) -> list[sqlite3.Row]:
        """Execute a read-only query and return results.

        This provides a centralized way to run custom queries against the Elo
        database while benefiting from connection pooling and thread-safety.

        Args:
            query: SQL query string (should be read-only SELECT)
            params: Query parameters tuple

        Returns:
            List of sqlite3.Row objects (supports both index and name access)

        Example:
            elo = get_elo_service()
            rows = elo.execute_query(
                "SELECT participant_id, rating FROM elo_ratings WHERE rating > ?",
                (1300,)
            )
            for row in rows:
                print(f"{row['participant_id']}: {row['rating']}")
        """
        conn = self._get_connection()
        cursor = conn.execute(query, params)
        return cursor.fetchall()

    @property
    def backend(self) -> EloBackendType:
        """Get the current backend type.

        Returns:
            EloBackendType.RAFT if using Raft consensus, EloBackendType.SQLITE otherwise
        """
        return self._backend

    def is_using_raft(self) -> bool:
        """Check if this service is using Raft backend.

        Returns:
            True if Raft backend is active
        """
        return self._backend == EloBackendType.RAFT

    def _record_match_raft(
        self,
        match_id: str,
        participant_a: str,
        participant_b: str,
        winner: str | None,
        board_type: str,
        num_players: int,
        game_length: int = 0,
        duration_sec: float = 0.0,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Record a match via Raft for cluster-wide consistency.

        Args:
            match_id: Unique match identifier
            participant_a: First participant
            participant_b: Second participant
            winner: Winner ID or None for draw
            board_type: Board type
            num_players: Number of players
            game_length: Number of moves
            duration_sec: Duration in seconds

        Returns:
            Tuple of (elo_before, elo_after, elo_changes) dicts
        """
        raft_store = get_raft_elo_store()
        if not raft_store:
            raise RuntimeError("Raft Elo store not available")

        result = raft_store.record_match(
            match_id=match_id,
            participant_a=participant_a,
            participant_b=participant_b,
            winner_id=winner,
            board_type=board_type,
            num_players=num_players,
            game_length=game_length,
            duration_sec=duration_sec,
            k_factor=self.K_FACTOR,
        )

        # Also update local SQLite cache for fast reads
        # This ensures local queries don't need to hit Raft
        elo_before = result.get("elo_before", {})
        elo_after = result.get("elo_after", {})
        elo_changes = result.get("elo_changes", {})

        for pid in [participant_a, participant_b]:
            pinned_rating = get_pinned_baseline_rating(pid)
            if pinned_rating is not None:
                elo_after[pid] = float(pinned_rating)
                elo_changes[pid] = 0.0

        with self._transaction() as conn:
            for pid in [participant_a, participant_b]:
                new_rating = elo_after.get(pid, self.INITIAL_ELO)
                score = 1.0 if winner == pid else (0.0 if winner and winner != pid else 0.5)
                win_inc = 1 if score == 1.0 else 0
                loss_inc = 1 if score == 0.0 else 0
                draw_inc = 1 if score == 0.5 else 0

                conn.execute("""
                    INSERT INTO elo_ratings (participant_id, board_type, num_players, rating,
                                           games_played, wins, losses, draws, peak_rating, last_update)
                    VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
                    ON CONFLICT(participant_id, board_type, num_players) DO UPDATE SET
                        rating = excluded.rating,
                        games_played = games_played + 1,
                        wins = wins + excluded.wins,
                        losses = losses + excluded.losses,
                        draws = draws + excluded.draws,
                        peak_rating = MAX(peak_rating, excluded.peak_rating),
                        last_update = excluded.last_update
                """, (
                    pid, board_type, num_players, new_rating,
                    win_inc, loss_inc, draw_inc, new_rating, time.time()
                ))

        return elo_before, elo_after, elo_changes

    def _init_db(self):
        """Initialize database schema with all required tables."""
        with self._transaction() as conn:
            # Participants table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS participants (
                    participant_id TEXT PRIMARY KEY,
                    participant_type TEXT NOT NULL DEFAULT 'model',
                    ai_type TEXT,
                    difficulty INTEGER,
                    use_neural_net INTEGER DEFAULT 0,
                    model_path TEXT,
                    model_version TEXT,
                    metadata TEXT,
                    created_at REAL,
                    last_seen REAL,
                    nn_model_id TEXT,
                    nn_model_path TEXT,
                    ai_algorithm TEXT,
                    algorithm_config TEXT,
                    is_composite INTEGER DEFAULT 0,
                    -- January 2026: Only track Elo for deployable models
                    -- Non-deployable models (ephemeral checkpoints) should not pollute Elo DB
                    is_deployable INTEGER DEFAULT 0
                )
            """)

            # Elo ratings per configuration
            conn.execute("""
                CREATE TABLE IF NOT EXISTS elo_ratings (
                    participant_id TEXT NOT NULL,
                    board_type TEXT NOT NULL,
                    num_players INTEGER NOT NULL,
                    rating REAL DEFAULT 1500.0,
                    games_played INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    draws INTEGER DEFAULT 0,
                    peak_rating REAL DEFAULT 1500.0,
                    last_update REAL,
                    -- Jan 2026: Harness tracking for composite Elo
                    harness_type TEXT,          -- e.g., "gumbel_mcts", "minimax", "policy_only"
                    simulation_count INTEGER,   -- e.g., 64, 200, 800, 1600
                    PRIMARY KEY (participant_id, board_type, num_players)
                )
            """)

            # Match history
            conn.execute("""
                CREATE TABLE IF NOT EXISTS match_history (
                    id TEXT PRIMARY KEY,
                    participant_ids TEXT NOT NULL,
                    winner_id TEXT,
                    game_length INTEGER,
                    duration_sec REAL,
                    board_type TEXT NOT NULL,
                    num_players INTEGER NOT NULL,
                    timestamp TEXT,
                    elo_before TEXT,
                    elo_after TEXT,
                    tournament_id TEXT,
                    metadata TEXT
                )
            """)

            # Migration: add metadata column if not exists (for existing DBs)
            try:
                conn.execute("ALTER TABLE match_history ADD COLUMN metadata TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Migration: add winner_id column if not exists (for existing DBs)
            try:
                conn.execute("ALTER TABLE match_history ADD COLUMN winner_id TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Migration: add harness_type column for multi-harness tracking (Jan 12, 2026)
            # This enables tracking which AI harness (gumbel_mcts, minimax, etc.) was used
            try:
                conn.execute("ALTER TABLE match_history ADD COLUMN harness_type TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Elo history for trend analysis
            conn.execute("""
                CREATE TABLE IF NOT EXISTS elo_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    participant_id TEXT NOT NULL,
                    board_type TEXT NOT NULL,
                    num_players INTEGER NOT NULL,
                    rating REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    iteration INTEGER
                )
            """)

            # Training feedback signals
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_feedback (
                    board_type TEXT NOT NULL,
                    num_players INTEGER NOT NULL,
                    iteration INTEGER NOT NULL,
                    best_elo REAL,
                    elo_delta REAL,
                    epochs_multiplier REAL DEFAULT 1.0,
                    lr_multiplier REAL DEFAULT 1.0,
                    curriculum_stage INTEGER DEFAULT 0,
                    timestamp REAL,
                    PRIMARY KEY (board_type, num_players, iteration)
                )
            """)

            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_elo_config ON elo_ratings(board_type, num_players)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_match_time ON match_history(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_participant ON elo_history(participant_id, board_type, num_players)")

            # Schema migrations for existing databases (December 2025)
            # Add peak_rating column if missing (older databases don't have it)
            cursor = conn.execute("PRAGMA table_info(elo_ratings)")
            columns = {row[1] for row in cursor.fetchall()}
            if "peak_rating" not in columns:
                conn.execute("ALTER TABLE elo_ratings ADD COLUMN peak_rating REAL DEFAULT 1500.0")
                logger.info("Migrated elo_ratings: added peak_rating column")

            # January 2026: Add is_deployable column for filtering ephemeral checkpoints
            cursor = conn.execute("PRAGMA table_info(participants)")
            participant_columns = {row[1] for row in cursor.fetchall()}
            participant_column_migrations = {
                "nn_model_id": "TEXT",
                "nn_model_path": "TEXT",
                "ai_algorithm": "TEXT",
                "algorithm_config": "TEXT",
                "is_composite": "INTEGER DEFAULT 0",
                "is_deployable": "INTEGER DEFAULT 0",
            }
            added_participant_columns: list[str] = []
            for column_name, column_def in participant_column_migrations.items():
                if column_name in participant_columns:
                    continue
                conn.execute(
                    f"ALTER TABLE participants ADD COLUMN {column_name} {column_def}"
                )
                participant_columns.add(column_name)
                added_participant_columns.append(column_name)
                logger.info(f"Migrated participants: added {column_name} column")

            if "is_deployable" in added_participant_columns:
                # Mark existing canonical models as deployable
                conn.execute("""
                    UPDATE participants SET is_deployable = 1
                    WHERE model_path LIKE '%canonical_%' OR model_path LIKE '%ringrift_best_%'
                """)
                logger.info("Migrated participants: backfilled is_deployable for canonical models")

            # January 2026: Add harness tracking columns for composite Elo
            cursor = conn.execute("PRAGMA table_info(elo_ratings)")
            elo_columns = {row[1] for row in cursor.fetchall()}
            if "harness_type" not in elo_columns:
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
                logger.info("Migrated elo_ratings: added harness_type and simulation_count columns")

            # Model identity tracking tables (January 2026)
            # Track model files by SHA256 hash for deduplication and alias resolution
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_identities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_path TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    file_size INTEGER,
                    first_seen_at REAL DEFAULT (strftime('%s', 'now')),
                    last_verified_at REAL DEFAULT (strftime('%s', 'now')),
                    UNIQUE(model_path, content_sha256)
                )
            """)

            # Participant aliases for same model content
            conn.execute("""
                CREATE TABLE IF NOT EXISTS participant_aliases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    primary_participant_id TEXT NOT NULL,
                    alias_participant_id TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    created_at REAL DEFAULT (strftime('%s', 'now')),
                    UNIQUE(primary_participant_id, alias_participant_id)
                )
            """)

            # Indexes for hash lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_identities_hash ON model_identities(content_sha256)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_identities_path ON model_identities(model_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_participant_aliases_primary ON participant_aliases(primary_participant_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_participant_aliases_alias ON participant_aliases(alias_participant_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_participant_aliases_hash ON participant_aliases(content_sha256)")

    def register_participant(
        self,
        participant_id: str,
        name: str | None = None,  # Deprecated: not stored in DB, use participant_id
        ai_type: str = "unknown",
        difficulty: int | None = None,
        use_neural_net: bool = False,
        model_path: str | None = None,
        metadata: dict | None = None,
        is_deployable: bool = False,
    ) -> None:
        """Register a new participant (model or AI baseline).

        Note: The `name` parameter is deprecated and ignored. The participant_id
        serves as the display name.

        Args:
            is_deployable: If True, this participant is eligible for production use.
                Only deployable models should have persistent Elo tracking.
        """
        with self._transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO participants
                (participant_id, ai_type, difficulty, use_neural_net, model_path, created_at, metadata, is_deployable)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                participant_id,
                ai_type,
                difficulty,
                int(use_neural_net),
                model_path,
                time.time(),
                json.dumps(metadata) if metadata else None,
                int(is_deployable),
            ))

    def _ensure_participant_exists(
        self,
        participant_id: str,
        ai_type: str = "neural_net",
        model_path: str | None = None,
    ) -> None:
        """Ensure participant exists in participants table before adding Elo ratings.

        January 17, 2026: Added to fix issue where new models got Elo ratings
        but weren't registered as participants, causing them to be invisible
        in get_leaderboard() which uses a JOIN with the participants table.

        This method uses INSERT OR IGNORE to avoid overwriting existing participant
        data while ensuring new participants are properly registered.

        Args:
            participant_id: Unique participant identifier
            ai_type: Type of AI (default: "neural_net")
            model_path: Optional path to model file
        """
        # Skip baseline participants (handled separately)
        if any(x in participant_id.lower() for x in ["random", "heuristic", "dummy", "baseline", "none:"]):
            return

        with self._transaction() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO participants
                (participant_id, participant_type, ai_type, use_neural_net, model_path,
                 created_at, last_seen, is_deployable)
                VALUES (?, 'AI', ?, 1, ?, ?, ?, 1)
            """, (
                participant_id,
                ai_type,
                model_path or f"models/{participant_id}.pth",
                time.time(),
                time.time(),
            ))

    # =========================================================================
    # Model Identity Tracking (January 2026)
    # Track model files by SHA256 hash for deduplication and alias resolution
    # =========================================================================

    def _compute_model_hash(self, model_path: str) -> str | None:
        """Compute SHA256 hash of model file content.

        Args:
            model_path: Path to the model file

        Returns:
            SHA256 hex digest or None if file not found
        """
        path = Path(model_path)

        # Check common model directories if not found directly
        if not path.exists():
            for model_dir in [Path("models"), Path("models_essential")]:
                candidate = model_dir / path.name
                if candidate.exists():
                    path = candidate
                    break

        if not path.exists():
            return None

        try:
            sha256 = hashlib.sha256()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except (OSError, IOError) as e:
            logger.warning(f"Could not compute hash for {model_path}: {e}")
            return None

    def _store_model_identity(
        self,
        model_path: str,
        content_sha256: str,
        file_size: int | None = None,
    ) -> None:
        """Store model file identity in database.

        Args:
            model_path: Path to the model file
            content_sha256: SHA256 hash of file content
            file_size: Optional file size in bytes
        """
        with self._transaction() as conn:
            conn.execute("""
                INSERT INTO model_identities (model_path, content_sha256, file_size, last_verified_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(model_path, content_sha256) DO UPDATE SET
                    last_verified_at = excluded.last_verified_at
            """, (model_path, content_sha256, file_size, time.time()))

    def _find_participant_by_hash(self, content_sha256: str) -> str | None:
        """Find an existing participant ID with the same model content hash.

        Args:
            content_sha256: SHA256 hash of model content

        Returns:
            Participant ID if found, None otherwise
        """
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT DISTINCT p.participant_id
            FROM model_identities mi
            JOIN participants p ON mi.model_path = p.model_path
            WHERE mi.content_sha256 = ?
            ORDER BY p.created_at ASC
            LIMIT 1
        """, (content_sha256,))
        row = cursor.fetchone()
        return row[0] if row else None

    def _create_participant_alias(
        self,
        primary_id: str,
        alias_id: str,
        content_sha256: str,
    ) -> None:
        """Create an alias relationship between two participant IDs.

        The primary_id is the canonical participant, and alias_id references
        the same model content.

        Args:
            primary_id: The canonical participant ID (with most games)
            alias_id: The alias participant ID pointing to same model
            content_sha256: SHA256 hash of the shared model content
        """
        with self._transaction() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO participant_aliases
                (primary_participant_id, alias_participant_id, content_sha256)
                VALUES (?, ?, ?)
            """, (primary_id, alias_id, content_sha256))
            logger.info(f"Created participant alias: {alias_id} -> {primary_id}")

    def _resolve_participant_alias(self, participant_id: str) -> str:
        """Resolve a participant ID to its primary ID if it's an alias.

        Args:
            participant_id: The participant ID to resolve

        Returns:
            The primary participant ID (or the original if not an alias)
        """
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT primary_participant_id
            FROM participant_aliases
            WHERE alias_participant_id = ?
            LIMIT 1
        """, (participant_id,))
        row = cursor.fetchone()
        return row[0] if row else participant_id

    def get_model_identity(self, model_path: str) -> dict[str, Any] | None:
        """Get stored identity information for a model file.

        Args:
            model_path: Path to the model file

        Returns:
            Dict with model_path, content_sha256, file_size, first_seen_at, last_verified_at
            or None if not found
        """
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT model_path, content_sha256, file_size, first_seen_at, last_verified_at
            FROM model_identities
            WHERE model_path = ?
            ORDER BY last_verified_at DESC
            LIMIT 1
        """, (model_path,))
        row = cursor.fetchone()
        if row:
            return {
                "model_path": row[0],
                "content_sha256": row[1],
                "file_size": row[2],
                "first_seen_at": row[3],
                "last_verified_at": row[4],
            }
        return None

    def get_participants_for_hash(self, content_sha256: str) -> list[str]:
        """Get all participant IDs associated with a model content hash.

        Args:
            content_sha256: SHA256 hash of model content

        Returns:
            List of participant IDs using this model content
        """
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT DISTINCT p.participant_id
            FROM model_identities mi
            JOIN participants p ON mi.model_path = p.model_path
            WHERE mi.content_sha256 = ?
        """, (content_sha256,))
        return [row[0] for row in cursor.fetchall()]

    def get_participant_by_model_path(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
    ) -> "EloRating | None":
        """Get Elo rating for a model file by computing its hash.

        January 2026: Public API for looking up Elo ratings by model file.
        Computes SHA256 hash of the model file and finds the matching participant.

        Args:
            model_path: Path to the model file
            board_type: Board type (e.g., 'square8', 'hex8')
            num_players: Number of players (2, 3, or 4)

        Returns:
            EloRating if model is found in database, None otherwise
        """
        content_hash = self._compute_model_hash(model_path)
        if content_hash is None:
            return None

        participant_id = self._find_participant_by_hash(content_hash)
        if participant_id is None:
            return None

        return self.get_rating(participant_id, board_type, num_players)

    def verify_and_update_model_identity(
        self,
        participant_id: str,
        model_path: str,
    ) -> tuple[bool, str | None]:
        """Verify model file and update identity tracking.

        Computes current hash and checks if it matches the stored hash.
        If different, the model file has changed (e.g., via promotion).

        Args:
            participant_id: The participant ID
            model_path: Path to the model file

        Returns:
            Tuple of (has_changed, new_hash). has_changed is True if model
            content differs from stored identity.
        """
        current_hash = self._compute_model_hash(model_path)
        if current_hash is None:
            return False, None

        stored_identity = self.get_model_identity(model_path)
        if stored_identity is None:
            # First time seeing this model, store identity
            file_size = Path(model_path).stat().st_size if Path(model_path).exists() else None
            self._store_model_identity(model_path, current_hash, file_size)
            return False, current_hash

        if stored_identity["content_sha256"] != current_hash:
            # Model file has changed
            logger.info(
                f"Model content changed for {participant_id}: "
                f"{stored_identity['content_sha256'][:12]}... -> {current_hash[:12]}..."
            )
            # Store new identity
            file_size = Path(model_path).stat().st_size if Path(model_path).exists() else None
            self._store_model_identity(model_path, current_hash, file_size)
            return True, current_hash

        # No change, but update last_verified timestamp
        self._store_model_identity(
            model_path,
            current_hash,
            stored_identity.get("file_size")
        )
        return False, current_hash

    def handle_model_promotion(
        self,
        source_model_path: str,
        target_model_path: str,
        source_participant_id: str,
        target_participant_id: str,
        board_type: str,
        num_players: int,
    ) -> dict:
        """Handle model promotion with hash-based identity tracking.

        When a model is promoted (copied to a canonical location), this method:
        1. Checks if the target already exists with a different hash
        2. If the source model has Elo data, transfers/aliases it to the target
        3. Updates model identity tracking

        This fixes the stale Elo problem where canonical models appear weak
        because their Elo was computed with an older model version.

        Args:
            source_model_path: Path to the model being promoted
            target_model_path: Path to the canonical location (will be overwritten)
            source_participant_id: Participant ID of the source model
            target_participant_id: Participant ID for the canonical model
            board_type: Board type for Elo lookup
            num_players: Number of players

        Returns:
            Dict with promotion status:
            - 'status': 'success', 'no_change', or 'error'
            - 'source_hash': SHA256 of source model
            - 'old_target_hash': SHA256 of previous target (if existed)
            - 'elo_transferred': True if Elo was transferred from source
            - 'elo_reset': True if Elo was reset due to model change
            - 'message': Human-readable status message

        January 2026: Added for Elo/Model Identity Tracking fix (Priority 0).
        """
        result = {
            "status": "success",
            "source_hash": None,
            "old_target_hash": None,
            "elo_transferred": False,
            "elo_reset": False,
            "message": "",
        }

        # Compute source model hash
        source_hash = self._compute_model_hash(source_model_path)
        if source_hash is None:
            result["status"] = "error"
            result["message"] = f"Could not compute hash for source model: {source_model_path}"
            logger.error(result["message"])
            return result

        result["source_hash"] = source_hash

        # Check if target already exists and get its hash
        old_target_hash = None
        if Path(target_model_path).exists():
            old_target_hash = self._compute_model_hash(target_model_path)
            result["old_target_hash"] = old_target_hash

        # If hashes are the same, no real change - just update tracking
        if old_target_hash == source_hash:
            result["status"] = "no_change"
            result["message"] = "Source and target models are identical (same hash)"
            logger.debug(f"[EloService] Promotion no-op: {source_hash[:12]}... unchanged")
            return result

        # Models are different - need to handle Elo tracking
        logger.info(
            f"[EloService] Model promotion detected: "
            f"{target_participant_id} changing from "
            f"{old_target_hash[:12] if old_target_hash else 'new'}... to {source_hash[:12]}..."
        )

        # Get source model's Elo if it exists
        source_rating = None
        try:
            source_rating = self.get_rating(source_participant_id, board_type, num_players)
            if source_rating.games_played > 0:
                logger.info(
                    f"[EloService] Source model {source_participant_id} has Elo "
                    f"{source_rating.rating:.0f} ({source_rating.games_played} games)"
                )
        except Exception as e:
            logger.debug(f"Could not get source rating: {e}")

        # Create alias from target to source if source has games
        if source_rating and source_rating.games_played > 0:
            # The source model's Elo should apply to the target (same content)
            self._create_participant_alias(
                primary_id=source_participant_id,
                alias_id=target_participant_id,
                content_sha256=source_hash,
            )
            result["elo_transferred"] = True
            result["message"] = (
                f"Elo transferred: {target_participant_id} -> {source_participant_id} "
                f"(Elo {source_rating.rating:.0f}, {source_rating.games_played} games)"
            )
            logger.info(f"[EloService] {result['message']}")
        else:
            # No source Elo - if target had Elo with old model, we need to note it's stale
            # The target's existing Elo is now invalid (different model content)
            if old_target_hash:
                # Mark that the old Elo is stale by storing the new identity
                # The alias system will handle lookups properly
                result["elo_reset"] = True
                result["message"] = (
                    f"Model content changed for {target_participant_id} - "
                    f"old Elo may be stale until re-evaluated"
                )
                logger.warning(f"[EloService] {result['message']}")
            else:
                result["message"] = f"New canonical model registered: {target_participant_id}"

        # Update model identity tracking for both source and target
        file_size = Path(source_model_path).stat().st_size if Path(source_model_path).exists() else None
        self._store_model_identity(source_model_path, source_hash, file_size)

        # Note: We store target identity AFTER the file copy happens (caller's responsibility)
        # But we return the hash so caller can verify

        return result

    def _validate_model_player_count(
        self, model_path: str, expected_num_players: int
    ) -> tuple[bool, int | None]:
        """Validate that a model's player count matches expected value.

        Args:
            model_path: Path to model checkpoint file
            expected_num_players: Expected number of players (2, 3, or 4)

        Returns:
            Tuple of (is_valid, actual_num_players). actual_num_players is None
            if it couldn't be determined from the checkpoint.
        """
        try:
            import torch
            path = Path(model_path)
            if not path.exists():
                # Check common directories
                for model_dir in [Path("models"), Path("models_essential")]:
                    candidate = model_dir / path
                    if candidate.exists():
                        path = candidate
                        break
                    candidate = model_dir / path.name
                    if candidate.exists():
                        path = candidate
                        break

            if not path.exists():
                return True, None  # Can't validate, assume OK

            checkpoint = safe_load_checkpoint(path, map_location="cpu", warn_on_unsafe=False)

            # Try to get num_players from checkpoint metadata
            actual_num_players = checkpoint.get("num_players")

            # If not in metadata, infer from value head shape
            if actual_num_players is None:
                state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
                if isinstance(state_dict, dict):
                    # Check value_fc2 shape - first dim is num_players
                    for key in ["value_fc2.weight", "value_head.fc2.weight"]:
                        if key in state_dict:
                            actual_num_players = state_dict[key].shape[0]
                            break

            if actual_num_players is None:
                return True, None  # Can't determine, assume OK

            return actual_num_players == expected_num_players, actual_num_players

        except Exception as e:
            logger.debug(f"Could not validate player count for {model_path}: {e}")
            return True, None  # On error, don't block registration

    def _validate_model_path(self, model_path: str | None) -> bool:
        """Validate that a model file exists.

        Args:
            model_path: Path to model file (absolute or relative)

        Returns:
            True if file exists or path is None, False otherwise
        """
        if not model_path:
            return True  # None is valid for baselines

        path = Path(model_path)

        # Check absolute path
        if path.is_absolute():
            return path.exists()

        # Check relative to common model directories
        model_dirs = [
            Path("models"),
            Path("models_essential"),
            Path("."),
        ]
        for model_dir in model_dirs:
            candidate = model_dir / path
            if candidate.exists():
                return True
            # Try just the filename
            candidate = model_dir / path.name
            if candidate.exists():
                return True

        return path.exists()

    def register_model(
        self,
        model_id: str,
        board_type: str,
        num_players: int,
        model_path: str | None = None,
        parent_model_id: str | None = None,
        validate_file: bool = True,
        is_deployable: bool = False,
    ) -> None:
        """Register a trained model and initialize its Elo rating.

        Args:
            model_id: Unique identifier for the model
            board_type: Board type (e.g., 'square8', 'hex8')
            num_players: Number of players (2, 3, or 4)
            model_path: Optional path to model file
            parent_model_id: Optional ID of parent model (for lineage tracking)
            validate_file: If True, verify model file exists before registering
            is_deployable: If True, this model is a candidate for production use.
                Only deployable models should have persistent Elo tracking.
                Set True for: canonical models, preserved high-Elo checkpoints.
                Set False for: ephemeral training checkpoints, experiments.
        """
        # Validate model file exists (prevent phantom entries)
        if validate_file and model_path and not self._validate_model_path(model_path):
            logger.warning(
                f"Model file not found, skipping registration: {model_path} "
                f"(model_id={model_id}). Use validate_file=False to override."
            )
            return

        # Validate player count matches (prevent player count mismatch - Dec 2025)
        if validate_file and model_path:
            is_valid, actual_players = self._validate_model_player_count(
                model_path, num_players
            )
            if not is_valid:
                logger.error(
                    f"Player count mismatch! Model {model_path} has {actual_players} "
                    f"players but trying to register for {num_players}-player config. "
                    f"Skipping registration to prevent invalid Elo entries."
                )
                return

        # Compute model content hash for identity tracking (January 2026)
        content_hash: str | None = None
        existing_participant: str | None = None
        if model_path:
            content_hash = self._compute_model_hash(model_path)
            if content_hash:
                # Check if this exact model content is already registered under another ID
                existing_participant = self._find_participant_by_hash(content_hash)
                if existing_participant and existing_participant != model_id:
                    # Create alias relationship - the existing participant becomes primary
                    self._create_participant_alias(
                        primary_id=existing_participant,
                        alias_id=model_id,
                        content_sha256=content_hash
                    )
                    logger.info(
                        f"Model {model_id} has same content as {existing_participant}, "
                        f"created alias (hash: {content_hash[:12]}...)"
                    )

        # Register as participant
        # Auto-detect deployable status from path if not explicitly set
        effective_deployable = is_deployable
        if not effective_deployable and model_path:
            # Canonical and best models are always deployable
            if "canonical_" in model_path or "ringrift_best_" in model_path:
                effective_deployable = True
            # Preserved high-Elo models are deployable
            if "/preserved/" in model_path:
                effective_deployable = True

        self.register_participant(
            participant_id=model_id,
            name=model_id,
            ai_type="neural_net",
            use_neural_net=True,
            model_path=model_path,
            metadata={"parent_model_id": parent_model_id, "content_sha256": content_hash},
            is_deployable=effective_deployable,
        )

        # Store model identity for future tracking
        if model_path and content_hash:
            file_size = None
            path = Path(model_path)
            if path.exists():
                file_size = path.stat().st_size
            self._store_model_identity(model_path, content_hash, file_size)

        # Initialize rating
        self.get_rating(model_id, board_type, num_players)

    def get_rating(
        self,
        participant_id: str,
        board_type: str,
        num_players: int
    ) -> EloRating:
        """Get participant's Elo rating, creating initial if needed.

        Note: Baseline players are anchored by thresholds.get_pinned_baseline_rating()
        to serve as fixed reference points and prevent rating inflation.
        """
        # Anchor canonical baseline participants at fixed Elo to prevent rating drift.
        pinned_rating = get_pinned_baseline_rating(participant_id)
        if pinned_rating is not None:
            # Still fetch games_played from DB for stats, but rating is fixed
            conn = self._get_connection()
            cursor = conn.execute("""
                SELECT games_played, wins, losses, draws, last_update
                FROM elo_ratings
                WHERE participant_id = ? AND board_type = ? AND num_players = ?
            """, (participant_id, board_type, num_players))
            row = cursor.fetchone()
            if row:
                return EloRating(
                    participant_id=participant_id,
                    rating=float(pinned_rating),  # ANCHORED
                    games_played=row["games_played"],
                    wins=row["wins"],
                    losses=row["losses"],
                    draws=row["draws"],
                    last_update=row["last_update"] or 0.0,
                    confidence=1.0  # Baselines are always reliable anchors
                )
            # Create entry for baseline participant at anchored rating
            with self._transaction() as txn_conn:
                txn_conn.execute("""
                    INSERT OR IGNORE INTO elo_ratings
                    (participant_id, board_type, num_players, rating, last_update)
                    VALUES (?, ?, ?, ?, ?)
                """, (participant_id, board_type, num_players, float(pinned_rating), time.time()))
            return EloRating(
                participant_id=participant_id,
                rating=float(pinned_rating),  # ANCHORED
                confidence=1.0
            )

        # Resolve participant alias (January 2026 - model identity tracking)
        # If this participant is an alias for a primary participant with more games,
        # use the primary's rating instead (they reference the same model content)
        resolved_id = self._resolve_participant_alias(participant_id)
        lookup_id = resolved_id if resolved_id != participant_id else participant_id
        if lookup_id != participant_id:
            logger.debug(f"Resolved alias {participant_id} -> {lookup_id} for Elo lookup")

        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT rating, games_played, wins, losses, draws, last_update
            FROM elo_ratings
            WHERE participant_id = ? AND board_type = ? AND num_players = ?
        """, (lookup_id, board_type, num_players))
        row = cursor.fetchone()

        if row:
            confidence = min(1.0, row["games_played"] / self.CONFIDENCE_GAMES)
            # Jan 13, 2026: Ensure row exists for original participant_id
            # When alias resolves to different lookup_id, we need both rows
            # so that record_match() UPDATE succeeds for participant_id
            if lookup_id != participant_id:
                with self._transaction() as txn_conn:
                    txn_conn.execute("""
                        INSERT OR IGNORE INTO elo_ratings
                        (participant_id, board_type, num_players, rating, last_update,
                         games_played, wins, losses, draws, peak_rating)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        participant_id, board_type, num_players,
                        row["rating"], row["last_update"],
                        row["games_played"], row["wins"], row["losses"], row["draws"],
                        row["rating"]  # peak_rating = current rating
                    ))
            return EloRating(
                participant_id=participant_id,
                rating=row["rating"],
                games_played=row["games_played"],
                wins=row["wins"],
                losses=row["losses"],
                draws=row["draws"],
                last_update=row["last_update"] or 0.0,
                confidence=confidence
            )

        # Create initial rating
        # January 17, 2026: Ensure participant is registered so it appears in leaderboard
        self._ensure_participant_exists(participant_id)

        with self._transaction() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO elo_ratings
                (participant_id, board_type, num_players, rating, last_update)
                VALUES (?, ?, ?, ?, ?)
            """, (participant_id, board_type, num_players, self.INITIAL_ELO, time.time()))

        return EloRating(
            participant_id=participant_id,
            rating=self.INITIAL_ELO,
            confidence=0.0
        )

    def record_match(
        self,
        participant_a: str,
        participant_b: str,
        winner: str | None,  # None for draw
        board_type: str,
        num_players: int,
        game_length: int = 0,
        duration_sec: float = 0.0,
        tournament_id: str | None = None,
        metadata: dict | None = None,
        # December 30, 2025: Multi-harness evaluation support
        # January 2026: Default to "gumbel_mcts" to ensure harness tracking
        harness_type: str = "gumbel_mcts",
        is_multi_harness: bool = False,
    ) -> MatchResult:
        """Record a match result and update Elo ratings.

        .. note:: Prefer using ``app.training.elo_recording.safe_record_elo()``

            The elo_recording facade provides better validation, required harness_type,
            model type detection, and DLQ integration. Direct calls to record_match()
            are supported for backwards compatibility but may miss important metadata.

            Example using facade::

                from app.training.elo_recording import safe_record_elo, EloMatchSpec, HarnessType

                result = safe_record_elo(EloMatchSpec(
                    participant_a="model_v1",
                    participant_b="heuristic",
                    winner="model_v1",
                    board_type="hex8",
                    num_players=2,
                    harness_type=HarnessType.GUMBEL_MCTS,  # REQUIRED
                ))

        Args:
            metadata: Optional dict with match metadata. Useful keys:
                - weight_profile_a: Heuristic weight profile ID for participant A
                - weight_profile_b: Heuristic weight profile ID for participant B
                - source: Origin of the match (e.g., "tournament", "selfplay")
            harness_type: AI harness type used for this match (e.g., "gumbel_mcts", "minimax").
                December 30, 2025: Added to support multi-harness evaluation tracking.
                January 2026: Now defaults to "gumbel_mcts" instead of None to ensure
                all matches have harness tracking. PREFER using elo_recording facade.
            is_multi_harness: True if this match is part of a multi-harness evaluation.
                When True, the harness_type is included in emitted events.
        """
        match_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # December 30, 2025: Merge harness info into metadata
        if harness_type or is_multi_harness:
            metadata = metadata.copy() if metadata else {}
            if harness_type:
                metadata["harness_type"] = harness_type
            if is_multi_harness:
                metadata["is_multi_harness"] = is_multi_harness

        # January 17, 2026: Ensure participants are registered so they appear in leaderboard
        self._ensure_participant_exists(participant_a)
        self._ensure_participant_exists(participant_b)

        # December 30, 2025 - P5.2: Route to Raft backend for cluster-wide consistency
        if self._backend == EloBackendType.RAFT:
            try:
                elo_before, elo_after, elo_changes = self._record_match_raft(
                    match_id=match_id,
                    participant_a=participant_a,
                    participant_b=participant_b,
                    winner=winner,
                    board_type=board_type,
                    num_players=num_players,
                    game_length=game_length,
                    duration_sec=duration_sec,
                )

                # Record match history in local SQLite for queries
                # match_history uses a TEXT primary key column named `id`.
                # Jan 11, 2026: Added harness_type column for multi-harness evaluation tracking
                with self._transaction() as conn:
                    conn.execute("""
                        INSERT INTO match_history
                        (id, participant_ids, winner_id, game_length, duration_sec,
                         board_type, num_players, timestamp, elo_before, elo_after,
                         tournament_id, metadata, harness_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        match_id,
                        json.dumps([participant_a, participant_b]),
                        winner,
                        game_length,
                        duration_sec,
                        board_type,
                        num_players,
                        timestamp,
                        json.dumps(elo_before),
                        json.dumps(elo_after),
                        tournament_id,
                        json.dumps(metadata) if metadata else None,
                        harness_type,
                    ))

                # Emit events (same as SQLite path)
                self._emit_elo_events(
                    participant_a, participant_b,
                    elo_before, elo_after, elo_changes,
                    board_type, num_players, duration_sec,
                )

                # December 30, 2025: Record Elo snapshots for longitudinal tracking
                self._record_elo_snapshot(participant_a, board_type, num_players)
                self._record_elo_snapshot(participant_b, board_type, num_players)

                return MatchResult(
                    match_id=match_id,
                    participant_ids=[participant_a, participant_b],
                    winner_id=winner,
                    game_length=game_length,
                    duration_sec=duration_sec,
                    board_type=board_type,
                    num_players=num_players,
                    timestamp=timestamp,
                    elo_changes=elo_changes,
                )
            except Exception as e:
                logger.warning(f"Raft record_match failed, falling back to SQLite: {e}")
                # Fall through to SQLite path

        # Get current ratings
        rating_a = self.get_rating(participant_a, board_type, num_players)
        rating_b = self.get_rating(participant_b, board_type, num_players)

        elo_before = {participant_a: rating_a.rating, participant_b: rating_b.rating}

        # Scale K-factor for multiplayer games
        # In N-player games, each pairwise matchup is 1/(N-1) of the rating info
        # This ensures consistent rating change magnitude across player counts
        base_k = self.K_FACTOR / (num_players - 1) if num_players > 2 else self.K_FACTOR

        computation = compute_elo_update(
            participant_a=participant_a,
            participant_b=participant_b,
            winner=winner,
            rating_a=rating_a.rating,
            rating_b=rating_b.rating,
            games_a=rating_a.games_played,
            games_b=rating_b.games_played,
            base_k=base_k,
            get_pinned_baseline_rating_fn=get_pinned_baseline_rating,
        )
        score_a = computation.score_a
        score_b = computation.score_b
        change_a = computation.change_a
        change_b = computation.change_b
        new_rating_a = computation.new_rating_a
        new_rating_b = computation.new_rating_b

        elo_after = {participant_a: new_rating_a, participant_b: new_rating_b}
        elo_changes = {participant_a: change_a, participant_b: change_b}

        # Update database
        with self._transaction() as conn:
            # Update ratings
            for pid, new_rating, score in [
                (participant_a, new_rating_a, score_a),
                (participant_b, new_rating_b, score_b)
            ]:
                win_inc = 1 if score == 1.0 else 0
                loss_inc = 1 if score == 0.0 else 0
                draw_inc = 1 if score == 0.5 else 0

                # Jan 13, 2026: Use cursor to check rowcount and handle missing rows
                cursor = conn.execute("""
                    UPDATE elo_ratings
                    SET rating = ?,
                        games_played = games_played + 1,
                        wins = wins + ?,
                        losses = losses + ?,
                        draws = draws + ?,
                        peak_rating = MAX(peak_rating, ?),
                        last_update = ?
                    WHERE participant_id = ? AND board_type = ? AND num_players = ?
                """, (
                    new_rating, win_inc, loss_inc, draw_inc,
                    new_rating, time.time(),
                    pid, board_type, num_players
                ))

                # Jan 13, 2026: Handle missing row by inserting if UPDATE affected 0 rows
                # Jan 16, 2026: Added INSERT result checking and retry with REPLACE
                if cursor.rowcount == 0:
                    logger.warning(
                        f"[EloService] UPDATE affected 0 rows for {pid}, "
                        f"inserting new row (board={board_type}, players={num_players})"
                    )
                    insert_cursor = conn.execute("""
                        INSERT OR IGNORE INTO elo_ratings
                        (participant_id, board_type, num_players, rating, last_update,
                         games_played, wins, losses, draws, peak_rating)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pid, board_type, num_players,
                        new_rating, time.time(),
                        1, win_inc, loss_inc, draw_inc, new_rating
                    ))
                    # Check if INSERT succeeded, retry with REPLACE if ignored
                    if insert_cursor.rowcount == 0:
                        logger.warning(
                            f"[EloService] INSERT OR IGNORE affected 0 rows for {pid}, "
                            f"using REPLACE to force upsert"
                        )
                        conn.execute("""
                            INSERT OR REPLACE INTO elo_ratings
                            (participant_id, board_type, num_players, rating, last_update,
                             games_played, wins, losses, draws, peak_rating)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            pid, board_type, num_players,
                            new_rating, time.time(),
                            1, win_inc, loss_inc, draw_inc, new_rating
                        ))
                        logger.info(f"[EloService] REPLACE succeeded for {pid}")

            # Record match with optional metadata (e.g., weight profiles used)
            # match_history uses a TEXT primary key column named `id`.
            # Jan 11, 2026: Added harness_type column for multi-harness evaluation tracking
            conn.execute("""
                INSERT INTO match_history
                (id, participant_ids, winner_id, game_length, duration_sec,
                 board_type, num_players, timestamp, elo_before, elo_after,
                 tournament_id, metadata, harness_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                match_id,
                json.dumps([participant_a, participant_b]),
                winner,
                game_length,
                duration_sec,
                board_type,
                num_players,
                timestamp,
                json.dumps(elo_before),
                json.dumps(elo_after),
                tournament_id,
                json.dumps(metadata) if metadata else None,
                harness_type,
            ))

        # Emit events (uses shared helper for both SQLite and Raft paths)
        self._emit_elo_events(
            participant_a, participant_b,
            elo_before, elo_after, elo_changes,
            board_type, num_players, duration_sec,
        )

        # December 30, 2025: Record Elo snapshots for longitudinal tracking
        self._record_elo_snapshot(participant_a, board_type, num_players)
        self._record_elo_snapshot(participant_b, board_type, num_players)

        return MatchResult(
            match_id=match_id,
            participant_ids=[participant_a, participant_b],
            winner_id=winner,
            game_length=game_length,
            duration_sec=duration_sec,
            board_type=board_type,
            num_players=num_players,
            timestamp=timestamp,
            elo_changes=elo_changes
        )

    def record_multi_harness_evaluation(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
        harness_results: dict[str, dict[str, Any]],
    ) -> dict[str, str]:
        """Record multi-harness evaluation results in the Elo system.

        December 30, 2025: Added to support multi-harness gauntlet integration.
        This method registers composite participant IDs for each (model, harness)
        combination and initializes their Elo ratings.

        Args:
            model_path: Path to the model being evaluated
            board_type: Board type (e.g., "hex8", "square8")
            num_players: Number of players (2, 3, or 4)
            harness_results: Dictionary mapping harness names to result dicts:
                {
                    "gumbel_mcts": {"elo": 1450.0, "games_played": 30, "win_rate": 0.65, ...},
                    "minimax": {"elo": 1380.0, "games_played": 30, "win_rate": 0.55, ...},
                }

        Returns:
            Dictionary mapping harness names to composite participant IDs:
                {"gumbel_mcts": "model_v5:gumbel_mcts:abc123", ...}

        Example:
            >>> elo = get_elo_service()
            >>> harness_results = {
            ...     "gumbel_mcts": {"elo": 1450, "games_played": 30, "wins": 20},
            ...     "minimax": {"elo": 1380, "games_played": 30, "wins": 17},
            ... }
            >>> participant_ids = elo.record_multi_harness_evaluation(
            ...     model_path="models/canonical_hex8_2p.pth",
            ...     board_type="hex8",
            ...     num_players=2,
            ...     harness_results=harness_results,
            ... )
        """
        from pathlib import Path as PathLib

        try:
            from app.training.composite_participant import make_composite_participant_id
        except ImportError:
            logger.warning("composite_participant module not available")
            return {}

        participant_ids: dict[str, str] = {}
        model_name = PathLib(model_path).stem

        for harness_name, result_data in harness_results.items():
            # Create composite participant ID
            participant_id = make_composite_participant_id(
                nn_id=model_name,
                ai_type=harness_name,
                config={"players": num_players},
            )

            # Extract rating data
            elo = result_data.get("elo", self.INITIAL_ELO)
            pinned_rating = get_pinned_baseline_rating(participant_id)
            if pinned_rating is not None:
                elo = float(pinned_rating)
            games_played = result_data.get("games_played", 0)
            wins = result_data.get("wins", 0)
            losses = result_data.get("losses", 0)
            draws = result_data.get("draws", 0)

            # Register as composite participant
            self.register_composite_participant(
                nn_id=model_name,
                ai_type=harness_name,
                config={"players": num_players},
                board_type=board_type,
                num_players=num_players,
                nn_model_path=model_path,
            )

            # Update the rating directly with provided values
            with self._transaction() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO elo_ratings
                    (participant_id, board_type, num_players, rating, games_played,
                     wins, losses, draws, peak_rating, last_update)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    participant_id,
                    board_type,
                    num_players,
                    elo,
                    games_played,
                    wins,
                    losses,
                    draws,
                    elo,  # peak_rating = current elo for new entry
                    time.time(),
                ))

            participant_ids[harness_name] = participant_id
            logger.debug(
                f"Registered multi-harness result: {participant_id} with Elo {elo:.0f}"
            )

        logger.info(
            f"Recorded {len(participant_ids)} harness ratings for {model_name} "
            f"({board_type}_{num_players}p)"
        )
        return participant_ids

    def register_composite_participant(
        self,
        nn_id: str | None,
        ai_type: str,
        config: dict[str, Any] | None = None,
        board_type: str = "square8",
        num_players: int = 2,
        nn_model_path: str | None = None,
    ) -> str:
        """Register a composite (NN, Algorithm) participant.

        Creates a composite participant ID and registers it with full metadata.

        Args:
            nn_id: Neural network identifier, or None for non-NN participants
            ai_type: Search algorithm type (e.g., "gumbel_mcts", "mcts")
            config: Algorithm configuration (uses defaults if None)
            board_type: Board type for rating
            num_players: Number of players
            nn_model_path: Path to NN model file

        Returns:
            Composite participant ID
        """
        from app.training.composite_participant import (
            encode_config_hash,
            get_standard_config,
            make_composite_participant_id,
        )

        # Create composite ID
        actual_config = config or get_standard_config(ai_type)
        participant_id = make_composite_participant_id(nn_id, ai_type, actual_config)
        config_hash = encode_config_hash(actual_config, ai_type)

        # Register as participant with extended metadata
        with self._transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO participants
                (participant_id, participant_type, ai_type, use_neural_net, model_path,
                 nn_model_id, nn_model_path, ai_algorithm, algorithm_config, is_composite,
                 created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                participant_id,
                "composite",
                ai_type,
                int(nn_id is not None),
                nn_model_path,
                nn_id,
                nn_model_path,
                ai_type,
                json.dumps(actual_config),
                1,  # is_composite = True
                time.time(),
            ))

        # Initialize rating for this config
        self.get_rating(participant_id, board_type, num_players)

        return participant_id

def get_elo_service(db_path: Path | None = None) -> EloService:
    """Get the singleton EloService instance."""
    global _elo_service_instance
    with _elo_service_lock:
        if _elo_service_instance is None:
            _elo_service_instance = EloService(db_path)
        return _elo_service_instance


def reset_elo_service() -> None:
    """Reset the singleton EloService instance (for testing).

    Dec 29, 2025: Added to fix test class leak issue in tournament tests.
    Ensures test isolation by clearing the cached service instance.
    Thread-local connections will be garbage collected.

    Usage in tests:
        @pytest.fixture(autouse=True)
        def cleanup_elo():
            yield
            reset_elo_service()
    """
    global _elo_service_instance
    with _elo_service_lock:
        if _elo_service_instance is not None:
            # Try to close thread-local connection if accessible
            try:
                if hasattr(_elo_service_instance, '_local'):
                    local = _elo_service_instance._local
                    if hasattr(local, 'connection') and local.connection is not None:
                        local.connection.close()
                        local.connection = None
            except (sqlite3.Error, AttributeError):
                pass  # Ignore close errors, connection will be GC'd
            _elo_service_instance = None


# =============================================================================
# Backwards Compatibility Layer
# =============================================================================
# These functions provide the same interface as scripts/run_model_elo_tournament.py
# to allow smooth migration of orchestrators to use this centralized service.


def init_elo_database(db_path: Path | None = None) -> EloService:
    """Initialize and return the Elo service (backwards compatible)."""
    return get_elo_service(db_path)
# Canonical path - orchestrators should use this
ELO_DB_PATH = DEFAULT_ELO_DB_PATH

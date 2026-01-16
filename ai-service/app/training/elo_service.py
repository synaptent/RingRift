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

import asyncio
import hashlib
import json
import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Path setup
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
    )
except ImportError:
    # Fallback defaults if thresholds not available
    INITIAL_ELO_RATING = 1500.0
    ELO_K_FACTOR = 32
    MIN_GAMES_FOR_ELO = 30
    BASELINE_ELO_RANDOM = 400


def _is_random_participant(participant_id: str) -> bool:
    """Check if participant is a random baseline that should be anchored at 400 Elo.

    Random players serve as the anchor point for the entire rating system.
    Their rating is fixed at BASELINE_ELO_RANDOM (400) to prevent rating inflation.
    """
    pid_lower = participant_id.lower()
    # Composite format: none:random:d1
    if pid_lower.startswith("none:random"):
        return True
    # Legacy formats
    if pid_lower in ("random", "baseline_random", "tier1_random"):
        return True
    # Robust check: "random" in name but not "heuristic" or other algorithms
    if "random" in pid_lower and not any(
        x in pid_lower for x in ("heuristic", "minimax", "mcts", "descent", "neural")
    ):
        return True
    return False

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

# ============================================
# Raft Elo Store Integration (Dec 30, 2025 - P5.2)
# Optional Raft backend for strong consistency across cluster
# ============================================

from enum import Enum


class EloBackendType(str, Enum):
    """Elo storage backend type."""
    RAFT = "raft"  # Raft-replicated store (cluster-wide consistency)
    SQLITE = "sqlite"  # Local SQLite (single-node, faster)


# Raft Elo store availability cache
_raft_elo_store_available: bool | None = None
_raft_elo_store: Any = None
_raft_elo_node_id: str | None = None


def _check_raft_elo_store_available() -> bool:
    """Check if Raft Elo store is available.

    Returns:
        True if Raft is enabled and the P2P orchestrator has a ReplicatedEloStore
    """
    global _raft_elo_store_available, _raft_elo_store, _raft_elo_node_id

    # Return cached result if available
    if _raft_elo_store_available is not None:
        return _raft_elo_store_available

    try:
        # Check if Raft is enabled
        from app.p2p.raft_state import RAFT_ENABLED, PYSYNCOBJ_AVAILABLE
        if not RAFT_ENABLED or not PYSYNCOBJ_AVAILABLE:
            logger.debug("Raft Elo store not available: Raft disabled or pysyncobj missing")
            _raft_elo_store_available = False
            return False

        # Try to get from P2P orchestrator singleton
        try:
            # Import inside try to avoid circular imports
            import sys
            if "scripts.p2p_orchestrator" in sys.modules:
                p2p_module = sys.modules["scripts.p2p_orchestrator"]
                if hasattr(p2p_module, "P2POrchestrator"):
                    orchestrator_cls = p2p_module.P2POrchestrator
                    if hasattr(orchestrator_cls, "_instance") and orchestrator_cls._instance:
                        orchestrator = orchestrator_cls._instance
                        if hasattr(orchestrator, "replicated_elo_store"):
                            elo_store = orchestrator.replicated_elo_store
                            if elo_store and hasattr(elo_store, "is_ready") and elo_store.is_ready:
                                _raft_elo_store = elo_store
                                _raft_elo_node_id = getattr(orchestrator, "node_id", None)
                                _raft_elo_store_available = True
                                logger.info(f"Raft Elo store available via P2P orchestrator")
                                return True
        except Exception as e:
            logger.debug(f"Could not get Raft Elo store from orchestrator: {e}")

        _raft_elo_store_available = False
        return False

    except ImportError as e:
        logger.debug(f"Raft Elo store not available: {e}")
        _raft_elo_store_available = False
        return False


def reset_raft_elo_store_cache() -> None:
    """Reset the Raft Elo store availability cache.

    Call this when the P2P orchestrator state changes.
    """
    global _raft_elo_store_available, _raft_elo_store, _raft_elo_node_id
    _raft_elo_store_available = None
    _raft_elo_store = None
    _raft_elo_node_id = None


def get_raft_elo_store() -> Any:
    """Get the cached Raft Elo store instance.

    Returns:
        ReplicatedEloStore instance or None if not available
    """
    if _check_raft_elo_store_available():
        return _raft_elo_store
    return None

# Event emission for ELO updates
# Jan 5, 2026: Fixed import path - emit_elo_updated is in data_events, not event_router
try:
    from app.distributed.data_events import emit_elo_updated
    HAS_ELO_EVENTS = True
except ImportError:
    HAS_ELO_EVENTS = False
    emit_elo_updated = None

# Event emission for Elo velocity/significant changes (December 2025 - training pipeline fix)
try:
    from app.distributed.data_events import emit_data_event, DataEventType
    HAS_ELO_VELOCITY_EVENTS = True
except ImportError:
    HAS_ELO_VELOCITY_EVENTS = False
    emit_data_event = None  # type: ignore
    DataEventType = None  # type: ignore

# Elo event thresholds (December 2025)
# ELO_SIGNIFICANT_CHANGE: Threshold for single-match Elo change to be "significant"
ELO_SIGNIFICANT_CHANGE_THRESHOLD = 25.0  # More than 25 Elo change in a single match

# ELO_VELOCITY_CHANGE: Minimum Elo change per hour to trigger velocity event
ELO_VELOCITY_THRESHOLD_PER_HOUR = 50.0  # 50+ Elo/hour indicates significant improvement rate

# Composite event emission (Sprint 5)
try:
    from app.training.event_integration import publish_composite_elo_updated_sync
    from app.training.composite_participant import (
        is_composite_id,
        parse_composite_participant_id,
    )
    HAS_COMPOSITE_EVENTS = True
except ImportError:
    HAS_COMPOSITE_EVENTS = False
    publish_composite_elo_updated_sync = None
    is_composite_id = None
    parse_composite_participant_id = None


@dataclass
class EloRating:
    """Elo rating with metadata.

    Note: Default rating uses INITIAL_ELO_RATING from app.config.thresholds (1500.0)
    """
    participant_id: str
    rating: float = 1500.0  # See app.config.thresholds.INITIAL_ELO_RATING
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    last_update: float = 0.0
    confidence: float = 0.0  # Based on games played

    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.5
        return (self.wins + 0.5 * self.draws) / self.games_played


@dataclass
class MatchResult:
    """Result of a single match."""
    match_id: str
    participant_ids: list[str]
    winner_id: str | None  # None for draw
    game_length: int
    duration_sec: float
    board_type: str
    num_players: int
    timestamp: str
    elo_changes: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingFeedback:
    """Feedback signals for training parameter adaptation."""
    board_type: str
    num_players: int

    # Elo trend analysis
    best_elo: float = 1500.0
    recent_elo_delta: float = 0.0  # Change over last N iterations
    elo_stagnating: bool = False  # True if no improvement in 5+ iterations
    elo_declining: bool = False  # True if negative trend

    # Win rate analysis
    best_win_rate: float = 0.5
    win_rate_vs_baseline: float = 0.5

    # Recommended adjustments
    epochs_multiplier: float = 1.0  # Multiply default epochs by this
    lr_multiplier: float = 1.0  # Multiply default LR by this
    exploration_boost: float = 0.0  # Add to temperature/noise

    # Curriculum recommendation
    recommended_curriculum_stage: int = 0


@dataclass
class LeaderboardEntry:
    """Entry in the Elo leaderboard."""
    rank: int
    participant_id: str
    name: str
    ai_type: str
    rating: float
    games_played: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    last_active: str


class EloService:
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
        if self._use_raft and _check_raft_elo_store_available():
            self._backend = EloBackendType.RAFT
            logger.info("EloService using Raft backend for cluster-wide consistency")
        else:
            logger.debug("EloService using SQLite backend")

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
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
            if "is_deployable" not in participant_columns:
                conn.execute("ALTER TABLE participants ADD COLUMN is_deployable INTEGER DEFAULT 0")
                # Mark existing canonical models as deployable
                conn.execute("""
                    UPDATE participants SET is_deployable = 1
                    WHERE model_path LIKE '%canonical_%' OR model_path LIKE '%ringrift_best_%'
                """)
                logger.info("Migrated participants: added is_deployable column")

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

        Note: Random players are anchored at BASELINE_ELO_RANDOM (400) to serve
        as the fixed reference point and prevent rating inflation.
        """
        # Anchor random participants at fixed Elo to prevent rating drift
        if _is_random_participant(participant_id):
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
                    rating=float(BASELINE_ELO_RANDOM),  # ANCHORED
                    games_played=row["games_played"],
                    wins=row["wins"],
                    losses=row["losses"],
                    draws=row["draws"],
                    last_update=row["last_update"] or 0.0,
                    confidence=1.0  # Random is always reliable as anchor
                )
            # Create entry for random participant at anchored rating
            with self._transaction() as txn_conn:
                txn_conn.execute("""
                    INSERT OR IGNORE INTO elo_ratings
                    (participant_id, board_type, num_players, rating, last_update)
                    VALUES (?, ?, ?, ?, ?)
                """, (participant_id, board_type, num_players, float(BASELINE_ELO_RANDOM), time.time()))
            return EloRating(
                participant_id=participant_id,
                rating=float(BASELINE_ELO_RANDOM),  # ANCHORED
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

    def _get_adaptive_k_factor(
        self,
        games_a: int,
        games_b: int,
        base_k: float,
    ) -> float:
        """Calculate adaptive K-factor based on games played.

        Adaptive K-factor improves rating convergence:
        - New models (< 30 games): K = 1.5x base (faster convergence)
        - Developing models (30-100 games): K = 1.25x base
        - Established models (100-300 games): K = 1.0x base
        - Mature models (300+ games): K = 0.75x base (stability)

        The K-factor is determined by the LESS confident participant,
        so matches involving new models are more impactful.

        Args:
            games_a: Games played by participant A
            games_b: Games played by participant B
            base_k: Base K-factor (after multiplayer scaling)

        Returns:
            Adaptive K-factor scaled by confidence

        December 2025: Added to improve rating accuracy for new models
        while maintaining stability for established models.
        """
        # Use the minimum games (less confident participant drives K)
        min_games = min(games_a, games_b)

        # Tiered K-factor multipliers
        if min_games < 30:
            # Provisional: High K for rapid convergence
            k_multiplier = 1.5
        elif min_games < 100:
            # Developing: Moderately high K
            k_multiplier = 1.25
        elif min_games < 300:
            # Established: Standard K
            k_multiplier = 1.0
        else:
            # Mature: Lower K for stability
            k_multiplier = 0.75

        return base_k * k_multiplier

    def _emit_elo_events(
        self,
        participant_a: str,
        participant_b: str,
        elo_before: dict[str, float],
        elo_after: dict[str, float],
        elo_changes: dict[str, float],
        board_type: str,
        num_players: int,
        duration_sec: float,
    ) -> None:
        """Emit Elo-related events for match recording.

        December 30, 2025: Extracted from record_match() to enable reuse
        in both SQLite and Raft code paths.

        This method emits:
        - ELO_UPDATED events for both participants
        - Composite ELO events for composite participants
        - ELO_SIGNIFICANT_CHANGE events for large rating changes
        - ELO_VELOCITY_CHANGED events for rapid improvement/decline

        Args:
            participant_a: First participant ID
            participant_b: Second participant ID
            elo_before: Dict mapping participant IDs to pre-match ratings
            elo_after: Dict mapping participant IDs to post-match ratings
            elo_changes: Dict mapping participant IDs to rating changes
            board_type: Board type (e.g., "hex8")
            num_players: Number of players (2, 3, or 4)
            duration_sec: Match duration in seconds (for velocity calculation)
        """
        config_key = f"{board_type}_{num_players}p"

        # Emit ELO_UPDATED events for both participants
        if HAS_ELO_EVENTS and emit_elo_updated is not None:
            try:
                # Try to get running event loop
                try:
                    asyncio.get_running_loop()
                    # Schedule coroutines in the running loop
                    for pid, old_elo, new_elo in [
                        (participant_a, elo_before[participant_a], elo_after[participant_a]),
                        (participant_b, elo_before[participant_b], elo_after[participant_b]),
                    ]:
                        asyncio.ensure_future(emit_elo_updated(
                            config=config_key,
                            model_id=pid,
                            new_elo=new_elo,
                            old_elo=old_elo,
                            games_played=1,
                            source="elo_service",
                        ))
                except RuntimeError:
                    # No running loop - create one for sync context
                    pass  # Skip in pure sync context to avoid blocking
            except (RuntimeError, AttributeError, TypeError, ValueError):
                pass  # Don't let event emission break match recording

        # Emit composite ELO events for composite participants (Sprint 5)
        if HAS_COMPOSITE_EVENTS and publish_composite_elo_updated_sync is not None:
            for pid, old_elo, new_elo in [
                (participant_a, elo_before[participant_a], elo_after[participant_a]),
                (participant_b, elo_before[participant_b], elo_after[participant_b]),
            ]:
                if is_composite_id and is_composite_id(pid):
                    try:
                        parsed = parse_composite_participant_id(pid)
                        if parsed:
                            nn_id, ai_type, config_hash = parsed
                            publish_composite_elo_updated_sync(
                                nn_id=nn_id,
                                ai_type=ai_type,
                                config_hash=config_hash,
                                participant_id=pid,
                                old_elo=old_elo,
                                new_elo=new_elo,
                                games_played=1,
                                board_type=board_type,
                                num_players=num_players,
                            )
                    except (RuntimeError, AttributeError, TypeError, ValueError, KeyError):
                        pass  # Don't let event emission break match recording

        # Emit ELO_SIGNIFICANT_CHANGE and ELO_VELOCITY_CHANGED events
        if HAS_ELO_VELOCITY_EVENTS and emit_data_event is not None:
            try:
                # Check for significant single-match Elo changes
                for pid, old_elo, new_elo in [
                    (participant_a, elo_before[participant_a], elo_after[participant_a]),
                    (participant_b, elo_before[participant_b], elo_after[participant_b]),
                ]:
                    elo_delta = new_elo - old_elo
                    if abs(elo_delta) > ELO_SIGNIFICANT_CHANGE_THRESHOLD:
                        try:
                            asyncio.get_running_loop()
                            asyncio.ensure_future(emit_data_event(
                                event_type=DataEventType.ELO_SIGNIFICANT_CHANGE,
                                payload={
                                    "config_key": config_key,
                                    "board_type": board_type,
                                    "num_players": num_players,
                                    "participant_id": pid,
                                    "old_elo": old_elo,
                                    "new_elo": new_elo,
                                    "delta": elo_delta,
                                },
                            ))
                        except RuntimeError:
                            pass  # No event loop - skip in sync context

                # Calculate velocity (Elo per hour) if we have duration
                if duration_sec and duration_sec > 0:
                    hours = duration_sec / 3600.0
                    for pid, old_elo, new_elo in [
                        (participant_a, elo_before[participant_a], elo_after[participant_a]),
                        (participant_b, elo_before[participant_b], elo_after[participant_b]),
                    ]:
                        elo_delta = new_elo - old_elo
                        elo_per_hour = elo_delta / hours if hours > 0 else 0.0

                        if abs(elo_per_hour) > ELO_VELOCITY_THRESHOLD_PER_HOUR:
                            try:
                                asyncio.get_running_loop()
                                asyncio.ensure_future(emit_data_event(
                                    event_type=DataEventType.ELO_VELOCITY_CHANGED,
                                    payload={
                                        "config_key": config_key,
                                        "board_type": board_type,
                                        "num_players": num_players,
                                        "participant_id": pid,
                                        "velocity": elo_per_hour,
                                        "trend": "improving" if elo_per_hour > 0 else "declining",
                                    },
                                ))
                            except RuntimeError:
                                pass  # No event loop - skip in sync context

            except (RuntimeError, AttributeError, TypeError, ValueError):
                pass  # Don't let event emission break match recording

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
                # Note: id is INTEGER PRIMARY KEY AUTOINCREMENT, so we use game_id for UUID
                # Jan 11, 2026: Added harness_type column for multi-harness evaluation tracking
                with self._transaction() as conn:
                    conn.execute("""
                        INSERT INTO match_history
                        (game_id, participant_ids, winner_id, game_length, duration_sec,
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

        # Calculate expected scores
        exp_a = 1.0 / (1.0 + math.pow(10, (rating_b.rating - rating_a.rating) / 400))
        exp_b = 1.0 - exp_a

        # Actual scores
        if winner == participant_a:
            score_a, score_b = 1.0, 0.0
        elif winner == participant_b:
            score_a, score_b = 0.0, 1.0
        else:
            score_a, score_b = 0.5, 0.5

        # Scale K-factor for multiplayer games
        # In N-player games, each pairwise matchup is 1/(N-1) of the rating info
        # This ensures consistent rating change magnitude across player counts
        base_k = self.K_FACTOR / (num_players - 1) if num_players > 2 else self.K_FACTOR

        # Apply adaptive K-factor based on confidence (December 2025)
        # New models (low games) get higher K for faster convergence
        # Established models (high games) get lower K for stability
        k = self._get_adaptive_k_factor(
            games_a=rating_a.games_played,
            games_b=rating_b.games_played,
            base_k=base_k,
        )

        # Calculate Elo changes
        change_a = k * (score_a - exp_a)
        change_b = k * (score_b - exp_b)

        new_rating_a = rating_a.rating + change_a
        new_rating_b = rating_b.rating + change_b

        # ANCHOR: Force random players back to fixed Elo (prevents rating inflation)
        # Random serves as the anchor point for the entire rating system
        if _is_random_participant(participant_a):
            new_rating_a = float(BASELINE_ELO_RANDOM)
            change_a = 0.0  # No effective change for anchored player
        if _is_random_participant(participant_b):
            new_rating_b = float(BASELINE_ELO_RANDOM)
            change_b = 0.0  # No effective change for anchored player

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
            # Note: id is INTEGER PRIMARY KEY AUTOINCREMENT, so we use game_id for UUID
            # Jan 11, 2026: Added harness_type column for multi-harness evaluation tracking
            conn.execute("""
                INSERT INTO match_history
                (game_id, participant_ids, winner_id, game_length, duration_sec,
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

    def get_leaderboard(
        self,
        board_type: str,
        num_players: int,
        limit: int = 50,
        min_games: int = 0
    ) -> list[LeaderboardEntry]:
        """Get the Elo leaderboard for a configuration."""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT
                e.participant_id,
                p.participant_id AS name,
                p.ai_type,
                e.rating,
                e.games_played,
                e.wins,
                e.losses,
                e.draws,
                e.last_update
            FROM elo_ratings e
            JOIN participants p ON e.participant_id = p.participant_id
            WHERE e.board_type = ? AND e.num_players = ? AND e.games_played >= ?
            ORDER BY e.rating DESC
            LIMIT ?
        """, (board_type, num_players, min_games, limit))

        entries = []
        for rank, row in enumerate(cursor.fetchall(), 1):
            games = row["games_played"]
            win_rate = (row["wins"] + 0.5 * row["draws"]) / games if games > 0 else 0.5
            entries.append(LeaderboardEntry(
                rank=rank,
                participant_id=row["participant_id"],
                name=row["name"],
                ai_type=row["ai_type"],
                rating=row["rating"],
                games_played=games,
                wins=row["wins"],
                losses=row["losses"],
                draws=row["draws"],
                win_rate=win_rate,
                last_active=datetime.fromtimestamp(row["last_update"] or 0).isoformat()
            ))
        return entries

    def log_leaderboard(
        self,
        board_type: str,
        num_players: int,
        top_n: int = 5,
    ) -> list[LeaderboardEntry]:
        """Log the Elo leaderboard for a configuration.

        January 5, 2026 (Session 17.34): Added for visibility into which models
        are improving. Called after evaluations to track progress.

        Args:
            board_type: Board type (e.g., 'hex8', 'square8')
            num_players: Number of players (2, 3, or 4)
            top_n: Number of top models to log (default: 5)

        Returns:
            List of top leaderboard entries
        """
        config_key = f"{board_type}_{num_players}p"
        entries = self.get_leaderboard(board_type, num_players, limit=top_n, min_games=1)

        if not entries:
            logger.info(f"[Elo] Leaderboard for {config_key}: (no entries with games)")
            return entries

        logger.info(f"[Elo] Leaderboard for {config_key}:")
        for entry in entries:
            model_name = entry.name[:30] if len(entry.name) > 30 else entry.name
            logger.info(
                f"  #{entry.rank}: {model_name} - Elo {entry.rating:.0f} "
                f"({entry.wins}W/{entry.losses}L, {entry.win_rate:.0%})"
            )
        return entries

    def get_training_feedback(
        self,
        board_type: str,
        num_players: int,
        lookback_iterations: int = 5
    ) -> TrainingFeedback:
        """Get feedback signals for training parameter adaptation."""
        conn = self._get_connection()

        # Get best model rating
        cursor = conn.execute("""
            SELECT MAX(rating) as best_rating, MAX(peak_rating) as peak
            FROM elo_ratings
            WHERE board_type = ? AND num_players = ?
        """, (board_type, num_players))
        row = cursor.fetchone()
        best_elo = row["best_rating"] or self.INITIAL_ELO

        # Get recent Elo history for trend analysis
        cursor = conn.execute("""
            SELECT rating, iteration FROM elo_history
            WHERE board_type = ? AND num_players = ?
            ORDER BY iteration DESC
            LIMIT ?
        """, (board_type, num_players, lookback_iterations))
        history = cursor.fetchall()

        # Calculate trend
        recent_elo_delta = 0.0
        elo_stagnating = False
        elo_declining = False

        if len(history) >= 2:
            recent_elo_delta = history[0]["rating"] - history[-1]["rating"]
            if abs(recent_elo_delta) < 10:  # Less than 10 Elo change
                elo_stagnating = True
            if recent_elo_delta < -20:  # More than 20 Elo drop
                elo_declining = True

        # Calculate recommended adjustments
        epochs_multiplier = 1.0
        lr_multiplier = 1.0
        exploration_boost = 0.0
        curriculum_stage = 0

        if elo_stagnating:
            # Try longer training with lower LR
            epochs_multiplier = 1.5
            lr_multiplier = 0.8
            exploration_boost = 0.1  # Add some exploration

        if elo_declining:
            # Something went wrong - increase epochs significantly
            epochs_multiplier = 2.0
            lr_multiplier = 0.5
            exploration_boost = 0.2

        # Curriculum progression based on Elo
        if best_elo > 1600:
            curriculum_stage = 1
        if best_elo > 1700:
            curriculum_stage = 2
        if best_elo > 1800:
            curriculum_stage = 3

        feedback = TrainingFeedback(
            board_type=board_type,
            num_players=num_players,
            best_elo=best_elo,
            recent_elo_delta=recent_elo_delta,
            elo_stagnating=elo_stagnating,
            elo_declining=elo_declining,
            epochs_multiplier=epochs_multiplier,
            lr_multiplier=lr_multiplier,
            exploration_boost=exploration_boost,
            recommended_curriculum_stage=curriculum_stage
        )

        # Notify callbacks
        for callback in self._feedback_callbacks:
            with suppress(Exception):
                callback(feedback)

        return feedback

    def record_training_feedback(
        self,
        board_type: str,
        num_players: int,
        iteration: int,
        best_elo: float,
        elo_delta: float,
        epochs_multiplier: float = 1.0,
        lr_multiplier: float = 1.0,
        curriculum_stage: int = 0,
    ) -> bool:
        """Record training feedback signal to database.

        January 2026: Added to fix empty training_feedback table.
        Called after TRAINING_COMPLETED to track Elo progression and
        enable curriculum velocity tracking.

        Args:
            board_type: Board type (e.g., "hex8", "square8")
            num_players: Number of players
            iteration: Training iteration number
            best_elo: Current best Elo for this config
            elo_delta: Elo change since last training
            epochs_multiplier: Training epochs multiplier used
            lr_multiplier: Learning rate multiplier used
            curriculum_stage: Current curriculum stage (0-3)

        Returns:
            True if recording succeeded, False otherwise
        """
        try:
            with self._transaction() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO training_feedback
                    (board_type, num_players, iteration, best_elo, elo_delta,
                     epochs_multiplier, lr_multiplier, curriculum_stage, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    board_type, num_players, iteration, best_elo, elo_delta,
                    epochs_multiplier, lr_multiplier, curriculum_stage,
                    time.time()
                ))
            logger.debug(
                f"Recorded training feedback: {board_type}_{num_players}p "
                f"iter={iteration} elo={best_elo:.0f} delta={elo_delta:+.0f}"
            )
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to record training feedback: {e}")
            return False

    def record_iteration_elo(
        self,
        participant_id: str,
        board_type: str,
        num_players: int,
        iteration: int
    ) -> None:
        """Record Elo snapshot for trend analysis."""
        rating = self.get_rating(participant_id, board_type, num_players)
        with self._transaction() as conn:
            conn.execute("""
                INSERT INTO elo_history
                (participant_id, board_type, num_players, rating, timestamp, iteration)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                participant_id, board_type, num_players,
                rating.rating, time.time(), iteration
            ))

    def _record_elo_snapshot(
        self,
        participant_id: str,
        board_type: str,
        num_players: int,
        min_interval_seconds: float = 300.0,
    ) -> None:
        """Record Elo snapshot for longitudinal tracking.

        December 30, 2025: Added to fix empty elo_history table.
        Unlike record_iteration_elo(), this doesn't require an iteration number
        and is called automatically after matches. Snapshots are rate-limited
        to avoid table bloat.

        Args:
            participant_id: Participant ID
            board_type: Board type
            num_players: Number of players
            min_interval_seconds: Minimum seconds between snapshots (default 5 min)
        """
        # Skip baseline participants (random, heuristic) - their Elo is fixed
        if _is_random_participant(participant_id):
            return
        pid_lower = participant_id.lower()
        if 'heuristic' in pid_lower or 'baseline' in pid_lower:
            return

        current_rating = self.get_rating(participant_id, board_type, num_players)
        if not current_rating or current_rating.games_played == 0:
            return

        conn = self._get_connection()

        # Check last snapshot time (rate limiting to avoid table bloat)
        cursor = conn.execute("""
            SELECT MAX(timestamp) FROM elo_history
            WHERE participant_id = ? AND board_type = ? AND num_players = ?
        """, (participant_id, board_type, num_players))
        row = cursor.fetchone()
        last_snapshot = row[0] if row and row[0] else 0.0

        if time.time() - last_snapshot < min_interval_seconds:
            return  # Too recent, skip

        # Record snapshot (iteration=NULL for automatic snapshots)
        with self._transaction() as txn_conn:
            txn_conn.execute("""
                INSERT INTO elo_history
                (participant_id, board_type, num_players, rating, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                participant_id, board_type, num_players,
                current_rating.rating, time.time()
            ))

    def register_feedback_callback(self, callback: Callable[[TrainingFeedback], None]) -> None:
        """Register a callback to be notified of training feedback."""
        self._feedback_callbacks.append(callback)

    def get_win_rate_vs_baseline(
        self,
        model_id: str,
        baseline_id: str,
        board_type: str,
        num_players: int
    ) -> tuple[float, int]:
        """Get win rate of model vs specific baseline."""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT winner_id FROM match_history
            WHERE board_type = ? AND num_players = ?
            AND participant_ids LIKE ? AND participant_ids LIKE ?
        """, (board_type, num_players, f'%{model_id}%', f'%{baseline_id}%'))

        wins = 0
        total = 0
        for row in cursor:
            total += 1
            if row["winner_id"] == model_id:
                wins += 1
            elif row["winner_id"] is None:
                wins += 0.5  # Draw counts as half

        win_rate = wins / total if total > 0 else 0.5
        return win_rate, total

    # =========================================================================
    # Composite Participant Methods (Sprint 1)
    # =========================================================================

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

    def get_nn_performance_summary(
        self,
        nn_model_id: str,
        board_type: str,
        num_players: int,
    ) -> dict[str, Any] | None:
        """Get aggregated performance summary for an NN across algorithms.

        Args:
            nn_model_id: Neural network identifier
            board_type: Board type
            num_players: Number of players

        Returns:
            Dict with best_algorithm, best_elo, avg_elo, algorithms_tested
            or None if no data
        """
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT best_algorithm, best_elo, avg_elo, algorithms_tested, last_updated
            FROM nn_performance_summary
            WHERE nn_model_id = ? AND board_type = ? AND num_players = ?
        """, (nn_model_id, board_type, num_players))

        row = cursor.fetchone()
        if not row:
            return None

        return {
            "nn_model_id": nn_model_id,
            "board_type": board_type,
            "num_players": num_players,
            "best_algorithm": row["best_algorithm"],
            "best_elo": row["best_elo"],
            "avg_elo": row["avg_elo"],
            "algorithms_tested": row["algorithms_tested"],
            "last_updated": row["last_updated"],
        }

    def get_algorithm_baseline(
        self,
        ai_algorithm: str,
        board_type: str,
        num_players: int,
    ) -> dict[str, Any] | None:
        """Get baseline rating for an algorithm.

        Args:
            ai_algorithm: Algorithm type
            board_type: Board type
            num_players: Number of players

        Returns:
            Dict with baseline_elo, games_played or None if not found
        """
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT baseline_elo, games_played, last_updated
            FROM algorithm_baselines
            WHERE ai_algorithm = ? AND board_type = ? AND num_players = ?
        """, (ai_algorithm, board_type, num_players))

        row = cursor.fetchone()
        if not row:
            return None

        return {
            "ai_algorithm": ai_algorithm,
            "board_type": board_type,
            "num_players": num_players,
            "baseline_elo": row["baseline_elo"],
            "games_played": row["games_played"],
            "last_updated": row["last_updated"],
        }

    def update_algorithm_baseline(
        self,
        ai_algorithm: str,
        board_type: str,
        num_players: int,
        baseline_elo: float,
        games_played: int = 0,
    ) -> None:
        """Update baseline rating for an algorithm.

        Args:
            ai_algorithm: Algorithm type
            board_type: Board type
            num_players: Number of players
            baseline_elo: New baseline Elo rating
            games_played: Number of games played
        """
        with self._transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO algorithm_baselines
                (ai_algorithm, board_type, num_players, baseline_elo, games_played, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (ai_algorithm, board_type, num_players, baseline_elo, games_played, time.time()))

    def get_composite_leaderboard(
        self,
        board_type: str,
        num_players: int,
        ai_algorithm: str | None = None,
        nn_model_id: str | None = None,
        limit: int = 50,
        min_games: int = 0,
    ) -> list[dict[str, Any]]:
        """Get leaderboard filtered by algorithm or NN.

        Args:
            board_type: Board type
            num_players: Number of players
            ai_algorithm: Filter by algorithm (optional)
            nn_model_id: Filter by NN model (optional)
            limit: Maximum entries
            min_games: Minimum games required

        Returns:
            List of leaderboard entries
        """
        conn = self._get_connection()

        query = """
            SELECT
                e.participant_id,
                p.nn_model_id,
                p.ai_algorithm,
                e.rating,
                e.games_played,
                e.wins,
                e.losses,
                e.draws,
                e.last_update
            FROM elo_ratings e
            JOIN participants p ON e.participant_id = p.participant_id
            WHERE e.board_type = ? AND e.num_players = ? AND e.games_played >= ?
        """
        params: list[Any] = [board_type, num_players, min_games]

        if ai_algorithm:
            query += " AND p.ai_algorithm = ?"
            params.append(ai_algorithm)

        if nn_model_id:
            query += " AND p.nn_model_id = ?"
            params.append(nn_model_id)

        query += " ORDER BY e.rating DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(query, params)

        entries = []
        for rank, row in enumerate(cursor.fetchall(), 1):
            games = row["games_played"]
            win_rate = (row["wins"] + 0.5 * row["draws"]) / games if games > 0 else 0.5
            entries.append({
                "rank": rank,
                "participant_id": row["participant_id"],
                "nn_model_id": row["nn_model_id"],
                "ai_algorithm": row["ai_algorithm"],
                "rating": row["rating"],
                "games_played": games,
                "wins": row["wins"],
                "losses": row["losses"],
                "draws": row["draws"],
                "win_rate": win_rate,
                "last_update": row["last_update"],
            })

        return entries

    def get_algorithm_rankings(
        self,
        board_type: str,
        num_players: int,
        min_games: int = 10,
    ) -> list[dict[str, Any]]:
        """Get algorithm rankings based on average performance.

        Aggregates ratings across all NNs for each algorithm.

        Args:
            board_type: Board type
            num_players: Number of players
            min_games: Minimum games for inclusion

        Returns:
            List of algorithm rankings with avg_elo, best_elo, nn_count
        """
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT
                p.ai_algorithm,
                AVG(e.rating) as avg_elo,
                MAX(e.rating) as best_elo,
                MIN(e.rating) as worst_elo,
                COUNT(DISTINCT p.nn_model_id) as nn_count,
                SUM(e.games_played) as total_games
            FROM elo_ratings e
            JOIN participants p ON e.participant_id = p.participant_id
            WHERE e.board_type = ? AND e.num_players = ?
                AND e.games_played >= ?
                AND p.ai_algorithm IS NOT NULL
            GROUP BY p.ai_algorithm
            ORDER BY avg_elo DESC
        """, (board_type, num_players, min_games))

        rankings = []
        for rank, row in enumerate(cursor.fetchall(), 1):
            rankings.append({
                "rank": rank,
                "ai_algorithm": row["ai_algorithm"],
                "avg_elo": row["avg_elo"],
                "best_elo": row["best_elo"],
                "worst_elo": row["worst_elo"],
                "nn_count": row["nn_count"],
                "total_games": row["total_games"],
                "elo_spread": row["best_elo"] - row["worst_elo"],
            })

        return rankings

    def get_nn_rankings(
        self,
        board_type: str,
        num_players: int,
        min_games: int = 10,
    ) -> list[dict[str, Any]]:
        """Get NN rankings based on best performance across algorithms.

        Args:
            board_type: Board type
            num_players: Number of players
            min_games: Minimum games for inclusion

        Returns:
            List of NN rankings with best_elo, best_algorithm, algorithm_count
        """
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT
                p.nn_model_id,
                MAX(e.rating) as best_elo,
                AVG(e.rating) as avg_elo,
                COUNT(DISTINCT p.ai_algorithm) as algorithm_count,
                SUM(e.games_played) as total_games
            FROM elo_ratings e
            JOIN participants p ON e.participant_id = p.participant_id
            WHERE e.board_type = ? AND e.num_players = ?
                AND e.games_played >= ?
                AND p.nn_model_id IS NOT NULL
                AND p.nn_model_id != 'none'
            GROUP BY p.nn_model_id
            ORDER BY best_elo DESC
        """, (board_type, num_players, min_games))

        rankings = []
        for rank, row in enumerate(cursor.fetchall(), 1):
            # Get best algorithm for this NN
            best_algo_cursor = conn.execute("""
                SELECT p.ai_algorithm
                FROM elo_ratings e
                JOIN participants p ON e.participant_id = p.participant_id
                WHERE p.nn_model_id = ? AND e.board_type = ? AND e.num_players = ?
                ORDER BY e.rating DESC
                LIMIT 1
            """, (row["nn_model_id"], board_type, num_players))
            best_algo_row = best_algo_cursor.fetchone()
            best_algorithm = best_algo_row["ai_algorithm"] if best_algo_row else None

            rankings.append({
                "rank": rank,
                "nn_model_id": row["nn_model_id"],
                "best_elo": row["best_elo"],
                "avg_elo": row["avg_elo"],
                "best_algorithm": best_algorithm,
                "algorithm_count": row["algorithm_count"],
                "total_games": row["total_games"],
            })

        return rankings

    def get_unevaluated_models(
        self,
        models_directory: Path | str | None = None,
        include_subdirs: bool = True,
    ) -> list[dict[str, Any]]:
        """Find model files that don't have Elo ratings in the database.

        January 3, 2026: Added for ModelDiscoveryDaemon to find models needing evaluation.

        Args:
            models_directory: Directory to scan for .pth files.
                Defaults to 'models/' directory relative to ai-service.
            include_subdirs: Whether to scan subdirectories recursively.

        Returns:
            List of dicts with model info:
                - model_path: str - Full path to model file
                - board_type: str - Extracted board type (hex8, square8, etc.)
                - num_players: int - Extracted player count
                - file_size_mb: float - File size in MB
                - modified_at: float - Last modified timestamp
        """
        if models_directory is None:
            # Default to models/ relative to this file's location
            from app.utils.paths import MODELS_DIR
            models_directory = MODELS_DIR

        models_dir = Path(models_directory)
        if not models_dir.exists():
            logger.warning(f"Models directory not found: {models_dir}")
            return []

        # Find all .pth files
        if include_subdirs:
            model_files = list(models_dir.rglob("*.pth"))
        else:
            model_files = list(models_dir.glob("*.pth"))

        if not model_files:
            logger.debug(f"No .pth files found in {models_dir}")
            return []

        # Known board types for parsing
        BOARD_TYPES = {"hex8", "square8", "square19", "hexagonal"}

        unevaluated = []
        for model_path in model_files:
            # Parse filename to extract board_type and num_players
            # Pattern: canonical_hex8_2p.pth or hex8_2p_v5heavy.pth
            model_name = model_path.stem.lower()
            board_type = None
            num_players = None

            # Extract board type
            for bt in BOARD_TYPES:
                if bt in model_name:
                    board_type = bt
                    break

            # Extract player count (e.g., "2p", "3p", "4p")
            import re
            player_match = re.search(r"(\d)p", model_name)
            if player_match:
                num_players = int(player_match.group(1))

            if not board_type or not num_players:
                logger.debug(
                    f"Could not parse config from filename: {model_path.name}"
                )
                continue

            # Check if this model has an Elo rating
            # Try to find by model path first
            model_path_str = str(model_path)
            conn = self._get_connection()

            # Check if participant exists with this model_path
            cursor = conn.execute("""
                SELECT p.participant_id, e.rating, e.games_played
                FROM participants p
                LEFT JOIN elo_ratings e ON p.participant_id = e.participant_id
                    AND e.board_type = ? AND e.num_players = ?
                WHERE p.model_path = ?
            """, (board_type, num_players, model_path_str))

            row = cursor.fetchone()

            # Model needs evaluation if:
            # 1. Not in participants table at all, OR
            # 2. In participants but no Elo rating, OR
            # 3. In participants but games_played = 0
            needs_evaluation = (
                row is None or
                row["rating"] is None or
                (row["games_played"] is not None and row["games_played"] == 0)
            )

            if needs_evaluation:
                stat = model_path.stat()
                unevaluated.append({
                    "model_path": model_path_str,
                    "board_type": board_type,
                    "num_players": num_players,
                    "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified_at": stat.st_mtime,
                })

        # Sort by modified time (newest first) so newer models get evaluated first
        unevaluated.sort(key=lambda x: x["modified_at"], reverse=True)

        logger.info(
            f"Found {len(unevaluated)} unevaluated models out of {len(model_files)} total"
        )
        return unevaluated


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


def register_models(db: EloService, models: list[dict[str, Any]]) -> None:
    """Register multiple models in the Elo database.

    Args:
        db: The EloService instance
        models: List of model dicts with keys: model_id, model_path, board_type, num_players
    """
    for model in models:
        model_id = model.get("model_id") or model.get("id", "")
        model_path = model.get("model_path", "")
        board_type = model.get("board_type", "square8")
        num_players = model.get("num_players", 2)

        if not model_id:
            continue

        db.register_model(
            model_id=model_id,
            board_type=board_type,
            num_players=num_players,
            model_path=model_path
        )


def update_elo_after_match(
    db: EloService,
    model_a_id: str,
    model_b_id: str,
    winner: str | None,  # model_a_id, model_b_id, "draw", or None
    board_type: str = "square8",
    num_players: int = 2,
    game_length: int = 0,
    duration_sec: float = 0.0,
    # December 30, 2025: Multi-harness support
    # January 13, 2026: Default to gumbel_mcts for consistency with record_match()
    harness_type: str = "gumbel_mcts",
    is_multi_harness: bool = False,
) -> dict[str, Any]:
    """Update Elo ratings after a match (backwards compatible).

    Args:
        db: The EloService instance
        model_a_id: First model ID
        model_b_id: Second model ID
        winner: Winner ID, "draw", or None for draw
        board_type: Board type string
        num_players: Number of players
        game_length: Number of moves in game
        duration_sec: Game duration in seconds
        harness_type: AI harness type (default: "gumbel_mcts")
        is_multi_harness: True if part of multi-harness evaluation

    Returns:
        Dict with rating changes: {"model_a": new_rating, "model_b": new_rating, "changes": {...}}
    """
    # Normalize winner for draw cases
    winner_id = None
    if winner == "draw" or winner is None:
        winner_id = None
    elif winner == model_a_id or winner == "model_a":
        winner_id = model_a_id
    elif winner == model_b_id or winner == "model_b":
        winner_id = model_b_id
    else:
        winner_id = winner

    result = db.record_match(
        participant_a=model_a_id,
        participant_b=model_b_id,
        winner=winner_id,
        board_type=board_type,
        num_players=num_players,
        game_length=game_length,
        duration_sec=duration_sec,
        harness_type=harness_type,
        is_multi_harness=is_multi_harness,
    )

    # Return in the format expected by legacy callers
    rating_a = db.get_rating(model_a_id, board_type, num_players)
    rating_b = db.get_rating(model_b_id, board_type, num_players)

    return {
        "model_a": rating_a.rating,
        "model_b": rating_b.rating,
        "changes": result.elo_changes,
        "match_id": result.match_id
    }


def get_leaderboard(
    db: EloService,
    board_type: str = "square8",
    num_players: int = 2,
    limit: int = 50,
    min_games: int = 0,
) -> list[dict[str, Any]]:
    """Get Elo leaderboard (backwards compatible).

    Args:
        db: The EloService instance
        board_type: Board type to query
        num_players: Number of players to query
        limit: Maximum entries to return
        min_games: Minimum games required

    Returns:
        List of leaderboard entries as dicts
    """
    entries = db.get_leaderboard(
        board_type=board_type,
        num_players=num_players,
        limit=limit,
        min_games=min_games
    )
    return [asdict(e) for e in entries]


# =============================================================================
# Additional Query Methods (merged from unified_elo_db.py)
# =============================================================================


def get_head_to_head(
    db: EloService,
    participant_a: str,
    participant_b: str,
    board_type: str | None = None,
    num_players: int | None = None,
) -> dict[str, Any]:
    """Get head-to-head stats between two participants.

    Args:
        db: The EloService instance
        participant_a: First participant ID
        participant_b: Second participant ID
        board_type: Filter by board type (optional)
        num_players: Filter by player count (optional)

    Returns:
        Dict with head-to-head stats
    """
    conn = db._get_connection()

    query = """
        SELECT winner_id FROM match_history
        WHERE participant_ids LIKE ? AND participant_ids LIKE ?
    """
    params: list[Any] = [f'%{participant_a}%', f'%{participant_b}%']

    if board_type:
        query += " AND board_type = ?"
        params.append(board_type)
    if num_players:
        query += " AND num_players = ?"
        params.append(num_players)

    cursor = conn.execute(query, params)

    a_wins = 0
    b_wins = 0
    draws = 0

    for row in cursor:
        winner = row["winner_id"]
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


def get_database_stats(db: EloService) -> dict[str, Any]:
    """Get overall database statistics.

    Args:
        db: The EloService instance

    Returns:
        Dict with database stats
    """
    conn = db._get_connection()

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
        SELECT board_type, num_players, COUNT(*) as count, MAX(rating) as top_rating
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


def get_match_history(
    db: EloService,
    participant_id: str | None = None,
    tournament_id: str | None = None,
    board_type: str | None = None,
    num_players: int | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Get match history with optional filters.

    Args:
        db: The EloService instance
        participant_id: Filter by participant
        tournament_id: Filter by tournament
        board_type: Filter by board type
        num_players: Filter by player count
        limit: Maximum results

    Returns:
        List of match records
    """
    conn = db._get_connection()

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

    cursor = conn.execute(query, params)
    results = []
    for row in cursor.fetchall():
        r = dict(row)
        # Parse JSON fields if present
        if r.get("participant_ids"):
            with suppress(json.JSONDecodeError, TypeError):
                r["participant_ids"] = json.loads(r["participant_ids"])
        if r.get("elo_before"):
            with suppress(json.JSONDecodeError, TypeError):
                r["elo_before"] = json.loads(r["elo_before"])
        if r.get("elo_after"):
            with suppress(json.JSONDecodeError, TypeError):
                r["elo_after"] = json.loads(r["elo_after"])
        results.append(r)
    return results


def get_rating_history(
    db: EloService,
    participant_id: str,
    board_type: str,
    num_players: int,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Get rating history for a participant in a specific config.

    Args:
        db: The EloService instance
        participant_id: Participant to query
        board_type: Board type
        num_players: Player count
        limit: Maximum results

    Returns:
        List of rating history records
    """
    conn = db._get_connection()
    cursor = conn.execute("""
        SELECT * FROM elo_history
        WHERE participant_id = ? AND board_type = ? AND num_players = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (participant_id, board_type, num_players, limit))
    return [dict(row) for row in cursor.fetchall()]


def get_elo_trend(
    db: EloService,
    participant_id: str,
    board_type: str,
    num_players: int,
    hours: int = 48,
    min_samples: int = 3,
) -> dict[str, Any]:
    """Get Elo trend (slope) over the last N hours.

    December 2025: Added for Phase 4 weak config detection.
    This enables detecting permanently stuck configs that should have
    their selfplay resource allocation reduced.

    Args:
        db: The EloService instance
        participant_id: Participant to query (e.g., "canonical_model")
        board_type: Board type (e.g., "square8")
        num_players: Player count
        hours: Time window in hours (default 48h)
        min_samples: Minimum data points required

    Returns:
        Dict with trend analysis:
        - slope: Elo change per hour (positive = improving)
        - start_elo: Rating at start of window
        - end_elo: Current/latest rating
        - total_change: end_elo - start_elo
        - duration_hours: Actual time span covered
        - sample_count: Number of data points
        - is_plateau: True if slope < 1.0 Elo/hour and 5+ samples
        - is_declining: True if slope < -1.0 Elo/hour
        - confidence: 0.0-1.0 based on sample count

    Example:
        trend = get_elo_trend(db, "canonical", "square8", 2, hours=48)
        if trend["is_plateau"] and trend["duration_hours"] >= 48:
            print(f"Config stuck for {trend['duration_hours']:.1f}h")
    """
    cutoff_time = time.time() - (hours * 3600)

    conn = db._get_connection()
    cursor = conn.execute("""
        SELECT rating, timestamp FROM elo_history
        WHERE participant_id = ? AND board_type = ? AND num_players = ?
          AND timestamp >= ?
        ORDER BY timestamp ASC
    """, (participant_id, board_type, num_players, cutoff_time))

    rows = cursor.fetchall()

    # Default return for insufficient data
    result = {
        "slope": 0.0,
        "start_elo": 0.0,
        "end_elo": 0.0,
        "total_change": 0.0,
        "duration_hours": 0.0,
        "sample_count": len(rows),
        "is_plateau": False,
        "is_declining": False,
        "confidence": 0.0,
    }

    if len(rows) < min_samples:
        return result

    # Extract ratings and timestamps
    ratings = [row["rating"] for row in rows]
    timestamps = [row["timestamp"] for row in rows]

    start_elo = ratings[0]
    end_elo = ratings[-1]
    duration_seconds = timestamps[-1] - timestamps[0]
    duration_hours = duration_seconds / 3600.0

    if duration_hours < 0.1:  # Less than 6 minutes
        return result

    # Calculate slope using simple linear regression
    total_change = end_elo - start_elo
    slope = total_change / duration_hours if duration_hours > 0 else 0.0

    # Calculate R² for confidence (optional, simple variance-based)
    mean_rating = sum(ratings) / len(ratings)
    ss_tot = sum((r - mean_rating) ** 2 for r in ratings)
    if ss_tot > 0 and len(ratings) >= 3:
        # Simple linear fit residuals
        predicted = [
            start_elo + slope * ((t - timestamps[0]) / 3600.0)
            for t in timestamps
        ]
        ss_res = sum((r - p) ** 2 for r, p in zip(ratings, predicted))
        r_squared = 1.0 - (ss_res / ss_tot)
        r_squared = max(0.0, min(1.0, r_squared))
    else:
        r_squared = 0.5  # Default moderate confidence

    # Confidence based on sample count and R²
    sample_confidence = min(1.0, len(rows) / 20.0)  # Max confidence at 20 samples
    confidence = (sample_confidence + r_squared) / 2.0

    # Plateau detection: less than 1 Elo/hour change with enough samples
    is_plateau = abs(slope) < 1.0 and len(rows) >= 5

    # Decline detection: losing more than 1 Elo/hour
    is_declining = slope < -1.0

    result.update({
        "slope": round(slope, 3),
        "start_elo": round(start_elo, 1),
        "end_elo": round(end_elo, 1),
        "total_change": round(total_change, 1),
        "duration_hours": round(duration_hours, 2),
        "sample_count": len(rows),
        "is_plateau": is_plateau,
        "is_declining": is_declining,
        "confidence": round(confidence, 3),
    })

    return result


def get_elo_trend_for_config(
    config_key: str,
    hours: int = 48,
    participant_id: str = "canonical",
) -> dict[str, Any]:
    """Convenience function to get Elo trend for a config key.

    December 2025: Added for Phase 4 weak config detection.

    Args:
        config_key: Config identifier like "square8_2p"
        hours: Time window in hours
        participant_id: Participant ID to query

    Returns:
        Trend analysis dict (see get_elo_trend)
    """
    # Parse config key
    parts = config_key.rsplit("_", 1)
    if len(parts) != 2 or not parts[1].endswith("p"):
        return {
            "slope": 0.0, "is_plateau": False, "confidence": 0.0,
            "error": f"Invalid config_key format: {config_key}"
        }

    board_type = parts[0]
    try:
        num_players = int(parts[1][:-1])
    except ValueError:
        return {
            "slope": 0.0, "is_plateau": False, "confidence": 0.0,
            "error": f"Invalid player count in config_key: {config_key}"
        }

    db = get_elo_service()
    return get_elo_trend(db, participant_id, board_type, num_players, hours)


# Canonical path - orchestrators should use this
ELO_DB_PATH = DEFAULT_ELO_DB_PATH

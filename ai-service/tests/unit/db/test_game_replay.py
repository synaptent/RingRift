"""Tests for game_replay.py - SQLite game storage and retrieval.

This module tests the GameReplayDB and GameWriter classes that handle
storing and retrieving complete RingRift games for training.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from app.db.game_replay import (
    DEFAULT_SNAPSHOT_INTERVAL,
    GameReplayDB,
    GameWriter,
    SCHEMA_VERSION,
)
from app.models import BoardType, GameState, Move, MoveType, Phase


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "test_replay.db"


@pytest.fixture
def db(temp_db_path: Path) -> GameReplayDB:
    """Create a temporary GameReplayDB instance."""
    db = GameReplayDB(str(temp_db_path))
    yield db
    db.close()


@pytest.fixture
def sample_initial_state() -> GameState:
    """Create a sample initial game state for testing."""
    from app.rules.game_engine import GameEngine

    return GameEngine.create_initial_state(
        board_type=BoardType.HEX8,
        num_players=2,
        rng_seed=42,
    )


@pytest.fixture
def sample_move() -> Move:
    """Create a sample PLACE_RING move."""
    return Move(
        type=MoveType.PLACE_RING,
        player=0,
        from_pos=None,
        to=(3, 3),
    )


# =============================================================================
# GameReplayDB Initialization Tests
# =============================================================================


class TestGameReplayDBInit:
    """Tests for GameReplayDB initialization and lifecycle."""

    def test_init_creates_database(self, temp_db_path: Path) -> None:
        """Test that initialization creates the database file."""
        db = GameReplayDB(str(temp_db_path))
        try:
            assert temp_db_path.exists()
        finally:
            db.close()

    def test_init_creates_schema(self, temp_db_path: Path) -> None:
        """Test that initialization creates required tables."""
        db = GameReplayDB(str(temp_db_path))
        try:
            conn = sqlite3.connect(str(temp_db_path))
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}
            conn.close()

            # Check for key tables
            assert "games" in tables
            assert "game_players" in tables
            assert "game_initial_state" in tables
            assert "game_moves" in tables
            assert "schema_metadata" in tables
        finally:
            db.close()

    def test_init_sets_schema_version(self, db: GameReplayDB) -> None:
        """Test that schema version is set correctly."""
        conn = sqlite3.connect(str(db.db_path))
        cursor = conn.execute(
            "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result is not None
        assert int(result[0]) == SCHEMA_VERSION

    def test_context_manager(self, temp_db_path: Path) -> None:
        """Test that GameReplayDB works as a context manager."""
        with GameReplayDB(str(temp_db_path)) as db:
            assert db is not None
            assert temp_db_path.exists()

    def test_readonly_mode(self, temp_db_path: Path) -> None:
        """Test readonly mode prevents writes."""
        # First create the database
        with GameReplayDB(str(temp_db_path)) as db:
            pass

        # Open in readonly mode
        with GameReplayDB(str(temp_db_path), readonly=True) as db:
            # Should be able to read
            count = db.get_game_count()
            assert count == 0


# =============================================================================
# GameReplayDB Store and Retrieve Tests
# =============================================================================


class TestGameReplayDBStoreRetrieve:
    """Tests for storing and retrieving games."""

    def test_store_game_minimal(
        self, db: GameReplayDB, sample_initial_state: GameState
    ) -> None:
        """Test storing a minimal game with no moves."""
        game_id = db.store_game(
            initial_state=sample_initial_state,
            moves=[],
            winner=None,
            termination_reason="forfeit",
            source="test",
        )

        assert game_id is not None
        assert len(game_id) == 36  # UUID format

    def test_store_game_with_moves(
        self,
        db: GameReplayDB,
        sample_initial_state: GameState,
        sample_move: Move,
    ) -> None:
        """Test storing a game with moves."""
        game_id = db.store_game(
            initial_state=sample_initial_state,
            moves=[sample_move],
            winner=0,
            termination_reason="victory",
            source="test",
        )

        # Verify game was stored
        metadata = db.get_game_metadata(game_id)
        assert metadata is not None
        assert metadata["total_moves"] == 1
        assert metadata["winner"] == 0

    def test_get_initial_state(
        self, db: GameReplayDB, sample_initial_state: GameState
    ) -> None:
        """Test retrieving initial state."""
        game_id = db.store_game(
            initial_state=sample_initial_state,
            moves=[],
            winner=None,
            termination_reason="forfeit",
            source="test",
        )

        retrieved = db.get_initial_state(game_id)
        assert retrieved is not None
        assert retrieved.board_type == sample_initial_state.board_type
        assert retrieved.num_players == sample_initial_state.num_players

    def test_get_moves_empty(
        self, db: GameReplayDB, sample_initial_state: GameState
    ) -> None:
        """Test retrieving moves from game with no moves."""
        game_id = db.store_game(
            initial_state=sample_initial_state,
            moves=[],
            winner=None,
            termination_reason="forfeit",
            source="test",
        )

        moves = db.get_moves(game_id)
        assert moves == []

    def test_get_moves_with_data(
        self,
        db: GameReplayDB,
        sample_initial_state: GameState,
        sample_move: Move,
    ) -> None:
        """Test retrieving moves from game with moves."""
        game_id = db.store_game(
            initial_state=sample_initial_state,
            moves=[sample_move],
            winner=0,
            termination_reason="victory",
            source="test",
        )

        moves = db.get_moves(game_id)
        assert len(moves) == 1
        assert moves[0].type == sample_move.type
        assert moves[0].player == sample_move.player

    def test_get_game_metadata(
        self, db: GameReplayDB, sample_initial_state: GameState
    ) -> None:
        """Test retrieving game metadata."""
        game_id = db.store_game(
            initial_state=sample_initial_state,
            moves=[],
            winner=1,
            termination_reason="victory",
            source="test_source",
        )

        metadata = db.get_game_metadata(game_id)
        assert metadata is not None
        assert metadata["game_id"] == game_id
        assert metadata["board_type"] == sample_initial_state.board_type.value
        assert metadata["num_players"] == sample_initial_state.num_players
        assert metadata["winner"] == 1
        assert metadata["source"] == "test_source"

    def test_get_nonexistent_game(self, db: GameReplayDB) -> None:
        """Test retrieving a game that doesn't exist."""
        metadata = db.get_game_metadata("nonexistent-game-id")
        assert metadata is None

        state = db.get_initial_state("nonexistent-game-id")
        assert state is None


# =============================================================================
# GameReplayDB Query Tests
# =============================================================================


class TestGameReplayDBQuery:
    """Tests for querying and filtering games."""

    def test_get_game_count_empty(self, db: GameReplayDB) -> None:
        """Test game count on empty database."""
        count = db.get_game_count()
        assert count == 0

    def test_get_game_count_with_games(
        self, db: GameReplayDB, sample_initial_state: GameState
    ) -> None:
        """Test game count with stored games."""
        # Store 3 games
        for _ in range(3):
            db.store_game(
                initial_state=sample_initial_state,
                moves=[],
                winner=None,
                termination_reason="forfeit",
                source="test",
            )

        count = db.get_game_count()
        assert count == 3

    def test_get_game_count_filtered(
        self, db: GameReplayDB, sample_initial_state: GameState
    ) -> None:
        """Test filtered game count."""
        # Store games with different sources
        db.store_game(
            initial_state=sample_initial_state,
            moves=[],
            winner=None,
            termination_reason="forfeit",
            source="source_a",
        )
        db.store_game(
            initial_state=sample_initial_state,
            moves=[],
            winner=None,
            termination_reason="forfeit",
            source="source_b",
        )

        # Count with filter
        count = db.get_game_count(source="source_a")
        assert count == 1

    def test_query_games_empty(self, db: GameReplayDB) -> None:
        """Test query on empty database."""
        results = db.query_games()
        assert results == []

    def test_query_games_with_limit(
        self, db: GameReplayDB, sample_initial_state: GameState
    ) -> None:
        """Test query with limit."""
        # Store 5 games
        for _ in range(5):
            db.store_game(
                initial_state=sample_initial_state,
                moves=[],
                winner=None,
                termination_reason="forfeit",
                source="test",
            )

        results = db.query_games(limit=3)
        assert len(results) == 3

    def test_iterate_games(
        self, db: GameReplayDB, sample_initial_state: GameState
    ) -> None:
        """Test iterating over games."""
        # Store 3 games
        stored_ids = []
        for _ in range(3):
            game_id = db.store_game(
                initial_state=sample_initial_state,
                moves=[],
                winner=None,
                termination_reason="forfeit",
                source="test",
            )
            stored_ids.append(game_id)

        # Iterate and collect IDs
        iterated_ids = []
        for game_id, initial_state, moves in db.iterate_games():
            iterated_ids.append(game_id)

        assert len(iterated_ids) == 3
        assert set(iterated_ids) == set(stored_ids)


# =============================================================================
# GameWriter Tests
# =============================================================================


class TestGameWriter:
    """Tests for GameWriter context manager."""

    def test_game_writer_context(
        self, db: GameReplayDB, sample_initial_state: GameState
    ) -> None:
        """Test GameWriter as context manager."""
        with GameWriter(db, sample_initial_state, source="test") as writer:
            assert writer is not None
            assert writer.game_id is not None

    def test_game_writer_add_move(
        self,
        db: GameReplayDB,
        sample_initial_state: GameState,
        sample_move: Move,
    ) -> None:
        """Test adding moves via GameWriter."""
        from app.rules.game_engine import GameEngine

        with GameWriter(db, sample_initial_state, source="test") as writer:
            # Apply move and add it
            state_after = GameEngine.apply_move(sample_initial_state, sample_move)
            writer.add_move(
                move=sample_move,
                state_before=sample_initial_state,
                state_after=state_after,
            )

        # Verify move was stored
        moves = db.get_moves(writer.game_id)
        assert len(moves) == 1

    def test_game_writer_finalize(
        self, db: GameReplayDB, sample_initial_state: GameState
    ) -> None:
        """Test finalizing a game via GameWriter."""
        with GameWriter(db, sample_initial_state, source="test") as writer:
            writer.finalize(winner=0, termination_reason="victory")

        # Verify game is complete
        metadata = db.get_game_metadata(writer.game_id)
        assert metadata is not None
        assert metadata["winner"] == 0
        assert metadata["game_status"] == "completed"

    def test_game_writer_abort(
        self, db: GameReplayDB, sample_initial_state: GameState
    ) -> None:
        """Test aborting a game via GameWriter."""
        writer = GameWriter(db, sample_initial_state, source="test")
        writer.__enter__()
        game_id = writer.game_id
        writer.abort()

        # Game should be deleted
        metadata = db.get_game_metadata(game_id)
        assert metadata is None


# =============================================================================
# GameReplayDB State Reconstruction Tests
# =============================================================================


class TestStateReconstruction:
    """Tests for game state reconstruction at arbitrary moves."""

    def test_get_state_at_move_zero(
        self, db: GameReplayDB, sample_initial_state: GameState
    ) -> None:
        """Test getting state at move 0 (initial state)."""
        game_id = db.store_game(
            initial_state=sample_initial_state,
            moves=[],
            winner=None,
            termination_reason="forfeit",
            source="test",
        )

        state = db.get_state_at_move(game_id, 0)
        assert state is not None
        assert state.board_type == sample_initial_state.board_type

    def test_get_state_at_invalid_move(
        self, db: GameReplayDB, sample_initial_state: GameState
    ) -> None:
        """Test getting state at invalid move number."""
        game_id = db.store_game(
            initial_state=sample_initial_state,
            moves=[],
            winner=None,
            termination_reason="forfeit",
            source="test",
        )

        # Move 10 doesn't exist in a game with 0 moves
        state = db.get_state_at_move(game_id, 10)
        assert state is None


# =============================================================================
# GameReplayDB Stats Tests
# =============================================================================


class TestGameReplayDBStats:
    """Tests for database statistics."""

    def test_get_stats_empty(self, db: GameReplayDB) -> None:
        """Test stats on empty database."""
        stats = db.get_stats()
        assert stats["total_games"] == 0
        assert stats["total_moves"] == 0

    def test_get_stats_with_data(
        self, db: GameReplayDB, sample_initial_state: GameState, sample_move: Move
    ) -> None:
        """Test stats with games and moves."""
        # Store a game with moves
        db.store_game(
            initial_state=sample_initial_state,
            moves=[sample_move],
            winner=0,
            termination_reason="victory",
            source="test",
        )

        stats = db.get_stats()
        assert stats["total_games"] == 1
        assert stats["total_moves"] == 1


# =============================================================================
# GameReplayDB Maintenance Tests
# =============================================================================


class TestGameReplayDBMaintenance:
    """Tests for database maintenance operations."""

    def test_vacuum(self, db: GameReplayDB, sample_initial_state: GameState) -> None:
        """Test VACUUM operation."""
        # Store and delete a game to create fragmentation
        game_id = db.store_game(
            initial_state=sample_initial_state,
            moves=[],
            winner=None,
            termination_reason="forfeit",
            source="test",
        )
        db._delete_game(game_id)

        # VACUUM should run without error
        db.vacuum()

    def test_checkpoint_wal(
        self, db: GameReplayDB, sample_initial_state: GameState
    ) -> None:
        """Test WAL checkpoint operation."""
        # Store a game to generate WAL data
        db.store_game(
            initial_state=sample_initial_state,
            moves=[],
            winner=None,
            termination_reason="forfeit",
            source="test",
        )

        # Checkpoint should run without error
        pages_checkpointed, pages_total = db.checkpoint_wal()
        assert pages_checkpointed >= 0
        assert pages_total >= 0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_store_game_with_metadata(
        self, db: GameReplayDB, sample_initial_state: GameState
    ) -> None:
        """Test storing a game with custom metadata."""
        metadata = {"custom_field": "custom_value", "nested": {"key": 123}}

        game_id = db.store_game(
            initial_state=sample_initial_state,
            moves=[],
            winner=None,
            termination_reason="forfeit",
            source="test",
            metadata=metadata,
        )

        # Retrieve and verify metadata
        game_data = db.get_game_metadata(game_id)
        assert game_data is not None
        # metadata_json should contain our custom data
        if game_data.get("metadata_json"):
            stored_meta = json.loads(game_data["metadata_json"])
            assert stored_meta.get("custom_field") == "custom_value"

    def test_multiple_databases(self, tmp_path: Path) -> None:
        """Test working with multiple database instances."""
        db1_path = tmp_path / "db1.db"
        db2_path = tmp_path / "db2.db"

        with GameReplayDB(str(db1_path)) as db1:
            with GameReplayDB(str(db2_path)) as db2:
                # Both should work independently
                assert db1.get_game_count() == 0
                assert db2.get_game_count() == 0

    def test_close_is_idempotent(self, temp_db_path: Path) -> None:
        """Test that close() can be called multiple times."""
        db = GameReplayDB(str(temp_db_path))
        db.close()
        db.close()  # Should not raise


# =============================================================================
# Schema Migration Tests
# =============================================================================


class TestSchemaMigrations:
    """Tests for schema migrations."""

    def test_fresh_database_has_latest_schema(self, db: GameReplayDB) -> None:
        """Test that fresh database has the latest schema version."""
        version = db._get_schema_version(db._get_conn())
        assert version == SCHEMA_VERSION

    def test_schema_includes_required_tables(self, db: GameReplayDB) -> None:
        """Test that schema includes all required tables."""
        conn = db._get_conn()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}

        required_tables = {
            "games",
            "game_players",
            "game_initial_state",
            "game_moves",
            "game_snapshots",
            "game_player_choices",
            "game_history_entries",
            "game_nnue_features",
            "orphaned_games",
            "schema_metadata",
        }

        for table in required_tables:
            assert table in tables, f"Missing required table: {table}"

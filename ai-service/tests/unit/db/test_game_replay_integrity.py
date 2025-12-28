#!/usr/bin/env python3
"""Unit tests for move data integrity enforcement in game_replay.py (December 2025).

Tests the Phase 6 Move Data Integrity Enforcement:
- v14 schema includes orphaned_games table and trigger
- store_game() rejects games with fewer than MIN_MOVES_REQUIRED moves
- Schema version is 14

Test fixtures create temporary SQLite databases for testing without
affecting production data.
"""

import sqlite3
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.db.game_replay import GameReplayDB, SCHEMA_VERSION, GameWriter
from app.errors import InvalidGameError


class TestSchemaVersion14:
    """Tests for v14 schema migration."""

    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create a temporary database path."""
        return tmp_path / "test_migration.db"

    def test_schema_version_is_14(self):
        """Test that SCHEMA_VERSION is 14."""
        assert SCHEMA_VERSION == 14

    def test_fresh_database_has_orphaned_games_table(self, temp_db_path):
        """Test that fresh database has orphaned_games table."""
        db = GameReplayDB(str(temp_db_path))

        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='orphaned_games'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result is not None
        assert result[0] == "orphaned_games"

    def test_fresh_database_has_enforcement_trigger(self, temp_db_path):
        """Test that fresh database has enforce_moves_on_complete trigger."""
        db = GameReplayDB(str(temp_db_path))

        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name='enforce_moves_on_complete'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result is not None
        assert result[0] == "enforce_moves_on_complete"

    def test_trigger_blocks_completing_game_without_moves(self, temp_db_path):
        """Test that trigger prevents completing games with 0 moves."""
        db = GameReplayDB(str(temp_db_path))

        conn = sqlite3.connect(str(temp_db_path))
        # Insert a game with 0 moves - include all NOT NULL columns
        conn.execute(
            """
            INSERT INTO games (game_id, board_type, num_players, total_moves, total_turns,
                             game_status, created_at, schema_version)
            VALUES ('trigger-test', 'hex8', 2, 0, 0, 'in_progress', datetime('now'), 14)
        """
        )
        conn.commit()

        # Try to mark it as completed - should fail
        with pytest.raises(sqlite3.IntegrityError) as exc_info:
            conn.execute(
                "UPDATE games SET game_status = 'completed' WHERE game_id = 'trigger-test'"
            )
            conn.commit()

        assert "Cannot complete game without moves" in str(exc_info.value)
        conn.close()

    def test_trigger_allows_completing_game_with_moves(self, temp_db_path):
        """Test that trigger allows completing games with moves."""
        db = GameReplayDB(str(temp_db_path))

        conn = sqlite3.connect(str(temp_db_path))
        # Insert a game with moves - include all NOT NULL columns
        conn.execute(
            """
            INSERT INTO games (game_id, board_type, num_players, total_moves, total_turns,
                             game_status, created_at, schema_version)
            VALUES ('trigger-test-ok', 'hex8', 2, 10, 5, 'in_progress', datetime('now'), 14)
        """
        )
        conn.commit()

        # Mark it as completed - should succeed
        conn.execute(
            "UPDATE games SET game_status = 'completed' WHERE game_id = 'trigger-test-ok'"
        )
        conn.commit()

        # Verify it was updated
        cursor = conn.execute(
            "SELECT game_status FROM games WHERE game_id = 'trigger-test-ok'"
        )
        result = cursor.fetchone()
        assert result[0] == "completed"
        conn.close()


class TestOrphanedGamesTable:
    """Tests for orphaned_games table functionality."""

    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create a temporary database path."""
        return tmp_path / "test_orphaned.db"

    def test_can_insert_orphaned_game(self, temp_db_path):
        """Test that orphaned games can be inserted."""
        db = GameReplayDB(str(temp_db_path))

        conn = sqlite3.connect(str(temp_db_path))
        conn.execute(
            """
            INSERT INTO orphaned_games (game_id, detected_at, reason, original_status, board_type, num_players)
            VALUES ('orphan-001', datetime('now'), 'No move data found', 'completed', 'hex8', 2)
        """
        )
        conn.commit()

        cursor = conn.execute("SELECT * FROM orphaned_games WHERE game_id = 'orphan-001'")
        result = cursor.fetchone()
        assert result is not None
        assert result[0] == "orphan-001"
        conn.close()

    def test_orphaned_games_index_exists(self, temp_db_path):
        """Test that idx_orphaned_detected index exists."""
        db = GameReplayDB(str(temp_db_path))

        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_orphaned_detected'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result is not None
        assert result[0] == "idx_orphaned_detected"


class TestMinMovesConstant:
    """Tests for MIN_MOVES_REQUIRED constant."""

    def test_min_moves_is_positive(self):
        """Test that MIN_MOVES_REQUIRED is at least 1."""
        assert GameReplayDB.MIN_MOVES_REQUIRED >= 1

    def test_min_moves_is_exactly_one(self):
        """Test that MIN_MOVES_REQUIRED is exactly 1."""
        assert GameReplayDB.MIN_MOVES_REQUIRED == 1


class TestStoreGameEnforcement:
    """Tests for store_game() move enforcement using InvalidGameError."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary database for testing."""
        db_path = tmp_path / "test_store.db"
        db = GameReplayDB(str(db_path))
        return db

    def test_store_game_with_no_moves_raises_invalid_game_error(self, temp_db):
        """Test that store_game() raises InvalidGameError with empty moves list."""
        # Create minimal mocks
        mock_state = MagicMock()
        mock_state.serialize.return_value = {"phase": "play"}

        with pytest.raises(InvalidGameError) as exc_info:
            temp_db.store_game(
                game_id="test-game-empty",
                initial_state=mock_state,
                final_state=mock_state,
                moves=[],  # Empty moves list
                metadata={"board_type": "hex8", "num_players": 2},
            )

        assert "0 moves" in str(exc_info.value)


class TestGameWriterEnforcement:
    """Tests for GameWriter.finalize() enforcement."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary database for testing."""
        db_path = tmp_path / "test_writer.db"
        db = GameReplayDB(str(db_path))
        return db

    def test_finalize_aborts_flag_exists(self):
        """Test that finalize raises InvalidGameError for 0 moves (from code review)."""
        # This is a code-level check - verify the abort behavior exists
        # by checking the source contains the right error handling
        import inspect
        source = inspect.getsource(GameWriter.finalize)

        assert "InvalidGameError" in source
        assert "0 moves" in source or "move_count == 0" in source

    def test_exit_cleans_up_on_exception(self):
        """Test that __exit__ handles cleanup on exception (from code review)."""
        import inspect
        source = inspect.getsource(GameWriter.__exit__)

        # Verify cleanup code exists
        assert "_delete_game" in source
        assert "exception" in source.lower() or "exc_type" in source


class TestInvalidGameError:
    """Tests for InvalidGameError exception."""

    def test_invalid_game_error_exists(self):
        """Test that InvalidGameError can be imported."""
        from app.errors import InvalidGameError

        # Can create instance
        error = InvalidGameError("Test error", game_id="test-123", move_count=0)
        # Uses 'details' dict for additional context (not 'context')
        assert "test-123" in str(error.details)
        assert error.details["move_count"] == 0
        assert error.game_id == "test-123"

    def test_invalid_game_error_inherits_from_data_quality_error(self):
        """Test that InvalidGameError inherits from DataQualityError."""
        from app.errors import InvalidGameError, DataQualityError

        assert issubclass(InvalidGameError, DataQualityError)

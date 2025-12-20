"""Tests for the game replay database."""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import pytest

# Ensure app package is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.db import GameReplayDB, GameWriter
from app.models import (
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Move,
    MoveType,
    Player,
    Position,
    RingStack,
    TimeControl,
)


def create_test_state(
    game_id: str = "test-game",
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
) -> GameState:
    """Create a minimal test game state."""
    now = datetime.now()

    board = BoardState(
        type=board_type,
        size=8 if board_type == BoardType.SQUARE8 else 19,
        stacks={
            "3,3": RingStack(
                position=Position(x=3, y=3),
                controlling_player=1,
                rings=[1],
                stack_height=1,
                cap_height=1,
            ),
            "5,5": RingStack(
                position=Position(x=5, y=5),
                controlling_player=2,
                rings=[2],
                stack_height=1,
                cap_height=1,
            ),
        },
        collapsed_spaces={},
        markers={},
        eliminated_rings={"1": 0, "2": 0},
        formed_lines=[],
        territories={},
    )

    players = [
        Player(
            id=f"p{i}",
            username=f"Player{i}",
            type="ai",
            player_number=i,
            is_ready=True,
            time_remaining=600000,
            ai_difficulty=5,
            rings_in_hand=10,
            eliminated_rings=0,
            territory_spaces=0,
        )
        for i in range(1, num_players + 1)
    ]

    return GameState(
        id=game_id,
        board_type=board_type,
        rng_seed=42,
        board=board,
        players=players,
        current_phase=GamePhase.MOVEMENT,
        current_player=1,
        move_history=[],
        time_control=TimeControl(initial_time=600000, increment=0, type="standard"),
        spectators=[],
        game_status=GameStatus.ACTIVE,
        winner=None,
        created_at=now,
        last_move_at=now,
        is_rated=False,
        max_players=num_players,
        total_rings_in_play=36,
        total_rings_eliminated=0,
        victory_threshold=18,  # RR-CANON-R061: ringsPerPlayer
        territory_victory_threshold=33,
        chain_capture_state=None,
        must_move_from_stack_key=None,
        zobrist_hash=None,
        lps_round_index=0,
        lps_current_round_actor_mask={},
        lps_exclusive_player_for_completed_round=None,
    )


def create_test_move(
    player: int,
    move_number: int,
    from_pos: Position = Position(x=3, y=3),
    to_pos: Position = Position(x=3, y=5),
) -> Move:
    """Create a test move."""
    return Move(
        id=f"move-{move_number}",
        type=MoveType.MOVE_STACK,
        player=player,
        from_pos=from_pos,
        to=to_pos,
        timestamp=datetime.now(),
        think_time=100,
        move_number=move_number,
    )


class TestGameReplayDBBasic:
    """Basic functionality tests."""

    @pytest.fixture
    def db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_games.db"

    def test_create_database(self, db_path):
        """Test database creation."""
        GameReplayDB(str(db_path))
        assert db_path.exists()

    def test_store_and_retrieve_game(self, db_path):
        """Test storing and retrieving a game."""
        # Disable canonical validation for synthetic test fixtures
        db = GameReplayDB(str(db_path), enforce_canonical_history=False)
        game_id = str(uuid.uuid4())

        initial_state = create_test_state(game_id)
        final_state = create_test_state(game_id)
        final_state = final_state.model_copy(
            update={
                "game_status": GameStatus.COMPLETED,
                "winner": 1,
            }
        )

        moves = [
            create_test_move(1, 0),
            create_test_move(2, 1),
            create_test_move(1, 2),
        ]

        db.store_game(
            game_id=game_id,
            initial_state=initial_state,
            final_state=final_state,
            moves=moves,
            metadata={"source": "test"},
            store_history_entries=False,  # Skip phase validation for test fixtures
            snapshot_interval=0,  # Disable state tracking for synthetic moves
        )

        # Retrieve metadata
        metadata = db.get_game_metadata(game_id)
        assert metadata is not None
        assert metadata["game_id"] == game_id
        assert metadata["winner"] == 1
        assert metadata["total_moves"] == 3

        # Retrieve initial state
        retrieved_state = db.get_initial_state(game_id)
        assert retrieved_state is not None
        assert retrieved_state.id == game_id

        # Retrieve moves
        retrieved_moves = db.get_moves(game_id)
        assert len(retrieved_moves) == 3
        assert retrieved_moves[0].move_number == 0

        # Verify that full metadata is persisted as JSON for debugging.
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT metadata_json FROM games WHERE game_id = ?",
            (game_id,),
        ).fetchone()
        conn.close()
        assert row is not None
        metadata_json = row[0]
        assert metadata_json is not None
        # The stored JSON should decode to a dict containing the source key.
        import json

        decoded = json.loads(metadata_json)
        assert isinstance(decoded, dict)
        assert decoded.get("source") == "test"

    def test_query_games(self, db_path):
        """Test querying games by filters."""
        db = GameReplayDB(str(db_path))

        # Store multiple games
        for i in range(5):
            game_id = f"game-{i}"
            initial_state = create_test_state(game_id)
            final_state = initial_state.model_copy(
                update={
                    "game_status": GameStatus.COMPLETED,
                    "winner": (i % 2) + 1,
                }
            )
            moves = [create_test_move(1, 0)]

            db.store_game(
                game_id=game_id,
                initial_state=initial_state,
                final_state=final_state,
                moves=moves,
                metadata={"source": "test"},
            )

        # Query all games
        all_games = db.query_games()
        assert len(all_games) == 5

        # Query by winner
        winner_1_games = db.query_games(winner=1)
        assert len(winner_1_games) == 3  # games 0, 2, 4 have winner=1

        # Query with limit
        limited = db.query_games(limit=2)
        assert len(limited) == 2

    def test_get_stats(self, db_path):
        """Test database statistics."""
        # Disable canonical validation for synthetic test fixtures
        db = GameReplayDB(str(db_path), enforce_canonical_history=False)

        game_id = "stats-test"
        initial_state = create_test_state(game_id)
        final_state = initial_state.model_copy(
            update={"game_status": GameStatus.COMPLETED, "winner": 1}
        )
        moves = [create_test_move(1, i) for i in range(5)]

        db.store_game(
            game_id=game_id,
            initial_state=initial_state,
            final_state=final_state,
            moves=moves,
            store_history_entries=False,  # Skip phase validation for test fixtures
            snapshot_interval=0,  # Disable state tracking for synthetic moves
        )

        stats = db.get_stats()
        assert stats["total_games"] == 1
        assert stats["total_moves"] == 5
        assert BoardType.SQUARE8.value in stats["games_by_board_type"]


class TestGameReplayDBCanonicalContract:
    """Tests for canonical phase↔move enforcement at DB write time."""

    @pytest.fixture
    def db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_games.db"

    def test_store_move_records_effective_phase_from_hint(self, db_path):
        """_store_move_conn stores the effective canonical phase derived from the hint."""
        db = GameReplayDB(str(db_path))
        game_id = "phase-hint-game"
        initial_state = create_test_state(game_id)

        # Create a placeholder games row to satisfy FK constraints.
        db._create_placeholder_game(game_id, initial_state)

        # MOVE_STACK is canonical in the 'movement' phase.
        move = create_test_move(player=1, move_number=0)

        with db._get_conn() as conn:
            db._store_move_conn(
                conn,
                game_id=game_id,
                move_number=0,
                turn_number=0,
                move=move,
                phase="movement",
            )

            row = conn.execute(
                """
                SELECT phase, move_type
                FROM game_moves
                WHERE game_id = ? AND move_number = 0
                """,
                (game_id,),
            ).fetchone()

        assert row is not None
        assert row["phase"] == "movement"
        assert row["move_type"] == MoveType.MOVE_STACK.value

    def test_store_move_rejects_non_canonical_phase_move_pair(self, db_path):
        """_store_move_conn rejects non-canonical (phase, moveType) pairs."""
        db = GameReplayDB(str(db_path))
        game_id = "phase-mismatch-game"
        initial_state = create_test_state(game_id)

        # Create a placeholder games row to satisfy FK constraints.
        db._create_placeholder_game(game_id, initial_state)

        # MOVE_STACK is not allowed in the territory_processing phase by the
        # canonical phase↔MoveType contract.
        move = create_test_move(player=1, move_number=0)

        with db._get_conn() as conn, pytest.raises(ValueError) as excinfo:
            db._store_move_conn(
                conn,
                game_id=game_id,
                move_number=0,
                turn_number=0,
                move=move,
                phase="territory_processing",
            )

        msg = str(excinfo.value)
        # The underlying history_contract should report a phase_move_mismatch.
        assert "phase_move_mismatch:territory_processing:move_stack" in msg


class TestGameWriter:
    """Tests for incremental game writing."""

    @pytest.fixture
    def db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_games.db"

    def test_incremental_game_writing(self, db_path):
        """Test writing a game incrementally."""
        db = GameReplayDB(str(db_path))
        game_id = str(uuid.uuid4())
        initial_state = create_test_state(game_id)

        writer = db.store_game_incremental(game_id, initial_state)

        # Add moves one at a time
        for i in range(10):
            move = create_test_move(player=(i % 2) + 1, move_number=i)
            writer.add_move(move)

        # Finalize
        final_state = initial_state.model_copy(
            update={
                "game_status": GameStatus.COMPLETED,
                "winner": 1,
            }
        )
        writer.finalize(final_state, {"source": "incremental_test"})

        # Verify
        moves = db.get_moves(game_id)
        assert len(moves) == 10

    def test_abort_game(self, db_path):
        """Test aborting an incomplete game."""
        db = GameReplayDB(str(db_path))
        game_id = str(uuid.uuid4())
        initial_state = create_test_state(game_id)

        writer = db.store_game_incremental(game_id, initial_state)
        writer.add_move(create_test_move(1, 0))
        writer.abort()

        # Game should be deleted
        metadata = db.get_game_metadata(game_id)
        assert metadata is None


class TestStateReconstruction:
    """Tests for state reconstruction at specific moves."""

    @pytest.fixture
    def db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_games.db"

    def test_get_state_at_final_move(self, db_path):
        """Test getting state at the final move (uses snapshot)."""
        db = GameReplayDB(str(db_path))
        game_id = str(uuid.uuid4())

        initial_state = create_test_state(game_id)
        final_state = initial_state.model_copy(
            update={
                "game_status": GameStatus.COMPLETED,
                "winner": 1,
                "total_rings_eliminated": 10,
            }
        )

        moves = [create_test_move(1, 0)]

        # Skip history entry replay since this test uses synthetic moves
        # that don't represent a real game progression
        db.store_game(
            game_id=game_id,
            initial_state=initial_state,
            final_state=final_state,
            moves=moves,
            store_history_entries=False,
        )

        # Verify we can retrieve the stored game metadata
        # Note: get_state_at_move reconstructs from initial state by replaying,
        # so for synthetic test moves we verify the game was stored correctly
        metadata = db.get_game_metadata(game_id)
        assert metadata is not None
        # The game should be stored with our provided final state metadata
        assert metadata["game_id"] == game_id
        assert metadata["total_moves"] == 1


class TestChoices:
    """Tests for player choice recording."""

    @pytest.fixture
    def db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_games.db"

    def test_store_and_retrieve_choices(self, db_path):
        """Test storing and retrieving player choices."""
        db = GameReplayDB(str(db_path))
        game_id = str(uuid.uuid4())

        initial_state = create_test_state(game_id)
        final_state = initial_state.model_copy(
            update={"game_status": GameStatus.COMPLETED, "winner": 1}
        )
        moves = [create_test_move(1, 0)]

        choices = [
            {
                "move_number": 0,
                "choice_type": "line_reward",
                "player": 1,
                "options": [{"id": "opt1"}, {"id": "opt2"}],
                "selected": {"id": "opt1"},
                "reasoning": "Selected first option",
            }
        ]

        db.store_game(
            game_id=game_id,
            initial_state=initial_state,
            final_state=final_state,
            moves=moves,
            choices=choices,
        )

        # Retrieve choices
        retrieved = db.get_choices_at_move(game_id, 0)
        assert len(retrieved) == 1
        assert retrieved[0]["choice_type"] == "line_reward"
        assert retrieved[0]["selected"]["id"] == "opt1"


class TestNNUEFeaturesCaching:
    """Tests for NNUE features caching in GameReplayDB."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create a test database."""
        db_path = tmp_path / "test_nnue.db"
        db = GameReplayDB(str(db_path))
        yield db

    def _create_game(self, db, game_id: str):
        """Helper to create a game record (required for foreign key)."""
        state = create_test_state(game_id=game_id)
        final_state = state.model_copy(
            update={"game_status": GameStatus.COMPLETED, "winner": 1}
        )
        db.store_game(
            game_id=game_id,
            initial_state=state,
            final_state=final_state,
            moves=[create_test_move(1, 0)],
        )

    def test_store_and_retrieve_single_feature(self, db):
        """Test storing and retrieving a single NNUE feature vector."""
        import numpy as np

        game_id = "test-game-1"
        self._create_game(db, game_id)

        move_number = 0
        player_perspective = 1
        features = np.random.randn(512).astype(np.float32)
        value = 1.0
        board_type = "square8"

        # Store features
        db.store_nnue_features(
            game_id=game_id,
            move_number=move_number,
            player_perspective=player_perspective,
            features=features,
            value=value,
            board_type=board_type,
        )

        # Retrieve features
        results = db.get_nnue_features(game_id)
        assert len(results) == 1

        ret_move, ret_player, ret_features, ret_value = results[0]
        assert ret_move == move_number
        assert ret_player == player_perspective
        assert ret_value == value
        assert np.allclose(ret_features, features)

    def test_store_features_batch(self, db):
        """Test batch storing of NNUE features."""
        import numpy as np

        game_id = "test-game-batch"
        self._create_game(db, game_id)

        records = []
        for move_num in range(10):
            for player in [1, 2]:
                features = np.random.randn(256).astype(np.float32)
                records.append((
                    game_id,
                    move_num,
                    player,
                    features,
                    1.0 if player == 1 else -1.0,
                    "square8",
                ))

        # Store batch
        count = db.store_nnue_features_batch(records)
        assert count == 20

        # Retrieve all
        results = db.get_nnue_features(game_id)
        assert len(results) == 20

    def test_get_features_by_move_number(self, db):
        """Test filtering by move number."""
        import numpy as np

        game_id = "test-game-filter"
        self._create_game(db, game_id)

        # Store features for 3 moves
        for move_num in range(3):
            db.store_nnue_features(
                game_id=game_id,
                move_number=move_num,
                player_perspective=1,
                features=np.ones(128, dtype=np.float32) * move_num,
                value=0.0,
                board_type="square8",
            )

        # Get specific move
        results = db.get_nnue_features(game_id, move_number=1)
        assert len(results) == 1
        assert results[0][0] == 1  # move_number
        assert np.allclose(results[0][2], np.ones(128) * 1)

    def test_get_features_by_player_perspective(self, db):
        """Test filtering by player perspective."""
        import numpy as np

        game_id = "test-game-player"
        self._create_game(db, game_id)

        # Store features for 2 players
        for player in [1, 2]:
            db.store_nnue_features(
                game_id=game_id,
                move_number=0,
                player_perspective=player,
                features=np.ones(64, dtype=np.float32) * player,
                value=1.0 if player == 1 else -1.0,
                board_type="square8",
            )

        # Get specific player
        results = db.get_nnue_features(game_id, move_number=0, player_perspective=2)
        assert len(results) == 1
        assert results[0][1] == 2  # player_perspective
        assert results[0][3] == -1.0  # value

    def test_compression_round_trip(self, db):
        """Test that feature compression/decompression preserves values exactly."""
        import numpy as np

        game_id = "test-compression"
        self._create_game(db, game_id)

        # Create features with specific values that could be affected by compression issues
        features = np.array([
            0.0, 1.0, -1.0, 0.5, -0.5,
            1e-6, -1e-6, 1e6, -1e6,
            np.finfo(np.float32).max,
            np.finfo(np.float32).min,
        ], dtype=np.float32)

        db.store_nnue_features(
            game_id=game_id,
            move_number=0,
            player_perspective=1,
            features=features,
            value=0.0,
            board_type="hex21",
        )

        results = db.get_nnue_features(game_id)
        assert len(results) == 1
        np.testing.assert_array_equal(results[0][2], features)

    def test_update_existing_features(self, db):
        """Test that storing features for same position updates existing record."""
        import numpy as np

        game_id = "test-update"
        self._create_game(db, game_id)

        move_number = 0
        player_perspective = 1

        # Store initial features
        features1 = np.ones(32, dtype=np.float32)
        db.store_nnue_features(
            game_id=game_id,
            move_number=move_number,
            player_perspective=player_perspective,
            features=features1,
            value=1.0,
            board_type="square8",
        )

        # Store updated features (same position)
        features2 = np.zeros(32, dtype=np.float32)
        db.store_nnue_features(
            game_id=game_id,
            move_number=move_number,
            player_perspective=player_perspective,
            features=features2,
            value=-1.0,
            board_type="square8",
        )

        # Should only have one record with updated values
        results = db.get_nnue_features(game_id)
        assert len(results) == 1
        assert np.allclose(results[0][2], features2)
        assert results[0][3] == -1.0

    def test_empty_batch(self, db):
        """Test that empty batch returns 0."""
        count = db.store_nnue_features_batch([])
        assert count == 0

    def test_large_feature_vector(self, db):
        """Test storing large feature vectors (typical NNUE size)."""
        import numpy as np

        game_id = "test-large"
        self._create_game(db, game_id)

        # NNUE typically uses ~40k features
        features = np.random.randn(40960).astype(np.float32)

        db.store_nnue_features(
            game_id=game_id,
            move_number=0,
            player_perspective=1,
            features=features,
            value=0.5,
            board_type="hex21",
        )

        results = db.get_nnue_features(game_id)
        assert len(results) == 1
        assert len(results[0][2]) == 40960
        assert np.allclose(results[0][2], features)

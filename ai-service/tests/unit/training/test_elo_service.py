"""Tests for the centralized EloService.

Comprehensive test coverage for the Elo rating service including:
- Rating calculations and updates
- Match recording
- Leaderboard generation
- Training feedback signals
- Thread safety
- Backwards compatibility layer
"""

import math
import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.training.elo_service import (
    EloRating,
    EloService,
    LeaderboardEntry,
    MatchResult,
    TrainingFeedback,
    get_database_stats,
    get_elo_service,
    get_head_to_head,
    get_leaderboard,
    get_match_history,
    get_rating_history,
    init_elo_database,
    register_models,
    update_elo_after_match,
)


class TestEloRating:
    """Test EloRating dataclass."""

    def test_default_values(self):
        """Test default Elo rating values."""
        rating = EloRating(participant_id="test")
        assert rating.rating == 1500.0
        assert rating.games_played == 0
        assert rating.wins == 0
        assert rating.losses == 0
        assert rating.draws == 0
        assert rating.confidence == 0.0

    def test_win_rate_no_games(self):
        """Test win rate with no games returns 0.5."""
        rating = EloRating(participant_id="test")
        assert rating.win_rate == 0.5

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        rating = EloRating(
            participant_id="test",
            games_played=10,
            wins=6,
            losses=2,
            draws=2,
        )
        # win_rate = (wins + 0.5 * draws) / games = (6 + 1) / 10 = 0.7
        assert rating.win_rate == 0.7

    def test_win_rate_all_draws(self):
        """Test win rate with all draws."""
        rating = EloRating(
            participant_id="test",
            games_played=10,
            wins=0,
            losses=0,
            draws=10,
        )
        assert rating.win_rate == 0.5


class TestEloServiceBasics:
    """Test basic EloService functionality."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        # Cleanup
        if db_path.exists():
            db_path.unlink()

    @pytest.fixture
    def service(self, temp_db):
        """Create an EloService with temp database."""
        return EloService(db_path=temp_db, enforce_single_writer=False)

    def test_init_creates_tables(self, service, temp_db):
        """Test that initialization creates required tables."""
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "participants" in tables
        assert "elo_ratings" in tables
        assert "match_history" in tables
        assert "elo_history" in tables
        assert "training_feedback" in tables

    def test_register_participant(self, service):
        """Test participant registration."""
        service.register_participant(
            participant_id="ai_1",
            name="AI Player 1",
            ai_type="heuristic",
            difficulty=5,
        )
        # Verify by getting rating (which requires participant to exist)
        rating = service.get_rating("ai_1", "square8", 2)
        assert rating.participant_id == "ai_1"

    def test_register_model(self, service):
        """Test model registration."""
        service.register_model(
            model_id="model_v1",
            board_type="square8",
            num_players=2,
            model_path="/path/to/model.pth",
        )
        rating = service.get_rating("model_v1", "square8", 2)
        assert rating.participant_id == "model_v1"
        assert rating.rating == service.INITIAL_ELO

    def test_get_rating_creates_if_not_exists(self, service):
        """Test that get_rating creates initial rating if participant doesn't exist."""
        rating = service.get_rating("new_player", "square8", 2)
        assert rating.participant_id == "new_player"
        assert rating.rating == service.INITIAL_ELO
        assert rating.games_played == 0


class TestMatchRecording:
    """Test match recording and Elo updates."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        if db_path.exists():
            db_path.unlink()

    @pytest.fixture
    def service(self, temp_db):
        """Create an EloService with temp database."""
        svc = EloService(db_path=temp_db, enforce_single_writer=False)
        # Register two players
        svc.register_participant("player_a", "Player A", "neural_net")
        svc.register_participant("player_b", "Player B", "neural_net")
        # Initialize their ratings
        svc.get_rating("player_a", "square8", 2)
        svc.get_rating("player_b", "square8", 2)
        return svc

    def test_record_match_winner(self, service):
        """Test recording a match with a winner."""
        result = service.record_match(
            participant_a="player_a",
            participant_b="player_b",
            winner="player_a",
            board_type="square8",
            num_players=2,
        )

        assert result.winner_id == "player_a"
        assert "player_a" in result.elo_changes
        assert "player_b" in result.elo_changes
        # Winner gains Elo, loser loses
        assert result.elo_changes["player_a"] > 0
        assert result.elo_changes["player_b"] < 0

    def test_record_match_draw(self, service):
        """Test recording a draw."""
        result = service.record_match(
            participant_a="player_a",
            participant_b="player_b",
            winner=None,  # Draw
            board_type="square8",
            num_players=2,
        )

        assert result.winner_id is None
        # Both players should have minimal Elo change for equal-rated draw
        assert abs(result.elo_changes["player_a"]) < 1.0
        assert abs(result.elo_changes["player_b"]) < 1.0

    def test_elo_calculation_expected_outcome(self, service):
        """Test Elo calculation when higher-rated player wins (expected)."""
        # Give player_a a higher rating first
        for _ in range(5):
            service.record_match(
                participant_a="player_a",
                participant_b="player_b",
                winner="player_a",
                board_type="square8",
                num_players=2,
            )

        rating_a_before = service.get_rating("player_a", "square8", 2).rating
        rating_b_before = service.get_rating("player_b", "square8", 2).rating

        # Higher-rated player wins - should gain less Elo
        result = service.record_match(
            participant_a="player_a",
            participant_b="player_b",
            winner="player_a",
            board_type="square8",
            num_players=2,
        )

        # Winner gains less when expected to win
        expected_win_prob = 1.0 / (1.0 + math.pow(10, (rating_b_before - rating_a_before) / 400))
        assert expected_win_prob > 0.5  # player_a is expected to win
        # Elo gain should be proportional to unexpectedness
        assert result.elo_changes["player_a"] < service.K_FACTOR / 2

    def test_elo_calculation_upset(self, service):
        """Test Elo calculation when lower-rated player wins (upset)."""
        # Give player_a a higher rating first
        for _ in range(5):
            service.record_match(
                participant_a="player_a",
                participant_b="player_b",
                winner="player_a",
                board_type="square8",
                num_players=2,
            )

        # Lower-rated player wins - should gain more Elo
        result = service.record_match(
            participant_a="player_a",
            participant_b="player_b",
            winner="player_b",
            board_type="square8",
            num_players=2,
        )

        # Upset winner gains more Elo
        assert result.elo_changes["player_b"] > service.K_FACTOR / 2

    def test_games_played_increments(self, service):
        """Test that games_played counter increments."""
        service.record_match(
            participant_a="player_a",
            participant_b="player_b",
            winner="player_a",
            board_type="square8",
            num_players=2,
        )

        rating_a = service.get_rating("player_a", "square8", 2)
        rating_b = service.get_rating("player_b", "square8", 2)

        assert rating_a.games_played == 1
        assert rating_b.games_played == 1
        assert rating_a.wins == 1
        assert rating_b.losses == 1


class TestLeaderboard:
    """Test leaderboard functionality."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        if db_path.exists():
            db_path.unlink()

    @pytest.fixture
    def populated_service(self, temp_db):
        """Create a service with multiple players and matches."""
        svc = EloService(db_path=temp_db, enforce_single_writer=False)

        # Register players
        for i in range(5):
            svc.register_participant(f"player_{i}", f"Player {i}", "neural_net")
            svc.get_rating(f"player_{i}", "square8", 2)

        # Create some matches to establish rankings
        # Player 0 beats everyone
        for i in range(1, 5):
            svc.record_match(
                participant_a="player_0",
                participant_b=f"player_{i}",
                winner="player_0",
                board_type="square8",
                num_players=2,
            )

        # Player 1 beats players 2-4
        for i in range(2, 5):
            svc.record_match(
                participant_a="player_1",
                participant_b=f"player_{i}",
                winner="player_1",
                board_type="square8",
                num_players=2,
            )

        return svc

    def test_leaderboard_ordering(self, populated_service):
        """Test that leaderboard is ordered by rating descending."""
        leaderboard = populated_service.get_leaderboard("square8", 2)

        assert len(leaderboard) >= 2
        # First entry should have highest rating
        ratings = [entry.rating for entry in leaderboard]
        assert ratings == sorted(ratings, reverse=True)

    def test_leaderboard_rank(self, populated_service):
        """Test that ranks are assigned correctly."""
        leaderboard = populated_service.get_leaderboard("square8", 2)

        for i, entry in enumerate(leaderboard, 1):
            assert entry.rank == i

    def test_leaderboard_limit(self, populated_service):
        """Test leaderboard limit parameter."""
        leaderboard = populated_service.get_leaderboard("square8", 2, limit=2)
        assert len(leaderboard) == 2

    def test_leaderboard_min_games(self, populated_service):
        """Test leaderboard min_games filter."""
        # Players have varying game counts
        # min_games=10 should filter most out
        leaderboard = populated_service.get_leaderboard(
            "square8", 2, min_games=10
        )
        for entry in leaderboard:
            assert entry.games_played >= 10


class TestTrainingFeedback:
    """Test training feedback signals."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        if db_path.exists():
            db_path.unlink()

    @pytest.fixture
    def service(self, temp_db):
        """Create an EloService with temp database."""
        return EloService(db_path=temp_db, enforce_single_writer=False)

    def test_feedback_default_values(self, service):
        """Test feedback with no data returns defaults."""
        feedback = service.get_training_feedback("square8", 2)

        assert feedback.board_type == "square8"
        assert feedback.num_players == 2
        assert feedback.epochs_multiplier == 1.0
        assert feedback.lr_multiplier == 1.0
        assert feedback.elo_stagnating is False
        assert feedback.elo_declining is False

    def test_feedback_stagnation_detection(self, service):
        """Test that stagnation is detected when Elo doesn't change."""
        # Record multiple iterations with similar Elo
        service.register_participant("model", "Model", "neural_net")
        service.get_rating("model", "square8", 2)

        for i in range(6):
            service.record_iteration_elo("model", "square8", 2, iteration=i)

        feedback = service.get_training_feedback("square8", 2)
        # Should detect stagnation (no change over iterations)
        assert feedback.elo_stagnating is True
        assert feedback.epochs_multiplier > 1.0  # Recommends longer training

    def test_feedback_callback(self, service):
        """Test that feedback callbacks are invoked."""
        callback_received = []

        def callback(feedback):
            callback_received.append(feedback)

        service.register_feedback_callback(callback)
        service.get_training_feedback("square8", 2)

        assert len(callback_received) == 1
        assert isinstance(callback_received[0], TrainingFeedback)


class TestBackwardsCompatibility:
    """Test backwards compatibility layer functions."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        if db_path.exists():
            db_path.unlink()

    @pytest.fixture
    def service(self, temp_db):
        """Create an EloService with temp database."""
        return EloService(db_path=temp_db, enforce_single_writer=False)

    def test_init_elo_database(self, temp_db):
        """Test init_elo_database returns service."""
        # Reset singleton
        import app.training.elo_service as elo_module
        elo_module._elo_service_instance = None

        svc = init_elo_database(temp_db)
        assert isinstance(svc, EloService)

    def test_register_models(self, service):
        """Test register_models bulk registration."""
        models = [
            {"model_id": "m1", "board_type": "square8", "num_players": 2},
            {"model_id": "m2", "board_type": "square8", "num_players": 2},
        ]
        register_models(service, models)

        # Both models should have ratings
        r1 = service.get_rating("m1", "square8", 2)
        r2 = service.get_rating("m2", "square8", 2)
        assert r1.participant_id == "m1"
        assert r2.participant_id == "m2"

    def test_update_elo_after_match(self, service):
        """Test update_elo_after_match wrapper."""
        service.register_participant("a", "A", "neural_net")
        service.register_participant("b", "B", "neural_net")
        service.get_rating("a", "square8", 2)
        service.get_rating("b", "square8", 2)

        result = update_elo_after_match(
            service,
            model_a_id="a",
            model_b_id="b",
            winner="a",
            board_type="square8",
            num_players=2,
        )

        assert "model_a" in result
        assert "model_b" in result
        assert "changes" in result
        assert "match_id" in result

    def test_update_elo_after_match_draw(self, service):
        """Test update_elo_after_match with draw."""
        service.register_participant("a", "A", "neural_net")
        service.register_participant("b", "B", "neural_net")
        service.get_rating("a", "square8", 2)
        service.get_rating("b", "square8", 2)

        result = update_elo_after_match(
            service,
            model_a_id="a",
            model_b_id="b",
            winner="draw",
            board_type="square8",
            num_players=2,
        )

        # Draw should result in minimal changes
        assert abs(result["changes"]["a"]) < 1.0

    def test_get_leaderboard_compat(self, service):
        """Test get_leaderboard backwards compat wrapper."""
        service.register_participant("p1", "P1", "neural_net")
        service.get_rating("p1", "square8", 2)

        result = get_leaderboard(service, "square8", 2)
        assert isinstance(result, list)
        if result:
            assert isinstance(result[0], dict)

    def test_get_head_to_head(self, service):
        """Test get_head_to_head stats."""
        service.register_participant("a", "A", "neural_net")
        service.register_participant("b", "B", "neural_net")
        service.get_rating("a", "square8", 2)
        service.get_rating("b", "square8", 2)

        # Play some games
        service.record_match("a", "b", "a", "square8", 2)
        service.record_match("a", "b", "a", "square8", 2)
        service.record_match("a", "b", "b", "square8", 2)

        stats = get_head_to_head(service, "a", "b", "square8", 2)

        assert stats["total_games"] == 3
        assert stats["a_wins"] == 2
        assert stats["b_wins"] == 1
        assert stats["a_win_rate"] == pytest.approx(2/3, rel=0.01)

    def test_get_database_stats(self, service):
        """Test get_database_stats."""
        service.register_participant("p1", "P1", "neural_net")
        service.get_rating("p1", "square8", 2)

        stats = get_database_stats(service)

        assert "total_participants" in stats
        assert "total_matches" in stats
        assert "configurations" in stats

    def test_get_match_history(self, service):
        """Test get_match_history."""
        service.register_participant("a", "A", "neural_net")
        service.register_participant("b", "B", "neural_net")
        service.get_rating("a", "square8", 2)
        service.get_rating("b", "square8", 2)

        service.record_match("a", "b", "a", "square8", 2)

        history = get_match_history(service, participant_id="a")

        assert len(history) >= 1
        assert history[0]["winner_id"] == "a"

    def test_get_rating_history(self, service):
        """Test get_rating_history."""
        service.register_participant("model", "Model", "neural_net")
        service.get_rating("model", "square8", 2)

        service.record_iteration_elo("model", "square8", 2, iteration=1)
        service.record_iteration_elo("model", "square8", 2, iteration=2)

        history = get_rating_history(service, "model", "square8", 2)

        assert len(history) == 2


class TestThreadSafety:
    """Test thread-safe operations."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        if db_path.exists():
            db_path.unlink()

    def test_concurrent_match_recording(self, temp_db):
        """Test concurrent match recording."""
        service = EloService(db_path=temp_db, enforce_single_writer=False)

        # Register players
        for i in range(10):
            service.register_participant(f"p{i}", f"Player {i}", "neural_net")
            service.get_rating(f"p{i}", "square8", 2)

        errors = []
        results = []

        def record_matches(player_a, player_b):
            try:
                for _ in range(5):
                    result = service.record_match(
                        participant_a=player_a,
                        participant_b=player_b,
                        winner=player_a,
                        board_type="square8",
                        num_players=2,
                    )
                    results.append(result)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(0, 10, 2):
            t = threading.Thread(target=record_matches, args=(f"p{i}", f"p{i+1}"))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 25  # 5 threads * 5 matches each


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        if db_path.exists():
            db_path.unlink()

    @pytest.fixture
    def service(self, temp_db):
        """Create an EloService with temp database."""
        return EloService(db_path=temp_db, enforce_single_writer=False)

    def test_self_match(self, service):
        """Test match between same participant."""
        service.register_participant("p1", "Player 1", "neural_net")
        service.get_rating("p1", "square8", 2)

        # This is technically allowed but shouldn't break
        result = service.record_match(
            participant_a="p1",
            participant_b="p1",
            winner="p1",
            board_type="square8",
            num_players=2,
        )
        assert result.match_id is not None

    def test_execute_query(self, service):
        """Test execute_query for custom queries."""
        service.register_participant("p1", "Player 1", "neural_net")
        service.get_rating("p1", "square8", 2)

        rows = service.execute_query(
            "SELECT * FROM participants WHERE id = ?",
            ("p1",)
        )

        assert len(rows) == 1
        assert rows[0]["id"] == "p1"

    def test_check_write_permission_no_coordinator(self, service):
        """Test check_write_permission without coordinator."""
        can_write, _reason = service.check_write_permission()
        assert can_write is True

    def test_different_configs_independent(self, service):
        """Test that different board configs have independent ratings."""
        service.register_participant("model", "Model", "neural_net")

        # Get ratings for two different configs
        rating_8x8 = service.get_rating("model", "square8", 2)
        rating_19x19 = service.get_rating("model", "square19", 2)

        # Both should be initial
        assert rating_8x8.rating == service.INITIAL_ELO
        assert rating_19x19.rating == service.INITIAL_ELO

        # Record match only for 8x8
        service.record_match(
            participant_a="model",
            participant_b="model",
            winner="model",
            board_type="square8",
            num_players=2,
        )

        # 8x8 should have games, 19x19 should not
        rating_8x8 = service.get_rating("model", "square8", 2)
        rating_19x19 = service.get_rating("model", "square19", 2)

        assert rating_8x8.games_played == 2  # Counted for both sides
        assert rating_19x19.games_played == 0

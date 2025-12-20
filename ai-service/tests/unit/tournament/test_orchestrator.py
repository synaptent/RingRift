"""Tests for TournamentOrchestrator (high-level tournament management).

Tests cover:
- EvaluationResult dataclass
- TournamentSummary dataclass
- TournamentOrchestrator initialization and methods
- Module functions
"""

import time
from dataclasses import dataclass
from datetime import datetime
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

# =============================================================================
# Test EvaluationResult Dataclass
# =============================================================================

class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_create_passing_result(self):
        """Test creating a passing evaluation result."""
        from app.tournament.orchestrator import EvaluationResult

        result = EvaluationResult(
            candidate_id="model_v2",
            baseline_id="model_v1",
            elo_delta=50.0,
            win_rate=0.65,
            games_played=100,
            passed=True,
        )

        assert result.candidate_id == "model_v2"
        assert result.baseline_id == "model_v1"
        assert result.elo_delta == 50.0
        assert result.win_rate == 0.65
        assert result.games_played == 100
        assert result.passed is True

    def test_create_failing_result(self):
        """Test creating a failing evaluation result."""
        from app.tournament.orchestrator import EvaluationResult

        result = EvaluationResult(
            candidate_id="model_v2",
            baseline_id="model_v1",
            elo_delta=-30.0,
            win_rate=0.40,
            games_played=50,
            passed=False,
        )

        assert result.passed is False
        assert result.elo_delta == -30.0

    def test_default_values(self):
        """Test default values."""
        from app.tournament.orchestrator import EvaluationResult

        result = EvaluationResult(
            candidate_id="test",
            baseline_id="baseline",
            elo_delta=0.0,
            win_rate=0.5,
            games_played=10,
            passed=False,
        )

        assert result.confidence == 0.0
        assert result.details == {}

    def test_with_confidence(self):
        """Test result with confidence score."""
        from app.tournament.orchestrator import EvaluationResult

        result = EvaluationResult(
            candidate_id="test",
            baseline_id="baseline",
            elo_delta=25.0,
            win_rate=0.55,
            games_played=200,
            passed=True,
            confidence=0.85,
        )

        assert result.confidence == 0.85

    def test_with_details(self):
        """Test result with additional details."""
        from app.tournament.orchestrator import EvaluationResult

        result = EvaluationResult(
            candidate_id="test",
            baseline_id="baseline",
            elo_delta=25.0,
            win_rate=0.55,
            games_played=50,
            passed=True,
            details={"head_to_head_wins": 30, "draws": 5},
        )

        assert result.details["head_to_head_wins"] == 30
        assert result.details["draws"] == 5


# =============================================================================
# Test TournamentSummary Dataclass
# =============================================================================

class TestTournamentSummary:
    """Tests for TournamentSummary dataclass."""

    def test_create_summary(self):
        """Test creating a tournament summary."""
        from app.tournament.orchestrator import TournamentSummary

        started = datetime.now()
        completed = datetime.now()

        summary = TournamentSummary(
            tournament_id="roundrobin_abc123",
            board_type="square8",
            num_players=2,
            started_at=started,
            completed_at=completed,
            total_games=100,
            duration_seconds=60.5,
            final_ratings={"agent_a": 1550, "agent_b": 1450},
            win_rates={"agent_a": 0.6, "agent_b": 0.4},
            agent_stats={"agent_a": {"wins": 60}, "agent_b": {"wins": 40}},
        )

        assert summary.tournament_id == "roundrobin_abc123"
        assert summary.board_type == "square8"
        assert summary.num_players == 2
        assert summary.total_games == 100
        assert summary.final_ratings["agent_a"] == 1550
        assert summary.win_rates["agent_b"] == 0.4

    def test_default_evaluation_results(self):
        """Test default evaluation results is empty list."""
        from app.tournament.orchestrator import TournamentSummary

        summary = TournamentSummary(
            tournament_id="test",
            board_type="square8",
            num_players=2,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            total_games=10,
            duration_seconds=5.0,
            final_ratings={},
            win_rates={},
            agent_stats={},
        )

        assert summary.evaluation_results == []

    def test_with_evaluation_results(self):
        """Test summary with evaluation results."""
        from app.tournament.orchestrator import EvaluationResult, TournamentSummary

        eval_result = EvaluationResult(
            candidate_id="new_model",
            baseline_id="old_model",
            elo_delta=30.0,
            win_rate=0.58,
            games_played=100,
            passed=True,
        )

        summary = TournamentSummary(
            tournament_id="test",
            board_type="square8",
            num_players=2,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            total_games=100,
            duration_seconds=30.0,
            final_ratings={"new_model": 1530, "old_model": 1500},
            win_rates={"new_model": 0.58},
            agent_stats={},
            evaluation_results=[eval_result],
        )

        assert len(summary.evaluation_results) == 1
        assert summary.evaluation_results[0].passed is True


# =============================================================================
# Test TournamentOrchestrator
# =============================================================================

class TestTournamentOrchestrator:
    """Tests for TournamentOrchestrator class."""

    def test_initialization_defaults(self):
        """Test orchestrator initialization with defaults."""
        from app.tournament.orchestrator import TournamentOrchestrator

        orch = TournamentOrchestrator()

        assert orch.board_type == "square8"
        assert orch.num_players == 2
        assert orch.max_workers == 4
        assert orch.persist_to_elo_db is True
        assert orch.record_metrics is True

    def test_initialization_custom(self):
        """Test orchestrator initialization with custom values."""
        from app.tournament.orchestrator import TournamentOrchestrator

        orch = TournamentOrchestrator(
            board_type="hex6",
            num_players=3,
            max_workers=8,
            persist_to_elo_db=False,
            record_metrics=False,
        )

        assert orch.board_type == "hex6"
        assert orch.num_players == 3
        assert orch.max_workers == 8
        assert orch.persist_to_elo_db is False
        assert orch.record_metrics is False

    def test_runner_lazy_load(self):
        """Test runner is lazy loaded."""
        from app.tournament.orchestrator import TournamentOrchestrator

        orch = TournamentOrchestrator()
        assert orch._runner is None

    def test_elo_db_lazy_load(self):
        """Test elo_db is lazy loaded."""
        from app.tournament.orchestrator import TournamentOrchestrator

        orch = TournamentOrchestrator()
        assert orch._elo_db is None

    def test_compute_confidence_low_games(self):
        """Test confidence computation with few games."""
        from app.tournament.orchestrator import TournamentOrchestrator

        orch = TournamentOrchestrator()
        confidence = orch._compute_confidence(10, 20.0)

        # 10 games / 100 = 0.1 game factor
        # 20 elo / 100 = 0.2 elo factor
        # (0.1 + 0.2) / 2 = 0.15
        assert confidence == pytest.approx(0.15, abs=0.01)

    def test_compute_confidence_high_games(self):
        """Test confidence computation with many games."""
        from app.tournament.orchestrator import TournamentOrchestrator

        orch = TournamentOrchestrator()
        confidence = orch._compute_confidence(200, 150.0)

        # 200 games capped at 100/100 = 1.0 game factor
        # 150 elo capped at 100/100 = 1.0 elo factor
        # (1.0 + 1.0) / 2 = 1.0
        assert confidence == pytest.approx(1.0, abs=0.01)

    def test_compute_confidence_medium(self):
        """Test confidence computation with medium values."""
        from app.tournament.orchestrator import TournamentOrchestrator

        orch = TournamentOrchestrator()
        confidence = orch._compute_confidence(50, 50.0)

        # 50/100 = 0.5, 50/100 = 0.5, avg = 0.5
        assert confidence == pytest.approx(0.5, abs=0.01)


# =============================================================================
# Test TournamentOrchestrator Methods with Mocking
# =============================================================================

class TestTournamentOrchestratorMocked:
    """Tests for TournamentOrchestrator methods with mocked dependencies."""

    @pytest.fixture
    def mock_runner(self):
        """Create a mock tournament runner."""
        mock = MagicMock()
        mock.run_tournament.return_value = MagicMock(
            match_results=[MagicMock() for _ in range(100)],
            final_ratings={"agent_a": 1550, "agent_b": 1450},
            agent_stats={
                "agent_a": {"win_rate": 0.6},
                "agent_b": {"win_rate": 0.4},
            },
            compute_stats=MagicMock(),
        )
        return mock

    @pytest.fixture
    def orchestrator(self, mock_runner):
        """Create orchestrator with mocked runner."""
        from app.tournament.orchestrator import TournamentOrchestrator

        orch = TournamentOrchestrator(record_metrics=False)
        orch._runner = mock_runner
        return orch

    def test_run_round_robin(self, orchestrator, mock_runner):
        """Test run_round_robin method."""
        summary = orchestrator.run_round_robin(
            agents=["agent_a", "agent_b"],
            games_per_pairing=50,
        )

        assert summary.tournament_id.startswith("roundrobin_")
        assert summary.board_type == "square8"
        assert summary.total_games == 100
        mock_runner.run_tournament.assert_called_once()

    def test_run_round_robin_with_callback(self, orchestrator, mock_runner):
        """Test run_round_robin with progress callback."""
        callback = MagicMock()

        orchestrator.run_round_robin(
            agents=["a", "b", "c"],
            games_per_pairing=20,
            progress_callback=callback,
        )

        mock_runner.run_tournament.assert_called_once()
        call_kwargs = mock_runner.run_tournament.call_args[1]
        assert call_kwargs["progress_callback"] == callback

    def test_run_evaluation(self, orchestrator, mock_runner):
        """Test run_evaluation method."""
        mock_runner.run_tournament.return_value.final_ratings = {
            "candidate": 1550,
            "baseline1": 1500,
            "baseline2": 1480,
        }
        mock_runner.run_tournament.return_value.agent_stats = {
            "candidate": {"win_rate": 0.6},
            "baseline1": {"win_rate": 0.45},
            "baseline2": {"win_rate": 0.4},
        }

        _summary, eval_results = orchestrator.run_evaluation(
            candidate_model="candidate",
            baseline_models=["baseline1", "baseline2"],
            games_per_pairing=30,
            elo_threshold=25.0,
        )

        assert len(eval_results) == 2
        assert eval_results[0].candidate_id == "candidate"
        assert eval_results[0].baseline_id == "baseline1"
        assert eval_results[0].elo_delta == 50.0  # 1550 - 1500
        assert eval_results[0].passed is True  # 50 > 25 threshold

    def test_run_evaluation_failing(self, orchestrator, mock_runner):
        """Test run_evaluation with failing candidate."""
        mock_runner.run_tournament.return_value.final_ratings = {
            "candidate": 1480,
            "baseline": 1520,
        }
        mock_runner.run_tournament.return_value.agent_stats = {
            "candidate": {"win_rate": 0.4},
            "baseline": {"win_rate": 0.6},
        }

        _summary, eval_results = orchestrator.run_evaluation(
            candidate_model="candidate",
            baseline_models=["baseline"],
            games_per_pairing=20,
            elo_threshold=25.0,
        )

        assert eval_results[0].elo_delta == -40.0  # 1480 - 1520
        assert eval_results[0].passed is False

    def test_run_shadow_eval_passing(self, orchestrator, mock_runner):
        """Test run_shadow_eval with passing candidate."""
        mock_runner.run_tournament.return_value.final_ratings = {
            "candidate": 1550,
            "heuristic": 1500,
            "mcts_100": 1520,
        }
        mock_runner.run_tournament.return_value.agent_stats = {
            "candidate": {"win_rate": 0.6},
            "heuristic": {"win_rate": 0.4},
            "mcts_100": {"win_rate": 0.45},
        }

        passed = orchestrator.run_shadow_eval(
            candidate="candidate",
            games=15,
            elo_threshold=25.0,
        )

        assert passed is True

    def test_run_shadow_eval_custom_baselines(self, orchestrator, mock_runner):
        """Test run_shadow_eval with custom baselines."""
        mock_runner.run_tournament.return_value.final_ratings = {
            "candidate": 1600,
            "random": 1400,
            "custom_baseline": 1500,
        }
        mock_runner.run_tournament.return_value.agent_stats = {
            "candidate": {"win_rate": 0.7},
        }

        passed = orchestrator.run_shadow_eval(
            candidate="candidate",
            games=10,
            baselines=["random", "custom_baseline"],
            elo_threshold=50.0,
        )

        assert passed is True


# =============================================================================
# Test Elo Database Methods
# =============================================================================

class TestEloDbMethods:
    """Tests for Elo database interaction methods."""

    @pytest.fixture
    def mock_elo_db(self):
        """Create a mock Elo database."""
        mock = MagicMock()
        mock_rating = MagicMock()
        mock_rating.elo = 1525.0
        mock_rating.uncertainty = 50.0
        mock_rating.games_played = 100
        mock.get_rating.return_value = mock_rating
        mock.get_ratings_for_config.return_value = [
            MagicMock(agent_id="agent_a", elo=1600, uncertainty=40, games_played=200),
            MagicMock(agent_id="agent_b", elo=1550, uncertainty=45, games_played=150),
            MagicMock(agent_id="agent_c", elo=1500, uncertainty=50, games_played=100),
        ]
        return mock

    @pytest.fixture
    def orchestrator(self, mock_elo_db):
        """Create orchestrator with mocked elo_db."""
        from app.tournament.orchestrator import TournamentOrchestrator

        orch = TournamentOrchestrator()
        orch._elo_db = mock_elo_db
        return orch

    def test_get_current_elo(self, orchestrator, mock_elo_db):
        """Test getting current Elo rating."""
        rating = orchestrator.get_current_elo("test_agent")

        assert rating == 1525.0
        mock_elo_db.get_rating.assert_called_once_with(
            agent_id="test_agent",
            board_type="square8",
            num_players=2,
        )

    def test_get_current_elo_not_found(self, orchestrator, mock_elo_db):
        """Test getting Elo for non-existent agent."""
        mock_elo_db.get_rating.return_value = None

        rating = orchestrator.get_current_elo("unknown_agent")

        assert rating is None

    def test_get_current_elo_handles_exception(self, orchestrator, mock_elo_db):
        """Test get_current_elo handles exceptions."""
        mock_elo_db.get_rating.side_effect = Exception("DB error")

        rating = orchestrator.get_current_elo("test_agent")

        assert rating is None

    def test_get_leaderboard(self, orchestrator, mock_elo_db):
        """Test getting leaderboard."""
        leaderboard = orchestrator.get_leaderboard(top_n=3)

        assert len(leaderboard) == 3
        assert leaderboard[0]["agent_id"] == "agent_a"
        assert leaderboard[0]["elo"] == 1600
        assert leaderboard[1]["agent_id"] == "agent_b"

    def test_get_leaderboard_limited(self, orchestrator, mock_elo_db):
        """Test getting limited leaderboard."""
        leaderboard = orchestrator.get_leaderboard(top_n=2)

        assert len(leaderboard) == 2

    def test_get_leaderboard_handles_exception(self, orchestrator, mock_elo_db):
        """Test get_leaderboard handles exceptions."""
        mock_elo_db.get_ratings_for_config.side_effect = Exception("DB error")

        leaderboard = orchestrator.get_leaderboard()

        assert leaderboard == []


# =============================================================================
# Test Module Functions
# =============================================================================

class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_run_quick_evaluation_defaults(self):
        """Test run_quick_evaluation with defaults."""
        from app.tournament.orchestrator import run_quick_evaluation

        with patch("app.tournament.orchestrator.TournamentOrchestrator") as MockOrch:
            mock_orch = MagicMock()
            mock_orch.run_evaluation.return_value = (
                MagicMock(),
                [
                    MagicMock(passed=True, elo_delta=30),
                    MagicMock(passed=True, elo_delta=40),
                ],
            )
            MockOrch.return_value = mock_orch

            passed, avg_delta = run_quick_evaluation("candidate")

            assert passed is True
            assert avg_delta == 35.0

    def test_run_quick_evaluation_custom_baselines(self):
        """Test run_quick_evaluation with custom baselines."""
        from app.tournament.orchestrator import run_quick_evaluation

        with patch("app.tournament.orchestrator.TournamentOrchestrator") as MockOrch:
            mock_orch = MagicMock()
            mock_orch.run_evaluation.return_value = (
                MagicMock(),
                [MagicMock(passed=True, elo_delta=50)],
            )
            MockOrch.return_value = mock_orch

            run_quick_evaluation(
                "candidate",
                baselines=["custom_baseline"],
                board_type="hex6",
                num_players=3,
            )

            MockOrch.assert_called_once_with(
                board_type="hex6",
                num_players=3,
            )

    def test_run_quick_evaluation_failing(self):
        """Test run_quick_evaluation with failing candidate."""
        from app.tournament.orchestrator import run_quick_evaluation

        with patch("app.tournament.orchestrator.TournamentOrchestrator") as MockOrch:
            mock_orch = MagicMock()
            mock_orch.run_evaluation.return_value = (
                MagicMock(),
                [
                    MagicMock(passed=True, elo_delta=30),
                    MagicMock(passed=False, elo_delta=-10),
                ],
            )
            MockOrch.return_value = mock_orch

            passed, avg_delta = run_quick_evaluation("candidate")

            assert passed is False
            assert avg_delta == 10.0  # (30 + -10) / 2

    def test_run_elo_calibration(self):
        """Test run_elo_calibration function."""
        from app.tournament.orchestrator import run_elo_calibration

        with patch("app.tournament.orchestrator.TournamentOrchestrator") as MockOrch:
            mock_orch = MagicMock()
            mock_summary = MagicMock()
            mock_summary.final_ratings = {
                "agent_a": 1550,
                "agent_b": 1500,
                "agent_c": 1450,
            }
            mock_orch.run_round_robin.return_value = mock_summary
            MockOrch.return_value = mock_orch

            ratings = run_elo_calibration(
                agents=["agent_a", "agent_b", "agent_c"],
                games_per_pairing=50,
            )

            assert ratings["agent_a"] == 1550
            assert ratings["agent_b"] == 1500
            assert ratings["agent_c"] == 1450

    def test_run_elo_calibration_custom_config(self):
        """Test run_elo_calibration with custom board config."""
        from app.tournament.orchestrator import run_elo_calibration

        with patch("app.tournament.orchestrator.TournamentOrchestrator") as MockOrch:
            mock_orch = MagicMock()
            mock_orch.run_round_robin.return_value = MagicMock(final_ratings={})
            MockOrch.return_value = mock_orch

            run_elo_calibration(
                agents=["a", "b"],
                board_type="hex6",
                num_players=4,
                games_per_pairing=100,
            )

            MockOrch.assert_called_once_with(
                board_type="hex6",
                num_players=4,
            )


# =============================================================================
# Integration Tests
# =============================================================================

class TestTournamentIntegration:
    """Integration tests for tournament orchestrator."""

    def test_evaluation_result_attributes(self):
        """Test EvaluationResult has all expected attributes."""
        from app.tournament.orchestrator import EvaluationResult

        result = EvaluationResult(
            candidate_id="test",
            baseline_id="baseline",
            elo_delta=0.0,
            win_rate=0.5,
            games_played=10,
            passed=False,
        )

        assert hasattr(result, "candidate_id")
        assert hasattr(result, "baseline_id")
        assert hasattr(result, "elo_delta")
        assert hasattr(result, "win_rate")
        assert hasattr(result, "games_played")
        assert hasattr(result, "passed")
        assert hasattr(result, "confidence")
        assert hasattr(result, "details")

    def test_tournament_summary_attributes(self):
        """Test TournamentSummary has all expected attributes."""
        from app.tournament.orchestrator import TournamentSummary

        summary = TournamentSummary(
            tournament_id="test",
            board_type="square8",
            num_players=2,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            total_games=100,
            duration_seconds=60.0,
            final_ratings={},
            win_rates={},
            agent_stats={},
        )

        assert hasattr(summary, "tournament_id")
        assert hasattr(summary, "board_type")
        assert hasattr(summary, "num_players")
        assert hasattr(summary, "started_at")
        assert hasattr(summary, "completed_at")
        assert hasattr(summary, "total_games")
        assert hasattr(summary, "duration_seconds")
        assert hasattr(summary, "final_ratings")
        assert hasattr(summary, "win_rates")
        assert hasattr(summary, "agent_stats")
        assert hasattr(summary, "evaluation_results")

    def test_orchestrator_has_expected_methods(self):
        """Test TournamentOrchestrator has expected methods."""
        from app.tournament.orchestrator import TournamentOrchestrator

        orch = TournamentOrchestrator()

        assert hasattr(orch, "run_round_robin")
        assert hasattr(orch, "run_evaluation")
        assert hasattr(orch, "run_shadow_eval")
        assert hasattr(orch, "get_current_elo")
        assert hasattr(orch, "get_leaderboard")
        assert callable(orch.run_round_robin)
        assert callable(orch.run_evaluation)

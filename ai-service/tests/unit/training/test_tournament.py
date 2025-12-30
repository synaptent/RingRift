"""Tests for app.training.tournament.

Tests the tournament system for evaluating AI models including:
- infer_victory_reason for determining game outcomes
- Tournament class for running model-vs-model matches
- run_tournament convenience function
- run_tournament_adaptive for early stopping
- Elo rating calculations
- Victory reason tracking
"""

from unittest.mock import MagicMock, patch

import pytest

from app.models import BoardType, GameStatus
from app.training.tournament import (
    VICTORY_REASONS,
    Tournament,
    infer_victory_reason,
    run_tournament,
    run_tournament_adaptive,
)


class TestVictoryReasons:
    """Tests for VICTORY_REASONS constant."""

    def test_victory_reasons_list(self):
        """Test VICTORY_REASONS contains expected values."""
        assert "ring_elimination" in VICTORY_REASONS
        assert "territory" in VICTORY_REASONS
        assert "last_player_standing" in VICTORY_REASONS
        assert "structural" in VICTORY_REASONS
        assert "unknown" in VICTORY_REASONS
        assert len(VICTORY_REASONS) == 5


class TestInferVictoryReason:
    """Tests for infer_victory_reason function."""

    def _create_mock_game_state(
        self,
        winner: int | None = 1,
        eliminated_rings: dict | None = None,
        collapsed_spaces: dict | None = None,
        lps_exclusive: int | None = None,
        stacks: dict | None = None,
        status: GameStatus = GameStatus.COMPLETED,
        victory_threshold: int = 10,
        territory_threshold: int = 30,
    ) -> MagicMock:
        """Helper to create a mock game state for testing."""
        if eliminated_rings is None:
            eliminated_rings = {}
        if collapsed_spaces is None:
            collapsed_spaces = {}
        if stacks is None:
            stacks = {"0,0": MagicMock()}

        state = MagicMock()
        state.game_status = status
        state.winner = winner
        state.victory_threshold = victory_threshold
        state.territory_victory_threshold = territory_threshold
        state.lps_exclusive_player_for_completed_round = lps_exclusive

        # Board mock
        state.board = MagicMock()
        state.board.eliminated_rings = eliminated_rings
        state.board.collapsed_spaces = collapsed_spaces
        state.board.stacks = stacks

        return state

    def test_unknown_for_incomplete_game(self):
        """Test returns 'unknown' for non-completed game."""
        state = self._create_mock_game_state(status=GameStatus.ACTIVE)

        result = infer_victory_reason(state)

        assert result == "unknown"

    def test_unknown_for_no_winner(self):
        """Test returns 'unknown' when winner is None."""
        state = self._create_mock_game_state(winner=None)

        result = infer_victory_reason(state)

        assert result == "unknown"

    def test_ring_elimination_victory(self):
        """Test detects ring elimination victory."""
        state = self._create_mock_game_state(
            winner=1,
            eliminated_rings={"1": 10},  # Winner eliminated 10 rings
            victory_threshold=10,
        )

        result = infer_victory_reason(state)

        assert result == "ring_elimination"

    def test_territory_victory(self):
        """Test detects territory victory."""
        # Create enough collapsed spaces for territory victory
        collapsed = {f"{i},0": 1 for i in range(30)}
        state = self._create_mock_game_state(
            winner=1,
            eliminated_rings={"1": 0},  # No ring elimination
            collapsed_spaces=collapsed,
            territory_threshold=30,
        )

        result = infer_victory_reason(state)

        assert result == "territory"

    def test_lps_victory(self):
        """Test detects last player standing (LPS) victory."""
        state = self._create_mock_game_state(
            winner=1,
            eliminated_rings={"1": 0},
            collapsed_spaces={},
            lps_exclusive=1,  # Winner is LPS exclusive
        )

        result = infer_victory_reason(state)

        assert result == "last_player_standing"

    def test_structural_victory(self):
        """Test detects structural termination victory."""
        state = self._create_mock_game_state(
            winner=1,
            eliminated_rings={"1": 0},
            collapsed_spaces={},
            stacks={},  # No stacks remaining
        )

        result = infer_victory_reason(state)

        assert result == "structural"


class TestTournament:
    """Tests for Tournament class."""

    def test_initialization(self):
        """Test Tournament initializes with correct defaults."""
        tournament = Tournament(
            model_path_a="models/a.pth",
            model_path_b="models/b.pth",
        )

        assert tournament.model_path_a == "models/a.pth"
        assert tournament.model_path_b == "models/b.pth"
        assert tournament.num_games == 20
        assert tournament.num_players == 2
        assert tournament.board_type == BoardType.SQUARE8
        assert tournament.max_moves == 10000
        assert tournament.k_elo == 32
        assert tournament.results == {"A": 0, "B": 0, "Draw": 0}
        assert tournament.ratings == {"A": 1500.0, "B": 1500.0}

    def test_initialization_custom_values(self):
        """Test Tournament with custom values."""
        tournament = Tournament(
            model_path_a="models/a.pth",
            model_path_b="models/b.pth",
            num_games=50,
            k_elo=16,
            board_type=BoardType.HEXAGONAL,
            num_players=4,
            max_moves=5000,
        )

        assert tournament.num_games == 50
        assert tournament.k_elo == 16
        assert tournament.board_type == BoardType.HEXAGONAL
        assert tournament.num_players == 4
        assert tournament.max_moves == 5000

    def test_victory_reasons_initialized(self):
        """Test victory_reasons dict is initialized."""
        tournament = Tournament("a.pth", "b.pth")

        for reason in VICTORY_REASONS:
            assert reason in tournament.victory_reasons
            assert tournament.victory_reasons[reason] == 0

    def test_update_elo_a_wins(self):
        """Test Elo update when A wins."""
        tournament = Tournament("a.pth", "b.pth")
        initial_a = tournament.ratings["A"]
        initial_b = tournament.ratings["B"]

        tournament._update_elo("A")

        # A should gain rating, B should lose
        assert tournament.ratings["A"] > initial_a
        assert tournament.ratings["B"] < initial_b

    def test_update_elo_b_wins(self):
        """Test Elo update when B wins."""
        tournament = Tournament("a.pth", "b.pth")
        initial_a = tournament.ratings["A"]
        initial_b = tournament.ratings["B"]

        tournament._update_elo("B")

        # B should gain rating, A should lose
        assert tournament.ratings["B"] > initial_b
        assert tournament.ratings["A"] < initial_a

    def test_update_elo_draw(self):
        """Test Elo update for draw."""
        tournament = Tournament("a.pth", "b.pth")
        initial_a = tournament.ratings["A"]
        initial_b = tournament.ratings["B"]

        tournament._update_elo(None)

        # Equal ratings, draw should leave ratings unchanged
        assert abs(tournament.ratings["A"] - initial_a) < 0.1
        assert abs(tournament.ratings["B"] - initial_b) < 0.1

    def test_update_elo_conserves_sum(self):
        """Test Elo update conserves rating sum."""
        tournament = Tournament("a.pth", "b.pth")
        initial_sum = tournament.ratings["A"] + tournament.ratings["B"]

        tournament._update_elo("A")
        tournament._update_elo("B")
        tournament._update_elo(None)

        final_sum = tournament.ratings["A"] + tournament.ratings["B"]
        assert abs(final_sum - initial_sum) < 0.01

    def test_create_initial_state_2p(self):
        """Test _create_initial_state for 2 players."""
        tournament = Tournament("a.pth", "b.pth", num_players=2)

        state = tournament._create_initial_state()

        assert len(state.players) == 2
        assert state.game_status == GameStatus.ACTIVE
        assert state.board_type == BoardType.SQUARE8

    def test_create_initial_state_4p(self):
        """Test _create_initial_state for 4 players."""
        tournament = Tournament(
            "a.pth", "b.pth",
            num_players=4,
            board_type=BoardType.SQUARE8,
        )

        state = tournament._create_initial_state()

        assert len(state.players) == 4
        assert state.max_players == 4

    def test_create_initial_state_different_seeds(self):
        """Test _create_initial_state uses different seeds per game."""
        tournament = Tournament("a.pth", "b.pth")

        state1 = tournament._create_initial_state(game_idx=0)
        state2 = tournament._create_initial_state(game_idx=1)

        assert state1.rng_seed != state2.rng_seed

    def test_determine_winner_by_tiebreaker(self):
        """Test tiebreaker winner determination."""
        tournament = Tournament("a.pth", "b.pth", num_players=2)

        # Create mock state with players
        state = MagicMock()
        player1 = MagicMock()
        player1.player_number = 1
        player1.rings_in_hand = 5
        player2 = MagicMock()
        player2.player_number = 2
        player2.rings_in_hand = 8

        state.players = [player1, player2]
        state.board = MagicMock()
        state.board.eliminated_rings = {"1": 5, "2": 3}
        state.board.collapsed_spaces = {"0,0": 1, "1,0": 1, "2,0": 2}

        winner = tournament._determine_winner_by_tiebreaker(state)

        # Player 1 has more eliminated rings (5 vs 3), so should win
        assert winner == 1

    @patch('app.training.tournament.DescentAI')
    @patch('app.training.tournament.GameEngine')
    def test_play_game_multiplayer_returns_state(self, mock_engine, mock_ai_class):
        """Test _play_game_multiplayer returns winner and state."""
        tournament = Tournament("a.pth", "b.pth", num_players=2)

        # Create mock AI that returns moves
        mock_ai = MagicMock()
        mock_move = MagicMock()
        mock_ai.select_move.return_value = mock_move

        # Create completed state
        final_state = MagicMock()
        final_state.game_status = GameStatus.COMPLETED
        final_state.winner = 1

        mock_engine.apply_move.return_value = final_state

        ais = {1: mock_ai, 2: mock_ai}
        winner, state = tournament._play_game_multiplayer(ais)

        assert state is not None
        assert state.game_status == GameStatus.COMPLETED

    @patch('app.training.tournament.DescentAI')
    def test_create_ai_uses_model_path(self, mock_ai_class):
        """Test _create_ai passes model path to AI."""
        tournament = Tournament("a.pth", "b.pth")

        ai = tournament._create_ai(player_number=1, model_path="custom/model.pth")

        # Verify AI was created with config containing model path
        mock_ai_class.assert_called_once()
        call_args = mock_ai_class.call_args
        config = call_args[0][1]  # Second positional arg
        assert config.nn_model_id == "custom/model.pth"

    @patch('app.training.tournament.DescentAI')
    def test_create_ai_unique_seeds(self, mock_ai_class):
        """Test _create_ai uses different seeds per game/player."""
        tournament = Tournament("a.pth", "b.pth")

        tournament._create_ai(1, "a.pth", game_idx=0)
        call1 = mock_ai_class.call_args[0][1].rng_seed

        mock_ai_class.reset_mock()
        tournament._create_ai(1, "a.pth", game_idx=1)
        call2 = mock_ai_class.call_args[0][1].rng_seed

        assert call1 != call2


class TestRunTournament:
    """Tests for run_tournament convenience function."""

    @patch('app.training.tournament.Tournament')
    def test_creates_tournament(self, mock_tournament_class):
        """Test run_tournament creates Tournament instance."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = {"A": 10, "B": 8, "Draw": 2}
        mock_instance.victory_reasons = dict.fromkeys(VICTORY_REASONS, 0)
        mock_instance.ratings = {"A": 1520, "B": 1480}
        mock_tournament_class.return_value = mock_instance

        result = run_tournament("a.pth", "b.pth", num_games=20)

        mock_tournament_class.assert_called_once()
        assert result["model_a_wins"] == 10
        assert result["model_b_wins"] == 8
        assert result["draws"] == 2

    @patch('app.training.tournament.Tournament')
    def test_validates_player_count(self, mock_tournament_class):
        """Test run_tournament clamps player count."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = {"A": 0, "B": 0, "Draw": 0}
        mock_instance.victory_reasons = {}
        mock_instance.ratings = {}
        mock_tournament_class.return_value = mock_instance

        # Too low
        run_tournament("a.pth", "b.pth", num_players=1)
        call_args = mock_tournament_class.call_args[1]
        assert call_args["num_players"] == 2

        mock_tournament_class.reset_mock()

        # Too high
        run_tournament("a.pth", "b.pth", num_players=10)
        call_args = mock_tournament_class.call_args[1]
        assert call_args["num_players"] == 4

    @patch('app.training.tournament.Tournament')
    def test_returns_results_dict(self, mock_tournament_class):
        """Test run_tournament returns proper results dict."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = {"A": 15, "B": 5, "Draw": 0}
        mock_instance.victory_reasons = {"ring_elimination": 20, "territory": 0}
        mock_instance.ratings = {"A": 1550, "B": 1450}
        mock_tournament_class.return_value = mock_instance

        result = run_tournament(
            "a.pth", "b.pth",
            num_games=20,
            board_type=BoardType.HEXAGONAL,
            num_players=3,
        )

        assert "model_a_wins" in result
        assert "model_b_wins" in result
        assert "draws" in result
        assert "total_games" in result
        assert "victory_reasons" in result
        assert "board_type" in result
        assert "num_players" in result
        assert "elo_ratings" in result


class TestRunTournamentAdaptive:
    """Tests for run_tournament_adaptive function."""

    @patch('app.training.tournament.Tournament')
    @patch('app.training.significance.wilson_score_interval')
    def test_obvious_winner_early_stop(self, mock_wilson, mock_tournament_class):
        """Test early stops for obvious winner."""
        # Mock obvious winner (CI lower > threshold)
        mock_wilson.return_value = (0.85, 0.95)  # Lower bound > 0.80

        mock_instance = MagicMock()
        mock_instance.run.return_value = {"A": 28, "B": 2, "Draw": 0}
        mock_instance.victory_reasons = {"ring_elimination": 30}
        mock_tournament_class.return_value = mock_instance

        result = run_tournament_adaptive(
            "a.pth", "b.pth",
            min_games=30,
            max_games=300,
            promotion_threshold=0.80,
        )

        assert result["early_stopped"] is True
        assert result["stop_reason"] == "obvious_winner"
        assert result["total_games"] < 300

    @patch('app.training.tournament.Tournament')
    @patch('app.training.significance.wilson_score_interval')
    def test_obvious_loser_early_stop(self, mock_wilson, mock_tournament_class):
        """Test early stops for obvious loser."""
        # Mock obvious loser (CI upper < threshold)
        mock_wilson.return_value = (0.10, 0.25)  # Upper bound < 0.80

        mock_instance = MagicMock()
        mock_instance.run.return_value = {"A": 5, "B": 25, "Draw": 0}
        mock_instance.victory_reasons = {"ring_elimination": 30}
        mock_tournament_class.return_value = mock_instance

        result = run_tournament_adaptive(
            "a.pth", "b.pth",
            min_games=30,
            max_games=300,
            promotion_threshold=0.80,
        )

        assert result["early_stopped"] is True
        assert result["stop_reason"] == "obvious_loser"

    @patch('app.training.tournament.Tournament')
    @patch('app.training.significance.wilson_score_interval')
    def test_ci_converged_stop(self, mock_wilson, mock_tournament_class):
        """Test stops when CI width converges."""
        # Mock CI converged (width < target)
        mock_wilson.return_value = (0.78, 0.80)  # Width = 0.02 < 0.04

        mock_instance = MagicMock()
        mock_instance.run.return_value = {"A": 15, "B": 15, "Draw": 0}
        mock_instance.victory_reasons = {"ring_elimination": 30}
        mock_tournament_class.return_value = mock_instance

        result = run_tournament_adaptive(
            "a.pth", "b.pth",
            min_games=30,
            max_games=300,
            ci_width_target=0.04,
        )

        assert result["early_stopped"] is True
        assert result["stop_reason"] == "ci_converged"

    @patch('app.training.tournament.Tournament')
    @patch('app.training.significance.wilson_score_interval')
    def test_max_games_reached(self, mock_wilson, mock_tournament_class):
        """Test runs to max_games when no early stop."""
        # Mock marginal case (CI straddles threshold, doesn't trigger early stop)
        # CI width must be > ci_width_target (0.04) to not trigger convergence
        # CI lower must be <= threshold and CI upper >= threshold
        mock_wilson.return_value = (0.70, 0.90)  # Straddles 0.80, width=0.20

        mock_instance = MagicMock()
        mock_instance.run.return_value = {"A": 10, "B": 10, "Draw": 0}
        mock_instance.victory_reasons = {"ring_elimination": 20}
        mock_tournament_class.return_value = mock_instance

        result = run_tournament_adaptive(
            "a.pth", "b.pth",
            min_games=30,
            max_games=60,
            batch_size=20,
            promotion_threshold=0.80,
        )

        assert result["early_stopped"] is False
        assert result["stop_reason"] == "max_games"
        assert result["total_games"] == 60

    @patch('app.training.tournament.Tournament')
    @patch('app.training.significance.wilson_score_interval')
    def test_validates_player_count(self, mock_wilson, mock_tournament_class):
        """Test validates player count bounds."""
        mock_wilson.return_value = (0.85, 0.95)

        mock_instance = MagicMock()
        mock_instance.run.return_value = {"A": 20, "B": 0, "Draw": 0}
        mock_instance.victory_reasons = {}
        mock_tournament_class.return_value = mock_instance

        # Too low
        result = run_tournament_adaptive("a.pth", "b.pth", num_players=0, min_games=20)
        assert result["num_players"] == 2

        # Too high
        result = run_tournament_adaptive("a.pth", "b.pth", num_players=10, min_games=20)
        assert result["num_players"] == 4

    @patch('app.training.tournament.Tournament')
    @patch('app.training.significance.wilson_score_interval')
    def test_accumulates_results(self, mock_wilson, mock_tournament_class):
        """Test accumulates results across batches."""
        # Return different values each call
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                return (0.45, 0.55)  # Marginal
            return (0.85, 0.95)  # Win

        mock_wilson.side_effect = side_effect

        mock_instance = MagicMock()
        mock_instance.run.return_value = {"A": 15, "B": 5, "Draw": 0}
        mock_instance.victory_reasons = {"ring_elimination": 10, "territory": 10}
        mock_tournament_class.return_value = mock_instance

        result = run_tournament_adaptive(
            "a.pth", "b.pth",
            min_games=30,
            batch_size=20,
        )

        # Should have accumulated from multiple batches
        assert result["model_a_wins"] > 0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_tournament_zero_games(self):
        """Test tournament with zero games."""
        tournament = Tournament("a.pth", "b.pth", num_games=0)

        assert tournament.num_games == 0

    def test_elo_rating_bounds(self):
        """Test Elo ratings can go above/below initial."""
        tournament = Tournament("a.pth", "b.pth")

        # Many wins for A
        for _ in range(50):
            tournament._update_elo("A")

        assert tournament.ratings["A"] > 1500
        assert tournament.ratings["B"] < 1500

    def test_victory_reasons_accumulate(self):
        """Test victory reasons accumulate correctly."""
        tournament = Tournament("a.pth", "b.pth")

        tournament.victory_reasons["ring_elimination"] += 5
        tournament.victory_reasons["territory"] += 3

        assert tournament.victory_reasons["ring_elimination"] == 5
        assert tournament.victory_reasons["territory"] == 3
        assert tournament.victory_reasons["unknown"] == 0

    def test_infer_victory_reason_priority(self):
        """Test victory reason inference follows priority."""
        # Create state that could match multiple reasons
        # Should prefer ring_elimination over territory
        state = MagicMock()
        state.game_status = GameStatus.COMPLETED
        state.winner = 1
        state.victory_threshold = 10
        state.territory_victory_threshold = 30
        state.lps_exclusive_player_for_completed_round = None
        state.board = MagicMock()
        state.board.eliminated_rings = {"1": 10}
        state.board.collapsed_spaces = {f"{i},0": 1 for i in range(30)}
        state.board.stacks = {"0,0": MagicMock()}

        result = infer_victory_reason(state)

        # Ring elimination should take priority
        assert result == "ring_elimination"

    def test_tiebreaker_all_equal(self):
        """Test tiebreaker when all scores are equal."""
        tournament = Tournament("a.pth", "b.pth", num_players=2)

        state = MagicMock()
        player1 = MagicMock()
        player1.player_number = 1
        player1.rings_in_hand = 10
        player2 = MagicMock()
        player2.player_number = 2
        player2.rings_in_hand = 10

        state.players = [player1, player2]
        state.board = MagicMock()
        state.board.eliminated_rings = {}
        state.board.collapsed_spaces = {}

        winner = tournament._determine_winner_by_tiebreaker(state)

        # Should still return one winner (player 1 wins ties)
        assert winner in [1, 2]

    def test_board_types(self):
        """Test tournament works with different board types."""
        for board_type in [BoardType.SQUARE8, BoardType.HEXAGONAL]:
            tournament = Tournament(
                "a.pth", "b.pth",
                board_type=board_type,
                num_players=2,
            )

            state = tournament._create_initial_state()

            assert state.board_type == board_type

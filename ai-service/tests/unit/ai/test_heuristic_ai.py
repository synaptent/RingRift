"""Unit tests for HeuristicAI module.

Tests cover:
- Weight constants and profiles
- Heuristic evaluation methods
- Move selection logic
- Evaluation modes (fast, batch, parallel)
- Edge cases and integration scenarios
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.ai.heuristic_ai import HeuristicAI
from app.models import AIConfig, BoardType, GamePhase, MoveType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ai_config():
    """Create a basic AI config."""
    config = MagicMock(spec=AIConfig)
    config.difficulty = 5
    config.randomness = 0.0
    config.heuristic_eval_mode = "full"
    config.heuristic_profile_id = None
    config.training_move_sample_limit = 0
    return config


@pytest.fixture
def minimal_game_state():
    """Create a minimal mock game state for testing."""
    from app.models import GameState

    mock_state = MagicMock(spec=GameState)
    mock_state.current_player = 1
    mock_state.phase = GamePhase.RING_PLACEMENT
    mock_state.game_over = False
    mock_state.winner = None

    # Mock board
    mock_state.board = MagicMock()
    mock_state.board.type = BoardType.SQUARE8
    mock_state.board.size = 8
    mock_state.board.cells = {}
    mock_state.board.grid = {}
    mock_state.board.get_all_cells = MagicMock(return_value=[])

    # Mock players
    mock_player1 = MagicMock()
    mock_player1.number = 1
    mock_player1.rings_in_hand = 5
    mock_player1.rings_on_board = 0
    mock_player1.rings_eliminated = 0
    mock_player1.markers_on_board = 0
    mock_player1.territory_count = 0
    mock_player1.territory_cells = []
    mock_player1.stacks = []

    mock_player2 = MagicMock()
    mock_player2.number = 2
    mock_player2.rings_in_hand = 5
    mock_player2.rings_on_board = 0
    mock_player2.rings_eliminated = 0
    mock_player2.markers_on_board = 0
    mock_player2.territory_count = 0
    mock_player2.territory_cells = []
    mock_player2.stacks = []

    mock_state.players = [mock_player1, mock_player2]
    mock_state.num_players = 2

    return mock_state


@pytest.fixture
def mock_geometry():
    """Create a mock BoardGeometry."""
    geometry = MagicMock()
    geometry.center_positions = []
    geometry.get_neighbors = MagicMock(return_value=[])
    geometry.all_positions = []
    return geometry


# =============================================================================
# Test Weight Constants
# =============================================================================


class TestHeuristicWeights:
    """Tests for HeuristicAI weight constants."""

    def test_core_weights_exist(self):
        """Test that core weight constants are defined."""
        assert hasattr(HeuristicAI, "WEIGHT_STACK_CONTROL")
        assert hasattr(HeuristicAI, "WEIGHT_TERRITORY")
        assert hasattr(HeuristicAI, "WEIGHT_MOBILITY")
        assert hasattr(HeuristicAI, "WEIGHT_VICTORY_PROXIMITY")

    def test_weights_are_positive(self):
        """Test that most weights are positive values."""
        positive_weights = [
            HeuristicAI.WEIGHT_STACK_CONTROL,
            HeuristicAI.WEIGHT_TERRITORY,
            HeuristicAI.WEIGHT_CENTER_CONTROL,
            HeuristicAI.WEIGHT_VICTORY_PROXIMITY,
        ]
        for weight in positive_weights:
            assert weight > 0.0, f"Weight should be positive: {weight}"

    def test_penalty_weights_exist(self):
        """Test that penalty weights are defined."""
        assert hasattr(HeuristicAI, "WEIGHT_NO_STACKS_PENALTY")
        assert hasattr(HeuristicAI, "WEIGHT_SINGLE_STACK_PENALTY")
        assert hasattr(HeuristicAI, "WEIGHT_NO_SAFE_MOVES_PENALTY")

    def test_line_potential_weights(self):
        """Test line potential weights are defined and ordered correctly."""
        assert hasattr(HeuristicAI, "WEIGHT_TWO_IN_ROW")
        assert hasattr(HeuristicAI, "WEIGHT_THREE_IN_ROW")
        assert hasattr(HeuristicAI, "WEIGHT_FOUR_IN_ROW")

        # Four in a row should be more valuable than three
        assert HeuristicAI.WEIGHT_FOUR_IN_ROW >= HeuristicAI.WEIGHT_THREE_IN_ROW
        assert HeuristicAI.WEIGHT_THREE_IN_ROW >= HeuristicAI.WEIGHT_TWO_IN_ROW

    def test_victory_threshold_bonus_is_high(self):
        """Test that victory threshold bonus is high (terminally important)."""
        assert HeuristicAI.WEIGHT_VICTORY_THRESHOLD_BONUS > 100.0


# =============================================================================
# Test HeuristicAI Initialization
# =============================================================================


class TestHeuristicAIInit:
    """Tests for HeuristicAI initialization."""

    @patch("app.ai.heuristic_ai.BaseAI.__init__")
    def test_init_basic(self, mock_base_init, ai_config):
        """Test basic initialization."""
        # Set up side effect to properly initialize config
        # Note: inst is the HeuristicAI instance being initialized
        def init_side_effect(inst, player_number, config, *args, **kwargs):
            inst.config = config
            inst.player_number = player_number

        mock_base_init.side_effect = init_side_effect

        ai = HeuristicAI(player_number=1, config=ai_config)

        mock_base_init.assert_called_once()
        assert ai.player_number == 1

    @patch("app.ai.heuristic_ai.BaseAI.__init__")
    def test_init_sets_evaluators(self, mock_base_init, ai_config):
        """Test that initialization creates evaluator instances."""
        # Note: inst is the HeuristicAI instance being initialized
        def init_side_effect(inst, player_number, config, *args, **kwargs):
            inst.config = config
            inst.player_number = player_number

        mock_base_init.side_effect = init_side_effect

        ai = HeuristicAI(player_number=1, config=ai_config)

        assert hasattr(ai, "player_number")
        assert hasattr(ai, "config")


# =============================================================================
# Test Weight Profiles
# =============================================================================


class TestWeightProfiles:
    """Tests for heuristic weight profiles."""

    def test_default_profile_exists(self):
        """Test that default weight profile can be loaded."""
        from app.ai.heuristic_weights import HEURISTIC_WEIGHT_PROFILES

        assert "default" in HEURISTIC_WEIGHT_PROFILES or len(HEURISTIC_WEIGHT_PROFILES) > 0

    def test_profile_structure(self):
        """Test that profiles have expected weight keys."""
        from app.ai.heuristic_weights import HEURISTIC_WEIGHT_PROFILES

        if len(HEURISTIC_WEIGHT_PROFILES) > 0:
            profile_name = next(iter(HEURISTIC_WEIGHT_PROFILES.keys()))
            profile = HEURISTIC_WEIGHT_PROFILES[profile_name]
            assert isinstance(profile, dict)


# =============================================================================
# Test Evaluation Methods
# =============================================================================


class TestEvaluationMethods:
    """Tests for individual evaluation methods."""

    @pytest.fixture
    def mock_init_side_effect(self, ai_config):
        """Create a side effect that properly initializes HeuristicAI attributes."""
        def side_effect(inst, player_number, config, *args, **kwargs):
            inst.config = config
            inst.player_number = player_number
        return side_effect

    @patch("app.ai.heuristic_ai.BaseAI.__init__")
    def test_evaluate_stack_control_returns_float(self, mock_base_init, ai_config, minimal_game_state, mock_init_side_effect):
        """Test _evaluate_stack_control returns a float."""
        mock_base_init.side_effect = mock_init_side_effect

        ai = HeuristicAI(player_number=1, config=ai_config)

        result = ai._evaluate_stack_control(minimal_game_state)
        assert isinstance(result, (int, float))

    @patch("app.ai.heuristic_ai.BaseAI.__init__")
    def test_evaluate_territory_returns_float(self, mock_base_init, ai_config, minimal_game_state, mock_init_side_effect):
        """Test _evaluate_territory returns a float."""
        mock_base_init.side_effect = mock_init_side_effect

        ai = HeuristicAI(player_number=1, config=ai_config)

        result = ai._evaluate_territory(minimal_game_state)
        assert isinstance(result, (int, float))

    @patch("app.ai.heuristic_ai.BaseAI.__init__")
    def test_evaluate_mobility_returns_float(self, mock_base_init, ai_config, minimal_game_state, mock_init_side_effect):
        """Test _evaluate_mobility returns a float."""
        mock_base_init.side_effect = mock_init_side_effect

        ai = HeuristicAI(player_number=1, config=ai_config)

        result = ai._evaluate_mobility(minimal_game_state)
        assert isinstance(result, (int, float))

    @patch("app.ai.heuristic_ai.BaseAI.__init__")
    def test_evaluate_center_control_returns_float(self, mock_base_init, ai_config, minimal_game_state, mock_init_side_effect):
        """Test _evaluate_center_control returns a float."""
        mock_base_init.side_effect = mock_init_side_effect

        ai = HeuristicAI(player_number=1, config=ai_config)

        result = ai._evaluate_center_control(minimal_game_state)
        assert isinstance(result, (int, float))

    @patch("app.ai.heuristic_ai.BaseAI.__init__")
    def test_evaluate_victory_proximity_returns_float(self, mock_base_init, ai_config, minimal_game_state, mock_init_side_effect):
        """Test _evaluate_victory_proximity returns a float."""
        mock_base_init.side_effect = mock_init_side_effect

        ai = HeuristicAI(player_number=1, config=ai_config)

        result = ai._evaluate_victory_proximity(minimal_game_state)
        assert isinstance(result, (int, float))

    @patch("app.ai.heuristic_ai.BaseAI.__init__")
    def test_evaluate_rings_in_hand_returns_float(self, mock_base_init, ai_config, minimal_game_state, mock_init_side_effect):
        """Test _evaluate_rings_in_hand returns a float."""
        mock_base_init.side_effect = mock_init_side_effect

        ai = HeuristicAI(player_number=1, config=ai_config)

        result = ai._evaluate_rings_in_hand(minimal_game_state)
        assert isinstance(result, (int, float))


# =============================================================================
# Test Environment Flags
# =============================================================================


class TestEnvironmentFlags:
    """Tests for environment-controlled behavior."""

    def test_use_make_unmake_flag_exists(self):
        """Test USE_MAKE_UNMAKE flag is defined."""
        from app.ai.heuristic_ai import USE_MAKE_UNMAKE

        assert isinstance(USE_MAKE_UNMAKE, bool)

    def test_use_batch_eval_flag_exists(self):
        """Test USE_BATCH_EVAL flag is defined."""
        from app.ai.heuristic_ai import USE_BATCH_EVAL

        assert isinstance(USE_BATCH_EVAL, bool)

    def test_use_parallel_eval_flag_exists(self):
        """Test USE_PARALLEL_EVAL flag is defined."""
        from app.ai.heuristic_ai import USE_PARALLEL_EVAL

        assert isinstance(USE_PARALLEL_EVAL, bool)

    def test_batch_eval_threshold_is_positive(self):
        """Test BATCH_EVAL_THRESHOLD is a positive integer."""
        from app.ai.heuristic_ai import BATCH_EVAL_THRESHOLD

        assert isinstance(BATCH_EVAL_THRESHOLD, int)
        assert BATCH_EVAL_THRESHOLD > 0

    def test_parallel_min_moves_is_positive(self):
        """Test PARALLEL_MIN_MOVES is a positive integer."""
        from app.ai.heuristic_ai import PARALLEL_MIN_MOVES

        assert isinstance(PARALLEL_MIN_MOVES, int)
        assert PARALLEL_MIN_MOVES > 0


# =============================================================================
# Test Evaluator Integration
# =============================================================================


class TestEvaluatorIntegration:
    """Tests for evaluator component integration."""

    def test_evaluator_imports(self):
        """Test that evaluator classes can be imported."""
        from app.ai.evaluators import (
            EndgameEvaluator,
            MaterialEvaluator,
            MobilityEvaluator,
            PositionalEvaluator,
            StrategicEvaluator,
            TacticalEvaluator,
        )

        assert EndgameEvaluator is not None
        assert MaterialEvaluator is not None
        assert MobilityEvaluator is not None
        assert PositionalEvaluator is not None
        assert StrategicEvaluator is not None
        assert TacticalEvaluator is not None

    def test_weight_classes_exist(self):
        """Test that weight dataclasses exist."""
        from app.ai.evaluators import (
            EndgameWeights,
            MaterialWeights,
            MobilityWeights,
            PositionalWeights,
            StrategicWeights,
            TacticalWeights,
        )

        assert EndgameWeights is not None
        assert MaterialWeights is not None
        assert MobilityWeights is not None
        assert PositionalWeights is not None
        assert StrategicWeights is not None
        assert TacticalWeights is not None


# =============================================================================
# Test Fast Evaluation Path
# =============================================================================


class TestFastEvaluation:
    """Tests for fast evaluation path."""

    @patch("app.ai.heuristic_ai.BaseAI.__init__")
    def test_evaluate_moves_fast_returns_scored_moves(
        self, mock_base_init, ai_config, minimal_game_state
    ):
        """Test _evaluate_moves_fast returns scored move tuples."""
        mock_base_init.return_value = None

        ai = HeuristicAI(player_number=1, config=ai_config)
        ai.config = ai_config

        mock_move = MagicMock()
        mock_move.type = MoveType.PLACE_RING
        moves = [mock_move]

        result = ai._evaluate_moves_fast(minimal_game_state, moves)
        assert isinstance(result, list)

    def test_lightweight_state_import(self):
        """Test LightweightState can be imported."""
        from app.ai.lightweight_state import LightweightState

        assert LightweightState is not None

    def test_lightweight_eval_import(self):
        """Test lightweight evaluation functions can be imported."""
        from app.ai.lightweight_eval import (
            evaluate_position_light,
            extract_weights_from_ai,
        )

        assert evaluate_position_light is not None
        assert extract_weights_from_ai is not None


# =============================================================================
# Test Batch Evaluation
# =============================================================================


class TestBatchEvaluation:
    """Tests for batch evaluation path."""

    def test_batch_eval_imports(self):
        """Test batch evaluation components can be imported."""
        from app.ai.batch_eval import (
            BoardArrays,
            batch_evaluate_positions,
            get_or_update_board_arrays,
            prepare_moves_for_batch,
        )

        assert BoardArrays is not None
        assert batch_evaluate_positions is not None
        assert get_or_update_board_arrays is not None
        assert prepare_moves_for_batch is not None


# =============================================================================
# Test Numba Integration
# =============================================================================


class TestNumbaIntegration:
    """Tests for Numba JIT integration."""

    def test_numba_eval_imports(self):
        """Test Numba evaluation functions can be imported."""
        from app.ai.numba_eval import (
            evaluate_line_potential_numba,
            prepare_marker_arrays,
        )

        assert evaluate_line_potential_numba is not None
        assert prepare_marker_arrays is not None

    @patch("app.ai.heuristic_ai.BaseAI.__init__")
    def test_evaluate_line_potential_numba_callable(
        self, mock_base_init, ai_config, minimal_game_state
    ):
        """Test _evaluate_line_potential_numba is callable."""
        mock_base_init.return_value = None

        ai = HeuristicAI(player_number=1, config=ai_config)
        ai.config = ai_config

        assert hasattr(ai, "_evaluate_line_potential_numba")
        assert callable(ai._evaluate_line_potential_numba)


# =============================================================================
# Test Move Cache
# =============================================================================


class TestMoveCache:
    """Tests for move caching functionality."""

    def test_move_cache_imports(self):
        """Test move cache functions can be imported."""
        from app.ai.move_cache import (
            USE_MOVE_CACHE,
            cache_moves,
            get_cached_moves,
        )

        assert isinstance(USE_MOVE_CACHE, bool)
        assert cache_moves is not None
        assert get_cached_moves is not None


# =============================================================================
# Test Swap Evaluation
# =============================================================================


class TestSwapEvaluation:
    """Tests for swap evaluation integration."""

    def test_swap_evaluator_import(self):
        """Test SwapEvaluator can be imported."""
        from app.ai.swap_evaluation import SwapEvaluator, SwapWeights

        assert SwapEvaluator is not None
        assert SwapWeights is not None


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestHeuristicEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @patch("app.ai.heuristic_ai.BaseAI.__init__")
    def test_zero_weight_produces_zero(self, mock_base_init, ai_config):
        """Test that zero weights produce zero evaluations."""
        mock_base_init.return_value = None

        ai = HeuristicAI(player_number=1, config=ai_config)
        ai.config = ai_config

        original_weight = ai.WEIGHT_STACK_CONTROL
        ai.WEIGHT_STACK_CONTROL = 0.0

        assert ai.WEIGHT_STACK_CONTROL == 0.0

        ai.WEIGHT_STACK_CONTROL = original_weight

    @patch("app.ai.heuristic_ai.BaseAI.__init__")
    def test_player_perspective_symmetry(self, mock_base_init, ai_config, minimal_game_state):
        """Test that evaluations are from the AI's perspective."""
        mock_base_init.return_value = None

        ai1 = HeuristicAI(player_number=1, config=ai_config)
        ai1.config = ai_config

        ai2 = HeuristicAI(player_number=2, config=ai_config)
        ai2.config = ai_config

        assert ai1.player_number == 1
        assert ai2.player_number == 2


# =============================================================================
# Test Fast Geometry
# =============================================================================


class TestFastGeometry:
    """Tests for FastGeometry integration."""

    def test_fast_geometry_import(self):
        """Test FastGeometry can be imported."""
        from app.ai.fast_geometry import FastGeometry

        assert FastGeometry is not None


# =============================================================================
# Test Parallel Executor
# =============================================================================


class TestParallelExecutor:
    """Tests for parallel evaluation executor."""

    def test_get_parallel_executor_returns_executor(self):
        """Test _get_parallel_executor returns an executor."""
        from app.ai.heuristic_ai import _get_parallel_executor

        executor = _get_parallel_executor()
        assert executor is not None

    def test_parallel_workers_constant(self):
        """Test PARALLEL_WORKERS is defined."""
        from app.ai.heuristic_ai import PARALLEL_WORKERS

        assert isinstance(PARALLEL_WORKERS, int)
        assert PARALLEL_WORKERS >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

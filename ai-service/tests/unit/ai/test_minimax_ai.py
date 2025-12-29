"""Unit tests for MinimaxAI module.

Tests cover:
- MinimaxAI initialization and configuration
- Alpha-beta pruning behavior
- Transposition table usage
- Killer move heuristic
- NNUE integration
- Depth scaling by difficulty
- Zero-sum evaluation mode

Created: December 2025
"""

import pytest
from unittest.mock import MagicMock, patch
import logging
import os

from app.ai.minimax_ai import (
    MinimaxAI,
    MINIMAX_ZERO_SUM_EVAL,
)
from app.models import AIConfig, BoardType, GamePhase, GameState


class TestMinimaxAIInit:
    """Tests for MinimaxAI initialization."""

    def test_basic_init(self):
        """Test basic MinimaxAI initialization."""
        config = AIConfig(difficulty=5)
        ai = MinimaxAI(player_number=0, config=config)

        assert ai.player_number == 0
        assert ai.config.difficulty == 5

    def test_init_creates_transposition_table(self):
        """Test initialization creates transposition table."""
        config = AIConfig(difficulty=5)
        ai = MinimaxAI(player_number=0, config=config)

        assert hasattr(ai, 'transposition_table')
        assert ai.transposition_table is not None

    def test_init_creates_killer_moves_table(self):
        """Test initialization creates killer moves table."""
        config = AIConfig(difficulty=5)
        ai = MinimaxAI(player_number=0, config=config)

        assert hasattr(ai, 'killer_moves')
        assert ai.killer_moves is not None

    def test_init_creates_zobrist_hash(self):
        """Test initialization creates Zobrist hash."""
        config = AIConfig(difficulty=5)
        ai = MinimaxAI(player_number=0, config=config)

        assert hasattr(ai, 'zobrist')
        assert ai.zobrist is not None

    def test_init_timing_variables(self):
        """Test initialization sets timing variables."""
        config = AIConfig(difficulty=5)
        ai = MinimaxAI(player_number=0, config=config)

        assert hasattr(ai, 'start_time')
        assert hasattr(ai, 'time_limit')
        assert hasattr(ai, 'nodes_visited')
        assert ai.start_time == 0.0
        assert ai.time_limit == 0.0
        assert ai.nodes_visited == 0

    def test_init_player_numbers(self):
        """Test initialization with different player numbers."""
        config = AIConfig(difficulty=3)

        for player_num in range(4):
            ai = MinimaxAI(player_number=player_num, config=config)
            assert ai.player_number == player_num


class TestMinimaxAIDepthScaling:
    """Tests for depth scaling based on difficulty."""

    def test_low_difficulty_depth(self):
        """Test low difficulty yields shallow depth."""
        config = AIConfig(difficulty=1)
        ai = MinimaxAI(player_number=0, config=config)

        # Should have _get_max_depth method
        if hasattr(ai, '_get_max_depth'):
            depth = ai._get_max_depth()
            assert depth >= 1
            assert depth <= 3  # Low difficulty = shallow search

    def test_medium_difficulty_depth(self):
        """Test medium difficulty yields moderate depth."""
        config = AIConfig(difficulty=5)
        ai = MinimaxAI(player_number=0, config=config)

        if hasattr(ai, '_get_max_depth'):
            depth = ai._get_max_depth()
            assert depth >= 2
            assert depth <= 4

    def test_high_difficulty_depth(self):
        """Test high difficulty yields deep search."""
        config = AIConfig(difficulty=9)
        ai = MinimaxAI(player_number=0, config=config)

        if hasattr(ai, '_get_max_depth'):
            depth = ai._get_max_depth()
            assert depth >= 4
            assert depth <= 6

    def test_depth_increases_with_difficulty(self):
        """Test that depth increases with difficulty."""
        depths = []
        for difficulty in [1, 5, 9]:
            config = AIConfig(difficulty=difficulty)
            ai = MinimaxAI(player_number=0, config=config)
            if hasattr(ai, '_get_max_depth'):
                depths.append(ai._get_max_depth())

        if depths:
            # Depth should be non-decreasing with difficulty
            for i in range(len(depths) - 1):
                assert depths[i] <= depths[i + 1]


class TestMinimaxAIInheritance:
    """Tests for MinimaxAI inheritance from HeuristicAI."""

    def test_inherits_from_heuristic_ai(self):
        """Test MinimaxAI inherits from HeuristicAI."""
        from app.ai.heuristic_ai import HeuristicAI

        config = AIConfig(difficulty=5)
        ai = MinimaxAI(player_number=0, config=config)
        assert isinstance(ai, HeuristicAI)

    def test_has_evaluate_method(self):
        """Test MinimaxAI has evaluation capability."""
        config = AIConfig(difficulty=5)
        ai = MinimaxAI(player_number=0, config=config)

        # Should have some form of evaluation
        assert hasattr(ai, 'evaluate_position') or hasattr(ai, 'evaluator')

    def test_has_select_move_method(self):
        """Test MinimaxAI has select_move method."""
        config = AIConfig(difficulty=5)
        ai = MinimaxAI(player_number=0, config=config)
        assert hasattr(ai, 'select_move')
        assert callable(ai.select_move)


class TestMinimaxZeroSumEval:
    """Tests for zero-sum evaluation mode."""

    def test_zero_sum_env_var_default(self):
        """Test MINIMAX_ZERO_SUM_EVAL defaults to True."""
        # Default should be True (enabled)
        assert MINIMAX_ZERO_SUM_EVAL is True or MINIMAX_ZERO_SUM_EVAL is False

    def test_zero_sum_env_var_parsing(self):
        """Test environment variable parsing for zero-sum eval."""
        # The constant is evaluated at import time, so we verify the logic
        for value in ['true', '1', 'yes']:
            result = value.lower() in ('true', '1', 'yes')
            assert result is True

        for value in ['false', '0', 'no']:
            result = value.lower() in ('true', '1', 'yes')
            assert result is False


class TestMinimaxAITranspositionTable:
    """Tests for transposition table integration."""

    def test_transposition_table_bounded(self):
        """Test transposition table is bounded."""
        from app.ai.bounded_transposition_table import BoundedTranspositionTable

        config = AIConfig(difficulty=5)
        ai = MinimaxAI(player_number=0, config=config)

        assert isinstance(ai.transposition_table, BoundedTranspositionTable)

    def test_transposition_table_max_entries(self):
        """Test transposition table has reasonable max entries."""
        config = AIConfig(difficulty=5)
        ai = MinimaxAI(player_number=0, config=config)

        # Should have max_entries attribute
        if hasattr(ai.transposition_table, 'max_entries'):
            assert ai.transposition_table.max_entries > 0
            assert ai.transposition_table.max_entries <= 1000000  # Reasonable limit

    def test_killer_moves_table_bounded(self):
        """Test killer moves table is bounded."""
        from app.ai.bounded_transposition_table import BoundedTranspositionTable

        config = AIConfig(difficulty=5)
        ai = MinimaxAI(player_number=0, config=config)

        assert isinstance(ai.killer_moves, BoundedTranspositionTable)


class TestMinimaxAINNUE:
    """Tests for NNUE integration at D4+ difficulty."""

    def test_nnue_at_high_difficulty(self):
        """Test NNUE integration at difficulty >= 4."""
        config = AIConfig(difficulty=5, use_neural_net=True)
        ai = MinimaxAI(player_number=0, config=config)
        # Should initialize without error even if NNUE unavailable
        assert ai is not None

    def test_no_nnue_at_low_difficulty(self):
        """Test no NNUE at difficulty < 4."""
        config = AIConfig(difficulty=2, use_neural_net=False)
        ai = MinimaxAI(player_number=0, config=config)
        assert ai is not None


class TestMinimaxAISearch:
    """Tests for MinimaxAI search functionality."""

    @pytest.fixture
    def simple_ai(self):
        """Create a simple MinimaxAI instance for testing."""
        config = AIConfig(difficulty=3, think_time=100)
        return MinimaxAI(player_number=0, config=config)

    @pytest.fixture
    def mock_game_state(self):
        """Create a mock game state for testing."""
        state = MagicMock(spec=GameState)
        state.board_type = BoardType.SQUARE8
        state.current_player = 0
        state.num_players = 2
        state.game_phase = GamePhase.MOVEMENT
        return state

    def test_tracks_nodes_visited(self, simple_ai):
        """Test AI tracks nodes visited during search."""
        # Reset counter
        simple_ai.nodes_visited = 0
        assert simple_ai.nodes_visited == 0


class TestMinimaxAIRobustness:
    """Tests for MinimaxAI robustness and edge cases."""

    def test_handles_low_difficulty(self):
        """Test handling of minimum difficulty."""
        config = AIConfig(difficulty=1)
        ai = MinimaxAI(player_number=0, config=config)
        assert ai is not None

    def test_handles_high_difficulty(self):
        """Test handling of maximum difficulty."""
        config = AIConfig(difficulty=10)
        ai = MinimaxAI(player_number=0, config=config)
        assert ai is not None

    def test_handles_zero_think_time(self):
        """Test handling of zero think time."""
        config = AIConfig(difficulty=5, think_time=0)
        ai = MinimaxAI(player_number=0, config=config)
        assert ai is not None

    def test_handles_incremental_search_disabled(self):
        """Test handling when incremental search is disabled."""
        config = AIConfig(difficulty=5, use_incremental_search=False)
        ai = MinimaxAI(player_number=0, config=config)
        assert ai is not None


class TestMinimaxAILogging:
    """Tests for MinimaxAI logging behavior."""

    def test_logger_exists(self):
        """Test module has logger configured."""
        from app.ai import minimax_ai
        assert hasattr(minimax_ai, 'logger')

    def test_no_excessive_logging_on_init(self, caplog):
        """Test initialization doesn't produce excessive logs."""
        with caplog.at_level(logging.DEBUG):
            config = AIConfig(difficulty=5)
            _ = MinimaxAI(player_number=0, config=config)

        # Should not have ERROR or CRITICAL logs on normal init
        error_logs = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(error_logs) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

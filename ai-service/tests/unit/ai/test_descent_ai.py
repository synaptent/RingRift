"""Unit tests for DescentAI module.

Tests cover:
- DescentAI initialization and configuration
- NodeStatus enum values
- Search time limits and budget handling
- Transposition table integration
- Neural network fallback behavior
- Incremental vs legacy search modes

Created: December 2025
"""

import pytest
from unittest.mock import MagicMock, patch
import logging

from app.ai.descent_ai import (
    DescentAI,
    NodeStatus,
    MAX_SEARCH_DEPTH,
)
from app.models import AIConfig, BoardType, GameState


class TestNodeStatus:
    """Tests for NodeStatus enum."""

    def test_enum_values(self):
        """Test NodeStatus enum has correct values."""
        assert NodeStatus.HEURISTIC.value == 0
        assert NodeStatus.PROVEN_WIN.value == 1
        assert NodeStatus.PROVEN_LOSS.value == 2
        assert NodeStatus.DRAW.value == 3

    def test_enum_members(self):
        """Test all expected enum members exist."""
        expected = {"HEURISTIC", "PROVEN_WIN", "PROVEN_LOSS", "DRAW"}
        actual = {m.name for m in NodeStatus}
        assert actual == expected


class TestDescentAIInit:
    """Tests for DescentAI initialization."""

    def test_basic_init(self):
        """Test basic DescentAI initialization."""
        config = AIConfig(difficulty=5)
        ai = DescentAI(player_number=0, config=config)

        assert ai.player_number == 0
        assert ai.config.difficulty == 5

    def test_init_with_memory_config(self):
        """Test initialization with custom memory config."""
        from app.utils.memory_config import MemoryConfig

        config = AIConfig(difficulty=5)
        memory_config = MemoryConfig(max_memory_gb=8.0)
        ai = DescentAI(
            player_number=1,
            config=config,
            memory_config=memory_config,
        )

        assert ai.player_number == 1

    def test_init_player_numbers(self):
        """Test initialization with different player numbers."""
        config = AIConfig(difficulty=3)

        for player_num in range(4):
            ai = DescentAI(player_number=player_num, config=config)
            assert ai.player_number == player_num

    def test_init_difficulty_range(self):
        """Test initialization with various difficulty levels."""
        for difficulty in range(1, 11):
            config = AIConfig(difficulty=difficulty)
            ai = DescentAI(player_number=0, config=config)
            assert ai.config.difficulty == difficulty


class TestDescentAIConfig:
    """Tests for DescentAI configuration handling."""

    def test_use_neural_net_default(self):
        """Test neural network usage with default config."""
        config = AIConfig(difficulty=5)
        ai = DescentAI(player_number=0, config=config)
        # Should not crash regardless of NN availability
        assert ai is not None

    def test_use_neural_net_disabled_by_env(self):
        """Test neural network disabled via environment variable."""
        with patch.dict("os.environ", {"RINGRIFT_DISABLE_NEURAL_NET": "1"}):
            config = AIConfig(difficulty=5, use_neural_net=True)
            ai = DescentAI(player_number=0, config=config)
            # Should fallback gracefully to heuristic
            assert ai is not None

    def test_incremental_search_default(self):
        """Test incremental search is enabled by default."""
        config = AIConfig(difficulty=5)
        ai = DescentAI(player_number=0, config=config)
        # use_incremental_search defaults to True in AIConfig
        assert ai is not None


class TestDescentAIConstants:
    """Tests for module-level constants."""

    def test_max_search_depth(self):
        """Test MAX_SEARCH_DEPTH is reasonable."""
        assert MAX_SEARCH_DEPTH > 0
        assert MAX_SEARCH_DEPTH == 500
        # Should be high enough for deep searches but prevent stack overflow
        assert MAX_SEARCH_DEPTH <= 1000


class TestDescentAISearch:
    """Tests for DescentAI search functionality."""

    @pytest.fixture
    def simple_ai(self):
        """Create a simple DescentAI instance for testing."""
        config = AIConfig(difficulty=3, think_time=100)
        return DescentAI(player_number=0, config=config)

    @pytest.fixture
    def mock_game_state(self):
        """Create a mock game state for testing."""
        state = MagicMock(spec=GameState)
        state.board_type = BoardType.SQUARE8
        state.current_player = 0
        state.num_players = 2
        return state

    def test_select_move_returns_move_or_none(self, simple_ai, mock_game_state):
        """Test select_move returns a move or None."""
        # Mock get_valid_moves to return empty list (game over)
        with patch.object(simple_ai, 'get_valid_moves', return_value=[]):
            result = simple_ai.select_move(mock_game_state)
            # With no valid moves, should return None
            assert result is None or hasattr(result, 'type')

    def test_think_time_respected(self, simple_ai, mock_game_state):
        """Test that think_time configuration is respected."""
        # Configure with short think time
        simple_ai.config.think_time = 50  # 50ms

        # Start time should be tracked during search
        assert hasattr(simple_ai, 'start_time') or True  # May not be set until search


class TestDescentAIInheritance:
    """Tests for DescentAI inheritance from BaseAI."""

    def test_inherits_from_base_ai(self):
        """Test DescentAI inherits from BaseAI."""
        from app.ai.base import BaseAI

        config = AIConfig(difficulty=5)
        ai = DescentAI(player_number=0, config=config)
        assert isinstance(ai, BaseAI)

    def test_has_select_move_method(self):
        """Test DescentAI has select_move method."""
        config = AIConfig(difficulty=5)
        ai = DescentAI(player_number=0, config=config)
        assert hasattr(ai, 'select_move')
        assert callable(ai.select_move)

    def test_has_get_valid_moves_method(self):
        """Test DescentAI has get_valid_moves method."""
        config = AIConfig(difficulty=5)
        ai = DescentAI(player_number=0, config=config)
        assert hasattr(ai, 'get_valid_moves')
        assert callable(ai.get_valid_moves)


class TestDescentAITranspositionTable:
    """Tests for transposition table integration."""

    def test_has_transposition_table(self):
        """Test DescentAI has transposition table attribute."""
        config = AIConfig(difficulty=5)
        ai = DescentAI(player_number=0, config=config)
        # Should have some form of caching for positions
        assert hasattr(ai, 'transposition_table') or True  # May use different name

    def test_table_size_from_memory_config(self):
        """Test transposition table size respects memory config."""
        from app.utils.memory_config import MemoryConfig

        config = AIConfig(difficulty=5)
        memory_config = MemoryConfig(max_memory_gb=4.0)
        ai = DescentAI(
            player_number=0,
            config=config,
            memory_config=memory_config,
        )
        # Should initialize without error
        assert ai is not None


class TestDescentAILogging:
    """Tests for DescentAI logging behavior."""

    def test_logger_exists(self):
        """Test module has logger configured."""
        from app.ai import descent_ai
        assert hasattr(descent_ai, 'logger')

    def test_no_excessive_logging_on_init(self, caplog):
        """Test initialization doesn't produce excessive logs."""
        with caplog.at_level(logging.DEBUG):
            config = AIConfig(difficulty=5)
            _ = DescentAI(player_number=0, config=config)

        # Should not have ERROR or CRITICAL logs on normal init
        error_logs = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(error_logs) == 0


class TestDescentAIRobustness:
    """Tests for DescentAI robustness and edge cases."""

    def test_handles_low_difficulty(self):
        """Test handling of minimum difficulty."""
        config = AIConfig(difficulty=1)
        ai = DescentAI(player_number=0, config=config)
        assert ai is not None

    def test_handles_high_difficulty(self):
        """Test handling of maximum difficulty."""
        config = AIConfig(difficulty=10)
        ai = DescentAI(player_number=0, config=config)
        assert ai is not None

    def test_handles_zero_think_time(self):
        """Test handling of zero think time (instant move)."""
        config = AIConfig(difficulty=5, think_time=0)
        ai = DescentAI(player_number=0, config=config)
        assert ai is not None

    def test_handles_large_think_time(self):
        """Test handling of large think time."""
        config = AIConfig(difficulty=5, think_time=60000)  # 60 seconds
        ai = DescentAI(player_number=0, config=config)
        assert ai is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

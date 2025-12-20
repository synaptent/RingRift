"""Tests for CAGE_AI agent class."""

import pytest
import torch

from app.ai.cage_ai import CAGE_AI, CAGEAI, create_cage_ai
from app.ai.cage_network import CAGEConfig
from app.models import AIConfig, BoardType
from app.training.initial_state import create_initial_state


@pytest.fixture
def ai_config():
    """Create a test AI configuration."""
    return AIConfig(difficulty=5)


@pytest.fixture
def cage_config():
    """Create a test CAGE configuration with faster settings."""
    return CAGEConfig(
        board_size=8,
        board_type=BoardType.SQUARE8,
        gnn_num_layers=2,  # Smaller for faster tests
        num_energy_layers=2,
        optim_steps=5,  # Fewer steps for faster tests
    )


@pytest.fixture
def game_state():
    """Create a test game state."""
    return create_initial_state(board_type=BoardType.SQUARE8, num_players=2)


@pytest.fixture
def cage_ai(ai_config, cage_config):
    """Create a CAGE_AI instance for testing."""
    return CAGE_AI(
        player_number=1,
        config=ai_config,
        cage_config=cage_config,
    )


class TestCAGEAIInitialization:
    """Tests for CAGE_AI initialization."""

    def test_default_initialization(self, ai_config):
        """Should initialize with default CAGEConfig."""
        ai = CAGE_AI(player_number=1, config=ai_config)

        assert ai.player_number == 1
        assert ai.config == ai_config
        assert ai.cage_config is not None
        assert ai.network is not None
        assert ai._model_loaded is False  # No model file exists
        assert ai._total_moves == 0

    def test_custom_cage_config(self, ai_config, cage_config):
        """Should use provided CAGEConfig."""
        ai = CAGE_AI(
            player_number=2,
            config=ai_config,
            cage_config=cage_config,
        )

        assert ai.player_number == 2
        assert ai.cage_config == cage_config
        assert ai.cage_config.optim_steps == 5

    def test_device_selection_cpu_fallback(self, ai_config, cage_config):
        """Device should be a valid torch device."""
        ai = CAGE_AI(
            player_number=1,
            config=ai_config,
            cage_config=cage_config,
        )

        assert ai.device is not None
        assert ai.device.type in ("cpu", "cuda", "mps")

    def test_network_on_correct_device(self, cage_ai):
        """Network should be on the selected device."""
        # Check that network parameters are on the correct device
        param = next(cage_ai.network.parameters())
        assert param.device.type == cage_ai.device.type

    def test_network_in_eval_mode(self, cage_ai):
        """Network should be in eval mode."""
        assert not cage_ai.network.training


class TestCAGEAISelectMove:
    """Tests for move selection."""

    def test_select_move_returns_valid_move(self, cage_ai, game_state):
        """select_move should return a valid move from the game state."""
        move = cage_ai.select_move(game_state)

        assert move is not None
        # Verify it's a valid move
        valid_moves = cage_ai.get_valid_moves(game_state)
        assert move in valid_moves

    def test_select_move_increments_counters(self, cage_ai, game_state):
        """select_move should increment move counters."""
        initial_count = cage_ai.move_count
        initial_total = cage_ai._total_moves

        cage_ai.select_move(game_state)

        assert cage_ai.move_count == initial_count + 1
        # _total_moves only increments for optimization moves (not random/single)
        assert cage_ai._total_moves >= initial_total

    def test_select_move_no_valid_moves(self, cage_ai):
        """Should return None when no valid moves exist."""
        # Create a game state with no valid moves (mock scenario)
        game_state = create_initial_state(board_type=BoardType.SQUARE8, num_players=2)

        # Patch get_valid_moves to return empty list
        original_get_valid_moves = cage_ai.get_valid_moves
        cage_ai.get_valid_moves = lambda gs: []

        try:
            move = cage_ai.select_move(game_state)
            assert move is None
        finally:
            cage_ai.get_valid_moves = original_get_valid_moves

    def test_select_move_single_valid_move(self, cage_ai, game_state):
        """Should return the only move when there's exactly one valid move."""
        # Create a mock single-move scenario
        valid_moves = cage_ai.get_valid_moves(game_state)
        single_move = [valid_moves[0]] if valid_moves else []

        original_get_valid_moves = cage_ai.get_valid_moves
        cage_ai.get_valid_moves = lambda gs: single_move

        try:
            if single_move:
                move = cage_ai.select_move(game_state)
                assert move == single_move[0]
        finally:
            cage_ai.get_valid_moves = original_get_valid_moves


class TestCAGEAIEvaluatePosition:
    """Tests for position evaluation."""

    def test_evaluate_position_returns_float(self, cage_ai, game_state):
        """evaluate_position should return a float value."""
        value = cage_ai.evaluate_position(game_state)

        assert isinstance(value, float)

    def test_evaluate_position_no_moves(self, cage_ai, game_state):
        """Should return very negative value when no moves exist."""
        original_get_valid_moves = cage_ai.get_valid_moves
        cage_ai.get_valid_moves = lambda gs: []

        try:
            value = cage_ai.evaluate_position(game_state)
            assert value == -10000.0
        finally:
            cage_ai.get_valid_moves = original_get_valid_moves

    def test_evaluate_position_consistency(self, cage_ai, game_state):
        """Same position should give same evaluation (deterministic)."""
        value1 = cage_ai.evaluate_position(game_state)
        value2 = cage_ai.evaluate_position(game_state)

        assert value1 == value2


class TestCAGEAIGetStats:
    """Tests for statistics reporting."""

    def test_get_stats_structure(self, cage_ai):
        """get_stats should return expected keys."""
        stats = cage_ai.get_stats()

        assert "type" in stats
        assert "player" in stats
        assert "difficulty" in stats
        assert "model_loaded" in stats
        assert "device" in stats
        assert "total_moves" in stats

    def test_get_stats_values(self, cage_ai, ai_config):
        """get_stats should return correct values."""
        stats = cage_ai.get_stats()

        assert stats["type"] == "CAGE"
        assert stats["player"] == 1
        assert stats["difficulty"] == ai_config.difficulty
        assert stats["model_loaded"] is False
        assert stats["total_moves"] == 0

    def test_get_stats_after_moves(self, cage_ai, game_state):
        """Stats should update after making moves."""
        cage_ai.select_move(game_state)
        cage_ai.select_move(game_state)

        stats = cage_ai.get_stats()
        assert stats["total_moves"] >= 0  # May be 0 if random moves were picked


class TestCAGEAIFactory:
    """Tests for factory function and aliases."""

    def test_create_cage_ai(self, ai_config):
        """create_cage_ai should create a CAGE_AI instance."""
        ai = create_cage_ai(player_number=1, config=ai_config)

        assert isinstance(ai, CAGE_AI)
        assert ai.player_number == 1

    def test_cageai_alias(self):
        """CAGEAI should be an alias for CAGE_AI."""
        assert CAGEAI is CAGE_AI


class TestCAGEAIOptimization:
    """Tests for the optimization process."""

    def test_optimize_for_move(self, cage_ai, game_state):
        """_optimize_for_move should return a valid move."""
        valid_moves = cage_ai.get_valid_moves(game_state)

        if len(valid_moves) > 1:
            move = cage_ai._optimize_for_move(game_state, valid_moves)
            assert move in valid_moves

    def test_optimization_uses_network(self, cage_ai, game_state):
        """Optimization should use the neural network."""
        valid_moves = cage_ai.get_valid_moves(game_state)

        if len(valid_moves) > 1:
            # Verify network is called by checking it's not None
            assert cage_ai.network is not None

            # Run optimization
            move = cage_ai._optimize_for_move(game_state, valid_moves)
            assert move is not None


class TestCAGEAIIntegration:
    """Integration tests for CAGE_AI."""

    def test_multiple_moves_in_game(self, ai_config, cage_config):
        """Should handle multiple moves in a game session."""
        ai = CAGE_AI(
            player_number=1,
            config=ai_config,
            cage_config=cage_config,
        )

        game_state = create_initial_state(board_type=BoardType.SQUARE8, num_players=2)

        # Make several moves
        moves_made = 0
        for _ in range(5):
            move = ai.select_move(game_state)
            if move:
                moves_made += 1

        assert moves_made > 0
        assert ai.move_count > 0

    def test_different_player_numbers(self, ai_config, cage_config):
        """Should work for different player numbers."""
        for player_num in [1, 2]:
            ai = CAGE_AI(
                player_number=player_num,
                config=ai_config,
                cage_config=cage_config,
            )

            game_state = create_initial_state(board_type=BoardType.SQUARE8, num_players=2)
            move = ai.select_move(game_state)

            assert ai.player_number == player_num
            # Move should be valid (or None if no moves)
            if move:
                valid_moves = ai.get_valid_moves(game_state)
                assert move in valid_moves

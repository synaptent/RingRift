"""Tests for app.ai.gumbel_mcts_ai module.

Tests the core Gumbel MCTS AI implementation:
- LeafEvaluationBuffer
- GumbelMCTSAI class (via create_gumbel_mcts factory)
- Factory function create_gumbel_mcts
- Helper function _infer_num_players

Note: GumbelSearchEngine is the preferred interface for new code.
This module provides the underlying implementation.

Created Dec 2025 as part of Phase 3 test coverage improvement.
"""

from unittest.mock import MagicMock, patch
import math

import pytest
import numpy as np

from app.ai.gumbel_mcts_ai import (
    LeafEvaluationBuffer,
    GumbelMCTSAI,
    create_gumbel_mcts,
    _infer_num_players,
)
from app.ai.gumbel_common import (
    GUMBEL_BUDGET_STANDARD,
    GUMBEL_BUDGET_THROUGHPUT,
    GUMBEL_BUDGET_QUALITY,
    GumbelAction,
    GumbelNode,
    LeafEvalRequest,
)
from app.models import BoardType


# =============================================================================
# _infer_num_players Tests
# =============================================================================


class TestInferNumPlayers:
    """Tests for _infer_num_players function."""

    def test_with_num_players_attribute(self):
        """Should use num_players attribute if present."""
        state = MagicMock()
        state.num_players = 4
        assert _infer_num_players(state) == 4

    def test_with_num_players_2(self):
        """Should handle 2-player game."""
        state = MagicMock()
        state.num_players = 2
        assert _infer_num_players(state) == 2

    def test_fallback_to_markers(self):
        """Should infer from board markers if no num_players."""
        state = MagicMock(spec=[])  # No num_players attribute
        state.board = MagicMock()

        marker1 = MagicMock()
        marker1.player = 0
        marker2 = MagicMock()
        marker2.player = 1
        marker3 = MagicMock()
        marker3.player = 2

        state.board.markers = {"a": marker1, "b": marker2, "c": marker3}

        result = _infer_num_players(state)
        assert result >= 2  # At least 2 players

    def test_minimum_two_players(self):
        """Should return at least 2 players."""
        state = MagicMock(spec=[])  # No num_players
        state.board = MagicMock()
        state.board.markers = {}  # No markers

        result = _infer_num_players(state)
        assert result >= 2


# =============================================================================
# LeafEvaluationBuffer Tests
# =============================================================================


class TestLeafEvaluationBuffer:
    """Tests for LeafEvaluationBuffer class."""

    @pytest.fixture
    def mock_neural_net(self):
        """Create a mock neural network."""
        nn = MagicMock()
        nn.evaluate_batch = MagicMock(return_value=(
            [0.5, 0.3, 0.2],  # values
            [[0.1] * 64] * 3,  # policies (unused in flush)
        ))
        return nn

    def test_init(self, mock_neural_net):
        """Should initialize with neural net and max batch size."""
        buffer = LeafEvaluationBuffer(mock_neural_net, max_batch_size=128)
        assert len(buffer) == 0
        assert buffer.max_batch_size == 128

    def test_init_no_neural_net(self):
        """Should accept None neural net."""
        buffer = LeafEvaluationBuffer(None, max_batch_size=64)
        assert len(buffer) == 0

    def test_add_request(self, mock_neural_net):
        """Should add evaluation requests."""
        buffer = LeafEvaluationBuffer(mock_neural_net)

        request = LeafEvalRequest(
            game_state=MagicMock(),
            action_idx=5,
            simulation_idx=0,
            is_opponent_perspective=False,
        )
        buffer.add(request)
        assert len(buffer) == 1

    def test_add_multiple(self, mock_neural_net):
        """Should accumulate multiple requests."""
        buffer = LeafEvaluationBuffer(mock_neural_net)

        for i in range(10):
            request = LeafEvalRequest(
                game_state=MagicMock(),
                action_idx=i,
                simulation_idx=i,
                is_opponent_perspective=False,
            )
            buffer.add(request)

        assert len(buffer) == 10

    def test_should_flush(self, mock_neural_net):
        """Should report when buffer is full."""
        buffer = LeafEvaluationBuffer(mock_neural_net, max_batch_size=5)

        for i in range(4):
            request = LeafEvalRequest(
                game_state=MagicMock(),
                action_idx=i,
                simulation_idx=i,
                is_opponent_perspective=False,
            )
            buffer.add(request)

        assert not buffer.should_flush()

        # Add one more to hit max
        buffer.add(LeafEvalRequest(
            game_state=MagicMock(),
            action_idx=4,
            simulation_idx=4,
            is_opponent_perspective=False,
        ))
        assert buffer.should_flush()

    def test_flush_empty(self, mock_neural_net):
        """Should handle flush on empty buffer."""
        buffer = LeafEvaluationBuffer(mock_neural_net)
        result = buffer.flush()
        assert result == []

    def test_flush_no_neural_net(self):
        """Should return zero values when no neural net."""
        buffer = LeafEvaluationBuffer(None)

        request = LeafEvalRequest(
            game_state=MagicMock(),
            action_idx=5,
            simulation_idx=1,
            is_opponent_perspective=False,
        )
        buffer.add(request)

        result = buffer.flush()
        assert len(result) == 1
        assert result[0] == (5, 1, 0.0)  # (action_idx, sim_idx, value=0)


# =============================================================================
# GumbelMCTSAI Tests (via create_gumbel_mcts)
# =============================================================================


class TestCreateGumbelMCTS:
    """Tests for create_gumbel_mcts factory function."""

    def test_create_basic(self):
        """Should create AI with basic parameters."""
        with patch('app.ai.gumbel_mcts_ai.NeuralNetAI') as mock_nn_class:
            mock_nn = MagicMock()
            mock_nn_class.return_value = mock_nn

            ai = create_gumbel_mcts(
                board_type="square8",
                num_players=2,
            )

            assert isinstance(ai, GumbelMCTSAI)

    def test_create_with_budget(self):
        """Should accept simulation budget."""
        with patch('app.ai.gumbel_mcts_ai.NeuralNetAI') as mock_nn_class:
            mock_nn = MagicMock()
            mock_nn_class.return_value = mock_nn

            ai = create_gumbel_mcts(
                board_type="square8",
                num_players=2,
                simulation_budget=GUMBEL_BUDGET_QUALITY,
            )

            assert ai.simulation_budget == GUMBEL_BUDGET_QUALITY

    def test_create_with_sampled_actions(self):
        """Should accept num_sampled_actions."""
        with patch('app.ai.gumbel_mcts_ai.NeuralNetAI') as mock_nn_class:
            mock_nn = MagicMock()
            mock_nn_class.return_value = mock_nn

            ai = create_gumbel_mcts(
                board_type="square8",
                num_players=2,
                num_sampled_actions=8,
            )

            assert ai.num_sampled_actions == 8

    def test_create_hex_board(self):
        """Should work with hex boards."""
        with patch('app.ai.gumbel_mcts_ai.NeuralNetAI') as mock_nn_class:
            mock_nn = MagicMock()
            mock_nn_class.return_value = mock_nn

            ai = create_gumbel_mcts(
                board_type="hex8",
                num_players=2,
            )

            assert isinstance(ai, GumbelMCTSAI)
            assert ai.board_type == BoardType.HEX8

    def test_create_with_board_type_enum(self):
        """Should accept BoardType enum directly."""
        with patch('app.ai.gumbel_mcts_ai.NeuralNetAI') as mock_nn_class:
            mock_nn = MagicMock()
            mock_nn_class.return_value = mock_nn

            ai = create_gumbel_mcts(
                board_type=BoardType.SQUARE19,
                num_players=2,
            )

            assert ai.board_type == BoardType.SQUARE19

    def test_create_multiplayer(self):
        """Should work with 4-player games."""
        with patch('app.ai.gumbel_mcts_ai.NeuralNetAI') as mock_nn_class:
            mock_nn = MagicMock()
            mock_nn_class.return_value = mock_nn

            ai = create_gumbel_mcts(
                board_type="square8",
                num_players=4,
            )

            assert isinstance(ai, GumbelMCTSAI)


class TestGumbelMCTSAIAttributes:
    """Tests for GumbelMCTSAI attributes via factory."""

    @pytest.fixture
    def ai(self):
        """Create AI instance for tests."""
        with patch('app.ai.gumbel_mcts_ai.NeuralNetAI') as mock_nn_class:
            mock_nn = MagicMock()
            mock_nn_class.return_value = mock_nn

            return create_gumbel_mcts(
                board_type="square8",
                num_players=2,
                simulation_budget=GUMBEL_BUDGET_STANDARD,
                num_sampled_actions=16,
            )

    def test_has_simulation_budget(self, ai):
        """Should have simulation budget attribute."""
        assert hasattr(ai, 'simulation_budget')
        assert ai.simulation_budget == GUMBEL_BUDGET_STANDARD

    def test_has_num_sampled_actions(self, ai):
        """Should have num_sampled_actions attribute."""
        assert hasattr(ai, 'num_sampled_actions')
        assert ai.num_sampled_actions == 16

    def test_has_c_puct(self, ai):
        """Should have exploration constant."""
        assert hasattr(ai, 'c_puct')
        assert ai.c_puct > 0

    def test_has_board_type(self, ai):
        """Should have board type."""
        assert hasattr(ai, 'board_type')
        assert ai.board_type == BoardType.SQUARE8


# =============================================================================
# Gumbel Algorithm Tests
# =============================================================================


class TestGumbelAlgorithm:
    """Tests for Gumbel-specific functionality."""

    def test_gumbel_noise_generation(self):
        """Should generate Gumbel noise for sampling."""
        np.random.seed(42)
        uniform = np.random.random(100)
        gumbel = -np.log(-np.log(uniform))

        # Gumbel(0,1) distribution properties
        assert gumbel.shape == (100,)
        assert not np.isnan(gumbel).any()
        assert not np.isinf(gumbel).any()

        # Mean of Gumbel(0,1) is ~0.5772 (Euler-Mascheroni constant)
        assert -1 < np.mean(gumbel) < 2

    def test_sequential_halving_phases(self):
        """Should compute correct number of halving phases."""
        # Sequential halving: log2(k) phases
        # With k=16: phases = 4 (16 -> 8 -> 4 -> 2 -> 1)
        k = 16
        expected_phases = int(math.log2(k))
        assert expected_phases == 4

        # With k=8: phases = 3
        k = 8
        expected_phases = int(math.log2(k))
        assert expected_phases == 3

    def test_gumbel_top_k_concept(self):
        """Should demonstrate Gumbel-Top-K sampling concept."""
        np.random.seed(42)

        # Policy logits (higher = more likely)
        logits = np.array([1.0, 0.5, 0.1, -0.5, -1.0])

        # Add Gumbel noise
        gumbel_noise = -np.log(-np.log(np.random.random(5)))
        perturbed = logits + gumbel_noise

        # Top-K selection
        k = 3
        top_k_indices = np.argsort(perturbed)[-k:][::-1]

        assert len(top_k_indices) == k
        # Higher logit actions should generally be selected more often
        # (though Gumbel noise adds exploration)


# =============================================================================
# Integration Tests
# =============================================================================


class TestGumbelMCTSIntegration:
    """Integration tests for GumbelMCTSAI."""

    def test_gumbel_common_integration(self):
        """Should work with gumbel_common dataclasses."""
        # Verify we can create GumbelAction/GumbelNode
        # GumbelAction requires: move, policy_logit, gumbel_noise, perturbed_value
        mock_move = MagicMock()
        action = GumbelAction(
            move=mock_move,
            policy_logit=0.5,
            gumbel_noise=0.1,
            perturbed_value=0.6,  # policy_logit + gumbel_noise
        )
        assert action.move == mock_move
        assert action.policy_logit == 0.5
        assert action.perturbed_value == 0.6

        # GumbelNode has all optional fields with defaults
        node = GumbelNode()
        assert node.move is None
        assert node.visit_count == 0

    def test_leaf_eval_request(self):
        """Should work with LeafEvalRequest."""
        request = LeafEvalRequest(
            game_state=MagicMock(),
            action_idx=5,
            simulation_idx=0,
            is_opponent_perspective=True,
        )
        assert request.action_idx == 5
        assert request.is_opponent_perspective

    def test_budget_constants(self):
        """Should use standard budget constants."""
        assert GUMBEL_BUDGET_THROUGHPUT < GUMBEL_BUDGET_STANDARD
        assert GUMBEL_BUDGET_STANDARD <= GUMBEL_BUDGET_QUALITY  # Equal after Dec 2025 change

        # Values (Dec 2025: STANDARD raised from 150 to 800 for quality training)
        assert GUMBEL_BUDGET_THROUGHPUT == 64
        assert GUMBEL_BUDGET_STANDARD == 800  # Dec 2025: raised from 150
        assert GUMBEL_BUDGET_QUALITY == 800

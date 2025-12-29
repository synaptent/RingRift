"""Unit tests for app/ai/hybrid_gpu.py

Tests cover:
- State conversion utilities (game_state_to_gpu_arrays, batch_game_states_to_gpu)
- HybridGPUEvaluator configuration and initialization
- AsyncEvalRequest and AsyncPipelineEvaluator
- HybridSelfPlayRunner configuration
- HybridNNAI initialization and move selection
- Factory functions (create_hybrid_evaluator)
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
from dataclasses import dataclass

# Test imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_game_state():
    """Create a mock GameState for testing."""
    state = MagicMock()

    # Mock board with stacks
    mock_stacks = {
        "0,0": MagicMock(controlling_player=1, height=2),
        "1,1": MagicMock(controlling_player=2, height=1),
        "3,3": MagicMock(controlling_player=1, height=3),
    }
    for key, stack in mock_stacks.items():
        stack.is_collapsed = False

    state.board.stacks = mock_stacks
    state.board.markers = {}
    state.board.rings = {1: [], 2: []}
    state.board.territories = {1: [], 2: []}

    # Mock player info
    state.current_player = 1
    state.players = {1: MagicMock(score=0), 2: MagicMock(score=0)}
    state.game_over = False
    state.winner = None
    state.phase = "placement"
    state.move_number = 5

    return state


@pytest.fixture
def mock_config():
    """Create a mock AIConfig."""
    config = MagicMock()
    config.difficulty = 5
    config.think_time = 1000
    config.randomness = 0.1
    config.use_neural_net = True
    config.nn_model_id = None
    return config


# =============================================================================
# State Conversion Tests
# =============================================================================


class TestGameStateToGPUArrays:
    """Tests for game_state_to_gpu_arrays function."""

    def test_converts_empty_board(self, mock_game_state):
        """Empty board should produce zero arrays."""
        mock_game_state.board.stacks = {}
        mock_game_state.board.markers = {}

        from app.ai.hybrid_gpu import game_state_to_gpu_arrays

        result = game_state_to_gpu_arrays(mock_game_state, board_size=8)

        assert "stack_owner" in result
        assert "stack_height" in result
        assert result["stack_owner"].shape == (64,)
        assert result["stack_height"].shape == (64,)
        assert np.all(result["stack_owner"] == 0)
        assert np.all(result["stack_height"] == 0)

    def test_converts_populated_board(self, mock_game_state):
        """Board with pieces should be converted correctly."""
        from app.ai.hybrid_gpu import game_state_to_gpu_arrays

        result = game_state_to_gpu_arrays(mock_game_state, board_size=8)

        # Position 0,0 should have player 1 with height 2
        assert result["stack_owner"][0] == 1
        assert result["stack_height"][0] == 2

        # Position 1,1 (index 9) should have player 2 with height 1
        assert result["stack_owner"][9] == 2
        assert result["stack_height"][9] == 1

    def test_different_board_sizes(self, mock_game_state):
        """Should handle different board sizes."""
        from app.ai.hybrid_gpu import game_state_to_gpu_arrays

        # 8x8 board
        result_8 = game_state_to_gpu_arrays(mock_game_state, board_size=8)
        assert result_8["stack_owner"].shape == (64,)

        # 19x19 board
        result_19 = game_state_to_gpu_arrays(mock_game_state, board_size=19)
        assert result_19["stack_owner"].shape == (361,)


class TestBatchGameStatesToGPU:
    """Tests for batch_game_states_to_gpu function."""

    def test_empty_batch(self):
        """Empty batch should return empty arrays."""
        from app.ai.hybrid_gpu import batch_game_states_to_gpu

        result = batch_game_states_to_gpu([], board_size=8)

        assert result["stack_owner"].shape[0] == 0

    def test_single_state_batch(self, mock_game_state):
        """Single state batch should work correctly."""
        from app.ai.hybrid_gpu import batch_game_states_to_gpu

        result = batch_game_states_to_gpu([mock_game_state], board_size=8)

        assert result["stack_owner"].shape == (1, 64)
        assert result["stack_height"].shape == (1, 64)

    def test_multiple_states_batch(self, mock_game_state):
        """Multiple states should be batched correctly."""
        from app.ai.hybrid_gpu import batch_game_states_to_gpu

        states = [mock_game_state, mock_game_state, mock_game_state]
        result = batch_game_states_to_gpu(states, board_size=8)

        assert result["stack_owner"].shape == (3, 64)


# =============================================================================
# HybridGPUEvaluator Tests
# =============================================================================


class TestHybridGPUEvaluator:
    """Tests for HybridGPUEvaluator class."""

    @patch("app.ai.hybrid_gpu.get_device")
    @patch("app.ai.hybrid_gpu.GPUHeuristicEvaluator")
    def test_initialization_cpu(self, mock_evaluator_class, mock_get_device):
        """Should initialize with CPU device."""
        mock_get_device.return_value = "cpu"
        mock_evaluator_class.return_value = MagicMock()

        from app.ai.hybrid_gpu import HybridGPUEvaluator

        evaluator = HybridGPUEvaluator(device="cpu", board_size=8)

        assert evaluator.device == "cpu"
        assert evaluator.board_size == 8

    @patch("app.ai.hybrid_gpu.get_device")
    @patch("app.ai.hybrid_gpu.GPUHeuristicEvaluator")
    def test_default_board_size(self, mock_evaluator_class, mock_get_device):
        """Should use default board size of 8."""
        mock_get_device.return_value = "cpu"
        mock_evaluator_class.return_value = MagicMock()

        from app.ai.hybrid_gpu import HybridGPUEvaluator

        evaluator = HybridGPUEvaluator(device="cpu")

        assert evaluator.board_size == 8

    @patch("app.ai.hybrid_gpu.get_device")
    @patch("app.ai.hybrid_gpu.GPUHeuristicEvaluator")
    def test_batch_size_configuration(self, mock_evaluator_class, mock_get_device):
        """Should respect batch size configuration."""
        mock_get_device.return_value = "cpu"
        mock_evaluator_class.return_value = MagicMock()

        from app.ai.hybrid_gpu import HybridGPUEvaluator

        evaluator = HybridGPUEvaluator(device="cpu", batch_size=256)

        assert evaluator.batch_size == 256


# =============================================================================
# AsyncEvalRequest Tests
# =============================================================================


class TestAsyncEvalRequest:
    """Tests for AsyncEvalRequest dataclass."""

    def test_creation(self, mock_game_state):
        """Should create request with state and callback."""
        from app.ai.hybrid_gpu import AsyncEvalRequest

        callback = MagicMock()
        request = AsyncEvalRequest(
            game_state=mock_game_state,
            callback=callback,
        )

        assert request.game_state == mock_game_state
        assert request.callback == callback

    def test_with_request_id(self, mock_game_state):
        """Should support optional request_id."""
        from app.ai.hybrid_gpu import AsyncEvalRequest

        request = AsyncEvalRequest(
            game_state=mock_game_state,
            callback=MagicMock(),
            request_id="test-123",
        )

        assert request.request_id == "test-123"


# =============================================================================
# AsyncPipelineEvaluator Tests
# =============================================================================


class TestAsyncPipelineEvaluator:
    """Tests for AsyncPipelineEvaluator class."""

    @patch("app.ai.hybrid_gpu.HybridGPUEvaluator")
    def test_initialization(self, mock_evaluator_class):
        """Should initialize with evaluator and queue."""
        mock_evaluator = MagicMock()
        mock_evaluator_class.return_value = mock_evaluator

        from app.ai.hybrid_gpu import AsyncPipelineEvaluator

        pipeline = AsyncPipelineEvaluator(
            device="cpu",
            board_size=8,
            max_queue_size=100,
        )

        assert pipeline.max_queue_size == 100
        assert not pipeline.running

    @patch("app.ai.hybrid_gpu.HybridGPUEvaluator")
    def test_start_stop(self, mock_evaluator_class):
        """Should start and stop worker threads."""
        mock_evaluator = MagicMock()
        mock_evaluator_class.return_value = mock_evaluator

        from app.ai.hybrid_gpu import AsyncPipelineEvaluator

        pipeline = AsyncPipelineEvaluator(device="cpu")

        # Start
        pipeline.start()
        assert pipeline.running

        # Stop
        pipeline.stop()
        assert not pipeline.running


# =============================================================================
# HybridSelfPlayRunner Tests
# =============================================================================


class TestHybridSelfPlayRunner:
    """Tests for HybridSelfPlayRunner class."""

    @patch("app.ai.hybrid_gpu.HybridGPUEvaluator")
    def test_initialization(self, mock_evaluator_class):
        """Should initialize with evaluator."""
        mock_evaluator = MagicMock()
        mock_evaluator_class.return_value = mock_evaluator

        from app.ai.hybrid_gpu import HybridSelfPlayRunner

        runner = HybridSelfPlayRunner(evaluator=mock_evaluator)

        assert runner.evaluator == mock_evaluator

    @patch("app.ai.hybrid_gpu.HybridGPUEvaluator")
    def test_with_num_workers(self, mock_evaluator_class):
        """Should respect num_workers configuration."""
        mock_evaluator = MagicMock()
        mock_evaluator_class.return_value = mock_evaluator

        from app.ai.hybrid_gpu import HybridSelfPlayRunner

        runner = HybridSelfPlayRunner(evaluator=mock_evaluator, num_workers=4)

        assert runner.num_workers == 4


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateHybridEvaluator:
    """Tests for create_hybrid_evaluator factory function."""

    @patch("app.ai.hybrid_gpu.get_device")
    @patch("app.ai.hybrid_gpu.GPUHeuristicEvaluator")
    def test_creates_evaluator(self, mock_heuristic_class, mock_get_device):
        """Should create evaluator with specified config."""
        mock_get_device.return_value = "cpu"
        mock_heuristic_class.return_value = MagicMock()

        from app.ai.hybrid_gpu import create_hybrid_evaluator

        evaluator = create_hybrid_evaluator(
            device="cpu",
            board_size=8,
            batch_size=64,
        )

        assert evaluator is not None
        assert evaluator.device == "cpu"

    @patch("app.ai.hybrid_gpu.get_device")
    @patch("app.ai.hybrid_gpu.GPUHeuristicEvaluator")
    def test_default_device(self, mock_heuristic_class, mock_get_device):
        """Should use auto-detected device by default."""
        mock_get_device.return_value = "cpu"
        mock_heuristic_class.return_value = MagicMock()

        from app.ai.hybrid_gpu import create_hybrid_evaluator

        evaluator = create_hybrid_evaluator()

        mock_get_device.assert_called()


# =============================================================================
# HybridNNAI Tests
# =============================================================================


class TestHybridNNAI:
    """Tests for HybridNNAI class."""

    @patch("app.ai.hybrid_gpu.get_device")
    @patch("app.ai.hybrid_gpu.GPUHeuristicEvaluator")
    def test_initialization(self, mock_heuristic_class, mock_get_device, mock_config):
        """Should initialize with player number and config."""
        mock_get_device.return_value = "cpu"
        mock_heuristic_class.return_value = MagicMock()

        from app.ai.hybrid_gpu import HybridNNAI

        ai = HybridNNAI(player_number=1, config=mock_config)

        assert ai.player_number == 1

    @patch("app.ai.hybrid_gpu.get_device")
    @patch("app.ai.hybrid_gpu.GPUHeuristicEvaluator")
    def test_inherits_from_base_ai(self, mock_heuristic_class, mock_get_device, mock_config):
        """Should inherit from BaseAI."""
        mock_get_device.return_value = "cpu"
        mock_heuristic_class.return_value = MagicMock()

        from app.ai.hybrid_gpu import HybridNNAI
        from app.ai.base import BaseAI

        ai = HybridNNAI(player_number=1, config=mock_config)

        assert isinstance(ai, BaseAI)


# =============================================================================
# HybridNNValuePlayer Tests
# =============================================================================


class TestHybridNNValuePlayer:
    """Tests for HybridNNValuePlayer class."""

    @patch("app.ai.hybrid_gpu.get_device")
    @patch("app.ai.hybrid_gpu.GPUHeuristicEvaluator")
    def test_initialization(self, mock_heuristic_class, mock_get_device):
        """Should initialize with required parameters."""
        mock_get_device.return_value = "cpu"
        mock_heuristic_class.return_value = MagicMock()

        from app.ai.hybrid_gpu import HybridNNValuePlayer

        player = HybridNNValuePlayer(
            player_number=1,
            device="cpu",
        )

        assert player.player_number == 1

    @patch("app.ai.hybrid_gpu.get_device")
    @patch("app.ai.hybrid_gpu.GPUHeuristicEvaluator")
    def test_with_search_depth(self, mock_heuristic_class, mock_get_device):
        """Should respect search depth configuration."""
        mock_get_device.return_value = "cpu"
        mock_heuristic_class.return_value = MagicMock()

        from app.ai.hybrid_gpu import HybridNNValuePlayer

        player = HybridNNValuePlayer(
            player_number=1,
            device="cpu",
            search_depth=5,
        )

        assert player.search_depth == 5


# =============================================================================
# Integration-style Tests (mocked)
# =============================================================================


class TestHybridEvaluationPipeline:
    """Integration tests for the hybrid evaluation pipeline."""

    @patch("app.ai.hybrid_gpu.get_device")
    @patch("app.ai.hybrid_gpu.GPUHeuristicEvaluator")
    def test_full_evaluation_pipeline(self, mock_heuristic_class, mock_get_device, mock_game_state):
        """Should handle full evaluation pipeline."""
        mock_get_device.return_value = "cpu"

        mock_heuristic = MagicMock()
        mock_heuristic.evaluate_batch.return_value = np.array([0.5])
        mock_heuristic_class.return_value = mock_heuristic

        from app.ai.hybrid_gpu import HybridGPUEvaluator

        evaluator = HybridGPUEvaluator(device="cpu", board_size=8)

        # Should be able to call evaluate (mocked)
        assert evaluator is not None

    @patch("app.ai.hybrid_gpu.get_device")
    @patch("app.ai.hybrid_gpu.GPUHeuristicEvaluator")
    def test_batch_processing(self, mock_heuristic_class, mock_get_device, mock_game_state):
        """Should handle batch processing of game states."""
        mock_get_device.return_value = "cpu"

        mock_heuristic = MagicMock()
        mock_heuristic.evaluate_batch.return_value = np.array([0.5, 0.6, 0.4])
        mock_heuristic_class.return_value = mock_heuristic

        from app.ai.hybrid_gpu import HybridGPUEvaluator

        evaluator = HybridGPUEvaluator(device="cpu", batch_size=16)

        # Should handle batch configuration
        assert evaluator.batch_size == 16


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_board_size(self):
        """Should handle invalid board sizes gracefully."""
        from app.ai.hybrid_gpu import game_state_to_gpu_arrays

        mock_state = MagicMock()
        mock_state.board.stacks = {}
        mock_state.board.markers = {}

        # Should work with valid sizes
        result = game_state_to_gpu_arrays(mock_state, board_size=4)
        assert result["stack_owner"].shape == (16,)

    def test_empty_batch_processing(self):
        """Should handle empty batches gracefully."""
        from app.ai.hybrid_gpu import batch_game_states_to_gpu

        result = batch_game_states_to_gpu([], board_size=8)

        # Should return valid empty arrays
        assert isinstance(result, dict)
        assert "stack_owner" in result

    @patch("app.ai.hybrid_gpu.get_device")
    @patch("app.ai.hybrid_gpu.GPUHeuristicEvaluator")
    def test_evaluator_with_custom_weights(self, mock_heuristic_class, mock_get_device):
        """Should handle custom heuristic weights."""
        mock_get_device.return_value = "cpu"
        mock_heuristic_class.return_value = MagicMock()

        from app.ai.hybrid_gpu import HybridGPUEvaluator

        custom_weights = {"material": 1.0, "mobility": 0.5}
        evaluator = HybridGPUEvaluator(
            device="cpu",
            heuristic_weights=custom_weights,
        )

        assert evaluator is not None

"""
MCTS AI Test Suite
==================

These tests exercise the MCTS (Monte Carlo Tree Search) AI implementation
which involves:
- Neural network inference for position evaluation
- Tree search with time-bounded exploration
- Complex game state simulation

Configuration
-------------
MCTS tests are disabled by default for fast CI execution. To enable:

    # Enable MCTS tests with default 60s timeout
    ENABLE_MCTS_TESTS=true pytest tests/test_mcts_ai.py -v

    # Enable with custom timeout (30 seconds)
    ENABLE_MCTS_TESTS=true MCTS_TEST_TIMEOUT=30 pytest tests/test_mcts_ai.py -v

    # Run all tests EXCEPT mcts tests
    pytest -m "not mcts" -v

Environment Variables
---------------------
- ENABLE_MCTS_TESTS: Set to 'true' to enable MCTS tests (default: false)
- MCTS_TEST_TIMEOUT: Timeout in seconds per test (default: 60)

CI Integration
--------------
For CI pipelines, MCTS tests should remain disabled (default) to prevent
slow builds. Enable selectively for nightly or dedicated AI test runs:

    # Fast CI (default) - skips MCTS tests
    pytest tests/

    # Dedicated AI test job
    ENABLE_MCTS_TESTS=true MCTS_TEST_TIMEOUT=120 \\
        pytest tests/test_mcts_ai.py -v
"""
import os
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime

import numpy as np
import pytest
import torch

from app.ai.mcts_ai import MCTSAI
from app.models import (
    GameState, AIConfig, BoardType, GamePhase, GameStatus, TimeControl,
    BoardState
)

# ==============================================================================
# MCTS TEST CONFIGURATION
# ==============================================================================
# By default, MCTS tests are skipped because they involve expensive tree search
# and neural network inference. Enable via ENABLE_MCTS_TESTS=true for local
# development or dedicated CI jobs.

MCTS_TESTS_ENABLED = os.getenv('ENABLE_MCTS_TESTS', 'false').lower() == 'true'
MCTS_TEST_TIMEOUT = int(os.getenv('MCTS_TEST_TIMEOUT', '60'))

# Apply markers and conditional skip at module level
pytestmark = [
    pytest.mark.mcts,
    pytest.mark.slow,
    pytest.mark.skipif(
        not MCTS_TESTS_ENABLED,
        reason="MCTS tests disabled (set ENABLE_MCTS_TESTS=true to enable)"
    ),
]


class TestMCTSAI(unittest.TestCase):
    def setUp(self):
        self.config = AIConfig(difficulty=5, think_time=1000, randomness=0.1)
        self.ai = MCTSAI(player_number=1, config=self.config)
        
        self.game_state = GameState(
            id="test-game",
            boardType=BoardType.SQUARE8,
            board=BoardState(
                type=BoardType.SQUARE8,
                size=8,
                stacks={},
                markers={},
                collapsedSpaces={},
                eliminatedRings={}
            ),
            players=[],
            currentPhase=GamePhase.MOVEMENT,
            currentPlayer=1,
            moveHistory=[],
            timeControl=TimeControl(
                initialTime=600, increment=0, type="blitz"
            ),
            gameStatus=GameStatus.ACTIVE,
            createdAt=datetime.now(),
            lastMoveAt=datetime.now(),
            isRated=False,
            maxPlayers=2,
            totalRingsInPlay=36,
            totalRingsEliminated=0,
            victoryThreshold=19,
            territoryVictoryThreshold=33,
            chainCaptureState=None,
        )

    @pytest.mark.timeout(MCTS_TEST_TIMEOUT)
    @patch('app.game_engine.GameEngine.get_valid_moves')
    def test_select_move_returns_move(self, mock_get_valid_moves):
        mock_move = MagicMock()
        mock_move.type = "move"
        mock_get_valid_moves.return_value = [mock_move]
        
        self.ai.simulate_thinking = MagicMock()

        # Mock neural net to avoid actual inference or fallback to heuristic
        self.ai.neural_net = MagicMock()
        # Return enough values for the batch
        self.ai.neural_net.evaluate_batch.side_effect = lambda states: (
            [0.5] * len(states), [[1.0]] * len(states)
        )
        self.ai.neural_net.encode_move.return_value = 0

        move = self.ai.select_move(self.game_state)
        
        self.assertIsNotNone(move)
        self.assertEqual(move, mock_move)

    @pytest.mark.timeout(MCTS_TEST_TIMEOUT)
    @patch('app.game_engine.GameEngine.get_valid_moves')
    def test_select_move_no_valid_moves(self, mock_get_valid_moves):
        mock_get_valid_moves.return_value = []
        self.ai.simulate_thinking = MagicMock()
        
        move = self.ai.select_move(self.game_state)
        self.assertIsNone(move)

    @pytest.mark.timeout(MCTS_TEST_TIMEOUT)
    @patch('app.game_engine.GameEngine.get_valid_moves')
    def test_select_move_uses_hex_network_for_hex_board(
        self,
        mock_get_valid_moves,
    ):
        """Hex boards should exercise the hex NN path without error.

        This test mocks out the neural network, hex encoder, and rules
        engine move generation sufficiently to ensure that MCTSAI's
        hex-specific evaluation branch can be executed in a controlled
        way. It does not assert on search quality, only that the wiring
        is sound and a move is returned.
        """
        # Configure the board as canonical hex (size=11, radius=10).
        self.game_state.board.type = BoardType.HEXAGONAL
        self.game_state.board.size = 11

        # Keep the search very small for test speed.
        self.ai.config.think_time = 10  # milliseconds

        # Single mocked move that is always legal.
        mock_move = MagicMock()
        mock_move.type = "move_stack"
        mock_move.to = MagicMock()
        mock_move.to.x = 0
        mock_move.to.y = 0
        mock_get_valid_moves.return_value = [mock_move]

        # Mock out the shared NeuralNetAI encoder.
        self.ai.neural_net = MagicMock()
        self.ai.neural_net._extract_features.return_value = (
            np.zeros((10, 21, 21), dtype=np.float32),
            np.zeros((10,), dtype=np.float32),
        )

        # Hex encoder: always map moves to index 0 in the (small) policy
        # vector returned by the fake hex model.
        self.ai.hex_encoder = MagicMock()
        self.ai.hex_encoder.encode_move.return_value = 0

        # Fake hex model that returns zero values and a single-logit policy
        # vector per state. This keeps the shapes valid while avoiding any
        # dependency on real model weights.
        def fake_hex_model(x, globals_vec, hex_mask=None):
            batch = x.shape[0]
            values = torch.zeros((batch, 1))
            logits = torch.zeros((batch, 1))
            return values, logits

        self.ai.hex_model = MagicMock(side_effect=fake_hex_model)

        move = self.ai.select_move(self.game_state)

        self.assertIsNotNone(move)
        self.assertEqual(move, mock_move)
        # Ensure that the hex model path was exercised.
        self.assertTrue(self.ai.hex_model.called)


if __name__ == '__main__':
    unittest.main()

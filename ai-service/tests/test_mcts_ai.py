"""
MCTS AI Test Suite
==================

These tests exercise the MCTS (Monte Carlo Tree Search) AI implementation
which involves:
- Neural network inference for position evaluation
- Tree search with time-bounded exploration
- Complex game state simulation
- Make/unmake pattern for incremental search

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

from app.ai.mcts_ai import MCTSAI, MCTSNodeLite
from app.models import (
    GameState, AIConfig, BoardType, GamePhase, GameStatus, TimeControl,
    BoardState, Move, MoveType, Position, Player, RingStack
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
        
        # Force legacy path for this test to ensure hex_model is exercised
        self.ai.use_incremental_search = False

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
        # Mock evaluate_batch for the incremental search path
        self.ai.neural_net.evaluate_batch.side_effect = lambda states: (
            [0.5] * len(states), [[1.0]] * len(states)
        )
        # Mock encode_move to return integer
        self.ai.neural_net.encode_move.return_value = 0

        # Hex encoder: always map moves to index 0 in the (small) policy
        # vector returned by the fake hex model.
        self.ai.hex_encoder = MagicMock()
        self.ai.hex_encoder.encode_move.return_value = 0  # Must be int

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


class TestMCTSNodeLite(unittest.TestCase):
    """Test the lightweight MCTSNodeLite class for incremental search."""
    
    def test_init_defaults(self):
        """Test MCTSNodeLite initialization with defaults."""
        node = MCTSNodeLite()
        self.assertIsNone(node.parent)
        self.assertIsNone(node.move)
        self.assertEqual(len(node.children), 0)
        self.assertEqual(node.wins, 0.0)
        self.assertEqual(node.visits, 0)
        self.assertEqual(node.amaf_wins, 0.0)
        self.assertEqual(node.amaf_visits, 0)
        self.assertEqual(node.prior, 0.0)
        self.assertEqual(len(node.untried_moves), 0)
        self.assertEqual(len(node.policy_map), 0)
    
    def test_is_leaf(self):
        """Test is_leaf detection."""
        node = MCTSNodeLite()
        self.assertTrue(node.is_leaf())
        
        # Add a child
        mock_move = MagicMock(spec=Move)
        mock_move.type = MoveType.PLACE_RING
        mock_move.to = Position(x=0, y=0)
        child = node.add_child(mock_move)
        self.assertFalse(node.is_leaf())
        self.assertTrue(child.is_leaf())
    
    def test_is_fully_expanded(self):
        """Test is_fully_expanded detection."""
        node = MCTSNodeLite()
        self.assertTrue(node.is_fully_expanded())
        
        # Add untried moves
        mock_move = MagicMock(spec=Move)
        mock_move.type = MoveType.PLACE_RING
        mock_move.to = Position(x=0, y=0)
        node.untried_moves = [mock_move]
        self.assertFalse(node.is_fully_expanded())
    
    def test_add_child(self):
        """Test adding child nodes."""
        parent = MCTSNodeLite()
        mock_move = MagicMock(spec=Move)
        mock_move.type = MoveType.PLACE_RING
        mock_move.to = Position(x=0, y=0)
        parent.untried_moves = [mock_move]
        
        child = parent.add_child(mock_move, prior=0.5)
        
        self.assertEqual(len(parent.children), 1)
        self.assertEqual(parent.children[0], child)
        self.assertEqual(child.parent, parent)
        self.assertEqual(child.move, mock_move)
        self.assertEqual(child.prior, 0.5)
        self.assertEqual(len(parent.untried_moves), 0)
    
    def test_update(self):
        """Test node update with result."""
        node = MCTSNodeLite()
        node.update(1.0)
        
        self.assertEqual(node.visits, 1)
        self.assertEqual(node.wins, 1.0)
        
        node.update(-0.5)
        self.assertEqual(node.visits, 2)
        self.assertEqual(node.wins, 0.5)
    
    def test_uct_select_child(self):
        """Test UCT child selection."""
        parent = MCTSNodeLite()
        parent.visits = 10
        
        # Add two children with different stats
        move1 = MagicMock(spec=Move)
        move1.type = MoveType.PLACE_RING
        move1.to = Position(x=0, y=0)
        move2 = MagicMock(spec=Move)
        move2.type = MoveType.PLACE_RING
        move2.to = Position(x=1, y=1)
        
        parent.untried_moves = [move1, move2]
        child1 = parent.add_child(move1)
        child2 = parent.add_child(move2)
        
        # Child1: low visits, high win rate
        child1.visits = 2
        child1.wins = 1.5
        
        # Child2: high visits, lower win rate
        child2.visits = 5
        child2.wins = 2.0
        
        # UCT should prefer exploration vs exploitation balance
        selected = parent.uct_select_child()
        self.assertIn(selected, [child1, child2])


class TestMCTSIncrementalSearch(unittest.TestCase):
    """Test MCTS incremental search using make/unmake pattern."""
    
    def setUp(self):
        self.config = AIConfig(difficulty=5, think_time=100, randomness=0.0)
        
        # Create a game state with some stacks to enable movement
        stacks = {
            "0,0": RingStack(
                position=Position(x=0, y=0),
                rings=[1],
                stackHeight=1,
                capHeight=1,
                controllingPlayer=1
            ),
            "1,1": RingStack(
                position=Position(x=1, y=1),
                rings=[2],
                stackHeight=1,
                capHeight=1,
                controllingPlayer=2
            ),
        }
        
        self.game_state = GameState(
            id="test-game",
            boardType=BoardType.SQUARE8,
            board=BoardState(
                type=BoardType.SQUARE8,
                size=8,
                stacks=stacks,
                markers={},
                collapsedSpaces={},
                eliminatedRings={}
            ),
            players=[
                Player(
                    id="1", username="Player1", type="ai",
                    playerNumber=1, isReady=True, timeRemaining=0,
                    ringsInHand=17, eliminatedRings=0, territorySpaces=0
                ),
                Player(
                    id="2", username="Player2", type="ai",
                    playerNumber=2, isReady=True, timeRemaining=0,
                    ringsInHand=17, eliminatedRings=0, territorySpaces=0
                ),
            ],
            currentPhase=GamePhase.RING_PLACEMENT,
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
    def test_incremental_search_enabled_by_default(self):
        """Test that incremental search is enabled by default."""
        ai = MCTSAI(player_number=1, config=self.config)
        self.assertTrue(ai.use_incremental_search)

    @pytest.mark.timeout(MCTS_TEST_TIMEOUT)
    def test_incremental_search_can_be_disabled(self):
        """Test that incremental search can be disabled via config."""
        config = AIConfig(difficulty=5, think_time=100, randomness=0.0)
        # Manually set use_incremental_search on config
        config.use_incremental_search = False
        ai = MCTSAI(player_number=1, config=config)
        self.assertFalse(ai.use_incremental_search)

    def _create_test_move(self, x: int = 2, y: int = 2) -> Move:
        """Create a valid Move object for testing."""
        return Move(
            id="test-move-1",
            type=MoveType.PLACE_RING,
            player=1,
            to=Position(x=x, y=y),
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=1,
        )

    @pytest.mark.timeout(MCTS_TEST_TIMEOUT)
    @patch('app.game_engine.GameEngine.get_valid_moves')
    def test_incremental_search_returns_valid_move(self, mock_get_valid_moves):
        """Test that incremental search returns a valid move."""
        real_move = self._create_test_move()
        mock_get_valid_moves.return_value = [real_move]
        
        ai = MCTSAI(player_number=1, config=self.config)
        ai.neural_net = None  # Force heuristic rollout path
        
        move = ai.select_move(self.game_state)
        
        self.assertIsNotNone(move)
        self.assertEqual(move, real_move)

    @pytest.mark.timeout(MCTS_TEST_TIMEOUT)
    @patch('app.game_engine.GameEngine.get_valid_moves')
    def test_legacy_search_returns_valid_move(self, mock_get_valid_moves):
        """Test that legacy search returns a valid move."""
        real_move = self._create_test_move()
        mock_get_valid_moves.return_value = [real_move]
        
        config = AIConfig(difficulty=5, think_time=100, randomness=0.0)
        config.use_incremental_search = False
        ai = MCTSAI(player_number=1, config=config)
        ai.neural_net = None  # Force heuristic rollout path
        
        move = ai.select_move(self.game_state)
        
        self.assertIsNotNone(move)
        self.assertEqual(move, real_move)

    @pytest.mark.timeout(MCTS_TEST_TIMEOUT)
    def test_both_paths_can_be_configured(self):
        """Test that both incremental and legacy paths can be configured.
        
        This test verifies the configuration options work correctly.
        Actual move selection is tested in separate tests with proper mocks.
        """
        # Test incremental path configuration
        config_incr = AIConfig(difficulty=5, think_time=50, randomness=0.0)
        ai_incr = MCTSAI(player_number=1, config=config_incr)
        self.assertTrue(ai_incr.use_incremental_search)
        self.assertIsNone(ai_incr.last_root_lite)  # Incremental uses last_root_lite
        
        # Test legacy path configuration
        config_legacy = AIConfig(difficulty=5, think_time=50, randomness=0.0)
        config_legacy.use_incremental_search = False
        ai_legacy = MCTSAI(player_number=1, config=config_legacy)
        self.assertFalse(ai_legacy.use_incremental_search)
        
        # Clean up
        ai_incr.clear_tree()
        ai_legacy.clear_tree()

    @pytest.mark.timeout(MCTS_TEST_TIMEOUT)
    def test_select_move_and_policy_returns_tuple(self):
        """Test that select_move_and_policy returns move and policy."""
        ai = MCTSAI(player_number=1, config=self.config)
        ai.neural_net = None
        
        real_move = self._create_test_move()
        
        with patch.object(
            ai.rules_engine, 'get_valid_moves', return_value=[real_move]
        ):
            move, policy = ai.select_move_and_policy(self.game_state)
            
            self.assertIsNotNone(move)
            self.assertIsNotNone(policy)
            self.assertIsInstance(policy, dict)


if __name__ == '__main__':
    unittest.main()

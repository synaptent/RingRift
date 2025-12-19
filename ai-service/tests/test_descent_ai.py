import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime

import numpy as np
import pytest
import torch

from app.ai.descent_ai import DescentAI
from app.models import (
    GameState,
    AIConfig,
    BoardType,
    GamePhase,
    GameStatus,
    TimeControl,
    BoardState,
    Player,
    Position,
    RingStack,
)

# Test timeout guards to prevent hanging in CI
TEST_TIMEOUT_SECONDS = 30


class TestDescentAIHex(unittest.TestCase):
    """Smoke tests for DescentAI on hex boards.

    These tests do not assert on search quality; they simply ensure that the
    hex-specific neural-network path (HexNeuralNet + ActionEncoderHex) can be
    exercised without errors when the board type is HEXAGONAL.
    """

    def setUp(self) -> None:
        self.config = AIConfig(difficulty=5, think_time=1000, randomness=0.1)
        self.ai = DescentAI(player_number=1, config=self.config)

        board = BoardState(
            type=BoardType.HEXAGONAL,
            size=13,  # canonical radius-12 hex (2N+1 = 25 frame)
            stacks={},
            markers={},
            collapsedSpaces={},
            eliminatedRings={},
        )

        players = [
            Player(
                id="p1",
                username="p1",
                type="human",
                playerNumber=1,
                isReady=True,
                timeRemaining=60,
                aiDifficulty=None,
                ringsInHand=18,
                eliminatedRings=0,
                territorySpaces=0,
            ),
            Player(
                id="p2",
                username="p2",
                type="human",
                playerNumber=2,
                isReady=True,
                timeRemaining=60,
                aiDifficulty=None,
                ringsInHand=18,
                eliminatedRings=0,
                territorySpaces=0,
            ),
        ]

        now = datetime.now()

        self.game_state = GameState(
            id="hex-descent-test",
            boardType=BoardType.HEXAGONAL,
            board=board,
            players=players,
            currentPhase=GamePhase.MOVEMENT,
            currentPlayer=1,
            moveHistory=[],
            timeControl=TimeControl(
                initialTime=60,
                increment=0,
                type="blitz",
            ),
            gameStatus=GameStatus.ACTIVE,
            createdAt=now,
            lastMoveAt=now,
            isRated=False,
            maxPlayers=2,
            totalRingsInPlay=36,
            totalRingsEliminated=0,
            victoryThreshold=18,  # RR-CANON-R061: ringsPerPlayer
            territoryVictoryThreshold=33,
            chainCaptureState=None,
            mustMoveFromStackKey=None,
            zobristHash=None,
        )

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    @pytest.mark.skip(
        reason="TODO-HEX-NN-PATH: Hex neural network integration changed during "
        "v3 architecture migration. The HexNeuralNet_v2 model path and HexStateEncoder "
        "initialization differ from this test's expectations. Needs update to use "
        "NeuralNetAI with board_type=BoardType.HEXAGONAL instead of legacy hex paths. "
        "Workaround: manually test hex NN via test_hex_parity.py"
    )
    def test_select_move_uses_hex_network_for_hex_board(self) -> None:
        """Selecting a move on a hex board should hit the hex NN path.

        The test stubs out the rules engine, neural-net feature encoder,
        hex encoder, and hex model so that DescentAI's hex-specific policy
        evaluation runs in a controlled way and returns a legal move.
        """
        # Keep the search very small for test speed.
        self.ai.config.think_time = 10  # milliseconds
        
        # Use legacy search path for this test since it specifically tests
        # the hex NN integration via the legacy apply_move pathway.
        self.ai.use_incremental_search = False

        # Single mocked move that is always legal.
        mock_move = MagicMock()
        mock_move.type = "move_stack"
        mock_move.to = MagicMock()
        mock_move.to.x = 0
        mock_move.to.y = 0

        # Stub the rules engine used by BaseAI / DescentAI.
        rules_engine = MagicMock()
        rules_engine.get_valid_moves.return_value = [mock_move]
        rules_engine.apply_move.side_effect = lambda state, move: state
        self.ai.rules_engine = rules_engine

        # Mock out the shared NeuralNetAI encoder.
        self.ai.neural_net = MagicMock()
        self.ai.neural_net._extract_features.return_value = (
            np.zeros((10, 21, 21), dtype=np.float32),
            np.zeros((10,), dtype=np.float32),
        )
        # Ensure evaluate_position returns a numeric value so that the
        # DescentAI.evaluate_position clamping logic operates on floats
        # rather than MagicMock instances.
        self.ai.neural_net.evaluate_position.return_value = 0.0

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


class TestDescentAIIncrementalSearch(unittest.TestCase):
    """Tests for the make/unmake incremental search integration."""

    def _create_game_state(self) -> GameState:
        """Create a minimal game state for testing."""
        return GameState(
            id="test-game",
            boardType=BoardType.SQUARE8,
            board=BoardState(
                type=BoardType.SQUARE8,
                size=8,
                stacks={
                    "2,2": RingStack(
                        position=Position(x=2, y=2),
                        rings=[1],
                        stackHeight=1,
                        capHeight=1,
                        controllingPlayer=1,
                    ),
                    "5,5": RingStack(
                        position=Position(x=5, y=5),
                        rings=[2],
                        stackHeight=1,
                        capHeight=1,
                        controllingPlayer=2,
                    ),
                },
                markers={},
                collapsedSpaces={},
                eliminatedRings={},
            ),
            players=[
                Player(
                    id="player1",
                    username="Player1",
                    type="human",
                    playerNumber=1,
                    isReady=True,
                    timeRemaining=600000,
                    ringsInHand=17,
                    eliminatedRings=0,
                    territorySpaces=0,
                ),
                Player(
                    id="player2",
                    username="Player2",
                    type="ai",
                    playerNumber=2,
                    isReady=True,
                    timeRemaining=600000,
                    ringsInHand=17,
                    eliminatedRings=0,
                    territorySpaces=0,
                ),
            ],
            currentPhase=GamePhase.MOVEMENT,
            currentPlayer=1,
            moveHistory=[],
            timeControl=TimeControl(
                initialTime=600000,
                increment=5000,
                type="fischer",
            ),
            gameStatus=GameStatus.ACTIVE,
            createdAt="2023-01-01T00:00:00Z",
            lastMoveAt="2023-01-01T00:00:00Z",
            isRated=False,
            maxPlayers=2,
            totalRingsInPlay=36,
            totalRingsEliminated=0,
            victoryThreshold=6,
            territoryVictoryThreshold=20,
        )

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_use_incremental_search_defaults_to_true(self) -> None:
        """Verify use_incremental_search defaults to True."""
        config = AIConfig(difficulty=5)
        ai = DescentAI(player_number=1, config=config)
        self.assertTrue(ai.use_incremental_search)

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_use_incremental_search_can_be_disabled(self) -> None:
        """Verify use_incremental_search can be set to False."""
        config = AIConfig(difficulty=5)
        ai = DescentAI(player_number=1, config=config)
        ai.use_incremental_search = False
        self.assertFalse(ai.use_incremental_search)

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_select_move_uses_incremental_when_enabled(self) -> None:
        """Verify select_move routes to incremental search when enabled."""
        config = AIConfig(difficulty=3, think_time=50)
        ai = DescentAI(player_number=1, config=config)
        ai.use_incremental_search = True

        mock_move = MagicMock()
        mock_move.type = "move_stack"

        with patch.object(
            ai,
            "get_valid_moves",
            return_value=[mock_move],
        ), patch.object(
            ai,
            "_select_move_incremental",
            return_value=mock_move,
        ) as mock_incremental, patch.object(
            ai,
            "_select_move_legacy",
            return_value=mock_move,
        ) as mock_legacy:
            ai.should_pick_random_move = MagicMock(return_value=False)
            ai.select_move(self._create_game_state())

        mock_incremental.assert_called_once()
        mock_legacy.assert_not_called()

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_select_move_uses_legacy_when_disabled(self) -> None:
        """Verify select_move routes to legacy search when disabled."""
        config = AIConfig(difficulty=3, think_time=50)
        ai = DescentAI(player_number=1, config=config)
        ai.use_incremental_search = False

        mock_move = MagicMock()
        mock_move.type = "move_stack"

        with patch.object(
            ai,
            "get_valid_moves",
            return_value=[mock_move],
        ), patch.object(
            ai,
            "_select_move_incremental",
            return_value=mock_move,
        ) as mock_incremental, patch.object(
            ai,
            "_select_move_legacy",
            return_value=mock_move,
        ) as mock_legacy:
            ai.should_pick_random_move = MagicMock(return_value=False)
            ai.select_move(self._create_game_state())

        mock_legacy.assert_called_once()
        mock_incremental.assert_not_called()

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_incremental_search_produces_valid_move(self) -> None:
        """Test that incremental search produces a valid move.
        
        Mocks the internal descent iteration to return immediately,
        allowing the search to complete and return a move from the
        transposition table.
        """
        config = AIConfig(difficulty=3, think_time=50)
        ai = DescentAI(player_number=1, config=config)
        ai.use_incremental_search = True
        ai.neural_net = None

        mock_move = MagicMock()
        mock_move.type = "move_stack"
        mock_move.__str__ = MagicMock(return_value="mock_move")

        game_state = self._create_game_state()

        # Pre-populate transposition table with a valid entry
        # This simulates what would happen after an iteration
        state_hash = 12345
        children_values = {
            "mock_move": (mock_move, 0.5, 0.8),
        }
        from app.ai.descent_ai import NodeStatus
        ai.transposition_table.put(
            state_hash,
            (0.5, children_values, NodeStatus.HEURISTIC),
        )

        # Mock MutableGameState.from_immutable to return a mock with the hash
        mock_mutable = MagicMock()
        mock_mutable.zobrist_hash = state_hash
        mock_mutable.current_player = 1

        with patch.object(
            ai,
            "get_valid_moves",
            return_value=[mock_move],
        ), patch(
            "app.ai.descent_ai.MutableGameState.from_immutable",
            return_value=mock_mutable,
        ), patch.object(
            ai,
            "_descent_iteration_mutable",
            return_value=0.5,
        ):
            result = ai.select_move(game_state)

        self.assertIsNotNone(result)
        self.assertEqual(result, mock_move)

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_legacy_search_produces_valid_move(self) -> None:
        """Test that legacy search produces a valid move.
        
        Mocks the internal descent iteration to return immediately,
        allowing the search to complete and return a move from the
        transposition table.
        """
        config = AIConfig(difficulty=3, think_time=50)
        ai = DescentAI(player_number=1, config=config)
        ai.use_incremental_search = False
        ai.neural_net = None

        mock_move = MagicMock()
        mock_move.type = "move_stack"
        mock_move.__str__ = MagicMock(return_value="mock_move")

        game_state = self._create_game_state()

        # Pre-populate transposition table with a valid entry
        # This simulates what would happen after an iteration
        state_hash = 67890
        children_values = {
            "mock_move": (mock_move, 0.5, 0.8),
        }
        from app.ai.descent_ai import NodeStatus
        ai.transposition_table.put(
            state_hash,
            (0.5, children_values, NodeStatus.HEURISTIC),
        )

        with patch.object(
            ai,
            "get_valid_moves",
            return_value=[mock_move],
        ), patch.object(
            ai,
            "_get_state_key",
            return_value=state_hash,
        ), patch.object(
            ai,
            "_descent_iteration",
            return_value=0.5,
        ):
            result = ai.select_move(game_state)

        self.assertIsNotNone(result)
        self.assertEqual(result, mock_move)

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_incremental_search_with_no_valid_moves(self) -> None:
        """Test that incremental search handles no valid moves."""
        config = AIConfig(difficulty=3, think_time=50)
        ai = DescentAI(player_number=1, config=config)
        ai.use_incremental_search = True
        ai.neural_net = None

        with patch.object(
            ai,
            "get_valid_moves",
            return_value=[],
        ):
            result = ai.select_move(self._create_game_state())

        self.assertIsNone(result)

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_random_move_selection_bypasses_search(self) -> None:
        """Test that random move selection doesn't run full search."""
        config = AIConfig(difficulty=1, think_time=50, randomness=1.0)
        ai = DescentAI(player_number=1, config=config)
        ai.use_incremental_search = True
        ai.neural_net = None

        mock_move = MagicMock()
        mock_move.type = "move_stack"

        with patch.object(
            ai,
            "get_valid_moves",
            return_value=[mock_move],
        ), patch.object(
            ai.rules_engine,
            "apply_move",
        ) as mock_apply:
            # Force random selection
            ai.should_pick_random_move = MagicMock(return_value=True)
            result = ai.select_move(self._create_game_state())

        self.assertIsNotNone(result)
        # Verify no search-related apply_move calls occurred
        mock_apply.assert_not_called()

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_incremental_search_uses_mutable_state(self) -> None:
        """Verify incremental search creates MutableGameState.
        
        This test verifies that when incremental search is enabled,
        MutableGameState.from_immutable is called to convert the
        game state for make/unmake pattern.
        """
        config = AIConfig(difficulty=2, think_time=50)
        ai = DescentAI(player_number=1, config=config)
        ai.use_incremental_search = True
        ai.neural_net = None

        game_state = self._create_game_state()

        mock_move = MagicMock()
        mock_move.type = "move_stack"
        mock_move.__str__ = MagicMock(return_value="mock_move")

        # Track if from_immutable is called
        from_immutable_called = [False]

        def mock_from_immutable(state):
            from_immutable_called[0] = True
            mock_mutable = MagicMock()
            mock_mutable.zobrist_hash = 11111
            mock_mutable.current_player = 1
            return mock_mutable

        # Pre-populate transposition table
        from app.ai.descent_ai import NodeStatus
        ai.transposition_table.put(
            11111,
            (0.5, {"mock_move": (mock_move, 0.5, 0.8)}, NodeStatus.HEURISTIC),
        )

        with patch.object(
            ai, "get_valid_moves", return_value=[mock_move]
        ), patch(
            "app.ai.descent_ai.MutableGameState.from_immutable",
            side_effect=mock_from_immutable,
        ), patch.object(
            ai,
            "_descent_iteration_mutable",
            return_value=0.5,
        ):
            ai.should_pick_random_move = MagicMock(return_value=False)
            ai.select_move(game_state)

        # MutableGameState.from_immutable should be called
        self.assertTrue(from_immutable_called[0])


if __name__ == "__main__":
    unittest.main()

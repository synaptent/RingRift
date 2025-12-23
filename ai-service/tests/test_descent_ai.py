import unittest
from unittest.mock import MagicMock, patch

import pytest

from app.ai.descent_ai import DescentAI
from app.models import (
    AIConfig,
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Player,
    Position,
    RingStack,
    TimeControl,
)

# Test timeout guards to prevent hanging in CI
TEST_TIMEOUT_SECONDS = 30


# NOTE: TestDescentAIHex class was removed (2025-12-23) because it tested
# the legacy hex_model attribute that was removed during v3 architecture
# migration. The hex NN path is now tested via NeuralNetAI's internal
# board-type dispatch in test_hex_training.py, test_hex_augmentation.py,
# and other hex-specific test files.


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

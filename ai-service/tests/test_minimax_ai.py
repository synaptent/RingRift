import unittest
from unittest.mock import MagicMock, patch

from app.ai.minimax_ai import MinimaxAI
from app.models import (
    AIConfig,
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    TimeControl,
)


class TestMinimaxAI(unittest.TestCase):
    """Unit tests for MinimaxAI that do NOT leak global GameEngine mocks.

    Earlier versions of this test assigned MagicMocks directly to
    ``app.game_engine.GameEngine.get_valid_moves`` and ``apply_move``, which
    persisted after the test finished and broke later tests such as
    ``tests/test_rules_evaluate_move.py``.

    Here we patch methods on the MinimaxAI instance (and its rules_engine)
    so that changes are strictly local to these tests.
    """

    def setUp(self) -> None:
        self.config = AIConfig(difficulty=5)
        self.ai = MinimaxAI(player_number=1, config=self.config)

        # Minimal but valid game state for MinimaxAI to consume.
        self.game_state = GameState(
            id="test-game",
            boardType=BoardType.SQUARE8,
            board=BoardState(
                type=BoardType.SQUARE8,
                size=8,
                stacks={},
                markers={},
                collapsedSpaces={},
                eliminatedRings={},
            ),
            players=[],
            currentPhase=GamePhase.MOVEMENT,
            currentPlayer=1,
            moveHistory=[],
            timeControl=TimeControl(
                initialTime=600,
                increment=0,
                type="blitz",
            ),
            gameStatus=GameStatus.ACTIVE,
            createdAt="2023-01-01T00:00:00Z",
            lastMoveAt="2023-01-01T00:00:00Z",
            isRated=False,
            maxPlayers=2,
            totalRingsInPlay=36,
            totalRingsEliminated=0,
            victoryThreshold=19,
            territoryVictoryThreshold=33,
        )

    def test_select_move_returns_move(self) -> None:
        """MinimaxAI.select_move returns a move when legal moves exist.

        We patch ``self.ai.get_valid_moves`` and ``self.ai.rules_engine.apply_move``
        so that no global GameEngine state is modified.
        """

        mock_move1 = MagicMock()
        mock_move1.type = "move_stack"
        mock_move2 = MagicMock()
        mock_move2.type = "move_stack"

        with patch.object(
            self.ai,
            "get_valid_moves",
            return_value=[mock_move1, mock_move2],
        ), patch.object(
            self.ai.rules_engine,
            "apply_move",
            return_value=self.game_state,
        ):
            # Simplify evaluation logic for the purposes of this unit test.
            self.ai.evaluate_position = MagicMock(return_value=10)
            self.ai.simulate_thinking = MagicMock()
            self.ai.should_pick_random_move = MagicMock(return_value=False)

            move = self.ai.select_move(self.game_state)

        self.assertIsNotNone(move)
        self.assertEqual(move, mock_move1)

    def test_select_move_no_valid_moves(self) -> None:
        """MinimaxAI.select_move returns None when there are no legal moves."""

        with patch.object(
            self.ai,
            "get_valid_moves",
            return_value=[],
        ):
            self.ai.simulate_thinking = MagicMock()
            move = self.ai.select_move(self.game_state)

        self.assertIsNone(move)


if __name__ == "__main__":
    unittest.main()

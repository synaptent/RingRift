import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime
from app.ai.mcts_ai import MCTSAI
from app.models import (
    GameState, AIConfig, BoardType, GamePhase, GameStatus, TimeControl,
    BoardState
)


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

    @patch('app.game_engine.GameEngine.get_valid_moves')
    def test_select_move_no_valid_moves(self, mock_get_valid_moves):
        mock_get_valid_moves.return_value = []
        self.ai.simulate_thinking = MagicMock()
        
        move = self.ai.select_move(self.game_state)
        self.assertIsNone(move)


if __name__ == '__main__':
    unittest.main()
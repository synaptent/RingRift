import unittest
from unittest.mock import MagicMock
from app.ai.mcts_ai import MCTSAI
from app.models import GameState, AIConfig, BoardType, GamePhase, GameStatus, TimeControl, BoardState

class TestMCTSAI(unittest.TestCase):
    def setUp(self):
        self.config = AIConfig(difficulty=5)
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
            timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
            gameStatus=GameStatus.ACTIVE,
            createdAt="2023-01-01T00:00:00Z",
            lastMoveAt="2023-01-01T00:00:00Z",
            isRated=False,
            maxPlayers=2,
            totalRingsInPlay=36,
            totalRingsEliminated=0,
            victoryThreshold=19,
            territoryVictoryThreshold=33
        )

    def test_select_move_returns_move(self):
        self.ai._get_valid_moves_for_phase = MagicMock(return_value=[
            {"type": "move", "from": None, "to": {"x": 0, "y": 0}},
            {"type": "move", "from": None, "to": {"x": 1, "y": 1}}
        ])
        self.ai._evaluate_move = MagicMock(return_value=0) # Mock heuristic fallback
        self.ai._create_move_object = MagicMock(return_value="mock_move_object")
        self.ai.simulate_thinking = MagicMock()

        move = self.ai.select_move(self.game_state)
        
        self.assertIsNotNone(move)
        self.assertEqual(move, "mock_move_object")

    def test_select_move_no_valid_moves(self):
        self.ai._get_valid_moves_for_phase = MagicMock(return_value=[])
        self.ai.simulate_thinking = MagicMock()
        
        move = self.ai.select_move(self.game_state)
        self.assertIsNone(move)

if __name__ == '__main__':
    unittest.main()
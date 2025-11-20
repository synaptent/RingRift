import unittest
from unittest.mock import MagicMock
from app.ai.minimax_ai import MinimaxAI
from app.models import GameState, AIConfig, BoardType, GamePhase, GameStatus, TimeControl, BoardState

class TestMinimaxAI(unittest.TestCase):
    def setUp(self):
        self.config = AIConfig(difficulty=5)
        self.ai = MinimaxAI(player_number=1, config=self.config)
        
        # Create a mock game state
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
        # Mock GameEngine.get_valid_moves
        from app.game_engine import GameEngine
        
        mock_move1 = MagicMock()
        mock_move1.type = "move_stack"
        mock_move2 = MagicMock()
        mock_move2.type = "move_stack"
        
        GameEngine.get_valid_moves = MagicMock(return_value=[mock_move1, mock_move2])
        
        # Mock GameEngine.apply_move to return a valid state
        GameEngine.apply_move = MagicMock(return_value=self.game_state)
        
        # Mock evaluate_position
        # We need to provide enough side effects for the recursive calls
        # The minimax algorithm will call evaluate_position multiple times
        # 1. Initial sort (2 calls)
        # 2. Minimax recursion (depth 2) -> Quiescence search -> evaluate_position
        # Let's just return a constant value or a simple function to avoid StopIteration
        self.ai.evaluate_position = MagicMock(return_value=10)
        
        # Mock simulate_thinking to avoid delay
        self.ai.simulate_thinking = MagicMock()
        
        # Mock should_pick_random_move to False
        self.ai.should_pick_random_move = MagicMock(return_value=False)

        move = self.ai.select_move(self.game_state)
        
        self.assertIsNotNone(move)
        self.assertEqual(move, mock_move1)

    def test_select_move_no_valid_moves(self):
        from app.game_engine import GameEngine
        GameEngine.get_valid_moves = MagicMock(return_value=[])
        self.ai.simulate_thinking = MagicMock()
        
        move = self.ai.select_move(self.game_state)
        self.assertIsNone(move)

if __name__ == '__main__':
    unittest.main()
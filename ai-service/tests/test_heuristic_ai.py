import unittest
from unittest.mock import MagicMock
from datetime import datetime
from app.ai.heuristic_ai import HeuristicAI
from app.models import GameState, AIConfig, BoardType, GamePhase, GameStatus, TimeControl, BoardState, Position, RingStack, MarkerInfo, Player

class TestHeuristicAI(unittest.TestCase):
    def setUp(self):
        self.config = AIConfig(difficulty=5)
        self.ai = HeuristicAI(player_number=1, config=self.config)
        
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
            players=[
                Player(id="p1", username="P1", type="human", playerNumber=1, isReady=True, timeRemaining=600, ringsInHand=10, eliminatedRings=0, territorySpaces=0, aiDifficulty=None),
                Player(id="p2", username="P2", type="human", playerNumber=2, isReady=True, timeRemaining=600, ringsInHand=10, eliminatedRings=0, territorySpaces=0, aiDifficulty=None)
            ],
            currentPhase=GamePhase.MOVEMENT,
            currentPlayer=1,
            moveHistory=[],
            timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
            gameStatus=GameStatus.ACTIVE,
            createdAt=datetime.now(),
            lastMoveAt=datetime.now(),
            isRated=False,
            maxPlayers=2,
            totalRingsInPlay=36,
            totalRingsEliminated=0,
            victoryThreshold=19,
            territoryVictoryThreshold=33
        )

    def test_evaluate_line_connectivity(self):
        # Setup: Two markers with a gap of 1
        self.game_state.board.markers["3,3"] = MarkerInfo(player=1, position=Position(x=3, y=3), type="regular")
        self.game_state.board.markers["5,3"] = MarkerInfo(player=1, position=Position(x=5, y=3), type="regular")
        
        score = self.ai._evaluate_line_connectivity(self.game_state)
        
        # Should have some positive score
        self.assertGreater(score, 0)

    def test_evaluate_territory_safety(self):
        # Setup: My marker near opponent stack
        self.game_state.board.markers["3,3"] = MarkerInfo(player=1, position=Position(x=3, y=3), type="regular")
        self.game_state.board.stacks["4,3"] = RingStack(position=Position(x=4, y=3), rings=[2], stackHeight=1, capHeight=1, controllingPlayer=2)
        
        score = self.ai._evaluate_territory_safety(self.game_state)
        
        # Should have negative score (penalty)
        self.assertLess(score, 0)

    def test_evaluate_stack_mobility(self):
        # Setup: My stack surrounded by collapsed spaces
        self.game_state.board.stacks["3,3"] = RingStack(position=Position(x=3, y=3), rings=[1], stackHeight=1, capHeight=1, controllingPlayer=1)
        
        # Block all 8 neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                self.game_state.board.collapsed_spaces[f"{3+dx},{3+dy}"] = 1
                
        score = self.ai._evaluate_stack_mobility(self.game_state)
        
        # Should have negative score (penalty for being blocked)
        self.assertLess(score, 0)

if __name__ == '__main__':
    unittest.main()
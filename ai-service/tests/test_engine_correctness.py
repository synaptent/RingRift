import copy
import unittest
from datetime import datetime

from app.game_engine import GameEngine
from app.models import BoardState, BoardType, GamePhase, GameState, GameStatus, Player, Position, RingStack, TimeControl


class TestEngineCorrectness(unittest.TestCase):
    def setUp(self):
        self.state = GameState(
            id="test-game",
            boardType=BoardType.SQUARE8,
            board=BoardState(
                type=BoardType.SQUARE8,
                size=8,
                stacks={
                    "3,3": RingStack(
                        position=Position(x=3, y=3),
                        rings=[1],
                        stackHeight=1,
                        capHeight=1,
                        controllingPlayer=1
                    )
                },
                markers={},
                collapsedSpaces={},
                eliminatedRings={}
            ),
            players=[
                Player(
                    id="p1", username="P1", type="human", playerNumber=1,
                    isReady=True, timeRemaining=600, ringsInHand=10,
                    eliminatedRings=0, territorySpaces=0, aiDifficulty=None
                ),
                Player(
                    id="p2", username="P2", type="human", playerNumber=2,
                    isReady=True, timeRemaining=600, ringsInHand=10,
                    eliminatedRings=0, territorySpaces=0, aiDifficulty=None
                )
            ],
            currentPhase=GamePhase.RING_PLACEMENT,
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
            victoryThreshold=18,  # RR-CANON-R061: ringsPerPlayer
            territoryVictoryThreshold=33
        )

    def test_apply_move_does_not_mutate_original(self):
        original_state = copy.deepcopy(self.state)

        # Get a move (placement)
        moves = GameEngine.get_valid_moves(self.state, 1)
        self.assertTrue(len(moves) > 0)
        move = moves[0]

        # Apply move
        new_state = GameEngine.apply_move(self.state, move)

        # Check if original state is modified
        # Specifically check the stack that might have been modified
        original_state.board.stacks.get("3,3")
        self.state.board.stacks.get("3,3")

        # If we placed on 3,3, it would change. If we placed elsewhere, a new stack would appear.
        # Let's check if any stack in original state changed.

        self.assertEqual(len(self.state.board.stacks), len(original_state.board.stacks))
        for k, v in self.state.board.stacks.items():
            self.assertEqual(v.rings, original_state.board.stacks[k].rings)
            self.assertEqual(v.stack_height, original_state.board.stacks[k].stack_height)

        # Also check if new_state is different
        self.assertNotEqual(len(new_state.board.stacks), len(self.state.board.stacks))

if __name__ == '__main__':
    unittest.main()

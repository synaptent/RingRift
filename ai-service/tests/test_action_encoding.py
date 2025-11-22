import unittest
import sys
import os
from datetime import datetime

# Ensure app package is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from app.models import (
    Position, Move, MoveType, AIConfig
)
from app.ai.neural_net import NeuralNetAI, INVALID_MOVE_INDEX


class TestActionEncoding(unittest.TestCase):
    def setUp(self):
        self.config = AIConfig(difficulty=10, randomness=0.1, think_time=500)
        self.ai = NeuralNetAI(player_number=1, config=self.config)
        self.board_size = 8

    def test_encode_placement(self):
        """Test encoding of placement moves"""
        move = Move(
            id="test",
            type=MoveType.PLACE_RING,
            player=1,
            to=Position(x=3, y=3),
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=1,
            placementCount=1,
            placedOnStack=False
        )

        idx = self.ai.encode_move(move, self.board_size)
        self.assertNotEqual(idx, INVALID_MOVE_INDEX)
        
        # Manual calculation check
        # Index = (y * 19 + x) * 3 + (count - 1)
        # (3 * 19 + 3) * 3 + 0 = 60 * 3 = 180
        expected = (3 * 19 + 3) * 3 + 0
        self.assertEqual(idx, expected)

    def test_encode_movement(self):
        """Test encoding of movement moves"""
        move = Move(
            id="test",
            type=MoveType.MOVE_STACK,
            player=1,
            from_pos=Position(x=3, y=3),
            to=Position(x=3, y=4),
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=1
        )

        idx = self.ai.encode_move(move, self.board_size)
        self.assertNotEqual(idx, INVALID_MOVE_INDEX)

    def test_encode_capture(self):
        """Test encoding of capture moves"""
        move = Move(
            id="test",
            type=MoveType.OVERTAKING_CAPTURE,
            player=1,
            from_pos=Position(x=3, y=3),
            to=Position(x=3, y=5),
            capture_target=Position(x=3, y=4),
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=1
        )

        idx = self.ai.encode_move(move, self.board_size)
        self.assertNotEqual(idx, INVALID_MOVE_INDEX)

    def test_encode_continue_capture(self):
        """Test encoding of continue capture moves"""
        move = Move(
            id="test",
            type=MoveType.CONTINUE_CAPTURE_SEGMENT,
            player=1,
            from_pos=Position(x=3, y=5),
            to=Position(x=5, y=5),
            capture_target=Position(x=4, y=5),
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=1
        )

        idx = self.ai.encode_move(move, self.board_size)
        self.assertNotEqual(idx, INVALID_MOVE_INDEX)


if __name__ == '__main__':
    unittest.main()
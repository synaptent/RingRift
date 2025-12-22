import os
import sys
import unittest
from datetime import datetime

# Ensure app package is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from app.ai.neural_net import (
    INVALID_MOVE_INDEX,
    P_HEX,
    SQUARE8_FORCED_ELIMINATION_IDX,
    SQUARE8_LINE_FORM_BASE,
    SQUARE8_NO_LINE_ACTION_IDX,
    SQUARE8_NO_MOVEMENT_ACTION_IDX,
    SQUARE8_NO_PLACEMENT_ACTION_IDX,
    SQUARE8_NO_TERRITORY_ACTION_IDX,
    SQUARE8_SKIP_CAPTURE_IDX,
    SQUARE8_SKIP_TERRITORY_PROCESSING_IDX,
    SQUARE19_FORCED_ELIMINATION_IDX,
    SQUARE19_LINE_FORM_BASE,
    SQUARE19_NO_LINE_ACTION_IDX,
    SQUARE19_NO_MOVEMENT_ACTION_IDX,
    SQUARE19_NO_PLACEMENT_ACTION_IDX,
    SQUARE19_NO_TERRITORY_ACTION_IDX,
    SQUARE19_SKIP_CAPTURE_IDX,
    SQUARE19_SKIP_TERRITORY_PROCESSING_IDX,
    ActionEncoderHex,
    NeuralNetAI,
    decode_move_for_board,
    encode_move_for_board,
)
from app.models import (
    AIConfig,
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    LineInfo,
    Move,
    MoveType,
    Player,
    Position,
    TimeControl,
)
from app.rules.core import (
    get_rings_per_player,
    get_territory_victory_threshold,
    get_victory_threshold,
)


def make_dummy_hex_game_state() -> GameState:
    """Create a minimal, canonical hex GameState for encoder tests.

    This mirrors the helper patterns in tests/rules/test_utils.py but uses a
    hexagonal board (BoardType.HEXAGONAL) with size=25 (bounding box = 2*radius+1),
    which matches the canonical N=12 hex used by the neural-net encoder.
    """
    board = BoardState(type=BoardType.HEXAGONAL, size=25)
    rings_per_player = get_rings_per_player(BoardType.HEXAGONAL)

    players = [
        Player(
            id="p1",
            username="p1",
            type="human",
            playerNumber=1,
            isReady=True,
            timeRemaining=60,
            aiDifficulty=None,
            ringsInHand=rings_per_player,
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
            ringsInHand=rings_per_player,
            eliminatedRings=0,
            territorySpaces=0,
        ),
    ]

    now = datetime.now()

    return GameState(
        id="hex-test",
        boardType=BoardType.HEXAGONAL,
        board=board,
        players=players,
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        timeControl=TimeControl(initialTime=60, increment=0, type="blitz"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=get_victory_threshold(BoardType.HEXAGONAL, 2),
        territoryVictoryThreshold=get_territory_victory_threshold(BoardType.HEXAGONAL),
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
    )


class TestActionEncoding(unittest.TestCase):
    def setUp(self):
        self.config = AIConfig(difficulty=10, randomness=0.1, think_time=500)
        self.ai = NeuralNetAI(player_number=1, config=self.config)
        self.board_size = 8

    def test_encode_placement(self):
        """Test encoding of placement moves on square boards."""
        move = Move(
            id="test",
            type=MoveType.PLACE_RING,
            player=1,
            to=Position(x=3, y=3),
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=1,
            placementCount=1,
            placedOnStack=False,
        )

        idx = self.ai.encode_move(move, self.board_size)
        self.assertNotEqual(idx, INVALID_MOVE_INDEX)

        # Manual calculation check
        # Index = (y * 19 + x) * 3 + (count - 1)
        # (3 * 19 + 3) * 3 + 0 = 60 * 3 = 180
        expected = (3 * 19 + 3) * 3 + 0
        self.assertEqual(idx, expected)

    def test_encode_movement(self):
        """Test encoding of movement moves on square boards."""
        move = Move(
            id="test",
            type=MoveType.MOVE_STACK,
            player=1,
            from_pos=Position(x=3, y=3),
            to=Position(x=3, y=4),
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=1,
        )

        idx = self.ai.encode_move(move, self.board_size)
        self.assertNotEqual(idx, INVALID_MOVE_INDEX)

    def test_encode_capture(self):
        """Test encoding of capture moves on square boards."""
        move = Move(
            id="test",
            type=MoveType.OVERTAKING_CAPTURE,
            player=1,
            from_pos=Position(x=3, y=3),
            to=Position(x=3, y=5),
            capture_target=Position(x=3, y=4),
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=1,
        )

        idx = self.ai.encode_move(move, self.board_size)
        self.assertNotEqual(idx, INVALID_MOVE_INDEX)

    def test_encode_continue_capture(self):
        """Test encoding of continue capture moves on square boards."""
        move = Move(
            id="test",
            type=MoveType.CONTINUE_CAPTURE_SEGMENT,
            player=1,
            from_pos=Position(x=3, y=5),
            to=Position(x=5, y=5),
            capture_target=Position(x=4, y=5),
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=1,
        )

        idx = self.ai.encode_move(move, self.board_size)
        self.assertNotEqual(idx, INVALID_MOVE_INDEX)


class TestActionEncodingSquarePolicies(unittest.TestCase):
    def setUp(self):
        self.board8 = BoardState(type=BoardType.SQUARE8, size=8)
        self.board19 = BoardState(type=BoardType.SQUARE19, size=19)
        self.now = datetime.now()

    def test_process_line_uses_anchor_position(self):
        """PROCESS_LINE encodes using the first line anchor position."""
        line = LineInfo(
            positions=[
                Position(x=2, y=1),
                Position(x=3, y=1),
                Position(x=4, y=1),
                Position(x=5, y=1),
            ],
            player=1,
            length=4,
            direction=Position(x=1, y=0),
        )
        move = Move(
            id="line-test",
            type=MoveType.PROCESS_LINE,
            player=1,
            to=None,
            formedLines=(line,),
            timestamp=self.now,
            thinkTime=0,
            moveNumber=1,
        )

        idx8 = encode_move_for_board(move, self.board8)
        pos_idx8 = 1 * 8 + 2
        self.assertEqual(idx8, SQUARE8_LINE_FORM_BASE + pos_idx8 * 4)

        idx19 = encode_move_for_board(move, self.board19)
        pos_idx19 = 1 * 19 + 2
        self.assertEqual(idx19, SQUARE19_LINE_FORM_BASE + pos_idx19 * 4)

    def test_extra_special_indices_round_trip(self):
        """Extra special actions map to fixed indices and decode as specials."""
        cases = [
            (MoveType.NO_PLACEMENT_ACTION, SQUARE8_NO_PLACEMENT_ACTION_IDX, SQUARE19_NO_PLACEMENT_ACTION_IDX, "no_placement_action"),
            (MoveType.NO_MOVEMENT_ACTION, SQUARE8_NO_MOVEMENT_ACTION_IDX, SQUARE19_NO_MOVEMENT_ACTION_IDX, "no_movement_action"),
            (MoveType.SKIP_CAPTURE, SQUARE8_SKIP_CAPTURE_IDX, SQUARE19_SKIP_CAPTURE_IDX, "skip_capture"),
            (MoveType.NO_LINE_ACTION, SQUARE8_NO_LINE_ACTION_IDX, SQUARE19_NO_LINE_ACTION_IDX, "no_line_action"),
            (MoveType.NO_TERRITORY_ACTION, SQUARE8_NO_TERRITORY_ACTION_IDX, SQUARE19_NO_TERRITORY_ACTION_IDX, "no_territory_action"),
            (MoveType.SKIP_TERRITORY_PROCESSING, SQUARE8_SKIP_TERRITORY_PROCESSING_IDX, SQUARE19_SKIP_TERRITORY_PROCESSING_IDX, "skip_territory_processing"),
            (MoveType.FORCED_ELIMINATION, SQUARE8_FORCED_ELIMINATION_IDX, SQUARE19_FORCED_ELIMINATION_IDX, "forced_elimination"),
        ]

        for move_type, idx8_expected, idx19_expected, action_type in cases:
            move = Move(
                id=f"special-{move_type.value}",
                type=move_type,
                player=1,
                timestamp=self.now,
                thinkTime=0,
                moveNumber=1,
            )

            idx8 = encode_move_for_board(move, self.board8)
            self.assertEqual(idx8, idx8_expected)
            decoded8 = decode_move_for_board(idx8, BoardType.SQUARE8)
            self.assertIsNotNone(decoded8)
            assert decoded8 is not None
            self.assertTrue(decoded8.is_special)
            self.assertEqual(decoded8.action_type, action_type)

            idx19 = encode_move_for_board(move, self.board19)
            self.assertEqual(idx19, idx19_expected)
            decoded19 = decode_move_for_board(idx19, BoardType.SQUARE19)
            self.assertIsNotNone(decoded19)
            assert decoded19 is not None
            self.assertTrue(decoded19.is_special)
            self.assertEqual(decoded19.action_type, action_type)


class TestActionEncodingHex(unittest.TestCase):
    """Hex-specific tests for ActionEncoderHex.

    These tests validate that the documented hex layout (placements,
    movement/capture, and special indices) is implemented correctly and
    round-trips through encode_move / decode_move for canonical cases.
    """

    def setUp(self):
        self.encoder = ActionEncoderHex()
        self.game_state = make_dummy_hex_game_state()
        self.board = self.game_state.board

    def test_hex_placement_round_trip(self):
        """Central placement (0,0,0) should round-trip via encoder."""
        move = Move(
            id="hex-place-center",
            type=MoveType.PLACE_RING,
            player=self.game_state.current_player,
            to=Position(x=0, y=0, z=0),
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=1,
            placementCount=1,
            placedOnStack=False,
        )

        idx = self.encoder.encode_move(move, self.board)
        self.assertNotEqual(idx, INVALID_MOVE_INDEX)
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, P_HEX)

        decoded = self.encoder.decode_move(idx, self.game_state)
        self.assertIsNotNone(decoded)
        assert decoded is not None  # for type-checkers
        self.assertEqual(decoded.type, MoveType.PLACE_RING)
        self.assertEqual(decoded.to.x, 0)
        self.assertEqual(decoded.to.y, 0)
        self.assertEqual(decoded.to.z, 0)
        self.assertEqual(decoded.placement_count, 1)

    def test_hex_movement_round_trip(self):
        """Simple 1-step movement along a hex direction should round-trip."""
        # From centre (0,0,0) to neighbour in direction (1,-1,0).
        move = Move(
            id="hex-move-step",
            type=MoveType.MOVE_STACK,
            player=self.game_state.current_player,
            from_pos=Position(x=0, y=0, z=0),
            to=Position(x=1, y=-1, z=0),
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=1,
        )

        idx = self.encoder.encode_move(move, self.board)
        self.assertNotEqual(idx, INVALID_MOVE_INDEX)

        decoded = self.encoder.decode_move(idx, self.game_state)
        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertEqual(decoded.type, MoveType.MOVE_STACK)
        self.assertEqual(decoded.from_pos.x, 0)
        self.assertEqual(decoded.from_pos.y, 0)
        self.assertEqual(decoded.from_pos.z, 0)
        self.assertEqual(decoded.to.x, 1)
        self.assertEqual(decoded.to.y, -1)
        self.assertEqual(decoded.to.z, 0)

    def test_hex_max_distance_round_trip(self):
        """Max-radius move from centre to edge should still decode validly."""
        # Radius N=12: from centre (0,0,0) to edge (12,-12,0) lies on-board.
        move = Move(
            id="hex-move-maxdist",
            type=MoveType.MOVE_STACK,
            player=self.game_state.current_player,
            from_pos=Position(x=0, y=0, z=0),
            to=Position(x=12, y=-12, z=0),
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=1,
        )

        idx = self.encoder.encode_move(move, self.board)
        self.assertNotEqual(idx, INVALID_MOVE_INDEX)

        decoded = self.encoder.decode_move(idx, self.game_state)
        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertEqual(decoded.type, MoveType.MOVE_STACK)
        self.assertEqual(decoded.from_pos.x, 0)
        self.assertEqual(decoded.from_pos.y, 0)
        self.assertEqual(decoded.from_pos.z, 0)
        self.assertEqual(decoded.to.x, 12)
        self.assertEqual(decoded.to.y, -12)
        self.assertEqual(decoded.to.z, 0)

    def test_hex_offboard_decode_returns_none(self):
        """Indices that map outside the true hex should decode to None.

        We pick an obviously off-board canonical coordinate (e.g. a corner
        of the 25×25 bounding box that lies outside the 469-cell hex) and
        construct a fake index targeting it, then assert decode_move
        returns None.
        """
        # Construct an index corresponding to a placement at (24, 13), which
        # lies outside the axial radius-12 hex when mapped back via
        # _from_canonical_xy / BoardGeometry.is_within_bounds.
        cx, cy = 24, 13
        pos_idx = cy * 25 + cx
        fake_index = pos_idx * 3  # count_idx = 0 ⇒ placementCount=1

        # Sanity check that this index is within the placement span.
        self.assertLess(fake_index, HEX_PLACEMENT_SPAN := 25 * 25 * 3)

        decoded = self.encoder.decode_move(fake_index, self.game_state)
        self.assertIsNone(decoded)


if __name__ == "__main__":
    unittest.main()

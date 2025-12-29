"""Unit tests for app.notation.algebraic module.

Tests algebraic notation conversion for RingRift game records including:
- Position conversion for square and hex boards
- Move notation conversion
- PGN-style game record generation and parsing
"""

import pytest
from datetime import datetime
from unittest.mock import patch
from uuid import uuid4

from app.models import BoardType, GameState, Move, MoveType, Position
from app.notation.algebraic import (
    # MoveType mappings
    MOVE_TYPE_TO_CODE,
    CODE_TO_MOVE_TYPE,
    # Position functions
    position_to_algebraic,
    algebraic_to_position,
    _position_to_algebraic_square,
    _position_to_algebraic_hex,
    _algebraic_to_position_square,
    _algebraic_to_position_hex,
    # Move functions
    move_to_algebraic,
    algebraic_to_move,
    moves_to_notation_list,
    # PGN functions
    game_to_pgn,
    parse_pgn,
)


class TestMoveTypeMappings:
    """Tests for MoveType <-> Code mappings."""

    def test_move_type_to_code_core_types(self):
        """Test core move types have codes."""
        assert MOVE_TYPE_TO_CODE[MoveType.PLACE_RING] == "P"
        assert MOVE_TYPE_TO_CODE[MoveType.SKIP_PLACEMENT] == "SP"
        assert MOVE_TYPE_TO_CODE[MoveType.SWAP_SIDES] == "SW"
        assert MOVE_TYPE_TO_CODE[MoveType.MOVE_STACK] == "M"
        assert MOVE_TYPE_TO_CODE[MoveType.MOVE_RING] == "MR"
        assert MOVE_TYPE_TO_CODE[MoveType.BUILD_STACK] == "B"
        assert MOVE_TYPE_TO_CODE[MoveType.OVERTAKING_CAPTURE] == "C"
        assert MOVE_TYPE_TO_CODE[MoveType.CONTINUE_CAPTURE_SEGMENT] == "CC"

    def test_move_type_to_code_line_territory(self):
        """Test line and territory types."""
        assert MOVE_TYPE_TO_CODE[MoveType.PROCESS_LINE] == "L"
        assert MOVE_TYPE_TO_CODE[MoveType.CHOOSE_LINE_REWARD] == "LR"
        assert MOVE_TYPE_TO_CODE[MoveType.PROCESS_TERRITORY_REGION] == "T"
        assert MOVE_TYPE_TO_CODE[MoveType.ELIMINATE_RINGS_FROM_STACK] == "E"

    def test_code_to_move_type_reverse(self):
        """Test reverse mapping."""
        assert CODE_TO_MOVE_TYPE["P"] == MoveType.PLACE_RING
        assert CODE_TO_MOVE_TYPE["SP"] == MoveType.SKIP_PLACEMENT
        assert CODE_TO_MOVE_TYPE["SW"] == MoveType.SWAP_SIDES
        assert CODE_TO_MOVE_TYPE["M"] == MoveType.MOVE_STACK
        assert CODE_TO_MOVE_TYPE["MR"] == MoveType.MOVE_RING
        assert CODE_TO_MOVE_TYPE["B"] == MoveType.BUILD_STACK
        assert CODE_TO_MOVE_TYPE["C"] == MoveType.OVERTAKING_CAPTURE
        assert CODE_TO_MOVE_TYPE["CC"] == MoveType.CONTINUE_CAPTURE_SEGMENT
        assert CODE_TO_MOVE_TYPE["L"] == MoveType.PROCESS_LINE
        assert CODE_TO_MOVE_TYPE["LR"] == MoveType.CHOOSE_LINE_REWARD
        assert CODE_TO_MOVE_TYPE["T"] == MoveType.PROCESS_TERRITORY_REGION
        assert CODE_TO_MOVE_TYPE["E"] == MoveType.ELIMINATE_RINGS_FROM_STACK

    def test_ec_code_alias(self):
        """Test EC (End Chain) alias."""
        assert CODE_TO_MOVE_TYPE["EC"] == MoveType.CONTINUE_CAPTURE_SEGMENT


class TestSquarePositionConversion:
    """Tests for square board position conversion."""

    def test_position_to_algebraic_a1(self):
        """Test bottom-left corner."""
        pos = Position(x=0, y=0)
        assert _position_to_algebraic_square(pos) == "a1"

    def test_position_to_algebraic_h8(self):
        """Test top-right corner of 8x8."""
        pos = Position(x=7, y=7)
        assert _position_to_algebraic_square(pos) == "h8"

    def test_position_to_algebraic_d4(self):
        """Test middle position."""
        pos = Position(x=3, y=3)
        assert _position_to_algebraic_square(pos) == "d4"

    def test_position_to_algebraic_s19(self):
        """Test corner of 19x19 board."""
        pos = Position(x=18, y=18)
        assert _position_to_algebraic_square(pos) == "s19"

    def test_algebraic_to_position_a1(self):
        """Test parsing a1."""
        pos = _algebraic_to_position_square("a1")
        assert pos.x == 0
        assert pos.y == 0

    def test_algebraic_to_position_h8(self):
        """Test parsing h8."""
        pos = _algebraic_to_position_square("h8")
        assert pos.x == 7
        assert pos.y == 7

    def test_algebraic_to_position_d4(self):
        """Test parsing d4."""
        pos = _algebraic_to_position_square("d4")
        assert pos.x == 3
        assert pos.y == 3

    def test_algebraic_to_position_s19(self):
        """Test parsing 19x19 corner."""
        pos = _algebraic_to_position_square("s19")
        assert pos.x == 18
        assert pos.y == 18

    def test_algebraic_to_position_case_insensitive(self):
        """Test that parsing is case-insensitive."""
        pos_lower = _algebraic_to_position_square("d4")
        pos_upper = _algebraic_to_position_square("D4")
        assert pos_lower.x == pos_upper.x
        assert pos_lower.y == pos_upper.y

    def test_algebraic_to_position_invalid_empty(self):
        """Test empty notation raises error."""
        with pytest.raises(ValueError, match="Invalid square notation"):
            _algebraic_to_position_square("")

    def test_algebraic_to_position_invalid_short(self):
        """Test too short notation raises error."""
        with pytest.raises(ValueError, match="Invalid square notation"):
            _algebraic_to_position_square("a")

    def test_algebraic_to_position_invalid_column(self):
        """Test invalid column raises error."""
        with pytest.raises(ValueError, match="Invalid column"):
            _algebraic_to_position_square("z1")

    def test_algebraic_to_position_invalid_row(self):
        """Test invalid row raises error."""
        with pytest.raises(ValueError, match="Invalid row"):
            _algebraic_to_position_square("ax")

    def test_roundtrip_square(self):
        """Test conversion roundtrip for square positions."""
        for x in range(8):
            for y in range(8):
                pos = Position(x=x, y=y)
                notation = _position_to_algebraic_square(pos)
                parsed = _algebraic_to_position_square(notation)
                assert parsed.x == pos.x
                assert parsed.y == pos.y


class TestHexPositionConversion:
    """Tests for hexagonal board position conversion."""

    def test_position_to_algebraic_origin(self):
        """Test origin position."""
        pos = Position(x=0, y=0, z=0)
        assert _position_to_algebraic_hex(pos) == "0.0"

    def test_position_to_algebraic_positive(self):
        """Test positive coordinates."""
        pos = Position(x=3, y=-1, z=-2)
        assert _position_to_algebraic_hex(pos) == "3.-1"

    def test_position_to_algebraic_negative(self):
        """Test negative coordinates."""
        pos = Position(x=-2, y=4, z=-2)
        assert _position_to_algebraic_hex(pos) == "-2.4"

    def test_algebraic_to_position_origin(self):
        """Test parsing origin."""
        pos = _algebraic_to_position_hex("0.0")
        assert pos.x == 0
        assert pos.y == 0
        assert pos.z == 0

    def test_algebraic_to_position_positive(self):
        """Test parsing positive coordinates."""
        pos = _algebraic_to_position_hex("3.-2")
        assert pos.x == 3
        assert pos.y == -2
        assert pos.z == -1  # Derived from x + y + z = 0

    def test_algebraic_to_position_negative(self):
        """Test parsing negative coordinates."""
        pos = _algebraic_to_position_hex("-4.1")
        assert pos.x == -4
        assert pos.y == 1
        assert pos.z == 3  # -4 + 1 + 3 = 0

    def test_algebraic_to_position_z_constraint(self):
        """Test z is correctly derived from cube coordinate constraint."""
        # x + y + z = 0, so z = -(x + y)
        pos = _algebraic_to_position_hex("5.-3")
        assert pos.z == -(5 + (-3))  # z = -2

    def test_algebraic_to_position_invalid_no_dot(self):
        """Test missing dot raises error."""
        with pytest.raises(ValueError, match="missing '\\.'"):
            _algebraic_to_position_hex("34")

    def test_algebraic_to_position_invalid_format(self):
        """Test invalid format raises error."""
        with pytest.raises(ValueError, match="Invalid hex notation format"):
            _algebraic_to_position_hex("1.2.3")

    def test_algebraic_to_position_invalid_values(self):
        """Test non-numeric values raise error."""
        with pytest.raises(ValueError, match="Invalid hex coordinate values"):
            _algebraic_to_position_hex("a.b")

    def test_roundtrip_hex(self):
        """Test conversion roundtrip for hex positions."""
        for x in range(-4, 5):
            for y in range(-4, 5):
                if abs(x) + abs(y) <= 4:  # Valid hex radius
                    z = -x - y
                    pos = Position(x=x, y=y, z=z)
                    notation = _position_to_algebraic_hex(pos)
                    parsed = _algebraic_to_position_hex(notation)
                    assert parsed.x == pos.x
                    assert parsed.y == pos.y
                    assert parsed.z == pos.z


class TestPositionToAlgebraic:
    """Tests for the main position_to_algebraic function."""

    def test_square8_board(self):
        """Test square8 uses square notation."""
        pos = Position(x=3, y=4)
        result = position_to_algebraic(pos, BoardType.SQUARE8)
        assert result == "d5"

    def test_square19_board(self):
        """Test square19 uses square notation."""
        pos = Position(x=10, y=15)
        result = position_to_algebraic(pos, BoardType.SQUARE19)
        assert result == "k16"

    def test_hexagonal_board(self):
        """Test hexagonal uses hex notation."""
        pos = Position(x=2, y=-1, z=-1)
        result = position_to_algebraic(pos, BoardType.HEXAGONAL)
        assert result == "2.-1"

    def test_hex8_board(self):
        """Test hex8 uses hex notation."""
        pos = Position(x=-3, y=2, z=1)
        result = position_to_algebraic(pos, BoardType.HEX8)
        assert result == "-3.2"


class TestAlgebraicToPosition:
    """Tests for the main algebraic_to_position function."""

    def test_square8_board(self):
        """Test square8 parses square notation."""
        pos = algebraic_to_position("d5", BoardType.SQUARE8)
        assert pos.x == 3
        assert pos.y == 4

    def test_hexagonal_board(self):
        """Test hexagonal parses hex notation."""
        pos = algebraic_to_position("2.-1", BoardType.HEXAGONAL)
        assert pos.x == 2
        assert pos.y == -1

    def test_hex8_board(self):
        """Test hex8 parses hex notation."""
        pos = algebraic_to_position("-3.2", BoardType.HEX8)
        assert pos.x == -3
        assert pos.y == 2


class TestMoveToAlgebraic:
    """Tests for move_to_algebraic function."""

    def _make_move(
        self,
        move_type: MoveType,
        player: int = 1,
        from_pos: Position | None = None,
        to: Position | None = None,
        capture_target: Position | None = None,
        marker_left: Position | None = None,
    ) -> Move:
        """Helper to create Move objects."""
        return Move(
            id=str(uuid4()),
            type=move_type,
            player=player,
            from_pos=from_pos,
            to=to or Position(x=0, y=0),
            capture_target=capture_target,
            marker_left=marker_left,
            timestamp=datetime.now(),
            think_time=0,
            move_number=1,
        )

    def test_place_ring(self):
        """Test PLACE_RING notation."""
        move = self._make_move(
            MoveType.PLACE_RING,
            to=Position(x=3, y=3),
        )
        result = move_to_algebraic(move, BoardType.SQUARE8)
        assert result == "P d4"

    def test_skip_placement(self):
        """Test SKIP_PLACEMENT notation."""
        move = self._make_move(MoveType.SKIP_PLACEMENT)
        result = move_to_algebraic(move, BoardType.SQUARE8)
        assert result == "SP"

    def test_swap_sides(self):
        """Test SWAP_SIDES notation."""
        move = self._make_move(MoveType.SWAP_SIDES)
        result = move_to_algebraic(move, BoardType.SQUARE8)
        assert result == "SW"

    def test_move_stack(self):
        """Test MOVE_STACK with from-to notation."""
        move = self._make_move(
            MoveType.MOVE_STACK,
            from_pos=Position(x=3, y=3),
            to=Position(x=4, y=4),
        )
        result = move_to_algebraic(move, BoardType.SQUARE8)
        assert result == "M d4-e5"

    def test_move_stack_with_marker(self):
        """Test MOVE_STACK with marker annotation."""
        move = self._make_move(
            MoveType.MOVE_STACK,
            from_pos=Position(x=3, y=3),
            to=Position(x=4, y=4),
            marker_left=Position(x=3, y=4),
        )
        result = move_to_algebraic(move, BoardType.SQUARE8)
        assert result == "M d4-e5 @d5"

    def test_overtaking_capture(self):
        """Test OVERTAKING_CAPTURE notation."""
        move = self._make_move(
            MoveType.OVERTAKING_CAPTURE,
            from_pos=Position(x=2, y=2),
            to=Position(x=4, y=4),
            capture_target=Position(x=3, y=3),
        )
        result = move_to_algebraic(move, BoardType.SQUARE8)
        assert result == "C c3-e5 xd4"

    def test_continue_capture(self):
        """Test CONTINUE_CAPTURE_SEGMENT notation."""
        move = self._make_move(
            MoveType.CONTINUE_CAPTURE_SEGMENT,
            from_pos=Position(x=4, y=4),
            to=Position(x=6, y=6),
            capture_target=Position(x=5, y=5),
        )
        result = move_to_algebraic(move, BoardType.SQUARE8)
        assert result == "CC e5-g7 xf6"

    def test_process_line(self):
        """Test PROCESS_LINE notation."""
        move = self._make_move(MoveType.PROCESS_LINE)
        result = move_to_algebraic(move, BoardType.SQUARE8)
        assert result == "L 1"

    def test_choose_line_reward(self):
        """Test CHOOSE_LINE_REWARD notation."""
        move = self._make_move(MoveType.CHOOSE_LINE_REWARD)
        result = move_to_algebraic(move, BoardType.SQUARE8)
        assert result == "LR 1"

    def test_process_territory(self):
        """Test PROCESS_TERRITORY_REGION notation."""
        move = self._make_move(MoveType.PROCESS_TERRITORY_REGION)
        result = move_to_algebraic(move, BoardType.SQUARE8)
        assert result == "T 1"

    def test_eliminate_rings(self):
        """Test ELIMINATE_RINGS_FROM_STACK notation."""
        move = self._make_move(
            MoveType.ELIMINATE_RINGS_FROM_STACK,
            to=Position(x=5, y=5),
        )
        result = move_to_algebraic(move, BoardType.SQUARE8)
        assert result == "E f6"

    def test_hex_move(self):
        """Test move notation on hex board."""
        move = self._make_move(
            MoveType.PLACE_RING,
            to=Position(x=2, y=-1, z=-1),
        )
        result = move_to_algebraic(move, BoardType.HEXAGONAL)
        assert result == "P 2.-1"

    def test_unknown_move_type(self):
        """Test unknown move type gets '?' code."""
        # Create a move with a hypothetical unknown type
        move = self._make_move(MoveType.PLACE_RING)
        # Temporarily modify the mapping
        original = MOVE_TYPE_TO_CODE.get(MoveType.PLACE_RING)
        del MOVE_TYPE_TO_CODE[MoveType.PLACE_RING]
        try:
            result = move_to_algebraic(move, BoardType.SQUARE8)
            assert "?" in result
        finally:
            MOVE_TYPE_TO_CODE[MoveType.PLACE_RING] = original


class TestAlgebraicToMove:
    """Tests for algebraic_to_move function."""

    def _make_state(self, board_type: BoardType):
        """Helper to create minimal GameState mock.

        algebraic_to_move only needs state.board_type, so we use a simple mock.
        """
        class MockGameState:
            def __init__(self, bt: BoardType):
                self.board_type = bt
        return MockGameState(board_type)

    def test_parse_place_ring(self):
        """Test parsing PLACE_RING notation."""
        state = self._make_state(BoardType.SQUARE8)
        move = algebraic_to_move("P d4", state, player=1, move_number=1)
        assert move.type == MoveType.PLACE_RING
        assert move.to.x == 3
        assert move.to.y == 3

    def test_parse_skip_placement(self):
        """Test parsing SKIP_PLACEMENT notation."""
        state = self._make_state(BoardType.SQUARE8)
        move = algebraic_to_move("SP", state, player=1, move_number=1)
        assert move.type == MoveType.SKIP_PLACEMENT

    def test_parse_swap_sides(self):
        """Test parsing SWAP_SIDES notation."""
        state = self._make_state(BoardType.SQUARE8)
        move = algebraic_to_move("SW", state, player=2, move_number=2)
        assert move.type == MoveType.SWAP_SIDES
        assert move.player == 2
        assert move.move_number == 2

    def test_parse_move_stack(self):
        """Test parsing MOVE_STACK notation."""
        state = self._make_state(BoardType.SQUARE8)
        move = algebraic_to_move("M d4-e5", state, player=1, move_number=5)
        assert move.type == MoveType.MOVE_STACK
        assert move.from_pos.x == 3
        assert move.from_pos.y == 3
        assert move.to.x == 4
        assert move.to.y == 4

    def test_parse_capture(self):
        """Test parsing capture notation."""
        state = self._make_state(BoardType.SQUARE8)
        move = algebraic_to_move("C c3-e5 xd4", state, player=1, move_number=10)
        assert move.type == MoveType.OVERTAKING_CAPTURE
        assert move.from_pos.x == 2
        assert move.from_pos.y == 2
        assert move.to.x == 4
        assert move.to.y == 4
        assert move.capture_target.x == 3
        assert move.capture_target.y == 3

    def test_parse_with_marker(self):
        """Test parsing move with marker annotation."""
        state = self._make_state(BoardType.SQUARE8)
        move = algebraic_to_move("M d4-e5 @d5", state, player=1, move_number=5)
        assert move.marker_left.x == 3
        assert move.marker_left.y == 4

    def test_parse_hex_notation(self):
        """Test parsing hex board notation."""
        state = self._make_state(BoardType.HEXAGONAL)
        move = algebraic_to_move("P 2.-1", state, player=1, move_number=1)
        assert move.type == MoveType.PLACE_RING
        assert move.to.x == 2
        assert move.to.y == -1

    def test_parse_empty_raises(self):
        """Test empty notation raises error."""
        state = self._make_state(BoardType.SQUARE8)
        with pytest.raises(ValueError, match="Empty notation"):
            algebraic_to_move("", state, player=1, move_number=1)

    def test_parse_unknown_code_raises(self):
        """Test unknown move code raises error."""
        state = self._make_state(BoardType.SQUARE8)
        with pytest.raises(ValueError, match="Unknown move code"):
            algebraic_to_move("XYZ d4", state, player=1, move_number=1)

    def test_move_has_uuid(self):
        """Test parsed move has valid UUID."""
        state = self._make_state(BoardType.SQUARE8)
        move = algebraic_to_move("P d4", state, player=1, move_number=1)
        assert move.id is not None
        assert len(move.id) > 0

    def test_move_has_timestamp(self):
        """Test parsed move has timestamp."""
        state = self._make_state(BoardType.SQUARE8)
        move = algebraic_to_move("P d4", state, player=1, move_number=1)
        assert move.timestamp is not None
        assert isinstance(move.timestamp, datetime)


class TestMovesToNotationList:
    """Tests for moves_to_notation_list function."""

    def _make_move(self, move_type: MoveType, to: Position) -> Move:
        """Helper to create Move objects."""
        return Move(
            id=str(uuid4()),
            type=move_type,
            player=1,
            from_pos=None,
            to=to,
            capture_target=None,
            marker_left=None,
            timestamp=datetime.now(),
            think_time=0,
            move_number=1,
        )

    def test_empty_list(self):
        """Test empty move list."""
        result = moves_to_notation_list([], BoardType.SQUARE8)
        assert result == []

    def test_single_move(self):
        """Test single move conversion."""
        moves = [self._make_move(MoveType.PLACE_RING, Position(x=3, y=3))]
        result = moves_to_notation_list(moves, BoardType.SQUARE8)
        assert result == ["P d4"]

    def test_multiple_moves(self):
        """Test multiple move conversion."""
        moves = [
            self._make_move(MoveType.PLACE_RING, Position(x=3, y=3)),
            self._make_move(MoveType.PLACE_RING, Position(x=4, y=4)),
            self._make_move(MoveType.PLACE_RING, Position(x=5, y=5)),
        ]
        result = moves_to_notation_list(moves, BoardType.SQUARE8)
        assert len(result) == 3
        assert result[0] == "P d4"
        assert result[1] == "P e5"
        assert result[2] == "P f6"


class TestGameToPGN:
    """Tests for game_to_pgn function."""

    def _make_moves(self, n: int) -> list[Move]:
        """Helper to create a list of placement moves."""
        moves = []
        for i in range(n):
            moves.append(Move(
                id=str(uuid4()),
                type=MoveType.PLACE_RING,
                player=(i % 2) + 1,
                from_pos=None,
                to=Position(x=i, y=i),
                capture_target=None,
                marker_left=None,
                timestamp=datetime.now(),
                think_time=0,
                move_number=i + 1,
            ))
        return moves

    def test_minimal_pgn(self):
        """Test minimal PGN output."""
        moves = self._make_moves(2)
        metadata: dict[str, str | int | None] = {}
        result = game_to_pgn(moves, metadata, BoardType.SQUARE8)

        assert '[Game "RingRift"]' in result
        assert '[Board "square8"]' in result
        assert '[TotalMoves "2"]' in result
        assert "1. P a1" in result

    def test_pgn_with_metadata(self):
        """Test PGN with full metadata."""
        moves = self._make_moves(4)
        metadata = {
            "date": "2025-12-28",
            "player1": "Alice",
            "player2": "Bob",
            "winner": 1,
            "termination": "normal",
            "rng_seed": "12345",
        }
        result = game_to_pgn(moves, metadata, BoardType.SQUARE8)

        assert '[Date "2025-12-28"]' in result
        assert '[Player1 "Alice"]' in result
        assert '[Player2 "Bob"]' in result
        assert '[Result "1-0"]' in result
        assert '[Termination "normal"]' in result
        assert '[RNGSeed "12345"]' in result

    def test_pgn_result_player2_wins(self):
        """Test PGN result when player 2 wins."""
        moves = self._make_moves(2)
        metadata = {"winner": 2}
        result = game_to_pgn(moves, metadata, BoardType.SQUARE8)
        assert '[Result "0-1"]' in result

    def test_pgn_result_draw(self):
        """Test PGN result for draw (winner != 1 or 2)."""
        moves = self._make_moves(2)
        metadata = {"winner": 0}
        result = game_to_pgn(moves, metadata, BoardType.SQUARE8)
        assert '[Result "1/2-1/2"]' in result

    def test_pgn_result_incomplete(self):
        """Test PGN result for incomplete game."""
        moves = self._make_moves(2)
        metadata: dict[str, str | int | None] = {}
        result = game_to_pgn(moves, metadata, BoardType.SQUARE8)
        assert '[Result "*"]' in result

    def test_pgn_board_type_square19(self):
        """Test PGN board type for square19."""
        moves = self._make_moves(2)
        result = game_to_pgn(moves, {}, BoardType.SQUARE19)
        assert '[Board "square19"]' in result

    def test_pgn_board_type_hexagonal(self):
        """Test PGN board type for hexagonal."""
        moves = self._make_moves(2)
        result = game_to_pgn(moves, {}, BoardType.HEXAGONAL)
        assert '[Board "hexagonal"]' in result

    def test_pgn_turn_numbering(self):
        """Test turn-based move notation."""
        moves = self._make_moves(6)
        result = game_to_pgn(moves, {}, BoardType.SQUARE8)

        # Should have turns 1, 2, 3
        assert "1. P a1" in result
        assert "2. P c3" in result
        assert "3. P e5" in result

    def test_pgn_odd_moves(self):
        """Test PGN with odd number of moves."""
        moves = self._make_moves(5)
        result = game_to_pgn(moves, {}, BoardType.SQUARE8)

        # Last turn should have only one move
        lines = result.split('\n')
        move_lines = [l for l in lines if l.strip().startswith('3.')]
        assert len(move_lines) == 1


class TestParsePGN:
    """Tests for parse_pgn function."""

    def test_parse_minimal_pgn(self):
        """Test parsing minimal PGN."""
        pgn = '''[Game "RingRift"]
[Board "square8"]

1. P d4      P e5
1-0'''
        metadata, moves = parse_pgn(pgn)

        assert metadata["game"] == "RingRift"
        assert metadata["board"] == "square8"
        assert len(moves) == 2
        assert "P" in moves[0]
        assert "P" in moves[1]

    def test_parse_full_metadata(self):
        """Test parsing full metadata."""
        pgn = '''[Game "RingRift"]
[Board "hexagonal"]
[Date "2025-12-28"]
[Player1 "Alice"]
[Player2 "Bob"]
[Result "1-0"]
[Termination "normal"]
[RNGSeed "42"]
[TotalMoves "10"]

1. P 0.0      P 1.0
*'''
        metadata, _ = parse_pgn(pgn)

        assert metadata["date"] == "2025-12-28"
        assert metadata["player1"] == "Alice"
        assert metadata["player2"] == "Bob"
        assert metadata["result"] == "1-0"
        assert metadata["termination"] == "normal"
        assert metadata["rngseed"] == "42"
        assert metadata["totalmoves"] == "10"

    def test_parse_result_indicators(self):
        """Test that result indicators are not included in moves."""
        pgn = '''[Game "RingRift"]

1. P d4
1-0'''
        _, moves = parse_pgn(pgn)
        assert "1-0" not in moves

        pgn2 = '''[Game "RingRift"]

1. P d4
0-1'''
        _, moves2 = parse_pgn(pgn2)
        assert "0-1" not in moves2

        pgn3 = '''[Game "RingRift"]

1. P d4
1/2-1/2'''
        _, moves3 = parse_pgn(pgn3)
        assert "1/2-1/2" not in moves3

    def test_parse_multiple_turns(self):
        """Test parsing multiple turns."""
        pgn = '''[Game "RingRift"]

1. P a1      P b2
2. M a1-c3   C b2-d4 xc3
3. L 1
*'''
        _, moves = parse_pgn(pgn)

        assert len(moves) == 5
        assert moves[0] == "P"  # Code only after removing position
        assert "M" in moves[2]
        assert "C" in moves[3]

    def test_parse_ignores_comments(self):
        """Test that curly-brace comments are ignored."""
        pgn = '''[Game "RingRift"]

1. P a1 {good move}     P b2
*'''
        _, moves = parse_pgn(pgn)

        # Comments should be filtered out
        for move in moves:
            assert "{" not in move
            assert "}" not in move

    def test_parse_empty_pgn(self):
        """Test parsing empty PGN."""
        pgn = ""
        metadata, moves = parse_pgn(pgn)
        assert metadata == {}
        assert moves == []

    def test_parse_roundtrip(self):
        """Test that generating and parsing PGN preserves structure."""
        # Create original moves
        original_moves = []
        for i in range(4):
            original_moves.append(Move(
                id=str(uuid4()),
                type=MoveType.PLACE_RING,
                player=(i % 2) + 1,
                from_pos=None,
                to=Position(x=i, y=i),
                capture_target=None,
                marker_left=None,
                timestamp=datetime.now(),
                think_time=0,
                move_number=i + 1,
            ))

        # Generate PGN
        metadata = {"date": "2025-12-28", "winner": 1}
        pgn = game_to_pgn(original_moves, metadata, BoardType.SQUARE8)

        # Parse PGN
        parsed_metadata, parsed_moves = parse_pgn(pgn)

        # Verify
        assert parsed_metadata["date"] == "2025-12-28"
        assert parsed_metadata["result"] == "1-0"
        assert len(parsed_moves) == 4

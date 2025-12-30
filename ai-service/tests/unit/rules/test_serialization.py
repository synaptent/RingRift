"""Tests for app.rules.serialization module.

Tests round-trip serialization/deserialization for cross-language parity.
Each type should survive: object -> serialize -> deserialize -> equals original.
"""

import pytest

from app.models import (
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    MarkerInfo,
    Move,
    MoveType,
    Player,
    Position,
    RingStack,
)
from app.rules.serialization import (
    deserialize_board_state,
    deserialize_game_state,
    deserialize_marker,
    deserialize_move,
    deserialize_player,
    deserialize_position,
    deserialize_stack,
    serialize_board_state,
    serialize_game_state,
    serialize_marker,
    serialize_move,
    serialize_player,
    serialize_position,
    serialize_stack,
)


class TestPositionSerialization:
    """Round-trip tests for Position serialization."""

    def test_position_2d_roundtrip(self):
        """2D position serializes and deserializes correctly."""
        pos = Position(x=3, y=5)
        serialized = serialize_position(pos)
        deserialized = deserialize_position(serialized)
        assert deserialized.x == pos.x
        assert deserialized.y == pos.y
        assert deserialized.z == pos.z

    def test_position_3d_roundtrip(self):
        """3D position (with z) serializes correctly."""
        pos = Position(x=1, y=2, z=3)
        serialized = serialize_position(pos)
        assert "z" in serialized
        assert serialized["z"] == 3
        deserialized = deserialize_position(serialized)
        assert deserialized.z == 3

    def test_position_zero_coords(self):
        """Position at origin serializes correctly."""
        pos = Position(x=0, y=0)
        serialized = serialize_position(pos)
        deserialized = deserialize_position(serialized)
        assert deserialized.x == 0
        assert deserialized.y == 0

    def test_position_negative_coords(self):
        """Negative coordinates (hex boards) serialize correctly."""
        pos = Position(x=-3, y=2)
        serialized = serialize_position(pos)
        deserialized = deserialize_position(serialized)
        assert deserialized.x == -3
        assert deserialized.y == 2


class TestRingStackSerialization:
    """Round-trip tests for RingStack serialization."""

    def test_empty_stack_roundtrip(self):
        """Empty stack serializes correctly."""
        stack = RingStack(
            position=Position(x=0, y=0),
            rings=[],
            stackHeight=0,
            capHeight=0,
            controllingPlayer=None,
        )
        serialized = serialize_stack(stack)
        deserialized = deserialize_stack(serialized)
        assert deserialized.stack_height == 0
        assert len(deserialized.rings) == 0

    def test_single_ring_stack_roundtrip(self):
        """Single-ring stack serializes correctly."""
        stack = RingStack(
            position=Position(x=2, y=3),
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )
        serialized = serialize_stack(stack)
        deserialized = deserialize_stack(serialized)
        assert deserialized.stack_height == 1
        assert deserialized.cap_height == 1
        assert deserialized.controlling_player == 1

    def test_multi_ring_stack_roundtrip(self):
        """Multi-ring stack with multiple players serializes correctly."""
        # Rings from bottom to top: player 1, player 2, player 1
        stack = RingStack(
            position=Position(x=4, y=4),
            rings=[1, 2, 1],
            stackHeight=3,
            capHeight=1,
            controllingPlayer=1,
        )
        serialized = serialize_stack(stack)
        deserialized = deserialize_stack(serialized)
        assert deserialized.stack_height == 3
        # Note: deserialization reverses rings for TS/Python parity
        assert len(deserialized.rings) == 3


class TestMarkerInfoSerialization:
    """Round-trip tests for MarkerInfo serialization."""

    def test_marker_roundtrip(self):
        """MarkerInfo serializes correctly."""
        marker = MarkerInfo(
            position=Position(x=1, y=2),
            player=2,
            type="regular",
        )
        serialized = serialize_marker(marker)
        deserialized = deserialize_marker(serialized)
        assert deserialized.position.x == 1
        assert deserialized.position.y == 2
        assert deserialized.player == 2


class TestPlayerSerialization:
    """Round-trip tests for Player serialization."""

    def test_active_player_roundtrip(self):
        """Active player with rings serializes correctly."""
        player = Player(
            id=1,
            ringsInHand=10,
            ringsOnBoard=5,
            capturedRings=2,
            isEliminated=False,
            eliminationTurn=None,
            territoryCells=8,
        )
        serialized = serialize_player(player)
        deserialized = deserialize_player(serialized, index=0)
        assert deserialized.id == 1
        assert deserialized.rings_in_hand == 10
        assert deserialized.rings_on_board == 5
        assert deserialized.captured_rings == 2
        assert deserialized.is_eliminated is False
        assert deserialized.territory_cells == 8

    def test_eliminated_player_roundtrip(self):
        """Eliminated player serializes correctly."""
        player = Player(
            id=2,
            ringsInHand=0,
            ringsOnBoard=0,
            capturedRings=0,
            isEliminated=True,
            eliminationTurn=15,
            territoryCells=0,
        )
        serialized = serialize_player(player)
        deserialized = deserialize_player(serialized, index=1)
        assert deserialized.is_eliminated is True
        assert deserialized.elimination_turn == 15


class TestMoveSerialization:
    """Round-trip tests for Move serialization."""

    def test_place_ring_move_roundtrip(self):
        """PLACE_RING move serializes correctly."""
        move = Move(
            id="test-move-1",
            type=MoveType.PLACE_RING,
            player=1,
            to=Position(x=3, y=4),
            timestamp=None,
            thinkTime=0,
            moveNumber=5,
        )
        serialized = serialize_move(move)
        deserialized = deserialize_move(serialized)
        assert deserialized is not None
        assert deserialized.type == MoveType.PLACE_RING
        assert deserialized.player == 1
        assert deserialized.to.x == 3
        assert deserialized.to.y == 4

    def test_move_stack_move_roundtrip(self):
        """MOVE_STACK move with from/to serializes correctly."""
        move = Move(
            id="test-move-2",
            type=MoveType.MOVE_STACK,
            player=2,
            **{"from": Position(x=1, y=1)},  # 'from' is reserved keyword
            to=Position(x=4, y=4),
            timestamp=None,
            thinkTime=100,
            moveNumber=10,
        )
        serialized = serialize_move(move)
        deserialized = deserialize_move(serialized)
        assert deserialized is not None
        assert deserialized.type == MoveType.MOVE_STACK
        assert deserialized.from_pos is not None
        assert deserialized.from_pos.x == 1
        assert deserialized.to.x == 4


class TestBoardStateSerialization:
    """Round-trip tests for BoardState serialization."""

    def test_empty_board_roundtrip(self):
        """Empty board serializes correctly."""
        board = BoardState(
            type=BoardType.SQUARE8,
            stacks={},
            markers={},
            collapsedSpaces=[],
            territories={},
        )
        serialized = serialize_board_state(board)
        deserialized = deserialize_board_state(serialized)
        assert deserialized.type == BoardType.SQUARE8
        assert len(deserialized.stacks) == 0
        assert len(deserialized.markers) == 0

    def test_board_with_stacks_roundtrip(self):
        """Board with stacks serializes correctly."""
        board = BoardState(
            type=BoardType.HEX8,
            stacks={
                "0,0": RingStack(
                    position=Position(x=0, y=0),
                    rings=[1],
                    stackHeight=1,
                    capHeight=1,
                    controllingPlayer=1,
                ),
            },
            markers={},
            collapsedSpaces=[],
            territories={},
        )
        serialized = serialize_board_state(board)
        deserialized = deserialize_board_state(serialized)
        assert deserialized.type == BoardType.HEX8
        assert "0,0" in deserialized.stacks
        assert deserialized.stacks["0,0"].controlling_player == 1


class TestGameStateSerialization:
    """Round-trip tests for GameState serialization."""

    @pytest.fixture
    def minimal_game_state(self):
        """Create a minimal valid game state."""
        return GameState(
            game_id="test-game-123",
            phase=GamePhase.RING_PLACEMENT,
            status=GameStatus.IN_PROGRESS,
            players=[
                Player(
                    id=1,
                    ringsInHand=15,
                    ringsOnBoard=0,
                    capturedRings=0,
                    isEliminated=False,
                    eliminationTurn=None,
                    territoryCells=0,
                ),
                Player(
                    id=2,
                    ringsInHand=15,
                    ringsOnBoard=0,
                    capturedRings=0,
                    isEliminated=False,
                    eliminationTurn=None,
                    territoryCells=0,
                ),
            ],
            board=BoardState(
                type=BoardType.SQUARE8,
                stacks={},
                markers={},
                collapsedSpaces=[],
                territories={},
            ),
            moveHistory=[],
            currentPlayer=1,
            turn=1,
        )

    def test_minimal_game_state_roundtrip(self, minimal_game_state):
        """Minimal game state serializes correctly."""
        serialized = serialize_game_state(minimal_game_state)
        deserialized = deserialize_game_state(serialized)
        assert deserialized.game_id == "test-game-123"
        assert deserialized.phase == GamePhase.RING_PLACEMENT
        assert deserialized.status == GameStatus.IN_PROGRESS
        assert deserialized.current_player == 1
        assert len(deserialized.players) == 2

    def test_game_state_preserves_phase(self, minimal_game_state):
        """Phase is preserved through serialization."""
        for phase in [
            GamePhase.RING_PLACEMENT,
            GamePhase.MOVEMENT,
            GamePhase.CAPTURE,
            GamePhase.LINE_PROCESSING,
        ]:
            state = minimal_game_state.model_copy(update={"phase": phase})
            serialized = serialize_game_state(state)
            deserialized = deserialize_game_state(serialized)
            assert deserialized.phase == phase

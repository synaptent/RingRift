"""
Unit tests for app.rules.mutable_state module.

Tests cover:
- MutableStack dataclass and conversions
- MutableMarker dataclass and conversions
- MutablePlayerState dataclass
- MoveUndo dataclass
- MutableGameState class

Created: December 2025
"""

import pytest
from unittest.mock import MagicMock, patch

from app.models import Position, BoardType, GamePhase


# =============================================================================
# MutableStack Tests
# =============================================================================


class TestMutableStack:
    """Tests for MutableStack dataclass."""

    def test_from_ring_stack(self):
        """MutableStack can be created from RingStack."""
        from app.rules.mutable_state import MutableStack
        from app.models import RingStack

        ring_stack = RingStack(
            position=Position(x=3, y=4),
            rings=[1, 2, 1],
            stackHeight=3,
            capHeight=1,
            controllingPlayer=1,
        )

        mutable = MutableStack.from_ring_stack(ring_stack)

        assert mutable.position == Position(x=3, y=4)
        assert mutable.rings == [1, 2, 1]
        assert mutable.stack_height == 3
        assert mutable.cap_height == 1
        assert mutable.controlling_player == 1

    def test_to_ring_stack(self):
        """MutableStack can be converted back to RingStack."""
        from app.rules.mutable_state import MutableStack

        mutable = MutableStack(
            position=Position(x=2, y=3),
            rings=[1, 1],
            stack_height=2,
            cap_height=2,
            controlling_player=1,
        )

        ring_stack = mutable.to_ring_stack()

        assert ring_stack.position == Position(x=2, y=3)
        assert ring_stack.rings == [1, 1]
        assert ring_stack.stack_height == 2

    def test_copy(self):
        """MutableStack copy creates independent copy."""
        from app.rules.mutable_state import MutableStack

        original = MutableStack(
            position=Position(x=1, y=1),
            rings=[1, 2],
            stack_height=2,
            cap_height=1,
            controlling_player=2,
        )

        copy = original.copy()

        # Modify original
        original.rings.append(1)

        # Copy should be unchanged
        assert len(copy.rings) == 2
        assert len(original.rings) == 3

    def test_recompute_properties_empty(self):
        """Recompute works for empty stack."""
        from app.rules.mutable_state import MutableStack

        stack = MutableStack(
            position=Position(x=0, y=0),
            rings=[],
            stack_height=0,
            cap_height=0,
            controlling_player=0,
        )

        stack.recompute_properties()

        assert stack.stack_height == 0
        assert stack.cap_height == 0
        assert stack.controlling_player == 0

    def test_recompute_properties_single_ring(self):
        """Recompute works for single ring."""
        from app.rules.mutable_state import MutableStack

        stack = MutableStack(
            position=Position(x=0, y=0),
            rings=[1],
            stack_height=0,  # Wrong, should be 1
            cap_height=0,
            controlling_player=0,
        )

        stack.recompute_properties()

        assert stack.stack_height == 1
        assert stack.cap_height == 1
        assert stack.controlling_player == 1

    def test_recompute_properties_mixed_rings(self):
        """Recompute works for mixed ring stack."""
        from app.rules.mutable_state import MutableStack

        stack = MutableStack(
            position=Position(x=0, y=0),
            rings=[1, 2, 2, 2],  # P2 controls with cap of 3
            stack_height=0,
            cap_height=0,
            controlling_player=0,
        )

        stack.recompute_properties()

        assert stack.stack_height == 4
        assert stack.cap_height == 3
        assert stack.controlling_player == 2


# =============================================================================
# MutableMarker Tests
# =============================================================================


class TestMutableMarker:
    """Tests for MutableMarker dataclass."""

    def test_from_marker_info(self):
        """MutableMarker can be created from MarkerInfo."""
        from app.rules.mutable_state import MutableMarker
        from app.models import MarkerInfo

        marker_info = MarkerInfo(
            player=1,
            position=Position(x=5, y=6),
            type="regular",
        )

        mutable = MutableMarker.from_marker_info(marker_info)

        assert mutable.player == 1
        assert mutable.position == Position(x=5, y=6)
        assert mutable.marker_type == "regular"

    def test_to_marker_info(self):
        """MutableMarker can be converted back to MarkerInfo."""
        from app.rules.mutable_state import MutableMarker

        mutable = MutableMarker(
            player=2,
            position=Position(x=3, y=4),
            marker_type="regular",
        )

        marker_info = mutable.to_marker_info()

        assert marker_info.player == 2
        assert marker_info.position == Position(x=3, y=4)

    def test_copy(self):
        """MutableMarker copy creates independent copy."""
        from app.rules.mutable_state import MutableMarker

        original = MutableMarker(
            player=1,
            position=Position(x=1, y=2),
            marker_type="regular",
        )

        copy = original.copy()

        original.player = 2

        assert copy.player == 1


# =============================================================================
# MutablePlayerState Tests
# =============================================================================


class TestMutablePlayerState:
    """Tests for MutablePlayerState dataclass."""

    def test_creation(self):
        """MutablePlayerState can be created."""
        from app.rules.mutable_state import MutablePlayerState

        state = MutablePlayerState(
            player_number=1,
            rings_in_hand=5,
            eliminated_rings=0,
            territory_spaces=0,
        )

        assert state.player_number == 1
        assert state.rings_in_hand == 5
        assert state.eliminated_rings == 0
        assert state.territory_spaces == 0


# =============================================================================
# MoveUndo Tests
# =============================================================================


class TestMoveUndo:
    """Tests for MoveUndo dataclass."""

    def test_creation(self):
        """MoveUndo can be created with required fields."""
        from app.rules.mutable_state import MoveUndo
        from app.models import Move, MoveType

        move = Move(
            id="move-1",
            type=MoveType.PLACE_RING,
            player=1,
            to=Position(x=3, y=3),
        )

        undo = MoveUndo(move=move)

        assert undo.move == move
        # Default collections should be empty
        assert len(undo.removed_stacks) == 0
        assert len(undo.added_stacks) == 0
        assert len(undo.modified_stacks) == 0


# =============================================================================
# MutableGameState Tests
# =============================================================================


class TestMutableGameState:
    """Tests for MutableGameState class."""

    def test_class_exists(self):
        """MutableGameState class exists."""
        from app.rules.mutable_state import MutableGameState
        assert MutableGameState is not None

    def test_has_from_immutable(self):
        """MutableGameState has from_immutable class method."""
        from app.rules.mutable_state import MutableGameState
        assert hasattr(MutableGameState, 'from_immutable')
        assert callable(MutableGameState.from_immutable)

    def test_has_make_move(self):
        """MutableGameState has make_move method."""
        from app.rules.mutable_state import MutableGameState
        assert hasattr(MutableGameState, 'make_move')

    def test_has_unmake_move(self):
        """MutableGameState has unmake_move method."""
        from app.rules.mutable_state import MutableGameState
        assert hasattr(MutableGameState, 'unmake_move')

    def test_has_to_immutable(self):
        """MutableGameState has to_immutable method."""
        from app.rules.mutable_state import MutableGameState
        assert hasattr(MutableGameState, 'to_immutable')


# =============================================================================
# Module-level Tests
# =============================================================================


class TestModuleImports:
    """Tests for module-level imports and availability."""

    def test_all_exports_available(self):
        """All main exports are available from module."""
        from app.rules.mutable_state import (
            MutableStack,
            MutableMarker,
            MutablePlayerState,
            MoveUndo,
            MutableGameState,
        )

        assert MutableStack is not None
        assert MutableMarker is not None
        assert MutablePlayerState is not None
        assert MoveUndo is not None
        assert MutableGameState is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

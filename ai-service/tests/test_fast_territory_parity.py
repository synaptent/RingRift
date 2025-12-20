"""
Test parity between fast territory detection and original implementation.

This test ensures that the optimized NumPy-based territory detection
produces identical results to the original Python implementation.
"""

import pytest
from typing import List, Set
from app.board_manager import BoardManager
from app.models import (
    BoardState, BoardType, Position, RingStack, MarkerInfo, Territory
)


def territories_equivalent(t1: list[Territory], t2: list[Territory]) -> bool:
    """Check if two lists of territories are equivalent (order-independent)."""
    if len(t1) != len(t2):
        return False

    # Convert to sets of frozen position sets for comparison
    def territory_to_positions(t: Territory) -> frozenset:
        return frozenset((p.x, p.y, p.z) for p in t.spaces)

    set1 = {territory_to_positions(t) for t in t1}
    set2 = {territory_to_positions(t) for t in t2}

    return set1 == set2


def create_board(
    board_type: BoardType,
    size: int,
    stacks: dict,
    markers: dict,
    collapsed: dict = None,
) -> BoardState:
    """Helper to create a BoardState."""
    return BoardState(
        type=board_type,
        size=size,
        stacks=stacks,
        markers=markers,
        collapsedSpaces=collapsed or {},
        eliminatedRings={},
        formedLines=[],
        territories={},
    )


def create_stack(x: int, y: int, player: int, height: int = 1) -> RingStack:
    """Helper to create a RingStack."""
    return RingStack(
        position=Position(x=x, y=y),
        rings=[player for _ in range(height)],  # List of player numbers
        controllingPlayer=player,
        stackHeight=height,
        capHeight=1,
    )


def create_marker(x: int, y: int, player: int) -> MarkerInfo:
    """Helper to create a MarkerInfo."""
    return MarkerInfo(
        position=Position(x=x, y=y),
        player=player,
        type='regular',
    )


class TestFastTerritoryParity:
    """Test cases for fast territory detection parity."""

    def run_parity_test(self, board: BoardState, player: int = 1) -> None:
        """Run both implementations and compare results."""
        import os

        # Run original implementation (with fast territory disabled)
        os.environ['RINGRIFT_USE_FAST_TERRITORY'] = 'false'
        # Need to reload to pick up the env change
        import importlib
        import app.board_manager as bm
        importlib.reload(bm)

        original_result = bm.BoardManager.find_disconnected_regions(board, player)

        # Run fast implementation
        os.environ['RINGRIFT_USE_FAST_TERRITORY'] = 'true'
        importlib.reload(bm)

        fast_result = bm.BoardManager.find_disconnected_regions(board, player)

        # Reset to default
        os.environ['RINGRIFT_USE_FAST_TERRITORY'] = 'false'
        importlib.reload(bm)

        # Compare results
        assert territories_equivalent(original_result, fast_result), (
            f"Territory mismatch!\n"
            f"Original: {len(original_result)} territories\n"
            f"Fast: {len(fast_result)} territories\n"
            f"Original positions: {[[(p.x, p.y) for p in t.spaces] for t in original_result]}\n"
            f"Fast positions: {[[(p.x, p.y) for p in t.spaces] for t in fast_result]}"
        )

    def test_empty_board_square8(self):
        """Empty board should have no disconnected regions."""
        board = create_board(BoardType.SQUARE8, 8, {}, {})
        self.run_parity_test(board)

    def test_single_player_square8(self):
        """Single player on board should have no disconnected regions."""
        stacks = {
            "3,3": create_stack(3, 3, 1),
            "4,4": create_stack(4, 4, 1),
        }
        board = create_board(BoardType.SQUARE8, 8, stacks, {})
        self.run_parity_test(board)

    def test_two_players_no_markers_square8(self):
        """Two players without markers - no disconnected regions."""
        stacks = {
            "2,2": create_stack(2, 2, 1),
            "5,5": create_stack(5, 5, 2),
        }
        board = create_board(BoardType.SQUARE8, 8, stacks, {})
        self.run_parity_test(board)

    def test_marker_border_creates_region_square8(self):
        """Markers forming a border should create disconnected region."""
        stacks = {
            "1,1": create_stack(1, 1, 1),
            "6,6": create_stack(6, 6, 2),
        }
        # Create a line of markers that could form a border
        markers = {
            "3,0": create_marker(3, 0, 1),
            "3,1": create_marker(3, 1, 1),
            "3,2": create_marker(3, 2, 1),
            "3,3": create_marker(3, 3, 1),
            "3,4": create_marker(3, 4, 1),
            "3,5": create_marker(3, 5, 1),
            "3,6": create_marker(3, 6, 1),
            "3,7": create_marker(3, 7, 1),
        }
        board = create_board(BoardType.SQUARE8, 8, stacks, markers)
        self.run_parity_test(board)

    def test_collapsed_spaces_square8(self):
        """Collapsed spaces should act as borders."""
        stacks = {
            "1,1": create_stack(1, 1, 1),
            "6,6": create_stack(6, 6, 2),
        }
        collapsed = {
            "3,0": True,
            "3,1": True,
            "3,2": True,
            "3,3": True,
            "3,4": True,
            "3,5": True,
            "3,6": True,
            "3,7": True,
        }
        board = create_board(BoardType.SQUARE8, 8, stacks, {}, collapsed)
        self.run_parity_test(board)

    def test_complex_scenario_square8(self):
        """Complex scenario with multiple stacks, markers, and collapsed."""
        stacks = {
            "1,1": create_stack(1, 1, 1),
            "2,2": create_stack(2, 2, 1),
            "5,5": create_stack(5, 5, 2),
            "6,6": create_stack(6, 6, 2),
        }
        markers = {
            "3,3": create_marker(3, 3, 1),
            "3,4": create_marker(3, 4, 1),
            "4,3": create_marker(4, 3, 2),
            "4,4": create_marker(4, 4, 2),
        }
        collapsed = {
            "0,0": True,
            "7,7": True,
        }
        board = create_board(BoardType.SQUARE8, 8, stacks, markers, collapsed)
        self.run_parity_test(board)

    def test_empty_board_hex(self):
        """Empty hex board should have no disconnected regions."""
        board = create_board(BoardType.HEXAGONAL, 7, {}, {})
        self.run_parity_test(board)

    def test_two_players_hex(self):
        """Two players on hex board."""
        stacks = {
            "0,0": create_stack(0, 0, 1),
            "3,0": create_stack(3, 0, 2),
        }
        board = create_board(BoardType.HEXAGONAL, 7, stacks, {})
        self.run_parity_test(board)

    def test_marker_border_hex(self):
        """Markers on hex board."""
        stacks = {
            "-2,0": create_stack(-2, 0, 1),
            "2,0": create_stack(2, 0, 2),
        }
        markers = {
            "0,-1": create_marker(0, -1, 1),
            "0,0": create_marker(0, 0, 1),
            "0,1": create_marker(0, 1, 1),
        }
        board = create_board(BoardType.HEXAGONAL, 7, stacks, markers)
        self.run_parity_test(board)

    def test_square19_basic(self):
        """Basic test on 19x19 board."""
        stacks = {
            "5,5": create_stack(5, 5, 1),
            "15,15": create_stack(15, 15, 2),
        }
        board = create_board(BoardType.SQUARE19, 19, stacks, {})
        self.run_parity_test(board)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

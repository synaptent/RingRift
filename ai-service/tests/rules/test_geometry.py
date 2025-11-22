import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.models import BoardType, Position  # noqa: E402
from app.rules.geometry import BoardGeometry  # noqa: E402


@pytest.mark.parametrize(
    "board_type, from_pos, to_pos, expected",
    [
        # Square boards use Chebyshev (king-move) distance
        (
            BoardType.SQUARE8,
            Position(x=0, y=0),
            Position(x=3, y=4),
            4,
        ),
        (
            BoardType.SQUARE8,
            Position(x=2, y=2),
            Position(x=5, y=2),
            3,
        ),
        (
            BoardType.SQUARE19,
            Position(x=8, y=8),
            Position(x=10, y=13),
            5,
        ),
        # Hex boards use cube distance
        (
            BoardType.HEXAGONAL,
            Position(x=0, y=0, z=0),
            Position(x=1, y=-1, z=0),
            1,
        ),
        (
            BoardType.HEXAGONAL,
            Position(x=0, y=0, z=0),
            Position(x=2, y=-1, z=-1),
            2,
        ),
    ],
)
def test_calculate_distance(
    board_type: BoardType,
    from_pos: Position,
    to_pos: Position,
    expected: int,
) -> None:
    """BoardGeometry.calculate_distance mirrors TS core.calculateDistance.

    These cases cover Chebyshev distance on square boards and cube distance
    on hex boards.
    """
    dist = BoardGeometry.calculate_distance(board_type, from_pos, to_pos)
    assert dist == expected


def test_get_path_positions_square_straight_and_diagonal() -> None:
    """Path positions on square boards should match TS getPathPositions."""
    start = Position(x=0, y=0)

    # Horizontal ray
    end_horizontal = Position(x=3, y=0)
    path_h = BoardGeometry.get_path_positions(start, end_horizontal)
    assert [p.to_key() for p in path_h] == [
        "0,0",
        "1,0",
        "2,0",
        "3,0",
    ]

    # Diagonal ray
    end_diagonal = Position(x=3, y=3)
    path_d = BoardGeometry.get_path_positions(start, end_diagonal)
    assert [p.to_key() for p in path_d] == [
        "0,0",
        "1,1",
        "2,2",
        "3,3",
    ]


def test_get_path_positions_hexagonal() -> None:
    """Hexagonal paths should step in cube coordinates like TS
    getPathPositions.

    For a move from (0,0,0) to (2,-1,-1), the intermediate cube
    coordinate should be (1,0,0) after rounding.
    """
    start = Position(x=0, y=0, z=0)
    end = Position(x=2, y=-1, z=-1)

    path = BoardGeometry.get_path_positions(start, end)

    # Endpoints preserved
    assert path[0] == start
    assert path[-1] == end

    # One intermediate step with rounded cube coordinates
    assert len(path) == 3
    assert path[1] == Position(x=1, y=0, z=0)


def test_get_path_positions_degenerate_single_point() -> None:
    """When from == to, the path should contain exactly that position."""
    pos = Position(x=5, y=7)
    path = BoardGeometry.get_path_positions(pos, pos)
    assert path == [pos]

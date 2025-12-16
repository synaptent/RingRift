"""
Shared geometry logic for RingRift AI and Rules Engine.
Centralizes adjacency, line-of-sight, and board coordinate calculations.
"""

from typing import List, Tuple, Set
from ..models import BoardType, Position


class BoardGeometry:
    """Helper class for board geometry calculations"""

    @staticmethod
    def get_adjacent_positions(
        position: Position,
        board_type: BoardType,
        board_size: int
    ) -> List[Position]:
        """
        Get all valid adjacent positions for a given position.

        Args:
            position: The center position
            board_type: Type of board (square8, square19, hexagonal)
            board_size: Size of the board

        Returns:
            List of valid adjacent Position objects
        """
        neighbors: List[Position] = []

        if board_type in (BoardType.SQUARE8, BoardType.SQUARE19):
            limit = 8 if board_type == BoardType.SQUARE8 else 19
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = position.x + dx, position.y + dy
                    if 0 <= nx < limit and 0 <= ny < limit:
                        neighbors.append(Position(x=nx, y=ny))

        elif board_type in (BoardType.HEX8, BoardType.HEXAGONAL):
            # HEX8: radius=4, HEXAGONAL: radius=12
            radius = 4 if board_type == BoardType.HEX8 else (board_size - 1)
            directions = [
                (1, 0, -1), (-1, 0, 1),
                (0, 1, -1), (0, -1, 1),
                (1, -1, 0), (-1, 1, 0)
            ]

            # Ensure z is calculated if missing
            px = position.x
            py = position.y
            pz = position.z if position.z is not None else -px - py

            for dx, dy, dz in directions:
                nx, ny, nz = px + dx, py + dy, pz + dz
                if (abs(nx) <= radius and
                        abs(ny) <= radius and
                        abs(nz) <= radius):
                    neighbors.append(Position(x=nx, y=ny, z=nz))

        return neighbors

    @staticmethod
    def get_line_of_sight_directions(
        board_type: BoardType
    ) -> List[Tuple[int, int, int]]:
        """Get all valid line-of-sight directions for the board type"""
        if board_type in (BoardType.SQUARE8, BoardType.SQUARE19):
            directions = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    directions.append((dx, dy, 0))
            return directions
        elif board_type in (BoardType.HEX8, BoardType.HEXAGONAL):
            return [
                (1, 0, -1), (-1, 0, 1),
                (0, 1, -1), (0, -1, 1),
                (1, -1, 0), (-1, 1, 0)
            ]
        return []

    @staticmethod
    def get_center_positions(
        board_type: BoardType,
        board_size: int
    ) -> Set[str]:
        """Get the set of position keys representing the center of the board"""
        center = set()

        if board_type == BoardType.SQUARE8:
            # Center 2x2 of 8x8 board
            for x in [3, 4]:
                for y in [3, 4]:
                    center.add(f"{x},{y}")

        elif board_type == BoardType.SQUARE19:
            # Center 3x3 of 19x19 board
            for x in [8, 9, 10]:
                for y in [8, 9, 10]:
                    center.add(f"{x},{y}")

        elif board_type == BoardType.HEX8:
            # Center hexagon for hex8 (distance 0-1 from origin, smaller board)
            for x in range(-1, 2):
                for y in range(-1, 2):
                    z = -x - y
                    if abs(x) <= 1 and abs(y) <= 1 and abs(z) <= 1:
                        center.add(f"{x},{y},{z}")

        elif board_type == BoardType.HEXAGONAL:
            # Center hexagon (distance 0-2 from origin)
            for x in range(-2, 3):
                for y in range(-2, 3):
                    z = -x - y
                    if abs(x) <= 2 and abs(y) <= 2 and abs(z) <= 2:
                        center.add(f"{x},{y},{z}")

        return center

    @staticmethod
    def is_within_bounds(
        pos: Position,
        board_type: BoardType,
        board_size: int
    ) -> bool:
        """Check if a position is within board bounds"""
        if board_type in (BoardType.SQUARE8, BoardType.SQUARE19):
            limit = 8 if board_type == BoardType.SQUARE8 else 19
            return 0 <= pos.x < limit and 0 <= pos.y < limit
        elif board_type in (BoardType.HEX8, BoardType.HEXAGONAL):
            # HEX8: radius=4, HEXAGONAL: radius=12
            radius = 4 if board_type == BoardType.HEX8 else (board_size - 1)
            z = pos.z if pos.z is not None else -pos.x - pos.y
            return (abs(pos.x) <= radius and
                    abs(pos.y) <= radius and
                    abs(z) <= radius)
        return False

    @staticmethod
    def calculate_distance(
        board_type: BoardType,
        from_pos: Position,
        to_pos: Position,
    ) -> int:
        """Calculate distance between two positions based on board type.

        Mirrors src/shared/engine/core.ts:calculateDistance and
        GameEngine._calculate_distance.
        """
        if board_type in (BoardType.HEX8, BoardType.HEXAGONAL):
            dx = to_pos.x - from_pos.x
            dy = to_pos.y - from_pos.y
            dz = (to_pos.z or 0) - (from_pos.z or 0)
            return int((abs(dx) + abs(dy) + abs(dz)) / 2)

        dx = abs(to_pos.x - from_pos.x)
        dy = abs(to_pos.y - from_pos.y)
        return max(dx, dy)

    @staticmethod
    def get_path_positions(
        from_pos: Position,
        to_pos: Position,
    ) -> List[Position]:
        """Get all positions along a straight-line path between two
        positions, inclusive of endpoints.

        Mirrors src/shared/engine/core.ts:getPathPositions and
        GameEngine._get_path_positions.
        """
        path = [from_pos]

        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y
        dz_from = from_pos.z or 0
        dz_to = to_pos.z or 0
        dz = dz_to - dz_from

        steps = max(abs(dx), abs(dy), abs(dz))
        if steps == 0:
            return path

        step_x = dx / steps
        step_y = dy / steps
        step_z = dz / steps

        for i in range(1, steps + 1):
            x = int(round(from_pos.x + step_x * i))
            y = int(round(from_pos.y + step_y * i))
            pos_kwargs = {"x": x, "y": y}
            if from_pos.z is not None or to_pos.z is not None:
                z = int(round(dz_from + step_z * i))
                pos_kwargs["z"] = z
            path.append(Position(**pos_kwargs))  # type: ignore[arg-type]

        return path

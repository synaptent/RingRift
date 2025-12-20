"""Fast geometry operations for AI evaluation.

Pre-computes adjacency tables and uses raw tuples instead of Position objects
to avoid Pydantic overhead in hot paths. This provides significant speedup
for heuristic evaluation which calls geometry functions millions of times.

Performance: ~3-5x faster than Position-based geometry for evaluation loops.

Usage:
    from app.ai.fast_geometry import FastGeometry

    # Get singleton instance (pre-computed tables)
    geo = FastGeometry.get_instance()

    # Fast adjacency lookup - returns list of (x, y) tuples
    neighbors = geo.get_adjacent_keys("3,4", BoardType.SQUARE8)

    # Fast bounds check
    is_valid = geo.is_within_bounds(5, 6, BoardType.SQUARE8)

    # Get all board keys (cached)
    all_keys = geo.get_all_board_keys(BoardType.SQUARE8)
"""

from __future__ import annotations

from functools import lru_cache

from ..models import BoardType

# Type aliases for clarity
CoordTuple = tuple[int, int, int | None]  # (x, y, z) or (x, y, None)
KeyString = str  # "x,y" or "x,y,z"


class FastGeometry:
    """Pre-computed geometry tables for fast AI evaluation.

    This class pre-computes adjacency relationships and board positions
    at initialization time, then provides O(1) lookups during evaluation.
    """

    _instance: FastGeometry | None = None

    # Square board directions (8 neighbors)
    SQUARE_DIRECTIONS: list[tuple[int, int]] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    # Line-of-sight directions for square boards
    SQUARE_LOS_DIRECTIONS: list[tuple[int, int, int]] = [(dx, dy, 0) for dx, dy in SQUARE_DIRECTIONS]

    # Hexagonal directions (6 neighbors)
    HEX_DIRECTIONS: list[tuple[int, int, int]] = [
        (1, 0, -1),
        (-1, 0, 1),
        (0, 1, -1),
        (0, -1, 1),
        (1, -1, 0),
        (-1, 1, 0),
    ]

    # Canonical hex board configuration (RR-CANON-R001 / BOARD_CONFIGS).
    # size=13 implies cube radius=12 and totalSpaces=469.
    HEX_SIZE: int = 13
    HEX_RADIUS: int = HEX_SIZE - 1

    def __init__(self):
        """Initialize pre-computed geometry tables."""
        # Adjacency tables: key -> list of adjacent keys
        self._adjacency_square8: dict[str, list[str]] = {}
        self._adjacency_square19: dict[str, list[str]] = {}
        self._adjacency_hex: dict[str, list[str]] = {}

        # All board keys (cached)
        self._all_keys_square8: list[str] = []
        self._all_keys_square19: list[str] = []
        self._all_keys_hex: list[str] = []

        # Center positions (cached)
        self._center_square8: frozenset[str] = frozenset()
        self._center_square19: frozenset[str] = frozenset()
        self._center_hex: frozenset[str] = frozenset()

        # Key -> coordinates tuple (avoids string parsing)
        self._coords_square8: dict[str, tuple[int, int, None]] = {}
        self._coords_square19: dict[str, tuple[int, int, None]] = {}
        self._coords_hex: dict[str, tuple[int, int, int]] = {}

        # Pre-computed offset tables: (key, direction_index, distance) -> result_key
        # For distances 1, 2, 3 (most common in line evaluation)
        self._offset_square8: dict[tuple[str, int, int], str | None] = {}
        self._offset_square19: dict[tuple[str, int, int], str | None] = {}
        self._offset_hex: dict[tuple[str, int, int], str | None] = {}

        # Pre-compute everything
        self._build_square_tables(8, self._adjacency_square8, self._all_keys_square8)
        self._build_square_tables(19, self._adjacency_square19, self._all_keys_square19)
        self._build_hex_tables(self.HEX_SIZE, self._adjacency_hex, self._all_keys_hex)
        self._build_center_tables()
        self._build_coords_tables()
        self._build_offset_tables()

    @classmethod
    def get_instance(cls) -> FastGeometry:
        """Get singleton instance of FastGeometry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _build_square_tables(
        self,
        size: int,
        adjacency: dict[str, list[str]],
        all_keys: list[str],
    ) -> None:
        """Pre-compute adjacency table for square board."""
        for x in range(size):
            for y in range(size):
                key = f"{x},{y}"
                all_keys.append(key)

                neighbors: list[str] = []
                for dx, dy in self.SQUARE_DIRECTIONS:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        neighbors.append(f"{nx},{ny}")
                adjacency[key] = neighbors

    def _build_hex_tables(
        self,
        size: int,
        adjacency: dict[str, list[str]],
        all_keys: list[str],
    ) -> None:
        """Pre-compute adjacency table for hexagonal board."""
        radius = size - 1  # canonical: size=13 -> radius=12

        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                z = -x - y
                if abs(x) <= radius and abs(y) <= radius and abs(z) <= radius:
                    key = f"{x},{y},{z}"
                    all_keys.append(key)

                    neighbors: list[str] = []
                    for dx, dy, dz in self.HEX_DIRECTIONS:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if abs(nx) <= radius and abs(ny) <= radius and abs(nz) <= radius:
                            neighbors.append(f"{nx},{ny},{nz}")
                    adjacency[key] = neighbors

    def _build_center_tables(self) -> None:
        """Pre-compute center position sets."""
        # Square8: center 2x2
        self._center_square8 = frozenset(f"{x},{y}" for x in [3, 4] for y in [3, 4])

        # Square19: center 3x3
        self._center_square19 = frozenset(f"{x},{y}" for x in [8, 9, 10] for y in [8, 9, 10])

        # Hex: center hexagon (distance 0-2 from origin)
        hex_center: set[str] = set()
        for x in range(-2, 3):
            for y in range(-2, 3):
                z = -x - y
                if abs(x) <= 2 and abs(y) <= 2 and abs(z) <= 2:
                    hex_center.add(f"{x},{y},{z}")
        self._center_hex = frozenset(hex_center)

    def _build_coords_tables(self) -> None:
        """Pre-compute key -> coordinates mappings to avoid string parsing."""
        # Square8
        for key in self._all_keys_square8:
            parts = key.split(",")
            self._coords_square8[key] = (int(parts[0]), int(parts[1]), None)

        # Square19
        for key in self._all_keys_square19:
            parts = key.split(",")
            self._coords_square19[key] = (int(parts[0]), int(parts[1]), None)

        # Hex
        for key in self._all_keys_hex:
            parts = key.split(",")
            self._coords_hex[key] = (int(parts[0]), int(parts[1]), int(parts[2]))

    def _build_offset_tables(self) -> None:
        """Pre-compute offset results for all (key, direction, distance) combinations.

        This eliminates the need for coordinate parsing and bounds checking
        at runtime for the most common offset operations (distances 1-3).
        """
        # Square8
        directions_sq = self.SQUARE_LOS_DIRECTIONS
        for key in self._all_keys_square8:
            x, y, _ = self._coords_square8[key]
            for dir_idx, (dx, dy, _) in enumerate(directions_sq):
                for dist in (1, 2, 3):
                    nx, ny = x + dx * dist, y + dy * dist
                    if 0 <= nx < 8 and 0 <= ny < 8:
                        self._offset_square8[(key, dir_idx, dist)] = f"{nx},{ny}"
                    else:
                        self._offset_square8[(key, dir_idx, dist)] = None

        # Square19
        for key in self._all_keys_square19:
            x, y, _ = self._coords_square19[key]
            for dir_idx, (dx, dy, _) in enumerate(directions_sq):
                for dist in (1, 2, 3):
                    nx, ny = x + dx * dist, y + dy * dist
                    if 0 <= nx < 19 and 0 <= ny < 19:
                        self._offset_square19[(key, dir_idx, dist)] = f"{nx},{ny}"
                    else:
                        self._offset_square19[(key, dir_idx, dist)] = None

        # Hex
        directions_hex = self.HEX_DIRECTIONS
        for key in self._all_keys_hex:
            x, y, z = self._coords_hex[key]
            for dir_idx, (dx, dy, dz) in enumerate(directions_hex):
                for dist in (1, 2, 3):
                    nx = x + dx * dist
                    ny = y + dy * dist
                    nz = z + dz * dist
                    if (
                        abs(nx) <= self.HEX_RADIUS
                        and abs(ny) <= self.HEX_RADIUS
                        and abs(nz) <= self.HEX_RADIUS
                    ):
                        self._offset_hex[(key, dir_idx, dist)] = f"{nx},{ny},{nz}"
                    else:
                        self._offset_hex[(key, dir_idx, dist)] = None

    def get_adjacent_keys(self, key: str, board_type) -> list[str]:
        """Get pre-computed adjacent position keys.

        Args:
            key: Position key (e.g., "3,4" or "1,2,-3")
            board_type: Board type (BoardType enum or string)

        Returns:
            List of adjacent position keys (empty list if key not found)
        """
        # Handle both BoardType enum and string for flexibility
        bt = str(board_type).lower().replace("boardtype.", "")
        if bt == "square8":
            return self._adjacency_square8.get(key, [])
        elif bt == "square19":
            return self._adjacency_square19.get(key, [])
        elif bt in ("hexagonal", "hex"):
            return self._adjacency_hex.get(key, [])
        return []

    def get_all_board_keys(self, board_type: BoardType) -> list[str]:
        """Get all valid position keys for a board type.

        Args:
            board_type: Board type

        Returns:
            List of all position keys
        """
        if board_type == BoardType.SQUARE8:
            return self._all_keys_square8
        elif board_type == BoardType.SQUARE19:
            return self._all_keys_square19
        elif board_type == BoardType.HEXAGONAL:
            return self._all_keys_hex
        return []

    def get_center_positions(self, board_type: BoardType) -> frozenset[str]:
        """Get pre-computed center position keys.

        Args:
            board_type: Board type

        Returns:
            Frozen set of center position keys
        """
        if board_type == BoardType.SQUARE8:
            return self._center_square8
        elif board_type == BoardType.SQUARE19:
            return self._center_square19
        elif board_type == BoardType.HEXAGONAL:
            return self._center_hex
        return frozenset()

    def is_within_bounds_tuple(
        self,
        x: int,
        y: int,
        z: int | None,
        board_type: BoardType,
    ) -> bool:
        """Fast bounds check using raw coordinates.

        Args:
            x, y, z: Coordinates (z can be None for square boards)
            board_type: Board type

        Returns:
            True if within bounds
        """
        if board_type == BoardType.SQUARE8:
            return 0 <= x < 8 and 0 <= y < 8
        elif board_type == BoardType.SQUARE19:
            return 0 <= x < 19 and 0 <= y < 19
        elif board_type == BoardType.HEXAGONAL:
            if z is None:
                z = -x - y
            return (
                abs(x) <= self.HEX_RADIUS
                and abs(y) <= self.HEX_RADIUS
                and abs(z) <= self.HEX_RADIUS
            )
        return False

    def key_to_coords(self, key: str) -> CoordTuple:
        """Parse a position key to coordinates.

        Args:
            key: Position key (e.g., "3,4" or "1,2,-3")

        Returns:
            Tuple of (x, y, z) where z is None for 2D keys
        """
        parts = key.split(",")
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]), None)
        else:
            return (int(parts[0]), int(parts[1]), int(parts[2]))

    def coords_to_key(self, x: int, y: int, z: int | None = None) -> str:
        """Convert coordinates to position key.

        Args:
            x, y, z: Coordinates

        Returns:
            Position key string
        """
        if z is None:
            return f"{x},{y}"
        return f"{x},{y},{z}"

    def get_los_directions(self, board_type: BoardType) -> list[tuple[int, int, int]]:
        """Get line-of-sight directions for a board type.

        Args:
            board_type: Board type

        Returns:
            List of (dx, dy, dz) direction tuples
        """
        if board_type in (BoardType.SQUARE8, BoardType.SQUARE19):
            return self.SQUARE_LOS_DIRECTIONS
        elif board_type == BoardType.HEXAGONAL:
            return self.HEX_DIRECTIONS
        return []

    def get_coords(self, key: str, board_type: BoardType) -> CoordTuple:
        """Get pre-computed coordinates for a key (avoids string parsing).

        Args:
            key: Position key
            board_type: Board type

        Returns:
            (x, y, z) tuple where z is None for square boards
        """
        if board_type == BoardType.SQUARE8:
            return self._coords_square8.get(key, (0, 0, None))
        elif board_type == BoardType.SQUARE19:
            return self._coords_square19.get(key, (0, 0, None))
        elif board_type == BoardType.HEXAGONAL:
            return self._coords_hex.get(key, (0, 0, 0))
        return (0, 0, None)

    def offset_key_fast(
        self,
        key: str,
        direction_index: int,
        distance: int,
        board_type: BoardType,
    ) -> str | None:
        """Ultra-fast offset lookup using pre-computed tables.

        This is O(1) with a single dict lookup - no coordinate parsing,
        no arithmetic, no bounds checking at runtime.

        Args:
            key: Starting position key
            direction_index: Index into the board's direction list (0-7 for square, 0-5 for hex)
            distance: Distance to offset (1, 2, or 3)
            board_type: Board type

        Returns:
            Result key or None if out of bounds
        """
        lookup_key = (key, direction_index, distance)
        if board_type == BoardType.SQUARE8:
            return self._offset_square8.get(lookup_key)
        elif board_type == BoardType.SQUARE19:
            return self._offset_square19.get(lookup_key)
        elif board_type == BoardType.HEXAGONAL:
            return self._offset_hex.get(lookup_key)
        return None

    def offset_key(
        self,
        key: str,
        direction: tuple[int, int, int],
        distance: int,
        board_type: BoardType,
    ) -> str | None:
        """Compute a position key offset by a direction, without bounds check.

        This is a fast alternative to BoardManager._add_direction that
        avoids creating Position objects.

        Args:
            key: Starting position key
            direction: (dx, dy, dz) direction tuple
            distance: Number of steps in direction
            board_type: Board type for bounds checking

        Returns:
            New position key, or None if out of bounds
        """
        x, y, z = self.key_to_coords(key)
        dx, dy, dz = direction

        nx = x + dx * distance
        ny = y + dy * distance
        nz = z + dz * distance if z is not None else None

        if not self.is_within_bounds_tuple(nx, ny, nz, board_type):
            return None

        return self.coords_to_key(nx, ny, nz)

    def offset_coords(
        self,
        x: int,
        y: int,
        z: int | None,
        direction: tuple[int, int, int],
        distance: int,
    ) -> tuple[int, int, int | None]:
        """Compute coordinates offset by a direction.

        Args:
            x, y, z: Starting coordinates
            direction: (dx, dy, dz) direction tuple
            distance: Number of steps in direction

        Returns:
            New (x, y, z) coordinates (not bounds-checked)
        """
        dx, dy, dz = direction
        nx = x + dx * distance
        ny = y + dy * distance
        nz = z + dz * distance if z is not None else None
        return (nx, ny, nz)

    @lru_cache(maxsize=1024)
    def get_visible_keys_from(
        self,
        key: str,
        board_type: BoardType,
    ) -> list[list[str]]:
        """Get all position keys visible from a position in each direction.

        This returns a list of rays, where each ray is a list of keys
        extending from the position in one direction until the board edge.

        Cached for performance.

        Args:
            key: Starting position key
            board_type: Board type

        Returns:
            List of rays, each ray is a list of position keys
        """
        x, y, z = self.key_to_coords(key)
        directions = self.get_los_directions(board_type)

        rays: list[list[str]] = []
        for dx, dy, dz in directions:
            ray: list[str] = []
            curr_x, curr_y, curr_z = x, y, z

            while True:
                curr_x += dx
                curr_y += dy
                if curr_z is not None:
                    curr_z += dz

                if not self.is_within_bounds_tuple(curr_x, curr_y, curr_z, board_type):
                    break

                ray.append(self.coords_to_key(curr_x, curr_y, curr_z))

            if ray:
                rays.append(ray)

        return rays


# Module-level convenience function
def get_fast_geometry() -> FastGeometry:
    """Get the singleton FastGeometry instance."""
    return FastGeometry.get_instance()

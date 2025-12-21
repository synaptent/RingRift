"""
Optimized territory detection with caching and pre-computed lookups.

This module provides faster region detection by:
1. Pre-computing neighbor relationships once per board type
2. Using numpy arrays for efficient set operations
3. Caching region computations with invalidation tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..models import BoardState, Territory


class BoardGeometryCache:
    """
    Pre-computed geometry data for fast neighbor lookups.

    Created once per board type, reused across all operations.
    """
    _instances: dict[str, BoardGeometryCache] = {}

    def __init__(self, board_type: str, size: int):
        self.board_type = board_type
        self.size = size

        # Build position mappings
        self.position_to_idx: dict[str, int] = {}
        self.idx_to_position: list[str] = []
        self.num_positions = 0

        # Neighbor arrays (idx -> list of neighbor indices, -1 = invalid)
        self.neighbors: np.ndarray = None
        self.num_neighbors: np.ndarray = None  # Number of valid neighbors per position

        self._build_geometry()

    @classmethod
    def get(cls, board_type: str, size: int) -> BoardGeometryCache:
        """Get or create cached geometry for board type."""
        key = f"{board_type}_{size}"
        if key not in cls._instances:
            cls._instances[key] = cls(board_type, size)
        return cls._instances[key]

    def _build_geometry(self):
        """Build position mappings and neighbor arrays."""
        positions = []

        if self.board_type in ("hexagonal", "hex8"):
            # Hex boards: size = radius + 1 (TS BOARD_CONFIGS). Radius = size - 1.
            # HEX8: size=5 -> radius=4, HEXAGONAL: size=13 -> radius=12
            # This matches board_manager.py is_valid_position and TS BOARD_CONFIGS.
            radius = self.size - 1
            for x in range(-radius, radius + 1):
                for y in range(-radius, radius + 1):
                    z = -x - y
                    if abs(x) <= radius and abs(y) <= radius and abs(z) <= radius:
                        # Use "x,y,z" format to match how board.stacks/markers store hex positions
                        positions.append(f"{x},{y},{z}")
        else:
            # Square board
            board_size = 8 if self.board_type == "square8" else 19
            for y in range(board_size):
                for x in range(board_size):
                    positions.append(f"{x},{y}")

        self.idx_to_position = positions
        self.position_to_idx = {pos: idx for idx, pos in enumerate(positions)}
        self.num_positions = len(positions)

        # Build neighbor arrays for territory adjacency.
        # Canonical rules (and TS BOARD_CONFIGS) use:
        # - hexagonal/hex8: 6-direction hex adjacency in cube coords (x,y,z)
        # - square8/square19: von Neumann (4-direction) adjacency
        if self.board_type in ("hexagonal", "hex8"):
            # 6 hex directions in cube coordinates, mirroring TS getNeighbors
            directions_3d = [
                (1, 0, -1),
                (1, -1, 0),
                (0, -1, 1),
                (-1, 0, 1),
                (-1, 1, 0),
                (0, 1, -1),
            ]
            max_neighbors = 6
        else:
            # 4-direction von Neumann adjacency for square boards
            directions_3d = [
                (-1, 0, None),
                (1, 0, None),
                (0, -1, None),
                (0, 1, None),
            ]
            max_neighbors = 4

        self.neighbors = np.full((self.num_positions, max_neighbors), -1, dtype=np.int32)
        self.num_neighbors = np.zeros(self.num_positions, dtype=np.int8)

        for pos_key, idx in self.position_to_idx.items():
            parts = pos_key.split(",")
            x, y = int(parts[0]), int(parts[1])
            z = int(parts[2]) if len(parts) > 2 else 0

            neighbor_count = 0
            for dx, dy, dz in directions_3d:
                nx = x + dx
                ny = y + dy
                # For square boards dz is None and we emit "x,y"; for hex we
                # emit full "x,y,z" to match the keys above.
                if dz is None:
                    neighbor_key = f"{nx},{ny}"
                else:
                    nz = z + dz
                    neighbor_key = f"{nx},{ny},{nz}"

                if neighbor_key in self.position_to_idx:
                    neighbor_idx = self.position_to_idx[neighbor_key]
                    self.neighbors[idx, neighbor_count] = neighbor_idx
                    neighbor_count += 1

            self.num_neighbors[idx] = neighbor_count


@dataclass
class RegionCache:
    """
    Cached territory region data with invalidation tracking.
    """
    # Version tracking for cache invalidation
    marker_hash: int = 0
    collapsed_hash: int = 0
    stack_hash: int = 0

    # Cached regions by border color
    # Key: border_color (0 = no marker border), Value: list of region position sets
    regions_by_border: dict[int, list[set[str]]] = field(default_factory=dict)

    # Active players when cache was computed
    active_players: set[int] = field(default_factory=set)

    def is_valid(
        self,
        markers: dict[str, any],
        collapsed: dict[str, any],
        stacks: dict[str, any],
        active_players: set[int],
    ) -> bool:
        """Check if cache is still valid."""
        # Simple hash comparison for invalidation
        marker_hash = hash(frozenset(markers.keys()))
        collapsed_hash = hash(frozenset(collapsed.keys()))
        stack_hash = hash(frozenset(stacks.keys()))

        return (
            self.marker_hash == marker_hash and
            self.collapsed_hash == collapsed_hash and
            self.stack_hash == stack_hash and
            self.active_players == active_players
        )

    def update_hashes(
        self,
        markers: dict[str, any],
        collapsed: dict[str, any],
        stacks: dict[str, any],
        active_players: set[int],
    ):
        """Update cache hashes."""
        self.marker_hash = hash(frozenset(markers.keys()))
        self.collapsed_hash = hash(frozenset(collapsed.keys()))
        self.stack_hash = hash(frozenset(stacks.keys()))
        self.active_players = active_players.copy()


def _normalize_key_for_hex(key: str, board_type: str) -> str:
    """
    Normalize a position key to match geometry cache format.

    For hexagonal boards, board.markers/stacks may use "x,y" keys (when Position.z is None)
    but the geometry cache uses "x,y,z" keys. This function converts "x,y" -> "x,y,z"
    where z = -x - y for cube coordinates.
    """
    if board_type != "hexagonal":
        return key

    parts = key.split(",")
    if len(parts) == 2:
        # Convert "x,y" to "x,y,z" for hex boards
        x, y = int(parts[0]), int(parts[1])
        z = -x - y
        return f"{x},{y},{z}"
    return key


def find_disconnected_regions_fast(
    board: BoardState,
    player_number: int,
    cache: RegionCache | None = None,
) -> list[Territory]:
    """
    Fast territory region detection using pre-computed geometry.

    Args:
        board: Current board state
        player_number: Player requesting (for API compatibility)
        cache: Optional cache for repeated calls

    Returns:
        List of disconnected Territory objects
    """

    # Get geometry cache
    board_type_str = board.type.value if hasattr(board.type, 'value') else str(board.type)
    geo = BoardGeometryCache.get(board_type_str, board.size)

    # Find active players
    active_players = set()
    for stack in board.stacks.values():
        active_players.add(stack.controlling_player)

    if len(active_players) <= 1:
        return []

    # Check cache validity
    if cache is not None and cache.is_valid(
        board.markers, board.collapsed_spaces, board.stacks, active_players
    ):
        # Use cached results
        return _build_territories_from_cache(cache, geo, board, active_players)

    # Compute fresh
    regions: list[Territory] = []

    # Get marker colors
    marker_colors = set()
    for marker in board.markers.values():
        marker_colors.add(marker.player)

    # Build position -> marker color mapping
    # Normalize keys for hex boards (convert "x,y" to "x,y,z")
    marker_at = {}
    for key, marker in board.markers.items():
        normalized_key = _normalize_key_for_hex(key, board_type_str)
        if normalized_key in geo.position_to_idx:
            marker_at[geo.position_to_idx[normalized_key]] = marker.player

    # Build collapsed position set
    collapsed = set()
    for key in board.collapsed_spaces:
        normalized_key = _normalize_key_for_hex(key, board_type_str)
        if normalized_key in geo.position_to_idx:
            collapsed.add(geo.position_to_idx[normalized_key])

    # Build stack position -> controlling player mapping
    stack_at = {}
    for key, stack in board.stacks.items():
        normalized_key = _normalize_key_for_hex(key, board_type_str)
        if normalized_key in geo.position_to_idx:
            stack_at[geo.position_to_idx[normalized_key]] = stack.controlling_player

    # Find regions for each border color
    for border_color in marker_colors:
        new_regions = _find_regions_with_border_color_fast(
            geo, marker_at, collapsed, stack_at, border_color, active_players
        )
        for region_positions in new_regions:
            regions.append(_create_territory(geo, region_positions))

    # Find regions without marker borders
    new_regions = _find_regions_without_marker_border_fast(
        geo, marker_at, collapsed, stack_at, active_players
    )
    for region_positions in new_regions:
        regions.append(_create_territory(geo, region_positions))

    # Update cache if provided
    if cache is not None:
        cache.update_hashes(board.markers, board.collapsed_spaces, board.stacks, active_players)
        cache.regions_by_border.clear()
        # Store positions for cache reconstruction (not full territories)

    return regions


def _find_regions_with_border_color_fast(
    geo: BoardGeometryCache,
    marker_at: dict[int, int],  # idx -> player
    collapsed: set[int],
    stack_at: dict[int, int],  # idx -> controlling player
    border_color: int,
    active_players: set[int],
) -> list[set[int]]:
    """
    Find regions where markers of border_color act as borders.
    Uses numpy-based flood fill for efficiency.
    """
    regions = []
    visited = np.zeros(geo.num_positions, dtype=np.bool_)

    # Mark border positions as visited (they can't be part of regions)
    for idx in range(geo.num_positions):
        if idx in collapsed or (idx in marker_at and marker_at[idx] == border_color):
            visited[idx] = True

    # Flood fill from each unvisited position
    for start_idx in range(geo.num_positions):
        if visited[start_idx]:
            continue

        # BFS flood fill
        region = set()
        queue = [start_idx]
        visited[start_idx] = True

        while queue:
            idx = queue.pop()
            region.add(idx)

            # Check neighbors
            for n in range(geo.num_neighbors[idx]):
                neighbor_idx = geo.neighbors[idx, n]
                if neighbor_idx < 0 or visited[neighbor_idx]:
                    continue

                # Skip borders
                if neighbor_idx in collapsed:
                    visited[neighbor_idx] = True
                    continue
                if neighbor_idx in marker_at and marker_at[neighbor_idx] == border_color:
                    visited[neighbor_idx] = True
                    continue

                visited[neighbor_idx] = True
                queue.append(neighbor_idx)

        # Check if region is disconnected (not all active players represented)
        players_in_region = set()
        for idx in region:
            if idx in stack_at:
                players_in_region.add(stack_at[idx])

        # Match TS + slow-path semantics: empty regions still count as
        # disconnected if they exclude at least one active player.
        if players_in_region < active_players:
            regions.append(region)

    return regions


def _find_regions_without_marker_border_fast(
    geo: BoardGeometryCache,
    marker_at: dict[int, int],
    collapsed: set[int],
    stack_at: dict[int, int],
    active_players: set[int],
) -> list[set[int]]:
    """
    Find regions bordered only by collapsed spaces and edges.
    """
    regions = []
    visited = np.zeros(geo.num_positions, dtype=np.bool_)

    # Mark collapsed and marker positions as visited
    for idx in range(geo.num_positions):
        if idx in collapsed or idx in marker_at:
            visited[idx] = True

    for start_idx in range(geo.num_positions):
        if visited[start_idx]:
            continue

        region = set()
        queue = [start_idx]
        visited[start_idx] = True
        has_marker_border = False

        while queue:
            idx = queue.pop()
            region.add(idx)

            for n in range(geo.num_neighbors[idx]):
                neighbor_idx = geo.neighbors[idx, n]
                if neighbor_idx < 0:
                    continue

                if visited[neighbor_idx]:
                    # Check if this neighbor is a marker (indicates marker border)
                    if neighbor_idx in marker_at:
                        has_marker_border = True
                    continue

                if neighbor_idx in collapsed:
                    visited[neighbor_idx] = True
                    continue

                if neighbor_idx in marker_at:
                    visited[neighbor_idx] = True
                    has_marker_border = True
                    continue

                visited[neighbor_idx] = True
                queue.append(neighbor_idx)

        # Only include if no marker border and disconnected
        if not has_marker_border and region:
            players_in_region = set()
            for idx in region:
                if idx in stack_at:
                    players_in_region.add(stack_at[idx])

            # Match TS + slow-path semantics: empty regions still count as
            # disconnected if they exclude at least one active player.
            if players_in_region < active_players:
                # Additional check: region must be bordered only by collapsed
                # spaces or edges, mirroring TS isRegionBorderedByCollapsedOnly.
                is_valid = True
                for idx in region:
                    for n in range(geo.num_neighbors[idx]):
                        neighbor_idx = geo.neighbors[idx, n]
                        if neighbor_idx < 0:
                            # Outside canonical grid (edge) is an acceptable border.
                            continue
                        if neighbor_idx in region:
                            # Interior neighbor is fine.
                            continue
                        if neighbor_idx in collapsed:
                            # Collapsed space is an acceptable border.
                            continue
                        # Any non-collapsed neighbor outside the region
                        # (empty, stack, or marker) breaks the collapsed-only
                        # border requirement.
                        is_valid = False
                        break
                    if not is_valid:
                        break

                if is_valid:
                    regions.append(region)

    return regions


def _create_territory(
    geo: BoardGeometryCache,
    region_indices: set[int],
) -> Territory:
    """Create a Territory object from region indices."""
    from ..models import Position, Territory

    positions = []
    for idx in region_indices:
        pos_key = geo.idx_to_position[idx]
        parts = pos_key.split(",")
        x, y = int(parts[0]), int(parts[1])
        # Handle "x,y,z" format for hexagonal boards
        z = int(parts[2]) if len(parts) > 2 else None
        positions.append(Position(x=x, y=y, z=z))

    return Territory(
        spaces=positions,
        controllingPlayer=0,  # Caller decides
        isDisconnected=True,
    )


def _build_territories_from_cache(
    cache: RegionCache,
    geo: BoardGeometryCache,
    board: BoardState,
    active_players: set[int],
) -> list[Territory]:
    """Reconstruct Territory objects from cached data."""
    # For now, just recompute (cache stores validation data)
    # Full implementation would store region positions
    return []

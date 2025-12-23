"""Board-level helpers for the RingRift AI service.

**SSoT (Single Source of Truth) Policy:**

The canonical rules defined in ``RULES_CANONICAL_SPEC.md`` are the ultimate
authority for RingRift game semantics. The TS shared engine
(``src/shared/engine/**``) is the primary executable derivation. This Python
module is a *host adapter* that must mirror the canonical rules. If this code
disagrees with the canonical rules or the validated TS engine behaviour, this
code must be updated—never the other way around.
"""
from __future__ import annotations

import os

from .models import (
    BoardState,
    BoardType,
    GameState,
    LineInfo,
    MarkerInfo,
    Position,
    ProgressSnapshot,
    RingStack,
    Territory,
)

__all__ = ["BoardManager"]

# Environment flag to use fast territory detection.
#
# Fast territory detection (app.ai.territory_cache.find_disconnected_regions_fast)
# is an optimisation layer over the canonical BoardManager implementation. It
# must exactly match the shared TS engine's territoryDetection helpers; any
# deviation is a *bug in the fast path*, not in the rules spec.
#
# The fast path has been validated for parity across all board types
# (square8, hex8, square19, hexagonal) including marker border detection on
# hexagonal boards. It's now enabled by default for performance gains.
#
# To disable and use the slow path:
#   export RINGRIFT_USE_FAST_TERRITORY=false
USE_FAST_TERRITORY = os.getenv('RINGRIFT_USE_FAST_TERRITORY', 'true').lower() == 'true'


def _get_position_keys_for_lookup(position: Position, board_type: BoardType) -> list[str]:
    """
    Get position keys to try when looking up stacks/markers.

    For hexagonal boards, data may be stored with either "x,y" or "x,y,z" keys.
    This helper returns both formats to ensure lookups succeed regardless of
    how the data was stored.

    Args:
        position: The position to look up
        board_type: The board type (affects key format for hex boards)

    Returns:
        List of keys to try, in order of preference
    """
    primary_key = position.to_key()
    keys = [primary_key]

    # For hex boards, also try the abbreviated "x,y" format if the primary
    # key includes z. Data may be stored in either format.
    if board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
        parts = primary_key.split(",")
        if len(parts) == 3:
            # Primary is "x,y,z", also try "x,y"
            short_key = f"{parts[0]},{parts[1]}"
            keys.append(short_key)
        elif len(parts) == 2 and position.z is None:
            # Primary is "x,y" without z, also compute and try "x,y,z"
            z = -position.x - position.y
            full_key = f"{position.x},{position.y},{z}"
            keys.append(full_key)

    return keys


class BoardManager:
    """Helper for board‑level operations in the AI service.

    **SSoT Policy:** This is a *host adapter*, NOT a rules SSoT. The canonical
    rules in ``RULES_CANONICAL_SPEC.md`` are the ultimate authority. If this
    code disagrees with the canonical rules or the validated TS engine
    behaviour, THIS CODE IS WRONG and must be fixed to match.

    This class mirrors the behaviour of the TypeScript ``BoardManager``
    used by the shared engine and backend host, providing:

    - stack/marker/collapsed‑space queries,
    - board hashing for trace/parity tooling, and
    - line/territory discovery helpers used by the Python rules engine.

    It is intentionally side‑effect‑free; callers pass in ``BoardState``
    instances and receive derived views or new value objects.
    """

    @staticmethod
    def get_stack(
        position: Position, board: BoardState
    ) -> RingStack | None:
        """Return the stack at ``position`` or ``None`` if empty."""
        # Try multiple key formats for hex boards (data may use "x,y" or "x,y,z")
        for pos_key in _get_position_keys_for_lookup(position, board.type):
            stack = board.stacks.get(pos_key)
            if stack is not None:
                return stack
        return None

    @staticmethod
    def is_valid_position(
        position: Position, board_type: BoardType, size: int
    ) -> bool:
        """Return True if ``position`` is on the board for ``board_type``.

        Note: Hex boards have two conventions for ``size``:
          - Convention A: size = radius + 1 (e.g., size=13 for radius=12)
          - Convention B: size = 2*radius + 1 (e.g., size=25 for radius=12)

        This function detects which convention is used based on size value
        and computes radius accordingly for backwards compatibility.
        """
        if board_type == BoardType.SQUARE8:
            return 0 <= position.x < 8 and 0 <= position.y < 8
        elif board_type == BoardType.SQUARE19:
            return 0 <= position.x < 19 and 0 <= position.y < 19
        elif board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
            # Compute z from x,y if not provided (hex constraint: x + y + z = 0)
            z = position.z if position.z is not None else -position.x - position.y

            # Detect convention based on size value:
            # - Convention A: size <= 13 means radius = size - 1
            #   (HEXAGONAL: size=13 -> radius=12, HEX8: size=5 -> radius=4)
            # - Convention B: size > 13 means radius = (size - 1) // 2
            #   (HEXAGONAL: size=25 -> radius=12, HEX8: size=9 -> radius=4)
            if size <= 13:
                # Convention A: size = radius + 1
                radius = size - 1
            else:
                # Convention B: size = 2*radius + 1
                radius = (size - 1) // 2

            return (abs(position.x) <= radius and
                    abs(position.y) <= radius and
                    abs(z) <= radius and
                    position.x + position.y + z == 0)
        return False

    @staticmethod
    def is_collapsed_space(position: Position, board: BoardState) -> bool:
        # Try multiple key formats for hex boards
        return any(pos_key in board.collapsed_spaces for pos_key in _get_position_keys_for_lookup(position, board.type))

    @staticmethod
    def get_marker(position: Position, board: BoardState) -> MarkerInfo | None:
        """Return the marker at ``position`` or ``None`` if no marker."""
        # Try multiple key formats for hex boards (data may use "x,y" or "x,y,z")
        for pos_key in _get_position_keys_for_lookup(position, board.type):
            marker = board.markers.get(pos_key)
            if marker is not None:
                return marker
        return None

    @staticmethod
    def get_player_stacks(
        board: BoardState, player_number: int
    ) -> list[RingStack]:
        return [
            stack for stack in board.stacks.values()
            if stack.controlling_player == player_number
        ]

    @staticmethod
    def hash_game_state(state: GameState) -> str:
        """
        Canonical hash of a GameState used by tests and diagnostic tooling to
        detect state changes and compare backend/sandbox traces.
        Matches TypeScript implementation in src/shared/engine/core.ts

        Delegates to app.rules.core.hash_game_state for consistent cross-engine
        parity with the TypeScript fingerprintGameState function.
        """
        from app.rules.core import hash_game_state as core_hash_game_state

        return core_hash_game_state(state)

    @staticmethod
    def find_all_lines(board: BoardState, num_players: int = 3) -> list[LineInfo]:
        """
        Find all marker lines on the board.

        Line length thresholds per RR-CANON-R120:
        - square8 2-player: 4
        - square8 3-4 player: 3
        - square19 / hexagonal: 4 (all player counts)

        Mirrors src/shared/engine/lineDetection.findAllLines and the server
        BoardManager.findAllLines implementation:
        - Lines are formed by MARKERS, not stacks.
        - Collapsed spaces and stacks act as hard blockers.
        - Uses canonical line directions per board type to avoid duplicates.
        """
        from app.rules.core import get_effective_line_length

        lines: list[LineInfo] = []
        processed_keys = set()

        # Determine line length based on board type AND player count.
        # Canonical rules (RR-CANON-R120) use player-count-aware thresholds.
        min_length = get_effective_line_length(board.type, num_players)

        directions = BoardManager._get_line_directions(board.type)

        # Iterate through all markers
        for _pos_key, marker in board.markers.items():
            start_pos = marker.position
            player = marker.player

            # Treat stacks and collapsed spaces as hard blockers that fully
            # remove this cell from line consideration.
            if BoardManager.is_collapsed_space(start_pos, board):
                continue
            if BoardManager.get_stack(start_pos, board):
                continue

            for direction in directions:
                current_line_positions = BoardManager._find_line_in_direction(
                    start_pos,
                    direction,
                    player,
                    board,
                )

                if len(current_line_positions) < min_length:
                    continue

                line_key = "|".join(
                    sorted(pos.to_key() for pos in current_line_positions),
                )
                if line_key in processed_keys:
                    continue

                processed_keys.add(line_key)
                lines.append(
                    LineInfo(
                        positions=current_line_positions,
                        player=player,
                        length=len(current_line_positions),
                        direction=Position(
                            x=direction[0],
                            y=direction[1],
                            z=direction[2],
                        ),
                    )
                )

        return lines

    @staticmethod
    def find_disconnected_regions(
        board: BoardState, player_number: int
    ) -> list[Territory]:
        """
        Python analogue of the TS BoardManager.findDisconnectedRegions.

        A region is considered disconnected when:

        1. Physical disconnection:
           It is separated from the rest of the board by a border composed of
           collapsed spaces, board edges, or markers belonging to a *single*
           border color (or no markers at all, in which case only collapsed
           spaces and edges form the border).

        2. Representation:
           The set of players with stacks inside the region is a proper subset
           of the active players on the board (at least one active player is
           not represented inside the region).

        The returned Territory objects have controlling_player=0, matching the
        TS implementation; the moving player decides how to process them.
        The `player_number` argument is kept for API compatibility but is not
        used in the discovery algorithm.
        """
        # Use fast path with pre-computed geometry when enabled
        if USE_FAST_TERRITORY:
            try:
                from .ai.territory_cache import find_disconnected_regions_fast
                return find_disconnected_regions_fast(board, player_number)
            except ImportError:
                pass  # Fall back to original implementation

        regions: list[Territory] = []

        # Identify active players (those with stacks on board)
        active_players = set()
        for stack in board.stacks.values():
            active_players.add(stack.controlling_player)

        # If there is only one or zero active players, there is no meaningful
        # notion of disconnection.
        if len(active_players) <= 1:
            return []

        # Collect all marker colours present on the board
        marker_colors = set()
        for marker in board.markers.values():
            marker_colors.add(marker.player)

        # Regions where markers of a specific colour act as borders
        for border_color in marker_colors:
            regions.extend(
                BoardManager._find_regions_with_border_color(
                    board,
                    border_color,
                    active_players,
                )
            )

        # Regions surrounded only by collapsed spaces and edges (no marker
        # borders).
        regions.extend(
            BoardManager._find_regions_without_marker_border(
                board,
                active_players,
            )
        )

        return regions

    @staticmethod
    def _generate_all_positions_for_board(board: BoardState) -> list[Position]:
        """Generate all valid positions for the given board."""
        positions: list[Position] = []
        if board.type == BoardType.SQUARE8:
            for x in range(8):
                for y in range(8):
                    positions.append(Position(x=x, y=y))
        elif board.type == BoardType.SQUARE19:
            for x in range(19):
                for y in range(19):
                    positions.append(Position(x=x, y=y))
        elif board.type in (BoardType.HEXAGONAL, BoardType.HEX8):
            # Hex boards: size = bounding box (2*radius + 1). Radius = (size - 1) // 2.
            radius = (board.size - 1) // 2
            for x in range(-radius, radius + 1):
                for y in range(-radius, radius + 1):
                    z = -x - y
                    if (
                        abs(x) <= radius
                        and abs(y) <= radius
                        and abs(z) <= radius
                    ):
                        # Include z for hexagonal boards - stacks/markers use "x,y,z" keys
                        positions.append(Position(x=x, y=y, z=z))
        return positions

    @staticmethod
    def _find_regions_with_border_color(
        board: BoardState,
        border_color: int,
        active_players: set,
    ) -> list[Territory]:
        """
        Find regions where markers of `border_color` act as borders.

        This mirrors the TS BoardManager.findRegionsWithBorderColor:

        - flood-fills regions while treating collapsed spaces and markers of
          `border_color` as boundaries;
        - then filters out regions that contain stacks for all active players.
        """
        disconnected_regions: list[Territory] = []
        visited: set = set()

        all_positions = BoardManager._generate_all_positions_for_board(board)

        for pos in all_positions:
            pos_key = pos.to_key()
            if pos_key in visited:
                continue

            # Skip borders: collapsed spaces or markers of border_color.
            if BoardManager.is_collapsed_space(pos, board):
                visited.add(pos_key)
                continue

            marker = BoardManager.get_marker(pos, board)
            if marker is not None and marker.player == border_color:
                visited.add(pos_key)
                continue

            region = BoardManager._explore_region_with_border_color(
                pos,
                board,
                border_color,
                visited,
            )
            if not region:
                continue

            represented_players = BoardManager._get_represented_players(
                region,
                board,
            )

            # Region must lack at least one active player's stacks.
            if len(represented_players) < len(active_players):
                disconnected_regions.append(
                    Territory(
                        **{
                            "spaces": region,
                            # Attribute control to the border color to avoid neutral regions.
                            "controllingPlayer": border_color,
                            "isDisconnected": True,
                        }
                    )
                )

        return disconnected_regions

    @staticmethod
    def _find_regions_without_marker_border(
        board: BoardState,
        active_players: set,
    ) -> list[Territory]:
        """
        Find regions surrounded only by collapsed spaces and edges (no marker
        borders).

        Mirrors the TS BoardManager.findRegionsWithoutMarkerBorder helper.
        """
        disconnected_regions: list[Territory] = []
        visited: set = set()

        all_positions = BoardManager._generate_all_positions_for_board(board)

        for pos in all_positions:
            pos_key = pos.to_key()
            if pos_key in visited:
                continue

            # Skip collapsed spaces outright.
            if BoardManager.is_collapsed_space(pos, board):
                visited.add(pos_key)
                continue

            region = BoardManager._explore_region_without_marker_border(
                pos,
                board,
                visited,
            )
            if not region:
                continue

            # Ensure region border is composed only of collapsed spaces and
            # edges (no markers).
            if not BoardManager._is_region_bordered_by_collapsed_only(
                region,
                board,
            ):
                continue

            represented_players = BoardManager._get_represented_players(
                region,
                board,
            )

            # Only keep the region if exactly one player is represented inside;
            # ambiguous/neutral regions are non-canonical and should be dropped.
            if (len(represented_players) < len(active_players)
                    and len(represented_players) == 1):
                sole_player = next(iter(represented_players))
                disconnected_regions.append(
                    Territory(
                        **{
                            "spaces": region,
                            "controllingPlayer": sole_player,
                            "isDisconnected": True,
                        }
                    )
                )

        return disconnected_regions

    @staticmethod
    def _explore_region_with_border_color(
        start: Position,
        board: BoardState,
        border_color: int,
        visited: set,
    ) -> list[Position]:
        """
        Flood-fill to find a region where markers of `border_color` act as
        borders.

        Collapsed spaces and markers of `border_color` are treated as borders;
        all other spaces (empty, stacks, other-colour markers) are part of the
        region.
        """
        region: list[Position] = []
        queue: list[Position] = [start]
        local_visited: set = set()

        while queue:
            current = queue.pop(0)
            current_key = current.to_key()
            if current_key in local_visited:
                continue
            local_visited.add(current_key)
            visited.add(current_key)

            # Borders: collapsed spaces or markers of border_color.
            if BoardManager.is_collapsed_space(current, board):
                continue

            marker = BoardManager.get_marker(current, board)
            if marker is not None and marker.player == border_color:
                continue

            # This space is part of the region.
            region.append(current)

            # Explore neighbors using territory adjacency (Von Neumann on
            # square boards, hex adjacency on hex boards).
            neighbors = BoardManager._get_territory_neighbors(
                current,
                board.type,
            )
            for neighbor in neighbors:
                if not BoardManager.is_valid_position(
                    neighbor,
                    board.type,
                    board.size,
                ):
                    continue
                n_key = neighbor.to_key()
                if n_key not in local_visited:
                    queue.append(neighbor)

        return region

    @staticmethod
    def _explore_region_without_marker_border(
        start: Position,
        board: BoardState,
        visited: set,
    ) -> list[Position]:
        """
        Flood-fill to find a region where only collapsed spaces and edges act
        as borders (markers do not terminate the fill).

        This mirrors TS BoardManager.exploreRegionWithoutMarkerBorder.
        """
        region: list[Position] = []
        queue: list[Position] = [start]
        local_visited: set = set()

        while queue:
            current = queue.pop(0)
            current_key = current.to_key()
            if current_key in local_visited:
                continue
            local_visited.add(current_key)
            visited.add(current_key)

            # Borders are only collapsed spaces; they are not part of the
            # region.
            if BoardManager.is_collapsed_space(current, board):
                continue

            region.append(current)

            neighbors = BoardManager._get_territory_neighbors(
                current,
                board.type,
            )
            for neighbor in neighbors:
                if not BoardManager.is_valid_position(
                    neighbor,
                    board.type,
                    board.size,
                ):
                    continue
                n_key = neighbor.to_key()
                if n_key not in local_visited:
                    queue.append(neighbor)

        return region

    @staticmethod
    def _is_region_bordered_by_collapsed_only(
        region_spaces: list[Position],
        board: BoardState,
    ) -> bool:
        """
        Check that a region is bordered only by collapsed spaces and edges,
        with no markers or open/stacked spaces on its perimeter.

        Mirrors TS BoardManager.isRegionBorderedByCollapsedOnly.
        """
        region_keys = {p.to_key() for p in region_spaces}

        for space in region_spaces:
            neighbors = BoardManager._get_territory_neighbors(
                space,
                board.type,
            )
            for neighbor in neighbors:
                n_key = neighbor.to_key()

                # Neighbor inside region is fine.
                if n_key in region_keys:
                    continue

                # Board edge is an acceptable border.
                if not BoardManager.is_valid_position(
                    neighbor,
                    board.type,
                    board.size,
                ):
                    continue

                # Collapsed space is an acceptable border.
                if BoardManager.is_collapsed_space(neighbor, board):
                    continue

                # If neighbor has a marker, region is NOT bordered by
                # collapsed-only.
                if BoardManager.get_marker(neighbor, board) is not None:
                    return False

                # Empty or stacked neighbor on the perimeter invalidates
                # collapsed-only status.
                return False

        return True

    @staticmethod
    def _get_represented_players(
        region_spaces: list[Position],
        board: BoardState,
    ) -> set:
        """Get all players represented in a region by their ring stacks."""
        represented = set()
        for space in region_spaces:
            stack = BoardManager.get_stack(space, board)
            if stack:
                represented.add(stack.controlling_player)
        return represented

    @staticmethod
    def get_border_marker_positions(
        spaces: list[Position],
        board: BoardState,
    ) -> list[Position]:
        """
        Get border marker positions for a disconnected region.

        Mirrors the TS BoardManager.getBorderMarkerPositions:

        - Seed border markers as any markers adjacent to the region using
          territory adjacency (Von Neumann / hex).
        - Flood-fill across connected markers using Moore adjacency (for
          square boards) to capture the entire connected marker ring,
          including diagonal corners.
        """
        region_keys = {p.to_key() for p in spaces}

        # Step 1: territory-adjacent marker seeds.
        seed_map: dict[str, Position] = {}
        for space in spaces:
            neighbors = BoardManager._get_territory_neighbors(
                space,
                board.type,
            )
            for neighbor in neighbors:
                n_key = neighbor.to_key()
                if n_key in region_keys:
                    continue
                marker = BoardManager.get_marker(neighbor, board)
                if marker is not None and n_key not in seed_map:
                    seed_map[n_key] = neighbor

        # No adjacent markers → no marker border.
        if not seed_map:
            return []

        # Step 2: BFS through markers to capture the full border ring.
        border_markers: dict[str, Position] = dict(seed_map)
        queue: list[Position] = list(seed_map.values())
        visited: set = set(seed_map.keys())

        while queue:
            current = queue.pop(0)

            # For square boards, expand using Moore adjacency (8 directions).
            # For hex boards, TS's getMooreNeighbors effectively contributes
            # no additional neighbors; we mirror that by skipping expansion.
            if board.type in (BoardType.HEXAGONAL, BoardType.HEX8):
                neighbors: list[Position] = []
            else:
                neighbors = []
                directions = BoardManager._get_all_directions(board.type)
                for direction in directions:
                    neighbors.append(
                        BoardManager._add_direction(
                            current,
                            direction,
                            1,
                        )
                    )

            for neighbor in neighbors:
                n_key = neighbor.to_key()
                if n_key in visited:
                    continue
                if n_key in region_keys:
                    continue
                if not BoardManager.is_valid_position(
                    neighbor,
                    board.type,
                    board.size,
                ):
                    continue

                marker = BoardManager.get_marker(neighbor, board)
                if marker is not None:
                    visited.add(n_key)
                    border_markers[n_key] = neighbor
                    queue.append(neighbor)

        return list(border_markers.values())

    @staticmethod
    def _get_territory_neighbors(
        pos: Position, board_type: BoardType
    ) -> list[Position]:
        if board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
            return [
                Position(
                    x=pos.x+1, y=pos.y,
                    z=pos.z-1 if pos.z is not None else None
                ),
                Position(
                    x=pos.x, y=pos.y+1,
                    z=pos.z-1 if pos.z is not None else None
                ),
                Position(
                    x=pos.x-1, y=pos.y+1,
                    z=pos.z if pos.z is not None else None
                ),
                Position(
                    x=pos.x-1, y=pos.y,
                    z=pos.z+1 if pos.z is not None else None
                ),
                Position(
                    x=pos.x, y=pos.y-1,
                    z=pos.z+1 if pos.z is not None else None
                ),
                Position(
                    x=pos.x+1, y=pos.y-1,
                    z=pos.z if pos.z is not None else None
                )
            ]
        else:
            # Von Neumann (4-direction)
            return [
                Position(x=pos.x+1, y=pos.y),
                Position(x=pos.x-1, y=pos.y),
                Position(x=pos.x, y=pos.y+1),
                Position(x=pos.x, y=pos.y-1)
            ]

    @staticmethod
    def _get_line_directions(
        board_type: BoardType,
    ) -> list[tuple[int, int, int | None]]:
        """
        Canonical line directions for line detection.

        Mirrors src/shared/engine/lineDetection.getLineDirections and the
        server BoardManager.getLineDirections helpers:
        - Square boards: E, SE, S, NE
        - Hex boards: three axial directions (E, NE, NW in cube coords)
        """
        if board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
            return [
                (1, 0, -1),   # East
                (1, -1, 0),   # Northeast
                (0, -1, 1),   # Northwest
            ]
        else:
            return [
                (1, 0, None),   # East
                (1, 1, None),   # Southeast
                (0, 1, None),   # South
                (1, -1, None),  # Northeast
            ]

    @staticmethod
    def _find_line_in_direction(
        start: Position,
        direction: tuple[int, int, int | None],
        player: int,
        board: BoardState,
    ) -> list[Position]:
        """
        Find consecutive markers in a given direction for a player.

        Mirrors shared lineDetection.findLineInDirection semantics:
        - Lines are formed by MARKERS, not stacks.
        - Collapsed spaces and stacks break lines.
        - Expands in both forward and backward directions.
        """
        line: list[Position] = [start]

        # Forward
        current = start
        while True:
            next_pos = BoardManager._add_direction(current, direction, 1)
            if not BoardManager.is_valid_position(next_pos, board.type, board.size):
                break
            if BoardManager.is_collapsed_space(next_pos, board):
                break
            if BoardManager.get_stack(next_pos, board):
                break

            marker = BoardManager.get_marker(next_pos, board)
            if marker is None or marker.player != player:
                break

            line.append(next_pos)
            current = next_pos

        # Backward
        current = start
        while True:
            prev_pos = BoardManager._add_direction(current, direction, -1)
            if not BoardManager.is_valid_position(prev_pos, board.type, board.size):
                break
            if BoardManager.is_collapsed_space(prev_pos, board):
                break
            if BoardManager.get_stack(prev_pos, board):
                break

            marker = BoardManager.get_marker(prev_pos, board)
            if marker is None or marker.player != player:
                break

            line.insert(0, prev_pos)
            current = prev_pos

        return line

    @staticmethod
    def _get_all_directions(
        board_type: BoardType
    ) -> list[tuple[int, int, int | None]]:
        if board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
            return [
                (1, 0, -1), (0, 1, -1), (-1, 1, 0),
                (-1, 0, 1), (0, -1, 1), (1, -1, 0)
            ]
        else:
            # Moore neighborhood for square boards
            dirs = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    dirs.append((dx, dy, None))
            return dirs

    @staticmethod
    def _add_direction(
        pos: Position,
        direction: tuple[int, int, int | None],
        scale: int = 1
    ) -> Position:
        return Position(
            x=pos.x + direction[0] * scale,
            y=pos.y + direction[1] * scale,
            z=(pos.z + direction[2] * scale) if (
                pos.z is not None and direction[2] is not None
            ) else None
        )

    @staticmethod
    def set_collapsed_space(
        position: Position, player_number: int, board: BoardState
    ) -> None:
        pos_key = position.to_key()
        board.collapsed_spaces[pos_key] = player_number
        # Remove any marker at this position (try multiple key formats for hex)
        for key in _get_position_keys_for_lookup(position, board.type):
            if key in board.markers:
                del board.markers[key]
                break

    @staticmethod
    def remove_stack(position: Position, board: BoardState) -> None:
        # Try multiple key formats for hex boards
        for pos_key in _get_position_keys_for_lookup(position, board.type):
            if pos_key in board.stacks:
                del board.stacks[pos_key]
                break

    @staticmethod
    def set_stack(position: Position, stack: RingStack, board: BoardState) -> None:
        pos_key = position.to_key()
        board.stacks[pos_key] = stack

    @staticmethod
    def compute_progress_snapshot(game_state: GameState) -> ProgressSnapshot:
        """
        Compute the canonical S-invariant snapshot for a given GameState.
        S = M + C + E
        """
        markers = len(game_state.board.markers)
        collapsed = len(game_state.board.collapsed_spaces)

        # Calculate total eliminated rings
        eliminated_from_board = sum(game_state.board.eliminated_rings.values())
        eliminated = max(
            game_state.total_rings_eliminated, eliminated_from_board
        )

        S = markers + collapsed + eliminated

        return ProgressSnapshot(
            markers=markers,
            collapsed=collapsed,
            eliminated=eliminated,
            S=S
        )

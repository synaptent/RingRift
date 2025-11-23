from typing import List, Optional, Tuple
from .models import (
    BoardState, Position, RingStack, BoardType, LineInfo, Territory,
    ProgressSnapshot, GameState
)


class BoardManager:
    """
    Helper class for board operations, mirroring TypeScript BoardManager
    """

    @staticmethod
    def get_stack(
        position: Position, board: BoardState
    ) -> Optional[RingStack]:
        pos_key = position.to_key()
        return board.stacks.get(pos_key)

    @staticmethod
    def is_valid_position(
        position: Position, board_type: BoardType, size: int
    ) -> bool:
        if board_type == BoardType.SQUARE8:
            return 0 <= position.x < 8 and 0 <= position.y < 8
        elif board_type == BoardType.SQUARE19:
            return 0 <= position.x < 19 and 0 <= position.y < 19
        elif board_type == BoardType.HEXAGONAL:
            if position.z is None:
                return False
            radius = size - 1
            return (abs(position.x) <= radius and 
                    abs(position.y) <= radius and 
                    abs(position.z) <= radius and
                    position.x + position.y + position.z == 0)
        return False

    @staticmethod
    def is_collapsed_space(position: Position, board: BoardState) -> bool:
        pos_key = position.to_key()
        return pos_key in board.collapsed_spaces

    @staticmethod
    def get_player_stacks(
        board: BoardState, player_number: int
    ) -> List[RingStack]:
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
        """
        # Summarize board
        stacks = []
        for key, stack in state.board.stacks.items():
            stacks.append(f"{key}:{stack.controlling_player}:{stack.stack_height}:{stack.cap_height}")
        stacks.sort()

        markers = []
        for key, marker in state.board.markers.items():
            markers.append(f"{key}:{marker.player}")
        markers.sort()

        collapsed_spaces = []
        for key, owner in state.board.collapsed_spaces.items():
            collapsed_spaces.append(f"{key}:{owner}")
        collapsed_spaces.sort()

        # Summarize players
        players_meta = []
        for p in state.players:
            players_meta.append(f"{p.player_number}:{p.rings_in_hand}:{p.eliminated_rings}:{p.territory_spaces}")
        players_meta.sort()
        players_meta_str = "|".join(players_meta)

        # Meta info
        meta = f"{state.current_player}:{state.current_phase.value}:{state.game_status.value}"
        if state.must_move_from_stack_key:
            meta += f":must_move={state.must_move_from_stack_key}"
        
        # Include capture context in hash
        if state.current_phase in ["capture", "chain_capture"]:
            if state.chain_capture_state:
                meta += f":chain={state.chain_capture_state.current_position.to_key()}"
                meta += f":visited={','.join(sorted(state.chain_capture_state.visited_positions))}"
            elif state.move_history:
                # Initial capture depends on last move's destination (attacker position)
                last_move = state.move_history[-1]
                if last_move.to:
                    meta += f":last_to={last_move.to.to_key()}"

        return "#".join([
            meta,
            players_meta_str,
            "|".join(stacks),
            "|".join(markers),
            "|".join(collapsed_spaces)
        ])

    @staticmethod
    def find_all_lines(board: BoardState) -> List[LineInfo]:
        """
        Find all marker lines on the board (3+ for 8x8, 4+ for 19x19/hex).

        Mirrors src/shared/engine/lineDetection.findAllLines and the server
        BoardManager.findAllLines implementation:
        - Lines are formed by MARKERS, not stacks.
        - Collapsed spaces and stacks act as hard blockers.
        - Uses canonical line directions per board type to avoid duplicates.
        """
        lines: List[LineInfo] = []
        processed_keys = set()

        # Determine line length based on board type
        min_length = 3 if board.type == BoardType.SQUARE8 else 4

        directions = BoardManager._get_line_directions(board.type)

        # Iterate through all markers
        for pos_key, marker in board.markers.items():
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
    ) -> List[Territory]:
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
        regions: List[Territory] = []

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
    def _generate_all_positions_for_board(board: BoardState) -> List[Position]:
        """Generate all valid positions for the given board."""
        positions: List[Position] = []
        if board.type == BoardType.SQUARE8:
            for x in range(8):
                for y in range(8):
                    positions.append(Position(x=x, y=y))
        elif board.type == BoardType.SQUARE19:
            for x in range(19):
                for y in range(19):
                    positions.append(Position(x=x, y=y))
        elif board.type == BoardType.HEXAGONAL:
            radius = board.size - 1
            for x in range(-radius, radius + 1):
                for y in range(-radius, radius + 1):
                    z = -x - y
                    if (
                        abs(x) <= radius
                        and abs(y) <= radius
                        and abs(z) <= radius
                    ):
                        positions.append(Position(x=x, y=y, z=z))
        return positions

    @staticmethod
    def _find_regions_with_border_color(
        board: BoardState,
        border_color: int,
        active_players: set,
    ) -> List[Territory]:
        """
        Find regions where markers of `border_color` act as borders.

        This mirrors the TS BoardManager.findRegionsWithBorderColor:

        - flood-fills regions while treating collapsed spaces and markers of
          `border_color` as boundaries;
        - then filters out regions that contain stacks for all active players.
        """
        disconnected_regions: List[Territory] = []
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

            marker = board.markers.get(pos_key)
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
                            "controllingPlayer": 0,
                            "isDisconnected": True,
                        }
                    )
                )

        return disconnected_regions

    @staticmethod
    def _find_regions_without_marker_border(
        board: BoardState,
        active_players: set,
    ) -> List[Territory]:
        """
        Find regions surrounded only by collapsed spaces and edges (no marker
        borders).

        Mirrors the TS BoardManager.findRegionsWithoutMarkerBorder helper.
        """
        disconnected_regions: List[Territory] = []
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
            if len(represented_players) < len(active_players):
                disconnected_regions.append(
                    Territory(
                        **{
                            "spaces": region,
                            "controllingPlayer": 0,
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
    ) -> List[Position]:
        """
        Flood-fill to find a region where markers of `border_color` act as
        borders.

        Collapsed spaces and markers of `border_color` are treated as borders;
        all other spaces (empty, stacks, other-colour markers) are part of the
        region.
        """
        region: List[Position] = []
        queue: List[Position] = [start]
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

            marker = board.markers.get(current_key)
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
    ) -> List[Position]:
        """
        Flood-fill to find a region where only collapsed spaces and edges act
        as borders (markers do not terminate the fill).

        This mirrors TS BoardManager.exploreRegionWithoutMarkerBorder.
        """
        region: List[Position] = []
        queue: List[Position] = [start]
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
        region_spaces: List[Position],
        board: BoardState,
    ) -> bool:
        """
        Check that a region is bordered only by collapsed spaces and edges,
        with no markers or open/stacked spaces on its perimeter.

        Mirrors TS BoardManager.isRegionBorderedByCollapsedOnly.
        """
        region_keys = set(p.to_key() for p in region_spaces)

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
                if n_key in board.markers:
                    return False

                # Empty or stacked neighbor on the perimeter invalidates
                # collapsed-only status.
                return False

        return True

    @staticmethod
    def _get_represented_players(
        region_spaces: List[Position],
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
        spaces: List[Position],
        board: BoardState,
    ) -> List[Position]:
        """
        Get border marker positions for a disconnected region.

        Mirrors the TS BoardManager.getBorderMarkerPositions:

        - Seed border markers as any markers adjacent to the region using
          territory adjacency (Von Neumann / hex).
        - Flood-fill across connected markers using Moore adjacency (for
          square boards) to capture the entire connected marker ring,
          including diagonal corners.
        """
        region_keys = set(p.to_key() for p in spaces)

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
                marker = board.markers.get(n_key)
                if marker is not None and n_key not in seed_map:
                    seed_map[n_key] = neighbor

        # No adjacent markers → no marker border.
        if not seed_map:
            return []

        # Step 2: BFS through markers to capture the full border ring.
        border_markers: dict[str, Position] = dict(seed_map)
        queue: List[Position] = list(seed_map.values())
        visited: set = set(seed_map.keys())

        while queue:
            current = queue.pop(0)

            # For square boards, expand using Moore adjacency (8 directions).
            # For hex boards, TS's getMooreNeighbors effectively contributes
            # no additional neighbors; we mirror that by skipping expansion.
            if board.type == BoardType.HEXAGONAL:
                neighbors: List[Position] = []
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

                marker = board.markers.get(n_key)
                if marker is not None:
                    visited.add(n_key)
                    border_markers[n_key] = neighbor
                    queue.append(neighbor)

        return list(border_markers.values())

    @staticmethod
    def _get_territory_neighbors(
        pos: Position, board_type: BoardType
    ) -> List[Position]:
        if board_type == BoardType.HEXAGONAL:
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
    ) -> List[Tuple[int, int, Optional[int]]]:
        """
        Canonical line directions for line detection.

        Mirrors src/shared/engine/lineDetection.getLineDirections and the
        server BoardManager.getLineDirections helpers:
        - Square boards: E, SE, S, NE
        - Hex boards: three axial directions (E, NE, NW in cube coords)
        """
        if board_type == BoardType.HEXAGONAL:
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
        direction: Tuple[int, int, Optional[int]],
        player: int,
        board: BoardState,
    ) -> List[Position]:
        """
        Find consecutive markers in a given direction for a player.

        Mirrors shared lineDetection.findLineInDirection semantics:
        - Lines are formed by MARKERS, not stacks.
        - Collapsed spaces and stacks break lines.
        - Expands in both forward and backward directions.
        """
        line: List[Position] = [start]

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

            marker = board.markers.get(next_pos.to_key())
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

            marker = board.markers.get(prev_pos.to_key())
            if marker is None or marker.player != player:
                break

            line.insert(0, prev_pos)
            current = prev_pos

        return line

    @staticmethod
    def _get_all_directions(
        board_type: BoardType
    ) -> List[Tuple[int, int, Optional[int]]]:
        if board_type == BoardType.HEXAGONAL:
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
        direction: Tuple[int, int, Optional[int]],
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
    ):
        pos_key = position.to_key()
        board.collapsed_spaces[pos_key] = player_number
        # Remove any marker at this position
        if pos_key in board.markers:
            del board.markers[pos_key]
            
    @staticmethod
    def remove_stack(position: Position, board: BoardState):
        pos_key = position.to_key()
        if pos_key in board.stacks:
            del board.stacks[pos_key]

    @staticmethod
    def set_stack(position: Position, stack: RingStack, board: BoardState):
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

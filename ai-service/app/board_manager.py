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

        return "#".join([
            meta,
            players_meta_str,
            "|".join(stacks),
            "|".join(markers),
            "|".join(collapsed_spaces)
        ])

    @staticmethod
    def find_all_lines(board: BoardState) -> List[LineInfo]:
        lines = []
        # Determine line length based on board type
        min_length = 4 if board.type == BoardType.SQUARE8 else 5
        
        # Get all directions
        directions = BoardManager._get_all_directions(board.type)
        
        # Iterate through all markers
        for pos_key, marker in board.markers.items():
            start_pos = marker.position
            player = marker.player
            
            for direction in directions:
                # Check if this is the start of a line in this direction
                # (i.e., previous position is not same player marker)
                prev_pos = BoardManager._add_direction(
                    start_pos, direction, -1
                )
                prev_key = prev_pos.to_key()
                if (prev_key in board.markers and
                        board.markers[prev_key].player == player):
                    continue
                
                # Trace line
                current_line_positions = [start_pos]
                current_pos = start_pos
                
                while True:
                    next_pos = BoardManager._add_direction(
                        current_pos, direction, 1
                    )
                    next_key = next_pos.to_key()

                    if (next_key in board.markers and
                            board.markers[next_key].player == player):
                        current_line_positions.append(next_pos)
                        current_pos = next_pos
                    else:
                        break
                
                if len(current_line_positions) >= min_length:
                    lines.append(LineInfo(
                        positions=current_line_positions,
                        player=player,
                        length=len(current_line_positions),
                        direction=Position(
                            x=direction[0],
                            y=direction[1],
                            z=direction[2] if len(direction) > 2 else None
                        )
                    ))
                    
        return lines

    @staticmethod
    def find_disconnected_regions(
        board: BoardState, player_number: int
    ) -> List[Territory]:
        regions = []
        visited = set()
        
        # Get all valid positions
        all_positions = []
        if board.type == BoardType.SQUARE8:
            for x in range(8):
                for y in range(8):
                    all_positions.append(Position(x=x, y=y))
        elif board.type == BoardType.SQUARE19:
            for x in range(19):
                for y in range(19):
                    all_positions.append(Position(x=x, y=y))
        elif board.type == BoardType.HEXAGONAL:
            radius = board.size - 1
            for x in range(-radius, radius + 1):
                for y in range(-radius, radius + 1):
                    z = -x - y
                    if (abs(x) <= radius and
                            abs(y) <= radius and
                            abs(z) <= radius):
                        all_positions.append(Position(x=x, y=y, z=z))

        # Identify active players (those with rings on board)
        active_players = set()
        for stack in board.stacks.values():
            active_players.add(stack.controlling_player)
            
        for pos in all_positions:
            pos_key = pos.to_key()
            if pos_key in visited:
                continue
                
            # Skip collapsed spaces and markers of the player
            if pos_key in board.collapsed_spaces:
                visited.add(pos_key)
                continue

            if (pos_key in board.markers and
                    board.markers[pos_key].player == player_number):
                visited.add(pos_key)
                continue
                
            # Start flood fill
            region_spaces = []
            region_visited = set()
            queue = [pos]
            region_visited.add(pos_key)
            visited.add(pos_key)
            
            represented_players = set()

            while queue:
                curr = queue.pop(0)
                region_spaces.append(curr)
                curr_key = curr.to_key()

                # Check representation
                if curr_key in board.stacks:
                    represented_players.add(
                        board.stacks[curr_key].controlling_player
                    )

                # Get neighbors (Von Neumann for square, Hex for hex)
                neighbors = BoardManager._get_territory_neighbors(
                    curr, board.type
                )

                for neighbor in neighbors:
                    if not BoardManager.is_valid_position(
                        neighbor, board.type, board.size
                    ):
                        # Hit edge of board - region is NOT surrounded
                        continue

                    n_key = neighbor.to_key()

                    if n_key in region_visited:
                        continue

                    # Check if boundary
                    if n_key in board.collapsed_spaces:
                        continue

                    if (n_key in board.markers and
                            board.markers[n_key].player == player_number):
                        continue
                        
                    # Not a boundary, add to region
                    region_visited.add(n_key)
                    visited.add(n_key)
                    queue.append(neighbor)
            
            # Check if region is disconnected
            # 1. Physically disconnected
            #    Flood fill logic above implicitly handles this by only
            #    visiting non-boundary nodes.
            #    If we exhausted the queue, we found a connected component
            #    bounded by boundaries/edges.
            
            # 2. Color Representation
            #    Region must lack representation from at least one active
            #    player
            
            # If region contains ALL active players, it's NOT disconnected
            if active_players.issubset(represented_players):
                continue
                
            # If we found a valid region
            if region_spaces:
                regions.append(Territory(
                    spaces=region_spaces,
                    controlling_player=player_number,
                    is_disconnected=True
                ))
                
        return regions

    @staticmethod
    def get_border_marker_positions(
        spaces: List[Position], board: BoardState
    ) -> List[Position]:
        border_markers = []
        region_set = set(p.to_key() for p in spaces)
        
        for pos in spaces:
            neighbors = BoardManager._get_territory_neighbors(pos, board.type)
            for neighbor in neighbors:
                n_key = neighbor.to_key()
                if n_key not in region_set:
                    # Neighbor is outside region
                    if n_key in board.markers:
                        # It's a marker, add to border
                        # Note: We should check if it's the player's marker,
                        # but caller usually handles context
                        border_markers.append(neighbor)
                        
        # Deduplicate
        unique_markers = []
        seen = set()
        for m in border_markers:
            k = m.to_key()
            if k not in seen:
                seen.add(k)
                unique_markers.append(m)
                
        return unique_markers

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
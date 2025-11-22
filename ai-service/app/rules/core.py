"""
Shared core utilities for RingRift rules.
Mirrors src/shared/engine/core.ts
"""

from typing import List, Optional, Protocol, Any, Dict
from app.models import Position, BoardType, GameState, BoardState


class BoardView(Protocol):
    """Minimal interface for board access needed by core validators."""
    def is_valid_position(self, pos: Position) -> bool: ...
    def is_collapsed_space(self, pos: Position) -> bool: ...
    # Returns RingStack-like
    def get_stack_at(self, pos: Position) -> Optional[Any]: ...
    def get_marker_owner(self, pos: Position) -> Optional[int]: ...


def calculate_cap_height(rings: List[int]) -> int:
    """
    Calculate the cap height of a stack.
    Mirrors core.ts:calculateCapHeight
    """
    if not rings:
        return 0

    controlling_player = rings[-1]
    height = 0
    for r in reversed(rings):
        if r == controlling_player:
            height += 1
        else:
            break
    return height


def calculate_distance(
    board_type: BoardType, from_pos: Position, to_pos: Position
) -> int:
    """
    Calculate distance between two positions.
    Mirrors core.ts:calculateDistance
    """
    if board_type == BoardType.HEXAGONAL:
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y
        dz = (to_pos.z or 0) - (from_pos.z or 0)
        return int((abs(dx) + abs(dy) + abs(dz)) / 2)

    dx = abs(to_pos.x - from_pos.x)
    dy = abs(to_pos.y - from_pos.y)
    return max(dx, dy)


def get_path_positions(from_pos: Position, to_pos: Position) -> List[Position]:
    """
    Get all positions along a straight-line path, inclusive.
    Mirrors core.ts:getPathPositions
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
        path.append(Position(**pos_kwargs))

    return path


def summarize_board(board: BoardState) -> Dict[str, List[str]]:
    """
    Build a lightweight, order-independent summary of a BoardState.
    Mirrors core.ts:summarizeBoard
    """
    stacks = []
    for key, stack in board.stacks.items():
        stacks.append(
            f"{key}:{stack.controlling_player}:"
            f"{stack.stack_height}:{stack.cap_height}"
        )
    stacks.sort()

    markers = []
    for key, marker in board.markers.items():
        markers.append(f"{key}:{marker.player}")
    markers.sort()

    collapsed_spaces = []
    for key, owner in board.collapsed_spaces.items():
        collapsed_spaces.append(f"{key}:{owner}")
    collapsed_spaces.sort()

    return {
        "stacks": stacks,
        "markers": markers,
        "collapsedSpaces": collapsed_spaces
    }


def compute_progress_snapshot(state: GameState) -> Dict[str, int]:
    """
    Compute the canonical S-invariant snapshot for a given GameState.
    Mirrors core.ts:computeProgressSnapshot
    """
    markers = len(state.board.markers)
    collapsed = len(state.board.collapsed_spaces)

    # In the Python GameState model, total_rings_eliminated is required,
    # so we use it directly to match the primary logic of the TS
    # implementation.
    eliminated = state.total_rings_eliminated

    S = markers + collapsed + eliminated
    return {
        "markers": markers,
        "collapsed": collapsed,
        "eliminated": eliminated,
        "S": S
    }


def hash_game_state(state: GameState) -> str:
    """
    Canonical hash of a GameState used by tests and diagnostic tooling.
    Mirrors core.ts:hashGameState
    """
    board_summary = summarize_board(state.board)

    players_meta_list = [
        f"{p.player_number}:{p.rings_in_hand}:"
        f"{p.eliminated_rings}:{p.territory_spaces}"
        for p in state.players
    ]
    players_meta_list.sort()
    players_meta = "|".join(players_meta_list)

    # Ensure we use the string value of enums
    current_phase = (
        state.current_phase.value
        if hasattr(state.current_phase, "value")
        else state.current_phase
    )
    game_status = (
        state.game_status.value
        if hasattr(state.game_status, "value")
        else state.game_status
    )

    meta = f"{state.current_player}:{current_phase}:{game_status}"

    return "#".join([
        meta,
        players_meta,
        "|".join(board_summary["stacks"]),
        "|".join(board_summary["markers"]),
        "|".join(board_summary["collapsedSpaces"])
    ])
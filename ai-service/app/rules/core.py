"""
Shared core utilities for RingRift rules.
Mirrors src/shared/engine/core.ts
"""

from typing import List, Optional, Protocol, Any, Dict, NamedTuple
from app.models import Position, BoardType, GameState, GameStatus, BoardState


class BoardConfig(NamedTuple):
    """Board configuration matching TS BOARD_CONFIGS."""
    size: int
    total_spaces: int
    rings_per_player: int
    line_length: int


# Mirrors src/shared/types/game.ts BOARD_CONFIGS
BOARD_CONFIGS: Dict[BoardType, BoardConfig] = {
    BoardType.SQUARE8: BoardConfig(
        size=8,
        total_spaces=64,
        rings_per_player=18,
        line_length=3,  # 8x8 uses 3-marker lines, larger boards use 4
    ),
    BoardType.SQUARE19: BoardConfig(
        size=19,
        total_spaces=361,
        rings_per_player=48,
        line_length=4,
    ),
    BoardType.HEXAGONAL: BoardConfig(
        size=13,
        total_spaces=469,
        rings_per_player=72,
        line_length=4,
    ),
}


def get_line_length_for_board(board_type: BoardType) -> int:
    """
    Return the BASE line length for the given board type.
    Mirrors TS BOARD_CONFIGS[boardType].lineLength

    NOTE: For player-count-aware line length, use get_effective_line_length.
    """
    return BOARD_CONFIGS[board_type].line_length


def get_effective_line_length(board_type: BoardType, num_players: int) -> int:
    """
    Return the effective line length threshold for the given board and player count.

    Canonical semantics (RR-CANON-R120):
    - square8 2-player: line length = 4
    - square8 3-4 player: line length = 3
    - square19 and hexagonal: line length = 4 (all player counts)
    """
    # Per RR-CANON-R120: square8 2-player games require line length 4,
    # while 3-4 player games require line length 3.
    if board_type == BoardType.SQUARE8 and num_players == 2:
        return 4

    # For all other configurations, use the base line_length from BOARD_CONFIGS:
    # - square8 3-4p: 3
    # - square19: 4
    # - hexagonal: 4
    return BOARD_CONFIGS[board_type].line_length


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

    Mirrors src/shared/engine/core.ts:computeProgressSnapshot:

      S = M + C + E
        - M = markers.size
        - C = collapsedSpaces.size
        - E = totalRingsEliminated, falling back to the sum of
          board.eliminatedRings when needed.
    """
    markers = len(state.board.markers)
    collapsed = len(state.board.collapsed_spaces)

    # Align with the TS implementation: prefer the aggregated
    # total_rings_eliminated field when it is present, but fall back
    # to the board-level eliminated_rings summary when it is not.
    eliminated_from_board = sum(state.board.eliminated_rings.values())
    eliminated = state.total_rings_eliminated if state.total_rings_eliminated is not None else eliminated_from_board

    S = markers + collapsed + eliminated
    return {
        "markers": markers,
        "collapsed": collapsed,
        "eliminated": eliminated,
        "S": S,
    }


def count_rings_in_play_for_player(
    state: GameState,
    player_number: int,
) -> int:
    """
    Count all rings of a given player's colour that are currently in play.

    Mirrors the TypeScript core.countRingsInPlayForPlayer helper:

    - Iterate all stacks on the board and count rings whose value equals
      ``player_number`` (including buried rings, not just the cap).
    - Add the player's rings_in_hand, if present in ``state.players``.
    """
    rings_on_board = 0
    for stack in state.board.stacks.values():
        # Rings are stored bottom→top as player numbers.
        for ring_owner in stack.rings:
            if ring_owner == player_number:
                rings_on_board += 1

    rings_in_hand = 0
    for player in state.players:
        if player.player_number == player_number:
            rings_in_hand = player.rings_in_hand
            break

    return rings_on_board + rings_in_hand


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

    # For terminal states, currentPlayer/currentPhase are host-local
    # metadata and not semantically meaningful. Canonicalise them so
    # that engines which differ only in their choice of terminal
    # phase/player still produce identical fingerprints.
    # Note: 'game_over' is the canonical terminal phase as of RR-PARITY-FIX-2024-12.
    is_terminal = game_status in {
        GameStatus.COMPLETED.value,
        GameStatus.ABANDONED.value,
    }
    meta_player = 0 if is_terminal else state.current_player
    meta_phase = "game_over" if is_terminal else current_phase

    meta = f"{meta_player}:{meta_phase}:{game_status}"

    return "#".join(
        [
            meta,
            players_meta,
            "|".join(board_summary["stacks"]),
            "|".join(board_summary["markers"]),
            "|".join(board_summary["collapsedSpaces"]),
        ]
    )


# ═══════════════════════════════════════════════════════════════════════════
# RECOVERY HELPERS (RR-CANON-R110–R115)
# Mirrors src/shared/engine/playerStateHelpers.ts
# ═══════════════════════════════════════════════════════════════════════════


def player_controls_any_stack(board: BoardState, player_number: int) -> bool:
    """
    Check if a player controls any stacks on the board.

    A player controls a stack when their ring is on top (the controlling player).
    """
    for stack in board.stacks.values():
        if stack.controlling_player == player_number:
            return True
    return False


def player_has_markers(board: BoardState, player_number: int) -> bool:
    """
    Check if a player owns any markers on the board.

    Mirrors TS playerStateHelpers.ts:playerHasMarkers
    """
    for marker in board.markers.values():
        if marker.player == player_number:
            return True
    return False


def count_buried_rings(board: BoardState, player_number: int) -> int:
    """
    Count buried rings for a player.

    A buried ring is a ring of the player's colour that is in an opponent-
    controlled stack (not the top ring). These are used for recovery action
    costs per RR-CANON-R113.

    Mirrors TS playerStateHelpers.ts:countBuriedRings
    """
    count = 0
    for stack in board.stacks.values():
        # Only count rings in opponent-controlled stacks
        if stack.controlling_player == player_number:
            continue

        # Count rings belonging to this player (excluding top ring)
        for i in range(len(stack.rings) - 1):
            if stack.rings[i] == player_number:
                count += 1

    return count


def is_eligible_for_recovery(state: GameState, player_number: int) -> bool:
    """
    Check if a player is eligible for recovery action.

    A player is eligible per RR-CANON-R110 if:
    - They control no stacks
    - They have no rings in hand
    - They own at least one marker
    - They have at least one buried ring

    Mirrors TS playerStateHelpers.ts:isEligibleForRecovery
    """
    player = None
    for p in state.players:
        if p.player_number == player_number:
            player = p
            break

    if not player:
        return False

    # Must have no rings in hand
    if player.rings_in_hand > 0:
        return False

    # Must control no stacks
    if player_controls_any_stack(state.board, player_number):
        return False

    # Must own at least one marker
    if not player_has_markers(state.board, player_number):
        return False

    # Must have at least one buried ring
    if count_buried_rings(state.board, player_number) < 1:
        return False

    return True

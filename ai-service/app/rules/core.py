"""
Shared core utilities for RingRift rules.
Mirrors src/shared/engine/core.ts
"""

from typing import Any, NamedTuple, Protocol

from app.models import BoardState, BoardType, GameState, GameStatus, Position


class BoardConfig(NamedTuple):
    """Board configuration matching TS BOARD_CONFIGS."""
    size: int
    total_spaces: int
    rings_per_player: int
    line_length: int


# Mirrors src/shared/types/game.ts BOARD_CONFIGS
BOARD_CONFIGS: dict[BoardType, BoardConfig] = {
    BoardType.SQUARE8: BoardConfig(
        size=8,
        total_spaces=64,
        rings_per_player=18,
        line_length=3,  # 8x8 uses 3-marker lines, larger boards use 4
    ),
    BoardType.SQUARE19: BoardConfig(
        size=19,
        total_spaces=361,
        rings_per_player=72,
        line_length=4,
    ),
    BoardType.HEXAGONAL: BoardConfig(
        size=25,                 # Bounding box = 2*radius + 1 = 25 for radius=12
        total_spaces=469,
        rings_per_player=96,
        line_length=4,
    ),
    BoardType.HEX8: BoardConfig(
        size=9,                  # Bounding box = 2*radius + 1 = 9 for radius=4
        total_spaces=61,         # 3r² + 3r + 1 = 61 for r=4
        rings_per_player=18,     # Same as square8
        line_length=4,           # Standard line length for hex boards
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
    - hex8 2-player: line length = 4
    - hex8 3-4 player: line length = 3
    - square19 and hexagonal: line length = 4 (all player counts)
    """
    # Per RR-CANON-R120: square8 and hex8 2-player games require line length 4,
    # while 3-4 player games require line length 3.
    if board_type in (BoardType.SQUARE8, BoardType.HEX8):
        return 4 if num_players == 2 else 3

    # For all other configurations, use the base line_length from BOARD_CONFIGS:
    # - square19: 4
    # - hexagonal: 4
    return BOARD_CONFIGS[board_type].line_length


def get_victory_threshold(
    board_type: BoardType,
    num_players: int,
    rings_per_player_override: int | None = None,
) -> int:
    """
    Calculate the ring elimination victory threshold for the given board and player count.

    Per RR-CANON-R061:
    victoryThreshold = round(ringsPerPlayer × (2/3 + 1/3 × (numPlayers - 1)))

    This is: (2/3 × ownStartingRings) + (1/3 × combinedOpponentRings)
    For 2-player games, this equals ringsPerPlayer (must eliminate all opponent rings).
    For 3-player: 24 (8×8), 96 (19×19), 128 (hex)
    For 4-player: 30 (8×8), 120 (19×19), 160 (hex)

    Args:
        board_type: The board type.
        num_players: Number of players.
        rings_per_player_override: Optional override for rings per player (for experiments).
    """
    rings_per_player = rings_per_player_override or BOARD_CONFIGS[board_type].rings_per_player
    return round(rings_per_player * (2/3 + (1/3) * (num_players - 1)))


def get_territory_victory_minimum(board_type: BoardType, num_players: int) -> int:
    """
    Calculate the minimum territory required for victory.

    Per RR-CANON-R062-v2:
    territoryVictoryMinimum = floor(totalSpaces / numPlayers) + 1

    Victory also requires more territory than all opponents combined.
    This dual-condition rule makes territory victory more achievable in
    multiplayer games while maintaining balance.

    Args:
        board_type: The board type.
        num_players: Number of players in the game.

    Returns:
        Minimum territory spaces required for victory consideration.
    """
    total_spaces = BOARD_CONFIGS[board_type].total_spaces
    return (total_spaces // num_players) + 1


def get_territory_victory_threshold(board_type: BoardType) -> int:
    """
    DEPRECATED: Use get_territory_victory_minimum() with num_players.

    Legacy >50% threshold for 2-player games and backward compatibility.

    Per RR-CANON-R062 (legacy):
    territoryVictoryThreshold = floor(totalSpaces / 2) + 1
    """
    total_spaces = BOARD_CONFIGS[board_type].total_spaces
    return (total_spaces // 2) + 1


def get_rings_per_player(
    board_type: BoardType,
    override: int | None = None,
) -> int:
    """
    Return the starting ring supply per player for the given board type.

    Per BOARD_CONFIGS:
    - square8: 18
    - square19: 72
    - hex8: 18
    - hexagonal: 96

    Args:
        board_type: The board type.
        override: Optional override for rings per player (for experiments).
    """
    if override is not None:
        return override
    return BOARD_CONFIGS[board_type].rings_per_player


def get_board_size(board_type: BoardType) -> int:
    """
    Return the board size (embedding grid dimension) for the given board type.

    Per BOARD_CONFIGS, size = bounding box for hex boards (2*radius + 1):
    - square8: 8
    - square19: 19
    - hex8: 9 (bounding box for radius=4)
    - hexagonal: 25 (bounding box for radius=12)

    To derive the radius for hex boards: radius = (size - 1) // 2
    """
    return BOARD_CONFIGS[board_type].size


def get_total_spaces(board_type: BoardType) -> int:
    """
    Return the total number of spaces for the given board type.

    Per BOARD_CONFIGS:
    - square8: 64
    - square19: 361
    - hex8: 61
    - hexagonal: 469
    """
    return BOARD_CONFIGS[board_type].total_spaces


class BoardView(Protocol):
    """Minimal interface for board access needed by core validators."""
    def is_valid_position(self, pos: Position) -> bool: ...
    def is_collapsed_space(self, pos: Position) -> bool: ...
    # Returns RingStack-like
    def get_stack_at(self, pos: Position) -> Any | None: ...
    def get_marker_owner(self, pos: Position) -> int | None: ...


def calculate_cap_height(rings: list[int]) -> int:
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
    if board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y
        dz = (to_pos.z or 0) - (from_pos.z or 0)
        return int((abs(dx) + abs(dy) + abs(dz)) / 2)

    dx = abs(to_pos.x - from_pos.x)
    dy = abs(to_pos.y - from_pos.y)
    return max(dx, dy)


def get_path_positions(from_pos: Position, to_pos: Position) -> list[Position]:
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
        x = round(from_pos.x + step_x * i)
        y = round(from_pos.y + step_y * i)
        pos_kwargs = {"x": x, "y": y}
        # Mirror TS: only include z when the destination carries z.
        if to_pos.z is not None:
            z = round(dz_from + step_z * i)
            pos_kwargs["z"] = z
        path.append(Position(**pos_kwargs))

    return path


def summarize_board(board: BoardState) -> dict[str, list[str]]:
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


def compute_progress_snapshot(state: GameState) -> dict[str, int]:
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

    # RR-PARITY-FIX-2025-12-21: Include pendingLineRewardElimination in hash
    # to detect ANM divergence between TS and Python. This flag affects
    # has_phase_local_interactive_move in LINE_PROCESSING phase.
    pending_line = "1" if getattr(state, "pending_line_reward_elimination", False) else "0"

    meta = f"{meta_player}:{meta_phase}:{game_status}:{pending_line}"

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
    return any(stack.controlling_player == player_number for stack in board.stacks.values())


def player_has_markers(board: BoardState, player_number: int) -> bool:
    """
    Check if a player owns any markers on the board.

    Mirrors TS playerStateHelpers.ts:playerHasMarkers
    """
    return any(marker.player == player_number for marker in board.markers.values())


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
    - They own at least one marker
    - They have at least one buried ring

    Note: Recovery eligibility is independent of rings in hand.
    Players with rings in hand may reach recovery by voluntarily recording
    skip_placement in ring_placement and then using recovery in movement.

    Mirrors TS playerStateHelpers.ts:isEligibleForRecovery
    """
    player = None
    for p in state.players:
        if p.player_number == player_number:
            player = p
            break

    if not player:
        return False

    # Must control no stacks
    if player_controls_any_stack(state.board, player_number):
        return False

    # Must own at least one marker
    if not player_has_markers(state.board, player_number):
        return False

    # Must have at least one buried ring
    return not count_buried_rings(state.board, player_number) < 1

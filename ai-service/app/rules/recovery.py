"""
Recovery Action Implementation for RingRift AI Service.

This module implements the recovery action (RR-CANON-R110–R115) which allows
temporarily eliminated players (no stacks, no rings in hand, but with markers
and buried rings) to slide a marker to complete a line.

Mirrors src/shared/engine/aggregates/RecoveryAggregate.ts

Cost Model (Option 1 / Option 2):
- Exact-length lines: Always collapse all markers, cost = 1 buried ring (Option 1 only)
- Overlength lines: Player chooses:
  - Option 1: Collapse all markers, cost = 1 buried ring
  - Option 2: Collapse exactly lineLength consecutive markers, cost = 0 (free)

**Important:** Recovery moves use buried ring extraction, NOT stack cap elimination.
This is intentionally cheaper than normal territory processing which requires an
entire stack cap. Stack cap eligibility: mixed-colour (buried rings of other
colours beneath) OR single-colour stack height > 1.

Recovery moves CAN cause territory disconnection. If a recovery slide (line or
fallback) results in territory disconnection, the triggered territory processing
uses the recovery exception cost (1 buried ring), not an entire stack cap.

**SSoT Policy:**
The canonical rules defined in RULES_CANONICAL_SPEC.md are the ultimate
authority. The TS shared engine (src/shared/engine/**) is the primary
executable derivation. This Python module must mirror the canonical rules
and TS implementation. If this code disagrees with either, this code must
be updated—never the other way around.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal
from datetime import datetime

from app.models import (
    GameState,
    Move,
    Position,
    MoveType,
    GamePhase,
    BoardState,
    BoardType,
    MarkerInfo,
)
from app.board_manager import BoardManager
from app.rules.core import (
    count_buried_rings,
    is_eligible_for_recovery,
    get_effective_line_length,
)

# Recovery option type alias (for line recovery Option 1/2)
RecoveryOption = Literal[1, 2]

# Recovery mode type alias (which success criterion was met per RR-CANON-R112)
# - "line": Condition (a) - completes a line of at least lineLength markers
# - "fallback": Condition (b) - no line available, but any adjacent slide is legal
# - "stack_strike": Experimental (v1) fallback-class recovery. When enabled via
#   RINGRIFT_RECOVERY_STACK_STRIKE_V1=1 and no line-forming recovery exists,
#   a player may slide a marker onto an adjacent stack to eliminate that stack's
#   top ring; the marker is removed. Costs 1 buried ring extraction.
# Note: Territory disconnection was removed as a recovery criterion
RecoveryMode = Literal["line", "fallback", "stack_strike"]


# ═══════════════════════════════════════════════════════════════════════════
# RECOVERY TYPES
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class RecoverySlideTarget:
    """A valid target position for a recovery slide move (before option selection)."""

    from_pos: Position
    to_pos: Position
    markers_in_line: int  # number of markers (including the sliding marker) that form the line
    is_overlength: bool  # whether this is an overlength line (> lineLength)
    option1_cost: int  # always 1 for Option 1 (collapse all)
    option2_available: bool  # True only for overlength lines
    option2_cost: int  # always 0 for Option 2 (collapse lineLength, free)
    line_positions: List[Position]  # all marker positions in the line
    # Deprecated: use option1_cost/option2_cost instead
    cost: int = 1  # kept for backwards compatibility


@dataclass
class RecoveryValidationResult:
    """Result of validating a recovery slide move."""

    valid: bool
    reason: Optional[str] = None
    markers_in_line: int = 0
    is_overlength: bool = False
    option_used: Optional[RecoveryOption] = None
    cost: int = 0  # Effective cost based on option used
    line_positions: List[Position] = field(default_factory=list)  # Positions in the formed line


@dataclass
class RecoveryApplicationOutcome:
    """Outcome of applying a recovery slide."""

    success: bool
    error: Optional[str] = None
    option_used: Optional[RecoveryOption] = None
    rings_extracted: int = 0  # 1 for Option 1, 0 for Option 2
    line_positions: List[Position] = field(default_factory=list)  # All markers in the formed line
    collapsed_positions: List[Position] = field(default_factory=list)  # Markers that were collapsed


@dataclass
class EligibleExtractionStack:
    """Information about a stack eligible for buried ring extraction."""

    position_key: str  # Position key of the stack (e.g., "3,4")
    position: Position  # Position of the stack
    bottom_ring_index: int  # Index of the bottommost buried ring in the stack.rings list
    stack_height: int  # Total stack height
    controlling_player: int  # Current controlling player (top ring owner)


@dataclass
class ExpandedRecoveryTarget:
    """
    An expanded recovery target supporting line and fallback criteria (RR-CANON-R112).

    This replaces RecoverySlideTarget for the expanded recovery rules.
    Note: Territory disconnection was removed as a recovery criterion.
    """

    from_pos: Position
    to_pos: Position
    recovery_mode: RecoveryMode  # "line" or "fallback"

    # Line recovery fields (mode == "line")
    markers_in_line: int = 0
    is_overlength: bool = False
    line_positions: List[Position] = field(default_factory=list)

    # Cost information
    # - Line: Option 1 = 1, Option 2 (overlength only) = 0
    # - Fallback: 1
    min_cost: int = 1  # Minimum buried rings needed


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _get_moore_directions(board_type: BoardType) -> List[Tuple[int, int, Optional[int]]]:
    """
    Get Moore neighborhood directions (8 directions for square boards,
    6 axial directions for hex).
    """
    if board_type == BoardType.HEXAGONAL:
        return [(1, 0, -1), (0, 1, -1), (-1, 1, 0), (-1, 0, 1), (0, -1, 1), (1, -1, 0)]
    else:
        # Moore neighborhood: 8 directions
        directions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                directions.append((dx, dy, None))
        return directions


def _add_direction(pos: Position, direction: Tuple[int, int, Optional[int]], scale: int = 1) -> Position:
    """Add a direction vector to a position."""
    new_x = pos.x + direction[0] * scale
    new_y = pos.y + direction[1] * scale
    new_z = None
    if pos.z is not None and direction[2] is not None:
        new_z = pos.z + direction[2] * scale
    return Position(x=new_x, y=new_y, z=new_z)


def _is_adjacent(board_type: BoardType, pos1: Position, pos2: Position) -> bool:
    """Check if two positions are adjacent (Moore neighborhood for square, hex-adjacent for hex)."""
    directions = _get_moore_directions(board_type)
    for direction in directions:
        neighbor = _add_direction(pos1, direction, 1)
        if neighbor.x == pos2.x and neighbor.y == pos2.y:
            if board_type == BoardType.HEXAGONAL:
                # For hex, also check z coordinate
                if neighbor.z == pos2.z:
                    return True
            else:
                return True
    return False


def _can_marker_slide_to(
    board: BoardState,
    from_pos: Position,
    to_pos: Position,
    player: int,
) -> bool:
    """
    Check if a marker can slide from from_pos to to_pos.

    Requirements:
    - Target must be adjacent (Moore/hex-adjacent)
    - Target must be empty (no stack, no marker, not collapsed)
    - Target must be a valid board position
    """
    # Must be adjacent
    if not _is_adjacent(board.type, from_pos, to_pos):
        return False

    # Must be valid position on board
    if not BoardManager.is_valid_position(to_pos, board.type, board.size):
        return False

    # Must not be collapsed
    if BoardManager.is_collapsed_space(to_pos, board):
        return False

    # Must not have a stack
    if BoardManager.get_stack(to_pos, board) is not None:
        return False

    # Must not have a marker
    to_key = to_pos.to_key()
    if to_key in board.markers:
        return False

    return True


def _count_markers_in_line_through(
    board: BoardState,
    marker_pos: Position,
    player: int,
    direction: Tuple[int, int, Optional[int]],
) -> Tuple[int, List[Position]]:
    """
    Count consecutive markers of the player's colour in a line through marker_pos
    in the given direction, returning the count and positions.

    Mirrors the TS findLineInDirection logic but returns a tuple.
    """
    positions = [marker_pos]

    # Forward
    current = marker_pos
    while True:
        next_pos = _add_direction(current, direction, 1)
        if not BoardManager.is_valid_position(next_pos, board.type, board.size):
            break
        if BoardManager.is_collapsed_space(next_pos, board):
            break
        if BoardManager.get_stack(next_pos, board) is not None:
            break

        marker = board.markers.get(next_pos.to_key())
        if marker is None or marker.player != player:
            break

        positions.append(next_pos)
        current = next_pos

    # Reverse direction
    reverse_dir = (
        -direction[0],
        -direction[1],
        -direction[2] if direction[2] is not None else None,
    )

    # Backward
    current = marker_pos
    while True:
        prev_pos = _add_direction(current, reverse_dir, 1)
        if not BoardManager.is_valid_position(prev_pos, board.type, board.size):
            break
        if BoardManager.is_collapsed_space(prev_pos, board):
            break
        if BoardManager.get_stack(prev_pos, board) is not None:
            break

        marker = board.markers.get(prev_pos.to_key())
        if marker is None or marker.player != player:
            break

        positions.insert(0, prev_pos)
        current = prev_pos

    return len(positions), positions


def _would_complete_line_at(
    board: BoardState,
    player: int,
    to_pos: Position,
    line_length: int,
) -> Tuple[bool, int, List[Position]]:
    """
    Check if placing a marker at to_pos would complete a line of at least line_length.

    Returns (completes_line, markers_count, line_positions).
    """
    # Simulate the marker being at to_pos for line checking
    # We'll create a virtual marker map for checking
    directions = BoardManager._get_line_directions(board.type)

    for direction in directions:
        # Count markers through to_pos in this direction
        # We need to simulate the marker being at to_pos
        positions = [to_pos]

        # Forward
        current = to_pos
        while True:
            next_pos = _add_direction(current, direction, 1)
            if not BoardManager.is_valid_position(next_pos, board.type, board.size):
                break
            if BoardManager.is_collapsed_space(next_pos, board):
                break
            if BoardManager.get_stack(next_pos, board) is not None:
                break

            marker = board.markers.get(next_pos.to_key())
            if marker is None or marker.player != player:
                break

            positions.append(next_pos)
            current = next_pos

        # Backward (reverse direction)
        reverse_dir = (
            -direction[0],
            -direction[1],
            -direction[2] if direction[2] is not None else None,
        )
        current = to_pos
        while True:
            prev_pos = _add_direction(current, reverse_dir, 1)
            if not BoardManager.is_valid_position(prev_pos, board.type, board.size):
                break
            if BoardManager.is_collapsed_space(prev_pos, board):
                break
            if BoardManager.get_stack(prev_pos, board) is not None:
                break

            marker = board.markers.get(prev_pos.to_key())
            if marker is None or marker.player != player:
                break

            positions.insert(0, prev_pos)
            current = prev_pos

        if len(positions) >= line_length:
            return True, len(positions), positions

    return False, 0, []


# NOTE: _would_cause_territory_disconnection() was removed per RR-CANON-R112(b) update.
# Recovery fallback moves now ALLOW territory disconnection - no restriction applies.
# The function was previously used to PREVENT fallback recovery moves that would cause
# territory disconnection, but the rules have changed to permit such moves.


# ═══════════════════════════════════════════════════════════════════════════
# RECOVERY COST CALCULATION
# ═══════════════════════════════════════════════════════════════════════════


def calculate_recovery_cost(option: RecoveryOption) -> int:
    """
    Calculate the recovery cost for a given option.

    Cost Model (Option 1 / Option 2):
    - Option 1 (collapse all): 1 buried ring extraction
    - Option 2 (collapse lineLength, overlength only): 0 (free)

    Per RR-CANON-R113.

    Args:
        option: Which option (1 or 2)

    Returns:
        Number of buried rings to extract (1 for Option 1, 0 for Option 2)
    """
    return 1 if option == 1 else 0


def enumerate_eligible_extraction_stacks(
    board: BoardState,
    player: int,
) -> List[EligibleExtractionStack]:
    """
    Enumerate all stacks from which a player can extract a buried ring.

    Per RR-CANON-R113: The player chooses which stack to extract from if
    multiple stacks contain their buried rings. The bottommost ring from
    the chosen stack is extracted.

    A stack is eligible if:
    1. It contains at least one of the player's rings
    2. At least one of those rings is buried (not the top ring)

    Args:
        board: Current board state
        player: Player seeking extraction

    Returns:
        List of eligible extraction stacks with metadata
    """
    eligible_stacks: List[EligibleExtractionStack] = []

    for pos_key, stack in board.stacks.items():
        # Find the bottommost ring of this player
        try:
            bottom_ring_index = stack.rings.index(player)
        except ValueError:
            continue  # Player has no ring in this stack

        # Check if it's buried (not the top ring)
        is_top_ring = bottom_ring_index == len(stack.rings) - 1
        if is_top_ring:
            continue  # Not buried, cannot extract

        # Position.from_key is not available; parse manually.
        parts = pos_key.split(",")
        pos = Position(
            x=int(parts[0]),
            y=int(parts[1]),
            z=int(parts[2]) if len(parts) > 2 else None,
        )

        eligible_stacks.append(
            EligibleExtractionStack(
                position_key=pos_key,
                position=pos,
                bottom_ring_index=bottom_ring_index,
                stack_height=stack.stack_height,
                controlling_player=stack.controlling_player,
            )
        )

    return eligible_stacks


# ═══════════════════════════════════════════════════════════════════════════
# RECOVERY MOVE ENUMERATION
# ═══════════════════════════════════════════════════════════════════════════


def enumerate_recovery_slide_targets(
    state: GameState,
    player: int,
) -> List[RecoverySlideTarget]:
    """
    Enumerate all valid recovery slide targets for a player.

    A valid recovery slide:
    1. Moves one of the player's markers
    2. To an adjacent empty space (Moore/hex-adjacent)
    3. Where the final position completes a line of at least lineLength markers

    Cost Model (Option 1 / Option 2):
    - Exact-length lines: Only Option 1 available (cost = 1 buried ring)
    - Overlength lines: Player chooses:
      - Option 1: Collapse all markers, cost = 1 buried ring
      - Option 2: Collapse exactly lineLength markers, cost = 0 (free)

    Mirrors src/shared/engine/aggregates/RecoveryAggregate.enumerateRecoverySlideTargets.

    Args:
        state: Current game state
        player: Player to enumerate recovery moves for

    Returns:
        List of valid recovery slide targets
    """
    targets: List[RecoverySlideTarget] = []
    board = state.board
    line_length = get_effective_line_length(board.type, len(state.players))
    buried_ring_count = count_buried_rings(board, player)

    # Find all markers owned by the player
    player_marker_positions: List[Position] = []
    for pos_key, marker in board.markers.items():
        if marker.player == player:
            player_marker_positions.append(marker.position)

    # For each marker, check adjacent empty spaces
    for marker_pos in player_marker_positions:
        directions = _get_moore_directions(board.type)

        for direction in directions:
            to_pos = _add_direction(marker_pos, direction, 1)

            # Check if slide is valid
            if not _can_marker_slide_to(board, marker_pos, to_pos, player):
                continue

            # Temporarily "move" the marker by removing from original and adding to target
            # for line checking purposes
            original_marker = board.markers.get(marker_pos.to_key())
            del board.markers[marker_pos.to_key()]
            board.markers[to_pos.to_key()] = MarkerInfo(
                position=to_pos,
                player=player,
                type="regular",
            )

            # Check if this completes a line
            completes, markers_count, line_positions = _would_complete_line_at(board, player, to_pos, line_length)

            # Restore the marker
            del board.markers[to_pos.to_key()]
            board.markers[marker_pos.to_key()] = original_marker

            if completes:
                is_overlength = markers_count > line_length

                # For exact-length: need 1 buried ring (Option 1 only)
                # For overlength: Option 2 is free, so always legal
                can_use_option1 = buried_ring_count >= 1
                can_use_option2 = is_overlength  # Option 2 is free but only for overlength

                # At least one option must be available
                if can_use_option1 or can_use_option2:
                    targets.append(
                        RecoverySlideTarget(
                            from_pos=marker_pos,
                            to_pos=to_pos,
                            markers_in_line=markers_count,
                            is_overlength=is_overlength,
                            option1_cost=1,
                            option2_available=is_overlength,
                            option2_cost=0,
                            line_positions=line_positions,
                            cost=1,  # Deprecated, kept for backwards compatibility
                        )
                    )

    return targets


def enumerate_expanded_recovery_targets(
    state: GameState,
    player: int,
) -> List[ExpandedRecoveryTarget]:
    """
    Enumerate all valid recovery targets using the expanded criteria (RR-CANON-R112).

    A recovery slide is legal if ANY of these conditions are satisfied:
    (a) Line formation: Completes a line of at least lineLength markers
    (b) Fallback repositioning: If no slide satisfies (a), any slide is legal

    Note: Territory disconnection is NOT a recovery criterion. Territory may be
    disconnected as a side effect of line formation, but cannot be the primary
    reason for a recovery slide.

    Args:
        state: Current game state
        player: Player to enumerate recovery moves for

    Returns:
        List of expanded recovery targets
    """
    targets: List[ExpandedRecoveryTarget] = []
    board = state.board
    line_length = get_effective_line_length(board.type, len(state.players))
    buried_ring_count = count_buried_rings(board, player)

    # Stack-strike recovery is enabled by default (v1 rules)
    # Set RINGRIFT_RECOVERY_STACK_STRIKE_V1=0 to disable
    stack_strike_enabled = os.getenv("RINGRIFT_RECOVERY_STACK_STRIKE_V1", "1").lower() in {
        "1",
        "true",
    }

    # Track all valid slide destinations for fallback
    all_valid_slides: List[Tuple[Position, Position]] = []
    all_valid_stack_strikes: List[Tuple[Position, Position]] = []

    # Find all markers owned by the player
    player_marker_positions: List[Position] = []
    for pos_key, marker in board.markers.items():
        if marker.player == player:
            player_marker_positions.append(marker.position)

    # Check each marker for line recovery
    has_line_recovery = False

    for marker_pos in player_marker_positions:
        directions = _get_moore_directions(board.type)

        for direction in directions:
            to_pos = _add_direction(marker_pos, direction, 1)

            # Shared destination validity checks (for empty fallback vs stack-strike).
            if not BoardManager.is_valid_position(to_pos, board.type, board.size):
                continue
            if BoardManager.is_collapsed_space(to_pos, board):
                continue
            to_key = to_pos.to_key()
            if to_key in board.markers:
                continue

            dest_stack = BoardManager.get_stack(to_pos, board)
            if dest_stack is not None:
                if stack_strike_enabled:
                    all_valid_stack_strikes.append((marker_pos, to_pos))
                continue

            if not _can_marker_slide_to(board, marker_pos, to_pos, player):
                continue

            # Track for potential fallback
            all_valid_slides.append((marker_pos, to_pos))

            # Check for line formation (condition a)
            original_marker = board.markers.get(marker_pos.to_key())
            del board.markers[marker_pos.to_key()]
            board.markers[to_pos.to_key()] = MarkerInfo(
                position=to_pos,
                player=player,
                type="regular",
            )

            completes, markers_count, line_positions = _would_complete_line_at(
                board, player, to_pos, line_length
            )

            # Restore marker
            del board.markers[to_pos.to_key()]
            board.markers[marker_pos.to_key()] = original_marker

            if completes:
                is_overlength = markers_count > line_length
                # For exact-length: need 1 buried ring (Option 1 only)
                # For overlength: Option 2 is free, so always legal
                can_use_option1 = buried_ring_count >= 1
                can_use_option2 = is_overlength

                if can_use_option1 or can_use_option2:
                    min_cost = 0 if can_use_option2 else 1
                    targets.append(
                        ExpandedRecoveryTarget(
                            from_pos=marker_pos,
                            to_pos=to_pos,
                            recovery_mode="line",
                            markers_in_line=markers_count,
                            is_overlength=is_overlength,
                            line_positions=line_positions,
                            min_cost=min_cost,
                        )
                    )
                    has_line_recovery = True

    # If no line recovery options exist, add fallback options (condition b)
    # Per RR-CANON-R112(b): Fallback slides now ALLOW territory disconnection
    if not has_line_recovery and buried_ring_count >= 1:
        for from_pos, to_pos in all_valid_slides:
            # All adjacent slides are permitted for fallback (including territory disconnect)
            targets.append(
                ExpandedRecoveryTarget(
                    from_pos=from_pos,
                    to_pos=to_pos,
                    recovery_mode="fallback",
                    min_cost=1,
                )
            )
        for from_pos, to_pos in all_valid_stack_strikes:
            targets.append(
                ExpandedRecoveryTarget(
                    from_pos=from_pos,
                    to_pos=to_pos,
                    recovery_mode="stack_strike",
                    min_cost=1,
                )
            )

    return targets


def has_any_recovery_move(state: GameState, player: int) -> bool:
    """
    Check if a player has any valid recovery moves.

    This is an optimized check that can short-circuit early.

    Uses the expanded recovery criteria (RR-CANON-R112):
    (a) Line formation - completes a line of lineLength markers
    (b) Fallback repositioning - if no line available, any adjacent slide is permitted
        (including slides that cause territory disconnection)

    Args:
        state: Current game state
        player: Player to check

    Returns:
        True if the player has at least one valid recovery move
    """
    # First check eligibility (cheap check)
    if not is_eligible_for_recovery(state, player):
        return False

    buried_ring_count = count_buried_rings(state.board, player)
    if buried_ring_count < 1:
        return False

    board = state.board
    line_length = get_effective_line_length(board.type, len(state.players))
    # Stack-strike recovery is enabled by default (v1 rules)
    # Set RINGRIFT_RECOVERY_STACK_STRIKE_V1=0 to disable
    stack_strike_enabled = os.getenv("RINGRIFT_RECOVERY_STACK_STRIKE_V1", "1").lower() in {
        "1",
        "true",
    }
    valid_fallback_exists = False

    # Check for line recovery or valid fallback
    # Use list() to create a snapshot - we modify markers dict during iteration
    for pos_key, marker in list(board.markers.items()):
        if marker.player != player:
            continue

        directions = _get_moore_directions(board.type)
        for direction in directions:
            to_pos = _add_direction(marker.position, direction, 1)
            # Experimental stack-strike recovery counts as a fallback-class move.
            if (
                stack_strike_enabled
                and BoardManager.is_valid_position(to_pos, board.type, board.size)
                and not BoardManager.is_collapsed_space(to_pos, board)
                and to_pos.to_key() not in board.markers
                and BoardManager.get_stack(to_pos, board) is not None
            ):
                valid_fallback_exists = True
                continue

            if not _can_marker_slide_to(board, marker.position, to_pos, player):
                continue

            # Check for line formation
            original_marker = board.markers.get(marker.position.to_key())
            del board.markers[marker.position.to_key()]
            board.markers[to_pos.to_key()] = MarkerInfo(
                position=to_pos,
                player=player,
                type="regular",
            )

            completes, _, _ = _would_complete_line_at(board, player, to_pos, line_length)

            # Restore marker
            del board.markers[to_pos.to_key()]
            board.markers[marker.position.to_key()] = original_marker

            if completes:
                return True  # Line recovery found

            # Any valid adjacent slide is a valid fallback (per RR-CANON-R112(b))
            # Territory disconnection is now ALLOWED for fallback recovery
            valid_fallback_exists = True

    # If no line recovery, but a valid fallback exists
    return valid_fallback_exists


# ═══════════════════════════════════════════════════════════════════════════
# RECOVERY MOVE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════


def validate_recovery_slide(
    state: GameState,
    move: Move,
) -> RecoveryValidationResult:
    """
    Validate a recovery slide move.

    Checks:
    1. Player is eligible for recovery
    2. From position has a marker owned by the player
    3. To position is adjacent and empty
    4. Move completes a line of at least lineLength markers
    5. Player has enough buried rings to pay the cost

    Args:
        state: Current game state
        move: The recovery slide move to validate

    Returns:
        RecoveryValidationResult with validity and details
    """
    player = move.player
    board = state.board

    # Check eligibility
    if not is_eligible_for_recovery(state, player):
        return RecoveryValidationResult(
            valid=False,
            reason="Player is not eligible for recovery",
        )

    # Check phase
    if state.current_phase != GamePhase.MOVEMENT:
        return RecoveryValidationResult(
            valid=False,
            reason="Recovery slide only valid in movement phase",
        )

    # Check from_pos has player's marker
    if move.from_pos is None:
        return RecoveryValidationResult(
            valid=False,
            reason="Recovery slide requires from position",
        )

    from_key = move.from_pos.to_key()
    from_marker = board.markers.get(from_key)
    if from_marker is None or from_marker.player != player:
        return RecoveryValidationResult(
            valid=False,
            reason="No marker owned by player at from position",
        )

    # Check to position
    if move.to is None:
        return RecoveryValidationResult(
            valid=False,
            reason="Recovery slide requires to position",
        )
    recovery_mode = getattr(move, "recovery_mode", None)
    # Stack-strike recovery is enabled by default (v1 rules)
    # Set RINGRIFT_RECOVERY_STACK_STRIKE_V1=0 to disable
    stack_strike_enabled = os.getenv("RINGRIFT_RECOVERY_STACK_STRIKE_V1", "1").lower() in {
        "1",
        "true",
    }

    # Fallback-class recovery (fallback or stack_strike) bypasses line checks.
    if recovery_mode in {"fallback", "stack_strike"}:
        line_targets = enumerate_recovery_slide_targets(state, player)
        if line_targets:
            return RecoveryValidationResult(
                valid=False,
                reason="Fallback-class recovery is only allowed when no line-forming recovery exists",
            )

        buried_rings = count_buried_rings(board, player)
        if buried_rings < 1:
            return RecoveryValidationResult(
                valid=False,
                reason="Fallback-class recovery requires at least 1 buried ring",
            )

        if recovery_mode == "fallback":
            if not _can_marker_slide_to(board, move.from_pos, move.to, player):
                return RecoveryValidationResult(
                    valid=False,
                    reason="Invalid fallback destination (not adjacent or occupied)",
                )
        else:
            if not stack_strike_enabled:
                return RecoveryValidationResult(
                    valid=False,
                    reason="Stack-strike recovery is not enabled",
                )
            if not _is_adjacent(board.type, move.from_pos, move.to):
                return RecoveryValidationResult(
                    valid=False,
                    reason="Destination is not adjacent",
                )
            if not BoardManager.is_valid_position(move.to, board.type, board.size):
                return RecoveryValidationResult(
                    valid=False,
                    reason="Invalid destination position",
                )
            if BoardManager.is_collapsed_space(move.to, board):
                return RecoveryValidationResult(
                    valid=False,
                    reason="Destination is collapsed space",
                )
            if move.to.to_key() in board.markers:
                return RecoveryValidationResult(
                    valid=False,
                    reason="Destination has a marker",
                )
            if BoardManager.get_stack(move.to, board) is None:
                return RecoveryValidationResult(
                    valid=False,
                    reason="Stack-strike recovery requires a destination stack",
                )

        return RecoveryValidationResult(
            valid=True,
            line_positions=[],
            is_overlength=False,
            option_used=None,
            cost=1,
        )

    if not _can_marker_slide_to(board, move.from_pos, move.to, player):
        return RecoveryValidationResult(
            valid=False,
            reason="Invalid slide destination (not adjacent or occupied)",
        )

    # Temporarily move marker and check line
    del board.markers[from_key]
    to_key = move.to.to_key()
    board.markers[to_key] = MarkerInfo(
        position=move.to,
        player=player,
        type="regular",
    )

    line_length = get_effective_line_length(board.type, len(state.players))
    completes, markers_count, line_positions = _would_complete_line_at(
        board,
        player,
        move.to,
        line_length,
    )

    # Restore marker for line-based validation
    del board.markers[to_key]
    board.markers[from_key] = from_marker

    if not completes:
        return RecoveryValidationResult(
            valid=False,
            reason=f"Move does not complete a line (need {line_length} markers)",
        )

    # Option 1/2 cost model (RR-CANON-R113)
    is_overlength = markers_count > line_length
    buried_rings = count_buried_rings(board, player)

    # Determine requested option if provided
    requested_option = getattr(move, "recovery_option", None)
    option_used: RecoveryOption
    cost: int

    if requested_option is not None:
        if requested_option == 2 and not is_overlength:
            return RecoveryValidationResult(
                valid=False,
                reason="Option 2 is only available for overlength lines",
            )
        if requested_option == 1 and buried_rings < 1:
            return RecoveryValidationResult(
                valid=False,
                reason="Option 1 requires 1 buried ring",
            )
        option_used = requested_option  # type: ignore[assignment]
        cost = 0 if option_used == 2 else 1
    else:
        # Option 2 is free but only available for overlength lines
        # Option 1 costs 1 buried ring
        can_use_option1 = buried_rings >= 1
        can_use_option2 = is_overlength  # Free, only for overlength

        if not can_use_option1 and not can_use_option2:
            return RecoveryValidationResult(
                valid=False,
                reason="Insufficient buried rings: need 1, have 0 (and no free Option 2 available)",
            )

        # Determine which option to use (prefer free Option 2 when available)
        if can_use_option2:
            option_used = 2
            cost = 0
        else:
            option_used = 1
            cost = 1

    # Validate collapse positions for Option 2 when provided
    if option_used == 2:
        collapse_positions = getattr(move, "collapse_positions", None)
        if collapse_positions is None:
            return RecoveryValidationResult(
                valid=False,
                reason="Option 2 requires collapse_positions",
            )
        if len(collapse_positions) != line_length:
            return RecoveryValidationResult(
                valid=False,
                reason=f"Option 2 requires exactly {line_length} collapse positions",
            )
        # Ensure all collapse positions are drawn from the formed line and include destination
        line_keys = {p.to_key() for p in line_positions}
        dest_key = move.to.to_key()
        for pos in collapse_positions:
            if pos.to_key() not in line_keys:
                return RecoveryValidationResult(
                    valid=False,
                    reason="Collapse positions must be part of the formed line",
                )
        if dest_key not in {p.to_key() for p in collapse_positions}:
            return RecoveryValidationResult(
                valid=False,
                reason="Collapse positions must include the destination",
            )

    return RecoveryValidationResult(
        valid=True,
        markers_in_line=markers_count,
        is_overlength=is_overlength,
        option_used=option_used,
        cost=cost,
    )


# ═══════════════════════════════════════════════════════════════════════════
# RECOVERY MOVE APPLICATION
# ═══════════════════════════════════════════════════════════════════════════


def apply_recovery_slide(
    state: GameState,
    move: Move,
    option: Optional[RecoveryOption] = None,
    collapse_positions: Optional[List[Position]] = None,
) -> RecoveryApplicationOutcome:
    """
    Apply a recovery slide move to the game state (mutates in place).

    Steps:
    1. Move the marker from from_pos to to_pos
    2. Find the completed line
    3. Collapse markers based on option:
       - Option 1: Collapse all markers in the line
       - Option 2: Collapse only the specified lineLength consecutive markers
    4. Extract buried rings as self-elimination cost (Option 1 = 1 ring, Option 2 = 0)

    Cost Model (Option 1 / Option 2):
    - Option 1 (collapse all): 1 buried ring extraction
    - Option 2 (collapse lineLength, overlength only): 0 (free)

    Args:
        state: Current game state (will be mutated)
        move: The recovery slide move to apply
        option: Which option to use (1 or 2). For overlength, required.
                For exact-length, defaults to Option 1.
        collapse_positions: For Option 2, which positions to collapse.
                           Must be exactly lineLength consecutive positions.

    Returns:
        RecoveryApplicationOutcome with details
    """
    # Validate first
    validation = validate_recovery_slide(state, move)
    if not validation.valid:
        return RecoveryApplicationOutcome(
            success=False,
            error=validation.reason,
        )

    player = move.player
    board = state.board

    from_key = move.from_pos.to_key()
    to_key = move.to.to_key()
    recovery_mode = getattr(move, "recovery_mode", None)

    if recovery_mode == "stack_strike":
        # Experimental stack-strike recovery (v1): sacrifice marker to strike adjacent stack.
        del board.markers[from_key]

        attacked = board.stacks.get(to_key)
        if not attacked or attacked.stack_height == 0:
            return RecoveryApplicationOutcome(success=False, error="No stack to strike")

        attacked.rings.pop()  # rings stored bottom->top; pop removes top
        attacked.stack_height -= 1

        # Credit elimination to recovering player.
        player_id_str = str(player)
        board.eliminated_rings[player_id_str] = board.eliminated_rings.get(player_id_str, 0) + 1
        state.total_rings_eliminated += 1
        for p in state.players:
            if p.player_number == player:
                p.eliminated_rings += 1
                break

        if attacked.stack_height == 0:
            del board.stacks[to_key]
        else:
            attacked.controlling_player = attacked.rings[-1]
            cap_height = 0
            for r in reversed(attacked.rings):
                if r == attacked.controlling_player:
                    cap_height += 1
                else:
                    break
            attacked.cap_height = cap_height

        cost = 1
        collapsed_positions = []
        effective_option = None
        line_positions = []
    else:
        # Move the marker
        del board.markers[from_key]
        board.markers[to_key] = MarkerInfo(
            position=move.to,
            player=player,
            type="regular",
        )

        if recovery_mode == "fallback":
            # Fallback mode: no line collapse, just marker move + ring extraction
            cost = 1
            collapsed_positions = []
            effective_option = None
            line_positions = []
        else:
            # Line mode: find and collapse the completed line
            line_length = get_effective_line_length(board.type, len(state.players))
            _, markers_count, line_positions = _would_complete_line_at(
                board, player, move.to, line_length
            )

            is_overlength = markers_count > line_length

            if option is not None:
                effective_option = option
            elif getattr(move, "recovery_option", None) is not None:
                effective_option = getattr(move, "recovery_option")
            elif is_overlength:
                effective_option = 2
            else:
                effective_option = 1

            if effective_option == 2 and collapse_positions is not None:
                positions_to_collapse = collapse_positions
            elif effective_option == 2 and is_overlength:
                positions_to_collapse = line_positions[:line_length]
            else:
                positions_to_collapse = line_positions

            collapsed_positions = list(positions_to_collapse)
            for pos in collapsed_positions:
                key = pos.to_key()
                if key in board.markers:
                    del board.markers[key]
                board.collapsed_spaces[key] = player

            for p in state.players:
                if p.player_number == player:
                    p.territory_spaces = getattr(p, "territory_spaces", 0) + len(
                        collapsed_positions
                    )
                    break

            cost = calculate_recovery_cost(effective_option)

    # Extract buried rings to pay cost
    # Per RR-CANON-R113: Extract the bottommost ring from chosen stack(s)
    rings_extracted = 0

    if cost > 0:
        # Get extraction stacks from move, or auto-select if not provided
        extraction_stacks = getattr(move, "extraction_stacks", None)
        if not extraction_stacks:
            # Auto-select: find eligible stacks and pick the first one(s)
            eligible = enumerate_eligible_extraction_stacks(board, player)
            extraction_stacks = tuple(es.position_key for es in eligible[:cost])

        # Extract from each specified stack
        for stack_key in extraction_stacks:
            if rings_extracted >= cost:
                break

            stack = board.stacks.get(stack_key)
            if not stack:
                continue

            # Find player's bottommost ring (first occurrence = bottommost)
            try:
                bottom_index = stack.rings.index(player)
            except ValueError:
                continue  # Player has no ring in this stack

            # Extract (remove) the bottommost ring
            stack.rings.pop(bottom_index)
            stack.stack_height = len(stack.rings)
            rings_extracted += 1

            # Update stack state after extraction
            if stack.rings:
                # Recalculate cap height and controlling player
                cap_height = 0
                controlling = stack.rings[-1]
                for r in reversed(stack.rings):
                    if r == controlling:
                        cap_height += 1
                    else:
                        break
                stack.cap_height = cap_height
                stack.controlling_player = controlling
            else:
                # Stack is empty, remove it
                del board.stacks[stack_key]

    # Extracted rings are eliminated (self-elimination cost)
    # Per RR-CANON-R113: Extracted ring is credited as self-eliminated, NOT returned
    # to hand. This is the mandatory cost for recovery actions.
    if rings_extracted > 0:
        for p in state.players:
            if p.player_number == player:
                p.eliminated_rings = getattr(p, "eliminated_rings", 0) + rings_extracted
                # NOTE: Do NOT add to rings_in_hand - extracted rings are eliminated,
                # not returned. This was a bug that caused ring count divergence vs TS.
                break

    return RecoveryApplicationOutcome(
        success=True,
        option_used=effective_option,
        rings_extracted=rings_extracted,
        line_positions=line_positions,
        collapsed_positions=collapsed_positions,
    )


# ═══════════════════════════════════════════════════════════════════════════
# MOVE GENERATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def get_recovery_moves(state: GameState, player: int) -> List[Move]:
    """
    Get all valid recovery slide moves for a player.

    This is the main entry point for recovery move generation, used by
    GameEngine.get_valid_moves.

    Args:
        state: Current game state
        player: Player to generate moves for

    Returns:
        List of recovery slide moves
    """
    # Check eligibility first
    if not is_eligible_for_recovery(state, player):
        return []

    targets = enumerate_recovery_slide_targets(state, player)
    moves: List[Move] = []

    for target in targets:
        move_number = len(state.move_history) + 1
        base_kwargs = dict(
            id=f"recovery-{target.from_pos.to_key()}-{target.to_pos.to_key()}-{move_number}",
            type=MoveType.RECOVERY_SLIDE,
            player=player,
            from_pos=target.from_pos,
            to=target.to_pos,
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=move_number,
        )

        # Overlength: generate both options
        if target.is_overlength:
            # Option 2 (free) with default collapse subset (first lineLength)
            moves.append(
                Move(
                    recoveryOption=2,
                    collapsePositions=tuple(
                        target.line_positions[: get_effective_line_length(state.board.type, len(state.players))]
                    ),
                    **base_kwargs,
                )
            )
            # Option 1 (cost 1)
            moves.append(
                Move(
                    recoveryOption=1,
                    **base_kwargs,
                )
            )
        else:
            # Exact length: only Option 1
            moves.append(
                Move(
                    recoveryOption=1,
                    **base_kwargs,
                )
            )

    return moves


def get_expanded_recovery_moves(state: GameState, player: int) -> List[Move]:
    """
    Get all valid recovery moves using the expanded criteria (RR-CANON-R112).

    This includes:
    - Line recovery moves (condition a) - forms a line of lineLength markers
    - Fallback repositioning moves (condition b) - if no line available
    - Skip recovery option (pass turn without action)

    Note: Territory disconnection is NOT a recovery criterion. Territory may be
    disconnected as a side effect of line formation, but cannot be the primary
    reason for a recovery slide.

    Args:
        state: Current game state
        player: Player to generate moves for

    Returns:
        List of recovery moves including all modes and skip option
    """
    # Check eligibility first
    if not is_eligible_for_recovery(state, player):
        return []

    targets = enumerate_expanded_recovery_targets(state, player)
    moves: List[Move] = []
    move_number = len(state.move_history) + 1

    for target in targets:
        base_kwargs = dict(
            id=f"recovery-{target.recovery_mode}-{target.from_pos.to_key()}-{target.to_pos.to_key()}-{move_number}",
            type=MoveType.RECOVERY_SLIDE,
            player=player,
            from_pos=target.from_pos,
            to=target.to_pos,
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=move_number,
        )

        if target.recovery_mode == "line":
            # Line recovery: generate Option 1 and/or Option 2
            if target.is_overlength:
                # Option 2 (free) with default collapse subset
                moves.append(
                    Move(
                        recoveryOption=2,
                        recoveryMode="line",
                        collapsePositions=tuple(
                            target.line_positions[
                                : get_effective_line_length(state.board.type, len(state.players))
                            ]
                        ),
                        **base_kwargs,
                    )
                )
                # Option 1 (cost 1)
                moves.append(
                    Move(
                        recoveryOption=1,
                        recoveryMode="line",
                        **base_kwargs,
                    )
                )
            else:
                # Exact length: only Option 1
                moves.append(
                    Move(
                        recoveryOption=1,
                        recoveryMode="line",
                        **base_kwargs,
                    )
                )

        elif target.recovery_mode == "fallback":
            # Fallback repositioning: costs 1 buried ring (per RR-CANON-R112(b))
            # Must provide extraction_stacks for TS parity
            eligible = enumerate_eligible_extraction_stacks(state.board, player)
            if eligible:
                # Use the first eligible stack for extraction
                extraction_stack_key = eligible[0].position_key
                moves.append(
                    Move(
                        recoveryMode="fallback",
                        extraction_stacks=(extraction_stack_key,),
                        **base_kwargs,
                    )
                )
        elif target.recovery_mode == "stack_strike":
            eligible = enumerate_eligible_extraction_stacks(state.board, player)
            if eligible:
                extraction_stack_key = eligible[0].position_key
                moves.append(
                    Move(
                        recoveryMode="stack_strike",
                        extraction_stacks=(extraction_stack_key,),
                        **base_kwargs,
                    )
                )

    # Always add skip recovery option if player is eligible
    # Skip allows preserving buried rings for later
    moves.append(
        Move(
            id=f"skip-recovery-{move_number}",
            type=MoveType.SKIP_RECOVERY,
            player=player,
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=move_number,
        )
    )

    return moves

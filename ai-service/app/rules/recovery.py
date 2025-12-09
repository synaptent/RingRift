"""
Recovery Action Implementation for RingRift AI Service.

This module implements the recovery action (RR-CANON-R110–R115) which allows
temporarily eliminated players (no stacks, no rings in hand, but with markers
and buried rings) to slide a marker to complete a line.

Mirrors src/shared/engine/aggregates/RecoveryAggregate.ts

**SSoT Policy:**
The canonical rules defined in RULES_CANONICAL_SPEC.md are the ultimate
authority. The TS shared engine (src/shared/engine/**) is the primary
executable derivation. This Python module must mirror the canonical rules
and TS implementation. If this code disagrees with either, this code must
be updated—never the other way around.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
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
    RingStack,
)
from app.board_manager import BoardManager
from app.rules.core import (
    count_buried_rings,
    player_has_markers,
    is_eligible_for_recovery,
    get_effective_line_length,
)


# ═══════════════════════════════════════════════════════════════════════════
# RECOVERY TYPES
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class RecoverySlideTarget:
    """A valid target position for a recovery slide move."""
    from_pos: Position
    to_pos: Position
    markers_in_line: int  # number of markers (including the sliding marker) that form the line
    cost: int  # number of buried rings extracted (1 + excess)


@dataclass
class RecoveryValidationResult:
    """Result of validating a recovery slide move."""
    valid: bool
    reason: Optional[str] = None
    markers_in_line: int = 0
    cost: int = 0


@dataclass
class RecoveryApplicationOutcome:
    """Outcome of applying a recovery slide."""
    success: bool
    error: Optional[str] = None
    rings_extracted: int = 0
    line_positions: List[Position] = None

    def __post_init__(self):
        if self.line_positions is None:
            self.line_positions = []


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _get_moore_directions(board_type: BoardType) -> List[Tuple[int, int, Optional[int]]]:
    """
    Get Moore neighborhood directions (8 directions for square boards,
    6 axial directions for hex).
    """
    if board_type == BoardType.HEXAGONAL:
        return [
            (1, 0, -1), (0, 1, -1), (-1, 1, 0),
            (-1, 0, 1), (0, -1, 1), (1, -1, 0)
        ]
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


# ═══════════════════════════════════════════════════════════════════════════
# RECOVERY COST CALCULATION
# ═══════════════════════════════════════════════════════════════════════════


def calculate_recovery_cost(
    board: BoardState,
    player: int,
    markers_in_line: int,
) -> int:
    """
    Calculate the recovery cost (number of buried rings to extract).

    Cost = 1 + max(0, markers_in_line - line_length)

    Per RR-CANON-R113:
    - Base extraction is 1 ring
    - Each marker beyond lineLength costs an additional ring

    Args:
        board: Current board state
        player: Player performing recovery
        markers_in_line: Number of markers in the line (including the sliding marker)

    Returns:
        Number of buried rings to extract
    """
    line_length = get_effective_line_length(board.type, 3)  # num_players not used for line length
    excess = max(0, markers_in_line - line_length)
    return 1 + excess


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
            completes, markers_count, _ = _would_complete_line_at(
                board, player, to_pos, line_length
            )

            # Restore the marker
            del board.markers[to_pos.to_key()]
            board.markers[marker_pos.to_key()] = original_marker

            if completes:
                cost = calculate_recovery_cost(board, player, markers_count)
                targets.append(RecoverySlideTarget(
                    from_pos=marker_pos,
                    to_pos=to_pos,
                    markers_in_line=markers_count,
                    cost=cost,
                ))

    return targets


def has_any_recovery_move(state: GameState, player: int) -> bool:
    """
    Check if a player has any valid recovery moves.

    This is an optimized check that can short-circuit early.

    Args:
        state: Current game state
        player: Player to check

    Returns:
        True if the player has at least one valid recovery move
    """
    # First check eligibility (cheap check)
    if not is_eligible_for_recovery(state, player):
        return False

    # Then enumerate (expensive) - if any targets exist, return True
    targets = enumerate_recovery_slide_targets(state, player)
    return len(targets) > 0


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
    completes, markers_count, _ = _would_complete_line_at(
        board, player, move.to, line_length
    )

    # Restore marker
    del board.markers[to_key]
    board.markers[from_key] = from_marker

    if not completes:
        return RecoveryValidationResult(
            valid=False,
            reason=f"Move does not complete a line (need {line_length} markers)",
        )

    cost = calculate_recovery_cost(board, player, markers_count)
    buried_rings = count_buried_rings(board, player)

    if buried_rings < cost:
        return RecoveryValidationResult(
            valid=False,
            reason=f"Insufficient buried rings: need {cost}, have {buried_rings}",
        )

    return RecoveryValidationResult(
        valid=True,
        markers_in_line=markers_count,
        cost=cost,
    )


# ═══════════════════════════════════════════════════════════════════════════
# RECOVERY MOVE APPLICATION
# ═══════════════════════════════════════════════════════════════════════════


def apply_recovery_slide(
    state: GameState,
    move: Move,
) -> RecoveryApplicationOutcome:
    """
    Apply a recovery slide move to the game state (mutates in place).

    Steps:
    1. Move the marker from from_pos to to_pos
    2. Find the completed line
    3. Extract buried rings (cost = 1 + excess)
    4. Return extracted rings to player's hand

    Args:
        state: Current game state (will be mutated)
        move: The recovery slide move to apply

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

    # Move the marker
    from_key = move.from_pos.to_key()
    to_key = move.to.to_key()

    del board.markers[from_key]
    board.markers[to_key] = MarkerInfo(
        position=move.to,
        player=player,
        type="regular",
    )

    # Find the completed line
    line_length = get_effective_line_length(board.type, len(state.players))
    _, markers_count, line_positions = _would_complete_line_at(
        board, player, move.to, line_length
    )

    # Calculate cost and extract rings
    cost = calculate_recovery_cost(board, player, markers_count)

    # Find stacks with buried rings and extract them
    rings_extracted = 0
    for stack_key, stack in list(board.stacks.items()):
        if stack.controlling_player == player:
            continue  # Skip player's own stacks

        # Count and extract this player's rings from this stack
        rings_to_remove = []
        for i, ring in enumerate(stack.rings):
            if ring == player and rings_extracted < cost:
                rings_to_remove.append(i)
                rings_extracted += 1

        # Remove rings from bottom to top (to maintain indices)
        for i in reversed(rings_to_remove):
            stack.rings.pop(i)

        # Update stack height
        stack.stack_height = len(stack.rings)

        # Recalculate cap height
        if stack.rings:
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

        if rings_extracted >= cost:
            break

    # Return extracted rings to player's hand
    for p in state.players:
        if p.player_number == player:
            p.rings_in_hand += rings_extracted
            break

    return RecoveryApplicationOutcome(
        success=True,
        rings_extracted=rings_extracted,
        line_positions=line_positions,
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
        move = Move(
            id=f"recovery-{target.from_pos.to_key()}-{target.to_pos.to_key()}-{move_number}",
            type=MoveType.RECOVERY_SLIDE,
            player=player,
            from_pos=target.from_pos,
            to=target.to_pos,
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=move_number,
        )
        moves.append(move)

    return moves

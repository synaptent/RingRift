from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from app.models import (
    GameState,
    Move,
    Position,
    BoardState,
    BoardType,
    MoveType,
)
from app.board_manager import BoardManager


"""
Canonical placement helpers for the Python rules engine.

Python analogue of the shared TS PlacementAggregate / placementHelpers
modules. This module centralises ring placement validation,
enumeration, skip-placement eligibility, and application helpers
so that mutators/validators and higher-level engines can delegate
to a single canonical implementation.

Semantics are aligned with the P2.1 placement spec and the TS
PlacementAggregate, but implemented in terms of the existing
GameEngine helpers to avoid duplicating low-level movement or
capture geometry.
"""


@dataclass
class PlacementContextPy:
    """
    Context required to validate a prospective ring placement.

    Mirrors the TS PlacementContext type from PlacementAggregate.
    """

    board_type: BoardType
    player: int
    rings_in_hand: int
    rings_per_player_cap: int
    # Optional precomputed values for efficiency.
    rings_on_board: Optional[int] = None
    max_available_global: Optional[int] = None


@dataclass
class PlacementValidationResultPy:
    """
    Result of validating a placement at a specific cell.
    """

    valid: bool
    max_placement_count: int = 0
    error_code: Optional[str] = None
    message: Optional[str] = None


@dataclass
class PlacementApplicationOutcomePy:
    """
    Result of applying a canonical PLACE_RING move.
    """

    next_state: GameState
    applied_count: int
    placed_on_stack: bool


@dataclass
class SkipPlacementEligibilityResultPy:
    """
    Result of evaluating whether skip-placement is legal.
    """

    eligible: bool
    reason: Optional[str] = None
    code: Optional[str] = None


def _count_rings_on_board_for_player(board: BoardState, player: int) -> int:
    """
    Count rings of `player` that are currently on the board.

    This is the Python analogue of the TS core.countRingsOnBoardForPlayer
    helper used by PlacementAggregate.
    """
    total = 0
    for stack in board.stacks.values():
        for owner in stack.rings:
            if owner == player:
                total += 1
    return total


def validate_placement_on_board_py(
    board: BoardState,
    to: Position,
    requested_count: int,
    ctx: PlacementContextPy,
) -> PlacementValidationResultPy:
    """
    Canonical, board-local validator for ring placement.

    Responsibilities (mirrors TS validatePlacementOnBoard):

    - Enforce board geometry / collapsed-space / marker-stack exclusivity.
    - Enforce per-player ring cap and rings-in-hand supply.
    - Enforce per-cell caps:
      * Existing stacks: at most 1 ring per action.
      * Empty cells: up to 3 rings per action (subject to capacity).
    - Enforce no-dead-placement via hypothetical placement +
      hasAnyLegalMoveOrCaptureFromOnBoard semantics.

    The helper is phase-agnostic; callers are responsible for ensuring it
    is only used for the active player during the ring-placement phase.
    """
    from app.game_engine import GameEngine

    # 1. Basic supply check.
    if ctx.rings_in_hand <= 0:
        return PlacementValidationResultPy(
            valid=False,
            max_placement_count=0,
            error_code="INSUFFICIENT_RINGS",
            message="No rings in hand to place",
        )

    # 2. Position validity.
    if not BoardManager.is_valid_position(to, ctx.board_type, board.size):
        return PlacementValidationResultPy(
            valid=False,
            max_placement_count=0,
            error_code="INVALID_POSITION",
            message="Position off board",
        )

    pos_key = to.to_key()

    # 3. Collapsed spaces are never legal for placement.
    if pos_key in board.collapsed_spaces:
        return PlacementValidationResultPy(
            valid=False,
            max_placement_count=0,
            error_code="COLLAPSED_SPACE",
            message="Cannot place on collapsed space",
        )

    # 4. Stack/marker exclusivity: cannot place on a marker.
    if pos_key in board.markers:
        return PlacementValidationResultPy(
            valid=False,
            max_placement_count=0,
            error_code="MARKER_BLOCKED",
            message="Cannot place on a marker",
        )

    existing_stack = board.stacks.get(pos_key)
    is_occupied = bool(existing_stack and existing_stack.stack_height > 0)

    # 5. Global capacity and supply.
    if ctx.rings_on_board is not None:
        rings_on_board = ctx.rings_on_board
    else:
        rings_on_board = _count_rings_on_board_for_player(board, ctx.player)

    remaining_by_cap = ctx.rings_per_player_cap - rings_on_board
    remaining_by_supply = ctx.rings_in_hand

    if ctx.max_available_global is not None:
        max_available_global = ctx.max_available_global
    else:
        max_available_global = min(remaining_by_cap, remaining_by_supply)

    if max_available_global <= 0:
        return PlacementValidationResultPy(
            valid=False,
            max_placement_count=0,
            error_code="NO_RINGS_AVAILABLE",
            message=(
                "No rings available to place (cap reached or no rings "
                "in hand)"
            ),
        )

    # 6. Per-cell cap: existing stacks → 1, empty cells → up to 3.
    per_cell_cap = 1 if is_occupied else 3
    max_placement_count = min(per_cell_cap, max_available_global)

    if max_placement_count <= 0:
        return PlacementValidationResultPy(
            valid=False,
            max_placement_count=0,
            error_code="NO_RINGS_AVAILABLE",
            message="No rings available to place at this position",
        )

    # 7. Requested count must lie within [1, max_placement_count].
    count = requested_count
    if count < 1 or count > max_placement_count:
        message = (
            "Can only place 1 ring on an existing stack"
            if is_occupied
            else "Must place between 1 and 3 rings on an empty space"
        )
        return PlacementValidationResultPy(
            valid=False,
            max_placement_count=max_placement_count,
            error_code="INVALID_COUNT",
            message=message,
        )

    # 8. No-dead-placement: resulting stack must have at least one legal
    #    movement or capture available.
    hyp_board = GameEngine._create_hypothetical_board_with_placement(
        board,
        to,
        ctx.player,
        count,
    )

    has_legal_move = GameEngine._has_any_legal_move_or_capture_from_on_board(
        ctx.board_type,
        to,
        ctx.player,
        hyp_board,
    )

    if not has_legal_move:
        return PlacementValidationResultPy(
            valid=False,
            max_placement_count=max_placement_count,
            error_code="NO_LEGAL_MOVES",
            message=(
                "Placement would result in a stack with no legal "
                "moves or captures"
            ),
        )

    return PlacementValidationResultPy(
        valid=True,
        max_placement_count=max_placement_count,
        error_code=None,
        message=None,
    )


def enumerate_placement_moves_py(state: GameState, player: int) -> List[Move]:
    """
    Enumerate legal PLACE_RING moves for `player` in `state`.

    Thin wrapper around GameEngine._get_ring_placement_moves so that
    callers can rely on a stable surface mirroring the TS
    PlacementAggregate.enumeratePlacementPositions behaviour while
    reusing the existing Python engine implementation.
    """
    from app.game_engine import GameEngine

    return GameEngine._get_ring_placement_moves(state, player)


def evaluate_skip_placement_eligibility_py(
    state: GameState,
    player: int,
) -> SkipPlacementEligibilityResultPy:
    """
    Evaluate canonical skip-placement eligibility for `player` in `state`.

    Mirrors TS PlacementAggregate.evaluateSkipPlacementEligibility.
    """
    from app.models import GamePhase

    # Phase check.
    if state.current_phase != GamePhase.RING_PLACEMENT:
        return SkipPlacementEligibilityResultPy(
            eligible=False,
            reason="Not in ring placement phase",
            code="INVALID_PHASE",
        )

    # Turn check.
    if player != state.current_player:
        return SkipPlacementEligibilityResultPy(
            eligible=False,
            reason="Not your turn",
            code="NOT_YOUR_TURN",
        )

    player_obj = next(
        (p for p in state.players if p.player_number == player),
        None,
    )
    if not player_obj:
        return SkipPlacementEligibilityResultPy(
            eligible=False,
            reason="Player not found",
            code="PLAYER_NOT_FOUND",
        )

    # Canonical rule: when rings_in_hand == 0, skip_placement is not a valid
    # voluntary action. Players must emit `no_placement_action` instead.
    if player_obj.rings_in_hand <= 0:
        return SkipPlacementEligibilityResultPy(
            eligible=False,
            reason="Cannot skip placement with no rings in hand",
            code="NO_RINGS_IN_HAND",
        )

    from app.game_engine import GameEngine

    board = state.board
    has_controlled_stack = False
    has_legal_action_from_stack = False

    for stack in board.stacks.values():
        if stack.controlling_player != player or stack.stack_height <= 0:
            continue
        has_controlled_stack = True

        if GameEngine._has_any_legal_move_or_capture_from_on_board(
            board.type,
            stack.position,
            player,
            board,
        ):
            has_legal_action_from_stack = True
            break

    if not has_controlled_stack:
        return SkipPlacementEligibilityResultPy(
            eligible=False,
            reason=(
                "Cannot skip placement when you control no stacks on the "
                "board"
            ),
            code="NO_CONTROLLED_STACKS",
        )

    if not has_legal_action_from_stack:
        return SkipPlacementEligibilityResultPy(
            eligible=False,
            reason=(
                "Cannot skip placement when no legal moves or captures "
                "are available"
            ),
            code="NO_LEGAL_ACTIONS",
        )

    return SkipPlacementEligibilityResultPy(eligible=True)


def apply_place_ring_py(
    state: GameState,
    move: Move,
) -> PlacementApplicationOutcomePy:
    """
    Apply a PLACE_RING move and return the resulting GameState and metadata.

    This helper delegates the actual mutation semantics to the canonical
    GameEngine.apply_move implementation to avoid duplicating placement
    mutation logic. The input ``state`` is treated as immutable; the
    returned ``next_state`` is a new GameState instance.
    """
    from app.game_engine import GameEngine

    if move.type != MoveType.PLACE_RING:
        raise ValueError(f"Expected PLACE_RING move, got {move.type}")

    if move.to is None:
        raise ValueError("PLACE_RING move must have a destination (Move.to)")

    # Snapshot acting player's rings in hand before the move.
    player_before = next(
        (p for p in state.players if p.player_number == move.player),
        None,
    )
    rings_before = player_before.rings_in_hand if player_before else 0

    # Determine whether this placement lands on an existing stack.
    pos_key = move.to.to_key()
    existing_stack = state.board.stacks.get(pos_key)
    placed_on_stack = bool(existing_stack and existing_stack.stack_height > 0)

    next_state = GameEngine.apply_move(state, move)

    # Derive the effective number of rings placed from rings_in_hand delta.
    player_after = next(
        (p for p in next_state.players if p.player_number == move.player),
        None,
    )
    rings_after = player_after.rings_in_hand if player_after else rings_before
    applied_count = max(0, rings_before - rings_after)

    return PlacementApplicationOutcomePy(
        next_state=next_state,
        applied_count=applied_count,
        placed_on_stack=placed_on_stack,
    )

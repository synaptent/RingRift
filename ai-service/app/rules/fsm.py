"""
Python Turn FSM for phase-to-move validation.

This module provides a Finite State Machine for validating that moves are
appropriate for the current game phase. It mirrors the semantics of the
TypeScript TurnStateMachine in src/shared/engine/fsm/TurnStateMachine.ts.

Architecture Note (2025-12-09):
-------------------------------
FSM validation and phase orchestration are separate concerns:

1. FSM Validation (validate_move_for_phase): ACTIVE and canonical
   - Validates that moves are appropriate for current phase
   - Raises FSMValidationError on violations
   - Enabled by default (RINGRIFT_FSM_VALIDATION_MODE=active)

2. Phase Orchestration: Uses phase_machine.py (proven, stable)
   - The compute_fsm_orchestration() function below is EXPERIMENTAL
   - Phase transitions use app.rules.phase_machine.advance_phases()
   - This maintains parity with proven Python phase transition logic

Modes:
- off: No validation (not recommended, legacy only)
- active: Raises FSMValidationError on violations (default, canonical)

Environment variable RINGRIFT_FSM_VALIDATION_MODE controls the mode.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Set, Tuple

from app.models import GamePhase, GameState, Move, MoveType
from app.rules.history_contract import phase_move_contract

logger = logging.getLogger(__name__)

FSMValidationMode = Literal["off", "active"]


def get_fsm_mode() -> FSMValidationMode:
    """Get the FSM validation mode from environment variable.

    Defaults to 'active' (FSM is canonical).
    """
    mode = os.environ.get("RINGRIFT_FSM_VALIDATION_MODE", "active").lower()
    if mode == "off":
        return "off"
    # FSM validation is canonical (RR-CANON compliance)
    return "active"


class FSMValidationError(Exception):
    """Raised when FSM validation fails in active mode."""

    def __init__(
        self,
        code: str,
        message: str,
        current_phase: str,
        move_type: str,
        player: int,
    ):
        self.code = code
        self.message = message
        self.current_phase = current_phase
        self.move_type = move_type
        self.player = player
        super().__init__(
            f"[FSM] {code}: {message} (phase={current_phase}, move_type={move_type}, player={player})"
        )


@dataclass(frozen=True)
class FSMValidationResult:
    """Result of validating a move against the FSM."""

    ok: bool
    code: str | None = None
    message: str | None = None


# Map from GamePhase enum to canonical phase string
_PHASE_TO_CANONICAL: Dict[GamePhase, str] = {
    GamePhase.RING_PLACEMENT: "ring_placement",
    GamePhase.MOVEMENT: "movement",
    GamePhase.CAPTURE: "capture",
    GamePhase.CHAIN_CAPTURE: "chain_capture",
    GamePhase.LINE_PROCESSING: "line_processing",
    GamePhase.TERRITORY_PROCESSING: "territory_processing",
    GamePhase.FORCED_ELIMINATION: "forced_elimination",
}


def _get_allowed_move_types_for_phase(phase: GamePhase) -> Set[str]:
    """Get the set of allowed move types for a given phase."""
    canonical_phase = _PHASE_TO_CANONICAL.get(phase)
    if canonical_phase is None:
        return set()

    contract = phase_move_contract()
    allowed = contract.get(canonical_phase, ())  # type: ignore[arg-type]
    return set(allowed)


def _is_bookkeeping_move(move_type: MoveType) -> bool:
    """Check if a move type is a forced bookkeeping move."""
    return move_type in {
        MoveType.NO_PLACEMENT_ACTION,
        MoveType.NO_MOVEMENT_ACTION,
        MoveType.NO_LINE_ACTION,
        MoveType.NO_TERRITORY_ACTION,
        MoveType.FORCED_ELIMINATION,
    }


def validate_move_for_phase(
    phase: GamePhase,
    move: Move,
    game_state: Optional[GameState] = None,
) -> FSMValidationResult:
    """
    Validate that a move is allowed in the given phase.

    This is the core FSM validation function. It checks:
    1. Phase-to-move type contract (from history_contract.py)
    2. Special guards for bookkeeping moves

    Args:
        phase: Current game phase
        move: Move to validate
        game_state: Optional game state for additional context checks

    Returns:
        FSMValidationResult with ok=True if valid, ok=False with error details if invalid
    """
    move_type_str = move.type.value
    allowed = _get_allowed_move_types_for_phase(phase)

    # Check basic phase-to-move contract
    if not allowed:
        return FSMValidationResult(
            ok=False,
            code="UNKNOWN_PHASE",
            message=f"Phase {phase.value} has no defined move contract",
        )

    if move_type_str not in allowed:
        return FSMValidationResult(
            ok=False,
            code="INVALID_MOVE_FOR_PHASE",
            message=f"Move type '{move_type_str}' not allowed in phase '{phase.value}'. "
            f"Allowed: {sorted(allowed)}",
        )

    # Additional guards for specific move types
    if move.type == MoveType.NO_TERRITORY_ACTION:
        # NO_TERRITORY_ACTION is only valid when there are no territory regions
        # This mirrors handleTerritoryProcessing in TurnStateMachine.ts:781-793
        if game_state is not None:
            # Check if there are disconnected regions
            # If regions exist, NO_TERRITORY_ACTION is invalid
            from app.game_engine import GameEngine

            territory_moves = GameEngine._get_territory_processing_moves(
                game_state, game_state.current_player
            )
            if territory_moves:
                return FSMValidationResult(
                    ok=False,
                    code="CANNOT_SKIP_TERRITORY_WITH_REGIONS",
                    message="Cannot emit NO_TERRITORY_ACTION when territory regions exist. "
                    f"Found {len(territory_moves)} territory moves available.",
                )

    if move.type == MoveType.NO_LINE_ACTION:
        # NO_LINE_ACTION is only valid when there are no lines to process
        if game_state is not None:
            from app.game_engine import GameEngine

            line_moves = [
                m
                for m in GameEngine._get_line_processing_moves(
                    game_state, game_state.current_player
                )
                if m.type != MoveType.NO_LINE_ACTION
            ]
            if line_moves:
                return FSMValidationResult(
                    ok=False,
                    code="CANNOT_SKIP_LINE_WITH_LINES",
                    message="Cannot emit NO_LINE_ACTION when lines exist to process. "
                    f"Found {len(line_moves)} line moves available.",
                )

    if move.type == MoveType.NO_MOVEMENT_ACTION:
        # NO_MOVEMENT_ACTION is only valid when there are no movement/capture moves
        # Per RR-CANON-R070 (7-phase model) and TS FSMAdapter.deriveMovementState,
        # we must use get_valid_moves() to check the phase-local interactive surface,
        # NOT the global _has_valid_movements/_has_valid_captures helpers (which are
        # for FE eligibility checks). This ensures the FSM guard aligns with what
        # moves the env/host would actually surface to the player.
        if game_state is not None:
            from app.game_engine import GameEngine

            valid_moves = GameEngine.get_valid_moves(
                game_state, game_state.current_player
            )
            # Filter to movement-like moves (same filtering as TS FSMAdapter)
            movement_like = [
                m for m in valid_moves
                if m.type in (
                    MoveType.MOVE_STACK,
                    MoveType.MOVE_RING,
                    MoveType.OVERTAKING_CAPTURE,
                    MoveType.CONTINUE_CAPTURE_SEGMENT,
                    MoveType.RECOVERY_SLIDE,
                )
            ]
            if movement_like:
                return FSMValidationResult(
                    ok=False,
                    code="CANNOT_SKIP_MOVEMENT_WITH_MOVES",
                    message="Cannot emit NO_MOVEMENT_ACTION when movement or capture is available.",
                )

    return FSMValidationResult(ok=True)


class TurnFSM:
    """
    Python Turn FSM for validating phase transitions.

    This class provides a stateful FSM that validates moves against the
    current phase. It supports off/active modes controlled by
    the RINGRIFT_FSM_VALIDATION_MODE environment variable.

    Usage:
        fsm = TurnFSM(mode="active")
        fsm.validate_and_send(game_state.current_phase, move, game_state)

    In active mode (default), validation failures raise FSMValidationError.
    In off mode, validation is skipped entirely.
    """

    def __init__(self, mode: Optional[FSMValidationMode] = None):
        """
        Initialize the FSM.

        Args:
            mode: Validation mode. If None, reads from RINGRIFT_FSM_VALIDATION_MODE.
        """
        self.mode = mode or get_fsm_mode()
        self.history: List[Tuple[GamePhase, Move]] = []
        self.violation_count = 0

    def validate_and_send(
        self,
        phase: GamePhase,
        move: Move,
        game_state: Optional[GameState] = None,
    ) -> FSMValidationResult:
        """
        Validate a move and record it if valid.

        Args:
            phase: Current game phase
            move: Move to validate
            game_state: Optional game state for context checks

        Returns:
            FSMValidationResult

        Raises:
            FSMValidationError: In active mode when validation fails
        """
        if self.mode == "off":
            self.history.append((phase, move))
            return FSMValidationResult(ok=True)

        result = validate_move_for_phase(phase, move, game_state)

        if not result.ok:
            self.violation_count += 1
            # In active mode (default), raise FSMValidationError
            raise FSMValidationError(
                code=result.code or "UNKNOWN",
                message=result.message or "Unknown error",
                current_phase=phase.value,
                move_type=move.type.value,
                player=move.player,
            )

        self.history.append((phase, move))
        return result

    def get_required_bookkeeping_move(
        self,
        phase: GamePhase,
        game_state: GameState,
    ) -> Optional[MoveType]:
        """
        Determine if a bookkeeping move is required for the current phase.

        This mirrors the TS FSM's requirement detection logic. When a phase
        is entered but no interactive moves are available, a bookkeeping
        move must be emitted to mark that the phase was visited.

        Args:
            phase: Current game phase
            game_state: Current game state

        Returns:
            MoveType if a bookkeeping move is required, None otherwise
        """
        from app.game_engine import GameEngine

        player = game_state.current_player

        if phase == GamePhase.RING_PLACEMENT:
            # Check if any placement moves exist
            valid_moves = GameEngine.get_valid_moves(game_state, player)
            has_placement = any(
                m.type in (MoveType.PLACE_RING, MoveType.SKIP_PLACEMENT)
                for m in valid_moves
            )
            if not has_placement:
                return MoveType.NO_PLACEMENT_ACTION

        elif phase == GamePhase.MOVEMENT:
            has_moves = GameEngine._has_valid_movements(game_state, player)
            has_captures = GameEngine._has_valid_captures(game_state, player)
            if not has_moves and not has_captures:
                return MoveType.NO_MOVEMENT_ACTION

        elif phase == GamePhase.LINE_PROCESSING:
            line_moves = [
                m
                for m in GameEngine._get_line_processing_moves(game_state, player)
                if m.type != MoveType.NO_LINE_ACTION
            ]
            if not line_moves:
                return MoveType.NO_LINE_ACTION

        elif phase == GamePhase.TERRITORY_PROCESSING:
            territory_moves = GameEngine._get_territory_processing_moves(
                game_state, player
            )
            if not territory_moves:
                return MoveType.NO_TERRITORY_ACTION

        elif phase == GamePhase.FORCED_ELIMINATION:
            # FORCED_ELIMINATION is always required when in this phase
            return MoveType.FORCED_ELIMINATION

        return None

    def reset(self) -> None:
        """Reset the FSM state for a new game."""
        self.history.clear()
        self.violation_count = 0


# ═══════════════════════════════════════════════════════════════════════════
# FSM TRANSITION TYPES (mirrors TurnStateMachine.ts)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DetectedLine:
    """A detected line requiring processing."""

    positions: List[Tuple[int, int]]
    player: int
    requires_choice: bool


@dataclass(frozen=True)
class DisconnectedRegion:
    """A disconnected territory region requiring processing."""

    positions: List[Tuple[int, int]]
    controlling_player: int
    eliminations_required: int


@dataclass(frozen=True)
class ChainContinuation:
    """A chain capture continuation target."""

    target: Tuple[int, int]


@dataclass
class FSMDecisionSurface:
    """
    Decision surface information for hosts to construct valid decisions.

    Mirrors the TypeScript FSMDecisionSurface interface from FSMAdapter.ts.
    When the FSM transitions to a phase requiring player decisions, this
    structure provides the concrete data needed to build the decision UI
    and valid move options.
    """

    pending_lines: List[DetectedLine]
    """Detected lines for the current player (line_processing phase)."""

    pending_regions: List[DisconnectedRegion]
    """Territory regions for the current player (territory_processing phase)."""

    chain_continuations: List[ChainContinuation]
    """Available capture targets (chain_capture phase)."""

    forced_elimination_count: int
    """Number of rings that must be eliminated (forced_elimination phase)."""

    @classmethod
    def empty(cls) -> "FSMDecisionSurface":
        """Create an empty decision surface."""
        return cls(
            pending_lines=[],
            pending_regions=[],
            chain_continuations=[],
            forced_elimination_count=0,
        )


PendingDecisionType = Literal[
    "chain_capture",
    "line_order_required",
    "no_line_action_required",
    "region_order_required",
    "no_territory_action_required",
    "forced_elimination",
]


@dataclass
class FSMOrchestrationResult:
    """
    Result of FSM-driven orchestration.

    Mirrors the TypeScript FSMOrchestrationResult interface from FSMAdapter.ts.
    Contains the derived phase, player, and any pending decision information.
    """

    success: bool
    """Whether the FSM transition was successful."""

    next_phase: GamePhase
    """The next phase according to FSM."""

    next_player: int
    """The next player according to FSM."""

    pending_decision_type: Optional[PendingDecisionType] = None
    """Type of pending decision if any."""

    decision_surface: Optional[FSMDecisionSurface] = None
    """Decision surface data for the pending decision."""

    error_code: Optional[str] = None
    """Error code if transition failed."""

    error_message: Optional[str] = None
    """Error message if transition failed."""


# ═══════════════════════════════════════════════════════════════════════════
# FSM TRANSITION FUNCTION (mirrors TurnStateMachine.transition)
# ═══════════════════════════════════════════════════════════════════════════
#
# EXPERIMENTAL: This function is not currently used for phase transitions.
# Phase orchestration uses app.rules.phase_machine.advance_phases() instead.
# See the module docstring for architecture details.
# ═══════════════════════════════════════════════════════════════════════════


def compute_fsm_orchestration(
    game_state: GameState,
    last_move: Move,
) -> FSMOrchestrationResult:
    """
    EXPERIMENTAL: Compute the next phase and player using FSM transition logic.

    WARNING: This function is experimental and not currently used for production
    phase transitions. The proven phase_machine.advance_phases() is used instead.
    This function may have incomplete parity with TypeScript FSM orchestration.

    This mirrors the TypeScript computeFSMOrchestration function from FSMAdapter.ts.
    It determines the next phase based on the current phase and move type.

    Args:
        game_state: Current game state (after move application)
        last_move: The move that was just applied

    Returns:
        FSMOrchestrationResult with next phase, player, and decision surface
    """
    from app.game_engine import GameEngine
    from app.rules.phase_machine import (
        compute_had_any_action_this_turn,
        player_has_stacks_on_board,
    )

    current_phase = game_state.current_phase
    current_player = game_state.current_player
    move_type = last_move.type

    # Default result
    next_phase = current_phase
    next_player = current_player
    pending_decision_type: Optional[PendingDecisionType] = None
    decision_surface: Optional[FSMDecisionSurface] = None

    # Phase transitions based on move type (mirrors TurnStateMachine handlers)
    if current_phase == GamePhase.RING_PLACEMENT:
        if move_type == MoveType.PLACE_RING:
            # After placement, check if movement/capture is available
            has_moves = GameEngine._has_valid_movements(game_state, current_player)
            has_captures = GameEngine._has_valid_captures(game_state, current_player)
            if has_moves or has_captures:
                next_phase = GamePhase.MOVEMENT
            else:
                next_phase = GamePhase.LINE_PROCESSING
        elif move_type in (MoveType.SKIP_PLACEMENT, MoveType.NO_PLACEMENT_ACTION):
            # Skip/no-op placement always leads to movement phase
            next_phase = GamePhase.MOVEMENT

    elif current_phase == GamePhase.MOVEMENT:
        if move_type == MoveType.MOVE_STACK:
            # After movement, check for captures
            capture_moves = GameEngine._get_capture_moves(game_state, current_player)
            if capture_moves:
                next_phase = GamePhase.CAPTURE
            else:
                next_phase = GamePhase.LINE_PROCESSING
        elif move_type == MoveType.NO_MOVEMENT_ACTION:
            next_phase = GamePhase.LINE_PROCESSING
        elif move_type == MoveType.RECOVERY_SLIDE:
            next_phase = GamePhase.LINE_PROCESSING
        elif move_type == MoveType.OVERTAKING_CAPTURE:
            # Initial capture may lead to chain capture
            capture_moves = GameEngine._get_capture_moves(game_state, current_player)
            if capture_moves:
                next_phase = GamePhase.CHAIN_CAPTURE
            else:
                next_phase = GamePhase.LINE_PROCESSING

    elif current_phase == GamePhase.CAPTURE:
        if move_type == MoveType.OVERTAKING_CAPTURE:
            capture_moves = GameEngine._get_capture_moves(game_state, current_player)
            if capture_moves:
                next_phase = GamePhase.CHAIN_CAPTURE
            else:
                next_phase = GamePhase.LINE_PROCESSING

    elif current_phase == GamePhase.CHAIN_CAPTURE:
        if move_type == MoveType.CONTINUE_CAPTURE_SEGMENT:
            capture_moves = GameEngine._get_capture_moves(game_state, current_player)
            if capture_moves:
                next_phase = GamePhase.CHAIN_CAPTURE  # Stay in chain
            else:
                next_phase = GamePhase.LINE_PROCESSING

    elif current_phase == GamePhase.LINE_PROCESSING:
        if move_type == MoveType.NO_LINE_ACTION:
            # RR-CANON-R075: Always visit TERRITORY_PROCESSING as a distinct phase.
            # When no regions exist, the host emits NO_TERRITORY_ACTION next.
            next_phase = GamePhase.TERRITORY_PROCESSING
        elif move_type in (MoveType.PROCESS_LINE, MoveType.CHOOSE_LINE_OPTION, MoveType.CHOOSE_LINE_REWARD):
            # Check for more lines
            line_moves = [
                m
                for m in GameEngine._get_line_processing_moves(game_state, current_player)
                if m.type != MoveType.NO_LINE_ACTION
            ]
            if line_moves:
                next_phase = GamePhase.LINE_PROCESSING  # Stay
            else:
                # RR-CANON-R075: After finishing line decisions, always enter
                # TERRITORY_PROCESSING. FE/turn-end is resolved after territory.
                next_phase = GamePhase.TERRITORY_PROCESSING

    elif current_phase == GamePhase.TERRITORY_PROCESSING:
        if move_type == MoveType.NO_TERRITORY_ACTION:
            had_action = compute_had_any_action_this_turn(game_state)
            has_stacks = player_has_stacks_on_board(game_state, current_player)
            if not had_action and has_stacks:
                next_phase = GamePhase.FORCED_ELIMINATION
            else:
                next_phase = GamePhase.RING_PLACEMENT
                next_player = _next_active_player(game_state)
        elif move_type == MoveType.SKIP_TERRITORY_PROCESSING:
            # Voluntary early stop when region decisions exist.
            # This counts as an action for FE gating (see compute_had_any_action_this_turn).
            had_action = compute_had_any_action_this_turn(game_state)
            has_stacks = player_has_stacks_on_board(game_state, current_player)
            if not had_action and has_stacks:
                next_phase = GamePhase.FORCED_ELIMINATION
            else:
                next_phase = GamePhase.RING_PLACEMENT
                next_player = _next_active_player(game_state)
        elif move_type in (
            MoveType.CHOOSE_TERRITORY_OPTION,
            MoveType.PROCESS_TERRITORY_REGION,
            MoveType.ELIMINATE_RINGS_FROM_STACK,
        ):
            # Check for more regions
            territory_moves = GameEngine._get_territory_processing_moves(
                game_state, current_player
            )
            if territory_moves:
                next_phase = GamePhase.TERRITORY_PROCESSING  # Stay
            else:
                had_action = compute_had_any_action_this_turn(game_state)
                has_stacks = player_has_stacks_on_board(game_state, current_player)
                if not had_action and has_stacks:
                    next_phase = GamePhase.FORCED_ELIMINATION
                else:
                    next_phase = GamePhase.RING_PLACEMENT
                    next_player = _next_active_player(game_state)

    elif current_phase == GamePhase.FORCED_ELIMINATION:
        if move_type == MoveType.FORCED_ELIMINATION:
            # After forced elimination, turn ends
            next_phase = GamePhase.RING_PLACEMENT
            next_player = _next_active_player(game_state)

    # Build decision surface for the next phase
    if next_phase == GamePhase.CHAIN_CAPTURE:
        pending_decision_type = "chain_capture"
        capture_moves = GameEngine._get_capture_moves(game_state, next_player)
        decision_surface = FSMDecisionSurface(
            pending_lines=[],
            pending_regions=[],
            chain_continuations=[
                ChainContinuation(target=(m.capture_target.x, m.capture_target.y))
                for m in capture_moves
                if m.capture_target
            ],
            forced_elimination_count=0,
        )

    elif next_phase == GamePhase.LINE_PROCESSING:
        line_moves = [
            m
            for m in GameEngine._get_line_processing_moves(game_state, next_player)
            if m.type != MoveType.NO_LINE_ACTION
        ]
        if line_moves:
            pending_decision_type = "line_order_required"
        elif move_type not in (
            MoveType.NO_LINE_ACTION,
            MoveType.PROCESS_LINE,
            MoveType.CHOOSE_LINE_OPTION,
            MoveType.CHOOSE_LINE_REWARD,  # legacy alias
        ):
            pending_decision_type = "no_line_action_required"

        if pending_decision_type:
            decision_surface = FSMDecisionSurface(
                pending_lines=[],  # Would need line detection here
                pending_regions=[],
                chain_continuations=[],
                forced_elimination_count=0,
            )

    elif next_phase == GamePhase.TERRITORY_PROCESSING:
        territory_moves = GameEngine._get_territory_processing_moves(
            game_state, next_player
        )
        if territory_moves:
            pending_decision_type = "region_order_required"
        elif move_type not in (
            MoveType.NO_TERRITORY_ACTION,
            MoveType.SKIP_TERRITORY_PROCESSING,
            MoveType.CHOOSE_TERRITORY_OPTION,
            MoveType.PROCESS_TERRITORY_REGION,
            MoveType.ELIMINATE_RINGS_FROM_STACK,
        ):
            pending_decision_type = "no_territory_action_required"

        if pending_decision_type:
            decision_surface = FSMDecisionSurface(
                pending_lines=[],
                pending_regions=[],  # Would need territory detection here
                chain_continuations=[],
                forced_elimination_count=0,
            )

    elif next_phase == GamePhase.FORCED_ELIMINATION:
        pending_decision_type = "forced_elimination"
        decision_surface = FSMDecisionSurface(
            pending_lines=[],
            pending_regions=[],
            chain_continuations=[],
            forced_elimination_count=1,  # Simplified; actual count from game state
        )

    return FSMOrchestrationResult(
        success=True,
        next_phase=next_phase,
        next_player=next_player,
        pending_decision_type=pending_decision_type,
        decision_surface=decision_surface,
    )


def _player_has_any_rings(game_state: GameState, player: int) -> bool:
    """
    Check if a player has any rings anywhere per RR-CANON-R201.

    A player has rings anywhere if:
    - They have rings in hand (rings_in_hand > 0), OR
    - They have rings in any stack (controlled or buried)

    Players without ANY rings are permanently eliminated and must be skipped
    when advancing the turn. Players with buried rings (but no controlled
    stacks and no rings in hand) are NOT skipped - they may be recovery-eligible.

    This is different from "turn-material" (controlled stacks OR rings in hand).
    The key distinction:
    - turn-material: used to determine what moves are available
    - any-rings: used to determine if player is permanently eliminated

    Mirrors the check in TS turnLogic.ts playerHasAnyRings delegate.
    """
    # Check rings in hand first (fast path)
    player_state = next(
        (p for p in game_state.players if p.player_number == player),
        None,
    )
    if player_state is not None and player_state.rings_in_hand > 0:
        return True

    # Check all stacks for any ring belonging to this player
    # This includes both controlled stacks AND buried rings
    for stack in game_state.board.stacks.values():
        if stack.stack_height <= 0:
            continue
        # The rings list contains player numbers from bottom to top
        if stack.rings and player in stack.rings:
            return True

    return False


def _next_active_player(game_state: GameState) -> int:
    """
    Get the next active player who has any rings anywhere.

    Per RR-CANON-R201 and TS turnLogic.ts:
    - Players without ANY rings (controlled, buried, or in hand) are permanently
      eliminated and are skipped
    - Players with buried rings (but no stacks/hand) are NOT skipped - they may
      be recovery-eligible
    - Continue around the table until finding a player with rings
    - If all players are exhausted, return the initial next player (terminal state)

    This mirrors the TS behavior where workingState.currentPlayer is set to
    initialNextPlayer at the start, then cycles through players. When all are
    skipped, workingState.currentPlayer ends up back at the initial next player
    due to how the modular arithmetic works.
    """
    num_players = len(game_state.players)
    if num_players == 0:
        return game_state.current_player

    # Find index of the current player in the players list
    current_index = 0
    for i, p in enumerate(game_state.players):
        if p.player_number == game_state.current_player:
            current_index = i
            break

    # Compute initial next player (simple modular rotation)
    initial_next_index = (current_index + 1) % num_players
    initial_next_player = game_state.players[initial_next_index].player_number

    max_skips = num_players
    skips = 0
    idx = initial_next_index

    while skips < max_skips:
        candidate = game_state.players[idx]

        # Check if candidate has any rings anywhere (RR-CANON-R201):
        # - rings in controlled stacks (top ring is player's colour)
        # - rings buried inside stacks controlled by others
        # - rings in hand (not yet placed)
        # Players without ANY rings are permanently eliminated and must be
        # skipped. Players with buried rings but no stacks/hand are NOT
        # skipped - they may be recovery-eligible.
        has_any_rings = _player_has_any_rings(
            game_state, candidate.player_number
        )

        if has_any_rings:
            return candidate.player_number

        # Player has no rings anywhere (permanently eliminated); skip to next seat
        idx = (idx + 1) % num_players
        skips += 1

    # All players exhausted (no one has any rings). Per TS turnLogic.ts
    # behavior, return the initial next player. Victory logic will handle
    # the global stalemate.
    return initial_next_player


def compare_fsm_with_legacy(
    fsm_result: FSMOrchestrationResult,
    legacy_phase: GamePhase,
    legacy_player: int,
) -> Dict[str, object]:
    """
    Compare FSM orchestration result with legacy orchestration result.

    Mirrors the TypeScript compareFSMWithLegacy function from FSMAdapter.ts.
    Returns divergence details if they differ.

    Args:
        fsm_result: Result from compute_fsm_orchestration
        legacy_phase: Phase from legacy phase_machine
        legacy_player: Player from legacy phase_machine

    Returns:
        Dict with divergence details
    """
    # Map FSM RING_PLACEMENT at turn boundary to legacy semantics
    effective_fsm_phase = fsm_result.next_phase

    phase_diverged = effective_fsm_phase != legacy_phase
    player_diverged = fsm_result.next_player != legacy_player
    diverged = phase_diverged or player_diverged

    if diverged:
        return {
            "diverged": True,
            "phase_diverged": phase_diverged,
            "player_diverged": player_diverged,
            "details": {
                "fsm_phase": effective_fsm_phase.value,
                "legacy_phase": legacy_phase.value,
                "fsm_player": fsm_result.next_player,
                "legacy_player": legacy_player,
            },
        }

    return {"diverged": False, "phase_diverged": False, "player_diverged": False}

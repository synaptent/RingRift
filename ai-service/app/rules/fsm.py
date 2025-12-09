"""
Python Turn FSM for phase-to-move validation.

This module provides a Finite State Machine for validating that moves are
appropriate for the current game phase. It mirrors the semantics of the
TypeScript TurnStateMachine in src/shared/engine/fsm/TurnStateMachine.ts.

Modes:
- off: No validation (default for backwards compatibility)
- shadow: Logs violations but allows processing to continue
- active: Raises FSMValidationError on violations (fail-fast)

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

FSMValidationMode = Literal["off", "shadow", "active"]


def get_fsm_mode() -> FSMValidationMode:
    """Get the FSM validation mode from environment variable."""
    mode = os.environ.get("RINGRIFT_FSM_VALIDATION_MODE", "off").lower()
    if mode in ("off", "shadow", "active"):
        return mode  # type: ignore[return-value]
    logger.warning(f"Invalid RINGRIFT_FSM_VALIDATION_MODE={mode}, defaulting to 'off'")
    return "off"


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
        if game_state is not None:
            from app.game_engine import GameEngine

            has_moves = GameEngine._has_valid_movements(
                game_state, game_state.current_player
            )
            has_captures = GameEngine._has_valid_captures(
                game_state, game_state.current_player
            )
            if has_moves or has_captures:
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
    current phase. It supports off/shadow/active modes controlled by
    the RINGRIFT_FSM_VALIDATION_MODE environment variable.

    Usage:
        fsm = TurnFSM(mode="active")
        fsm.validate_and_send(game_state.current_phase, move, game_state)

    In shadow mode, validation failures are logged but don't raise.
    In active mode, validation failures raise FSMValidationError.
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

            if self.mode == "shadow":
                logger.warning(
                    f"[FSM shadow] Violation #{self.violation_count}: {result.code} - {result.message} "
                    f"(phase={phase.value}, move={move.type.value}, player={move.player})"
                )
            elif self.mode == "active":
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

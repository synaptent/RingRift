"""Bookkeeping move handler mixin for RingRiftEnv.

This module provides the BookkeepingMoveHandlerMixin class which handles
auto-generated bookkeeping moves per RR-CANON-R075/R076, extracted from
RingRiftEnv.step() for improved testability.
"""

import logging
import os
from typing import TYPE_CHECKING, Callable

from app.game_engine import GameEngine, PhaseRequirement, PhaseRequirementType
from app.models import GamePhase, GameStatus, MoveType

if TYPE_CHECKING:
    from app.models import GameState, Move
    from app.rules.default_engine import DefaultRulesEngine
    from app.rules.fsm import TurnFSM


logger = logging.getLogger(__name__)


class BookkeepingMoveHandlerMixin:
    """Mixin that handles auto-bookkeeping move generation.

    Per RR-CANON-R075/R076, the host must emit bookkeeping moves
    (no_*_action, forced_elimination, no_placement_action) when no
    interactive moves exist. This mixin encapsulates that logic.

    Attributes expected from host class:
        _state: GameState | None
        _move_count: int
        _rules_engine: DefaultRulesEngine | None
        _fsm: TurnFSM | None
        _force_bookkeeping_moves: bool
    """

    _state: "GameState | None"
    _move_count: int
    _rules_engine: "DefaultRulesEngine | None"
    _fsm: "TurnFSM | None"
    _force_bookkeeping_moves: bool

    def _apply_auto_bookkeeping_moves(self) -> list["Move"]:
        """Auto-satisfy pending phase requirements per RR-CANON-R075/R076.

        This method generates and applies bookkeeping moves that the host
        must emit when no interactive moves exist. It continues until:
        - Game ends (not ACTIVE)
        - Enters CAPTURE/CHAIN_CAPTURE phase (requires player decision)
        - No more phase requirements exist

        Returns
        -------
        list[Move]
            List of auto-generated bookkeeping moves that were applied.
        """
        if self._state is None:
            return []

        auto_generated_moves: list["Move"] = []

        while self._state.game_status == GameStatus.ACTIVE:
            # Skip auto-bookkeeping during capture phases - they require decisions
            if self._should_break_for_capture_phase():
                break

            # Try to get and apply the next bookkeeping move
            auto_move = self._get_next_bookkeeping_move()
            if auto_move is None:
                break

            # Apply the bookkeeping move
            self._apply_bookkeeping_move(auto_move)
            auto_generated_moves.append(auto_move)

        return auto_generated_moves

    def _should_break_for_capture_phase(self) -> bool:
        """Check if we should break the bookkeeping loop for capture phases.

        CAPTURE and CHAIN_CAPTURE phases require player decisions (selecting
        captures or declining). Bookkeeping moves are only valid in phases
        where no interactive moves exist.

        Returns
        -------
        bool
            True if current phase is CAPTURE or CHAIN_CAPTURE.
        """
        if self._state is None:
            return False

        return self._state.current_phase in (
            GamePhase.CAPTURE,
            GamePhase.CHAIN_CAPTURE,
        )

    def _get_next_bookkeeping_move(self) -> "Move | None":
        """Get the next bookkeeping move to apply, if any.

        Checks for phase requirements and synthesizes the appropriate
        bookkeeping move. Also handles forced bookkeeping in LINE/TERRITORY
        processing phases when enabled.

        Returns
        -------
        Move | None
            The bookkeeping move to apply, or None if none needed.
        """
        if self._state is None:
            return None

        current_player = self._state.current_player

        # Check for standard phase requirement
        requirement = GameEngine.get_phase_requirement(self._state, current_player)

        if requirement is None:
            # No standard requirement - check for forced bookkeeping
            return self._get_forced_bookkeeping_move(current_player)

        # Synthesize bookkeeping move from requirement
        return GameEngine.synthesize_bookkeeping_move(requirement, self._state)

    def _get_forced_bookkeeping_move(self, current_player: int) -> "Move | None":
        """Get a forced bookkeeping move when enabled.

        When _force_bookkeeping_moves is True and we're in LINE/TERRITORY
        processing, check if the only legal move is a bookkeeping no-op.

        Parameters
        ----------
        current_player:
            The current player number.

        Returns
        -------
        Move | None
            The forced bookkeeping move, or None if not applicable.
        """
        if self._state is None:
            return None

        if not self._force_bookkeeping_moves:
            return None

        if self._state.current_phase not in (
            GamePhase.LINE_PROCESSING,
            GamePhase.TERRITORY_PROCESSING,
        ):
            return None

        # Check available moves
        forced_moves = GameEngine.get_valid_moves(self._state, current_player)

        if not forced_moves:
            # No moves at all - synthesize the mandatory no-op
            req_type = (
                PhaseRequirementType.NO_LINE_ACTION_REQUIRED
                if self._state.current_phase == GamePhase.LINE_PROCESSING
                else PhaseRequirementType.NO_TERRITORY_ACTION_REQUIRED
            )
            requirement = PhaseRequirement(
                type=req_type,
                player=current_player,
                eligible_positions=[],
            )
            return GameEngine.synthesize_bookkeeping_move(requirement, self._state)

        if (
            len(forced_moves) == 1
            and forced_moves[0].type
            in (MoveType.NO_LINE_ACTION, MoveType.NO_TERRITORY_ACTION)
        ):
            # Single bookkeeping move available - use it directly
            return forced_moves[0]

        # Multiple moves or non-bookkeeping moves - stop
        return None

    def _apply_bookkeeping_move(self, auto_move: "Move") -> None:
        """Apply a bookkeeping move to the current state.

        Parameters
        ----------
        auto_move:
            The bookkeeping move to apply.
        """
        if self._state is None:
            return

        # Defensive assertion: verify synthesized move is valid for current phase
        try:
            GameEngine._assert_phase_move_invariant(self._state, auto_move)
        except RuntimeError as e:
            logger.error(
                "Phase/move invariant violation in bookkeeping loop: %s. "
                "State: phase=%s, player=%s, move_type=%s",
                str(e),
                self._state.current_phase.value,
                self._state.current_player,
                auto_move.type.value,
            )
            raise

        # FSM validation for auto-generated bookkeeping moves
        if self._fsm is not None:
            self._fsm.validate_and_send(
                self._state.current_phase, auto_move, self._state
            )

        # Apply the move
        if self._rules_engine is not None:
            self._state = self._rules_engine.apply_move(
                self._state, auto_move, trace_mode=True
            )
        else:
            self._state = GameEngine.apply_move(
                self._state, auto_move, trace_mode=True
            )

        self._move_count += 1

"""Move generation mixin for RingRiftEnv.

This module provides the MoveGenerationMixin class which handles legal move
generation with phase requirement handling, extracted from RingRiftEnv
for improved testability.
"""

from typing import TYPE_CHECKING

from app.game_engine import GameEngine
from app.models import GamePhase

if TYPE_CHECKING:
    from app.models import GameState, Move
    from app.rules.default_engine import DefaultRulesEngine


class MoveGenerationMixin:
    """Mixin that handles legal move generation with phase requirements.

    Per RR-CANON-R076, the core rules layer (GameEngine.get_valid_moves)
    returns ONLY interactive moves. When there are no interactive moves,
    this mixin checks for phase requirements and synthesizes the appropriate
    bookkeeping move (no_*_action, forced_elimination).

    Attributes expected from host class:
        _state: GameState | None
        _rules_engine: DefaultRulesEngine | None
    """

    _state: "GameState | None"
    _rules_engine: "DefaultRulesEngine | None"

    @property
    def state(self) -> "GameState":
        """Return the current GameState (expected to be implemented by host)."""
        assert self._state is not None, "Call reset() before using env"
        return self._state

    def _get_interactive_moves(self) -> list["Move"]:
        """Get interactive moves from the core rules layer.

        Returns
        -------
        list[Move]
            List of interactive moves for the current player.
        """
        if self._rules_engine is not None:
            return self._rules_engine.get_valid_moves(
                self.state,
                self.state.current_player,
            )
        return GameEngine.get_valid_moves(
            self.state,
            self.state.current_player,
        )

    def _get_phase_local_moves(self) -> list["Move"]:
        """Get phase-local moves when interactive moves are empty.

        In LINE/TERRITORY processing, cache/metadata drift can cause
        get_valid_moves() to return an empty list even though the
        phase-local decision enumerators would surface interactive moves.

        Returns
        -------
        list[Move]
            Phase-local moves if in LINE/TERRITORY processing phase,
            otherwise empty list.
        """
        if self._state is None:
            return []

        if self._state.current_phase == GamePhase.LINE_PROCESSING:
            return GameEngine._get_line_processing_moves(
                self._state, self._state.current_player
            )
        elif self._state.current_phase == GamePhase.TERRITORY_PROCESSING:
            return GameEngine._get_territory_processing_moves(
                self._state, self._state.current_player
            )
        return []

    def _synthesize_bookkeeping_move_if_needed(self) -> list["Move"]:
        """Synthesize bookkeeping move if phase requirement exists.

        Per RR-CANON-R076, when there are no interactive moves but a
        phase requirement exists, the host must synthesize the
        appropriate bookkeeping move.

        Returns
        -------
        list[Move]
            List containing the synthesized bookkeeping move, or empty list.
        """
        if self._state is None:
            return []

        requirement = GameEngine.get_phase_requirement(
            self._state,
            self._state.current_player,
        )
        if requirement is not None:
            bookkeeping_move = GameEngine.synthesize_bookkeeping_move(
                requirement,
                self._state,
            )
            return [bookkeeping_move]
        return []

    def _validate_move_phase_invariants(self, moves: list["Move"]) -> None:
        """Validate that all moves are legal for the current phase.

        Parameters
        ----------
        moves:
            List of moves to validate.

        Raises
        ------
        RuntimeError
            If any move violates the phase/move invariant.
        """
        if self._state is None:
            return

        for move in moves:
            GameEngine._assert_phase_move_invariant(self._state, move)

    def _validate_phase_requirement_consistency(
        self, moves: list["Move"]
    ) -> None:
        """Validate that phase requirement is satisfied by exactly one move.

        If the core engine reports a pending phase requirement, ensure we
        surfaced exactly one corresponding bookkeeping move.

        Parameters
        ----------
        moves:
            List of legal moves to check.

        Raises
        ------
        RuntimeError
            If phase requirement exists but no moves, or wrong move type.
        """
        if self._state is None:
            return

        requirement = GameEngine.get_phase_requirement(
            self._state,
            self._state.current_player,
        )
        if requirement is None:
            return

        expected = GameEngine.synthesize_bookkeeping_move(
            requirement,
            self._state,
        )
        if not moves:
            raise RuntimeError(
                "RingRiftEnv.legal_moves: phase requirement exists "
                f"({requirement.type.value}) but no legal moves were "
                "returned"
            )
        if len(moves) != 1 or moves[0].type != expected.type:
            raise RuntimeError(
                "RingRiftEnv.legal_moves: inconsistent bookkeeping move "
                f"for requirement {requirement.type.value}: "
                f"got {moves[0].type.value}, expected {expected.type.value}"
            )

    def _get_legal_moves(self) -> list["Move"]:
        """Return legal moves for the current player.

        This is the main entry point for move generation, implementing
        RR-CANON-R076 semantics with fallbacks for phase-local moves
        and bookkeeping synthesis.

        Returns
        -------
        list[Move]
            Legal moves for the current player.
        """
        if self._state is None:
            return []

        # Get interactive moves from the core rules layer
        moves = self._get_interactive_moves()

        # Defensive decision-surface check for LINE/TERRITORY processing
        if not moves and self._state.current_phase in (
            GamePhase.LINE_PROCESSING,
            GamePhase.TERRITORY_PROCESSING,
        ):
            moves = self._get_phase_local_moves()

        # If no interactive moves, check for phase requirements (R076)
        if not moves:
            moves = self._synthesize_bookkeeping_move_if_needed()

        # Defensive validations
        self._validate_move_phase_invariants(moves)
        self._validate_phase_requirement_consistency(moves)

        return moves

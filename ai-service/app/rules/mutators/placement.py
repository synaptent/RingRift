from app.models import GameState, Move, MoveType
from app.rules.interfaces import Mutator
from app.rules.placement import apply_place_ring_py


class PlacementMutator(Mutator):
    def apply(self, state: GameState, move: Move) -> None:
        """Apply a PLACE_RING move to ``state`` in-place.

        This mutator delegates core placement semantics to the canonical
        helper layer in ``app.rules.placement``, mirroring the pattern
        used by CaptureMutator:

        - ``apply_place_ring_py`` performs the mutation via
          ``GameEngine.apply_move`` on a working copy of the state.
        - The resulting GameState is then copied back onto ``state`` so
          that DefaultRulesEngine mutator-first orchestration can compare
          against GameEngine.apply_move without duplicating algorithms.

        Timeline bookkeeping (e.g. ``last_move_at``, phase updates,
        move history, hashes) remains the responsibility of the
        orchestrator (``GameEngine.apply_move`` /
        ``DefaultRulesEngine.apply_move``).
        """
        if move.type != MoveType.PLACE_RING:
            return

        # Work on a deep copy to avoid mutating the caller's state via
        # the intermediate engine call. Copy the resulting canonical
        # GameState back field-by-field, as in CaptureMutator.
        working_state = state.model_copy(deep=True)
        outcome = apply_place_ring_py(working_state, move)

        next_state = outcome.next_state
        for field_name in state.model_fields:
            setattr(state, field_name, getattr(next_state, field_name))

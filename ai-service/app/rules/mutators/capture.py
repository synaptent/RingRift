from app.models import GameState, Move
from app.rules.capture_chain import apply_capture_py
from app.rules.interfaces import Mutator


class CaptureMutator(Mutator):
    def apply(self, state: GameState, move: Move) -> None:
        """
        Delegate capture semantics to the canonical helper layer.

        This mirrors the TS CaptureAggregate.applyCapture path by routing
        through app.rules.capture_chain.apply_capture_py, which in turn
        leverages GameEngine.apply_move for full-state side effects.

        The mutator updates the input ``state`` in-place so that its
        board, players, and other GameState fields match the canonical
        GameEngine.apply_move outcome for the same move.
        """
        # Work on a deep copy to avoid mutating the caller's state via the
        # intermediate engine call. We then copy the resulting fields back
        # onto the original state so that DefaultRulesEngine mutator-first
        # orchestration can compare against GameEngine.apply_move.
        working_state = state.model_copy(deep=True)
        success, next_state, _ = apply_capture_py(working_state, move)
        if not success or next_state is None:
            return

        # Align all GameState fields with the canonical engine result.
        for field_name in state.model_fields:
            setattr(state, field_name, getattr(next_state, field_name))

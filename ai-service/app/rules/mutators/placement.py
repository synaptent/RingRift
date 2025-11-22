from app.models import GameState, Move
from app.rules.interfaces import Mutator
# from app.game_engine import GameEngine


class PlacementMutator(Mutator):
    def apply(self, state: GameState, move: Move) -> None:
        """Apply a PLACE_RING move to ``state`` in-place.

        This is a thin wrapper around ``GameEngine._apply_place_ring`` so
        that placement behaviour has a single source of truth. Timeline
        bookkeeping (e.g. ``last_move_at``, phase updates, move history)
        remains the responsibility of the orchestrator
        (``GameEngine.apply_move`` / ``DefaultRulesEngine.apply_move``).
        """
        # GameEngine._apply_place_ring mutates the provided GameState in-place
        # and keeps board/players in sync with the TS semantics.
        from app.game_engine import GameEngine
        GameEngine._apply_place_ring(state, move)

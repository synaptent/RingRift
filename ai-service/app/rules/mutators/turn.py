from app.models import GameState, Move
from app.rules.interfaces import Mutator

# from app.game_engine import GameEngine


class TurnMutator(Mutator):
    def apply(self, state: GameState, move: Move) -> None:
        # Phase updates are handled in GameEngine._update_phase called by
        # apply_move. But if we want to explicitly handle turn changes here:

        # Note: GameEngine.apply_move calls _update_phase at the end.
        # So we might not need to do anything here if we are just wrapping
        # GameEngine. However, the plan says:
        # 1. TurnChange: Rotate player, reset phase to ring_placement.
        # 2. PhaseChange: Update phase.

        # Since we are delegating to GameEngine for the main logic, and
        # GameEngine handles phase transitions internally, we can just ensure
        # the phase is updated correctly.

        from app.game_engine import GameEngine
        GameEngine._update_phase(state, move)

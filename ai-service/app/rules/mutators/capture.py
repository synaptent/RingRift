from app.models import GameState, Move
from app.rules.interfaces import Mutator
# from app.game_engine import GameEngine


class CaptureMutator(Mutator):
    def apply(self, state: GameState, move: Move) -> None:
        # Delegate to GameEngine's static method
        from app.game_engine import GameEngine
        GameEngine._apply_chain_capture(state, move)
        state.last_move_at = move.timestamp
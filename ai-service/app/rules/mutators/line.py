from app.models import GameState, Move
from app.rules.interfaces import Mutator

# from app.game_engine import GameEngine


class LineMutator(Mutator):
    def apply(self, state: GameState, move: Move) -> None:
        # Delegate to GameEngine's static method
        from app.game_engine import GameEngine
        GameEngine._apply_line_formation(state, move)
        state.last_move_at = move.timestamp

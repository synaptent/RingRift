from app.models import GameState, Move, MoveType
from app.rules.interfaces import Mutator
# from app.game_engine import GameEngine


class TerritoryMutator(Mutator):
    def apply(self, state: GameState, move: Move) -> None:
        # Delegate to GameEngine's static methods
        from app.game_engine import GameEngine
        if move.type == MoveType.ELIMINATE_RINGS_FROM_STACK:
            GameEngine._apply_forced_elimination(state, move)
        else:
            GameEngine._apply_territory_claim(state, move)
        
        state.last_move_at = move.timestamp
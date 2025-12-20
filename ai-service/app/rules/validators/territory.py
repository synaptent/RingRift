from app.board_manager import BoardManager
from app.models import GamePhase, GameState, Move, MoveType
from app.rules.elimination import EliminationContext, is_stack_eligible_for_elimination
from app.rules.interfaces import Validator


class TerritoryValidator(Validator):
    def validate(self, state: GameState, move: Move) -> bool:
        # 1. Phase Check
        if state.current_phase != GamePhase.TERRITORY_PROCESSING:
            return False

        # 2. Turn Check
        if move.player != state.current_player:
            return False

        # 3. Move Type Check
        if move.type not in (
            MoveType.PROCESS_TERRITORY_REGION,
            MoveType.ELIMINATE_RINGS_FROM_STACK,
            MoveType.TERRITORY_CLAIM,
            MoveType.CHOOSE_TERRITORY_OPTION,
        ):
            return False

        # 4. Region Existence & Disconnection Check
        # For PROCESS_TERRITORY_REGION, we are choosing a region to keep.
        # The move should correspond to a disconnected region in the state.

        if move.type == MoveType.PROCESS_TERRITORY_REGION:
            # In TS, we check if the region ID exists in the disconnected
            # regions. Here, we check if there ARE any disconnected regions for
            # the player. A more strict check would verify the specific region
            # ID if available.

            # We can check state.board.territories for disconnected regions
            # belonging to the player.
            has_disconnected = False
            for t in state.board.territories.values():
                if t.controlling_player == move.player and t.is_disconnected:
                    has_disconnected = True
                    break

            if not has_disconnected:
                return False

        # 5. Prerequisite Check (Self-Elimination capability)
        # For ELIMINATE_RINGS_FROM_STACK (self-elimination), the player must
        # have rings outside the disconnected region (or generally available).
        # This is a complex check in TS (canEliminateFromStack).
        # RR-CANON-R082/R145: All controlled stacks outside the region are eligible
        # cap targets, including:
        # (1) Multicolor stacks (with other players' rings buried beneath),
        # (2) Single-color stacks of height > 1 (all player's rings), AND
        # (3) Height-1 standalone rings.

        if move.type == MoveType.ELIMINATE_RINGS_FROM_STACK:
            if not move.to:
                return False
            stack = BoardManager.get_stack(move.to, state.board)
            if not stack:
                return False
            if stack.stack_height == 0:
                return False
            # RR-CANON-R082: Delegate eligibility check to EliminationAggregate
            eligibility = is_stack_eligible_for_elimination(
                rings=stack.rings,
                controlling_player=stack.controlling_player,
                context=EliminationContext.TERRITORY,
                player=move.player,
            )
            if not eligibility.eligible:
                return False

        return True

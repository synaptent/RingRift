from app.models import GameState, Move, GamePhase, MoveType
from app.rules.interfaces import Validator
from app.board_manager import BoardManager


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
        # RR-CANON-R082: An eligible cap target must be either:
        # (1) A multicolor stack controlled by the player (with other players'
        #     rings buried beneath the player's cap), OR
        # (2) A single-color stack of height > 1 consisting entirely of the
        #     player's colour.
        # A height-1 standalone ring is NOT an eligible cap target.

        if move.type == MoveType.ELIMINATE_RINGS_FROM_STACK:
            if not move.to:
                return False
            stack = BoardManager.get_stack(move.to, state.board)
            if not stack or stack.controlling_player != move.player:
                return False
            if stack.stack_height == 0:
                return False
            # RR-CANON-R082: Check eligible cap target criteria
            is_multicolor = stack.stack_height > stack.cap_height
            is_single_color_tall = (
                stack.stack_height == stack.cap_height and stack.stack_height > 1
            )
            if not is_multicolor and not is_single_color_tall:
                # Height-1 standalone ring is not an eligible cap target
                return False

        return True

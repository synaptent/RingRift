from app.models import GameState, Move, MoveType, GamePhase
from app.rules.interfaces import Validator
# from app.game_engine import GameEngine
from app.board_manager import BoardManager

class PlacementValidator(Validator):
    def validate(self, state: GameState, move: Move) -> bool:
        # 1. Phase Check
        if state.current_phase != GamePhase.RING_PLACEMENT:
            return False

        # 1.5. Skip Placement Check
        if move.type == MoveType.SKIP_PLACEMENT:
            # Delegate to GameEngine helper which mirrors TS logic
            from app.game_engine import GameEngine
            skip_moves = GameEngine._get_skip_placement_moves(state, move.player)
            return any(m.type == MoveType.SKIP_PLACEMENT for m in skip_moves)

        # 2. Turn Check
        if move.player != state.current_player:
            return False

        player = next((p for p in state.players if p.player_number == move.player), None)
        if not player:
            return False

        # 3. Rings in Hand Check
        count = move.placement_count or 1
        if player.rings_in_hand < count:
            return False

        # 4. Position Validity Check
        if not BoardManager.is_valid_position(move.to, state.board.type, state.board.size):
            return False

        pos_key = move.to.to_key()

        # 5. Collapsed Space Check
        if pos_key in state.board.collapsed_spaces:
            return False
        
        # Cannot place on markers
        if pos_key in state.board.markers:
            return False

        existing_stack = state.board.stacks.get(pos_key)

        # 6. Placement Logic Checks
        if existing_stack and existing_stack.stack_height > 0:
            # Placing on existing stack
            if count != 1:
                return False
        else:
            # Placing on empty space
            if count < 1 or count > 3:
                return False
            # Note: ringsInHand check above covers the capacity limit implicitly

        # 7. No-Dead-Placement Rule Check
        # Construct a hypothetical board representing the state AFTER placement.
        from app.game_engine import GameEngine
        hyp_board = GameEngine._create_hypothetical_board_with_placement(
            state.board,
            move.to,
            move.player,
            count,
        )

        has_legal_move = GameEngine._has_any_movement_or_capture_after_hypothetical_placement(
            state,
            move.player,
            move.to,
            hyp_board,
        )

        if not has_legal_move:
            return False

        return True
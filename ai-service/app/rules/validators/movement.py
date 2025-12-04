from app.models import GameState, Move, GamePhase
from app.rules.interfaces import Validator
# from app.game_engine import GameEngine
from app.board_manager import BoardManager


class MovementValidator(Validator):
    def validate(self, state: GameState, move: Move) -> bool:
        # 1. Phase Check
        if state.current_phase != GamePhase.MOVEMENT:
            return False

        # 1.5. Must-Move Check
        # If a ring was placed this turn, only the updated stack may move.
        if state.must_move_from_stack_key:
            if not move.from_pos:
                return False
            if move.from_pos.to_key() != state.must_move_from_stack_key:
                return False

        # 2. Turn Check
        if move.player != state.current_player:
            return False

        # 3. Position Validity
        if not move.from_pos or not move.to:
            return False

        if not BoardManager.is_valid_position(
            move.from_pos, state.board.type, state.board.size
        ):
            return False
        if not BoardManager.is_valid_position(
            move.to, state.board.type, state.board.size
        ):
            return False

        # 4. Stack Ownership
        stack = BoardManager.get_stack(move.from_pos, state.board)
        if not stack or stack.controlling_player != move.player:
            return False

        # 5. Collapsed Space Check (Destination)
        if BoardManager.is_collapsed_space(move.to, state.board):
            return False

        # 6. Direction Check
        # 6. Direction Check
        from app.game_engine import GameEngine
        if not GameEngine._is_straight_line_movement(
            state.board.type, move.from_pos, move.to
        ):
            return False

        # 7. Minimum Distance Check
        distance = GameEngine._calculate_distance(
            state.board.type, move.from_pos, move.to
        )
        if distance < stack.stack_height:
            return False

        # 8. Path Check
        if not GameEngine._is_path_clear_for_movement(
            state.board, move.from_pos, move.to
        ):
            return False
        # 9. Landing Check
        landing_stack = BoardManager.get_stack(move.to, state.board)
        landing_marker = state.board.markers.get(move.to.to_key())

        if landing_stack:
            # Cannot land on existing stack in simple movement
            return False

        # Per RR-CANON-R091/R092: landing on any marker (own or opponent) is legal.
        # The marker is removed and a ring from the cap is eliminated.
        # Landing on any marker is allowed (results in elimination).

        return True

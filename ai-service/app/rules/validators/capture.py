from app.models import GameState, Move, GamePhase
from app.rules.interfaces import Validator
# from app.game_engine import GameEngine
from app.board_manager import BoardManager


class CaptureValidator(Validator):
    def validate(self, state: GameState, move: Move) -> bool:
        # 1. Phase Check
        # Captures can happen in MOVEMENT (initial), CAPTURE (chain),
        # or CHAIN_CAPTURE (legacy/canonical)
        allowed_phases = {
            GamePhase.MOVEMENT,
            GamePhase.CAPTURE,
            GamePhase.CHAIN_CAPTURE,
        }
        if state.current_phase not in allowed_phases:
            return False

        # 2. Turn Check
        if move.player != state.current_player:
            return False

        # 2.5. Must-Move Check
        # If a ring was placed this turn, only the updated stack may capture.
        if state.must_move_from_stack_key:
            if not move.from_pos:
                return False
            if move.from_pos.to_key() != state.must_move_from_stack_key:
                return False

        # 3. Position Validity
        if not move.from_pos or not move.to or not move.capture_target:
            return False

        if not BoardManager.is_valid_position(
            move.from_pos, state.board.type, state.board.size
        ):
            return False
        if not BoardManager.is_valid_position(
            move.to, state.board.type, state.board.size
        ):
            return False
        if not BoardManager.is_valid_position(
            move.capture_target, state.board.type, state.board.size
        ):
            return False

        # 4. Core Validation
        # Delegate to the shared helper that implements the full geometry
        # and rules check.
        from app.game_engine import GameEngine
        return GameEngine._validate_capture_segment_on_board_for_reachability(
            state.board.type,
            move.from_pos,
            move.capture_target,
            move.to,
            move.player,
            state.board,
        )
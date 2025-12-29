from typing import TYPE_CHECKING

from app.models import GamePhase, GameState, Move, MoveType, Position
from app.rules.core import get_effective_line_length
from app.rules.interfaces import Validator

if TYPE_CHECKING:
    from app.models import LineInfo


def _position_to_string(pos: Position) -> str:
    """Convert position to string key for comparison."""
    if pos.z is not None:
        return f"{pos.x},{pos.y},{pos.z}"
    return f"{pos.x},{pos.y}"


def _position_to_marker_key(pos: Position) -> str:
    """Convert position to marker key (always x,y format, no z)."""
    return f"{pos.x},{pos.y}"


class LineValidator(Validator):
    """Validator for line processing moves during the LINE_PROCESSING phase.

    Validates PROCESS_LINE, CHOOSE_LINE_REWARD, and CHOOSE_LINE_OPTION moves,
    ensuring the selected line matches pending lines in the game state.
    """

    def validate(self, state: GameState, move: Move) -> bool:
        # 1. Phase Check
        if state.current_phase != GamePhase.LINE_PROCESSING:
            return False

        # 2. Turn Check
        if move.player != state.current_player:
            return False

        # 3. Move Type Check
        if move.type not in (
            MoveType.PROCESS_LINE,
            MoveType.CHOOSE_LINE_REWARD,
            MoveType.CHOOSE_LINE_OPTION,
        ):
            return False

        # 4. Line Index Check (mirrors TS validateProcessLine)
        # TS requires action.lineIndex to be a valid index into formedLines.
        if move.line_index is None:
            return False

        if move.line_index < 0 or move.line_index >= len(state.board.formed_lines):
            return False

        line = state.board.formed_lines[move.line_index]

        # 5. Line Ownership Check
        if line.player != move.player:
            return False

        # 6. Marker Existence Check (RR-CANON-R120)
        # Per RR-CANON-R120: "Each pi currently contains a marker of P"
        # Verify that markers actually exist at the line positions on the board.
        for pos in line.positions:
            # Use x,y format for marker lookup (markers don't include z coord)
            pos_key = _position_to_marker_key(pos)
            marker = state.board.markers.get(pos_key)
            if marker is None:
                return False
            # marker can be int (player number) or dict/MarkerInfo with .player
            marker_player = marker.player if hasattr(marker, 'player') else marker
            if marker_player != move.player:
                return False

        # 7. Option Validity for CHOOSE_LINE_REWARD / CHOOSE_LINE_OPTION
        # Mirrors TS LineValidator.validateChooseLineReward:
        # If collapsed_markers provided, validate count & consecutiveness
        if move.type in (
            MoveType.CHOOSE_LINE_REWARD, MoveType.CHOOSE_LINE_OPTION
        ):
            return self._validate_line_reward_choice(state, move, line)

        return True

    def _validate_line_reward_choice(
        self,
        state: GameState,
        move: Move,
        target_line: "LineInfo"
    ) -> bool:
        """
        Validate collapsed_markers for line reward choice.
        Mirrors TS validateChooseLineReward logic.

        Option 1 (collapse all): collapsed_markers is None OR equals full line
        Option 2 (minimum collapse): collapsed_markers == required_length,
                                     must be consecutive positions from line
        """
        required_length = get_effective_line_length(state.board_type, len(state.players))

        # If collapsed_markers is not provided, it's Option 1 (collapse all)
        # which is always valid for any line.
        if not move.collapsed_markers:
            return True

        collapsed = move.collapsed_markers

        # If collapsed_markers length == line length, it's Option 1 (all)
        # This is valid for any overlength line.
        if len(collapsed) == target_line.length:
            # Verify all positions are actually from the line
            line_pos_keys: set[str] = {
                _position_to_string(p) for p in target_line.positions
            }
            return all(_position_to_string(pos) in line_pos_keys for pos in collapsed)

        # Otherwise it's Option 2 (minimum collapse)
        # Must have exactly required_length positions
        if len(collapsed) != required_length:
            return False

        # Cannot do minimum collapse on exact-length line
        if target_line.length == required_length:
            return False

        # Verify all collapsed positions are part of the line
        line_pos_keys = {
            _position_to_string(p) for p in target_line.positions
        }
        for pos in collapsed:
            if _position_to_string(pos) not in line_pos_keys:
                return False

        # Verify collapsed positions are consecutive within the line
        # Map collapsed positions to their indices in line.positions
        indices = []
        for pos in collapsed:
            pos_key = _position_to_string(pos)
            for idx, line_pos in enumerate(target_line.positions):
                if _position_to_string(line_pos) == pos_key:
                    indices.append(idx)
                    break

        if len(indices) != len(collapsed):
            # Some positions weren't found in line
            return False

        indices.sort()
        return all(indices[i + 1] == indices[i] + 1 for i in range(len(indices) - 1))

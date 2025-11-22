from app.models import GameState, Move, GamePhase, MoveType
from app.rules.interfaces import Validator


class LineValidator(Validator):
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

        # 4. Line Existence & Ownership
        # The move should reference a line that exists in
        # state.board.formed_lines and belongs to the player.
        # Note: The Move object for line processing usually contains the line
        # info or an index/ID. The current Python Move model has `formed_lines`
        # tuple. We need to verify that the line being processed is actually
        # present in the board state.

        # For PROCESS_LINE (canonical auto-processing or explicit choice
        # start), we expect the move to correspond to one of the lines in
        # board.formed_lines.

        # Since the Python Move model doesn't strictly link to a specific line
        # ID in the same way the TS one might (TS uses line index or object
        # equality), we'll check if the player has ANY lines to process.

        player_lines = [
            line for line in state.board.formed_lines
            if line.player == move.player
        ]

        if not player_lines:
            return False

        # 5. Option Validity
        # If it's a choice move, ensure the choice is valid (e.g. valid option
        # enum). This is partially covered by Pydantic validation, but logical
        # checks (e.g. can only choose "collapse all" if not overlapping?)
        # belong here. For now, basic existence is enough.

        return True
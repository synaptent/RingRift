"""RingRift game notation module.

Provides algebraic notation conversion for game records per GAME_NOTATION_SPEC.md.
"""

from app.notation.algebraic import (
    CODE_TO_MOVE_TYPE,
    MOVE_TYPE_TO_CODE,
    algebraic_to_move,
    algebraic_to_position,
    game_to_pgn,
    move_to_algebraic,
    moves_to_notation_list,
    parse_pgn,
    position_to_algebraic,
)

__all__ = [
    "CODE_TO_MOVE_TYPE",
    "MOVE_TYPE_TO_CODE",
    "algebraic_to_move",
    "algebraic_to_position",
    "game_to_pgn",
    "move_to_algebraic",
    "moves_to_notation_list",
    "parse_pgn",
    "position_to_algebraic",
]

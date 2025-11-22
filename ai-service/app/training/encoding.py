from typing import List, Union
from app.models import Move, BoardState, GameState
from app.ai.neural_net import NeuralNetAI, INVALID_MOVE_INDEX


def encode_legal_moves(
    moves: List[Move],
    neural_net: NeuralNetAI,
    board_context: Union[BoardState, GameState],
) -> List[int]:
    """
    Encode a list of legal moves into their policy indices for the given
    board context.

    The returned indices live in the fixed MAX_N×MAX_N policy head used by
    NeuralNetAI. Moves that cannot be encoded (INVALID_MOVE_INDEX) are
    filtered out.
    """
    encoded_moves: List[int] = []
    for m in moves:
        idx = neural_net.encode_move(m, board_context)
        if idx != INVALID_MOVE_INDEX:
            encoded_moves.append(idx)
    return encoded_moves
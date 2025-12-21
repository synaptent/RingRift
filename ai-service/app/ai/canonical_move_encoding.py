"""Canonical move encoding/decoding for all board types.

This module provides unified move encoding functions that dispatch to the
appropriate board-specific encoder. It serves as the single entry point for
converting between Move objects and policy indices.

Usage:
    from app.ai.canonical_move_encoding import (
        encode_move_for_board,
        decode_move_for_board,
    )

    # Encode a move to policy index
    idx = encode_move_for_board(move, board_or_state)

    # Decode policy index to components
    decoded = decode_move_for_board(policy_idx, board_type)

This module consolidates move encoding from _neural_net_legacy.py as part of
the December 2025 encoding consolidation effort.
"""

from __future__ import annotations

from app.models import BoardState, BoardType, GameState, Move

from .neural_net.constants import (
    HEX8_BOARD_SIZE,
    HEX_BOARD_SIZE,
    INVALID_MOVE_INDEX,
    P_HEX,
    POLICY_SIZE_HEX8,
)
from .neural_net.hex_encoding import ActionEncoderHex
from .neural_net.square_encoding import (
    ActionEncoderSquare8,
    ActionEncoderSquare19,
    DecodedPolicyIndex,
)

# Singleton encoder instances for performance
_encoder_square8 = ActionEncoderSquare8()
_encoder_square19 = ActionEncoderSquare19()
_encoder_hex8 = ActionEncoderHex(board_size=HEX8_BOARD_SIZE, policy_size=POLICY_SIZE_HEX8)
_encoder_hexagonal = ActionEncoderHex(board_size=HEX_BOARD_SIZE, policy_size=P_HEX)


def encode_move_for_board(
    move: Move,
    board: BoardState | GameState,
) -> int:
    """Encode a move to a policy index using board-type-specific encoding.

    Uses the proper policy layout for each board type:
    - SQUARE8: policy_size=7000 (compact 8x8 encoding)
    - SQUARE19: policy_size=67000 (19x19 encoding)
    - HEX8: policy_size=4500 (hex8 encoding)
    - HEXAGONAL: policy_size=91876 (hexagonal encoding)

    Args:
        move: The move to encode
        board: Board context for coordinate mapping (BoardState or GameState)

    Returns:
        Policy index in [0, policy_size) for the board type,
        or INVALID_MOVE_INDEX (-1) if the move cannot be encoded
    """
    if isinstance(board, GameState):
        board = board.board

    board_type = board.type

    if board_type == BoardType.SQUARE8:
        return _encoder_square8.encode_move(move, board)
    elif board_type == BoardType.SQUARE19:
        return _encoder_square19.encode_move(move, board)
    elif board_type == BoardType.HEX8:
        return _encoder_hex8.encode_move(move, board)
    elif board_type == BoardType.HEXAGONAL:
        return _encoder_hexagonal.encode_move(move, board)
    else:
        return INVALID_MOVE_INDEX


def decode_move_for_board(
    policy_idx: int,
    board_type: BoardType,
) -> DecodedPolicyIndex | None:
    """Decode a policy index back to its components for transformation.

    This is the inverse of encode_move_for_board, used for data augmentation
    on square boards where we need to transform policy indices under
    rotations and reflections.

    Args:
        policy_idx: The policy index to decode
        board_type: The board type (SQUARE8, SQUARE19, HEX8, or HEXAGONAL)

    Returns:
        DecodedPolicyIndex with decoded components, or None if index is invalid.
        For hex boards, returns None (hex uses different augmentation system).
    """
    if board_type == BoardType.SQUARE8:
        return _encoder_square8.decode_to_components(policy_idx)
    elif board_type == BoardType.SQUARE19:
        return _encoder_square19.decode_to_components(policy_idx)
    else:
        # Hex boards use a different transformation system
        # and don't currently support decode_to_components
        return None


def get_encoder_for_board(board_type: BoardType):
    """Get the appropriate action encoder for a board type.

    Args:
        board_type: The board type

    Returns:
        The action encoder instance for the board type
    """
    if board_type == BoardType.SQUARE8:
        return _encoder_square8
    elif board_type == BoardType.SQUARE19:
        return _encoder_square19
    elif board_type == BoardType.HEX8:
        return _encoder_hex8
    elif board_type == BoardType.HEXAGONAL:
        return _encoder_hexagonal
    return None


__all__ = [
    "DecodedPolicyIndex",
    "decode_move_for_board",
    "encode_move_for_board",
    "get_encoder_for_board",
]

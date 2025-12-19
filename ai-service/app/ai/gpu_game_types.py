"""GPU parallel games type definitions and utilities.

This module provides shared types and simple utilities for the GPU parallel
games system. Extracted from gpu_parallel_games.py for better modularity.

December 2025: Extracted as part of R4 refactoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple

import torch


# =============================================================================
# MPS Compatibility Helpers
# =============================================================================


def get_int_dtype(device: torch.device) -> torch.dtype:
    """Get the appropriate integer dtype for index tensors on the device.

    MPS doesn't support int16 for certain operations like index_put_,
    so we use int32 on MPS devices.
    """
    if device.type == "mps":
        return torch.int32
    return torch.int16


# =============================================================================
# Game Status and Move Enums
# =============================================================================


class GameStatus(IntEnum):
    """Status of a game in the batch."""
    ACTIVE = 0       # Game is ongoing
    COMPLETED = 1    # Game ended with a winner
    DRAW = 2         # Game ended in draw
    MAX_MOVES = 3    # Game ended due to move limit


class MoveType(IntEnum):
    """Types of moves that can be made.

    Values match app.models.MoveType for compatibility.
    """
    PLACEMENT = 0
    MOVEMENT = 1
    CAPTURE = 2
    LINE_FORMATION = 3
    TERRITORY_CLAIM = 4
    SKIP = 5
    NO_ACTION = 6
    RECOVERY_SLIDE = 7


class GamePhase(IntEnum):
    """Current phase of a game.

    Order matters for phase progression logic.
    """
    RING_PLACEMENT = 0    # Initial ring placement
    MOVEMENT = 1          # Normal movement/capture phase
    LINE_PROCESSING = 2   # Processing formed lines
    TERRITORY_PROCESSING = 3  # Processing territory claims
    END_TURN = 4          # End of turn processing


# =============================================================================
# Line Detection Types
# =============================================================================


@dataclass
class DetectedLine:
    """A detected marker line with metadata for processing."""
    positions: List[Tuple[int, int]]  # All marker positions in the line
    length: int                        # Total length of line
    is_overlength: bool               # True if len > required_length
    direction: Tuple[int, int]        # Direction vector (dy, dx)


def get_required_line_length(board_size: int, num_players: int) -> int:
    """Get required line length per RR-CANON-R120.

    Args:
        board_size: Board dimension
        num_players: Number of players

    Returns:
        Required line length (3 or 4)
    """
    # square8 (8x8) with 3-4 players uses line length 3, all others use 4
    if board_size == 8 and num_players >= 3:
        return 3
    return 4


# =============================================================================
# Move Constants
# =============================================================================

# Maximum number of stacks a position can have
MAX_STACK_HEIGHT = 8

# Direction vectors for square boards (8 directions)
SQUARE_DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),          (0, 1),
    (1, -1), (1, 0), (1, 1),
]

# Direction vectors for line checking (4 directions - no need to check both ways)
LINE_DIRECTIONS = [
    (0, 1),   # Horizontal
    (1, 0),   # Vertical
    (1, 1),   # Diagonal down-right
    (1, -1),  # Diagonal down-left
]


__all__ = [
    # Functions
    'get_int_dtype',
    'get_required_line_length',
    # Enums
    'GameStatus',
    'MoveType',
    'GamePhase',
    # Dataclasses
    'DetectedLine',
    # Constants
    'MAX_STACK_HEIGHT',
    'SQUARE_DIRECTIONS',
    'LINE_DIRECTIONS',
]

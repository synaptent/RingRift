"""GPU parallel games type definitions and utilities.

This module provides shared types and simple utilities for the GPU parallel
games system. Extracted from gpu_parallel_games.py for better modularity.

December 2025: Extracted as part of R4 refactoring.

MPS (Apple Silicon) Limitations:
--------------------------------
1. **index_put_ with accumulate=True**: Only supports float, int32, or bool.
   Use int32 instead of int8/int16 for tensors that need accumulate operations.
   Affected tensors: rings_in_hand, buried_rings, eliminated_rings, etc.

2. **Performance**: MPS can be slower than CPU due to:
   - Excessive .item() calls causing CPU-GPU synchronization (avoid in hot paths)
   - Small tensor operations where kernel launch overhead dominates
   - nonzero() operations which are expensive on GPU

   Profiling shows ~100x slowdown on MPS vs CPU for game simulation due to
   synchronization overhead. For production training, prefer CUDA or CPU batching.

3. **Dtype limitations**: Some operations don't support int16. Use get_int_dtype()
   to get the appropriate dtype for the current device.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

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

    Values 0-7 match legacy GPU types for backwards compatibility.
    Values 8+ are canonical types added December 2025 for parity.
    """
    # Legacy GPU types (0-7)
    PLACEMENT = 0
    MOVEMENT = 1
    CAPTURE = 2              # Generic capture (legacy) - use OVERTAKING_CAPTURE for canonical
    LINE_FORMATION = 3
    TERRITORY_CLAIM = 4
    SKIP = 5                 # Generic skip (legacy)
    NO_ACTION = 6            # Generic no-action (legacy) - use phase-specific for canonical
    RECOVERY_SLIDE = 7

    # Canonical phase-specific no-action types (8-11)
    NO_PLACEMENT_ACTION = 8
    NO_MOVEMENT_ACTION = 9
    NO_LINE_ACTION = 10
    NO_TERRITORY_ACTION = 11

    # Canonical capture types (12-14)
    OVERTAKING_CAPTURE = 12      # First capture in sequence
    CONTINUE_CAPTURE_SEGMENT = 13  # Chain capture continuation
    SKIP_CAPTURE = 14            # Voluntarily skip available capture

    # Canonical recovery types (15)
    SKIP_RECOVERY = 15           # Skip recovery slide

    # Canonical forced elimination (16)
    FORCED_ELIMINATION = 16      # Player eliminated due to no real actions

    # Canonical line/territory choice types (17-18)
    CHOOSE_LINE_OPTION = 17      # Line reward selection
    CHOOSE_TERRITORY_OPTION = 18  # Territory reward selection

    # Canonical skip placement (19)
    SKIP_PLACEMENT = 19          # Skip placement phase


class GamePhase(IntEnum):
    """Current phase of a game.

    Order matters for phase progression logic.
    Values 0-4 are legacy GPU phases for backwards compatibility.
    Values 5+ are canonical phases added December 2025 for parity.

    Canonical phase order:
    ring_placement → movement → capture → chain_capture → line_processing →
    territory_processing → forced_elimination → game_over
    (Recovery is a movement-phase action; RECOVERY here is GPU-internal only.)
    """
    # Legacy GPU phases (0-4)
    RING_PLACEMENT = 0       # Initial ring placement
    MOVEMENT = 1             # Normal movement phase (pre-capture)
    LINE_PROCESSING = 2      # Processing formed lines
    TERRITORY_PROCESSING = 3 # Processing territory claims
    END_TURN = 4             # End of turn processing (legacy)

    # Canonical phases (5-9) - December 2025
    CAPTURE = 5              # First capture opportunity
    CHAIN_CAPTURE = 6        # Continuation capture after first capture
    # GPU-internal recovery phase; exported as movement per canonical rules.
    RECOVERY = 7
    FORCED_ELIMINATION = 8   # Player had no real actions this turn
    GAME_OVER = 9            # Game has ended


# =============================================================================
# Line Detection Types
# =============================================================================


@dataclass
class DetectedLine:
    """A detected marker line with metadata for processing."""
    positions: list[tuple[int, int]]  # All marker positions in the line
    length: int                        # Total length of line
    is_overlength: bool               # True if len > required_length
    direction: tuple[int, int]        # Direction vector (dy, dx)


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
    'LINE_DIRECTIONS',
    # Constants
    'MAX_STACK_HEIGHT',
    'SQUARE_DIRECTIONS',
    # Dataclasses
    'DetectedLine',
    'GamePhase',
    # Enums
    'GameStatus',
    'MoveType',
    # Functions
    'get_int_dtype',
    'get_required_line_length',
]

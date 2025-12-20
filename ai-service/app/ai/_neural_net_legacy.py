"""Neural-network-backed AI implementation for RingRift.

This module implements the convolutional policy/value network used by the
Python AI service and the :class:`NeuralNetAI` wrapper that integrates it
with the shared :class:`BaseAI` interface.

The same model architecture is used for both inference (online play,
parity tests) and training. Behaviour is configured via :class:`AIConfig`
fields such as ``nn_model_id``, ``allow_fresh_weights``, and
``history_length``; see :class:`NeuralNetAI` for details.

Memory management
-----------------
To prevent OOM issues in long soak tests and selfplay runs, this module
uses a singleton model cache (via :mod:`.model_cache`) that shares model
instances across multiple :class:`NeuralNetAI` instances. Call
:func:`.model_cache.clear_model_cache` to release GPU/MPS memory between
games or soak batches.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
import logging
import os

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union
from datetime import datetime

from .base import BaseAI
from ..models import (
    GameState,
    Move,
    BoardType,
    Position,
    BoardState,
    MoveType,
    GamePhase,
)
from ..rules.geometry import BoardGeometry
from ..rules.core import get_rings_per_player
from .game_state_utils import (
    infer_num_players,
    infer_rings_per_player,
    select_threat_opponent,
)
from .model_cache import (
    get_cache_ref as _get_model_cache,
    evict_stale_models as _evict_stale_models,
    clear_model_cache,
    get_cached_model_count,
    strip_module_prefix as _strip_module_prefix,
)
from .neural_losses import (
    multi_player_value_loss,
    rank_distribution_loss,
    ranks_from_game_result,
)

logger = logging.getLogger(__name__)

# Reference to the shared model cache (for direct access in this module)
_MODEL_CACHE = _get_model_cache()

INVALID_MOVE_INDEX = -1
MAX_N = 19  # Canonical maximum side length for policy encoding (19x19 grid)

# =============================================================================
# Board-Specific Policy Sizes
# =============================================================================
#
# Each board type has an optimal policy head size based on its action space.
# Using board-specific sizes reduces wasted parameters and improves training.

# Square 8x8 Policy Layout:
#   Placement:       3 * 8 * 8 = 192
#   Movement:        8 * 8 * 8 * 7 = 3,584  (8 directions, max 7 distance)
#   Line Formation:  8 * 8 * 4 = 256
#   Territory Claim: 8 * 8 = 64
#   Skip Placement:  1
#   Swap Sides:      1
#   Line Choice:     4
#   Territory Choice: 64 * 8 * 4 = 2,048
#   Total: ~6,150 → 7,000 (with padding)
POLICY_SIZE_8x8 = 7000

# Square 19x19 Policy Layout (v2 - with robust territory choice encoding):
#   0-1082:       Placement (3 * 19 * 19 = 1,083)
#   1083-53066:   Movement/Capture (19 * 19 * 8 * 18 = 51,984)
#   53067-54510:  Line Formation (19 * 19 * 4 = 1,444)
#   54511-54871:  Territory Claim (19 * 19 = 361)
#   54872:        Skip Placement (1)
#   54873:        Swap Sides (1)
#   54874-54877:  Line Choice (4)
#   54878-66429:  Territory Choice (361 * 8 * 4 = 11,552)
#   Total: 66,430 → 67,000 (with padding)
POLICY_SIZE_19x19 = 67000

# Legacy alias for backwards compatibility
POLICY_SIZE = POLICY_SIZE_19x19

# Maximum number of players for multi-player value head
MAX_PLAYERS = 4

# Board type to policy size mapping
BOARD_POLICY_SIZES: Dict[BoardType, int] = {
    BoardType.SQUARE8: POLICY_SIZE_8x8,
    BoardType.SQUARE19: POLICY_SIZE_19x19,
    BoardType.HEX8: 4500,  # POLICY_SIZE_HEX8 defined below
    BoardType.HEXAGONAL: 91876,  # P_HEX defined below
}

# Board type to spatial size mapping
BOARD_SPATIAL_SIZES: Dict[BoardType, int] = {
    BoardType.SQUARE8: 8,
    BoardType.SQUARE19: 19,
    BoardType.HEX8: 9,  # HEX8_BOARD_SIZE defined below
    BoardType.HEXAGONAL: 25,  # HEX_BOARD_SIZE
}


def get_policy_size_for_board(board_type: BoardType) -> int:
    """Get the optimal policy head size for a board type."""
    return BOARD_POLICY_SIZES.get(board_type, POLICY_SIZE_19x19)


def get_spatial_size_for_board(board_type: BoardType) -> int:
    """Get the spatial (H, W) size for a board type."""
    return BOARD_SPATIAL_SIZES.get(board_type, 19)


# Hex-specific canonical geometry and policy layout constants.
#
# The canonical competitive hex board has radius N = 12, which yields
# 3N^2 + 3N + 1 = 469 cells. We embed this hex into a fixed 25×25
# bounding box (2N + 1 on each axis) using _to_canonical_xy /
# _from_canonical_xy, and define a dedicated hex action space of size
# P_HEX = 91_876 as documented in AI_ARCHITECTURE.md.
HEX_BOARD_SIZE = 25
HEX_MAX_DIST = HEX_BOARD_SIZE - 1  # 24 distance buckets (1..24)

# Canonical axial directions in the 2D embedding.
HEX_DIRS = [
    (1, 0),  # +q
    (0, 1),  # +r
    (-1, 1),  # -q + r
    (-1, 0),  # -q
    (0, -1),  # -r
    (1, -1),  # +q - r
]
NUM_HEX_DIRS = len(HEX_DIRS)

# Layout spans for the hex policy head (see AI_ARCHITECTURE.md):
#
# Placements: 25 × 25 × 3 = 1_875
# Movement/capture: 25 × 25 × 6 × 24 = 90_000
# Special: 1 (skip_placement)
# Total: P_HEX = 91_876
HEX_PLACEMENT_SPAN = HEX_BOARD_SIZE * HEX_BOARD_SIZE * 3
HEX_MOVEMENT_BASE = HEX_PLACEMENT_SPAN
HEX_MOVEMENT_SPAN = HEX_BOARD_SIZE * HEX_BOARD_SIZE * NUM_HEX_DIRS * HEX_MAX_DIST
HEX_SPECIAL_BASE = HEX_MOVEMENT_BASE + HEX_MOVEMENT_SPAN
P_HEX = HEX_SPECIAL_BASE + 1

# =============================================================================
# Hex8 Board Policy Constants (radius-4 hexagonal board, 61 cells)
# =============================================================================
#
# Hex8 is a smaller hexagonal board (radius=4) parallel to square8, just as
# hexagonal (radius=12) is parallel to square19. Uses the same 6-direction
# hex geometry but with a 9×9 bounding box.
#
# Hex8 Board: radius=4, cells=61 (3*16 + 12 + 1), bounding box 9×9
HEX8_BOARD_SIZE = 9  # 2 * 4 + 1
HEX8_RADIUS = 4
HEX8_MAX_DIST = HEX8_BOARD_SIZE - 1  # 8 distance buckets (1..8)

# Layout spans for the hex8 policy head:
#
# Placements: 9 × 9 × 3 = 243
# Movement/capture: 9 × 9 × 6 × 8 = 3,888
# Special: 1 (skip_placement)
# Total: P_HEX8 = 4,132 → 4,500 (with padding)
HEX8_PLACEMENT_SPAN = HEX8_BOARD_SIZE * HEX8_BOARD_SIZE * 3  # 243
HEX8_MOVEMENT_BASE = HEX8_PLACEMENT_SPAN
HEX8_MOVEMENT_SPAN = HEX8_BOARD_SIZE * HEX8_BOARD_SIZE * NUM_HEX_DIRS * HEX8_MAX_DIST  # 3,888
HEX8_SPECIAL_BASE = HEX8_MOVEMENT_BASE + HEX8_MOVEMENT_SPAN
P_HEX8 = HEX8_SPECIAL_BASE + 1  # 4,132
POLICY_SIZE_HEX8 = 4500  # Padded policy size

# =============================================================================
# Square Board Spatial Policy Constants (V3)
# =============================================================================
#
# These constants define the spatial policy layout for V3 architecture.
# V3 uses Conv1x1 spatial heads instead of GAP→FC policy heads.

# Square board directions (8 cardinal + diagonal)
NUM_SQUARE_DIRS = 8

# Maximum movement distance for each board type
MAX_DIST_SQUARE8 = 7  # Max diagonal: sqrt(7^2 + 7^2) ≈ 9.9
MAX_DIST_SQUARE19 = 18  # Max diagonal: sqrt(18^2 + 18^2) ≈ 25.5

# Line formation directions (horizontal, vertical, diagonal)
NUM_LINE_DIRS = 4

# Territory choice encoding dimensions
TERRITORY_SIZE_BUCKETS = 8
TERRITORY_MAX_PLAYERS = 4

# Square8 Policy Layout (V3):
#   Placement:       [0, 191]       = 3 * 8 * 8 = 192
#   Movement:        [192, 3775]    = 8 * 8 * 8 * 7 = 3,584
#   Line Formation:  [3776, 4031]   = 8 * 8 * 4 = 256
#   Territory Claim: [4032, 4095]   = 8 * 8 = 64
#   Skip Placement:  [4096]         = 1
#   Swap Sides:      [4097]         = 1
#   Line Choice:     [4098, 4101]   = 4
#   Territory Choice:[4102, 6149]   = 64 * 8 * 4 = 2,048
#   Extra Special:   [6150, 6156]   = canonical no/skip/forced actions
#   Total: 6,157 (POLICY_SIZE_8x8 = 7000 with padding)

SQUARE8_PLACEMENT_SPAN = 3 * 8 * 8  # 192
SQUARE8_MOVEMENT_BASE = SQUARE8_PLACEMENT_SPAN
SQUARE8_MOVEMENT_SPAN = 8 * 8 * NUM_SQUARE_DIRS * MAX_DIST_SQUARE8  # 3,584
SQUARE8_LINE_FORM_BASE = SQUARE8_MOVEMENT_BASE + SQUARE8_MOVEMENT_SPAN
SQUARE8_LINE_FORM_SPAN = 8 * 8 * NUM_LINE_DIRS  # 256
SQUARE8_TERRITORY_CLAIM_BASE = SQUARE8_LINE_FORM_BASE + SQUARE8_LINE_FORM_SPAN
SQUARE8_TERRITORY_CLAIM_SPAN = 8 * 8  # 64
SQUARE8_SPECIAL_BASE = SQUARE8_TERRITORY_CLAIM_BASE + SQUARE8_TERRITORY_CLAIM_SPAN
SQUARE8_SKIP_PLACEMENT_IDX = SQUARE8_SPECIAL_BASE  # 4096
SQUARE8_SWAP_SIDES_IDX = SQUARE8_SPECIAL_BASE + 1  # 4097
SQUARE8_SKIP_RECOVERY_IDX = SQUARE8_SPECIAL_BASE + 2  # 4098 (RR-CANON-R112)
SQUARE8_LINE_CHOICE_BASE = SQUARE8_SPECIAL_BASE + 3  # 4099
SQUARE8_LINE_CHOICE_SPAN = 4
SQUARE8_TERRITORY_CHOICE_BASE = SQUARE8_LINE_CHOICE_BASE + SQUARE8_LINE_CHOICE_SPAN
SQUARE8_TERRITORY_CHOICE_SPAN = 8 * 8 * TERRITORY_SIZE_BUCKETS * TERRITORY_MAX_PLAYERS  # 2,048
SQUARE8_EXTRA_SPECIAL_BASE = (
    SQUARE8_TERRITORY_CHOICE_BASE + SQUARE8_TERRITORY_CHOICE_SPAN
)
SQUARE8_NO_PLACEMENT_ACTION_IDX = SQUARE8_EXTRA_SPECIAL_BASE
SQUARE8_NO_MOVEMENT_ACTION_IDX = SQUARE8_EXTRA_SPECIAL_BASE + 1
SQUARE8_SKIP_CAPTURE_IDX = SQUARE8_EXTRA_SPECIAL_BASE + 2
SQUARE8_NO_LINE_ACTION_IDX = SQUARE8_EXTRA_SPECIAL_BASE + 3
SQUARE8_NO_TERRITORY_ACTION_IDX = SQUARE8_EXTRA_SPECIAL_BASE + 4
SQUARE8_SKIP_TERRITORY_PROCESSING_IDX = SQUARE8_EXTRA_SPECIAL_BASE + 5
SQUARE8_FORCED_ELIMINATION_IDX = SQUARE8_EXTRA_SPECIAL_BASE + 6
SQUARE8_EXTRA_SPECIAL_SPAN = 7

# Square19 Policy Layout (V3):
#   Placement:       [0, 1082]      = 3 * 19 * 19 = 1,083
#   Movement:        [1083, 53066]  = 19 * 19 * 8 * 18 = 51,984
#   Line Formation:  [53067, 54510] = 19 * 19 * 4 = 1,444
#   Territory Claim: [54511, 54871] = 19 * 19 = 361
#   Skip Placement:  [54872]        = 1
#   Swap Sides:      [54873]        = 1
#   Skip Recovery:   [54874]        = 1 (RR-CANON-R112)
#   Line Choice:     [54875, 54878] = 4
#   Territory Choice:[54879, 66430] = 361 * 8 * 4 = 11,552
#   Extra Special:   [66431, 66437] = canonical no/skip/forced actions
#   Total: 66,438 (POLICY_SIZE_19x19 = 67000 with padding)

SQUARE19_PLACEMENT_SPAN = 3 * 19 * 19  # 1,083
SQUARE19_MOVEMENT_BASE = SQUARE19_PLACEMENT_SPAN
SQUARE19_MOVEMENT_SPAN = 19 * 19 * NUM_SQUARE_DIRS * MAX_DIST_SQUARE19  # 51,984
SQUARE19_LINE_FORM_BASE = SQUARE19_MOVEMENT_BASE + SQUARE19_MOVEMENT_SPAN
SQUARE19_LINE_FORM_SPAN = 19 * 19 * NUM_LINE_DIRS  # 1,444
SQUARE19_TERRITORY_CLAIM_BASE = SQUARE19_LINE_FORM_BASE + SQUARE19_LINE_FORM_SPAN
SQUARE19_TERRITORY_CLAIM_SPAN = 19 * 19  # 361
SQUARE19_SPECIAL_BASE = SQUARE19_TERRITORY_CLAIM_BASE + SQUARE19_TERRITORY_CLAIM_SPAN
SQUARE19_SKIP_PLACEMENT_IDX = SQUARE19_SPECIAL_BASE  # 54872
SQUARE19_SWAP_SIDES_IDX = SQUARE19_SPECIAL_BASE + 1  # 54873
SQUARE19_SKIP_RECOVERY_IDX = SQUARE19_SPECIAL_BASE + 2  # 54874 (RR-CANON-R112)
SQUARE19_LINE_CHOICE_BASE = SQUARE19_SPECIAL_BASE + 3  # 54875
SQUARE19_LINE_CHOICE_SPAN = 4
SQUARE19_TERRITORY_CHOICE_BASE = SQUARE19_LINE_CHOICE_BASE + SQUARE19_LINE_CHOICE_SPAN
SQUARE19_TERRITORY_CHOICE_SPAN = 19 * 19 * TERRITORY_SIZE_BUCKETS * TERRITORY_MAX_PLAYERS  # 11,552
SQUARE19_EXTRA_SPECIAL_BASE = (
    SQUARE19_TERRITORY_CHOICE_BASE + SQUARE19_TERRITORY_CHOICE_SPAN
)
SQUARE19_NO_PLACEMENT_ACTION_IDX = SQUARE19_EXTRA_SPECIAL_BASE
SQUARE19_NO_MOVEMENT_ACTION_IDX = SQUARE19_EXTRA_SPECIAL_BASE + 1
SQUARE19_SKIP_CAPTURE_IDX = SQUARE19_EXTRA_SPECIAL_BASE + 2
SQUARE19_NO_LINE_ACTION_IDX = SQUARE19_EXTRA_SPECIAL_BASE + 3
SQUARE19_NO_TERRITORY_ACTION_IDX = SQUARE19_EXTRA_SPECIAL_BASE + 4
SQUARE19_SKIP_TERRITORY_PROCESSING_IDX = SQUARE19_EXTRA_SPECIAL_BASE + 5
SQUARE19_FORCED_ELIMINATION_IDX = SQUARE19_EXTRA_SPECIAL_BASE + 6
SQUARE19_EXTRA_SPECIAL_SPAN = 7


def encode_move_for_board(
    move: "Move",
    board: Union["BoardState", "GameState"],
) -> int:
    """
    Encode a move to a policy index using board-type-specific encoding.

    Unlike NeuralNetAI.encode_move which uses a fixed MAX_N=19 layout for all boards,
    this function uses the proper policy layout for each board type:
    - SQUARE8: policy_size=7000 (compact 8x8 encoding)
    - SQUARE19: policy_size=67000 (19x19 encoding)
    - HEXAGONAL: policy_size=91876 (hex-specific encoding)

    This should be used for selfplay data generation to create training data
    with correct policy sizes.

    Parameters
    ----------
    move : Move
        The move to encode
    board : BoardState or GameState
        Board context for coordinate mapping

    Returns
    -------
    int
        Policy index in [0, policy_size) for the board type,
        or INVALID_MOVE_INDEX (-1) if the move cannot be encoded
    """
    if isinstance(board, GameState):
        board = board.board

    board_type = board.type

    # Dispatch to board-specific encoder
    if board_type == BoardType.SQUARE8:
        return _encode_move_square8(move, board)
    elif board_type == BoardType.SQUARE19:
        return _encode_move_square19(move, board)
    elif board_type == BoardType.HEX8:
        # Use the hex-specific encoder with hex8 parameters (policy_size=4500)
        hex8_encoder = ActionEncoderHex(board_size=HEX8_BOARD_SIZE, policy_size=POLICY_SIZE_HEX8)
        return hex8_encoder.encode_move(move, board)
    elif board_type == BoardType.HEXAGONAL:
        # Use the hex-specific encoder (policy_size=91876)
        hex_encoder = ActionEncoderHex(board_size=HEX_BOARD_SIZE, policy_size=P_HEX)
        return hex_encoder.encode_move(move, board)
    else:
        return INVALID_MOVE_INDEX


def _line_anchor_position(move: "Move") -> Optional[Position]:
    """Best-effort anchor for line-processing style moves."""
    if move.to is not None:
        return move.to
    if move.formed_lines:
        try:
            line = move.formed_lines[0]
            if hasattr(line, "positions") and line.positions:
                return line.positions[0]
        except Exception:
            return None
    return None


def _encode_move_square8(move: "Move", board: "BoardState") -> int:
    """Encode move for square8 board using compact 8x8 policy layout (max ~7000)."""
    N = 8  # Board size
    MAX_DIST = MAX_DIST_SQUARE8  # 7
    move_type = move.type.value if hasattr(move.type, "value") else str(move.type)

    # Placement: 0..191 (3 * 8 * 8)
    if move_type == "place_ring":
        cx, cy = move.to.x, move.to.y
        if not (0 <= cx < N and 0 <= cy < N):
            return INVALID_MOVE_INDEX
        pos_idx = cy * N + cx
        count_idx = (move.placement_count or 1) - 1
        return pos_idx * 3 + count_idx

    # Movement: 192..3775 (8 * 8 * 8 * 7)
    if move_type in [
        "move_stack", "move_ring", "overtaking_capture",
        "chain_capture", "continue_capture_segment", "recovery_slide",
    ]:
        if not move.from_pos:
            return INVALID_MOVE_INDEX
        cfx, cfy = move.from_pos.x, move.from_pos.y
        ctx, cty = move.to.x, move.to.y

        if not (0 <= cfx < N and 0 <= cfy < N and 0 <= ctx < N and 0 <= cty < N):
            return INVALID_MOVE_INDEX

        from_idx = cfy * N + cfx
        dx, dy = ctx - cfx, cty - cfy
        dist = max(abs(dx), abs(dy))
        if dist == 0 or dist > MAX_DIST:
            return INVALID_MOVE_INDEX

        dir_x = dx // dist if dist > 0 else 0
        dir_y = dy // dist if dist > 0 else 0

        dirs = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        try:
            dir_idx = dirs.index((dir_x, dir_y))
        except ValueError:
            return INVALID_MOVE_INDEX

        return SQUARE8_MOVEMENT_BASE + from_idx * (8 * MAX_DIST) + dir_idx * MAX_DIST + (dist - 1)

    # Line formation: 3776..4031
    if move_type in ("line_formation", "process_line"):
        line_pos = _line_anchor_position(move)
        if line_pos is None:
            return INVALID_MOVE_INDEX
        cx, cy = line_pos.x, line_pos.y
        if not (0 <= cx < N and 0 <= cy < N):
            return INVALID_MOVE_INDEX
        pos_idx = cy * N + cx
        return SQUARE8_LINE_FORM_BASE + pos_idx * 4  # 4 directions

    # Territory claim: 4032..4095
    if move_type in ("territory_claim", "eliminate_rings_from_stack"):
        if move.to is None:
            return INVALID_MOVE_INDEX
        cx, cy = move.to.x, move.to.y
        if not (0 <= cx < N and 0 <= cy < N):
            return INVALID_MOVE_INDEX
        pos_idx = cy * N + cx
        return SQUARE8_TERRITORY_CLAIM_BASE + pos_idx

    # Special actions
    if move_type == "skip_placement":
        return SQUARE8_SKIP_PLACEMENT_IDX

    if move_type == "swap_sides":
        return SQUARE8_SWAP_SIDES_IDX

    if move_type == "skip_recovery":
        return SQUARE8_SKIP_RECOVERY_IDX

    if move_type == "no_placement_action":
        return SQUARE8_NO_PLACEMENT_ACTION_IDX

    if move_type == "no_movement_action":
        return SQUARE8_NO_MOVEMENT_ACTION_IDX

    if move_type == "skip_capture":
        return SQUARE8_SKIP_CAPTURE_IDX

    if move_type == "no_line_action":
        return SQUARE8_NO_LINE_ACTION_IDX

    if move_type == "no_territory_action":
        return SQUARE8_NO_TERRITORY_ACTION_IDX

    if move_type == "skip_territory_processing":
        return SQUARE8_SKIP_TERRITORY_PROCESSING_IDX

    if move_type == "forced_elimination":
        return SQUARE8_FORCED_ELIMINATION_IDX

    # Line choice: 4099..4102
    if move_type in ("choose_line_reward", "choose_line_option"):
        option = (move.placement_count or 1) - 1
        option = max(0, min(3, option))
        return SQUARE8_LINE_CHOICE_BASE + option

    # Territory choice: 4103..6150
    if move_type in ("choose_territory_option", "process_territory_region"):
        canonical_pos = move.to
        region_size = 1
        controlling_player = move.player

        if move.disconnected_regions:
            regions = list(move.disconnected_regions)
            if regions:
                region = regions[0]
                if hasattr(region, "spaces") and region.spaces:
                    spaces = list(region.spaces)
                    region_size = len(spaces)
                    canonical_pos = min(spaces, key=lambda p: (p.y, p.x))
                if hasattr(region, "controlling_player"):
                    controlling_player = region.controlling_player

        if canonical_pos is None:
            return INVALID_MOVE_INDEX
        cx, cy = canonical_pos.x, canonical_pos.y
        if not (0 <= cx < N and 0 <= cy < N):
            return INVALID_MOVE_INDEX

        pos_idx = cy * N + cx
        size_bucket = min(region_size - 1, TERRITORY_SIZE_BUCKETS - 1)
        player_idx = (controlling_player - 1) % TERRITORY_MAX_PLAYERS

        return (
            SQUARE8_TERRITORY_CHOICE_BASE
            + pos_idx * (TERRITORY_SIZE_BUCKETS * TERRITORY_MAX_PLAYERS)
            + size_bucket * TERRITORY_MAX_PLAYERS
            + player_idx
        )

    return INVALID_MOVE_INDEX


def _encode_move_square19(move: "Move", board: "BoardState") -> int:
    """Encode move for square19 board using 19x19 policy layout (max ~67000)."""
    N = 19  # Board size
    MAX_DIST = MAX_DIST_SQUARE19  # 18
    move_type = move.type.value if hasattr(move.type, "value") else str(move.type)

    # Placement: 0..1082 (3 * 19 * 19)
    if move_type == "place_ring":
        cx, cy = move.to.x, move.to.y
        if not (0 <= cx < N and 0 <= cy < N):
            return INVALID_MOVE_INDEX
        pos_idx = cy * N + cx
        count_idx = (move.placement_count or 1) - 1
        return pos_idx * 3 + count_idx

    # Movement: 1083..53066
    if move_type in [
        "move_stack", "move_ring", "overtaking_capture",
        "chain_capture", "continue_capture_segment", "recovery_slide",
    ]:
        if not move.from_pos:
            return INVALID_MOVE_INDEX
        cfx, cfy = move.from_pos.x, move.from_pos.y
        ctx, cty = move.to.x, move.to.y

        if not (0 <= cfx < N and 0 <= cfy < N and 0 <= ctx < N and 0 <= cty < N):
            return INVALID_MOVE_INDEX

        from_idx = cfy * N + cfx
        dx, dy = ctx - cfx, cty - cfy
        dist = max(abs(dx), abs(dy))
        if dist == 0 or dist > MAX_DIST:
            return INVALID_MOVE_INDEX

        dir_x = dx // dist if dist > 0 else 0
        dir_y = dy // dist if dist > 0 else 0

        dirs = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        try:
            dir_idx = dirs.index((dir_x, dir_y))
        except ValueError:
            return INVALID_MOVE_INDEX

        return SQUARE19_MOVEMENT_BASE + from_idx * (8 * MAX_DIST) + dir_idx * MAX_DIST + (dist - 1)

    # Line formation: 53067..54510
    if move_type in ("line_formation", "process_line"):
        line_pos = _line_anchor_position(move)
        if line_pos is None:
            return INVALID_MOVE_INDEX
        cx, cy = line_pos.x, line_pos.y
        if not (0 <= cx < N and 0 <= cy < N):
            return INVALID_MOVE_INDEX
        pos_idx = cy * N + cx
        return SQUARE19_LINE_FORM_BASE + pos_idx * 4

    # Territory claim: 54511..54871
    if move_type in ("territory_claim", "eliminate_rings_from_stack"):
        if move.to is None:
            return INVALID_MOVE_INDEX
        cx, cy = move.to.x, move.to.y
        if not (0 <= cx < N and 0 <= cy < N):
            return INVALID_MOVE_INDEX
        pos_idx = cy * N + cx
        return SQUARE19_TERRITORY_CLAIM_BASE + pos_idx

    # Special actions
    if move_type == "skip_placement":
        return SQUARE19_SKIP_PLACEMENT_IDX

    if move_type == "swap_sides":
        return SQUARE19_SWAP_SIDES_IDX

    if move_type == "skip_recovery":
        return SQUARE19_SKIP_RECOVERY_IDX

    if move_type == "no_placement_action":
        return SQUARE19_NO_PLACEMENT_ACTION_IDX

    if move_type == "no_movement_action":
        return SQUARE19_NO_MOVEMENT_ACTION_IDX

    if move_type == "skip_capture":
        return SQUARE19_SKIP_CAPTURE_IDX

    if move_type == "no_line_action":
        return SQUARE19_NO_LINE_ACTION_IDX

    if move_type == "no_territory_action":
        return SQUARE19_NO_TERRITORY_ACTION_IDX

    if move_type == "skip_territory_processing":
        return SQUARE19_SKIP_TERRITORY_PROCESSING_IDX

    if move_type == "forced_elimination":
        return SQUARE19_FORCED_ELIMINATION_IDX

    # Line choice: 54875..54878
    if move_type in ("choose_line_reward", "choose_line_option"):
        option = (move.placement_count or 1) - 1
        option = max(0, min(3, option))
        return SQUARE19_LINE_CHOICE_BASE + option

    # Territory choice: 54879..66430
    if move_type in ("choose_territory_option", "process_territory_region"):
        canonical_pos = move.to
        region_size = 1
        controlling_player = move.player

        if move.disconnected_regions:
            regions = list(move.disconnected_regions)
            if regions:
                region = regions[0]
                if hasattr(region, "spaces") and region.spaces:
                    spaces = list(region.spaces)
                    region_size = len(spaces)
                    canonical_pos = min(spaces, key=lambda p: (p.y, p.x))
                if hasattr(region, "controlling_player"):
                    controlling_player = region.controlling_player

        if canonical_pos is None:
            return INVALID_MOVE_INDEX
        cx, cy = canonical_pos.x, canonical_pos.y
        if not (0 <= cx < N and 0 <= cy < N):
            return INVALID_MOVE_INDEX

        pos_idx = cy * N + cx
        size_bucket = min(region_size - 1, TERRITORY_SIZE_BUCKETS - 1)
        player_idx = (controlling_player - 1) % TERRITORY_MAX_PLAYERS

        return (
            SQUARE19_TERRITORY_CHOICE_BASE
            + pos_idx * (TERRITORY_SIZE_BUCKETS * TERRITORY_MAX_PLAYERS)
            + size_bucket * TERRITORY_MAX_PLAYERS
            + player_idx
        )

    return INVALID_MOVE_INDEX


# ---------------------------------------------------------------------------
# decode_move_for_board: Inverse of encode_move_for_board for data augmentation
# ---------------------------------------------------------------------------

@dataclass
class DecodedPolicyIndex:
    """Decoded policy index components for transformation.

    This structure contains enough information to apply symmetry transformations
    (rotations, reflections) and re-encode the policy index.
    """
    action_type: str  # "placement", "movement", "line_formation", etc.
    board_size: int   # 8 for square8, 19 for square19
    x: int = 0        # Primary position x
    y: int = 0        # Primary position y
    count_idx: int = 0     # For placement: (placement_count - 1)
    dir_idx: int = 0       # For movement: direction index 0-7
    dist: int = 0          # For movement: distance 1..MAX_DIST
    option: int = 0        # For line/territory choice
    size_bucket: int = 0   # For territory choice
    player_idx: int = 0    # For territory choice
    is_special: bool = False  # Special actions don't transform


# Direction vectors for square boards (8 directions)
SQUARE_DIRS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


def decode_move_for_board(
    policy_idx: int,
    board_type: BoardType,
) -> Optional[DecodedPolicyIndex]:
    """
    Decode a policy index back to its components for transformation.

    This is the inverse of encode_move_for_board, used for data augmentation
    on square boards where we need to transform policy indices under
    rotations and reflections.

    Parameters
    ----------
    policy_idx : int
        The policy index to decode
    board_type : BoardType
        The board type (SQUARE8 or SQUARE19)

    Returns
    -------
    DecodedPolicyIndex or None
        Decoded components, or None if index is invalid
    """
    if board_type == BoardType.SQUARE8:
        return _decode_move_square8(policy_idx)
    elif board_type == BoardType.SQUARE19:
        return _decode_move_square19(policy_idx)
    else:
        # Hex boards use different augmentation system
        return None


def _decode_move_square8(idx: int) -> Optional[DecodedPolicyIndex]:
    """Decode square8 policy index."""
    N = 8
    MAX_DIST = MAX_DIST_SQUARE8

    # Placement: [0, 191]
    if idx < SQUARE8_MOVEMENT_BASE:
        pos_idx = idx // 3
        count_idx = idx % 3
        x = pos_idx % N
        y = pos_idx // N
        return DecodedPolicyIndex(
            action_type="placement", board_size=N,
            x=x, y=y, count_idx=count_idx
        )

    # Movement: [192, 3775]
    if idx < SQUARE8_LINE_FORM_BASE:
        rel_idx = idx - SQUARE8_MOVEMENT_BASE
        from_idx = rel_idx // (8 * MAX_DIST)
        remainder = rel_idx % (8 * MAX_DIST)
        dir_idx = remainder // MAX_DIST
        dist = (remainder % MAX_DIST) + 1
        x = from_idx % N
        y = from_idx // N
        return DecodedPolicyIndex(
            action_type="movement", board_size=N,
            x=x, y=y, dir_idx=dir_idx, dist=dist
        )

    # Line formation: [3776, 4031]
    if idx < SQUARE8_TERRITORY_CLAIM_BASE:
        rel_idx = idx - SQUARE8_LINE_FORM_BASE
        pos_idx = rel_idx // 4
        dir_idx = rel_idx % 4
        x = pos_idx % N
        y = pos_idx // N
        return DecodedPolicyIndex(
            action_type="line_formation", board_size=N,
            x=x, y=y, dir_idx=dir_idx
        )

    # Territory claim: [4032, 4095]
    if idx < SQUARE8_SPECIAL_BASE:
        pos_idx = idx - SQUARE8_TERRITORY_CLAIM_BASE
        x = pos_idx % N
        y = pos_idx // N
        return DecodedPolicyIndex(
            action_type="territory_claim", board_size=N,
            x=x, y=y
        )

    # Special actions: skip_placement, swap_sides, skip_recovery
    if idx == SQUARE8_SKIP_PLACEMENT_IDX:
        return DecodedPolicyIndex(action_type="skip_placement", board_size=N, is_special=True)
    if idx == SQUARE8_SWAP_SIDES_IDX:
        return DecodedPolicyIndex(action_type="swap_sides", board_size=N, is_special=True)
    if idx == SQUARE8_SKIP_RECOVERY_IDX:
        return DecodedPolicyIndex(action_type="skip_recovery", board_size=N, is_special=True)

    # Line choice: [4099, 4102]
    if SQUARE8_LINE_CHOICE_BASE <= idx < SQUARE8_TERRITORY_CHOICE_BASE:
        option = idx - SQUARE8_LINE_CHOICE_BASE
        return DecodedPolicyIndex(
            action_type="line_choice", board_size=N,
            option=option, is_special=True
        )

    # Territory choice: [4103, 6150]
    if SQUARE8_TERRITORY_CHOICE_BASE <= idx < SQUARE8_TERRITORY_CHOICE_BASE + SQUARE8_TERRITORY_CHOICE_SPAN:
        rel_idx = idx - SQUARE8_TERRITORY_CHOICE_BASE
        pos_idx = rel_idx // (TERRITORY_SIZE_BUCKETS * TERRITORY_MAX_PLAYERS)
        remainder = rel_idx % (TERRITORY_SIZE_BUCKETS * TERRITORY_MAX_PLAYERS)
        size_bucket = remainder // TERRITORY_MAX_PLAYERS
        player_idx = remainder % TERRITORY_MAX_PLAYERS
        x = pos_idx % N
        y = pos_idx // N
        return DecodedPolicyIndex(
            action_type="territory_choice", board_size=N,
            x=x, y=y, size_bucket=size_bucket, player_idx=player_idx
        )

    if (
        SQUARE8_EXTRA_SPECIAL_BASE
        <= idx
        < SQUARE8_EXTRA_SPECIAL_BASE + SQUARE8_EXTRA_SPECIAL_SPAN
    ):
        offset = idx - SQUARE8_EXTRA_SPECIAL_BASE
        action_types = (
            "no_placement_action",
            "no_movement_action",
            "skip_capture",
            "no_line_action",
            "no_territory_action",
            "skip_territory_processing",
            "forced_elimination",
        )
        if 0 <= offset < len(action_types):
            return DecodedPolicyIndex(
                action_type=action_types[offset], board_size=N, is_special=True
            )

    return None


def _decode_move_square19(idx: int) -> Optional[DecodedPolicyIndex]:
    """Decode square19 policy index."""
    N = 19
    MAX_DIST = MAX_DIST_SQUARE19

    # Placement: [0, 1082]
    if idx < SQUARE19_MOVEMENT_BASE:
        pos_idx = idx // 3
        count_idx = idx % 3
        x = pos_idx % N
        y = pos_idx // N
        return DecodedPolicyIndex(
            action_type="placement", board_size=N,
            x=x, y=y, count_idx=count_idx
        )

    # Movement: [1083, 53066]
    if idx < SQUARE19_LINE_FORM_BASE:
        rel_idx = idx - SQUARE19_MOVEMENT_BASE
        from_idx = rel_idx // (8 * MAX_DIST)
        remainder = rel_idx % (8 * MAX_DIST)
        dir_idx = remainder // MAX_DIST
        dist = (remainder % MAX_DIST) + 1
        x = from_idx % N
        y = from_idx // N
        return DecodedPolicyIndex(
            action_type="movement", board_size=N,
            x=x, y=y, dir_idx=dir_idx, dist=dist
        )

    # Line formation: [53067, 54510]
    if idx < SQUARE19_TERRITORY_CLAIM_BASE:
        rel_idx = idx - SQUARE19_LINE_FORM_BASE
        pos_idx = rel_idx // 4
        dir_idx = rel_idx % 4
        x = pos_idx % N
        y = pos_idx // N
        return DecodedPolicyIndex(
            action_type="line_formation", board_size=N,
            x=x, y=y, dir_idx=dir_idx
        )

    # Territory claim: [54511, 54871]
    if idx < SQUARE19_SPECIAL_BASE:
        pos_idx = idx - SQUARE19_TERRITORY_CLAIM_BASE
        x = pos_idx % N
        y = pos_idx // N
        return DecodedPolicyIndex(
            action_type="territory_claim", board_size=N,
            x=x, y=y
        )

    # Special actions
    if idx == SQUARE19_SKIP_PLACEMENT_IDX:
        return DecodedPolicyIndex(action_type="skip_placement", board_size=N, is_special=True)
    if idx == SQUARE19_SWAP_SIDES_IDX:
        return DecodedPolicyIndex(action_type="swap_sides", board_size=N, is_special=True)
    if idx == SQUARE19_SKIP_RECOVERY_IDX:
        return DecodedPolicyIndex(action_type="skip_recovery", board_size=N, is_special=True)

    # Line choice: [54875, 54878]
    if SQUARE19_LINE_CHOICE_BASE <= idx < SQUARE19_TERRITORY_CHOICE_BASE:
        option = idx - SQUARE19_LINE_CHOICE_BASE
        return DecodedPolicyIndex(
            action_type="line_choice", board_size=N,
            option=option, is_special=True
        )

    # Territory choice: [54879, 66430]
    if SQUARE19_TERRITORY_CHOICE_BASE <= idx < SQUARE19_TERRITORY_CHOICE_BASE + SQUARE19_TERRITORY_CHOICE_SPAN:
        rel_idx = idx - SQUARE19_TERRITORY_CHOICE_BASE
        pos_idx = rel_idx // (TERRITORY_SIZE_BUCKETS * TERRITORY_MAX_PLAYERS)
        remainder = rel_idx % (TERRITORY_SIZE_BUCKETS * TERRITORY_MAX_PLAYERS)
        size_bucket = remainder // TERRITORY_MAX_PLAYERS
        player_idx = remainder % TERRITORY_MAX_PLAYERS
        x = pos_idx % N
        y = pos_idx // N
        return DecodedPolicyIndex(
            action_type="territory_choice", board_size=N,
            x=x, y=y, size_bucket=size_bucket, player_idx=player_idx
        )

    if (
        SQUARE19_EXTRA_SPECIAL_BASE
        <= idx
        < SQUARE19_EXTRA_SPECIAL_BASE + SQUARE19_EXTRA_SPECIAL_SPAN
    ):
        offset = idx - SQUARE19_EXTRA_SPECIAL_BASE
        action_types = (
            "no_placement_action",
            "no_movement_action",
            "skip_capture",
            "no_line_action",
            "no_territory_action",
            "skip_territory_processing",
            "forced_elimination",
        )
        if 0 <= offset < len(action_types):
            return DecodedPolicyIndex(
                action_type=action_types[offset], board_size=N, is_special=True
            )

    return None


def transform_policy_index_square(
    policy_idx: int,
    board_type: BoardType,
    rotation: int,
    flip_horizontal: bool,
) -> int:
    """
    Transform a policy index under rotation and reflection.

    Used for data augmentation on square boards. Applies the same
    transformation to the policy that would be applied to the board state.

    Parameters
    ----------
    policy_idx : int
        Original policy index
    board_type : BoardType
        SQUARE8 or SQUARE19
    rotation : int
        Number of 90-degree clockwise rotations (0-3)
    flip_horizontal : bool
        Whether to flip horizontally (before rotation)

    Returns
    -------
    int
        Transformed policy index, or original if transformation fails
    """
    decoded = decode_move_for_board(policy_idx, board_type)
    if decoded is None:
        return policy_idx

    # Special actions don't transform
    if decoded.is_special:
        return policy_idx

    N = decoded.board_size
    x, y = decoded.x, decoded.y

    # Apply flip first (horizontal = flip x)
    if flip_horizontal:
        x = N - 1 - x

    # Apply rotation (clockwise)
    for _ in range(rotation % 4):
        x, y = N - 1 - y, x

    # Re-encode based on action type
    if decoded.action_type == "placement":
        pos_idx = y * N + x
        return pos_idx * 3 + decoded.count_idx

    if decoded.action_type == "movement":
        # Transform direction as well
        dir_idx = decoded.dir_idx
        dx, dy = SQUARE_DIRS[dir_idx]

        # Apply flip to direction
        if flip_horizontal:
            dx = -dx

        # Apply rotation to direction
        for _ in range(rotation % 4):
            dx, dy = -dy, dx

        # Find new direction index
        try:
            new_dir_idx = SQUARE_DIRS.index((dx, dy))
        except ValueError:
            return policy_idx  # Shouldn't happen

        from_idx = y * N + x
        MAX_DIST = MAX_DIST_SQUARE8 if N == 8 else MAX_DIST_SQUARE19
        BASE = SQUARE8_MOVEMENT_BASE if N == 8 else SQUARE19_MOVEMENT_BASE
        return BASE + from_idx * (8 * MAX_DIST) + new_dir_idx * MAX_DIST + (decoded.dist - 1)

    if decoded.action_type == "line_formation":
        # Transform direction for line formation
        # Line directions are: horizontal(0), vertical(1), diagonal1(2), diagonal2(3)
        # Need to map these under transformation
        line_dir = decoded.dir_idx

        # Line direction vectors: horizontal, vertical, diag down-right, diag down-left
        line_dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        ldx, ldy = line_dirs[line_dir]

        if flip_horizontal:
            ldx = -ldx

        for _ in range(rotation % 4):
            ldx, ldy = -ldy, ldx

        # Normalize direction (lines are symmetric)
        if ldx < 0 or (ldx == 0 and ldy < 0):
            ldx, ldy = -ldx, -ldy

        try:
            new_line_dir = line_dirs.index((ldx, ldy))
        except ValueError:
            # Try normalized forms
            if (ldx, ldy) == (0, 1) or (ldx, ldy) == (0, -1):
                new_line_dir = 1  # vertical
            elif (ldx, ldy) == (1, 0) or (ldx, ldy) == (-1, 0):
                new_line_dir = 0  # horizontal
            elif ldx * ldy > 0:
                new_line_dir = 2  # diag down-right
            else:
                new_line_dir = 3  # diag down-left

        pos_idx = y * N + x
        BASE = SQUARE8_LINE_FORM_BASE if N == 8 else SQUARE19_LINE_FORM_BASE
        return BASE + pos_idx * 4 + new_line_dir

    if decoded.action_type == "territory_claim":
        pos_idx = y * N + x
        BASE = SQUARE8_TERRITORY_CLAIM_BASE if N == 8 else SQUARE19_TERRITORY_CLAIM_BASE
        return BASE + pos_idx

    if decoded.action_type == "territory_choice":
        pos_idx = y * N + x
        BASE = SQUARE8_TERRITORY_CHOICE_BASE if N == 8 else SQUARE19_TERRITORY_CHOICE_BASE
        return (
            BASE
            + pos_idx * (TERRITORY_SIZE_BUCKETS * TERRITORY_MAX_PLAYERS)
            + decoded.size_bucket * TERRITORY_MAX_PLAYERS
            + decoded.player_idx
        )

    return policy_idx


def _infer_board_size(board: Union[BoardState, GameState]) -> int:
    """
    Infer the canonical 2D board_size for CNN feature tensors.

    For SQUARE8: 8
    For SQUARE19: 19
    For HEXAGONAL: 2 * radius + 1, where radius = board.size - 1

    The returned value is the height/width of the (10, board_size, board_size)
    feature planes used by the CNN. Raises if the logical size exceeds MAX_N
    for square boards.
    """
    # Allow a GameState to be passed directly
    if isinstance(board, GameState):
        board = board.board

    board_type = board.type

    if board_type == BoardType.SQUARE8:
        return 8
    if board_type == BoardType.SQUARE19:
        return 19
    if board_type == BoardType.HEXAGONAL:
        radius = board.size - 1  # board.size=13 → radius=12
        return 2 * radius + 1    # → bounding box = 25
    if board_type == BoardType.HEX8:
        radius = board.size - 1  # board.size=5 → radius=4
        return 2 * radius + 1    # → bounding box = 9

    # Defensive fallback: use board.size but guard against unsupported sizes
    size = getattr(board, "size", 8)
    if size > MAX_N:
        raise ValueError(f"Unsupported board size {size}; MAX_N={MAX_N} is the current " "canonical maximum.")
    return int(size)


def _pos_from_key(pos_key: str) -> Position:
    """Parse a BoardState dict key like 'x,y' or 'x,y,z' into a Position."""
    parts = pos_key.split(",")
    if len(parts) == 2:
        x, y = int(parts[0]), int(parts[1])
        return Position(x=x, y=y)
    if len(parts) == 3:
        x, y, z = int(parts[0]), int(parts[1]), int(parts[2])
        return Position(x=x, y=y, z=z)
    raise ValueError(f"Invalid position key: {pos_key!r}")


def _to_canonical_xy(board: BoardState, pos: Position) -> tuple[int, int]:
    """
    Map a logical Position on this board into canonical (cx, cy) in
    [0, board_size) × [0, board_size), where board_size depends on
    board.type and board.size.

    For SQUARE8/SQUARE19: return (pos.x, pos.y) directly.

    For HEXAGONAL:
      - Interpret pos.(x,y,z) as cube/axial coords where x,y lie in
        [-radius, radius].
      - Let radius = board.size - 1.
      - Map x → cx = x + radius, y → cy = y + radius.
      - Return (cx, cy).
    """
    # We still allow callers to pass a GameState into _infer_board_size, but
    # here we require a BoardState to access geometry metadata consistently.
    if board.type in (BoardType.SQUARE8, BoardType.SQUARE19):
        return pos.x, pos.y

    if board.type in (BoardType.HEXAGONAL, BoardType.HEX8):
        # Hex boards use cube coordinates where x,y in [-radius, radius]
        # Map to bounding box [0, 2*radius+1) via cx = x + radius
        radius = board.size - 1  # HEXAGONAL: 12, HEX8: 4
        cx = pos.x + radius
        cy = pos.y + radius
        return cx, cy

    # Fallback: treat as generic square coordinates.
    return pos.x, pos.y


def _from_canonical_xy(
    board: BoardState,
    cx: int,
    cy: int,
) -> Optional[Position]:
    """
    Inverse of _to_canonical_xy.

    Returns a Position instance whose coordinates lie on this board, or None
    if (cx, cy) is outside [0, board_size) × [0, board_size).

    For HEXAGONAL:
      - radius = board.size - 1
      - x = cx - radius, y = cy - radius, z = -x - y
    """
    board_size = _infer_board_size(board)
    if not (0 <= cx < board_size and 0 <= cy < board_size):
        return None

    if board.type in (BoardType.SQUARE8, BoardType.SQUARE19):
        return Position(x=cx, y=cy)

    if board.type in (BoardType.HEXAGONAL, BoardType.HEX8):
        # Inverse of _to_canonical_xy for hex boards
        # Map from bounding box [0, 2*radius+1) back to cube coords [-radius, radius]
        radius = board.size - 1  # HEXAGONAL: 12, HEX8: 4
        x = cx - radius
        y = cy - radius
        z = -x - y
        return Position(x=x, y=y, z=z)

    # Fallback generic square position.
    return Position(x=cx, y=cy)


class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutions and skip connection."""

    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class SEResidualBlock(nn.Module):
    """Squeeze-and-Excitation enhanced residual block for v2 architectures.

    SE blocks improve global pattern recognition by adaptively recalibrating
    channel-wise feature responses. This is particularly valuable for RingRift
    where global dependencies (territory connectivity, line formation) are critical.

    The SE mechanism:
    1. Squeeze: Global average pooling to get channel descriptors
    2. Excitation: FC layers to learn channel interdependencies
    3. Scale: Multiply original features by learned channel weights

    Adds ~1% parameter overhead but significantly improves pattern recognition.

    Reference: Hu et al., "Squeeze-and-Excitation Networks" (CVPR 2018)
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels: Number of input/output channels
            reduction: Reduction ratio for SE bottleneck (default 16)
        """
        super(SEResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        # Squeeze-and-Excitation layers
        reduced_channels = max(channels // reduction, 8)  # Minimum 8 channels
        self.se_fc1 = nn.Linear(channels, reduced_channels)
        self.se_fc2 = nn.Linear(reduced_channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze-and-Excitation
        # Squeeze: Global average pooling [B, C, H, W] -> [B, C]
        se = torch.mean(out, dim=[-2, -1])
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        se = self.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        # Scale: Multiply features by channel attention
        out = out * se.unsqueeze(-1).unsqueeze(-1)

        out += residual
        out = self.relu(out)
        return out


def create_hex_mask(radius: int, bounding_size: int) -> torch.Tensor:
    """Create a hex board validity mask for the given radius.

    For a hex board embedded in a square bounding box, this creates a mask
    where valid hex cells are 1.0 and invalid (padding) cells are 0.0.

    Args:
        radius: Hex board radius (e.g., 12 for 469-cell board)
        bounding_size: Size of the square bounding box (e.g., 25)

    Returns:
        Tensor of shape [1, 1, bounding_size, bounding_size] with valid hex cells as 1.0
    """
    mask = torch.zeros(1, 1, bounding_size, bounding_size)
    center = bounding_size // 2

    for row in range(bounding_size):
        for col in range(bounding_size):
            # Convert to axial coordinates (q, r) centered at origin
            q = col - center
            r = row - center

            # Check if within hex radius using axial distance formula
            # For axial coords: distance = max(|q|, |r|, |q + r|)
            if max(abs(q), abs(r), abs(q + r)) <= radius:
                mask[0, 0, row, col] = 1.0

    return mask


# =============================================================================
# Memory-Tiered Architectures (v2)
# =============================================================================
#
# These architectures are designed for specific memory budgets:
# - v2 (High Memory): Optimized for 96GB systems with 2 NNs loaded
# - v2_Lite (Low Memory): Optimized for 48GB systems with 2 NNs loaded
#
# All v2 architectures use torch.mean() for global pooling, ensuring
# compatibility with both CUDA and MPS backends.


class RingRiftCNN_v2(nn.Module):
    """
    High-capacity CNN for 19x19 square boards (96GB memory target).

    This architecture is designed for maximum playing strength on systems
    with sufficient memory (96GB+) to run two instances simultaneously
    for comparison matches with MCTS search overhead.

    Key improvements over RingRiftCNN_MPS:
    - 12 SE residual blocks with Squeeze-and-Excitation for global patterns
    - 192 filters for richer representations
    - 14 base input channels capturing stack/cap/territory mechanics
    - 20 global features for multi-player state tracking
    - Multi-player value head (outputs per-player win probability)
    - 384-dim policy intermediate for better move discrimination

    Input Feature Channels (14 base × 4 frames = 56 total):
        1-4: Per-player stack presence (binary, one per player)
        5-8: Per-player marker presence (binary)
        9: Stack height (normalized 0-1)
        10: Cap height (normalized)
        11: Collapsed territory (binary)
        12-14: Territory ownership channels

    Global Features (20):
        1-4: Rings in hand (per player)
        5-8: Eliminated rings (per player)
        9-12: Territory count (per player)
        13-16: Line count (per player)
        17: Current player indicator
        18: Game phase (early/mid/late)
        19: Total rings in play
        20: LPS threat indicator

    Memory profile (FP32):
    - Model weights: ~150 MB
    - Per-model with activations: ~350 MB
    - Two models + MCTS: ~18 GB total

    Architecture Version:
        v2.0.0 - High-capacity SE architecture for 96GB systems.
    """

    ARCHITECTURE_VERSION = "v2.0.0"

    def __init__(
        self,
        board_size: int = 19,
        in_channels: int = 14,
        global_features: int = 20,
        num_res_blocks: int = 12,
        num_filters: int = 192,
        history_length: int = 3,
        policy_size: Optional[int] = None,
        policy_intermediate: int = 384,
        value_intermediate: int = 128,
        num_players: int = 4,
        se_reduction: int = 16,
    ):
        super(RingRiftCNN_v2, self).__init__()
        self.board_size = board_size
        self.num_filters = num_filters
        self.num_players = num_players
        self.global_features = global_features

        # Input channels = base_channels * (history_length + 1)
        self.total_in_channels = in_channels * (history_length + 1)

        # Initial convolution
        self.conv1 = nn.Conv2d(self.total_in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()

        # SE-enhanced residual blocks for global pattern recognition
        self.res_blocks = nn.ModuleList(
            [SEResidualBlock(num_filters, reduction=se_reduction) for _ in range(num_res_blocks)]
        )

        # Multi-player value head (outputs per-player win probability)
        self.value_fc1 = nn.Linear(num_filters + global_features, value_intermediate)
        self.value_fc2 = nn.Linear(value_intermediate, num_players)
        self.tanh = nn.Tanh()

        # Policy head with larger intermediate
        self.policy_fc1 = nn.Linear(num_filters + global_features, policy_intermediate)
        if policy_size is not None:
            self.policy_size = policy_size
        elif board_size == 8:
            self.policy_size = POLICY_SIZE_8x8
        elif board_size == 19:
            self.policy_size = POLICY_SIZE_19x19
        else:
            self.policy_size = POLICY_SIZE
        self.policy_fc2 = nn.Linear(policy_intermediate, self.policy_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor, globals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Backbone with SE blocks
        x = self.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            x = block(x)

        # MPS-compatible global average pooling
        x = torch.mean(x, dim=[-2, -1])

        # Concatenate global features
        x = torch.cat((x, globals), dim=1)

        # Multi-player value head: outputs [batch, num_players]
        v = self.relu(self.value_fc1(x))
        v = self.dropout(v)
        value = self.tanh(self.value_fc2(v))  # [-1, 1] per player

        # Policy head
        p = self.relu(self.policy_fc1(x))
        p = self.dropout(p)
        policy = self.policy_fc2(p)

        return value, policy

    def forward_single(
        self, feature: np.ndarray, globals_vec: np.ndarray, player_idx: int = 0
    ) -> tuple[float, np.ndarray]:
        """Convenience method for single-sample inference.

        Args:
            feature: Board features [C, H, W]
            globals_vec: Global features [G]
            player_idx: Which player's value to return (default 0)

        Returns:
            Tuple of (value for player, policy logits)
        """
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(feature[None, ...]).float().to(next(self.parameters()).device)
            g = torch.from_numpy(globals_vec[None, ...]).float().to(next(self.parameters()).device)
            v, p = self.forward(x, g)
        return float(v[0, player_idx].item()), p.cpu().numpy()[0]


class RingRiftCNN_v2_Lite(nn.Module):
    """
    Memory-efficient CNN for 19x19 square boards (48GB memory target).

    This architecture is designed for systems with limited memory (48GB)
    while maintaining reasonable playing strength. Suitable for running
    two instances simultaneously for comparison matches.

    Key trade-offs vs RingRiftCNN_v2:
    - 6 SE residual blocks (vs 12) - faster but shallower
    - 96 filters (vs 192) - smaller representations
    - 192-dim policy intermediate (vs 384)
    - 12 base input channels (vs 14) - reduced history
    - 3 history frames (vs 4) - reduced temporal context

    Input Feature Channels (12 base × 3 frames = 36 total):
        1-4: Per-player stack presence (binary)
        5-8: Per-player marker presence (binary)
        9: Stack height (normalized)
        10: Cap height (normalized)
        11: Collapsed territory (binary)
        12: Current player territory

    Global Features (20):
        Same as RingRiftCNN_v2 for compatibility

    Memory profile (FP32):
    - Model weights: ~60 MB
    - Per-model with activations: ~130 MB
    - Two models + MCTS: ~8 GB total

    Architecture Version:
        v2.0.0-lite - Memory-efficient SE architecture for 48GB systems.
    """

    ARCHITECTURE_VERSION = "v2.0.0-lite"

    def __init__(
        self,
        board_size: int = 19,
        in_channels: int = 12,
        global_features: int = 20,
        num_res_blocks: int = 6,
        num_filters: int = 96,
        history_length: int = 2,
        policy_size: Optional[int] = None,
        policy_intermediate: int = 192,
        value_intermediate: int = 64,
        num_players: int = 4,
        se_reduction: int = 16,
    ):
        super(RingRiftCNN_v2_Lite, self).__init__()
        self.board_size = board_size
        self.num_filters = num_filters
        self.num_players = num_players
        self.global_features = global_features

        self.total_in_channels = in_channels * (history_length + 1)

        # Initial convolution
        self.conv1 = nn.Conv2d(self.total_in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()

        # SE-enhanced residual blocks
        self.res_blocks = nn.ModuleList(
            [SEResidualBlock(num_filters, reduction=se_reduction) for _ in range(num_res_blocks)]
        )

        # Multi-player value head
        self.value_fc1 = nn.Linear(num_filters + global_features, value_intermediate)
        self.value_fc2 = nn.Linear(value_intermediate, num_players)
        self.tanh = nn.Tanh()

        # Policy head
        self.policy_fc1 = nn.Linear(num_filters + global_features, policy_intermediate)
        if policy_size is not None:
            self.policy_size = policy_size
        elif board_size == 8:
            self.policy_size = POLICY_SIZE_8x8
        elif board_size == 19:
            self.policy_size = POLICY_SIZE_19x19
        else:
            self.policy_size = POLICY_SIZE
        self.policy_fc2 = nn.Linear(policy_intermediate, self.policy_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor, globals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Backbone with SE blocks
        x = self.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            x = block(x)

        # MPS-compatible global average pooling
        x = torch.mean(x, dim=[-2, -1])

        # Concatenate global features
        x = torch.cat((x, globals), dim=1)

        # Multi-player value head: outputs [batch, num_players]
        v = self.relu(self.value_fc1(x))
        v = self.dropout(v)
        value = self.tanh(self.value_fc2(v))

        # Policy head
        p = self.relu(self.policy_fc1(x))
        p = self.dropout(p)
        policy = self.policy_fc2(p)

        return value, policy

    def forward_single(
        self, feature: np.ndarray, globals_vec: np.ndarray, player_idx: int = 0
    ) -> tuple[float, np.ndarray]:
        """Convenience method for single-sample inference."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(feature[None, ...]).float().to(next(self.parameters()).device)
            g = torch.from_numpy(globals_vec[None, ...]).float().to(next(self.parameters()).device)
            v, p = self.forward(x, g)
        return float(v[0, player_idx].item()), p.cpu().numpy()[0]


class RingRiftCNN_v3(nn.Module):
    """
    V3 architecture with spatial policy heads for square boards.

    This architecture improves on V2 by using spatially-structured policy heads
    that preserve the geometric relationship between positions and actions,
    rather than collapsing everything through global average pooling.

    Key improvements over V2:
    - Spatial placement head: Conv1×1 → [B, 3, H, W] for (cell, ring_count) logits
    - Spatial movement head: Conv1×1 → [B, 8*max_dist, H, W] for movement logits
    - Spatial line formation head: Conv1×1 → [B, 4, H, W] for line directions
    - Spatial territory claim head: Conv1×1 → [B, 1, H, W] for territory claims
    - Spatial territory choice head: Conv1×1 → [B, 32, H, W] for territory choice
    - Small FC for special actions (skip_placement, swap_sides, line_choice)
    - Preserves spatial locality during policy computation

    Why spatial heads are better:
    1. No spatial information loss - each cell produces its own policy logits
    2. Better gradient flow - actions at position (x,y) directly update features at (x,y)
    3. Reduced parameter count - Conv1×1 vs large FC layer
    4. Natural position encoding - the network learns to associate positions with actions

    Architecture Version:
        v3.1.0 - Spatial policy heads, SE backbone, MPS compatible, rank distribution output.

    Rank Distribution Output (v3.1.0):
        The value head now outputs a rank probability distribution for each player:
        - Shape: [B, num_players, num_players] where rank_dist[b, p, r] = P(player p finishes at rank r)
        - Uses softmax over ranks (dim=-1) so each player's rank probabilities sum to 1
        - Ranks are 0-indexed: rank 0 = 1st place (winner), rank 1 = 2nd place, etc.
        - Also outputs legacy value for backward compatibility: [B, num_players] in [-1, 1]
    """

    ARCHITECTURE_VERSION = "v3.1.0"

    def __init__(
        self,
        board_size: int = 19,
        in_channels: int = 14,
        global_features: int = 20,
        num_res_blocks: int = 12,
        num_filters: int = 192,
        history_length: int = 3,
        policy_size: Optional[int] = None,
        value_intermediate: int = 128,
        num_players: int = 4,
        se_reduction: int = 16,
        num_ring_counts: int = 3,
        num_directions: int = NUM_SQUARE_DIRS,  # 8 directions
        num_line_dirs: int = NUM_LINE_DIRS,  # 4 line directions
        territory_size_buckets: int = TERRITORY_SIZE_BUCKETS,  # 8
        territory_max_players: int = TERRITORY_MAX_PLAYERS,  # 4
    ) -> None:
        super(RingRiftCNN_v3, self).__init__()
        self.board_size = board_size
        self.num_filters = num_filters
        self.num_players = num_players
        self.global_features = global_features

        # Determine max distance based on board size
        if board_size == 8:
            self.max_distance = MAX_DIST_SQUARE8  # 7
        else:
            self.max_distance = MAX_DIST_SQUARE19  # 18

        # Spatial policy dimensions
        self.num_ring_counts = num_ring_counts
        self.num_directions = num_directions
        self.num_line_dirs = num_line_dirs
        self.territory_size_buckets = territory_size_buckets
        self.territory_max_players = territory_max_players
        self.movement_channels = num_directions * self.max_distance  # 8 × 7 or 8 × 18
        self.territory_choice_channels = territory_size_buckets * territory_max_players  # 32

        # Input channels = base_channels * (history_length + 1)
        self.total_in_channels = in_channels * (history_length + 1)

        # Determine policy size
        if policy_size is not None:
            self.policy_size = policy_size
        elif board_size == 8:
            self.policy_size = POLICY_SIZE_8x8
        elif board_size == 19:
            self.policy_size = POLICY_SIZE_19x19
        else:
            self.policy_size = POLICY_SIZE

        # Pre-compute index scatter tensors for policy assembly
        self._register_policy_indices(board_size)

        # Initial convolution
        self.conv1 = nn.Conv2d(self.total_in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()

        # SE-enhanced residual blocks
        self.res_blocks = nn.ModuleList(
            [SEResidualBlock(num_filters, reduction=se_reduction) for _ in range(num_res_blocks)]
        )

        # Multi-player value head (legacy, kept for backward compatibility)
        self.value_fc1 = nn.Linear(num_filters + global_features, value_intermediate)
        self.value_fc2 = nn.Linear(value_intermediate, num_players)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3)

        # === V3.1 Rank Distribution Head ===
        # Outputs P(player p finishes at rank r) for each player
        # Shape: [B, num_players, num_players] with softmax over ranks (dim=-1)
        rank_dist_intermediate = value_intermediate * 2  # 256 for full, 128 for lite
        self.rank_dist_fc1 = nn.Linear(num_filters + global_features, rank_dist_intermediate)
        self.rank_dist_fc2 = nn.Linear(rank_dist_intermediate, num_players * num_players)
        self.rank_softmax = nn.Softmax(dim=-1)

        # === V3 Spatial Policy Heads ===
        # Placement head: [B, 3, H, W] for (cell, ring_count)
        self.placement_conv = nn.Conv2d(num_filters, num_ring_counts, kernel_size=1)

        # Movement head: [B, movement_channels, H, W] for (cell, dir, dist)
        self.movement_conv = nn.Conv2d(num_filters, self.movement_channels, kernel_size=1)

        # Line formation head: [B, 4, H, W] for (cell, line_direction)
        self.line_form_conv = nn.Conv2d(num_filters, num_line_dirs, kernel_size=1)

        # Territory claim head: [B, 1, H, W] for (cell)
        self.territory_claim_conv = nn.Conv2d(num_filters, 1, kernel_size=1)

        # Territory choice head: [B, 32, H, W] for (cell, size_bucket, player)
        self.territory_choice_conv = nn.Conv2d(num_filters, self.territory_choice_channels, kernel_size=1)

        # Special actions FC: skip_placement (1) + swap_sides (1) + skip_recovery (1) + line_choice (4) = 7
        self.special_fc = nn.Linear(num_filters + global_features, 7)

    def _register_policy_indices(self, board_size: int) -> None:
        """
        Pre-compute index tensors for scattering spatial logits into flat policy.

        For Square8:
            Placement: idx = y * 8 * 3 + x * 3 + ring_count
            Movement: idx = MOVEMENT_BASE + y * 8 * 8 * 7 + x * 8 * 7 + dir * 7 + (dist - 1)
            Line Form: idx = LINE_FORM_BASE + y * 8 * 4 + x * 4 + line_dir
            Territory Claim: idx = TERRITORY_CLAIM_BASE + y * 8 + x
            Territory Choice: idx = TERRITORY_CHOICE_BASE + y * 8 * 32 + x * 32 + size * 4 + player

        For Square19: same pattern with 19x19 dimensions
        """
        H, W = board_size, board_size

        # Get board-specific constants
        if board_size == 8:
            movement_base = SQUARE8_MOVEMENT_BASE
            line_form_base = SQUARE8_LINE_FORM_BASE
            territory_claim_base = SQUARE8_TERRITORY_CLAIM_BASE
            territory_choice_base = SQUARE8_TERRITORY_CHOICE_BASE
            max_dist = MAX_DIST_SQUARE8
        else:
            movement_base = SQUARE19_MOVEMENT_BASE
            line_form_base = SQUARE19_LINE_FORM_BASE
            territory_claim_base = SQUARE19_TERRITORY_CLAIM_BASE
            territory_choice_base = SQUARE19_TERRITORY_CHOICE_BASE
            max_dist = MAX_DIST_SQUARE19

        # Placement indices: [3, H, W] → flat index
        placement_idx = torch.zeros(self.num_ring_counts, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for r in range(self.num_ring_counts):
                    placement_idx[r, y, x] = y * W * 3 + x * 3 + r
        self.register_buffer("placement_idx", placement_idx)

        # Movement indices: [movement_channels, H, W] → flat index
        movement_idx = torch.zeros(self.movement_channels, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for d in range(self.num_directions):
                    for dist_minus_1 in range(max_dist):
                        channel = d * max_dist + dist_minus_1
                        flat_idx = (
                            movement_base
                            + y * W * self.num_directions * max_dist
                            + x * self.num_directions * max_dist
                            + d * max_dist
                            + dist_minus_1
                        )
                        movement_idx[channel, y, x] = flat_idx
        self.register_buffer("movement_idx", movement_idx)

        # Line formation indices: [4, H, W] → flat index
        line_form_idx = torch.zeros(self.num_line_dirs, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for ld in range(self.num_line_dirs):
                    line_form_idx[ld, y, x] = line_form_base + y * W * self.num_line_dirs + x * self.num_line_dirs + ld
        self.register_buffer("line_form_idx", line_form_idx)

        # Territory claim indices: [1, H, W] → flat index
        territory_claim_idx = torch.zeros(1, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                territory_claim_idx[0, y, x] = territory_claim_base + y * W + x
        self.register_buffer("territory_claim_idx", territory_claim_idx)

        # Territory choice indices: [32, H, W] → flat index
        territory_choice_idx = torch.zeros(self.territory_choice_channels, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for size_bucket in range(self.territory_size_buckets):
                    for player_idx in range(self.territory_max_players):
                        channel = size_bucket * self.territory_max_players + player_idx
                        flat_idx = (
                            territory_choice_base
                            + y * W * self.territory_choice_channels
                            + x * self.territory_choice_channels
                            + size_bucket * self.territory_max_players
                            + player_idx
                        )
                        territory_choice_idx[channel, y, x] = flat_idx
        self.register_buffer("territory_choice_idx", territory_choice_idx)

        # Store special action indices
        if board_size == 8:
            self.skip_placement_idx = SQUARE8_SKIP_PLACEMENT_IDX
            self.swap_sides_idx = SQUARE8_SWAP_SIDES_IDX
            self.skip_recovery_idx = SQUARE8_SKIP_RECOVERY_IDX
            self.line_choice_base = SQUARE8_LINE_CHOICE_BASE
            self.extra_special_base = SQUARE8_EXTRA_SPECIAL_BASE
            self.extra_special_span = SQUARE8_EXTRA_SPECIAL_SPAN
        else:
            self.skip_placement_idx = SQUARE19_SKIP_PLACEMENT_IDX
            self.swap_sides_idx = SQUARE19_SWAP_SIDES_IDX
            self.skip_recovery_idx = SQUARE19_SKIP_RECOVERY_IDX
            self.line_choice_base = SQUARE19_LINE_CHOICE_BASE
            self.extra_special_base = SQUARE19_EXTRA_SPECIAL_BASE
            self.extra_special_span = SQUARE19_EXTRA_SPECIAL_SPAN

    def _scatter_policy_logits(
        self,
        placement_logits: torch.Tensor,
        movement_logits: torch.Tensor,
        line_form_logits: torch.Tensor,
        territory_claim_logits: torch.Tensor,
        territory_choice_logits: torch.Tensor,
        special_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Scatter spatial policy logits into flat policy vector.

        Args:
            placement_logits: [B, 3, H, W]
            movement_logits: [B, movement_channels, H, W]
            line_form_logits: [B, 4, H, W]
            territory_claim_logits: [B, 1, H, W]
            territory_choice_logits: [B, 32, H, W]
            special_logits: [B, 7] (skip_placement, swap_sides, skip_recovery, line_choice[4])

        Returns:
            policy_logits: [B, policy_size] flat policy vector
        """
        B = placement_logits.size(0)
        device = placement_logits.device
        dtype = placement_logits.dtype

        # Initialize flat policy with large negative (will be masked by legal moves)
        # Use -1e4 instead of -1e9 to avoid float16 overflow in mixed precision
        policy = torch.full((B, self.policy_size), -1e4, device=device, dtype=dtype)

        # Flatten and scatter placement logits
        placement_flat = placement_logits.view(B, -1)
        placement_idx_flat = self.placement_idx.view(-1).expand(B, -1)
        policy.scatter_(1, placement_idx_flat, placement_flat)

        # Flatten and scatter movement logits
        movement_flat = movement_logits.view(B, -1)
        movement_idx_flat = self.movement_idx.view(-1).expand(B, -1)
        policy.scatter_(1, movement_idx_flat, movement_flat)

        # Flatten and scatter line formation logits
        line_form_flat = line_form_logits.view(B, -1)
        line_form_idx_flat = self.line_form_idx.view(-1).expand(B, -1)
        policy.scatter_(1, line_form_idx_flat, line_form_flat)

        # Flatten and scatter territory claim logits
        territory_claim_flat = territory_claim_logits.view(B, -1)
        territory_claim_idx_flat = self.territory_claim_idx.view(-1).expand(B, -1)
        policy.scatter_(1, territory_claim_idx_flat, territory_claim_flat)

        # Flatten and scatter territory choice logits
        territory_choice_flat = territory_choice_logits.view(B, -1)
        territory_choice_idx_flat = self.territory_choice_idx.view(-1).expand(B, -1)
        policy.scatter_(1, territory_choice_idx_flat, territory_choice_flat)

        # Add special action logits
        policy[:, self.skip_placement_idx] = special_logits[:, 0]
        policy[:, self.swap_sides_idx] = special_logits[:, 1]
        policy[:, self.skip_recovery_idx] = special_logits[:, 2]  # RR-CANON-R112
        policy[:, self.line_choice_base : self.line_choice_base + 4] = special_logits[:, 3:7]
        # Map canonical no/skip/forced actions to the skip_placement logit by default.
        if self.extra_special_span > 0:
            policy[
                :,
                self.extra_special_base : self.extra_special_base + self.extra_special_span,
            ] = special_logits[:, 0:1]

        return policy

    def forward(
        self,
        x: torch.Tensor,
        globals: torch.Tensor,
        return_features: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass with spatial policy heads and rank distribution output.

        Args:
            x: Input features [B, in_channels, H, W]
            globals: Global features [B, global_features]
            return_features: If True, also return backbone features for auxiliary tasks

        Returns:
            value: [B, num_players] per-player expected outcome (legacy, tanh in [-1, 1])
            policy: [B, policy_size] flat policy logits
            rank_dist: [B, num_players, num_players] rank probability distribution
                       where rank_dist[b, p, r] = P(player p finishes at rank r)
                       Softmax applied over ranks (dim=-1), so each player's probs sum to 1
            features: (optional) [B, num_filters + global_features] backbone features for aux tasks
        """
        # Backbone with SE blocks
        out = self.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            out = block(out)

        # === Value Head (legacy, for backward compatibility) ===
        v_pooled = torch.mean(out, dim=[-2, -1])  # Global average pooling
        v_cat = torch.cat([v_pooled, globals], dim=1)
        v_hidden = self.relu(self.value_fc1(v_cat))
        v_hidden = self.dropout(v_hidden)
        v_out = self.tanh(self.value_fc2(v_hidden))  # [B, num_players]

        # === Rank Distribution Head (V3.1) ===
        # Compute rank probability distribution for each player
        rank_hidden = self.relu(self.rank_dist_fc1(v_cat))
        rank_hidden = self.dropout(rank_hidden)
        rank_logits = self.rank_dist_fc2(rank_hidden)  # [B, num_players * num_players]
        rank_logits = rank_logits.view(-1, self.num_players, self.num_players)  # [B, P, P]
        rank_dist = self.rank_softmax(rank_logits)  # Softmax over ranks (dim=-1)

        # === Spatial Policy Heads (V3) ===
        placement_logits = self.placement_conv(out)  # [B, 3, H, W]
        movement_logits = self.movement_conv(out)  # [B, movement_channels, H, W]
        line_form_logits = self.line_form_conv(out)  # [B, 4, H, W]
        territory_claim_logits = self.territory_claim_conv(out)  # [B, 1, H, W]
        territory_choice_logits = self.territory_choice_conv(out)  # [B, 32, H, W]

        # Special actions from pooled features
        special_input = torch.cat([v_pooled, globals], dim=1)
        special_logits = self.special_fc(special_input)  # [B, 6]

        # Scatter into flat policy vector
        policy_logits = self._scatter_policy_logits(
            placement_logits,
            movement_logits,
            line_form_logits,
            territory_claim_logits,
            territory_choice_logits,
            special_logits,
        )

        if return_features:
            # Return backbone features for auxiliary tasks
            # v_cat contains pooled spatial features + global features
            return v_out, policy_logits, rank_dist, v_cat

        return v_out, policy_logits, rank_dist

    def forward_single(
        self, feature: np.ndarray, globals_vec: np.ndarray, player_idx: int = 0
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Convenience method for single-sample inference.

        Returns:
            value: float, expected outcome for specified player (legacy)
            policy: np.ndarray, flat policy logits
            rank_dist: np.ndarray, shape [num_players, num_players], rank distribution for all players
        """
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(feature[None, ...]).float().to(next(self.parameters()).device)
            g = torch.from_numpy(globals_vec[None, ...]).float().to(next(self.parameters()).device)
            v, p, rank_dist = self.forward(x, g)
        return float(v[0, player_idx].item()), p.cpu().numpy()[0], rank_dist.cpu().numpy()[0]


class RingRiftCNN_v3_Lite(nn.Module):
    """
    Memory-efficient V3 architecture with spatial policy heads (48GB target).

    Same spatial policy head design as RingRiftCNN_v3 but with reduced capacity:
    - 6 SE residual blocks (vs 12)
    - 96 filters (vs 192)
    - 3 history frames (vs 4)
    - 12 base input channels (vs 14)

    Architecture Version:
        v3.1.0-lite - Spatial policy heads, reduced capacity, rank distribution output.

    Rank Distribution Output (v3.1.0):
        Same as RingRiftCNN_v3 - outputs rank probability distribution for each player.
    """

    ARCHITECTURE_VERSION = "v3.1.0-lite"

    def __init__(
        self,
        board_size: int = 19,
        in_channels: int = 12,
        global_features: int = 20,
        num_res_blocks: int = 6,
        num_filters: int = 96,
        history_length: int = 2,
        policy_size: Optional[int] = None,
        value_intermediate: int = 64,
        num_players: int = 4,
        se_reduction: int = 16,
        num_ring_counts: int = 3,
        num_directions: int = NUM_SQUARE_DIRS,
        num_line_dirs: int = NUM_LINE_DIRS,
        territory_size_buckets: int = TERRITORY_SIZE_BUCKETS,
        territory_max_players: int = TERRITORY_MAX_PLAYERS,
    ) -> None:
        super(RingRiftCNN_v3_Lite, self).__init__()
        self.board_size = board_size
        self.num_filters = num_filters
        self.num_players = num_players
        self.global_features = global_features

        # Determine max distance based on board size
        if board_size == 8:
            self.max_distance = MAX_DIST_SQUARE8
        else:
            self.max_distance = MAX_DIST_SQUARE19

        # Spatial policy dimensions
        self.num_ring_counts = num_ring_counts
        self.num_directions = num_directions
        self.num_line_dirs = num_line_dirs
        self.territory_size_buckets = territory_size_buckets
        self.territory_max_players = territory_max_players
        self.movement_channels = num_directions * self.max_distance
        self.territory_choice_channels = territory_size_buckets * territory_max_players

        # Input channels
        self.total_in_channels = in_channels * (history_length + 1)

        # Determine policy size
        if policy_size is not None:
            self.policy_size = policy_size
        elif board_size == 8:
            self.policy_size = POLICY_SIZE_8x8
        elif board_size == 19:
            self.policy_size = POLICY_SIZE_19x19
        else:
            self.policy_size = POLICY_SIZE

        # Pre-compute index scatter tensors
        self._register_policy_indices(board_size)

        # Initial convolution
        self.conv1 = nn.Conv2d(self.total_in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()

        # SE-enhanced residual blocks
        self.res_blocks = nn.ModuleList(
            [SEResidualBlock(num_filters, reduction=se_reduction) for _ in range(num_res_blocks)]
        )

        # Multi-player value head (legacy, kept for backward compatibility)
        self.value_fc1 = nn.Linear(num_filters + global_features, value_intermediate)
        self.value_fc2 = nn.Linear(value_intermediate, num_players)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3)

        # === V3.1 Rank Distribution Head ===
        # Outputs P(player p finishes at rank r) for each player
        rank_dist_intermediate = value_intermediate * 2  # 128 for lite
        self.rank_dist_fc1 = nn.Linear(num_filters + global_features, rank_dist_intermediate)
        self.rank_dist_fc2 = nn.Linear(rank_dist_intermediate, num_players * num_players)
        self.rank_softmax = nn.Softmax(dim=-1)

        # === V3 Spatial Policy Heads ===
        self.placement_conv = nn.Conv2d(num_filters, num_ring_counts, kernel_size=1)
        self.movement_conv = nn.Conv2d(num_filters, self.movement_channels, kernel_size=1)
        self.line_form_conv = nn.Conv2d(num_filters, num_line_dirs, kernel_size=1)
        self.territory_claim_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.territory_choice_conv = nn.Conv2d(num_filters, self.territory_choice_channels, kernel_size=1)
        self.special_fc = nn.Linear(num_filters + global_features, 6)

    def _register_policy_indices(self, board_size: int) -> None:
        """Pre-compute index tensors for policy assembly (same as V3 full)."""
        H, W = board_size, board_size

        if board_size == 8:
            movement_base = SQUARE8_MOVEMENT_BASE
            line_form_base = SQUARE8_LINE_FORM_BASE
            territory_claim_base = SQUARE8_TERRITORY_CLAIM_BASE
            territory_choice_base = SQUARE8_TERRITORY_CHOICE_BASE
            max_dist = MAX_DIST_SQUARE8
        else:
            movement_base = SQUARE19_MOVEMENT_BASE
            line_form_base = SQUARE19_LINE_FORM_BASE
            territory_claim_base = SQUARE19_TERRITORY_CLAIM_BASE
            territory_choice_base = SQUARE19_TERRITORY_CHOICE_BASE
            max_dist = MAX_DIST_SQUARE19

        # Placement indices
        placement_idx = torch.zeros(self.num_ring_counts, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for r in range(self.num_ring_counts):
                    placement_idx[r, y, x] = y * W * 3 + x * 3 + r
        self.register_buffer("placement_idx", placement_idx)

        # Movement indices
        movement_idx = torch.zeros(self.movement_channels, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for d in range(self.num_directions):
                    for dist_minus_1 in range(max_dist):
                        channel = d * max_dist + dist_minus_1
                        flat_idx = (
                            movement_base
                            + y * W * self.num_directions * max_dist
                            + x * self.num_directions * max_dist
                            + d * max_dist
                            + dist_minus_1
                        )
                        movement_idx[channel, y, x] = flat_idx
        self.register_buffer("movement_idx", movement_idx)

        # Line formation indices
        line_form_idx = torch.zeros(self.num_line_dirs, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for ld in range(self.num_line_dirs):
                    line_form_idx[ld, y, x] = line_form_base + y * W * self.num_line_dirs + x * self.num_line_dirs + ld
        self.register_buffer("line_form_idx", line_form_idx)

        # Territory claim indices
        territory_claim_idx = torch.zeros(1, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                territory_claim_idx[0, y, x] = territory_claim_base + y * W + x
        self.register_buffer("territory_claim_idx", territory_claim_idx)

        # Territory choice indices
        territory_choice_idx = torch.zeros(self.territory_choice_channels, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for size_bucket in range(self.territory_size_buckets):
                    for player_idx in range(self.territory_max_players):
                        channel = size_bucket * self.territory_max_players + player_idx
                        flat_idx = (
                            territory_choice_base
                            + y * W * self.territory_choice_channels
                            + x * self.territory_choice_channels
                            + size_bucket * self.territory_max_players
                            + player_idx
                        )
                        territory_choice_idx[channel, y, x] = flat_idx
        self.register_buffer("territory_choice_idx", territory_choice_idx)

        # Special action indices
        if board_size == 8:
            self.skip_placement_idx = SQUARE8_SKIP_PLACEMENT_IDX
            self.swap_sides_idx = SQUARE8_SWAP_SIDES_IDX
            self.skip_recovery_idx = SQUARE8_SKIP_RECOVERY_IDX
            self.line_choice_base = SQUARE8_LINE_CHOICE_BASE
            self.extra_special_base = SQUARE8_EXTRA_SPECIAL_BASE
            self.extra_special_span = SQUARE8_EXTRA_SPECIAL_SPAN
        else:
            self.skip_placement_idx = SQUARE19_SKIP_PLACEMENT_IDX
            self.swap_sides_idx = SQUARE19_SWAP_SIDES_IDX
            self.skip_recovery_idx = SQUARE19_SKIP_RECOVERY_IDX
            self.line_choice_base = SQUARE19_LINE_CHOICE_BASE
            self.extra_special_base = SQUARE19_EXTRA_SPECIAL_BASE
            self.extra_special_span = SQUARE19_EXTRA_SPECIAL_SPAN

    def _scatter_policy_logits(
        self,
        placement_logits: torch.Tensor,
        movement_logits: torch.Tensor,
        line_form_logits: torch.Tensor,
        territory_claim_logits: torch.Tensor,
        territory_choice_logits: torch.Tensor,
        special_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter spatial policy logits into flat policy vector."""
        B = placement_logits.size(0)
        device = placement_logits.device
        dtype = placement_logits.dtype

        # Use -1e4 instead of -1e9 to avoid float16 overflow in mixed precision
        policy = torch.full((B, self.policy_size), -1e4, device=device, dtype=dtype)

        # Scatter all spatial logits
        placement_flat = placement_logits.view(B, -1)
        placement_idx_flat = self.placement_idx.view(-1).expand(B, -1)
        policy.scatter_(1, placement_idx_flat, placement_flat)

        movement_flat = movement_logits.view(B, -1)
        movement_idx_flat = self.movement_idx.view(-1).expand(B, -1)
        policy.scatter_(1, movement_idx_flat, movement_flat)

        line_form_flat = line_form_logits.view(B, -1)
        line_form_idx_flat = self.line_form_idx.view(-1).expand(B, -1)
        policy.scatter_(1, line_form_idx_flat, line_form_flat)

        territory_claim_flat = territory_claim_logits.view(B, -1)
        territory_claim_idx_flat = self.territory_claim_idx.view(-1).expand(B, -1)
        policy.scatter_(1, territory_claim_idx_flat, territory_claim_flat)

        territory_choice_flat = territory_choice_logits.view(B, -1)
        territory_choice_idx_flat = self.territory_choice_idx.view(-1).expand(B, -1)
        policy.scatter_(1, territory_choice_idx_flat, territory_choice_flat)

        # Special actions
        policy[:, self.skip_placement_idx] = special_logits[:, 0]
        policy[:, self.swap_sides_idx] = special_logits[:, 1]
        policy[:, self.skip_recovery_idx] = special_logits[:, 2]  # RR-CANON-R112
        policy[:, self.line_choice_base : self.line_choice_base + 4] = special_logits[:, 3:7]
        if self.extra_special_span > 0:
            policy[
                :,
                self.extra_special_base : self.extra_special_base + self.extra_special_span,
            ] = special_logits[:, 0:1]

        return policy

    def forward(
        self,
        x: torch.Tensor,
        globals: torch.Tensor,
        return_features: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass with spatial policy heads and rank distribution output.

        Args:
            x: Input features [B, in_channels, H, W]
            globals: Global features [B, global_features]
            return_features: If True, also return backbone features for auxiliary tasks

        Returns:
            value: [B, num_players] per-player expected outcome (legacy)
            policy: [B, policy_size] flat policy logits
            rank_dist: [B, num_players, num_players] rank probability distribution
            features: (optional) [B, num_filters + global_features] backbone features for aux tasks
        """
        out = self.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            out = block(out)

        # Value head (legacy)
        v_pooled = torch.mean(out, dim=[-2, -1])
        v_cat = torch.cat([v_pooled, globals], dim=1)
        v_hidden = self.relu(self.value_fc1(v_cat))
        v_hidden = self.dropout(v_hidden)
        v_out = self.tanh(self.value_fc2(v_hidden))

        # Rank Distribution Head (V3.1)
        rank_hidden = self.relu(self.rank_dist_fc1(v_cat))
        rank_hidden = self.dropout(rank_hidden)
        rank_logits = self.rank_dist_fc2(rank_hidden)
        rank_logits = rank_logits.view(-1, self.num_players, self.num_players)
        rank_dist = self.rank_softmax(rank_logits)

        # Spatial policy heads
        placement_logits = self.placement_conv(out)
        movement_logits = self.movement_conv(out)
        line_form_logits = self.line_form_conv(out)
        territory_claim_logits = self.territory_claim_conv(out)
        territory_choice_logits = self.territory_choice_conv(out)

        special_input = torch.cat([v_pooled, globals], dim=1)
        special_logits = self.special_fc(special_input)

        policy_logits = self._scatter_policy_logits(
            placement_logits,
            movement_logits,
            line_form_logits,
            territory_claim_logits,
            territory_choice_logits,
            special_logits,
        )

        if return_features:
            return v_out, policy_logits, rank_dist, v_cat

        return v_out, policy_logits, rank_dist

    def forward_single(
        self, feature: np.ndarray, globals_vec: np.ndarray, player_idx: int = 0
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Convenience method for single-sample inference.

        Returns:
            value: float, expected outcome for specified player (legacy)
            policy: np.ndarray, flat policy logits
            rank_dist: np.ndarray, shape [num_players, num_players], rank distribution
        """
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(feature[None, ...]).float().to(next(self.parameters()).device)
            g = torch.from_numpy(globals_vec[None, ...]).float().to(next(self.parameters()).device)
            v, p, rank_dist = self.forward(x, g)
        return float(v[0, player_idx].item()), p.cpu().numpy()[0], rank_dist.cpu().numpy()[0]


# =============================================================================
# V4 Architecture (NAS-optimized)
# =============================================================================
#
# Architecture discovered by Neural Architecture Search (NAS).
# Key differences from V3:
# - Multi-head self-attention instead of SE blocks
# - Larger kernel (5x5) in initial conv
# - 13 res blocks, 128 filters (NAS optimal)
# - Deeper value head (3 layers)


class AttentionResidualBlock(nn.Module):
    """
    Residual block with multi-head self-attention.

    This block combines a standard residual convolution path with a spatial
    self-attention mechanism. The attention allows the network to capture
    long-range dependencies between board positions, which is critical for
    understanding territory connectivity and line formation patterns.

    NAS found this more effective than SE blocks for RingRift.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        dropout: float = 0.08,
        kernel_size: int = 3,
    ):
        super(AttentionResidualBlock, self).__init__()
        self.channels = channels
        self.num_heads = num_heads

        # Convolutional path
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channels)

        # Multi-head self-attention
        # We use 1x1 convs for Q, K, V projections to maintain spatial structure
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"

        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.attention_out = nn.Conv2d(channels, channels, kernel_size=1)

        self.attention_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm([channels])

        # Learnable mixing parameter for attention vs conv path
        self.attention_gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        # Convolutional path
        conv_out = self.relu(self.bn1(self.conv1(x)))
        conv_out = self.bn2(self.conv2(conv_out))

        # Attention path
        # Reshape for multi-head attention: [B, C, H, W] -> [B, heads, head_dim, H*W]
        q = self.query(x).view(B, self.num_heads, self.head_dim, H * W)
        k = self.key(x).view(B, self.num_heads, self.head_dim, H * W)
        v = self.value(x).view(B, self.num_heads, self.head_dim, H * W)

        # Compute attention: [B, heads, H*W, H*W]
        attn_weights = torch.matmul(q.transpose(-2, -1), k) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        # Apply attention: [B, heads, head_dim, H*W]
        attn_out = torch.matmul(v, attn_weights.transpose(-2, -1))
        attn_out = attn_out.view(B, C, H, W)
        attn_out = self.attention_out(attn_out)

        # Gated combination of conv and attention paths
        gate = torch.sigmoid(self.attention_gate)
        out = conv_out + gate * attn_out

        # Residual connection
        out = out + residual
        out = self.relu(out)

        return out


class RingRiftCNN_v4(nn.Module):
    """
    V4 architecture discovered by Neural Architecture Search (NAS).

    This architecture incorporates the optimal hyperparameters found by
    evolutionary NAS, combining the game-specific features of V3 (spatial
    policy heads, rank distribution) with NAS-optimized structural choices:

    NAS-Discovered Improvements:
    - Multi-head self-attention (4 heads) instead of SE blocks
    - 13 residual blocks (vs 12 in v3)
    - 128 filters (vs 192 in v3, more efficient)
    - 5x5 initial kernel (vs 3x3, better spatial coverage)
    - Deeper value head (3 layers vs 2)
    - Lower dropout (0.08 vs 0.3)

    Preserved from V3:
    - Spatial policy heads (placement, movement, line formation, territory)
    - Rank distribution output for multi-player games
    - Game-specific action encoding

    Architecture Version:
        v4.0.0 - NAS-optimized attention architecture.

    Performance Characteristics:
    - Slightly fewer parameters than v3 (128 vs 192 filters)
    - Better long-range pattern recognition (attention)
    - Improved training efficiency (deeper value head)
    """

    ARCHITECTURE_VERSION = "v4.0.0"

    def __init__(
        self,
        board_size: int = 19,
        in_channels: int = 14,
        global_features: int = 20,
        num_res_blocks: int = 13,  # NAS optimal
        num_filters: int = 128,  # NAS optimal
        history_length: int = 3,
        policy_size: Optional[int] = None,
        value_intermediate: int = 256,  # NAS optimal
        value_hidden: int = 256,  # NAS: deeper value head
        num_players: int = 4,
        num_attention_heads: int = 4,  # NAS optimal
        dropout: float = 0.08,  # NAS optimal
        initial_kernel_size: int = 5,  # NAS optimal
        num_ring_counts: int = 3,
        num_directions: int = NUM_SQUARE_DIRS,
        num_line_dirs: int = NUM_LINE_DIRS,
        territory_size_buckets: int = TERRITORY_SIZE_BUCKETS,
        territory_max_players: int = TERRITORY_MAX_PLAYERS,
    ) -> None:
        super(RingRiftCNN_v4, self).__init__()
        self.board_size = board_size
        self.num_filters = num_filters
        self.num_players = num_players
        self.global_features = global_features
        self.num_res_blocks = num_res_blocks
        self.dropout_rate = dropout

        # Determine max distance based on board size
        if board_size == 8:
            self.max_distance = MAX_DIST_SQUARE8
        else:
            self.max_distance = MAX_DIST_SQUARE19

        # Spatial policy dimensions
        self.num_ring_counts = num_ring_counts
        self.num_directions = num_directions
        self.num_line_dirs = num_line_dirs
        self.territory_size_buckets = territory_size_buckets
        self.territory_max_players = territory_max_players
        self.movement_channels = num_directions * self.max_distance
        self.territory_choice_channels = territory_size_buckets * territory_max_players

        # Input channels
        self.total_in_channels = in_channels * (history_length + 1)

        # Determine policy size
        if policy_size is not None:
            self.policy_size = policy_size
        elif board_size == 8:
            self.policy_size = POLICY_SIZE_8x8
        elif board_size == 19:
            self.policy_size = POLICY_SIZE_19x19
        else:
            self.policy_size = POLICY_SIZE

        # Pre-compute index scatter tensors
        self._register_policy_indices(board_size)

        # Initial convolution with larger kernel (NAS optimal: 5x5)
        self.conv1 = nn.Conv2d(
            self.total_in_channels,
            num_filters,
            kernel_size=initial_kernel_size,
            padding=initial_kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()

        # Attention-enhanced residual blocks (NAS optimal)
        self.res_blocks = nn.ModuleList([
            AttentionResidualBlock(
                num_filters,
                num_heads=num_attention_heads,
                dropout=dropout,
            )
            for _ in range(num_res_blocks)
        ])

        # === Deeper Value Head (NAS optimal: 3 layers) ===
        self.value_fc1 = nn.Linear(num_filters + global_features, value_intermediate)
        self.value_fc2 = nn.Linear(value_intermediate, value_hidden)
        self.value_fc3 = nn.Linear(value_hidden, num_players)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        # === Rank Distribution Head ===
        rank_dist_intermediate = value_intermediate
        self.rank_dist_fc1 = nn.Linear(num_filters + global_features, rank_dist_intermediate)
        self.rank_dist_fc2 = nn.Linear(rank_dist_intermediate, value_hidden)
        self.rank_dist_fc3 = nn.Linear(value_hidden, num_players * num_players)
        self.rank_softmax = nn.Softmax(dim=-1)

        # === Spatial Policy Heads (inherited from V3) ===
        self.placement_conv = nn.Conv2d(num_filters, num_ring_counts, kernel_size=1)
        self.movement_conv = nn.Conv2d(num_filters, self.movement_channels, kernel_size=1)
        self.line_form_conv = nn.Conv2d(num_filters, num_line_dirs, kernel_size=1)
        self.territory_claim_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.territory_choice_conv = nn.Conv2d(num_filters, self.territory_choice_channels, kernel_size=1)
        self.special_fc = nn.Linear(num_filters + global_features, 7)

    def _register_policy_indices(self, board_size: int) -> None:
        """Pre-compute index tensors for scattering spatial logits into flat policy."""
        H, W = board_size, board_size

        if board_size == 8:
            movement_base = SQUARE8_MOVEMENT_BASE
            line_form_base = SQUARE8_LINE_FORM_BASE
            territory_claim_base = SQUARE8_TERRITORY_CLAIM_BASE
            territory_choice_base = SQUARE8_TERRITORY_CHOICE_BASE
            max_dist = MAX_DIST_SQUARE8
        else:
            movement_base = SQUARE19_MOVEMENT_BASE
            line_form_base = SQUARE19_LINE_FORM_BASE
            territory_claim_base = SQUARE19_TERRITORY_CLAIM_BASE
            territory_choice_base = SQUARE19_TERRITORY_CHOICE_BASE
            max_dist = MAX_DIST_SQUARE19

        # Placement indices
        placement_idx = torch.zeros(self.num_ring_counts, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for r in range(self.num_ring_counts):
                    placement_idx[r, y, x] = y * W * 3 + x * 3 + r
        self.register_buffer("placement_idx", placement_idx)

        # Movement indices
        movement_idx = torch.zeros(self.movement_channels, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for d in range(self.num_directions):
                    for dist_minus_1 in range(max_dist):
                        channel = d * max_dist + dist_minus_1
                        flat_idx = (
                            movement_base
                            + y * W * self.num_directions * max_dist
                            + x * self.num_directions * max_dist
                            + d * max_dist
                            + dist_minus_1
                        )
                        movement_idx[channel, y, x] = flat_idx
        self.register_buffer("movement_idx", movement_idx)

        # Line formation indices
        line_form_idx = torch.zeros(self.num_line_dirs, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for ld in range(self.num_line_dirs):
                    line_form_idx[ld, y, x] = line_form_base + y * W * self.num_line_dirs + x * self.num_line_dirs + ld
        self.register_buffer("line_form_idx", line_form_idx)

        # Territory claim indices
        territory_claim_idx = torch.zeros(1, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                territory_claim_idx[0, y, x] = territory_claim_base + y * W + x
        self.register_buffer("territory_claim_idx", territory_claim_idx)

        # Territory choice indices
        territory_choice_idx = torch.zeros(self.territory_choice_channels, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for size_bucket in range(self.territory_size_buckets):
                    for player_idx in range(self.territory_max_players):
                        channel = size_bucket * self.territory_max_players + player_idx
                        flat_idx = (
                            territory_choice_base
                            + y * W * self.territory_choice_channels
                            + x * self.territory_choice_channels
                            + size_bucket * self.territory_max_players
                            + player_idx
                        )
                        territory_choice_idx[channel, y, x] = flat_idx
        self.register_buffer("territory_choice_idx", territory_choice_idx)

        # Special action indices
        if board_size == 8:
            self.skip_placement_idx = SQUARE8_SKIP_PLACEMENT_IDX
            self.swap_sides_idx = SQUARE8_SWAP_SIDES_IDX
            self.skip_recovery_idx = SQUARE8_SKIP_RECOVERY_IDX
            self.line_choice_base = SQUARE8_LINE_CHOICE_BASE
            self.extra_special_base = SQUARE8_EXTRA_SPECIAL_BASE
            self.extra_special_span = SQUARE8_EXTRA_SPECIAL_SPAN
        else:
            self.skip_placement_idx = SQUARE19_SKIP_PLACEMENT_IDX
            self.swap_sides_idx = SQUARE19_SWAP_SIDES_IDX
            self.skip_recovery_idx = SQUARE19_SKIP_RECOVERY_IDX
            self.line_choice_base = SQUARE19_LINE_CHOICE_BASE
            self.extra_special_base = SQUARE19_EXTRA_SPECIAL_BASE
            self.extra_special_span = SQUARE19_EXTRA_SPECIAL_SPAN

    def _scatter_policy_logits(
        self,
        placement_logits: torch.Tensor,
        movement_logits: torch.Tensor,
        line_form_logits: torch.Tensor,
        territory_claim_logits: torch.Tensor,
        territory_choice_logits: torch.Tensor,
        special_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter spatial policy logits into flat policy vector."""
        B = placement_logits.size(0)
        device = placement_logits.device
        dtype = placement_logits.dtype

        policy = torch.full((B, self.policy_size), -1e4, device=device, dtype=dtype)

        placement_flat = placement_logits.view(B, -1)
        placement_idx_flat = self.placement_idx.view(-1).expand(B, -1)
        policy.scatter_(1, placement_idx_flat, placement_flat)

        movement_flat = movement_logits.view(B, -1)
        movement_idx_flat = self.movement_idx.view(-1).expand(B, -1)
        policy.scatter_(1, movement_idx_flat, movement_flat)

        line_form_flat = line_form_logits.view(B, -1)
        line_form_idx_flat = self.line_form_idx.view(-1).expand(B, -1)
        policy.scatter_(1, line_form_idx_flat, line_form_flat)

        territory_claim_flat = territory_claim_logits.view(B, -1)
        territory_claim_idx_flat = self.territory_claim_idx.view(-1).expand(B, -1)
        policy.scatter_(1, territory_claim_idx_flat, territory_claim_flat)

        territory_choice_flat = territory_choice_logits.view(B, -1)
        territory_choice_idx_flat = self.territory_choice_idx.view(-1).expand(B, -1)
        policy.scatter_(1, territory_choice_idx_flat, territory_choice_flat)

        policy[:, self.skip_placement_idx] = special_logits[:, 0]
        policy[:, self.swap_sides_idx] = special_logits[:, 1]
        policy[:, self.skip_recovery_idx] = special_logits[:, 2]
        policy[:, self.line_choice_base : self.line_choice_base + 4] = special_logits[:, 3:7]
        if self.extra_special_span > 0:
            policy[
                :,
                self.extra_special_base : self.extra_special_base + self.extra_special_span,
            ] = special_logits[:, 0:1]

        return policy

    def forward(
        self,
        x: torch.Tensor,
        globals: torch.Tensor,
        return_features: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass with attention backbone and spatial policy heads.

        Args:
            x: Input features [B, in_channels, H, W]
            globals: Global features [B, global_features]
            return_features: If True, also return backbone features for auxiliary tasks

        Returns:
            value: [B, num_players] per-player expected outcome
            policy: [B, policy_size] flat policy logits
            rank_dist: [B, num_players, num_players] rank probability distribution
            features: (optional) [B, num_filters + global_features] backbone features for aux tasks
        """
        # Backbone with attention blocks
        out = self.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            out = block(out)

        # === Deeper Value Head (3 layers) ===
        v_pooled = torch.mean(out, dim=[-2, -1])
        v_cat = torch.cat([v_pooled, globals], dim=1)
        v_hidden = self.relu(self.value_fc1(v_cat))
        v_hidden = self.dropout(v_hidden)
        v_hidden = self.relu(self.value_fc2(v_hidden))
        v_hidden = self.dropout(v_hidden)
        v_out = self.tanh(self.value_fc3(v_hidden))

        # === Rank Distribution Head (3 layers) ===
        rank_hidden = self.relu(self.rank_dist_fc1(v_cat))
        rank_hidden = self.dropout(rank_hidden)
        rank_hidden = self.relu(self.rank_dist_fc2(rank_hidden))
        rank_hidden = self.dropout(rank_hidden)
        rank_logits = self.rank_dist_fc3(rank_hidden)
        rank_logits = rank_logits.view(-1, self.num_players, self.num_players)
        rank_dist = self.rank_softmax(rank_logits)

        # === Spatial Policy Heads ===
        placement_logits = self.placement_conv(out)
        movement_logits = self.movement_conv(out)
        line_form_logits = self.line_form_conv(out)
        territory_claim_logits = self.territory_claim_conv(out)
        territory_choice_logits = self.territory_choice_conv(out)

        special_input = torch.cat([v_pooled, globals], dim=1)
        special_logits = self.special_fc(special_input)

        policy_logits = self._scatter_policy_logits(
            placement_logits,
            movement_logits,
            line_form_logits,
            territory_claim_logits,
            territory_choice_logits,
            special_logits,
        )

        if return_features:
            return v_out, policy_logits, rank_dist, v_cat

        return v_out, policy_logits, rank_dist

    def forward_single(
        self, feature: np.ndarray, globals_vec: np.ndarray, player_idx: int = 0
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Convenience method for single-sample inference."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(feature[None, ...]).float().to(next(self.parameters()).device)
            g = torch.from_numpy(globals_vec[None, ...]).float().to(next(self.parameters()).device)
            v, p, rank_dist = self.forward(x, g)
        return float(v[0, player_idx].item()), p.cpu().numpy()[0], rank_dist.cpu().numpy()[0]


# =============================================================================
# Model Factory Functions
# =============================================================================
#
# These functions create board-specific model instances with optimal
# configurations for each board type.


def get_memory_tier() -> str:
    """
    Get the memory tier configuration from environment variable.

    The memory tier controls which model variant to use:
    - "high" (default, 96GB target): Full-capacity v2 models for maximum playing strength
    - "low" (48GB target): Memory-efficient v2-lite models for constrained systems
    - "v3-high": V3 models with spatial policy heads
    - "v3-low": V3-lite models with spatial policy heads
    - "v4": V4 NAS-optimized models with attention (square boards only)

    Returns
    -------
    str
        One of "high", "low", "v3-high", "v3-low", or "v4".
    """
    valid_tiers = ("high", "low", "v3-high", "v3-low", "v4")
    tier = os.environ.get("RINGRIFT_NN_MEMORY_TIER", "high").lower()
    if tier not in valid_tiers:
        logger.warning(f"Unknown memory tier '{tier}', defaulting to 'high'")
        return "high"
    return tier


def create_model_for_board(
    board_type: BoardType,
    in_channels: int = 14,
    global_features: int = 20,
    num_res_blocks: Optional[int] = None,
    num_filters: Optional[int] = None,
    history_length: int = 3,
    memory_tier: Optional[str] = None,
    model_class: Optional[str] = None,
    num_players: int = 4,
    policy_size: Optional[int] = None,
    **_: Any,
) -> nn.Module:
    """
    # model_class is accepted for backward compatibility with legacy callers
    # but is ignored; v2/v2-lite selection is handled via memory_tier.
    Create a neural network model optimized for a specific board type.

    This factory function instantiates the correct v2 model architecture with
    board-specific policy head sizes to avoid wasting parameters on unused
    action space. All models are CUDA and MPS compatible.

    Parameters
    ----------
    board_type : BoardType
        The board type (SQUARE8, SQUARE19, or HEXAGONAL).
    in_channels : int
        Number of input feature channels per frame (default 14).
    global_features : int
        Number of global feature dimensions (default 20).
    num_res_blocks : int, optional
        Number of residual blocks in the backbone (default depends on tier).
    num_filters : int, optional
        Number of convolutional filters (default depends on tier).
    history_length : int
        Number of historical frames to stack (default 3).
    memory_tier : str, optional
        Memory tier override. Valid values:
        - "high" (default, 96GB target): V2 models with GAP→FC policy heads
        - "low" (48GB target): V2-lite models with reduced capacity
        - "v3-high": V3 models with spatial policy heads (experimental)
        - "v3-low": V3-lite models with spatial policy heads (experimental)
        - "v4": V4 NAS-optimized architecture with attention (square boards only)
        If None, reads from RINGRIFT_NN_MEMORY_TIER environment variable.
        Defaults to "high".

    Returns
    -------
    nn.Module
        A model instance configured for the specified board type.

    Notes
    -----
    Memory tier selection:
    - "high" (default, 96GB target): Uses v2 models with 12-15 res blocks, 192 filters
    - "low" (48GB target): Uses v2-lite models with 6-8 res blocks, 96 filters
    - "v3-high" (experimental): Uses v3 models with spatial policy heads (~8M params)
    - "v3-low" (experimental): Uses v3-lite models with spatial policy heads (~4M params)
    - "v4" (NAS-optimized): Uses v4 models with attention, 13 res blocks, 128 filters

    V3 architectures use spatially-structured Conv1×1 policy heads instead of global
    average pooling + FC layers. This preserves spatial locality and reduces parameter
    count while potentially improving learning for position-dependent actions.

    V4 architecture was discovered by Neural Architecture Search (NAS). It uses multi-head
    self-attention instead of SE blocks, a 5x5 initial kernel, and a deeper value head.
    Currently only supports square boards.

    All models use torch.mean for global pooling, ensuring CUDA and MPS compatibility.

    Examples
    --------
    >>> # Create 8x8 model (default high tier)
    >>> model_8x8 = create_model_for_board(BoardType.SQUARE8)
    >>> assert isinstance(model_8x8, RingRiftCNN_v2)

    >>> # Create 19x19 model with high memory tier
    >>> model_19x19 = create_model_for_board(BoardType.SQUARE19, memory_tier="high")
    >>> assert isinstance(model_19x19, RingRiftCNN_v2)

    >>> # Create hex model with low memory tier
    >>> model_hex = create_model_for_board(BoardType.HEXAGONAL, memory_tier="low")
    >>> assert isinstance(model_hex, HexNeuralNet_v2_Lite)
    """
    # Get board-specific parameters
    board_size = get_spatial_size_for_board(board_type)
    # Use provided policy_size if available, otherwise get default for board type
    if policy_size is None:
        policy_size = get_policy_size_for_board(board_type)

    # Determine memory tier
    tier = memory_tier if memory_tier is not None else get_memory_tier()

    # Create model based on board type and memory tier
    # Both HEX8 (radius-4) and HEXAGONAL (radius-12) use hexagonal neural network architectures
    if board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
        if tier == "v4":
            raise ValueError(
                "V4 architecture is not yet available for hexagonal boards. "
                "Use 'v3-high', 'v3-low', 'high', or 'low' instead."
            )
        if tier == "v3-high":
            # HexNeuralNet_v3: Spatial policy heads, 12 res blocks, 192 filters, ~8M params
            return HexNeuralNet_v3(
                in_channels=in_channels * (history_length + 1),
                global_features=global_features,
                num_res_blocks=num_res_blocks or 12,
                num_filters=num_filters or 192,
                board_size=board_size,
                policy_size=policy_size,
                num_players=num_players,
            )
        elif tier == "v3-low":
            # HexNeuralNet_v3_Lite: Spatial policy heads, 6 res blocks, 96 filters, ~4M params
            return HexNeuralNet_v3_Lite(
                in_channels=in_channels * (history_length + 1),
                global_features=global_features,
                num_res_blocks=num_res_blocks or 6,
                num_filters=num_filters or 96,
                board_size=board_size,
                policy_size=policy_size,
                num_players=num_players,
            )
        elif tier == "high":
            # HexNeuralNet_v2: 12 res blocks, 192 filters, ~43M params
            return HexNeuralNet_v2(
                in_channels=in_channels * (history_length + 1),
                global_features=global_features,
                num_res_blocks=num_res_blocks or 12,
                num_filters=num_filters or 192,
                board_size=board_size,
                policy_size=policy_size,
                num_players=num_players,
            )
        else:  # low tier (default)
            # HexNeuralNet_v2_Lite: 6 res blocks, 96 filters, ~19M params
            return HexNeuralNet_v2_Lite(
                in_channels=in_channels * (history_length + 1),
                global_features=global_features,
                num_res_blocks=num_res_blocks or 6,
                num_filters=num_filters or 96,
                board_size=board_size,
                policy_size=policy_size,
                num_players=num_players,
            )
    else:
        # Square boards (8x8 and 19x19)
        if tier == "v4":
            # RingRiftCNN_v4: NAS-optimized, 13 res blocks, 128 filters, attention
            return RingRiftCNN_v4(
                board_size=board_size,
                in_channels=in_channels,
                global_features=global_features,
                num_res_blocks=num_res_blocks or 13,  # NAS optimal
                num_filters=num_filters or 128,  # NAS optimal
                num_attention_heads=4,  # NAS optimal
                dropout=0.08,  # NAS optimal
                initial_kernel_size=5,  # NAS optimal
                history_length=history_length,
                policy_size=policy_size,
            )
        elif tier == "v3-high":
            # RingRiftCNN_v3: Spatial policy heads, 12 res blocks, 192 filters
            return RingRiftCNN_v3(
                board_size=board_size,
                in_channels=in_channels,
                global_features=global_features,
                num_res_blocks=num_res_blocks or 12,
                num_filters=num_filters or 192,
                history_length=history_length,
                policy_size=policy_size,
            )
        elif tier == "v3-low":
            # RingRiftCNN_v3_Lite: Spatial policy heads, 6 res blocks, 96 filters
            return RingRiftCNN_v3_Lite(
                board_size=board_size,
                in_channels=in_channels,
                global_features=global_features,
                num_res_blocks=num_res_blocks or 6,
                num_filters=num_filters or 96,
                history_length=history_length,
                policy_size=policy_size,
            )
        elif tier == "high":
            # RingRiftCNN_v2: 12 res blocks, 192 filters, ~34M params
            return RingRiftCNN_v2(
                board_size=board_size,
                in_channels=in_channels,
                global_features=global_features,
                num_res_blocks=num_res_blocks or 12,
                num_filters=num_filters or 192,
                history_length=history_length,
                policy_size=policy_size,
            )
        else:  # low tier
            # RingRiftCNN_v2_Lite: 6 res blocks, 96 filters, ~14M params
            return RingRiftCNN_v2_Lite(
                board_size=board_size,
                in_channels=in_channels,
                global_features=global_features,
                num_res_blocks=num_res_blocks or 6,
                num_filters=num_filters or 96,
                history_length=history_length,
                policy_size=policy_size,
            )


def get_model_config_for_board(
    board_type: BoardType,
    memory_tier: Optional[str] = None,
) -> Dict[str, any]:
    """
    Get recommended model configuration for a specific board type.

    Returns a dictionary of hyperparameters optimized for the board type,
    including recommended residual block count and filter count based on
    the complexity of the action space and memory tier.

    Parameters
    ----------
    board_type : BoardType
        The board type to get configuration for.
    memory_tier : str, optional
        Memory tier override: "high" (96GB) or "low" (48GB).
        If None, reads from RINGRIFT_NN_MEMORY_TIER environment variable.
        Defaults to "high".

    Returns
    -------
    Dict[str, any]
        Configuration dictionary with keys:
        - board_size: Spatial dimension of the board
        - policy_size: Action space size
        - num_res_blocks: Recommended residual block count
        - num_filters: Recommended filter count
        - recommended_model: Which model class to use
        - memory_tier: Active memory tier
        - estimated_params_m: Estimated parameter count in millions
    """
    tier = memory_tier if memory_tier is not None else get_memory_tier()

    config = {
        "board_size": get_spatial_size_for_board(board_type),
        "policy_size": get_policy_size_for_board(board_type),
        "memory_tier": tier,
    }

    # V3 models with spatial policy heads (more memory-efficient)
    if tier == "v3-high":
        if board_type == BoardType.HEXAGONAL:
            config.update(
                {
                    "num_res_blocks": 12,
                    "num_filters": 192,
                    "recommended_model": "HexNeuralNet_v3",
                    "description": "V3 spatial policy hex model for 96GB systems (~8M params)",
                    "estimated_params_m": 8.2,
                }
            )
        elif board_type == BoardType.HEX8:
            config.update(
                {
                    "num_res_blocks": 12,
                    "num_filters": 192,
                    "recommended_model": "HexNeuralNet_v3",
                    "description": "V3 spatial policy hex8 model for 96GB systems (~7M params)",
                    "estimated_params_m": 7.0,
                }
            )
        elif board_type == BoardType.SQUARE19:
            config.update(
                {
                    "num_res_blocks": 12,
                    "num_filters": 192,
                    "recommended_model": "RingRiftCNN_v3",
                    "description": "V3 spatial policy 19x19 model for 96GB systems (~7M params)",
                    "estimated_params_m": 7.0,
                }
            )
        else:  # SQUARE8
            config.update(
                {
                    "num_res_blocks": 12,
                    "num_filters": 192,
                    "recommended_model": "RingRiftCNN_v3",
                    "description": "V3 spatial policy 8x8 model for 96GB systems (~7M params)",
                    "estimated_params_m": 7.0,
                }
            )
    elif tier == "v3-low":
        if board_type == BoardType.HEXAGONAL:
            config.update(
                {
                    "num_res_blocks": 6,
                    "num_filters": 96,
                    "recommended_model": "HexNeuralNet_v3_Lite",
                    "description": "V3 spatial policy hex model for 48GB systems (~2M params)",
                    "estimated_params_m": 2.1,
                }
            )
        elif board_type == BoardType.HEX8:
            config.update(
                {
                    "num_res_blocks": 6,
                    "num_filters": 96,
                    "recommended_model": "HexNeuralNet_v3_Lite",
                    "description": "V3 spatial policy hex8 model for 48GB systems (~2M params)",
                    "estimated_params_m": 1.8,
                }
            )
        elif board_type == BoardType.SQUARE19:
            config.update(
                {
                    "num_res_blocks": 6,
                    "num_filters": 96,
                    "recommended_model": "RingRiftCNN_v3_Lite",
                    "description": "V3 spatial policy 19x19 model for 48GB systems (~2M params)",
                    "estimated_params_m": 1.8,
                }
            )
        else:  # SQUARE8
            config.update(
                {
                    "num_res_blocks": 6,
                    "num_filters": 96,
                    "recommended_model": "RingRiftCNN_v3_Lite",
                    "description": "V3 spatial policy 8x8 model for 48GB systems (~2M params)",
                    "estimated_params_m": 1.8,
                }
            )
    # V2 models with FC policy heads
    elif tier == "high":
        if board_type == BoardType.HEXAGONAL:
            config.update(
                {
                    "num_res_blocks": 12,
                    "num_filters": 192,
                    "recommended_model": "HexNeuralNet_v2",
                    "description": "High-capacity hex model for 96GB systems (~43M params)",
                    "estimated_params_m": 43.4,
                }
            )
        elif board_type == BoardType.HEX8:
            config.update(
                {
                    "num_res_blocks": 12,
                    "num_filters": 192,
                    "recommended_model": "HexNeuralNet_v2",
                    "description": "High-capacity hex8 model for 96GB systems (~34M params)",
                    "estimated_params_m": 34.0,
                }
            )
        elif board_type == BoardType.SQUARE19:
            config.update(
                {
                    "num_res_blocks": 12,
                    "num_filters": 192,
                    "recommended_model": "RingRiftCNN_v2",
                    "description": "High-capacity 19x19 model for 96GB systems (~34M params)",
                    "estimated_params_m": 34.0,
                }
            )
        else:  # SQUARE8
            config.update(
                {
                    "num_res_blocks": 12,
                    "num_filters": 192,
                    "recommended_model": "RingRiftCNN_v2",
                    "description": "High-capacity 8x8 model for 96GB systems (~34M params)",
                    "estimated_params_m": 34.0,
                }
            )
    else:  # low tier
        if board_type == BoardType.HEXAGONAL:
            config.update(
                {
                    "num_res_blocks": 6,
                    "num_filters": 96,
                    "recommended_model": "HexNeuralNet_v2_Lite",
                    "description": "Memory-efficient hex model for 48GB systems (~19M params)",
                    "estimated_params_m": 18.7,
                }
            )
        elif board_type == BoardType.HEX8:
            config.update(
                {
                    "num_res_blocks": 6,
                    "num_filters": 96,
                    "recommended_model": "HexNeuralNet_v2_Lite",
                    "description": "Memory-efficient hex8 model for 48GB systems (~14M params)",
                    "estimated_params_m": 14.0,
                }
            )
        elif board_type == BoardType.SQUARE19:
            config.update(
                {
                    "num_res_blocks": 6,
                    "num_filters": 96,
                    "recommended_model": "RingRiftCNN_v2_Lite",
                    "description": "Memory-efficient 19x19 model for 48GB systems (~14M params)",
                    "estimated_params_m": 14.3,
                }
            )
        else:  # SQUARE8
            config.update(
                {
                    "num_res_blocks": 6,
                    "num_filters": 96,
                    "recommended_model": "RingRiftCNN_v2_Lite",
                    "description": "Memory-efficient 8x8 model for 48GB systems (~14M params)",
                    "estimated_params_m": 14.0,
                }
            )

    return config


class NeuralNetAI(BaseAI):
    """AI that uses a CNN to evaluate positions.

    Configuration overview (:class:`AIConfig` / related fields):

    - ``nn_model_id``: Logical identifier for the model checkpoint
      (e.g. ``"ringrift_v5_sq8_2p_2xh100"``). Resolved to
      ``<base_dir>/models/<id>.pth`` (or ``<id>_mps.pth`` for MPS builds).
      When omitted, defaults to a board-aware model-id selection (see
      :meth:`_ensure_model_initialized`):
      - square8: prefer v3-family ("v5") when present, else v2-family ("v4")
      - square19/hex: v2-family ("v4") until v3-family checkpoints exist
    - ``allow_fresh_weights``: When ``True``, missing checkpoints are
      treated as intentional and the network starts from random weights
      without raising; when ``False`` (default), a WARNING is logged.
    - ``history_length`` (environment / training wiring): Number of
      previous board feature frames to include in the stacked CNN input
      (in addition to the current frame). This controls temporal context
      for both training and inference and must match the value used when
      the checkpoint was trained.

    Board-specific model selection:
        The model architecture is automatically selected based on the board
        type of the first game state processed:
        - SQUARE8: RingRiftCNN_MPS (7K policy head)
        - SQUARE19: RingRiftCNN_MPS (67K policy head)
        - HEXAGONAL: HexNeuralNet_v2 (92K policy head, with D6 symmetry)

        This is done via lazy initialization - the model is not created
        until the first game state is seen. You can also pass board_type
        to __init__ to force early initialization.

    Training vs inference:
        The class itself is agnostic to training vs inference. In
        production it is normally used in inference mode, with a
        single model instance loaded onto a chosen device (MPS, CUDA,
        or CPU). The :attr:`game_history` buffer accumulates per‑game
        feature history keyed by ``GameState.id`` and is truncated to
        ``history_length + 1`` frames per game to bound memory usage.
    """

    def __init__(
        self,
        player_number: int,
        config: Any,
        board_type: Optional[BoardType] = None,
    ):
        super().__init__(player_number, config)
        # Initialize model
        # Channels:
        # 0: My stacks (height normalized)
        # 1: Opponent stacks (height normalized)
        # 2: My markers
        # 3: Opponent markers
        # 4: My collapsed spaces
        # 5: Opponent collapsed spaces
        # 6: My liberties
        # 7: Opponent liberties
        # 8: My line potential
        # 9: Opponent line potential
        # Hint for tools that need the current spatial dimension (e.g. training
        # data augmentation). The encoder derives the true size from the
        # GameState/BoardState via _infer_board_size and keeps this field
        # updated at runtime.
        self.board_size = 8
        self.history_length = 3
        self.feature_version = int(getattr(config, "feature_version", 1) or 1)
        # Dict[str, List[np.ndarray]] - Keyed by game_id
        self.game_history = {}

        # Track which board type we're initialized for
        self._initialized_board_type: Optional[BoardType] = None
        # Best-effort metadata for observability.
        self.loaded_checkpoint_path: Optional[str] = None
        self.loaded_checkpoint_signature: Optional[tuple[int, int]] = None
        # Hex encoder for hex boards (initialized lazily in _ensure_model_initialized)
        self._hex_encoder: Optional[Any] = None

        # Device detection
        import os

        disable_mps = bool(os.environ.get("RINGRIFT_DISABLE_MPS") or os.environ.get("PYTORCH_MPS_DISABLE"))
        force_cpu = bool(os.environ.get("RINGRIFT_FORCE_CPU"))

        # Architecture selection
        # RINGRIFT_NN_ARCHITECTURE can be:
        # - "default": Use RingRiftCNN_v2 (MPS-compatible)
        # - "mps": Use RingRiftCNN_MPS (MPS-compatible)
        # - "auto": Auto-select MPS architecture if MPS available (RECOMMENDED)
        # Default is "auto" to avoid AdaptiveAvgPool2d crashes on MPS for 19x19
        arch_type = os.environ.get("RINGRIFT_NN_ARCHITECTURE", "auto")
        self._use_mps_arch = False

        if arch_type == "mps":
            self._use_mps_arch = True
        elif arch_type == "auto":
            # Auto-select MPS architecture if MPS is available
            if torch.backends.mps.is_available() and not disable_mps and not force_cpu:
                self._use_mps_arch = True

        # Device selection - prefer MPS when using MPS architecture
        if self._use_mps_arch:
            if torch.backends.mps.is_available() and not disable_mps and not force_cpu:
                self.device = torch.device("mps")
                logger.info("Using MPS device with MPS-compatible architecture")
            else:
                self.device = torch.device("cpu")
                logger.warning("MPS architecture selected but MPS not available, " "falling back to CPU")
        else:
            # Standard device selection for default architecture
            # NOTE: V2 models use torch.mean for pooling, which is MPS-compatible
            # for all input sizes. MPS is safe to use with V2 architecture.
            if torch.cuda.is_available() and not force_cpu:
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                # Warn if MPS is available but we're using CPU due to architecture
                if torch.backends.mps.is_available() and not disable_mps and not force_cpu:
                    logger.warning(
                        "Non-MPS architecture selected but MPS available. "
                        "Using CPU to avoid AdaptiveAvgPool2d MPS limitations. "
                        "Set RINGRIFT_NN_ARCHITECTURE=auto (default) or =mps "
                        "to use MPS-compatible architecture."
                    )

        # Determine architecture type
        self.architecture_type = "mps" if self._use_mps_arch else "default"

        # Store base_dir for model path resolution
        self._base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Model will be lazily initialized when we see the first game state
        # unless board_type is explicitly provided
        self.model: Optional[nn.Module] = None

        # If board_type is explicitly provided, initialize the model now
        if board_type is not None:
            self._ensure_model_initialized(board_type)

    def _ensure_model_initialized(self, board_type: BoardType, num_players: Optional[int] = None) -> None:
        """Ensure the model is initialized for the given board type.

        This method is called lazily when the first game state is processed,
        or eagerly if board_type was passed to __init__.

        Args:
            board_type: The board type to initialize the model for.
            num_players: Optional player count for model selection.
        """
        # Already initialized for this board type?
        if self._initialized_board_type == board_type and self.model is not None:
            return

        # Already initialized for a different board type? This is an error.
        if self._initialized_board_type is not None:
            raise RuntimeError(
                f"NeuralNetAI was initialized for {self._initialized_board_type} "
                f"but is now being used with {board_type}. Create a new instance "
                f"for different board types."
            )

        import os

        # Update board_size based on board_type
        self.board_size = get_spatial_size_for_board(board_type)

        # =====================================================================
        # In-memory state_dict loading path (zero disk I/O)
        # =====================================================================
        # If nn_state_dict is provided in config, skip all file-based loading
        # and load weights directly from memory. This is used by
        # BackgroundEvaluator for efficient in-process evaluation.
        nn_state_dict = getattr(self.config, "nn_state_dict", None)
        if nn_state_dict is not None and isinstance(nn_state_dict, dict):
            logger.info("Loading model from in-memory state_dict (zero disk I/O)")
            self._init_from_state_dict(nn_state_dict, board_type, num_players)
            return

        models_dir = os.path.join(self._base_dir, "models")

        model_id = getattr(self.config, "nn_model_id", None)
        if isinstance(model_id, str) and model_id.startswith("ringrift_best_") and not model_id.endswith(".pth"):
            import glob

            # Best-model aliases are intended to be stable identifiers that
            # may not exist on a fresh checkout. If missing, fall back to the
            # built-in default selection for this board.
            if not glob.glob(os.path.join(models_dir, f"{model_id}*.pth")):
                logger.warning(
                    "nn_model_id alias %s not found under %s; falling back to built-in defaults",
                    model_id,
                    models_dir,
                )
                model_id = None
        if not model_id:
            # Default model selection.
            #
            # We intentionally do NOT fall back to deprecated v1/v1_mps ids.
            # Instead, prefer the latest canonical models for the current board
            # and player count. Callers should explicitly set nn_model_id when
            # they require a specific checkpoint (e.g. ringrift_best_* aliases).
            #
            # "v4"/"v5" here are model-id lineage prefixes (checkpoint families),
            # not architecture class names. See ai-service/docs/MPS_ARCHITECTURE.md.
            def _pick_first_existing_model_id(candidates: list[str]) -> str:
                for candidate in candidates:
                    # Prefer the exact filename, but allow timestamped variants.
                    for suffix in ("_mps.pth", ".pth"):
                        if os.path.exists(os.path.join(models_dir, f"{candidate}{suffix}")):
                            return candidate
                    try:
                        from pathlib import Path

                        if list(Path(models_dir).glob(f"{candidate}_*.pth")):
                            return candidate
                    except Exception:
                        pass
                return candidates[0]

            players_for_alias = int(num_players or 2)

            if board_type == BoardType.SQUARE8:
                if num_players == 3:
                    model_id = _pick_first_existing_model_id(
                        [
                            f"ringrift_best_sq8_{players_for_alias}p",
                            "ringrift_v5_sq8_3p",
                            "ringrift_v4_sq8_3p",
                        ]
                    )
                elif num_players == 4:
                    model_id = _pick_first_existing_model_id(
                        [
                            f"ringrift_best_sq8_{players_for_alias}p",
                            "ringrift_v5_sq8_4p",
                            "ringrift_v4_sq8_4p",
                        ]
                    )
                else:
                    model_id = _pick_first_existing_model_id(
                        [
                            f"ringrift_best_sq8_{players_for_alias}p",
                            "ringrift_v5_sq8_2p_2xh100",
                            "ringrift_v4_sq8_2p",
                        ]
                    )
            elif board_type == BoardType.SQUARE19:
                model_id = _pick_first_existing_model_id(
                    [
                        f"ringrift_best_sq19_{players_for_alias}p",
                        f"ringrift_v4_sq19_{players_for_alias}p",
                    ]
                )
            elif board_type == BoardType.HEX8:
                model_id = _pick_first_existing_model_id(
                    [
                        f"ringrift_best_hex8_{players_for_alias}p",
                        f"ringrift_v5_hex8_{players_for_alias}p",
                    ]
                )
            else:  # HEXAGONAL
                model_id = _pick_first_existing_model_id(
                    [
                        f"ringrift_best_hex_{players_for_alias}p",
                        f"ringrift_v4_hex_{players_for_alias}p",
                    ]
                )
            logger.info(
                "AIConfig.nn_model_id not set; defaulting to %s for board=%s players=%s",
                model_id,
                board_type,
                num_players,
            )

        # Allow nn_model_id to be an explicit checkpoint path (useful for
        # evaluation harnesses that want to compare two different checkpoints in
        # the same process).
        explicit_checkpoint_path: Optional[str] = None
        if isinstance(model_id, str) and model_id.endswith(".pth"):
            from pathlib import Path

            candidate = Path(model_id).expanduser()
            candidates: list[Path] = []
            if candidate.is_absolute():
                candidates.append(candidate)
            else:
                candidates.append((Path.cwd() / candidate))
                candidates.append((Path(self._base_dir) / candidate))
            for path in candidates:
                try:
                    if path.is_file() and path.stat().st_size > 0:
                        explicit_checkpoint_path = str(path.resolve())
                        break
                except OSError:
                    continue

        def _model_id_includes_board_hint(value: str) -> bool:
            lower = value.lower()
            return any(
                token in lower
                for token in (
                    "sq8",
                    "square8",
                    "sq19",
                    "square19",
                    "19x19",
                    "hex8",
                    "hex",
                    "hexagonal",
                )
            )

        # When nn_model_id points at an explicit checkpoint file, skip the
        # model-id resolution logic and treat it as the resolved path.
        board_suffix = ""
        model_path: str
        if explicit_checkpoint_path is not None:
            model_path = explicit_checkpoint_path
        else:
            # Board-type-specific model path (e.g., ringrift_v4_hex_2p_mps.pth)
            if not _model_id_includes_board_hint(model_id):
                if board_type == BoardType.HEXAGONAL:
                    board_suffix = "_hex"
                elif board_type == BoardType.SQUARE19:
                    board_suffix = "_19x19"
            # SQUARE8 uses the base model name (legacy compatibility)

            # Architecture-specific checkpoint naming
            if self.architecture_type == "mps":
                model_filename = f"{model_id}{board_suffix}_mps.pth"
            else:
                model_filename = f"{model_id}{board_suffix}.pth"

            model_path = os.path.join(self._base_dir, "models", model_filename)

        # Resolve a usable checkpoint path before building the model so we can
        # match architecture hyperparameters to the checkpoint metadata.
        models_dir = os.path.join(self._base_dir, "models")
        allow_fresh = bool(getattr(self.config, "allow_fresh_weights", False))
        if os.environ.get("RINGRIFT_REQUIRE_NEURAL_NET", "").lower() in {"1", "true", "yes"}:
            # Fail-fast mode: never allow random-weight fallback when neural
            # tiers are expected to be functional.
            allow_fresh = False

        chosen_path = explicit_checkpoint_path
        if chosen_path is None:
            arch_suffix = "_mps" if self.architecture_type == "mps" else ""
            other_arch_suffix = "" if arch_suffix == "_mps" else "_mps"

            candidate_filenames = [
                f"{model_id}{board_suffix}{arch_suffix}.pth",
                f"{model_id}{board_suffix}{other_arch_suffix}.pth",
            ]
            if board_suffix:
                candidate_filenames.extend(
                    [
                        f"{model_id}{arch_suffix}.pth",
                        f"{model_id}{other_arch_suffix}.pth",
                    ]
                )

            seen: set[str] = set()
            candidate_paths: list[str] = []
            for filename in candidate_filenames:
                if filename in seen:
                    continue
                seen.add(filename)
                candidate_paths.append(os.path.join(models_dir, filename))

            def _is_usable_checkpoint(path: str) -> bool:
                try:
                    return os.path.isfile(path) and os.path.getsize(path) > 0
                except OSError:
                    return False

            chosen_path = next((p for p in candidate_paths if _is_usable_checkpoint(p)), None)
        if chosen_path is None:
            import glob

            prefix = f"{model_id}{board_suffix}"
            patterns = []
            if self.architecture_type == "mps":
                patterns.append(os.path.join(models_dir, f"{prefix}_*_mps.pth"))
            patterns.append(os.path.join(models_dir, f"{prefix}_*.pth"))

            latest_matches: list[str] = []
            for pattern in patterns:
                latest_matches.extend(glob.glob(pattern))
            latest_matches = sorted(
                p for p in set(latest_matches) if _is_usable_checkpoint(p)
            )

            def _is_loadable_checkpoint(path: str) -> bool:
                """Return True iff we can torch.load the checkpoint.

                Some long-running training jobs may leave behind truncated
                checkpoints (EOFError) even though the file exists and has a
                non-zero size. Selecting the "latest" file by mtime/name alone
                can therefore cause neural tiers to fail and silently fall back
                to heuristic rollouts.
                """
                try:
                    try:
                        checkpoint_obj = torch.load(
                            path,
                            map_location="cpu",
                            weights_only=False,
                        )
                    except TypeError:
                        checkpoint_obj = torch.load(
                            path,
                            map_location="cpu",
                        )
                except Exception:
                    return False

                if not isinstance(checkpoint_obj, dict):
                    return False

                # Accept versioned and legacy checkpoint layouts.
                if any(
                    k in checkpoint_obj
                    for k in (
                        "model_state_dict",
                        "state_dict",
                        "_versioning_metadata",
                        "conv1.weight",
                        "module.conv1.weight",
                        "policy_fc2.weight",
                        "module.policy_fc2.weight",
                    )
                ):
                    return True

                return False

            if latest_matches:
                # Iterate newest→oldest, selecting the first loadable checkpoint.
                max_probe = int(os.environ.get("RINGRIFT_NN_RESOLVE_MAX_PROBE", "25") or "25")
                for candidate in reversed(latest_matches[-max_probe:]):
                    if _is_loadable_checkpoint(candidate):
                        chosen_path = candidate
                        break

        effective_model_path = chosen_path or model_path

        # Include a lightweight file signature in the cache key so stable
        # aliases (e.g. ringrift_best_*) can be atomically replaced and picked
        # up by new games without restarting the service.
        checkpoint_signature: tuple[int, int] | None = None
        try:
            st = os.stat(effective_model_path)
            checkpoint_signature = (int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))), int(st.st_size))
        except OSError:
            checkpoint_signature = None

        self.loaded_checkpoint_path = effective_model_path
        self.loaded_checkpoint_signature = checkpoint_signature

        # Build cache key including the resolved checkpoint path and board_type.
        cache_key = (
            self.architecture_type,
            str(self.device),
            effective_model_path,
            checkpoint_signature,
            board_type.value if board_type else "unknown",
        )

        if cache_key in _MODEL_CACHE:
            import time
            model, created_at, _ = _MODEL_CACHE[cache_key]
            # Update last access time
            _MODEL_CACHE[cache_key] = (model, created_at, time.time())
            self.model = model
            cached_history_length = getattr(self.model, "_ringrift_history_length", None)
            if cached_history_length is not None:
                try:
                    self.history_length = int(cached_history_length)
                except Exception:
                    pass
            cached_feature_version = getattr(self.model, "_ringrift_feature_version", None)
            if cached_feature_version is None:
                cached_feature_version = getattr(self.model, "feature_version", None)
            if cached_feature_version is not None:
                try:
                    self.feature_version = int(cached_feature_version)
                except Exception:
                    pass
            self._initialized_board_type = board_type

            # Initialize hex encoder for hex boards (must also do this for cached models)
            # Check model's conv1 weight shape to determine encoder version
            if board_type in (BoardType.HEXAGONAL, BoardType.HEX8) and self._hex_encoder is None:
                # Infer encoder version from model's input channels
                model_in_channels = None
                if hasattr(self.model, 'conv1') and hasattr(self.model.conv1, 'weight'):
                    model_in_channels = self.model.conv1.weight.shape[1]

                use_v2_encoder = model_in_channels == 40

                if use_v2_encoder:
                    from ..training.encoding import HexStateEncoder
                    if board_type == BoardType.HEX8:
                        self._hex_encoder = HexStateEncoder(
                            board_size=HEX8_BOARD_SIZE,
                            policy_size=POLICY_SIZE_HEX8,
                            feature_version=self.feature_version,
                        )
                    else:
                        self._hex_encoder = HexStateEncoder(
                            board_size=HEX_BOARD_SIZE,
                            policy_size=P_HEX,
                            feature_version=self.feature_version,
                        )
                else:
                    from ..training.encoding import HexStateEncoderV3
                    if board_type == BoardType.HEX8:
                        self._hex_encoder = HexStateEncoderV3(
                            board_size=HEX8_BOARD_SIZE,
                            policy_size=POLICY_SIZE_HEX8,
                            feature_version=self.feature_version,
                        )
                    else:
                        self._hex_encoder = HexStateEncoderV3(
                            board_size=HEX_BOARD_SIZE,
                            policy_size=P_HEX,
                            feature_version=self.feature_version,
                        )

            logger.debug(
                f"Reusing cached model: board={board_type}, "
                f"arch={self.architecture_type}, device={self.device}"
            )
            return

        # Defaults for fresh weights / unknown metadata.
        #
        # IMPORTANT: Our current canonical v2 checkpoints use the "high" tier
        # (12 res blocks, 192 filters). Defaulting to the historical 10/128 can
        # trigger checkpoint shape mismatches (e.g., value_fc1 in_features
        # checkpoint=212 vs expected=148) when metadata cannot be read.
        num_res_blocks = 12
        num_filters = 192
        num_players_override = 4
        policy_size_override: Optional[int] = None
        in_channels_override: Optional[int] = None  # Input channels from checkpoint metadata
        model_class_name: Optional[str] = None
        memory_tier_override: Optional[str] = None
        history_length_override = self.history_length
        feature_version_override: Optional[int] = None

        if chosen_path is not None:
            try:
                # NOTE: PyTorch 2.6+ supports the `weights_only` kwarg; older
                # versions do not. Prefer weights_only=False so we can read
                # versioning metadata when present, but fall back gracefully.
                try:
                    checkpoint = torch.load(
                        chosen_path,
                        map_location="cpu",
                        weights_only=False,
                    )
                except TypeError:
                    checkpoint = torch.load(
                        chosen_path,
                        map_location="cpu",
                    )
                if isinstance(checkpoint, dict):
                    meta = checkpoint.get("_versioning_metadata") or {}
                    if isinstance(meta, dict):
                        cfg = meta.get("config") or {}
                        if isinstance(cfg, dict):
                            num_res_blocks = int(cfg.get("num_res_blocks") or num_res_blocks)
                            num_filters = int(cfg.get("num_filters") or num_filters)
                            num_players_override = int(cfg.get("num_players") or num_players_override)
                            if cfg.get("history_length") is not None:
                                history_length_override = int(cfg.get("history_length"))
                            if cfg.get("feature_version") is not None:
                                feature_version_override = int(cfg.get("feature_version"))
                            policy_size_override = (
                                int(cfg.get("policy_size"))
                                if cfg.get("policy_size") is not None
                                else None
                            )
                            # Extract in_channels from checkpoint metadata to match model architecture
                            if cfg.get("in_channels") is not None:
                                in_channels_override = int(cfg.get("in_channels"))
                        raw_class = meta.get("model_class")
                        if isinstance(raw_class, str):
                            model_class_name = raw_class
                            lower_name = raw_class.lower()
                            if "v3" in lower_name:
                                memory_tier_override = "v3-low" if "lite" in lower_name else "v3-high"
                            elif "lite" in lower_name:
                                memory_tier_override = "low"
                            else:
                                memory_tier_override = "high"
                    # Fallback: check for top-level policy_size (e.g., non-versioned checkpoints)
                    if policy_size_override is None and checkpoint.get("policy_size") is not None:
                        policy_size_override = int(checkpoint["policy_size"])
                        logger.info(
                            "Using top-level policy_size=%s from checkpoint",
                            policy_size_override,
                        )
                        # Also default to RingRiftCNN_v2 if no model class specified
                        # (ensures we use the direct instantiation path that respects policy_size)
                        if model_class_name is None:
                            model_class_name = "RingRiftCNN_v2"

                    # Cross-check metadata against the actual weight shapes.
                    # This hardens against metadata drift and protects callers
                    # that rely on model_id prefixes.
                    def _extract_state_dict(candidate: object) -> Optional[dict]:
                        """Extract a raw state_dict from a loaded checkpoint.

                        Supports:
                        - Versioned checkpoints (model_state_dict key)
                        - Common legacy keys (state_dict)
                        - Bare state_dict checkpoints (direct mapping of weight tensors)
                        """

                        if not isinstance(candidate, dict):
                            return None

                        state_dict_obj = candidate.get("model_state_dict")
                        if isinstance(state_dict_obj, dict):
                            return state_dict_obj

                        state_dict_obj = candidate.get("state_dict")
                        if isinstance(state_dict_obj, dict):
                            return state_dict_obj

                        # Best-effort: treat as bare state_dict when keys look like weights.
                        if any(
                            key in candidate
                            for key in (
                                "conv1.weight",
                                "module.conv1.weight",
                                "value_fc1.weight",
                                "module.value_fc1.weight",
                                "policy_fc2.weight",
                                "module.policy_fc2.weight",
                            )
                        ):
                            return candidate

                        return None

                    state_dict = _extract_state_dict(checkpoint)
                    if isinstance(state_dict, dict):
                        state_dict = _strip_module_prefix(state_dict)
                        conv1_weight = state_dict.get("conv1.weight")
                        if conv1_weight is not None and hasattr(conv1_weight, "shape"):
                            inferred_filters = int(conv1_weight.shape[0])
                            if inferred_filters and inferred_filters != num_filters:
                                logger.warning(
                                    "Checkpoint metadata num_filters=%s disagrees with weights (%s); "
                                    "using inferred value.",
                                    num_filters,
                                    inferred_filters,
                                )
                                num_filters = inferred_filters
                            inferred_in_channels = int(conv1_weight.shape[1])
                            if inferred_in_channels and inferred_in_channels % 14 == 0:
                                inferred_frames = inferred_in_channels // 14
                                inferred_history = inferred_frames - 1
                                if (
                                    inferred_history >= 0
                                    and inferred_history != history_length_override
                                ):
                                    logger.warning(
                                        "Checkpoint metadata history_length=%s disagrees with weights (%s); "
                                        "using inferred value.",
                                        history_length_override,
                                        inferred_history,
                                    )
                                    history_length_override = inferred_history

                        # Infer num_players from the value head when metadata is absent.
                        #
                        # V2 checkpoints are typically trained with num_players=4 even for 2p,
                        # but V3 checkpoints may be trained with num_players matching the
                        # target configuration (e.g. 2p → value_fc2.out_features == 2).
                        #
                        # If we initialize the model with the wrong num_players, we will hit
                        # shape mismatches for value_fc2 / rank_dist_fc2 and neural tiers will
                        # silently fall back to heuristic rollouts in search AIs.
                        value_fc2_weight = state_dict.get("value_fc2.weight")
                        if value_fc2_weight is not None and hasattr(value_fc2_weight, "shape"):
                            inferred_players = int(value_fc2_weight.shape[0])
                            if inferred_players in (2, 3, 4) and inferred_players != num_players_override:
                                logger.warning(
                                    "Checkpoint metadata num_players=%s disagrees with weights (%s); "
                                    "using inferred value.",
                                    num_players_override,
                                    inferred_players,
                                )
                                num_players_override = inferred_players

                        # Infer model architecture from weight keys when metadata is absent.
                        # V3 models have spatial policy heads; V2 models have FC policy heads.
                        # V4 models have attention blocks (query/key/value) in residual layers.
                        # This is the definitive way to distinguish them.
                        v3_spatial_keys = (
                            "placement_conv.weight",
                            "movement_conv.weight",
                            "line_form_conv.weight",
                            "territory_claim_conv.weight",
                            "territory_choice_conv.weight",
                        )
                        v3_rank_dist_keys = ("rank_dist_fc1.weight", "rank_dist_fc2.weight")
                        v2_policy_keys = ("policy_fc1.weight", "policy_fc2.weight")
                        # V4 uses AttentionResidualBlock with query/key/value convolutions
                        v4_attention_patterns = ("res_blocks.0.query.weight", "res_blocks.0.key.weight")

                        has_v3_spatial = any(k in state_dict for k in v3_spatial_keys)
                        has_v3_rank_dist = any(k in state_dict for k in v3_rank_dist_keys)
                        has_v2_policy = any(k in state_dict for k in v2_policy_keys)
                        has_v4_attention = any(k in state_dict for k in v4_attention_patterns)

                        if model_class_name is None:
                            if has_v4_attention:
                                # V4 model - NAS-optimized with attention blocks
                                model_class_name = "RingRiftCNN_v4"
                                memory_tier_override = "v3-high"  # V4 uses same tier as V3
                                logger.info(
                                    "Auto-detected RingRiftCNN_v4 from checkpoint weight keys "
                                    "(attention blocks detected)"
                                )
                            elif has_v3_spatial or has_v3_rank_dist:
                                # Definitely a V3 model - use spatial policy heads
                                # Check for lite variant by looking at filter count
                                is_lite = num_filters <= 96
                                model_class_name = "RingRiftCNN_v3_Lite" if is_lite else "RingRiftCNN_v3"
                                memory_tier_override = "v3-low" if is_lite else "v3-high"
                                logger.info(
                                    "Auto-detected %s from checkpoint weight keys "
                                    "(spatial_heads=%s, rank_dist=%s)",
                                    model_class_name,
                                    has_v3_spatial,
                                    has_v3_rank_dist,
                                )
                            elif has_v2_policy:
                                # Definitely a V2 model - use FC policy heads
                                is_lite = num_filters <= 96
                                model_class_name = "RingRiftCNN_v2_Lite" if is_lite else "RingRiftCNN_v2"
                                memory_tier_override = "low" if is_lite else "high"
                                logger.info(
                                    "Auto-detected %s from checkpoint weight keys (policy_fc heads)",
                                    model_class_name,
                                )

                        # Infer policy_size when metadata is absent (only for V2 models).
                        if policy_size_override is None and has_v2_policy:
                            policy_fc2_weight = state_dict.get("policy_fc2.weight")
                            if policy_fc2_weight is not None and hasattr(policy_fc2_weight, "shape"):
                                inferred_policy_size = int(policy_fc2_weight.shape[0])
                                if inferred_policy_size:
                                    policy_size_override = inferred_policy_size

                        # Infer res-block count from state_dict keys when possible.
                        # We avoid importing regex at module level to keep import
                        # time down for inference.
                        inferred_blocks = None
                        try:
                            import re

                            indices = set()
                            for key in state_dict.keys():
                                if not isinstance(key, str):
                                    continue
                                m = re.match(r"(?:module\.)?res_blocks\.(\d+)\.", key)
                                if m:
                                    indices.add(int(m.group(1)))
                            if indices:
                                inferred_blocks = max(indices) + 1
                        except Exception:
                            inferred_blocks = None

                        if inferred_blocks and inferred_blocks != num_res_blocks:
                            logger.warning(
                                "Checkpoint metadata num_res_blocks=%s disagrees with weights (%s); "
                                "using inferred value.",
                                num_res_blocks,
                                inferred_blocks,
                            )
                            num_res_blocks = inferred_blocks
            except Exception as e:
                logger.debug(
                    "Failed to read checkpoint metadata for %s: %s",
                    chosen_path,
                    e,
                )

        if feature_version_override is not None:
            self.feature_version = feature_version_override

        # Create new model. When metadata specifies a square-board model class,
        # instantiate it directly so we can respect a fixed policy_size from the
        # checkpoint (MAX_N head) even on square8. Otherwise fall back to the
        # board-based factory.
        square_model_classes: Dict[str, Any] = {
            "RingRiftCNN_v2": RingRiftCNN_v2,
            "RingRiftCNN_v2_Lite": RingRiftCNN_v2_Lite,
            "RingRiftCNN_v3": RingRiftCNN_v3,
            "RingRiftCNN_v3_Lite": RingRiftCNN_v3_Lite,
            "RingRiftCNN_v4": RingRiftCNN_v4,
        }

        if model_class_name in square_model_classes:
            cls = square_model_classes[model_class_name]
            self.model = cls(
                board_size=self.board_size,
                in_channels=14,
                global_features=20,
                num_res_blocks=num_res_blocks,
                num_filters=num_filters,
                history_length=history_length_override,
                policy_size=policy_size_override,
                num_players=num_players_override,
            )
        else:
            # Hex boards use different in_channels depending on encoder version:
            # - HexStateEncoderV3: 16 base channels (for HexNeuralNet_v3)
            # - HexStateEncoder (v2): 10 base channels (for HexNeuralNet_v2)
            # Square boards use 14 base channels.
            #
            # Note: Checkpoint metadata stores TOTAL in_channels (base × (history_length + 1)),
            # but create_model_for_board expects BASE channels.
            # We need to convert total to base by dividing by (history_length + 1).
            if in_channels_override is not None:
                # Convert total channels to base channels
                history_frames = history_length_override + 1  # e.g., 4 for history_length=3
                if in_channels_override % history_frames == 0:
                    effective_in_channels = in_channels_override // history_frames
                    logger.info(
                        "Converted total in_channels=%d to base in_channels=%d (history_frames=%d)",
                        in_channels_override,
                        effective_in_channels,
                        history_frames,
                    )
                else:
                    # If not evenly divisible, assume it's already base channels
                    effective_in_channels = in_channels_override
            elif board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
                # Default to V3 encoder (16 channels) for hex boards
                effective_in_channels = 16
            else:
                effective_in_channels = 14
            self.model = create_model_for_board(
                board_type=board_type,
                in_channels=effective_in_channels,
                global_features=20,
                num_res_blocks=num_res_blocks,
                num_filters=num_filters,
                history_length=history_length_override,
                memory_tier=memory_tier_override,
                num_players=num_players_override,
                policy_size=policy_size_override,
            )
        # Ensure the encoder uses the same history length as the loaded model.
        self.history_length = history_length_override
        logger.info(
            "Initialized %s for %s from %s (res_blocks=%s, filters=%s)",
            type(self.model).__name__,
            board_type,
            os.path.basename(effective_model_path),
            num_res_blocks,
            num_filters,
        )

        self.model.to(self.device)

        if chosen_path is not None:
            if chosen_path != model_path:
                logger.info(
                    "NeuralNetAI checkpoint fallback: requested=%s, using=%s",
                    model_path,
                    chosen_path,
                )
            try:
                self._load_model_checkpoint(chosen_path)
            except RuntimeError as e:
                if allow_fresh:
                    logger.warning(
                        "Checkpoint incompatible (%s); using fresh weights "
                        "(allow_fresh_weights=True).",
                        e,
                    )
                    self.model.eval()
                else:
                    raise
        else:
            if allow_fresh:
                logger.info(
                    "No model found at %s; using fresh weights "
                    "(allow_fresh_weights=True).",
                    model_path,
                )
                self.model.eval()
            else:
                raise FileNotFoundError(
                    f"No neural-net checkpoint found for nn_model_id={model_id!r} "
                    f"(looked for {model_path}).\n"
                    "Provide a matching checkpoint under ai-service/models/, "
                    "or set AIConfig.allow_fresh_weights=True for offline "
                    "experiments that intentionally start from random weights."
                )

        # Apply torch.compile() optimization for faster inference on CUDA
        # This provides 2-3x speedup for batch inference
        try:
            from .gpu_batch import compile_model
            dev_type = self.device.type if isinstance(self.device, torch.device) else str(self.device)
            if dev_type not in {"cpu", "mps"}:
                self.model = compile_model(
                    self.model,
                    device=torch.device(self.device) if isinstance(self.device, str) else self.device,
                    mode="reduce-overhead",
                )
        except ImportError:
            pass  # gpu_batch not available, skip compilation
        except Exception as e:
            logger.debug(f"torch.compile() skipped: {e}")

        # Record the expected history length on the cached model so future
        # NeuralNetAI wrappers that reuse it keep encoder/channel alignment.
        try:
            setattr(self.model, "_ringrift_history_length", int(self.history_length))
        except Exception:
            pass
        try:
            setattr(self.model, "_ringrift_feature_version", int(self.feature_version))
        except Exception:
            pass
        try:
            setattr(self.model, "feature_version", int(self.feature_version))
        except Exception:
            pass

        # Initialize hex encoder for hex boards
        # Select encoder version based on model architecture:
        # - HexNeuralNet_v2 models expect 40 channels (10 base × 4 frames), use HexStateEncoder
        # - HexNeuralNet_v3 models expect 64 channels (16 base × 4 frames), use HexStateEncoderV3
        if board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
            # Check if we need V2 encoder (for 40-channel models)
            use_v2_encoder = (
                (model_class_name and "v2" in model_class_name.lower() and "v3" not in model_class_name.lower())
                or (in_channels_override is not None and in_channels_override == 40)
            )

            if use_v2_encoder:
                # Use V2 encoder (10 base channels) for older models
                from ..training.encoding import HexStateEncoder

                if board_type == BoardType.HEX8:
                    self._hex_encoder = HexStateEncoder(
                        board_size=HEX8_BOARD_SIZE,
                        policy_size=POLICY_SIZE_HEX8,
                        feature_version=self.feature_version,
                    )
                else:
                    self._hex_encoder = HexStateEncoder(
                        board_size=HEX_BOARD_SIZE,
                        policy_size=P_HEX,
                        feature_version=self.feature_version,
                    )
                logger.info(
                    "Initialized HexStateEncoder (V2) for %s (board_size=%d, in_channels=%s)",
                    board_type,
                    self._hex_encoder.board_size,
                    in_channels_override or 40,
                )
            else:
                # Use V3 encoder (16 base channels) for newer models
                from ..training.encoding import HexStateEncoderV3

                if board_type == BoardType.HEX8:
                    self._hex_encoder = HexStateEncoderV3(
                        board_size=HEX8_BOARD_SIZE,
                        policy_size=POLICY_SIZE_HEX8,
                        feature_version=self.feature_version,
                    )
                else:
                    self._hex_encoder = HexStateEncoderV3(
                        board_size=HEX_BOARD_SIZE,
                        policy_size=P_HEX,
                        feature_version=self.feature_version,
                    )
                logger.info(
                    "Initialized HexStateEncoderV3 for %s (board_size=%d)",
                    board_type,
                    self._hex_encoder.board_size,
                )

        # Cache the model for reuse with LRU tracking
        import time
        now = time.time()
        _MODEL_CACHE[cache_key] = (self.model, now, now)  # (model, created_at, last_access)

        # Periodically evict stale models to prevent memory bloat
        _evict_stale_models()

        self._initialized_board_type = board_type
        logger.info(
            f"Cached model: board={board_type}, arch={self.architecture_type}, "
            f"device={self.device} (total cached: {len(_MODEL_CACHE)})"
        )

    def _init_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        board_type: BoardType,
        num_players: Optional[int] = None,
    ) -> None:
        """Initialize model from an in-memory state_dict (zero disk I/O).

        This is the fast path used by BackgroundEvaluator to avoid writing
        checkpoints to disk. Architecture parameters are inferred from tensor
        shapes in the state_dict.

        Args:
            state_dict: PyTorch state_dict (parameter name -> tensor)
            board_type: Board type for the model
            num_players: Optional player count override
        """
        import re

        # Strip module. prefix (DDP compatibility)
        state_dict = _strip_module_prefix(state_dict)

        # Infer architecture parameters from state_dict tensor shapes
        num_filters = 192  # Default
        num_res_blocks = 12  # Default
        num_players_override = num_players or 2
        history_length_override = 3
        policy_size_override: Optional[int] = None
        in_channels_override: Optional[int] = None

        # Infer num_filters from conv1.weight shape[0]
        conv1_weight = state_dict.get("conv1.weight")
        if conv1_weight is not None and hasattr(conv1_weight, "shape"):
            num_filters = int(conv1_weight.shape[0])

        # Infer in_channels from conv1.weight shape[1]
        # conv1.weight.shape[1] is TOTAL channels (base * (history_length + 1))
        # We need to extract BASE channels for model constructor
        if conv1_weight is not None and hasattr(conv1_weight, "shape"):
            total_channels = int(conv1_weight.shape[1])
            # Try to infer history_length from total_channels
            # Common base channels: 14 (square), 16 (hex v3), 10 (hex v2)
            for base in (14, 16, 10, 12):
                if total_channels % base == 0:
                    frames = total_channels // base
                    if frames in (1, 2, 3, 4, 5):  # history_length 0-4
                        in_channels_override = base
                        history_length_override = frames - 1
                        break
            else:
                # Fallback: assume 4 frames (history_length=3)
                in_channels_override = total_channels // 4 if total_channels >= 4 else total_channels

        # Infer num_players from value_fc2.weight shape[0]
        value_fc2_weight = state_dict.get("value_fc2.weight")
        if value_fc2_weight is not None and hasattr(value_fc2_weight, "shape"):
            inferred_players = int(value_fc2_weight.shape[0])
            if inferred_players in (2, 3, 4):
                num_players_override = inferred_players

        # Infer policy_size from policy_fc2.weight shape[0] (V2 models)
        policy_fc2_weight = state_dict.get("policy_fc2.weight")
        if policy_fc2_weight is not None and hasattr(policy_fc2_weight, "shape"):
            policy_size_override = int(policy_fc2_weight.shape[0])

        # Infer res-block count from state_dict keys
        indices: set[int] = set()
        for key in state_dict.keys():
            if isinstance(key, str):
                m = re.match(r"res_blocks\.(\d+)\.", key)
                if m:
                    indices.add(int(m.group(1)))
        if indices:
            num_res_blocks = max(indices) + 1

        # Determine model class from state_dict signatures
        v3_spatial_keys = ("spatial_policy_conv.weight",)
        v3_rank_dist_keys = ("rank_dist_fc1.weight", "rank_dist_fc2.weight")
        v2_policy_keys = ("policy_fc1.weight", "policy_fc2.weight")
        v4_attention_patterns = ("res_blocks.0.query.weight", "res_blocks.0.key.weight")

        has_v3_spatial = any(k in state_dict for k in v3_spatial_keys)
        has_v3_rank_dist = any(k in state_dict for k in v3_rank_dist_keys)
        has_v2_policy = any(k in state_dict for k in v2_policy_keys)
        has_v4_attention = any(k in state_dict for k in v4_attention_patterns)

        # Check value_fc1 hidden size to distinguish Lite from full models
        # V2/V3 full: 128 hidden units, Lite: 64 hidden units
        value_fc1_weight = state_dict.get("value_fc1.weight")
        is_lite_fc = False
        if value_fc1_weight is not None and hasattr(value_fc1_weight, "shape"):
            hidden_units = int(value_fc1_weight.shape[0])
            is_lite_fc = hidden_units <= 64

        if has_v4_attention:
            model_class_name = "RingRiftCNN_v4"
        elif has_v3_spatial and has_v3_rank_dist:
            # Lite models have smaller FC layers (is_lite_fc) - prioritize this check
            if is_lite_fc:
                model_class_name = "RingRiftCNN_v3_Lite"
            else:
                model_class_name = "RingRiftCNN_v3"
        elif has_v2_policy:
            # Lite models have smaller FC layers (is_lite_fc) - prioritize this check
            if is_lite_fc:
                model_class_name = "RingRiftCNN_v2_Lite"
            else:
                model_class_name = "RingRiftCNN_v2"
        else:
            model_class_name = "RingRiftCNN_v3"  # Default

        logger.info(
            "Inferred architecture from state_dict: class=%s, filters=%d, blocks=%d, players=%d",
            model_class_name,
            num_filters,
            num_res_blocks,
            num_players_override,
        )

        # Build the model
        square_model_classes = {
            "RingRiftCNN_v2": RingRiftCNN_v2,
            "RingRiftCNN_v2_Lite": RingRiftCNN_v2_Lite,
            "RingRiftCNN_v3": RingRiftCNN_v3,
            "RingRiftCNN_v3_Lite": RingRiftCNN_v3_Lite,
            "RingRiftCNN_v4": RingRiftCNN_v4,
        }

        if model_class_name in square_model_classes:
            cls = square_model_classes[model_class_name]
            self.model = cls(
                board_size=self.board_size,
                in_channels=in_channels_override or 14,
                global_features=20,
                num_res_blocks=num_res_blocks,
                num_filters=num_filters,
                history_length=history_length_override,
                policy_size=policy_size_override,
                num_players=num_players_override,
            )
        else:
            # Fallback to create_model_for_board
            self.model = create_model_for_board(
                board_type=board_type,
                in_channels=in_channels_override or 14,
                global_features=20,
                num_res_blocks=num_res_blocks,
                num_filters=num_filters,
                history_length=history_length_override,
                num_players=num_players_override,
                policy_size=policy_size_override,
            )

        self.history_length = history_length_override
        self.model.to(self.device)

        # Load the state_dict
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self._initialized_board_type = board_type
        self.loaded_checkpoint_path = "<in-memory>"
        logger.info(
            "Initialized %s from in-memory state_dict (filters=%d, blocks=%d, players=%d)",
            model_class_name,
            num_filters,
            num_res_blocks,
            num_players_override,
        )

    def _maybe_rebuild_model_to_match_checkpoint(
        self,
        state_dict: Dict[str, torch.Tensor],
        metadata: Any,
    ) -> bool:
        """Best-effort rebuild of ``self.model`` to match checkpoint shapes.

        This is a last-resort guard for situations where the runtime model was
        instantiated with stale hyperparameters (historically 10/128 filters)
        but the checkpoint was trained with the canonical 12/192 filters,
        leading to errors like:

            value_fc1 in_features: checkpoint=212, expected=148

        When possible, we infer the correct configuration from the checkpoint
        weights and reconstruct the model in-place so neural tiers do not
        silently fall back to heuristic rollouts.
        """
        if self.model is None:
            return False

        model_class = getattr(metadata, "model_class", None)
        if not isinstance(model_class, str):
            model_class = self.model.__class__.__name__

        # Only handle the square-board CNN families here. Hex networks have a
        # different value-head geometry (1 + global_features).
        if not model_class.startswith("RingRiftCNN_"):
            return False

        conv1_weight = state_dict.get("conv1.weight")
        value_fc1_weight = state_dict.get("value_fc1.weight")
        value_fc2_weight = state_dict.get("value_fc2.weight")
        policy_fc2_weight = state_dict.get("policy_fc2.weight")

        # Canonical encoder constants for square boards.
        base_in_channels = 14
        global_features = 20

        inferred_filters: Optional[int] = None
        inferred_history_length: Optional[int] = None
        if conv1_weight is not None and hasattr(conv1_weight, "shape"):
            inferred_filters = int(conv1_weight.shape[0])
            inferred_in_channels = int(conv1_weight.shape[1])
            if inferred_in_channels % base_in_channels == 0:
                frames = inferred_in_channels // base_in_channels
                inferred_history_length = max(frames - 1, 0)

        if inferred_filters is None and value_fc1_weight is not None and hasattr(value_fc1_weight, "shape"):
            inferred_filters = int(value_fc1_weight.shape[1]) - global_features

        inferred_num_players: Optional[int] = None
        if value_fc2_weight is not None and hasattr(value_fc2_weight, "shape"):
            inferred_num_players = int(value_fc2_weight.shape[0])

        inferred_policy_size: Optional[int] = None
        if policy_fc2_weight is not None and hasattr(policy_fc2_weight, "shape"):
            inferred_policy_size = int(policy_fc2_weight.shape[0])

        cfg = getattr(metadata, "config", None)
        if not isinstance(cfg, dict):
            cfg = {}

        if inferred_policy_size is None and cfg.get("policy_size") is not None:
            try:
                inferred_policy_size = int(cfg["policy_size"])
            except Exception:
                inferred_policy_size = None

        current_filters = getattr(self.model, "num_filters", None)
        current_policy_size = getattr(self.model, "policy_size", None)
        current_value_out = 0
        if hasattr(self.model, "value_fc2"):
            current_value_out = int(getattr(self.model.value_fc2, "out_features", 0) or 0)

        needs_rebuild = False
        if inferred_filters is not None and current_filters is not None:
            if int(current_filters) != inferred_filters:
                needs_rebuild = True
        if inferred_policy_size is not None and current_policy_size is not None:
            if int(current_policy_size) != inferred_policy_size:
                needs_rebuild = True
        if inferred_history_length is not None and int(self.history_length) != inferred_history_length:
            needs_rebuild = True
        if inferred_num_players in (2, 3, 4) and current_value_out and inferred_num_players != current_value_out:
            needs_rebuild = True

        if not needs_rebuild:
            return False

        num_filters = inferred_filters or (int(current_filters) if current_filters is not None else 192)
        history_length = inferred_history_length if inferred_history_length is not None else int(self.history_length)
        num_players = inferred_num_players if inferred_num_players in (2, 3, 4) else (current_value_out or 4)

        num_res_blocks = None
        if cfg.get("num_res_blocks") is not None:
            try:
                num_res_blocks = int(cfg["num_res_blocks"])
            except Exception:
                num_res_blocks = None
        if num_res_blocks is None and hasattr(self.model, "res_blocks"):
            try:
                num_res_blocks = len(self.model.res_blocks)  # type: ignore[arg-type]
            except Exception:
                num_res_blocks = None
        if num_res_blocks is None:
            # Infer from state_dict keys.
            try:
                import re

                indices = set()
                for key in state_dict.keys():
                    if not isinstance(key, str):
                        continue
                    m = re.match(r"res_blocks\.(\d+)\.", key)
                    if m:
                        indices.add(int(m.group(1)))
                if indices:
                    num_res_blocks = max(indices) + 1
            except Exception:
                num_res_blocks = None
        if num_res_blocks is None:
            num_res_blocks = 12

        policy_size = inferred_policy_size
        if policy_size is None:
            # For v3 models, policy head size is determined by board defaults
            # unless the checkpoint explicitly pins a legacy MAX_N layout.
            if getattr(self.model, "policy_size", None) is not None:
                policy_size = int(getattr(self.model, "policy_size"))

        square_model_classes: Dict[str, Any] = {
            "RingRiftCNN_v2": RingRiftCNN_v2,
            "RingRiftCNN_v2_Lite": RingRiftCNN_v2_Lite,
            "RingRiftCNN_v3": RingRiftCNN_v3,
            "RingRiftCNN_v3_Lite": RingRiftCNN_v3_Lite,
            "RingRiftCNN_v4": RingRiftCNN_v4,
        }
        cls = square_model_classes.get(model_class)
        if cls is None:
            return False

        board_size = getattr(self.model, "board_size", self.board_size)
        try:
            board_size = int(board_size)
        except Exception:
            board_size = self.board_size

        logger.warning(
            "Rebuilding %s to match checkpoint shapes (filters=%s, res_blocks=%s, "
            "history_length=%s, num_players=%s, policy_size=%s)",
            model_class,
            num_filters,
            num_res_blocks,
            history_length,
            num_players,
            policy_size,
        )

        self.model = cls(
            board_size=board_size,
            in_channels=base_in_channels,
            global_features=global_features,
            num_res_blocks=num_res_blocks,
            num_filters=num_filters,
            history_length=history_length,
            policy_size=policy_size,
            num_players=num_players,
        )
        self.model.to(self.device)
        self.model.eval()
        self.history_length = history_length
        return True

    def _load_model_checkpoint(self, model_path: str) -> None:
        """
        Load model checkpoint with version validation.

        Uses the model versioning system when available, falls back to
        direct state_dict loading for legacy checkpoints with explicit
        error handling.
        """
        import os

        allow_fresh = bool(getattr(self.config, "allow_fresh_weights", False))
        if os.environ.get("RINGRIFT_REQUIRE_NEURAL_NET", "").lower() in {"1", "true", "yes"}:
            allow_fresh = False

        try:
            # Try to use versioned loading first
            from ..training.model_versioning import (
                ModelVersionManager,
                VersionMismatchError,
                ChecksumMismatchError,
                LegacyCheckpointError,
            )

            manager = ModelVersionManager(default_device=self.device)

            try:
                # Try strict loading first
                # Use actual model class for version validation
                expected_version = getattr(self.model, "ARCHITECTURE_VERSION", "v2.0.0")
                # Handle torch.compile() wrapped models - extract original class name
                expected_class = self.model.__class__.__name__
                if expected_class == "OptimizedModule":
                    # torch.compile wraps the model - get original class
                    if hasattr(self.model, "_orig_mod"):
                        expected_class = self.model._orig_mod.__class__.__name__
                    elif hasattr(self.model, "module"):
                        expected_class = self.model.module.__class__.__name__
                state_dict, metadata = manager.load_checkpoint(
                    model_path,
                    strict=True,
                    expected_version=expected_version,
                    expected_class=expected_class,
                    verify_checksum=True,
                    device=self.device,
                )
                if isinstance(state_dict, dict):
                    state_dict = _strip_module_prefix(state_dict)

                # If the checkpoint config disagrees with the instantiated
                # model (common when callers build the historical 128-filter
                # variant), rebuild the model to match the checkpoint weights
                # before running strict shape guards.
                self._maybe_rebuild_model_to_match_checkpoint(state_dict, metadata)

                # Guard: reject checkpoints whose declared global_features do not
                # match the current encoder/output shape.
                expected_globals = getattr(self.model, "global_features", None)
                if expected_globals is not None:
                    meta_globals = None
                    if hasattr(metadata, "global_features"):
                        meta_globals = getattr(metadata, "global_features")
                    else:
                        meta_globals = metadata.config.get("global_features")
                    if meta_globals is not None and meta_globals != expected_globals:
                        msg = (
                            "Model checkpoint incompatible with current global_features "
                            f"(checkpoint={meta_globals}, expected={expected_globals})"
                        )
                        if allow_fresh:
                            logger.warning("%s; using fresh weights (allow_fresh_weights=True).", msg)
                            return
                        raise RuntimeError(msg)

                # Guard: reject checkpoints whose value_fc1 weight shape does not
                # match the current architecture (e.g., stale feature count).
                vf1_weight = state_dict.get("value_fc1.weight")
                if vf1_weight is not None and hasattr(self.model, "value_fc1"):
                    expected_in = self.model.value_fc1.in_features
                    actual_in = vf1_weight.shape[1]
                    if actual_in != expected_in:
                        msg = (
                            "Model checkpoint incompatible with current feature shape "
                            f"(value_fc1 in_features: checkpoint={actual_in}, expected={expected_in})"
                        )
                        if allow_fresh:
                            logger.warning("%s; using fresh weights (allow_fresh_weights=True).", msg)
                            return
                        raise RuntimeError(msg)

                # Guard: reject checkpoints whose value head output shape does not
                # match the current model's num_players.
                vf2_weight = state_dict.get("value_fc2.weight")
                if vf2_weight is not None and hasattr(self.model, "value_fc2"):
                    expected_out = int(getattr(self.model.value_fc2, "out_features", 0))
                    actual_out = int(vf2_weight.shape[0])
                    if expected_out and actual_out != expected_out:
                        msg = (
                            "Model checkpoint incompatible with current num_players "
                            f"(value_fc2 out_features: checkpoint={actual_out}, expected={expected_out})"
                        )
                        if allow_fresh:
                            logger.warning("%s; using fresh weights (allow_fresh_weights=True).", msg)
                            return
                        raise RuntimeError(msg)

                # Guard: V3.1 rank distribution head depends on num_players^2.
                rd_weight = state_dict.get("rank_dist_fc2.weight")
                if rd_weight is not None and hasattr(self.model, "rank_dist_fc2"):
                    expected_rd_out = int(getattr(self.model.rank_dist_fc2, "out_features", 0))
                    actual_rd_out = int(rd_weight.shape[0])
                    if expected_rd_out and actual_rd_out != expected_rd_out:
                        msg = (
                            "Model checkpoint incompatible with current num_players "
                            f"(rank_dist_fc2 out_features: checkpoint={actual_rd_out}, expected={expected_rd_out})"
                        )
                        if allow_fresh:
                            logger.warning("%s; using fresh weights (allow_fresh_weights=True).", msg)
                            return
                        raise RuntimeError(msg)

                self.model.load_state_dict(state_dict)
                self.model.eval()
                logger.info(f"Loaded versioned model from {model_path} " f"(version: {metadata.architecture_version})")
                return

            except LegacyCheckpointError:
                # Legacy checkpoint - load with warning
                logger.warning(
                    f"Loading legacy checkpoint without versioning: "
                    f"{model_path}. Consider migrating to versioned format."
                )
                state_dict, _ = manager.load_checkpoint(
                    model_path,
                    strict=False,
                    verify_checksum=False,
                    device=self.device,
                )
                self.model.load_state_dict(state_dict)
                self.model.eval()
                return

            except VersionMismatchError as e:
                # Version mismatch - FAIL EXPLICITLY instead of silent fallback
                logger.error(
                    f"ARCHITECTURE VERSION MISMATCH - Cannot load!\n"
                    f"  Checkpoint: {e.checkpoint_version}\n"
                    f"  Expected: {e.current_version}\n"
                    f"  Path: {model_path}\n"
                    f"  This is a P0 error. The checkpoint is incompatible "
                    f"with the current model architecture."
                )
                raise  # Re-raise to prevent silent fallback

            except ChecksumMismatchError as e:
                # Integrity failure - FAIL EXPLICITLY
                logger.error(
                    f"CHECKPOINT INTEGRITY FAILURE - File may be corrupted!\n"
                    f"  Path: {model_path}\n"
                    f"  Expected checksum: {e.expected[:16]}...\n"
                    f"  Actual checksum: {e.actual[:16]}..."
                )
                raise  # Re-raise to prevent silent fallback

        except ImportError:
            # model_versioning not available, fall back to direct loading
            logger.warning("model_versioning module not available, " "using legacy loading")
            self._load_legacy_checkpoint(model_path)

    def _load_legacy_checkpoint(self, model_path: str) -> None:
        """
        Legacy checkpoint loading with explicit error handling.

        This is used when the versioning module is not available or
        for backwards compatibility with existing code paths.
        """
        try:
            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False,
            )

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    # Assume it's a direct state dict
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            if isinstance(state_dict, dict):
                state_dict = _strip_module_prefix(state_dict)

            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info(f"Loaded legacy model from {model_path}")

        except RuntimeError as e:
            # Architecture mismatch - FAIL EXPLICITLY
            error_msg = str(e)
            if "size mismatch" in error_msg or "Missing key" in error_msg:
                logger.error(
                    f"ARCHITECTURE MISMATCH - Cannot load checkpoint!\n"
                    f"  Path: {model_path}\n"
                    f"  Error: {e}\n"
                    f"  This indicates the checkpoint was saved with a "
                    f"different model architecture. Silent fallback to "
                    f"fresh weights is DISABLED to prevent training bugs."
                )
                raise RuntimeError(
                    f"Architecture mismatch loading {model_path}: {e}. "
                    f"Silent fallback is disabled. Either use a compatible "
                    f"checkpoint or explicitly start with fresh weights."
                ) from e
            else:
                raise

        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def select_move(self, game_state: GameState) -> Optional[Move]:
        """
        Select the best move using neural network evaluation.
        """
        # Ensure model is initialized for this board type (lazy initialization)
        self._ensure_model_initialized(
            game_state.board.type,
            num_players=infer_num_players(game_state),
        )

        # Update history for the current game state
        current_features, _ = self._extract_features(game_state)
        game_id = game_state.id

        if game_id not in self.game_history:
            self.game_history[game_id] = []

        # Append current state to history
        # We only append if it's a new state (simple check: diff from last)
        # Or just append always? select_move is called once per turn.
        # But we might be called multiple times for same state if retrying?
        # Let's assume we append.
        self.game_history[game_id].append(current_features)

        # Keep only needed history (history_length + 1 for current)
        # Actually we need history_length previous states.
        # So we keep history_length + 1 (current) + maybe more?
        # We just need the last few.
        max_hist = self.history_length + 1
        if len(self.game_history[game_id]) > max_hist:
            self.game_history[game_id] = self.game_history[game_id][-max_hist:]

        # Get all valid moves for this AI player via the rules engine
        valid_moves = self.get_valid_moves(game_state)
        if not valid_moves:
            return None

        # Optional exploration/randomisation based on configured randomness
        if self.should_pick_random_move():
            selected = self.get_random_element(valid_moves)
        else:
            # Batch evaluation
            next_states: list[GameState] = []
            moves_list: list[Move] = []

            for move in valid_moves:
                next_states.append(self.rules_engine.apply_move(game_state, move))
                moves_list.append(move)

            # Construct stacked inputs for all next states
            # For each next_state, the history is:
            # [current_state, prev1, prev2, ...] (from self.game_history)
            # So the stack for next_state is:
            # next_state + current_state + prev1 + prev2 ...

            # Get the base history (current + previous)
            # self.game_history[game_id] contains [...prev2, prev1, current]
            # Reverse to get [current, prev1, prev2...]
            base_history = self.game_history[game_id][::-1]

            # Pad if necessary
            while len(base_history) < self.history_length:
                base_history.append(np.zeros_like(current_features))

            # Trim to history_length
            base_history = base_history[: self.history_length]

            # Now construct batch
            batch_stacks: list[np.ndarray] = []
            batch_globals: list[np.ndarray] = []

            for ns in next_states:
                ns_features, ns_globals = self._extract_features(ns)

                # Stack: [ns_features, base_history[0], base_history[1]...]
                stack_list = [ns_features] + base_history
                # Concatenate along channel dim (0)
                stack = np.concatenate(stack_list, axis=0)

                batch_stacks.append(stack)
                batch_globals.append(ns_globals)

            # Convert to tensor
            tensor_input = torch.FloatTensor(np.array(batch_stacks)).to(self.device)
            globals_input = torch.FloatTensor(np.array(batch_globals)).to(self.device)

            # Evaluate batch
            values, _ = self.evaluate_batch(
                next_states,
                tensor_input=tensor_input,
                globals_input=globals_input,
            )

            # Find best move
            if not values or not moves_list:
                # Defensive fallback if evaluation fails
                selected = valid_moves[0] if valid_moves else None
            else:
                best_idx = int(np.argmax(values))
                if best_idx >= len(moves_list):
                    # Defensive fallback if index is out of range
                    best_idx = 0
                selected = moves_list[best_idx]

        return selected

    def evaluate_position(self, game_state: GameState) -> float:
        """
        Evaluate position using neural network.
        """
        # Note: This method doesn't support history injection easily unless
        # we pass it. If called from outside select_move, it might lack
        # history context. We'll assume it uses the stored history for the
        # game_id if available.
        values, _ = self.evaluate_batch([game_state])
        return values[0] if values else 0.0

    def evaluate_batch(
        self,
        game_states: list[GameState],
        tensor_input: Optional[torch.Tensor] = None,
        globals_input: Optional[torch.Tensor] = None,
        value_head: Optional[int] = None,
    ) -> tuple[list[float], np.ndarray]:
        """
        Evaluate a batch of game states.

        All states in a batch must share the same board.type and board.size so
        that the stacked feature tensors have a consistent spatial shape. This
        invariant is enforced at runtime and a ValueError is raised if it is
        violated.
        """
        if not game_states and tensor_input is None:
            # Return empty batch - use default policy size if model not initialized
            policy_size = self.model.policy_size if self.model else POLICY_SIZE_8x8
            empty_policy = np.zeros((0, policy_size), dtype=np.float32)
            return [], empty_policy

        # Enforce homogeneous board geometry within a batch.
        if game_states:
            first_board = game_states[0].board
            first_type = first_board.type
            first_size = first_board.size
            for state in game_states[1:]:
                if state.board.type != first_type or state.board.size != first_size:
                    raise ValueError(
                        "NeuralNetAI.evaluate_batch requires all game_states "
                        "in a batch to share the same board.type and "
                        f"board.size; got {first_type}/{first_size} and "
                        f"{state.board.type}/{state.board.size}."
                    )

            # Ensure model is initialized for this board type (lazy initialization)
            self._ensure_model_initialized(
                first_type,
                num_players=infer_num_players(game_states[0]),
            )

            # Cache the canonical spatial dimension for downstream tools.
            self.board_size = _infer_board_size(first_board)

        if tensor_input is None:
            # Fallback: construct inputs from states, using stored history.
            batch_stacks: list[np.ndarray] = []
            batch_globals: list[np.ndarray] = []

            for state in game_states:
                features, globals_vec = self._extract_features(state)

                # Try to get history
                game_id = state.id
                history: list[np.ndarray] = []
                if game_id in self.game_history:
                    # History list is [oldest, ..., newest]; we want
                    # newest-first for stacking.
                    hist_list = self.game_history[game_id][::-1]
                    history = hist_list[: self.history_length]

                # Pad history
                while len(history) < self.history_length:
                    history.append(np.zeros_like(features))

                # Stack: [current, hist1, hist2...]
                stack_list = [features] + history
                stack = np.concatenate(stack_list, axis=0)

                batch_stacks.append(stack)
                batch_globals.append(globals_vec)

            tensor_input = torch.FloatTensor(np.array(batch_stacks)).to(self.device)
            globals_input = torch.FloatTensor(np.array(batch_globals)).to(self.device)

        assert globals_input is not None

        use_autocast = False
        device = self.device
        if isinstance(device, str):
            use_autocast = device.startswith("cuda")
        else:
            use_autocast = getattr(device, "type", "") == "cuda"

        with torch.no_grad():
            assert self.model is not None
            if use_autocast:
                with torch.amp.autocast('cuda'):
                    out = self.model(tensor_input, globals_input)
            else:
                out = self.model(tensor_input, globals_input)
            # V3 models return (values, policy_logits, rank_dist). Keep the
            # rank distribution for training-only use and ignore it here.
            if isinstance(out, tuple) and len(out) == 3:
                values, policy_logits, _rank_dist = out
            else:
                values, policy_logits = out

            # Apply softmax to logits to get probabilities for MCTS / Descent.
            policy_probs = torch.softmax(policy_logits, dim=1)

        # NOTE: Many RingRiftCNN models output a multi-value head by default.
        # The NeuralNetAI wrapper (and most search AIs) currently consume a
        # *single scalar* value per state. By default we return value_head=0,
        # but callers may request a specific column via value_head (e.g. for
        # multi-player value heads trained to predict per-player utilities).
        #
        # IMPORTANT: The value_head parameter should be set to the AI's player
        # number (0-indexed), NOT the current_player in the state. During search,
        # we evaluate many states including opponent turns, but we always want
        # the value from our own perspective.
        values_np = values.detach().cpu().numpy()
        if values_np.ndim == 2:
            head = 0
            if value_head is not None:
                try:
                    head = int(value_head)
                except Exception:
                    head = 0
            if head < 0 or head >= values_np.shape[1]:
                head = 0
            scalar_values = values_np[:, head]
        else:
            scalar_values = values_np.reshape(values_np.shape[0])

        return (scalar_values.astype(np.float32).tolist(), policy_probs.cpu().numpy())

    def encode_state_for_model(
        self,
        game_state: GameState,
        history_frames: list[np.ndarray],
        history_length: int = 3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (stacked_features[C,H,W], globals[20]) compatible with
        RingRiftCNN.
        history_frames: most recent feature frames for this game_id,
        newest last.
        """
        features, globals_vec = self._extract_features(game_state)
        # newest-first
        hist = history_frames[::-1][:history_length]
        while len(hist) < history_length:
            hist.append(np.zeros_like(features))
        stack = np.concatenate([features] + hist, axis=0)
        return stack, globals_vec

    def encode_move(
        self,
        move: Move,
        board_context: Union[BoardState, GameState, int],
    ) -> int:
        """
        Encode a move into a policy index.

        This method supports two policy layouts:

        1. Board-specific policy heads (preferred):
           - SQUARE8: policy_size=7000
           - SQUARE19: policy_size=67000
           - HEXAGONAL: policy_size=91876

           When the loaded checkpoint's ``model.policy_size`` matches the
           board-specific size for the current board, we use
           :func:`encode_move_for_board` to generate indices that are
           guaranteed to be in-range.

        2. Legacy MAX_N layout:
           Some older checkpoints (e.g. `ringrift_v4_*`) were trained against a
           fixed MAX_N=19 layout. When the checkpoint's policy head size does
           not match the board-specific layout, we fall back to the legacy
           encoding below.

        For backward compatibility, board_context may be an integer board_size
        (e.g. 8 or 19). In that case we treat coordinates as already expressed
        in the canonical 2D frame for a square board and bypass BoardState-
        based mapping.
        """
        board: Optional[BoardState] = None

        # Normalise context to a BoardState when possible.
        if isinstance(board_context, GameState):
            board = board_context.board
        elif isinstance(board_context, BoardState):
            board = board_context
        elif isinstance(board_context, int):
            # Legacy callers (tests, older tooling) pass a raw board_size.
            # We do not need the size here, because we only ever check against
            # the canonical MAX_N=19 grid.
            board = None
        else:
            raise TypeError(f"Unsupported board_context type for encode_move: " f"{type(board_context)!r}")

        # If we have an initialized model and a concrete board, prefer the
        # board-specific encoder when the checkpoint's policy head matches.
        model_policy_size = 0
        board_policy_size = 0
        if board is not None and self.model is not None:
            try:
                model_policy_size = int(getattr(self.model, "policy_size"))
            except Exception:
                model_policy_size = 0

            board_policy_size = int(get_policy_size_for_board(board.type))
            if model_policy_size and model_policy_size == board_policy_size:
                idx = encode_move_for_board(move, board)
                if 0 <= idx < model_policy_size:
                    return idx
                return INVALID_MOVE_INDEX

        # Pre-compute layout constants from MAX_N to avoid hard-coded offsets.
        placement_span = 3 * MAX_N * MAX_N  # 0..1082
        movement_base = placement_span  # 1083
        movement_span = MAX_N * MAX_N * (8 * (MAX_N - 1))
        line_base = movement_base + movement_span  # 53067
        line_span = MAX_N * MAX_N * 4
        territory_base = line_base + line_span  # 54511
        skip_index = territory_base + MAX_N * MAX_N  # 54872
        swap_sides_index = skip_index + 1  # 54873
        move_type = move.type.value if hasattr(move.type, "value") else str(move.type)
        effective_policy_size = model_policy_size or board_policy_size or 0

        # Placement: 0..1082 (3 * 19 * 19)
        if move_type == "place_ring":
            if board is not None:
                cx, cy = _to_canonical_xy(board, move.to)
            else:
                # Legacy integer board_size path (square boards).
                cx, cy = move.to.x, move.to.y

            # Guard against boards larger than MAX_N×MAX_N.
            if not (0 <= cx < MAX_N and 0 <= cy < MAX_N):
                return INVALID_MOVE_INDEX

            # Index = (y * MAX_N + x) * 3 + (count - 1)
            pos_idx = cy * MAX_N + cx
            count_idx = (move.placement_count or 1) - 1
            return pos_idx * 3 + count_idx

        # Movement: 1083..53066
        # Recovery_slide is encoded as movement since it has from/to positions
        if move_type in [
            "move_stack",
            "move_ring",
            "overtaking_capture",
            "chain_capture",
            "continue_capture_segment",
            "recovery_slide",  # RR-CANON-R110–R115: marker slide to adjacent cell
        ]:
            # Base = 1083 (3 * 19 * 19)
            # Index = Base + (from_y * MAX_N + from_x) * (8 * (MAX_N-1)) +
            #         (dir_idx * (MAX_N-1)) + (dist - 1)
            if not move.from_pos:
                return INVALID_MOVE_INDEX

            if board is not None:
                cfx, cfy = _to_canonical_xy(board, move.from_pos)
                ctx, cty = _to_canonical_xy(board, move.to)
            else:
                # Legacy integer board_size path (square boards).
                cfx, cfy = move.from_pos.x, move.from_pos.y
                ctx, cty = move.to.x, move.to.y

            # If either endpoint lies outside the canonical 19×19 grid, this
            # move cannot be represented in the fixed policy head.
            if not (0 <= cfx < MAX_N and 0 <= cfy < MAX_N and 0 <= ctx < MAX_N and 0 <= cty < MAX_N):
                return INVALID_MOVE_INDEX

            from_idx = cfy * MAX_N + cfx

            dx = ctx - cfx
            dy = cty - cfy

            # For square boards we use Chebyshev distance. For hex boards, the
            # canonical 2D embedding is a translation of axial coordinates, so
            # dx/dy are preserved and we can continue to use Chebyshev here as
            # long as encode/decode remain symmetric.
            dist = max(abs(dx), abs(dy))
            if dist == 0:
                return INVALID_MOVE_INDEX

            dir_x = dx // dist if dist > 0 else 0
            dir_y = dy // dist if dist > 0 else 0

            dirs = [
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
            ]
            try:
                dir_idx = dirs.index((dir_x, dir_y))
            except ValueError:
                # Direction not representable in our 8-direction scheme.
                return INVALID_MOVE_INDEX

            max_dist = MAX_N - 1
            return movement_base + from_idx * (8 * max_dist) + dir_idx * max_dist + (dist - 1)

        # Line: 53067..54510
        if move_type in ("line_formation", "process_line"):
            line_pos = _line_anchor_position(move)
            if line_pos is None:
                return INVALID_MOVE_INDEX
            if board is not None:
                cx, cy = _to_canonical_xy(board, line_pos)
            else:
                cx, cy = line_pos.x, line_pos.y
            if not (0 <= cx < MAX_N and 0 <= cy < MAX_N):
                return INVALID_MOVE_INDEX
            pos_idx = cy * MAX_N + cx
            # We currently ignore direction and always use dir_idx = 0, but
            # keep the 4-way slot layout for backward compatibility.
            return line_base + pos_idx * 4

        # Territory: 54511..54871
        if move_type in ("territory_claim", "eliminate_rings_from_stack"):
            if move.to is None:
                return INVALID_MOVE_INDEX
            if board is not None:
                cx, cy = _to_canonical_xy(board, move.to)
            else:
                cx, cy = move.to.x, move.to.y
            if not (0 <= cx < MAX_N and 0 <= cy < MAX_N):
                return INVALID_MOVE_INDEX
            pos_idx = cy * MAX_N + cx
            return territory_base + pos_idx

        # Skip placement: single terminal index
        if move_type == "skip_placement":
            return skip_index

        # Swap-sides (pie rule) decision. The decode_move implementation
        # already treats the index directly above skip_index as a canonical
        # SWAP_SIDES action; wiring encode_move to emit the same index ensures
        # that recorded swap_sides moves are represented in the policy head
        # and can be learned from training data.
        if move_type == "swap_sides":
            return swap_sides_index

        # Skip recovery (RR-CANON-R112): player elects not to perform recovery action
        skip_recovery_index = swap_sides_index + 1  # 54874 (sq19) / 4098 (sq8)
        if move_type == "skip_recovery":
            return skip_recovery_index

        # Choice moves: line and territory decision options
        # Line choices: 4 slots (options 0-3, typically option 1 = partial, 2 = full)
        line_choice_base = skip_recovery_index + 1  # 54875 (sq19) / 4099 (sq8)

        if move_type in ("choose_line_reward", "choose_line_option"):
            # Line choice uses placement_count to indicate option (1-based)
            option = (move.placement_count or 1) - 1  # Convert to 0-indexed
            option = max(0, min(3, option))  # Clamp to valid range
            return line_choice_base + option

        # Territory choice encoding: uniquely identify by (position, size, player)
        # This handles cases where canonical position alone is non-unique
        # (e.g., overlapping regions with different borders)
        #
        # Layout: base + pos_idx * (SIZE_BUCKETS * MAX_PLAYERS) + size_bucket * MAX_PLAYERS + player_idx
        # With SIZE_BUCKETS=8, MAX_PLAYERS=4: 361 * 8 * 4 = 11,552 slots
        territory_choice_base = line_choice_base + 4  # 54878
        TERRITORY_SIZE_BUCKETS = 8
        TERRITORY_MAX_PLAYERS = 4

        if move_type in ("choose_territory_option", "process_territory_region"):
            # Extract region information from the move
            canonical_pos = move.to  # Default to move.to
            region_size = 1
            controlling_player = move.player

            if move.disconnected_regions:
                regions = list(move.disconnected_regions)
                if regions:
                    region = regions[0]
                    # Get region spaces
                    if hasattr(region, "spaces") and region.spaces:
                        spaces = list(region.spaces)
                        region_size = len(spaces)
                        # Find canonical (lexicographically smallest) position
                        canonical_pos = min(spaces, key=lambda p: (p.y, p.x))
                    # Get controlling player (who owns the border)
                    if hasattr(region, "controlling_player"):
                        controlling_player = region.controlling_player

            # Convert position to canonical coordinates
            if canonical_pos is None:
                return INVALID_MOVE_INDEX
            if board is not None:
                cx, cy = _to_canonical_xy(board, canonical_pos)
            else:
                cx, cy = canonical_pos.x, canonical_pos.y

            if not (0 <= cx < MAX_N and 0 <= cy < MAX_N):
                return INVALID_MOVE_INDEX

            pos_idx = cy * MAX_N + cx
            size_bucket = min(region_size - 1, TERRITORY_SIZE_BUCKETS - 1)  # 0-7
            player_idx = (controlling_player - 1) % TERRITORY_MAX_PLAYERS  # 0-3

            return (
                territory_choice_base
                + pos_idx * (TERRITORY_SIZE_BUCKETS * TERRITORY_MAX_PLAYERS)
                + size_bucket * TERRITORY_MAX_PLAYERS
                + player_idx
            )

        territory_choice_span = MAX_N * MAX_N * TERRITORY_SIZE_BUCKETS * TERRITORY_MAX_PLAYERS
        extra_special_base = territory_choice_base + territory_choice_span
        extra_special_indices = {
            "no_placement_action": extra_special_base,
            "no_movement_action": extra_special_base + 1,
            "skip_capture": extra_special_base + 2,
            "no_line_action": extra_special_base + 3,
            "no_territory_action": extra_special_base + 4,
            "skip_territory_processing": extra_special_base + 5,
            "forced_elimination": extra_special_base + 6,
        }
        extra_idx = extra_special_indices.get(move_type)
        if extra_idx is not None:
            if effective_policy_size and extra_idx < effective_policy_size:
                return extra_idx
            return INVALID_MOVE_INDEX

        return INVALID_MOVE_INDEX

    def decode_move(self, index: int, game_state: GameState) -> Optional[Move]:
        """
        Decode a policy index into a Move.

        The inverse of encode_move, using the same MAX_N × MAX_N canonical
        grid. If the decoded coordinates fall outside the legal geometry of
        game_state.board, this returns None.
        """
        board = game_state.board

        # Pre-compute layout constants from MAX_N to align with encode_move.
        placement_span = 3 * MAX_N * MAX_N  # 0..1082
        movement_base = placement_span  # 1083
        movement_span = MAX_N * MAX_N * (8 * (MAX_N - 1))
        line_base = movement_base + movement_span  # 53067
        line_span = MAX_N * MAX_N * 4
        territory_base = line_base + line_span  # 54511
        skip_index = territory_base + MAX_N * MAX_N  # 54872

        if index < 0 or index >= self.model.policy_size:
            return None

        # Placement
        if index < placement_span:
            count_idx = index % 3
            pos_idx = index // 3
            cy = pos_idx // MAX_N
            cx = pos_idx % MAX_N

            pos = _from_canonical_xy(board, cx, cy)
            if pos is None or not BoardGeometry.is_within_bounds(pos, board.type, board.size):
                return None

            to_payload: dict[str, int] = {"x": pos.x, "y": pos.y}
            if pos.z is not None:
                to_payload["z"] = pos.z

            move_data = {
                "id": "decoded",
                "type": "place_ring",
                "player": game_state.current_player,
                "to": to_payload,
                "placementCount": count_idx + 1,
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)

        # Movement
        if index < line_base:
            max_dist = MAX_N - 1
            offset = index - movement_base

            dist_idx = offset % max_dist
            dist = dist_idx + 1
            offset //= max_dist

            dir_idx = offset % 8
            offset //= 8

            from_idx = offset
            cfy = from_idx // MAX_N
            cfx = from_idx % MAX_N

            from_pos = _from_canonical_xy(board, cfx, cfy)
            if from_pos is None or not BoardGeometry.is_within_bounds(from_pos, board.type, board.size):
                return None

            dirs = [
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
            ]
            dx, dy = dirs[dir_idx]

            ctx = cfx + dx * dist
            cty = cfy + dy * dist
            to_pos = _from_canonical_xy(board, ctx, cty)
            if to_pos is None or not BoardGeometry.is_within_bounds(to_pos, board.type, board.size):
                return None

            from_payload: dict[str, int] = {"x": from_pos.x, "y": from_pos.y}
            if from_pos.z is not None:
                from_payload["z"] = from_pos.z

            to_payload: dict[str, int] = {"x": to_pos.x, "y": to_pos.y}
            if to_pos.z is not None:
                to_payload["z"] = to_pos.z

            move_data = {
                "id": "decoded",
                "type": "move_stack",
                "player": game_state.current_player,
                "from": from_payload,
                "to": to_payload,
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)

        # Line formation
        if line_base <= index < territory_base:
            offset = index - line_base
            pos_idx = offset // 4  # Ignore dir_idx for now
            cy = pos_idx // MAX_N
            cx = pos_idx % MAX_N

            pos = _from_canonical_xy(board, cx, cy)
            if pos is None or not BoardGeometry.is_within_bounds(pos, board.type, board.size):
                return None

            to_payload: dict[str, int] = {"x": pos.x, "y": pos.y}
            if pos.z is not None:
                to_payload["z"] = pos.z

            move_data = {
                "id": "decoded",
                "type": "line_formation",
                "player": game_state.current_player,
                "to": to_payload,
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)

        # Territory claim
        if index < skip_index:
            pos_idx = index - territory_base
            cy = pos_idx // MAX_N
            cx = pos_idx % MAX_N

            pos = _from_canonical_xy(board, cx, cy)
            if pos is None or not BoardGeometry.is_within_bounds(pos, board.type, board.size):
                return None

            to_payload: dict[str, int] = {"x": pos.x, "y": pos.y}
            if pos.z is not None:
                to_payload["z"] = pos.z

            move_data = {
                "id": "decoded",
                "type": "territory_claim",
                "player": game_state.current_player,
                "to": to_payload,
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)

        # Skip placement
        if index == skip_index:
            move_data = {
                "id": "decoded",
                "type": "skip_placement",
                "player": game_state.current_player,
                "to": {"x": 0, "y": 0},
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)

        # Swap sides (pie rule)
        swap_sides_index = skip_index + 1
        if index == swap_sides_index:
            move_data = {
                "id": "decoded",
                "type": "swap_sides",
                "player": game_state.current_player,
                "to": {"x": 0, "y": 0},
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)

        # Skip recovery (RR-CANON-R112): player elects not to perform recovery action
        skip_recovery_index = swap_sides_index + 1
        if index == skip_recovery_index:
            move_data = {
                "id": "decoded",
                "type": "skip_recovery",
                "player": game_state.current_player,
                "to": {"x": 0, "y": 0},
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)

        # Choice moves: line and territory options
        line_choice_base = skip_recovery_index + 1  # 54875
        territory_choice_base = line_choice_base + 4  # 54879

        # Line choice (indices 54874-54877)
        if line_choice_base <= index < territory_choice_base:
            option = index - line_choice_base + 1  # Convert to 1-indexed
            move_data = {
                "id": "decoded",
                "type": "choose_line_option",
                "player": game_state.current_player,
                "to": {"x": 0, "y": 0},
                "placementCount": option,
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)

        # Territory choice: (position, size_bucket, player) encoding
        # Layout: base + pos_idx * (SIZE_BUCKETS * MAX_PLAYERS) + size_bucket * MAX_PLAYERS + player_idx
        TERRITORY_SIZE_BUCKETS = 8
        TERRITORY_MAX_PLAYERS = 4
        territory_choice_span = MAX_N * MAX_N * TERRITORY_SIZE_BUCKETS * TERRITORY_MAX_PLAYERS

        if territory_choice_base <= index < territory_choice_base + territory_choice_span:
            offset = index - territory_choice_base
            _player_idx = offset % TERRITORY_MAX_PLAYERS  # noqa: F841 - reserved for future use
            offset //= TERRITORY_MAX_PLAYERS
            size_bucket = offset % TERRITORY_SIZE_BUCKETS
            pos_idx = offset // TERRITORY_SIZE_BUCKETS

            cy = pos_idx // MAX_N
            cx = pos_idx % MAX_N

            pos = _from_canonical_xy(board, cx, cy)
            if pos is None:
                return None

            to_payload: dict[str, int] = {"x": pos.x, "y": pos.y}
            if pos.z is not None:
                to_payload["z"] = pos.z

            move_data = {
                "id": "decoded",
                "type": "choose_territory_option",
                "player": game_state.current_player,
                "to": to_payload,
                # Size and player info embedded in the index, used for matching
                "placementCount": size_bucket + 1,  # 1-indexed size bucket
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)

        extra_special_base = territory_choice_base + territory_choice_span
        extra_special_span = 7
        if extra_special_base <= index < extra_special_base + extra_special_span:
            action_types = (
                "no_placement_action",
                "no_movement_action",
                "skip_capture",
                "no_line_action",
                "no_territory_action",
                "skip_territory_processing",
                "forced_elimination",
            )
            offset = index - extra_special_base
            if 0 <= offset < len(action_types):
                move_data = {
                    "id": "decoded",
                    "type": action_types[offset],
                    "player": game_state.current_player,
                    "timestamp": datetime.now(),
                    "thinkTime": 0,
                    "moveNumber": 0,
                }
                return Move(**move_data)

        return None

    def _extract_features(
        self,
        game_state: GameState,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert game state to feature tensor for CNN and global features.

        Returns:
            (board_features, global_features)

        The board_features tensor has shape:
        - Hex boards (with _hex_encoder): (16, board_size, board_size)
        - Square boards: (14, board_size, board_size)

        Board size is derived from the logical board via _infer_board_size so
        that this encoder works for 8×8, 19×19, and hexagonal boards.
        """
        # Use hex encoder for hex boards (produces 16 channels matching V3 models)
        if self._hex_encoder is not None:
            board_features, global_features = self._hex_encoder.encode_state(game_state)
            # Update board_size hint for external callers
            self.board_size = self._hex_encoder.board_size
            return board_features, global_features

        board = game_state.board
        # Derive spatial dimension from logical board geometry and keep a hint
        # for components (e.g. training augmentation) that still need to know
        # the current spatial dimension.
        board_size = _infer_board_size(board)
        self.board_size = board_size

        # Board features: 14 channels
        # 0: My stacks (height normalized)
        # 1: Opponent stacks (height normalized)
        # 2: My markers
        # 3: Opponent markers
        # 4: My collapsed spaces
        # 5: Opponent collapsed spaces
        # 6: My liberties
        # 7: Opponent liberties
        # 8: My line potential
        # 9: Opponent line potential
        # 10: Cap presence - current player
        # 11: Cap presence - opponent
        # 12: Valid board position mask
        # 13: Reserved (zeros)
        features = np.zeros((14, board_size, board_size), dtype=np.float32)

        is_hex = board.type == BoardType.HEXAGONAL

        # --- Stacks: channels 0/1 ---
        for pos_key, stack in board.stacks.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            cx, cy = _to_canonical_xy(board, pos)
            if not (0 <= cx < board_size and 0 <= cy < board_size):
                # Robust to any stray off-board keys.
                continue

            val = min(stack.stack_height / 5.0, 1.0)  # Normalize height
            if stack.controlling_player == game_state.current_player:
                features[0, cx, cy] = val
            else:
                features[1, cx, cy] = val

        # --- Markers: channels 2/3 ---
        for pos_key, marker in board.markers.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            cx, cy = _to_canonical_xy(board, pos)
            if not (0 <= cx < board_size and 0 <= cy < board_size):
                continue

            if marker.player == game_state.current_player:
                features[2, cx, cy] = 1.0
            else:
                features[3, cx, cy] = 1.0

        # --- Collapsed spaces: channels 4/5 ---
        for pos_key, owner in board.collapsed_spaces.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            cx, cy = _to_canonical_xy(board, pos)
            if not (0 <= cx < board_size and 0 <= cy < board_size):
                continue

            if owner == game_state.current_player:
                features[4, cx, cy] = 1.0
            else:
                features[5, cx, cy] = 1.0

        # --- Liberties: channels 6/7 ---
        # Simple approximation based on adjacency; uses BoardGeometry so that
        # hex and square boards share the same logic.
        for pos_key, stack in board.stacks.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            cx, cy = _to_canonical_xy(board, pos)
            if not (0 <= cx < board_size and 0 <= cy < board_size):
                continue

            neighbors = BoardGeometry.get_adjacent_positions(
                pos,
                board.type,
                board.size,
            )
            liberties = 0
            for npos in neighbors:
                if not BoardGeometry.is_within_bounds(
                    npos,
                    board.type,
                    board.size,
                ):
                    continue
                n_key = npos.to_key()
                if n_key in board.stacks or n_key in board.collapsed_spaces:
                    continue
                liberties += 1

            max_libs = 6.0 if is_hex else 8.0
            val = min(liberties / max_libs, 1.0)
            if stack.controlling_player == game_state.current_player:
                features[6, cx, cy] = val
            else:
                features[7, cx, cy] = val

        # --- Line potential: channels 8/9 ---
        # Simplified: markers with neighbours of same colour.
        for pos_key, marker in board.markers.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            cx, cy = _to_canonical_xy(board, pos)
            if not (0 <= cx < board_size and 0 <= cy < board_size):
                continue

            neighbors = BoardGeometry.get_adjacent_positions(
                pos,
                board.type,
                board.size,
            )
            neighbor_count = 0
            for npos in neighbors:
                if not BoardGeometry.is_within_bounds(
                    npos,
                    board.type,
                    board.size,
                ):
                    continue
                n_key = npos.to_key()
                neighbor_marker = board.markers.get(n_key)
                if neighbor_marker is not None and neighbor_marker.player == marker.player:
                    neighbor_count += 1

            max_neighbors = 6.0 if is_hex else 8.0
            val = min(neighbor_count / (max_neighbors / 2.0), 1.0)
            if marker.player == game_state.current_player:
                features[8, cx, cy] = val
            else:
                features[9, cx, cy] = val

        # --- Channels 10-13: Extended features ---
        # Channels 10/11: Reserved for future cap features (zeros for now)
        # Channel 12: Valid board position mask (important for hex)
        # Channel 13: Reserved (zeros)

        # For hex boards, mark valid positions (not all grid cells are playable)
        if is_hex:
            # Use the board's valid positions from stacks, markers, collapsed_spaces
            # or compute from board geometry
            for pos_key in board.stacks.keys():
                try:
                    pos = _pos_from_key(pos_key)
                    cx, cy = _to_canonical_xy(board, pos)
                    if 0 <= cx < board_size and 0 <= cy < board_size:
                        features[12, cx, cy] = 1.0
                except ValueError:
                    continue
            for pos_key in board.markers.keys():
                try:
                    pos = _pos_from_key(pos_key)
                    cx, cy = _to_canonical_xy(board, pos)
                    if 0 <= cx < board_size and 0 <= cy < board_size:
                        features[12, cx, cy] = 1.0
                except ValueError:
                    continue
            for pos_key in board.collapsed_spaces.keys():
                try:
                    pos = _pos_from_key(pos_key)
                    cx, cy = _to_canonical_xy(board, pos)
                    if 0 <= cx < board_size and 0 <= cy < board_size:
                        features[12, cx, cy] = 1.0
                except ValueError:
                    continue
        else:
            # Square boards: all positions are valid
            features[12, :, :] = 1.0

        # --- Global features: 20 dims ---
        # Phase (5), Rings in hand (2), Eliminated rings (2), Turn (1),
        # Extras (2: chain_capture, forced_elimination), Reserved (8)
        # Hex network model was trained with global_features=20 (value_fc1 expects 1+20=21 inputs)
        globals = np.zeros(20, dtype=np.float32)

        # Phase one-hot
        phases = [
            "ring_placement",
            "movement",
            "capture",
            "line_processing",
            "territory_processing",
        ]
        try:
            phase_idx = phases.index(game_state.current_phase.value)
            globals[phase_idx] = 1.0
        except ValueError:
            pass
        # Extra phase flags (reserved slots) for chain capture / forced elimination.
        globals[10] = 1.0 if game_state.current_phase == GamePhase.CHAIN_CAPTURE else 0.0
        globals[11] = 1.0 if game_state.current_phase == GamePhase.FORCED_ELIMINATION else 0.0

        # Rings info (current-player perspective, plus a single "threat opponent"
        # for multi-player Paranoid reductions).
        my_player = next(
            (
                p
                for p in game_state.players
                if p.player_number == game_state.current_player
            ),
            None,
        )

        opp_player = None
        num_players = infer_num_players(game_state)
        if num_players <= 2:
            opp_player = next(
                (
                    p
                    for p in game_state.players
                    if p.player_number != game_state.current_player
                ),
                None,
            )
        else:
            threat_pid = select_threat_opponent(
                game_state, game_state.current_player
            )
            if threat_pid is not None:
                opp_player = next(
                    (
                        p
                        for p in game_state.players
                        if p.player_number == threat_pid
                    ),
                    None,
                )
            if opp_player is None:
                # Defensive fallback: pick any non-current opponent.
                opp_player = next(
                    (
                        p
                        for p in game_state.players
                        if p.player_number != game_state.current_player
                    ),
                    None,
                )

        ring_norm = float(infer_rings_per_player(game_state))
        if my_player:
            globals[5] = my_player.rings_in_hand / ring_norm
            globals[7] = my_player.eliminated_rings / ring_norm

        if opp_player:
            globals[6] = opp_player.rings_in_hand / ring_norm
            globals[8] = opp_player.eliminated_rings / ring_norm

        # Is it my turn? (always yes for current_player perspective)
        globals[9] = 1.0

        return features, globals

    def _extract_phase_chain_planes(
        self,
        game_state: GameState,
        board_size: int,
    ) -> np.ndarray:
        """
        Extract V3 spatial encoding planes for phase and chain-capture state.

        Returns 8 planes:
            0-5: Phase one-hot (broadcast across all cells)
                 0: ring_placement, 1: movement, 2: capture,
                 3: chain_capture, 4: line_processing, 5: territory_processing
            6: Chain-capture start position (binary mask)
            7: Chain-capture current position (binary mask)

        These planes provide explicit spatial context for the policy heads,
        helping the network understand which actions are contextually relevant.
        """
        planes = np.zeros((8, board_size, board_size), dtype=np.float32)
        board = game_state.board

        # Phase one-hot encoding (channels 0-5)
        phase_map = {
            GamePhase.RING_PLACEMENT: 0,
            GamePhase.MOVEMENT: 1,
            GamePhase.CAPTURE: 2,
            GamePhase.CHAIN_CAPTURE: 3,
            GamePhase.LINE_PROCESSING: 4,
            GamePhase.TERRITORY_PROCESSING: 5,
        }
        # Handle forced_elimination by mapping to territory_processing
        phase_idx = phase_map.get(game_state.current_phase, 5)
        planes[phase_idx, :, :] = 1.0  # Broadcast across spatial dims

        # Chain-capture position encoding (channels 6-7)
        if game_state.chain_capture_state is not None:
            ccs = game_state.chain_capture_state
            # Start position (channel 6)
            try:
                start_cx, start_cy = _to_canonical_xy(board, ccs.start_position)
                if 0 <= start_cx < board_size and 0 <= start_cy < board_size:
                    planes[6, start_cy, start_cx] = 1.0
            except (ValueError, AttributeError):
                pass

            # Current position (channel 7)
            try:
                cur_cx, cur_cy = _to_canonical_xy(board, ccs.current_position)
                if 0 <= cur_cx < board_size and 0 <= cur_cy < board_size:
                    planes[7, cur_cy, cur_cx] = 1.0
            except (ValueError, AttributeError):
                pass

        return planes

    def encode_state_for_model_v3(
        self,
        game_state: GameState,
        history_frames: list[np.ndarray],
        history_length: int = 3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        V3 state encoding with spatial phase and chain-capture planes.

        Returns (stacked_features[64, H, W], globals[20]) for V3 models.

        Channel layout (64 total):
            0-9:   Current frame base features (10 channels)
            10-19: History frame 1 (most recent)
            20-29: History frame 2
            30-39: History frame 3
            40-49: History frame 4 (oldest)
            50-55: Reserved for future (zeros)
            56-63: Phase/chain-capture planes (current state only)

        Actually we use 56 (14 base × 4 frames) + 8 = 64:
            0-55:  14 base × 4 history frames
            56-63: Phase/chain-capture planes
        """
        features, globals_vec = self._extract_features(game_state)
        board_size = features.shape[1]

        # Build history stack (14 base × 4 frames = 56 channels)
        # newest-first for history frames
        hist = history_frames[::-1][:history_length]
        while len(hist) < history_length:
            hist.append(np.zeros_like(features))
        base_stack = np.concatenate([features] + hist, axis=0)  # [40, H, W] (10 × 4)

        # For V3, we need 14 base channels (not 10). Check and pad if needed.
        expected_base = 14 * (history_length + 1)  # 14 × 4 = 56
        actual_channels = base_stack.shape[0]

        if actual_channels < expected_base:
            # Pad with zeros to reach 56 channels
            padding = np.zeros((expected_base - actual_channels, board_size, board_size), dtype=np.float32)
            base_stack = np.concatenate([base_stack, padding], axis=0)

        # Extract phase/chain planes (8 channels for current state)
        phase_chain_planes = self._extract_phase_chain_planes(game_state, board_size)

        # Concatenate: 56 base + 8 phase/chain = 64 total
        full_stack = np.concatenate([base_stack, phase_chain_planes], axis=0)

        return full_stack, globals_vec

    # _evaluate_move_with_net is deprecated


class ActionEncoderHex:
    """Hex-only action encoder for the canonical N=12 board.

    The concrete layout matches the design in AI_ARCHITECTURE.md:

      * Spatial frame: 25×25 canonical hex bounding box.
      * Placements (0 .. HEX_PLACEMENT_SPAN-1):
          index = (cy * 25 + cx) * 3 + (count - 1)
        where count ∈ {1,2,3} is the number of rings placed.

      * Movement / capture (HEX_MOVEMENT_BASE .. HEX_SPECIAL_BASE-1):
          index = HEX_MOVEMENT_BASE
                  + from_idx * (6 * HEX_MAX_DIST)
                  + dir_idx * HEX_MAX_DIST
                  + (dist - 1)
        where from_idx = from_cy * 25 + from_cx, dir_idx ∈ [0,6),
        dist ∈ [1,HEX_MAX_DIST]. This shared layout is used for MOVE_STACK,
        OVERTAKING_CAPTURE, and CONTINUE_CAPTURE_SEGMENT.

      * Special (HEX_SPECIAL_BASE):
          SKIP_PLACEMENT sentinel.

    Any decoded index that maps to a canonical cell outside the true hex
    (469-cell) region is treated as invalid and returns None.
    """

    def __init__(
        self,
        board_size: int = HEX_BOARD_SIZE,
        policy_size: int = 0,
    ) -> None:
        # Spatial dimension of the hex bounding box (2N+1 for side N).
        self.board_size = board_size

        # Compute layout constants based on board_size
        self.max_dist = board_size - 1  # max distance on hex board
        self.placement_span = board_size * board_size * 3
        self.movement_base = self.placement_span
        self.movement_span = board_size * board_size * NUM_HEX_DIRS * self.max_dist
        self.special_base = self.movement_base + self.movement_span

        # Hex-specific action space dimension.
        self.policy_size = policy_size or (self.special_base + 1)

    def _encode_canonical_xy(
        self,
        board: BoardState,
        pos: Position,
    ) -> Optional[tuple[int, int]]:
        """Return (cx, cy) in [0, board_size)×[0, board_size) or None if off-grid."""
        cx, cy = _to_canonical_xy(board, pos)
        if not (0 <= cx < self.board_size and 0 <= cy < self.board_size):
            return None
        return cx, cy

    def encode_move(self, move: Move, board: BoardState) -> int:
        """Map a hex move into a [0, policy_size) index.

        This encoder supports both HEXAGONAL (radius-12) and HEX8 (radius-4)
        board types. The board size must match the encoder's configured
        board_size. For non-hex geometries or size mismatches, the move is
        treated as unrepresentable and INVALID_MOVE_INDEX is returned.
        """
        if board.type not in (BoardType.HEXAGONAL, BoardType.HEX8):
            return INVALID_MOVE_INDEX

        # Verify board size matches encoder configuration
        if _infer_board_size(board) != self.board_size:
            return INVALID_MOVE_INDEX

        # --- Placements ---
        if move.type == MoveType.PLACE_RING:
            canon = self._encode_canonical_xy(board, move.to)
            if canon is None:
                return INVALID_MOVE_INDEX
            cx, cy = canon

            pos_idx = cy * self.board_size + cx
            count = move.placement_count or 1
            if count < 1 or count > 3:
                return INVALID_MOVE_INDEX

            return pos_idx * 3 + (count - 1)

        # --- Movement / capture / recovery ---
        if move.type in (
            MoveType.MOVE_STACK,
            MoveType.OVERTAKING_CAPTURE,
            MoveType.CONTINUE_CAPTURE_SEGMENT,
            MoveType.RECOVERY_SLIDE,  # RR-CANON-R110–R115: marker slide
        ):
            if move.from_pos is None:
                return INVALID_MOVE_INDEX

            from_canon = self._encode_canonical_xy(board, move.from_pos)
            to_canon = self._encode_canonical_xy(board, move.to)
            if from_canon is None or to_canon is None:
                return INVALID_MOVE_INDEX

            from_cx, from_cy = from_canon
            to_cx, to_cy = to_canon

            dx = to_cx - from_cx
            dy = to_cy - from_cy
            dist = max(abs(dx), abs(dy))
            if dist == 0 or dist > self.max_dist:
                return INVALID_MOVE_INDEX

            dir_x = dx // dist
            dir_y = dy // dist
            try:
                dir_idx = HEX_DIRS.index((dir_x, dir_y))
            except ValueError:
                # Direction not representable in our 6-direction scheme.
                return INVALID_MOVE_INDEX

            from_idx = from_cy * self.board_size + from_cx
            return self.movement_base + from_idx * (NUM_HEX_DIRS * self.max_dist) + dir_idx * self.max_dist + (dist - 1)

        # --- Special ---
        if move.type in (
            MoveType.SKIP_PLACEMENT,
            MoveType.NO_PLACEMENT_ACTION,
            MoveType.NO_MOVEMENT_ACTION,
            MoveType.SKIP_CAPTURE,
            MoveType.NO_LINE_ACTION,
            MoveType.NO_TERRITORY_ACTION,
            MoveType.SKIP_TERRITORY_PROCESSING,
            MoveType.FORCED_ELIMINATION,
        ):
            return self.special_base

        return INVALID_MOVE_INDEX

    def decode_move(
        self,
        index: int,
        game_state: GameState,
    ) -> Optional[Move]:
        """Inverse of encode_move for hex boards.

        Returns a Move instance for valid indices whose endpoints lie on the
        true hex board, or None if the index is out of range or maps off
        the playable hex.
        """
        board = game_state.board

        if board.type not in (BoardType.HEXAGONAL, BoardType.HEX8):
            return None
        if _infer_board_size(board) != self.board_size:
            return None
        if index < 0 or index >= self.policy_size:
            return None

        # --- Placements ---
        if index < self.placement_span:
            count_idx = index % 3
            pos_idx = index // 3
            cy = pos_idx // self.board_size
            cx = pos_idx % self.board_size

            pos = _from_canonical_xy(board, cx, cy)
            if pos is None or not BoardGeometry.is_within_bounds(pos, board.type, board.size):
                return None

            to_payload: dict[str, int] = {"x": pos.x, "y": pos.y}
            if pos.z is not None:
                to_payload["z"] = pos.z

            move_data = {
                "id": "decoded",
                "type": MoveType.PLACE_RING,
                "player": game_state.current_player,
                "to": to_payload,
                "placementCount": count_idx + 1,
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)

        # --- Movement / capture ---
        if index < self.special_base:
            offset = index - self.movement_base

            dist_idx = offset % self.max_dist
            dist = dist_idx + 1
            offset //= self.max_dist

            dir_idx = offset % NUM_HEX_DIRS
            offset //= NUM_HEX_DIRS

            from_idx = offset
            from_cy = from_idx // self.board_size
            from_cx = from_idx % self.board_size

            from_pos = _from_canonical_xy(board, from_cx, from_cy)
            if from_pos is None or not BoardGeometry.is_within_bounds(from_pos, board.type, board.size):
                return None

            dx, dy = HEX_DIRS[dir_idx]
            to_cx = from_cx + dx * dist
            to_cy = from_cy + dy * dist

            to_pos = _from_canonical_xy(board, to_cx, to_cy)
            if to_pos is None or not BoardGeometry.is_within_bounds(to_pos, board.type, board.size):
                return None

            from_payload: dict[str, int] = {"x": from_pos.x, "y": from_pos.y}
            if from_pos.z is not None:
                from_payload["z"] = from_pos.z

            to_payload: dict[str, int] = {"x": to_pos.x, "y": to_pos.y}
            if to_pos.z is not None:
                to_payload["z"] = to_pos.z

            move_data = {
                "id": "decoded",
                "type": MoveType.MOVE_STACK,
                "player": game_state.current_player,
                "from": from_payload,
                "to": to_payload,
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)

        # --- Special ---
        if index == self.special_base:
            move_data = {
                "id": "decoded",
                "type": MoveType.SKIP_PLACEMENT,
                "player": game_state.current_player,
                "to": {"x": 0, "y": 0},
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)

        return None


class HexNeuralNet_v2(nn.Module):
    """
    High-capacity CNN for hexagonal boards (96GB memory target).

    This architecture fixes the critical bug in HexNeuralNet where the policy
    head flattened full spatial features (80,000 dims) directly to policy logits,
    resulting in 7.35 billion parameters. The v2 architecture uses global average
    pooling before the policy FC layer, reducing parameters by 169×.

    Key improvements over HexNeuralNet:
    - Policy head uses global avg pool → FC (like RingRiftCNN_MPS)
    - 12 SE residual blocks with Squeeze-and-Excitation
    - 192 filters for richer hex representations
    - 14 base input channels capturing stack/cap/territory mechanics
    - 20 global features for multi-player state tracking
    - Multi-player value head with masked pooling for hex grid
    - 384-dim policy intermediate for better move discrimination

    Hex-specific features:
    - Automatic hex mask generation and caching
    - Masked global average pooling for valid cells only
    - 469 valid cells in 25×25 bounding box (radius 12)

    Input Feature Channels (14 base × 4 frames = 56 total):
        1-4: Per-player stack presence (binary, one per player)
        5-8: Per-player marker presence (binary)
        9: Stack height (normalized 0-1)
        10: Cap height (normalized)
        11: Collapsed territory (binary)
        12-14: Territory ownership channels

    Global Features (20):
        1-4: Rings in hand (per player)
        5-8: Eliminated rings (per player)
        9-12: Territory count (per player)
        13-16: Line count (per player)
        17: Current player indicator
        18: Game phase (early/mid/late)
        19: Total rings in play
        20: LPS threat indicator

    Memory profile (FP32):
    - Model weights: ~180 MB (vs ~29 GB in original!)
    - Per-model with activations: ~380 MB
    - Two models + MCTS: ~18 GB total

    Architecture Version:
        v2.0.0 - Fixed policy head, SE blocks, high-capacity for 96GB systems.
    """

    ARCHITECTURE_VERSION = "v2.0.0"

    def __init__(
        self,
        in_channels: int = 40,  # 10 base × 4 frames (hex uses fewer channels than square)
        global_features: int = 20,
        num_res_blocks: int = 12,
        num_filters: int = 192,
        board_size: int = HEX_BOARD_SIZE,
        policy_size: int = P_HEX,
        policy_intermediate: int = 384,
        value_intermediate: int = 128,
        num_players: int = 4,
        se_reduction: int = 16,
        hex_radius: int = 12,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.global_features = global_features
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.board_size = board_size
        self.policy_size = policy_size
        self.num_players = num_players

        # Pre-compute hex validity mask
        self.register_buffer("hex_mask", create_hex_mask(hex_radius, board_size))

        # Shared backbone with SE blocks
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()
        self.res_blocks = nn.ModuleList(
            [SEResidualBlock(num_filters, reduction=se_reduction) for _ in range(num_res_blocks)]
        )

        # Multi-player value head with masked pooling
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 + global_features, value_intermediate)
        self.value_fc2 = nn.Linear(value_intermediate, num_players)
        self.tanh = nn.Tanh()

        # Policy head: 1×1 conv → global avg pool → FC (FIXED!)
        self.policy_conv = nn.Conv2d(num_filters, num_filters, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(num_filters)
        self.policy_fc1 = nn.Linear(num_filters + global_features, policy_intermediate)
        self.policy_fc2 = nn.Linear(policy_intermediate, policy_size)
        self.dropout = nn.Dropout(0.3)

    def _masked_global_avg_pool(
        self,
        x: torch.Tensor,
        hex_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform masked global average pooling over [H, W] for hex grid."""
        mask = hex_mask if hex_mask is not None else self.hex_mask
        if mask is None:
            return x.mean(dim=(2, 3))
        mask = mask.to(dtype=x.dtype, device=x.device)
        masked = x * mask
        num = masked.sum(dim=(2, 3))
        denom = mask.sum(dim=(2, 3)).clamp_min(1.0)
        return num / denom

    def forward(
        self,
        x: torch.Tensor,
        globals: torch.Tensor,
        hex_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Apply hex mask to input to prevent information bleeding
        if hex_mask is not None:
            x = x * hex_mask.to(dtype=x.dtype, device=x.device)
        elif self.hex_mask is not None:
            x = x * self.hex_mask.to(dtype=x.dtype, device=x.device)

        # Backbone with SE blocks
        out = self.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            out = block(out)

        # Multi-player value head with masked pooling
        v = self.value_conv(out)
        v = self.relu(self.value_bn(v))
        v_pooled = self._masked_global_avg_pool(v, hex_mask)
        v_cat = torch.cat([v_pooled, globals], dim=1)
        v_hidden = self.relu(self.value_fc1(v_cat))
        v_hidden = self.dropout(v_hidden)
        v_out = self.tanh(self.value_fc2(v_hidden))  # [B, num_players]

        # Policy head with masked global avg pool
        p = self.policy_conv(out)
        p = self.relu(self.policy_bn(p))
        p_pooled = self._masked_global_avg_pool(p, hex_mask)
        p_cat = torch.cat([p_pooled, globals], dim=1)
        p_hidden = self.relu(self.policy_fc1(p_cat))
        p_hidden = self.dropout(p_hidden)
        p_logits = self.policy_fc2(p_hidden)

        return v_out, p_logits


class HexNeuralNet_v2_Lite(nn.Module):
    """
    Memory-efficient CNN for hexagonal boards (48GB memory target).

    This architecture provides the same bug fix as HexNeuralNet_v2 but with
    reduced capacity for systems with limited memory (48GB). Suitable for
    running two instances simultaneously for comparison matches.

    Key trade-offs vs HexNeuralNet_v2:
    - 6 SE residual blocks (vs 12) - faster but shallower
    - 96 filters (vs 192) - smaller representations
    - 192-dim policy intermediate (vs 384)
    - 12 base input channels (vs 14) - reduced history
    - 3 history frames (vs 4) - reduced temporal context

    Hex-specific features:
    - Automatic hex mask generation and caching
    - Masked global average pooling for valid cells only
    - Input masking to prevent information bleeding
    - 469 valid cells in 25×25 bounding box (radius 12)

    Input Feature Channels (12 base × 3 frames = 36 total):
        1-4: Per-player stack presence (binary)
        5-8: Per-player marker presence (binary)
        9: Stack height (normalized)
        10: Cap height (normalized)
        11: Collapsed territory (binary)
        12: Current player territory

    Global Features (20):
        Same as HexNeuralNet_v2 for compatibility

    Memory profile (FP32):
    - Model weights: ~75 MB
    - Per-model with activations: ~150 MB
    - Two models + MCTS: ~10 GB total

    Architecture Version:
        v2.0.0-lite - SE blocks, hex masking, memory-efficient for 48GB.
    """

    ARCHITECTURE_VERSION = "v2.0.0-lite"

    def __init__(
        self,
        in_channels: int = 36,  # 12 base × 3 frames
        global_features: int = 20,
        num_res_blocks: int = 6,
        num_filters: int = 96,
        board_size: int = HEX_BOARD_SIZE,
        policy_size: int = P_HEX,
        policy_intermediate: int = 192,
        value_intermediate: int = 64,
        num_players: int = 4,
        se_reduction: int = 16,
        hex_radius: int = 12,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.global_features = global_features
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.board_size = board_size
        self.policy_size = policy_size
        self.num_players = num_players

        # Pre-compute hex validity mask
        self.register_buffer("hex_mask", create_hex_mask(hex_radius, board_size))

        # Shared backbone with SE blocks
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()
        self.res_blocks = nn.ModuleList(
            [SEResidualBlock(num_filters, reduction=se_reduction) for _ in range(num_res_blocks)]
        )

        # Multi-player value head with masked pooling
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 + global_features, value_intermediate)
        self.value_fc2 = nn.Linear(value_intermediate, num_players)
        self.tanh = nn.Tanh()

        # Policy head: 1×1 conv → masked global avg pool → FC (FIXED!)
        self.policy_conv = nn.Conv2d(num_filters, num_filters, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(num_filters)
        self.policy_fc1 = nn.Linear(num_filters + global_features, policy_intermediate)
        self.policy_fc2 = nn.Linear(policy_intermediate, policy_size)
        self.dropout = nn.Dropout(0.3)

    def _masked_global_avg_pool(
        self,
        x: torch.Tensor,
        hex_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform masked global average pooling over [H, W] for hex grid."""
        mask = hex_mask if hex_mask is not None else self.hex_mask
        if mask is None:
            return x.mean(dim=(2, 3))
        mask = mask.to(dtype=x.dtype, device=x.device)
        masked = x * mask
        num = masked.sum(dim=(2, 3))
        denom = mask.sum(dim=(2, 3)).clamp_min(1.0)
        return num / denom

    def forward(
        self,
        x: torch.Tensor,
        globals: torch.Tensor,
        hex_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Apply hex mask to input to prevent information bleeding
        if hex_mask is not None:
            x = x * hex_mask.to(dtype=x.dtype, device=x.device)
        elif self.hex_mask is not None:
            x = x * self.hex_mask.to(dtype=x.dtype, device=x.device)

        # Backbone with SE blocks
        out = self.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            out = block(out)

        # Multi-player value head with masked pooling
        v = self.value_conv(out)
        v = self.relu(self.value_bn(v))
        v_pooled = self._masked_global_avg_pool(v, hex_mask)
        v_cat = torch.cat([v_pooled, globals], dim=1)
        v_hidden = self.relu(self.value_fc1(v_cat))
        v_hidden = self.dropout(v_hidden)
        v_out = self.tanh(self.value_fc2(v_hidden))  # [B, num_players]

        # Policy head with masked global avg pool
        p = self.policy_conv(out)
        p = self.relu(self.policy_bn(p))
        p_pooled = self._masked_global_avg_pool(p, hex_mask)
        p_cat = torch.cat([p_pooled, globals], dim=1)
        p_hidden = self.relu(self.policy_fc1(p_cat))
        p_hidden = self.dropout(p_hidden)
        p_logits = self.policy_fc2(p_hidden)

        return v_out, p_logits


class HexNeuralNet_v3(nn.Module):
    """
    V3 architecture with spatial policy heads for hexagonal boards.

    This architecture improves on V2 by using spatially-structured policy heads
    that preserve the geometric relationship between positions and actions,
    rather than collapsing everything through global average pooling.

    Key improvements over V2:
    - Spatial placement head: Conv1×1 → [B, 3, H, W] for (cell, ring_count) logits
    - Spatial movement head: Conv1×1 → [B, 144, H, W] for (cell, dir, dist) logits
    - Small FC for special actions (skip_placement only)
    - Logits are scattered into canonical P_HEX=91,876 flat policy vector
    - Preserves spatial locality during policy computation

    Why spatial heads are better:
    1. No spatial information loss - each cell produces its own policy logits
    2. Better gradient flow - actions at position (x,y) directly update features at (x,y)
    3. Reduced parameter count - Conv1×1 vs large FC layer
    4. Natural hex masking - invalid cells produce masked logits

    Policy Layout (P_HEX = 91,876):
        Placements:  [0, 1874]     = 25×25×3 = 1,875 (cell × ring_count)
        Movements:   [1875, 91874] = 25×25×6×24 = 90,000 (cell × dir × dist)
        Special:     [91875]       = 1 (skip_placement)

    Architecture Version:
        v3.0.0 - Spatial policy heads, SE backbone, MPS compatible.
    """

    ARCHITECTURE_VERSION = "v3.0.0"

    def __init__(
        self,
        in_channels: int = 64,  # 16 base × 4 frames for V3 encoder
        global_features: int = 20,  # V3 encoder provides 20 global features
        num_res_blocks: int = 12,
        num_filters: int = 192,
        board_size: int = HEX_BOARD_SIZE,
        policy_size: int = P_HEX,
        value_intermediate: int = 128,
        num_players: int = 4,
        se_reduction: int = 16,
        hex_radius: int = 12,
        num_ring_counts: int = 3,  # Ring count options (1, 2, 3)
        num_directions: int = NUM_HEX_DIRS,  # 6 hex directions
        max_distance: int = HEX_MAX_DIST,  # 24 distance buckets
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.global_features = global_features
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.board_size = board_size
        self.policy_size = policy_size
        self.num_players = num_players

        # Spatial policy dimensions
        self.num_ring_counts = num_ring_counts
        self.num_directions = num_directions
        self.max_distance = max_distance
        self.movement_channels = num_directions * max_distance  # 6 × 24 = 144

        # Pre-compute hex validity mask
        self.register_buffer("hex_mask", create_hex_mask(hex_radius, board_size))

        # Pre-compute index scatter tensors for policy assembly
        self._register_policy_indices(board_size)

        # Shared backbone with SE blocks (same as V2)
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()
        self.res_blocks = nn.ModuleList(
            [SEResidualBlock(num_filters, reduction=se_reduction) for _ in range(num_res_blocks)]
        )

        # Multi-player value head with masked pooling (same as V2)
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 + global_features, value_intermediate)
        self.value_fc2 = nn.Linear(value_intermediate, num_players)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3)

        # === V3 Spatial Policy Heads ===
        # Placement head: produces logits for each (cell, ring_count) tuple
        # Output shape: [B, 3, 25, 25] → indices [0, 1874]
        self.placement_conv = nn.Conv2d(num_filters, num_ring_counts, kernel_size=1)

        # Movement head: produces logits for each (cell, dir, dist) tuple
        # Output shape: [B, 144, 25, 25] → indices [1875, 91874]
        self.movement_conv = nn.Conv2d(num_filters, self.movement_channels, kernel_size=1)

        # Special actions head: small FC for skip_placement
        # Uses global pooled features → single logit
        self.special_fc = nn.Linear(num_filters + global_features, 1)

    def _register_policy_indices(self, board_size: int) -> None:
        """
        Pre-compute index tensors for scattering spatial logits into flat policy.

        Placement indexing: idx = y * W * 3 + x * 3 + ring_count
        Movement indexing: idx = HEX_MOVEMENT_BASE + y * W * 6 * 24 + x * 6 * 24 + dir * 24 + (dist - 1)
        """
        H, W = board_size, board_size

        # Placement indices: [3, H, W] → flat index in [0, 1874]
        placement_idx = torch.zeros(self.num_ring_counts, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for r in range(self.num_ring_counts):
                    placement_idx[r, y, x] = y * W * 3 + x * 3 + r
        self.register_buffer("placement_idx", placement_idx)

        # Movement indices: [144, H, W] → flat index in [1875, 91874]
        movement_idx = torch.zeros(self.movement_channels, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for d in range(self.num_directions):
                    for dist_minus_1 in range(self.max_distance):
                        channel = d * self.max_distance + dist_minus_1
                        flat_idx = (
                            HEX_MOVEMENT_BASE
                            + y * W * self.num_directions * self.max_distance
                            + x * self.num_directions * self.max_distance
                            + d * self.max_distance
                            + dist_minus_1
                        )
                        movement_idx[channel, y, x] = flat_idx
        self.register_buffer("movement_idx", movement_idx)

    def _masked_global_avg_pool(
        self,
        x: torch.Tensor,
        hex_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform masked global average pooling over [H, W] for hex grid."""
        mask = hex_mask if hex_mask is not None else self.hex_mask
        if mask is None:
            return x.mean(dim=(2, 3))
        mask = mask.to(dtype=x.dtype, device=x.device)
        masked = x * mask
        num = masked.sum(dim=(2, 3))
        denom = mask.sum(dim=(2, 3)).clamp_min(1.0)
        return num / denom

    def _scatter_policy_logits(
        self,
        placement_logits: torch.Tensor,
        movement_logits: torch.Tensor,
        special_logits: torch.Tensor,
        hex_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Scatter spatial policy logits into flat P_HEX policy vector.

        Args:
            placement_logits: [B, 3, H, W] placement logits
            movement_logits: [B, 144, H, W] movement logits
            special_logits: [B, 1] special action logits
            hex_mask: Optional [1, H, W] validity mask

        Returns:
            policy_logits: [B, P_HEX] flat policy vector
        """
        B = placement_logits.size(0)
        device = placement_logits.device
        dtype = placement_logits.dtype

        # Initialize flat policy with large negative (will be masked anyway)
        # Use -1e4 instead of -1e9 to avoid float16 overflow in mixed precision
        policy = torch.full((B, self.policy_size), -1e4, device=device, dtype=dtype)

        # Flatten spatial dimensions for scatter
        # placement_logits: [B, 3, H, W] → [B, 3*H*W]
        placement_flat = placement_logits.view(B, -1)
        placement_idx_flat = self.placement_idx.view(-1).expand(B, -1)

        # movement_logits: [B, 144, H, W] → [B, 144*H*W]
        movement_flat = movement_logits.view(B, -1)
        movement_idx_flat = self.movement_idx.view(-1).expand(B, -1)

        # Scatter placement and movement logits
        policy.scatter_(1, placement_idx_flat, placement_flat)
        policy.scatter_(1, movement_idx_flat, movement_flat)

        # Add special action logit at index HEX_SPECIAL_BASE
        policy[:, HEX_SPECIAL_BASE : HEX_SPECIAL_BASE + 1] = special_logits

        return policy

    def forward(
        self,
        x: torch.Tensor,
        globals: torch.Tensor,
        hex_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with spatial policy heads.

        Args:
            x: Input features [B, in_channels, H, W]
            globals: Global features [B, global_features]
            hex_mask: Optional validity mask [1, H, W]

        Returns:
            value: [B, num_players] per-player win probability
            policy: [B, P_HEX] flat policy logits
        """
        # Apply hex mask to input to prevent information bleeding
        mask = hex_mask if hex_mask is not None else self.hex_mask
        if mask is not None:
            x = x * mask.to(dtype=x.dtype, device=x.device)

        # Backbone with SE blocks
        out = self.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            out = block(out)

        # === Value Head (same as V2) ===
        v = self.value_conv(out)
        v = self.relu(self.value_bn(v))
        v_pooled = self._masked_global_avg_pool(v, hex_mask)
        v_cat = torch.cat([v_pooled, globals], dim=1)
        v_hidden = self.relu(self.value_fc1(v_cat))
        v_hidden = self.dropout(v_hidden)
        v_out = self.tanh(self.value_fc2(v_hidden))  # [B, num_players]

        # === Spatial Policy Heads (V3) ===
        # Placement logits: [B, 3, H, W]
        placement_logits = self.placement_conv(out)

        # Movement logits: [B, 144, H, W]
        movement_logits = self.movement_conv(out)

        # Apply hex mask to spatial logits (invalid cells get -inf)
        if mask is not None:
            mask_expanded = mask.to(dtype=out.dtype, device=out.device)
            # Broadcast mask to all channels
            placement_logits = placement_logits * mask_expanded + (1.0 - mask_expanded) * (-1e9)
            movement_logits = movement_logits * mask_expanded + (1.0 - mask_expanded) * (-1e9)

        # Special action logits from pooled features
        out_pooled = self._masked_global_avg_pool(out, hex_mask)
        special_input = torch.cat([out_pooled, globals], dim=1)
        special_logits = self.special_fc(special_input)  # [B, 1]

        # Scatter into flat policy vector
        policy_logits = self._scatter_policy_logits(placement_logits, movement_logits, special_logits, hex_mask)

        return v_out, policy_logits


class HexNeuralNet_v3_Lite(nn.Module):
    """
    Memory-efficient V3 architecture with spatial policy heads (48GB target).

    Same spatial policy head design as HexNeuralNet_v3 but with reduced capacity:
    - 6 SE residual blocks (vs 12)
    - 96 filters (vs 192)
    - 3 history frames (vs 4)
    - 12 base input channels (vs 14)

    Architecture Version:
        v3.0.0-lite - Spatial policy heads, reduced capacity for 48GB systems.
    """

    ARCHITECTURE_VERSION = "v3.0.0-lite"

    def __init__(
        self,
        in_channels: int = 44,  # 12 base × 3 frames + 8 phase/chain planes
        global_features: int = 20,
        num_res_blocks: int = 6,
        num_filters: int = 96,
        board_size: int = HEX_BOARD_SIZE,
        policy_size: int = P_HEX,
        value_intermediate: int = 64,
        num_players: int = 4,
        se_reduction: int = 8,
        hex_radius: int = 12,
        num_ring_counts: int = 3,
        num_directions: int = NUM_HEX_DIRS,
        max_distance: int = HEX_MAX_DIST,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.global_features = global_features
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.board_size = board_size
        self.policy_size = policy_size
        self.num_players = num_players

        # Spatial policy dimensions
        self.num_ring_counts = num_ring_counts
        self.num_directions = num_directions
        self.max_distance = max_distance
        self.movement_channels = num_directions * max_distance

        # Pre-compute hex validity mask
        self.register_buffer("hex_mask", create_hex_mask(hex_radius, board_size))

        # Pre-compute index scatter tensors
        self._register_policy_indices(board_size)

        # Shared backbone with SE blocks
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()
        self.res_blocks = nn.ModuleList(
            [SEResidualBlock(num_filters, reduction=se_reduction) for _ in range(num_res_blocks)]
        )

        # Multi-player value head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 + global_features, value_intermediate)
        self.value_fc2 = nn.Linear(value_intermediate, num_players)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3)

        # Spatial policy heads
        self.placement_conv = nn.Conv2d(num_filters, num_ring_counts, kernel_size=1)
        self.movement_conv = nn.Conv2d(num_filters, self.movement_channels, kernel_size=1)
        self.special_fc = nn.Linear(num_filters + global_features, 1)

    def _register_policy_indices(self, board_size: int) -> None:
        """Pre-compute index tensors for scattering spatial logits."""
        H, W = board_size, board_size

        placement_idx = torch.zeros(self.num_ring_counts, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for r in range(self.num_ring_counts):
                    placement_idx[r, y, x] = y * W * 3 + x * 3 + r
        self.register_buffer("placement_idx", placement_idx)

        movement_idx = torch.zeros(self.movement_channels, H, W, dtype=torch.long)
        for y in range(H):
            for x in range(W):
                for d in range(self.num_directions):
                    for dist_minus_1 in range(self.max_distance):
                        channel = d * self.max_distance + dist_minus_1
                        flat_idx = (
                            HEX_MOVEMENT_BASE
                            + y * W * self.num_directions * self.max_distance
                            + x * self.num_directions * self.max_distance
                            + d * self.max_distance
                            + dist_minus_1
                        )
                        movement_idx[channel, y, x] = flat_idx
        self.register_buffer("movement_idx", movement_idx)

    def _masked_global_avg_pool(
        self,
        x: torch.Tensor,
        hex_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform masked global average pooling over [H, W] for hex grid."""
        mask = hex_mask if hex_mask is not None else self.hex_mask
        if mask is None:
            return x.mean(dim=(2, 3))
        mask = mask.to(dtype=x.dtype, device=x.device)
        masked = x * mask
        num = masked.sum(dim=(2, 3))
        denom = mask.sum(dim=(2, 3)).clamp_min(1.0)
        return num / denom

    def _scatter_policy_logits(
        self,
        placement_logits: torch.Tensor,
        movement_logits: torch.Tensor,
        special_logits: torch.Tensor,
        hex_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Scatter spatial policy logits into flat P_HEX policy vector."""
        B = placement_logits.size(0)
        device = placement_logits.device
        dtype = placement_logits.dtype

        # Use -1e4 instead of -1e9 to avoid float16 overflow in mixed precision
        policy = torch.full((B, self.policy_size), -1e4, device=device, dtype=dtype)

        placement_flat = placement_logits.view(B, -1)
        placement_idx_flat = self.placement_idx.view(-1).expand(B, -1)

        movement_flat = movement_logits.view(B, -1)
        movement_idx_flat = self.movement_idx.view(-1).expand(B, -1)

        policy.scatter_(1, placement_idx_flat, placement_flat)
        policy.scatter_(1, movement_idx_flat, movement_flat)
        policy[:, HEX_SPECIAL_BASE : HEX_SPECIAL_BASE + 1] = special_logits

        return policy

    def forward(
        self,
        x: torch.Tensor,
        globals: torch.Tensor,
        hex_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with spatial policy heads."""
        mask = hex_mask if hex_mask is not None else self.hex_mask
        if mask is not None:
            x = x * mask.to(dtype=x.dtype, device=x.device)

        out = self.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            out = block(out)

        # Value head
        v = self.value_conv(out)
        v = self.relu(self.value_bn(v))
        v_pooled = self._masked_global_avg_pool(v, hex_mask)
        v_cat = torch.cat([v_pooled, globals], dim=1)
        v_hidden = self.relu(self.value_fc1(v_cat))
        v_hidden = self.dropout(v_hidden)
        v_out = self.tanh(self.value_fc2(v_hidden))

        # Spatial policy heads
        placement_logits = self.placement_conv(out)
        movement_logits = self.movement_conv(out)

        if mask is not None:
            mask_expanded = mask.to(dtype=out.dtype, device=out.device)
            placement_logits = placement_logits * mask_expanded + (1.0 - mask_expanded) * (-1e9)
            movement_logits = movement_logits * mask_expanded + (1.0 - mask_expanded) * (-1e9)

        out_pooled = self._masked_global_avg_pool(out, hex_mask)
        special_input = torch.cat([out_pooled, globals], dim=1)
        special_logits = self.special_fc(special_input)

        policy_logits = self._scatter_policy_logits(placement_logits, movement_logits, special_logits, hex_mask)

        return v_out, policy_logits


# Compatibility aliases: older callers expect these legacy names; map to v2 implementations.
HexNeuralNet = HexNeuralNet_v2
HexNeuralNet_Lite = HexNeuralNet_v2_Lite

# Explicit export list for consumers relying on __all__
__all__ = [
    "NeuralNetAI",
    "HexNeuralNet",
    "HexNeuralNet_Lite",
    "HexNeuralNet_v2",
    "HexNeuralNet_v2_Lite",
    "HexNeuralNet_v3",
    "HexNeuralNet_v3_Lite",
    "RingRiftCNN_v2",
    "RingRiftCNN_v3",
    "clear_model_cache",
    "encode_move_for_board",
    "get_cached_model_count",
    "get_policy_size_for_board",
    "get_spatial_size_for_board",
]

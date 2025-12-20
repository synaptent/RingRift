"""Neural network constants and policy size definitions.

This module contains all the constants used across the neural network
implementation, including policy sizes for different board types,
hex geometry constants, and encoding layout spans.
"""

from typing import Dict
from app.models import BoardType

# Invalid move marker
INVALID_MOVE_INDEX = -1

# Canonical maximum side length for policy encoding (19x19 grid)
MAX_N = 19

# Maximum number of players for multi-player value head
MAX_PLAYERS = 4

# =============================================================================
# Square Board Policy Sizes
# =============================================================================

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

# =============================================================================
# Square8 Policy Layout Constants
# =============================================================================

# Detailed layout for 8x8 board:
# Placement: 192 indices (64 positions × 3 counts)
SQUARE8_PLACEMENT_SPAN = 8 * 8 * 3  # 192

# Movement: 3584 indices (64 positions × 8 directions × 7 distances)
SQUARE8_MOVEMENT_BASE = SQUARE8_PLACEMENT_SPAN  # 192
SQUARE8_MOVEMENT_SPAN = 8 * 8 * 8 * 7  # 3584

# Line formation: 256 indices (64 positions × 4 directions)
SQUARE8_LINE_FORM_BASE = SQUARE8_MOVEMENT_BASE + SQUARE8_MOVEMENT_SPAN  # 3776
SQUARE8_LINE_FORM_SPAN = 8 * 8 * 4  # 256

# Territory claim: 64 indices (64 positions)
SQUARE8_TERRITORY_CLAIM_BASE = SQUARE8_LINE_FORM_BASE + SQUARE8_LINE_FORM_SPAN  # 4032
SQUARE8_TERRITORY_CLAIM_SPAN = 8 * 8  # 64

# Special actions
SQUARE8_SPECIAL_BASE = SQUARE8_TERRITORY_CLAIM_BASE + SQUARE8_TERRITORY_CLAIM_SPAN  # 4096
SQUARE8_SKIP_PLACEMENT_IDX = SQUARE8_SPECIAL_BASE  # 4096
SQUARE8_SWAP_SIDES_IDX = SQUARE8_SPECIAL_BASE + 1  # 4097

# Line choice: 4 indices
SQUARE8_LINE_CHOICE_BASE = SQUARE8_SPECIAL_BASE + 2  # 4098
SQUARE8_LINE_CHOICE_SPAN = 4  # 4

# Territory choice: 2048 indices (64 × 8 × 4)
SQUARE8_TERRITORY_CHOICE_BASE = SQUARE8_LINE_CHOICE_BASE + SQUARE8_LINE_CHOICE_SPAN  # 4102
SQUARE8_TERRITORY_CHOICE_SPAN = 8 * 8 * 8 * 4  # 2048

# =============================================================================
# Hexagonal Board Constants
# =============================================================================

# Canonical competitive hex board has radius N = 12, yielding 469 cells.
# We embed into a fixed 25×25 bounding box (2N + 1 on each axis).
HEX_BOARD_SIZE = 25
HEX_MAX_DIST = HEX_BOARD_SIZE - 1  # 24 distance buckets (1..24)

# Canonical axial directions in the 2D embedding.
HEX_DIRS = [
    (1, 0),   # +q
    (0, 1),   # +r
    (-1, 1),  # -q + r
    (-1, 0),  # -q
    (0, -1),  # -r
    (1, -1),  # +q - r
]
NUM_HEX_DIRS = len(HEX_DIRS)

# Layout spans for the hex policy head:
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
# Hex8 Board Constants (radius-4 hexagonal board, 61 cells)
# =============================================================================

HEX8_BOARD_SIZE = 9  # 2 * 4 + 1 = 9
HEX8_MAX_DIST = HEX8_BOARD_SIZE - 1  # 8 distance buckets

# Hex8 policy layout:
# Placements: 9 × 9 × 3 = 243
# Movement: 9 × 9 × 6 × 8 = 3_888
# Special: 1
# Total: 4_132 → 4_500 (with padding)
HEX8_PLACEMENT_SPAN = HEX8_BOARD_SIZE * HEX8_BOARD_SIZE * 3
HEX8_MOVEMENT_BASE = HEX8_PLACEMENT_SPAN
HEX8_MOVEMENT_SPAN = HEX8_BOARD_SIZE * HEX8_BOARD_SIZE * NUM_HEX_DIRS * HEX8_MAX_DIST
HEX8_SPECIAL_BASE = HEX8_MOVEMENT_BASE + HEX8_MOVEMENT_SPAN
POLICY_SIZE_HEX8 = 4500

# =============================================================================
# Board Type Mappings
# =============================================================================

# Board type to policy size mapping
BOARD_POLICY_SIZES: Dict[BoardType, int] = {
    BoardType.SQUARE8: POLICY_SIZE_8x8,
    BoardType.SQUARE19: POLICY_SIZE_19x19,
    BoardType.HEX8: POLICY_SIZE_HEX8,
    BoardType.HEXAGONAL: P_HEX,
}

# Board type to spatial size mapping
BOARD_SPATIAL_SIZES: Dict[BoardType, int] = {
    BoardType.SQUARE8: 8,
    BoardType.SQUARE19: 19,
    BoardType.HEX8: HEX8_BOARD_SIZE,
    BoardType.HEXAGONAL: HEX_BOARD_SIZE,
}


def get_policy_size_for_board(board_type: BoardType) -> int:
    """Get the optimal policy head size for a board type."""
    return BOARD_POLICY_SIZES.get(board_type, POLICY_SIZE_19x19)


def get_spatial_size_for_board(board_type: BoardType) -> int:
    """Get the spatial (H, W) size for a board type."""
    return BOARD_SPATIAL_SIZES.get(board_type, 19)

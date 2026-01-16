"""Neural network constants and policy size definitions.

This module contains all the constants used across the neural network
implementation, including policy sizes for different board types,
hex geometry constants, and encoding layout spans.
"""


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
SQUARE8_SKIP_RECOVERY_IDX = SQUARE8_SPECIAL_BASE + 2  # 4098 (RR-CANON-R112)

# Line choice: 4 indices
SQUARE8_LINE_CHOICE_BASE = SQUARE8_SPECIAL_BASE + 3  # 4099
SQUARE8_LINE_CHOICE_SPAN = 4  # 4

# Territory choice: 2048 indices (64 × 8 × 4)
SQUARE8_TERRITORY_CHOICE_BASE = SQUARE8_LINE_CHOICE_BASE + SQUARE8_LINE_CHOICE_SPAN  # 4103
SQUARE8_TERRITORY_CHOICE_SPAN = 8 * 8 * 8 * 4  # 2048

# Extra special actions (after territory choice block)
SQUARE8_EXTRA_SPECIAL_BASE = (
    SQUARE8_TERRITORY_CHOICE_BASE + SQUARE8_TERRITORY_CHOICE_SPAN
)  # 6151
SQUARE8_NO_PLACEMENT_ACTION_IDX = SQUARE8_EXTRA_SPECIAL_BASE
SQUARE8_NO_MOVEMENT_ACTION_IDX = SQUARE8_EXTRA_SPECIAL_BASE + 1
SQUARE8_SKIP_CAPTURE_IDX = SQUARE8_EXTRA_SPECIAL_BASE + 2
SQUARE8_NO_LINE_ACTION_IDX = SQUARE8_EXTRA_SPECIAL_BASE + 3
SQUARE8_NO_TERRITORY_ACTION_IDX = SQUARE8_EXTRA_SPECIAL_BASE + 4
SQUARE8_SKIP_TERRITORY_PROCESSING_IDX = SQUARE8_EXTRA_SPECIAL_BASE + 5
SQUARE8_FORCED_ELIMINATION_IDX = SQUARE8_EXTRA_SPECIAL_BASE + 6

# Square8 distance and extra special span
MAX_DIST_SQUARE8 = 7  # Max diagonal on 8x8 board
SQUARE8_EXTRA_SPECIAL_SPAN = 7  # Number of extra special action indices

# =============================================================================
# Square19 Policy Layout Constants
# =============================================================================

# Direction and distance constants
NUM_SQUARE_DIRS = 8  # 8 cardinal + diagonal directions
NUM_LINE_DIRS = 4  # 4 line formation directions
MAX_DIST_SQUARE19 = 18  # Max diagonal: sqrt(18^2 + 18^2) ≈ 25.5

# Territory encoding
TERRITORY_SIZE_BUCKETS = 8
TERRITORY_MAX_PLAYERS = 4

# Detailed layout for 19x19 board:
SQUARE19_PLACEMENT_SPAN = 3 * 19 * 19  # 1,083
SQUARE19_MOVEMENT_BASE = SQUARE19_PLACEMENT_SPAN
SQUARE19_MOVEMENT_SPAN = 19 * 19 * NUM_SQUARE_DIRS * MAX_DIST_SQUARE19  # 51,984
SQUARE19_LINE_FORM_BASE = SQUARE19_MOVEMENT_BASE + SQUARE19_MOVEMENT_SPAN
SQUARE19_LINE_FORM_SPAN = 19 * 19 * NUM_LINE_DIRS  # 1,444
SQUARE19_TERRITORY_CLAIM_BASE = SQUARE19_LINE_FORM_BASE + SQUARE19_LINE_FORM_SPAN
SQUARE19_TERRITORY_CLAIM_SPAN = 19 * 19  # 361

# Special actions
SQUARE19_SPECIAL_BASE = SQUARE19_TERRITORY_CLAIM_BASE + SQUARE19_TERRITORY_CLAIM_SPAN
SQUARE19_SKIP_PLACEMENT_IDX = SQUARE19_SPECIAL_BASE  # 54872
SQUARE19_SWAP_SIDES_IDX = SQUARE19_SPECIAL_BASE + 1  # 54873
SQUARE19_SKIP_RECOVERY_IDX = SQUARE19_SPECIAL_BASE + 2  # 54874 (RR-CANON-R112)
SQUARE19_LINE_CHOICE_BASE = SQUARE19_SPECIAL_BASE + 3  # 54875
SQUARE19_LINE_CHOICE_SPAN = 4
SQUARE19_TERRITORY_CHOICE_BASE = SQUARE19_LINE_CHOICE_BASE + SQUARE19_LINE_CHOICE_SPAN
SQUARE19_TERRITORY_CHOICE_SPAN = 19 * 19 * TERRITORY_SIZE_BUCKETS * TERRITORY_MAX_PLAYERS  # 11,552

# Extra special actions (after territory choice block)
SQUARE19_EXTRA_SPECIAL_BASE = (
    SQUARE19_TERRITORY_CHOICE_BASE + SQUARE19_TERRITORY_CHOICE_SPAN
)  # 66431
SQUARE19_NO_PLACEMENT_ACTION_IDX = SQUARE19_EXTRA_SPECIAL_BASE
SQUARE19_NO_MOVEMENT_ACTION_IDX = SQUARE19_EXTRA_SPECIAL_BASE + 1
SQUARE19_SKIP_CAPTURE_IDX = SQUARE19_EXTRA_SPECIAL_BASE + 2
SQUARE19_NO_LINE_ACTION_IDX = SQUARE19_EXTRA_SPECIAL_BASE + 3
SQUARE19_NO_TERRITORY_ACTION_IDX = SQUARE19_EXTRA_SPECIAL_BASE + 4
SQUARE19_SKIP_TERRITORY_PROCESSING_IDX = SQUARE19_EXTRA_SPECIAL_BASE + 5
SQUARE19_FORCED_ELIMINATION_IDX = SQUARE19_EXTRA_SPECIAL_BASE + 6
SQUARE19_EXTRA_SPECIAL_SPAN = 7  # Number of extra special action indices

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
BOARD_POLICY_SIZES: dict[BoardType, int] = {
    BoardType.SQUARE8: POLICY_SIZE_8x8,
    BoardType.SQUARE19: POLICY_SIZE_19x19,
    BoardType.HEX8: POLICY_SIZE_HEX8,
    BoardType.HEXAGONAL: P_HEX,
}

# Board type to spatial size mapping
BOARD_SPATIAL_SIZES: dict[BoardType, int] = {
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


def compute_hex_policy_size(
    board_size: int,
    num_ring_counts: int = 3,
    num_directions: int = 6,
) -> int:
    """Compute the policy size for a hex board of given size.

    This is the canonical formula used by V3/V4 spatial policy heads:
    - Placement span: board_size * board_size * num_ring_counts
    - Movement span: board_size * board_size * num_directions * (board_size - 1)
    - Special actions: 1 (skip_placement)

    Args:
        board_size: Bounding box size (9 for hex8, 25 for hexagonal)
        num_ring_counts: Number of ring placement options (default: 3)
        num_directions: Number of hex directions (default: 6)

    Returns:
        Total policy vector size

    Examples:
        >>> compute_hex_policy_size(9)   # hex8
        4132
        >>> compute_hex_policy_size(25)  # hexagonal
        91876
    """
    max_distance = board_size - 1
    placement_span = board_size * board_size * num_ring_counts
    movement_span = board_size * board_size * num_directions * max_distance
    return placement_span + movement_span + 1

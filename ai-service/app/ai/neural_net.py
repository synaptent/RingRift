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
uses a singleton model cache (``_MODEL_CACHE``) that shares model instances
across multiple :class:`NeuralNetAI` instances. Call :func:`clear_model_cache`
to release GPU/MPS memory between games or soak batches.
"""

import gc
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

logger = logging.getLogger(__name__)

# =============================================================================
# Model Cache for Memory Efficiency
# =============================================================================
#
# Singleton cache to share model instances across NeuralNetAI instances.
# Key: (architecture_type, device_str, model_path)
# Value: loaded model instance
_MODEL_CACHE: Dict[Tuple[str, str, str], nn.Module] = {}


def clear_model_cache() -> None:
    """Clear the model cache and release GPU/MPS memory.

    Call this function between games or soak batches to prevent OOM issues.
    This is especially important for MPS where memory management is more
    aggressive than CUDA.
    """
    global _MODEL_CACHE
    cache_size = len(_MODEL_CACHE)

    # Move models to CPU before clearing to release GPU memory
    for model in _MODEL_CACHE.values():
        try:
            model.cpu()
        except Exception:
            pass

    _MODEL_CACHE.clear()

    # Clear PyTorch caches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Clear MPS cache if available (PyTorch 2.0+)
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    # Force garbage collection
    gc.collect()

    if cache_size > 0:
        logger.info(f"Cleared model cache ({cache_size} models)")


def get_cached_model_count() -> int:
    """Return the number of models currently in the cache."""
    return len(_MODEL_CACHE)


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
    BoardType.HEXAGONAL: 91876,  # P_HEX defined below
}

# Board type to spatial size mapping
BOARD_SPATIAL_SIZES: Dict[BoardType, int] = {
    BoardType.SQUARE8: 8,
    BoardType.SQUARE19: 19,
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
#   Total: 6,150 (POLICY_SIZE_8x8 = 7000 with padding)

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
#   Total: 66,431 (POLICY_SIZE_19x19 = 67000 with padding)

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
        radius = board.size - 1
        return 2 * radius + 1

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

    if board.type == BoardType.HEXAGONAL:
        radius = board.size - 1
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

    if board.type == BoardType.HEXAGONAL:
        radius = board.size - 1
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
        else:
            self.skip_placement_idx = SQUARE19_SKIP_PLACEMENT_IDX
            self.swap_sides_idx = SQUARE19_SWAP_SIDES_IDX
            self.skip_recovery_idx = SQUARE19_SKIP_RECOVERY_IDX
            self.line_choice_base = SQUARE19_LINE_CHOICE_BASE

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
        policy = torch.full((B, self.policy_size), -1e9, device=device, dtype=dtype)

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

        return policy

    def forward(self, x: torch.Tensor, globals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with spatial policy heads and rank distribution output.

        Args:
            x: Input features [B, in_channels, H, W]
            globals: Global features [B, global_features]

        Returns:
            value: [B, num_players] per-player expected outcome (legacy, tanh in [-1, 1])
            policy: [B, policy_size] flat policy logits
            rank_dist: [B, num_players, num_players] rank probability distribution
                       where rank_dist[b, p, r] = P(player p finishes at rank r)
                       Softmax applied over ranks (dim=-1), so each player's probs sum to 1
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
        else:
            self.skip_placement_idx = SQUARE19_SKIP_PLACEMENT_IDX
            self.swap_sides_idx = SQUARE19_SWAP_SIDES_IDX
            self.skip_recovery_idx = SQUARE19_SKIP_RECOVERY_IDX
            self.line_choice_base = SQUARE19_LINE_CHOICE_BASE

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

        policy = torch.full((B, self.policy_size), -1e9, device=device, dtype=dtype)

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

        return policy

    def forward(self, x: torch.Tensor, globals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with spatial policy heads and rank distribution output.

        Returns:
            value: [B, num_players] per-player expected outcome (legacy)
            policy: [B, policy_size] flat policy logits
            rank_dist: [B, num_players, num_players] rank probability distribution
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


def multi_player_value_loss(
    pred_values: torch.Tensor,
    target_values: torch.Tensor,
    num_players: Union[int, torch.Tensor],
) -> torch.Tensor:
    """
    MSE loss for multi-player value predictions.

    Only computes loss over active players (slots 0 to num_players-1).
    Inactive slots (for games with fewer than MAX_PLAYERS) are masked out.

    Parameters
    ----------
    pred_values : torch.Tensor
        Predicted values of shape (batch, MAX_PLAYERS).
    target_values : torch.Tensor
        Target values of shape (batch, MAX_PLAYERS).
    num_players : int | torch.Tensor
        Either a single integer active player count (2, 3, or 4), or a
        per-sample tensor of shape (batch,) with values in [1, MAX_PLAYERS].

    Returns
    -------
    torch.Tensor
        Scalar MSE loss averaged over active players.
    """
    if pred_values.shape != target_values.shape:
        raise ValueError(
            "multi_player_value_loss expects pred_values and target_values to "
            f"share the same shape; got pred_values={tuple(pred_values.shape)} "
            f"target_values={tuple(target_values.shape)}."
        )
    if pred_values.ndim != 2:
        raise ValueError(
            "multi_player_value_loss expects 2D tensors of shape "
            "(batch, max_players); got "
            f"pred_values.ndim={pred_values.ndim}."
        )

    batch_size, max_players = target_values.shape

    # Create mask for active players.
    if isinstance(num_players, int):
        n = int(num_players)
        if n < 1 or n > max_players:
            raise ValueError(
                f"num_players must be in [1, {max_players}], got {n}."
            )
        mask = torch.zeros_like(target_values)
        mask[:, :n] = 1.0
    else:
        num_players_tensor = num_players.to(
            device=target_values.device,
            dtype=torch.long,
        )
        if num_players_tensor.ndim == 0:
            n = int(num_players_tensor.item())
            if n < 1 or n > max_players:
                raise ValueError(
                    f"num_players must be in [1, {max_players}], got {n}."
                )
            mask = torch.zeros_like(target_values)
            mask[:, :n] = 1.0
        elif num_players_tensor.ndim == 1:
            if int(num_players_tensor.shape[0]) != int(batch_size):
                raise ValueError(
                    "Per-sample num_players tensor must have shape (batch,), "
                    f"got {tuple(num_players_tensor.shape)} for batch_size={batch_size}."
                )
            if torch.any(num_players_tensor < 1) or torch.any(num_players_tensor > max_players):
                raise ValueError(
                    "Per-sample num_players tensor contains values outside "
                    f"[1, {max_players}]."
                )

            player_idx = torch.arange(
                max_players,
                device=target_values.device,
            ).unsqueeze(0)
            mask = (
                player_idx < num_players_tensor.unsqueeze(1)
            ).to(dtype=target_values.dtype)
        else:
            raise ValueError(
                "Per-sample num_players tensor must be a scalar or 1D tensor; "
                f"got ndim={num_players_tensor.ndim}."
            )

    # Compute masked MSE
    squared_errors = ((pred_values - target_values) ** 2) * mask
    denom = mask.sum()
    if float(denom.item()) <= 0.0:
        raise ValueError(
            "multi_player_value_loss mask has zero active entries; check num_players."
        )
    loss = squared_errors.sum() / denom

    return loss


def rank_distribution_loss(
    pred_rank_dist: torch.Tensor,
    target_ranks: torch.Tensor,
    num_players: int,
) -> torch.Tensor:
    """
    Cross-entropy loss for rank distribution predictions.

    For each player, computes the cross-entropy loss between the predicted
    rank probability distribution and the actual rank (one-hot encoded).

    Parameters
    ----------
    pred_rank_dist : torch.Tensor
        Predicted rank distributions of shape (batch, MAX_PLAYERS, MAX_PLAYERS).
        pred_rank_dist[b, p, r] = P(player p finishes at rank r).
        Must be probability distributions (sum to 1 over rank dimension).
    target_ranks : torch.Tensor
        Target rank indices of shape (batch, MAX_PLAYERS).
        target_ranks[b, p] = actual rank of player p (0 = 1st place, 1 = 2nd, etc.)
        Values should be in range [0, num_players-1] for active players.
        Inactive player slots (index >= num_players) are ignored.
    num_players : int
        Number of active players in the game (2, 3, or 4).

    Returns
    -------
    torch.Tensor
        Scalar cross-entropy loss averaged over active players.

    Example
    -------
    For a 3-player game where player 0 won (rank 0), player 1 came 2nd (rank 1),
    and player 2 came last (rank 2):
        target_ranks = [0, 1, 2, -1]  # -1 for inactive player 4
        pred_rank_dist = [[0.8, 0.15, 0.05, 0],  # Player 0's rank dist
                          [0.1, 0.7, 0.2, 0],    # Player 1's rank dist
                          [0.1, 0.15, 0.75, 0],  # Player 2's rank dist
                          [0.25, 0.25, 0.25, 0.25]]  # Inactive, ignored
    """
    batch_size = pred_rank_dist.size(0)
    max_players = pred_rank_dist.size(1)

    # Clamp predictions to avoid log(0)
    pred_rank_dist = torch.clamp(pred_rank_dist, min=1e-8)

    # Create mask for active players
    player_mask = torch.zeros(batch_size, max_players, device=pred_rank_dist.device)
    player_mask[:, :num_players] = 1.0

    # Compute cross-entropy for each player
    # For each player p, we want: -log(pred_rank_dist[b, p, target_ranks[b, p]])
    # Use gather to select the predicted probability for the target rank
    target_ranks_clamped = target_ranks.clamp(0, num_players - 1)  # Clamp to valid range
    target_ranks_expanded = target_ranks_clamped.unsqueeze(-1)  # [B, P, 1]
    pred_at_target = pred_rank_dist.gather(dim=2, index=target_ranks_expanded).squeeze(-1)  # [B, P]

    # Negative log likelihood
    nll = -torch.log(pred_at_target)  # [B, P]

    # Apply mask and compute mean
    masked_nll = nll * player_mask
    loss = masked_nll.sum() / player_mask.sum()

    return loss


def ranks_from_game_result(
    winner: int,
    num_players: int,
    player_territories: list[int] | None = None,
    player_eliminated_rings: list[int] | None = None,
    player_markers_on_board: list[int] | None = None,
    elimination_order: list[int] | None = None,
) -> torch.Tensor:
    """
    Compute rank indices from game result using canonical ranking rules.

    Follows Section 8 of ringrift_compact_rules.md:
    1. Winner gets rank 0 (1st place)
    2. Remaining players ranked by: territory → eliminated rings → markers → elimination order

    Parameters
    ----------
    winner : int
        Index of winning player (0-indexed).
    num_players : int
        Number of active players (2, 3, or 4).
    player_territories : list[int] | None
        Territory count per player, or None if not available.
    player_eliminated_rings : list[int] | None
        Total eliminated rings per player, or None.
    player_markers_on_board : list[int] | None
        Markers remaining on board per player, or None.
    elimination_order : list[int] | None
        Order of elimination (later = better), or None.

    Returns
    -------
    torch.Tensor
        Rank indices of shape (MAX_PLAYERS,) where ranks[p] = rank of player p.
        Inactive players get rank num_players (outside valid range).
    """
    MAX_PLAYERS = 4
    ranks = torch.full((MAX_PLAYERS,), MAX_PLAYERS, dtype=torch.long)

    # Winner gets rank 0
    ranks[winner] = 0

    # For remaining players, assign ranks 1, 2, ...
    remaining = [p for p in range(num_players) if p != winner]

    if len(remaining) == 0:
        return ranks

    # Build scoring tuples for sorting (higher is better)
    def score(p: int) -> tuple:
        territory = player_territories[p] if player_territories else 0
        elim_rings = player_eliminated_rings[p] if player_eliminated_rings else 0
        markers = player_markers_on_board[p] if player_markers_on_board else 0
        elim_order = elimination_order.index(p) if elimination_order and p in elimination_order else -1
        return (territory, elim_rings, markers, elim_order)

    # Sort remaining players by score (descending)
    remaining_sorted = sorted(remaining, key=score, reverse=True)

    # Assign ranks
    for rank_idx, player in enumerate(remaining_sorted, start=1):
        ranks[player] = rank_idx

    return ranks


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

    Returns
    -------
    str
        One of "high" or "low".
    """
    tier = os.environ.get("RINGRIFT_NN_MEMORY_TIER", "high").lower()
    if tier not in ("high", "low"):
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

    V3 architectures use spatially-structured Conv1×1 policy heads instead of global
    average pooling + FC layers. This preserves spatial locality and reduces parameter
    count while potentially improving learning for position-dependent actions.

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
    policy_size = get_policy_size_for_board(board_type)

    # Determine memory tier
    tier = memory_tier if memory_tier is not None else get_memory_tier()

    # Create model based on board type and memory tier
    if board_type == BoardType.HEXAGONAL:
        if tier == "v3-high":
            # HexNeuralNet_v3: Spatial policy heads, 12 res blocks, 192 filters, ~8M params
            return HexNeuralNet_v3(
                in_channels=in_channels * (history_length + 1),
                global_features=global_features,
                num_res_blocks=num_res_blocks or 12,
                num_filters=num_filters or 192,
                board_size=board_size,
                policy_size=policy_size,
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
            )
    else:
        # Square boards (8x8 and 19x19)
        if tier == "v3-high":
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
      (e.g. ``"ringrift_v1"``). Resolved to ``<base_dir>/models/<id>.pth``
      (or ``<id>_mps.pth`` for MPS builds). When omitted, falls back to
      ``"ringrift_v1"``.
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
        # Dict[str, List[np.ndarray]] - Keyed by game_id
        self.game_history = {}

        # Track which board type we're initialized for
        self._initialized_board_type: Optional[BoardType] = None

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

    def _ensure_model_initialized(self, board_type: BoardType) -> None:
        """Ensure the model is initialized for the given board type.

        This method is called lazily when the first game state is processed,
        or eagerly if board_type was passed to __init__.

        Args:
            board_type: The board type to initialize the model for.
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

        model_id = getattr(self.config, "nn_model_id", None)
        if not model_id:
            model_id = "ringrift_v1"

        # Board-type-specific model path (e.g., ringrift_v1_hex_mps.pth)
        board_suffix = ""
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

        # Build cache key including board_type
        cache_key = (
            self.architecture_type,
            str(self.device),
            model_path,
            board_type.value if board_type else "unknown",
        )

        # Check cache for existing model
        if cache_key in _MODEL_CACHE:
            self.model = _MODEL_CACHE[cache_key]
            self._initialized_board_type = board_type
            logger.debug(
                f"Reusing cached model: board={board_type}, " f"arch={self.architecture_type}, device={self.device}"
            )
            return

        # Create new model using the factory function
        # NOTE: in_channels=14 is the canonical value for all board types.
        # Old hex models used in_channels=10 but are deprecated (radius-10 geometry).
        # Current hex uses radius-12 geometry which requires 14 channels.
        self.model = create_model_for_board(
            board_type=board_type,
            model_class="auto",  # Let factory choose best class
            use_mps=self._use_mps_arch,
            in_channels=14,
            global_features=20,  # Must match _extract_features() which returns 20 globals
            num_res_blocks=10,
            num_filters=128,
            history_length=self.history_length,
        )
        logger.info(
            f"Initialized {type(self.model).__name__} for {board_type} " f"(policy_size={self.model.policy_size})"
        )

        self.model.to(self.device)

        # Load weights if available.
        #
        # Checkpoints are named with an optional board suffix (e.g. _hex) and
        # an optional architecture suffix (_mps). In practice, most training
        # runs publish a single CPU/CUDA checkpoint without the _mps suffix,
        # so when running with the MPS-friendly architecture we also try the
        # non-MPS filename as a compatibility fallback.
        models_dir = os.path.join(self._base_dir, "models")
        allow_fresh = bool(getattr(self.config, "allow_fresh_weights", False))

        arch_suffix = "_mps" if self.architecture_type == "mps" else ""
        other_arch_suffix = "" if arch_suffix == "_mps" else "_mps"

        candidate_filenames = [
            f"{model_id}{board_suffix}{arch_suffix}.pth",
            f"{model_id}{board_suffix}{other_arch_suffix}.pth",
        ]

        # If we had a board suffix, also try the base model id without the
        # suffix (both arch variants) as a fallback.
        if board_suffix:
            candidate_filenames.extend(
                [
                    f"{model_id}{arch_suffix}.pth",
                    f"{model_id}{other_arch_suffix}.pth",
                ]
            )

        # Deduplicate while preserving order.
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
            # Convenience: allow a stable nn_model_id prefix (e.g.
            # "sq8_2p_nn_baseline") to resolve to the latest timestamped
            # checkpoint "sq8_2p_nn_baseline_<ts>.pth" in the models dir.
            #
            # This keeps ladder configs and training tooling from having to
            # rewrite a new model id every time a checkpoint is produced.
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
            if latest_matches:
                chosen_path = latest_matches[-1]

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
            # No model found - this is often a configuration error in production
            # but may be intentional for training.
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
            if self.device != "cpu" and self.device != "mps":
                self.model = compile_model(
                    self.model,
                    device=torch.device(self.device) if isinstance(self.device, str) else self.device,
                    mode="reduce-overhead",
                )
        except ImportError:
            pass  # gpu_batch not available, skip compilation
        except Exception as e:
            logger.debug(f"torch.compile() skipped: {e}")

        # Cache the model for reuse
        _MODEL_CACHE[cache_key] = self.model
        self._initialized_board_type = board_type
        logger.info(
            f"Cached model: board={board_type}, arch={self.architecture_type}, "
            f"device={self.device} (total cached: {len(_MODEL_CACHE)})"
        )

    def _load_model_checkpoint(self, model_path: str) -> None:
        """
        Load model checkpoint with version validation.

        Uses the model versioning system when available, falls back to
        direct state_dict loading for legacy checkpoints with explicit
        error handling.
        """
        import os

        allow_fresh = bool(getattr(self.config, "allow_fresh_weights", False))

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
                expected_class = self.model.__class__.__name__
                state_dict, metadata = manager.load_checkpoint(
                    model_path,
                    strict=True,
                    expected_version=expected_version,
                    expected_class=expected_class,
                    verify_checksum=True,
                    device=self.device,
                )
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
        self._ensure_model_initialized(game_state.board.type)

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
            self._ensure_model_initialized(first_type)

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

        with torch.no_grad():
            assert self.model is not None
            out = self.model(tensor_input, globals_input)
            # V3 models return (values, policy_logits, rank_dist). Keep the
            # rank distribution for training-only use and ignore it here.
            if isinstance(out, tuple) and len(out) == 3:
                values, policy_logits, _rank_dist = out
            else:
                values, policy_logits = out

            # Apply softmax to logits to get probabilities for MCTS / Descent.
            policy_probs = torch.softmax(policy_logits, dim=1)

        # NOTE: RingRiftCNN v2/v3 models output a multi-value head by default.
        # The NeuralNetAI wrapper (and all search AIs) currently consume a
        # *single scalar* value per state. We treat the first value head as
        # the canonical scalar value and ignore any additional heads.
        values_np = values.detach().cpu().numpy()
        if values_np.ndim == 2:
            scalar_values = values_np[:, 0]
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
        Returns (stacked_features[C,H,W], globals[10]) compatible with
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

        The encoding uses a fixed MAX_N × MAX_N = 19 × 19 canonical grid so
        that a single policy head can serve 8×8, 19×19, and hexagonal boards.
        Moves that require coordinates outside this canonical grid return
        INVALID_MOVE_INDEX and are simply omitted from the policy.

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

        # Pre-compute layout constants from MAX_N to avoid hard-coded offsets.
        placement_span = 3 * MAX_N * MAX_N  # 0..1082
        movement_base = placement_span  # 1083
        movement_span = MAX_N * MAX_N * (8 * (MAX_N - 1))
        line_base = movement_base + movement_span  # 53067
        line_span = MAX_N * MAX_N * 4
        territory_base = line_base + line_span  # 54511
        skip_index = territory_base + MAX_N * MAX_N  # 54872
        swap_sides_index = skip_index + 1  # 54873

        # Placement: 0..1082 (3 * 19 * 19)
        if move.type == "place_ring":
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
        if move.type in [
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
        if move.type == "line_formation":
            if board is not None:
                cx, cy = _to_canonical_xy(board, move.to)
            else:
                cx, cy = move.to.x, move.to.y
            if not (0 <= cx < MAX_N and 0 <= cy < MAX_N):
                return INVALID_MOVE_INDEX
            pos_idx = cy * MAX_N + cx
            # We currently ignore direction and always use dir_idx = 0, but
            # keep the 4-way slot layout for backward compatibility.
            return line_base + pos_idx * 4

        # Territory: 54511..54871
        if move.type == "territory_claim":
            if board is not None:
                cx, cy = _to_canonical_xy(board, move.to)
            else:
                cx, cy = move.to.x, move.to.y
            if not (0 <= cx < MAX_N and 0 <= cy < MAX_N):
                return INVALID_MOVE_INDEX
            pos_idx = cy * MAX_N + cx
            return territory_base + pos_idx

        # Skip placement: single terminal index
        if move.type == "skip_placement":
            return skip_index

        # Swap-sides (pie rule) decision. The decode_move implementation
        # already treats the index directly above skip_index as a canonical
        # SWAP_SIDES action; wiring encode_move to emit the same index ensures
        # that recorded swap_sides moves are represented in the policy head
        # and can be learned from training data.
        if move.type == "swap_sides":
            return swap_sides_index

        # Skip recovery (RR-CANON-R112): player elects not to perform recovery action
        skip_recovery_index = swap_sides_index + 1  # 54874 (sq19) / 4098 (sq8)
        if move.type == "skip_recovery":
            return skip_recovery_index

        # Choice moves: line and territory decision options
        # Line choices: 4 slots (options 0-3, typically option 1 = partial, 2 = full)
        line_choice_base = skip_recovery_index + 1  # 54875 (sq19) / 4099 (sq8)

        if move.type in ("choose_line_reward", "choose_line_option"):
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

        if move.type == "choose_territory_option":
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

        return None

    def _extract_features(
        self,
        game_state: GameState,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert game state to feature tensor for CNN and global features.

        Returns:
            (board_features, global_features)

        The board_features tensor has shape
        (10, board_size, board_size), where board_size is derived from the
        logical board via _infer_board_size so that this encoder works for
        8×8, 19×19, and hexagonal boards.
        """
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
        # Phase (5), Rings in hand (2), Eliminated rings (2), Turn (1), Reserved (10)
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

        # Rings info
        my_player = next(
            (p for p in game_state.players if p.player_number == game_state.current_player),
            None,
        )
        opp_player = next(
            (p for p in game_state.players if p.player_number != game_state.current_player),
            None,
        )

        ring_norm = 48.0  # hex supply per player
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

        Returns (stacked_features[64, H, W], globals[10]) for V3 models.

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
        # Hex-specific action space dimension.
        self.policy_size = policy_size or P_HEX

    def _encode_canonical_xy(
        self,
        board: BoardState,
        pos: Position,
    ) -> Optional[tuple[int, int]]:
        """Return (cx, cy) in [0, 21)×[0, 21) or None if off-grid."""
        cx, cy = _to_canonical_xy(board, pos)
        if not (0 <= cx < HEX_BOARD_SIZE and 0 <= cy < HEX_BOARD_SIZE):
            return None
        return cx, cy

    def encode_move(self, move: Move, board: BoardState) -> int:
        """Map a hex move into a [0, policy_size) index.

        This encoder is only valid for BoardType.HEXAGONAL boards of the
        canonical radius (size == 13, radius == 12). For any non-hex geometry or
        geometries that do not match the canonical frame, the move is
        treated as unrepresentable and INVALID_MOVE_INDEX is returned.
        """
        if board.type != BoardType.HEXAGONAL:
            return INVALID_MOVE_INDEX

        # Defensive guard: only support the canonical N=12 hex for now.
        if _infer_board_size(board) != HEX_BOARD_SIZE:
            return INVALID_MOVE_INDEX

        # --- Placements ---
        if move.type == MoveType.PLACE_RING:
            canon = self._encode_canonical_xy(board, move.to)
            if canon is None:
                return INVALID_MOVE_INDEX
            cx, cy = canon

            pos_idx = cy * HEX_BOARD_SIZE + cx
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
            if dist == 0 or dist > HEX_MAX_DIST:
                return INVALID_MOVE_INDEX

            dir_x = dx // dist
            dir_y = dy // dist
            try:
                dir_idx = HEX_DIRS.index((dir_x, dir_y))
            except ValueError:
                # Direction not representable in our 6-direction scheme.
                return INVALID_MOVE_INDEX

            from_idx = from_cy * HEX_BOARD_SIZE + from_cx
            return HEX_MOVEMENT_BASE + from_idx * (NUM_HEX_DIRS * HEX_MAX_DIST) + dir_idx * HEX_MAX_DIST + (dist - 1)

        # --- Special ---
        if move.type == MoveType.SKIP_PLACEMENT:
            return HEX_SPECIAL_BASE

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

        if board.type != BoardType.HEXAGONAL:
            return None
        if _infer_board_size(board) != HEX_BOARD_SIZE:
            return None
        if index < 0 or index >= self.policy_size:
            return None

        # --- Placements ---
        if index < HEX_PLACEMENT_SPAN:
            count_idx = index % 3
            pos_idx = index // 3
            cy = pos_idx // HEX_BOARD_SIZE
            cx = pos_idx % HEX_BOARD_SIZE

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
        if index < HEX_SPECIAL_BASE:
            offset = index - HEX_MOVEMENT_BASE

            dist_idx = offset % HEX_MAX_DIST
            dist = dist_idx + 1
            offset //= HEX_MAX_DIST

            dir_idx = offset % NUM_HEX_DIRS
            offset //= NUM_HEX_DIRS

            from_idx = offset
            from_cy = from_idx // HEX_BOARD_SIZE
            from_cx = from_idx % HEX_BOARD_SIZE

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
        if index == HEX_SPECIAL_BASE:
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
        in_channels: int = 56,  # 14 base × 4 frames
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
        in_channels: int = 64,  # 14 base × 4 frames + 8 phase/chain planes
        global_features: int = 20,
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
        policy = torch.full((B, self.policy_size), -1e9, device=device, dtype=dtype)

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

        policy = torch.full((B, self.policy_size), -1e9, device=device, dtype=dtype)

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
    "get_cached_model_count",
    "get_policy_size_for_board",
    "get_spatial_size_for_board",
]

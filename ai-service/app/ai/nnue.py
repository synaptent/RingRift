"""NNUE (Efficiently Updatable Neural Network) evaluation for RingRift Minimax.

This module implements a lightweight neural network evaluator inspired by
Stockfish NNUE, designed for fast CPU inference during alpha-beta search.

Architecture Overview:
- Small network (~256 hidden units) optimized for CPU inference
- Sparse input features (12 planes per position)
- ClippedReLU activations for efficient integer inference
- Output in [-1, 1] scaled to centipawn-like score for Minimax

The NNUE model is gated on:
- config.difficulty >= 4 (D4 is the neural Minimax tier)
- config.use_neural_net != False
- Model checkpoint availability

Memory Management:
Uses a singleton model cache similar to neural_net.py to share model
instances and prevent memory leaks in long-running sessions.
"""

import contextlib
import gc
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ..models import BoardType, GameState, Position
from ..rules.mutable_state import MutableGameState
from ..utils.torch_utils import safe_load_checkpoint

logger = logging.getLogger(__name__)

# =============================================================================
# Model Cache
# =============================================================================

_NNUE_CACHE: dict[tuple[str, str], nn.Module] = {}


def clear_nnue_cache() -> None:
    """Clear the NNUE model cache and release memory."""
    cache_size = len(_NNUE_CACHE)

    for model in _NNUE_CACHE.values():
        with contextlib.suppress(Exception):
            model.cpu()

    _NNUE_CACHE.clear()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        with contextlib.suppress(Exception):
            torch.mps.empty_cache()

    gc.collect()

    if cache_size > 0:
        logger.info(f"Cleared NNUE cache ({cache_size} models)")


# =============================================================================
# Feature Dimensions
# =============================================================================

# Input features: 26 spatial planes per position + 32 global features
#
# Spatial planes (26 total):
# - Planes 0-3: Ring presence per player (4 planes for 4 players max)
# - Planes 4-7: Stack height per player (4 planes, normalized by 5.0)
# - Planes 8-11: Territory ownership per player (4 planes)
# - Planes 12-15: Marker presence per player (4 planes) [V3]
# - Planes 16-19: Cap height per player (4 planes, normalized by 5.0) [V3]
# - Planes 20-23: Line threat per direction (4 planes: H, V, diag, anti-diag) [V3]
# - Planes 24-25: Capture threat map (vulnerability, opportunity) [V3]
#
# Global features (32 total):
# - Indices 0-3: Rings in hand per player (4 values, normalized 0-1)
# - Indices 4-11: Game phase one-hot (8 values for 8 phases)
# - Indices 12-15: Eliminated rings per player (4 values, normalized 0-1)
# - Indices 16-19: Territory count per player (4 values, normalized 0-1)
# - Indices 20-23: Victory proximity per player (4 values) [V3]
# - Index 24: Board occupancy (stacks / total_spaces) [V3]
# - Index 25: Marker density (markers / total_spaces) [V3]
# - Index 26: Turn number normalized (turn / 100) [V3]
# - Indices 27-30: Ring supply ratio per player (rings / total_remaining) [V3]
# - Index 31: Reserved [V3]

# V2 constants (backwards compatible)
FEATURE_PLANES_V2 = 12
GLOBAL_FEATURES_V2 = 20

# V3 constants (new features)
FEATURE_PLANES = 26  # 12 original + 4 marker + 4 cap height + 4 line threat + 2 capture threat
GLOBAL_FEATURES = 32  # 20 original + 12 enhanced

BOARD_SIZES: dict[BoardType, int] = {
    BoardType.SQUARE8: 8,
    BoardType.SQUARE19: 19,
    BoardType.HEXAGONAL: 25,  # Embedded in 25x25 grid
    BoardType.HEX8: 9,        # Radius 4, embedded in 9x9 grid
}

# V2 dimensions (backwards compatible: 12 spatial planes + 20 global)
FEATURE_DIMS_V2: dict[BoardType, int] = {
    BoardType.SQUARE8: 8 * 8 * FEATURE_PLANES_V2 + GLOBAL_FEATURES_V2,      # 768 + 20 = 788
    BoardType.SQUARE19: 19 * 19 * FEATURE_PLANES_V2 + GLOBAL_FEATURES_V2,   # 4332 + 20 = 4352
    BoardType.HEXAGONAL: 25 * 25 * FEATURE_PLANES_V2 + GLOBAL_FEATURES_V2,  # 7500 + 20 = 7520
    BoardType.HEX8: 9 * 9 * FEATURE_PLANES_V2 + GLOBAL_FEATURES_V2,         # 972 + 20 = 992
}

# V3 dimensions (new features: 26 spatial planes + 32 global)
FEATURE_DIMS: dict[BoardType, int] = {
    BoardType.SQUARE8: 8 * 8 * FEATURE_PLANES + GLOBAL_FEATURES,      # 1664 + 32 = 1696
    BoardType.SQUARE19: 19 * 19 * FEATURE_PLANES + GLOBAL_FEATURES,   # 9386 + 32 = 9418
    BoardType.HEXAGONAL: 25 * 25 * FEATURE_PLANES + GLOBAL_FEATURES,  # 16250 + 32 = 16282
    BoardType.HEX8: 9 * 9 * FEATURE_PLANES + GLOBAL_FEATURES,         # 2106 + 32 = 2138
}

# =============================================================================
# Feature Versioning
# =============================================================================
# Feature version tracks input encoding changes (separate from architecture version)
# V1: Spatial features only (768 for SQUARE8) - 12 planes, 0 global
# V2: Spatial + 20 global features (788 for SQUARE8) - 12 planes, 20 global
# V3: Enhanced features (1696 for SQUARE8) - 26 planes, 32 global [Dec 22, 2025]
#     Added: marker planes, cap height, line threats, capture threats, victory proximity
NNUE_FEATURE_V1 = 1
NNUE_FEATURE_V2 = 2
NNUE_FEATURE_V3 = 3
CURRENT_NNUE_FEATURE_VERSION = NNUE_FEATURE_V3

# Spatial-only dimensions (V1 feature encoding)
SPATIAL_DIMS_V1: dict[BoardType, int] = {
    BoardType.SQUARE8: 8 * 8 * FEATURE_PLANES_V2,      # 768
    BoardType.SQUARE19: 19 * 19 * FEATURE_PLANES_V2,   # 4332
    BoardType.HEXAGONAL: 25 * 25 * FEATURE_PLANES_V2,  # 7500
    BoardType.HEX8: 9 * 9 * FEATURE_PLANES_V2,         # 972
}

# V2 spatial dimensions (for backwards compatibility detection)
SPATIAL_DIMS: dict[BoardType, int] = SPATIAL_DIMS_V1  # Alias for V2 detection


def get_feature_dim_for_version(board_type: BoardType, feature_version: int) -> int:
    """Get feature dimension for a specific feature version.

    Args:
        board_type: Board type
        feature_version: 1 for V1 (spatial only), 2 for V2, 3 for V3 (enhanced)

    Returns:
        Feature dimension for the given version
    """
    if feature_version >= NNUE_FEATURE_V3:
        return FEATURE_DIMS.get(board_type, FEATURE_DIMS[BoardType.SQUARE8])
    elif feature_version >= NNUE_FEATURE_V2:
        return FEATURE_DIMS_V2.get(board_type, FEATURE_DIMS_V2[BoardType.SQUARE8])
    else:
        return SPATIAL_DIMS_V1.get(board_type, SPATIAL_DIMS_V1[BoardType.SQUARE8])


def detect_feature_version_from_accumulator(
    acc_weight_shape: tuple[int, ...],
    board_type: BoardType,
) -> int:
    """Detect feature version from accumulator weight shape.

    Args:
        acc_weight_shape: Shape of accumulator.weight tensor (hidden_dim, input_dim)
        board_type: Board type for the model

    Returns:
        Detected feature version (1, 2, or 3)
    """
    input_size = acc_weight_shape[1] if len(acc_weight_shape) >= 2 else 0
    spatial_v1 = SPATIAL_DIMS_V1.get(board_type, SPATIAL_DIMS_V1[BoardType.SQUARE8])
    v2_size = FEATURE_DIMS_V2.get(board_type, FEATURE_DIMS_V2[BoardType.SQUARE8])
    v3_size = FEATURE_DIMS.get(board_type, FEATURE_DIMS[BoardType.SQUARE8])

    if input_size == spatial_v1:
        return NNUE_FEATURE_V1
    elif input_size == v2_size:
        return NNUE_FEATURE_V2
    elif input_size == v3_size:
        return NNUE_FEATURE_V3
    else:
        # Unknown size - assume latest version
        logger.warning(f"Unknown accumulator input size {input_size} for {board_type}, assuming V{CURRENT_NNUE_FEATURE_VERSION}")
        return CURRENT_NNUE_FEATURE_VERSION


def get_feature_dim(board_type: BoardType) -> int:
    """Get the input feature dimension for a board type."""
    return FEATURE_DIMS.get(board_type, FEATURE_DIMS[BoardType.SQUARE8])


def get_board_size(board_type: BoardType) -> int:
    """Get the spatial size for a board type."""
    return BOARD_SIZES.get(board_type, 8)


# =============================================================================
# NNUE Model Architecture
# =============================================================================

class ClippedReLU(nn.Module):
    """ClippedReLU activation: clamp(x, 0, 1).

    Used in NNUE for efficient quantized inference (values in [0, 127] range
    when using int8 weights). For float inference, we use [0, 1] range.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, 1.0)


class StochasticDepthLayer(nn.Module):
    """Stochastic Depth: randomly skip layers during training.

    Implements 'Deep Networks with Stochastic Depth' (Huang et al., 2016).
    During training, the layer is skipped with probability p, and during
    inference it's always applied with scaled output.

    Args:
        p: Drop probability (0=never skip, 1=always skip)
        mode: "row" for per-sample dropping, "batch" for entire batch
    """

    def __init__(self, p: float = 0.1, mode: str = "batch"):
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth to residual connection.

        Args:
            x: Input tensor (skip connection)
            residual: Output from the dropped layer

        Returns:
            x + scaled_residual (or just x if dropped)
        """
        if not self.training or self.p == 0.0:
            # During inference, apply with survival probability scaling
            return x + residual

        survival_prob = 1.0 - self.p

        if self.mode == "row":
            # Per-sample dropping: different samples may be dropped
            mask = torch.empty(x.shape[0], 1, device=x.device).bernoulli_(survival_prob)
            return x + residual * mask / survival_prob
        else:
            # Batch dropping: entire batch is dropped or not
            if torch.rand(1, device=x.device).item() < self.p:
                return x
            return x + residual / survival_prob


class ResidualBlock(nn.Module):
    """Residual block with optional stochastic depth.

    A basic residual block with skip connection and optional stochastic depth
    for regularization during training.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_spectral_norm: bool = False,
        stochastic_depth_prob: float = 0.0,
    ):
        super().__init__()

        def maybe_spectral_norm(layer: nn.Linear) -> nn.Linear:
            if use_spectral_norm:
                return nn.utils.spectral_norm(layer)
            return layer

        self.fc = maybe_spectral_norm(nn.Linear(in_dim, out_dim))
        self.activation = ClippedReLU()

        # Optional stochastic depth
        self.stochastic_depth = None
        if stochastic_depth_prob > 0:
            self.stochastic_depth = StochasticDepthLayer(p=stochastic_depth_prob)

        # Projection for dimension mismatch
        self.projection = None
        if in_dim != out_dim:
            self.projection = maybe_spectral_norm(nn.Linear(in_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.activation(self.fc(x))

        # Project input if dimensions don't match
        if self.projection is not None:
            x = self.projection(x)

        # Apply stochastic depth if enabled
        if self.stochastic_depth is not None:
            return self.stochastic_depth(x, residual)

        return x + residual


class RingRiftNNUE(nn.Module):
    """NNUE-style evaluation network for RingRift Minimax search.

    Architecture inspired by Stockfish NNUE with RingRift-specific features.
    Designed for efficient CPU inference during alpha-beta search.

    Input: Sparse feature vector (12 planes flattened)
    Output: Scalar value in [-1, 1]

    The network uses ClippedReLU activations and small hidden layers
    to enable fast inference without GPU.
    """

    ARCHITECTURE_VERSION = "v1.4.0"  # V3 features: markers, cap height, line threats, capture threats
    FEATURE_VERSION = CURRENT_NNUE_FEATURE_VERSION  # V3: 26 spatial planes + 32 global features

    def __init__(
        self,
        board_type: BoardType = BoardType.SQUARE8,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        use_spectral_norm: bool = False,
        use_batch_norm: bool = False,
        num_heads: int = 1,
        stochastic_depth_prob: float = 0.0,
    ):
        super().__init__()
        self.board_type = board_type
        self.use_spectral_norm = use_spectral_norm
        self.use_batch_norm = use_batch_norm
        self.num_heads = num_heads
        self.stochastic_depth_prob = stochastic_depth_prob
        input_dim = get_feature_dim(board_type)

        # Helper to optionally apply spectral normalization
        def maybe_spectral_norm(layer: nn.Linear) -> nn.Linear:
            if use_spectral_norm:
                return nn.utils.spectral_norm(layer)
            return layer

        # Multi-head feature projection
        if num_heads > 1:
            # Split features into num_heads groups (e.g., by player)
            # Each head projects its subset independently
            head_dim = hidden_dim // num_heads
            self.head_projections = nn.ModuleList([
                maybe_spectral_norm(nn.Linear(input_dim // num_heads, head_dim, bias=True))
                for _ in range(num_heads)
            ])
            self.accumulator = None  # Using head projections instead
            acc_output_dim = hidden_dim
        else:
            # Single accumulator (original behavior)
            self.accumulator = maybe_spectral_norm(nn.Linear(input_dim, hidden_dim, bias=True))
            self.head_projections = None
            acc_output_dim = hidden_dim

        # Optional batch normalization after accumulator
        self.acc_batch_norm = nn.BatchNorm1d(acc_output_dim) if use_batch_norm else None

        # Hidden layers with ClippedReLU and optional stochastic depth
        # Use progressive drop rate: earlier layers have lower drop rate
        self.hidden_blocks = nn.ModuleList()
        current_dim = acc_output_dim * 2  # Concatenate player perspectives

        for i in range(num_hidden_layers):
            out_dim = 32
            # Progressive drop rate: increases for deeper layers
            layer_drop_prob = stochastic_depth_prob * (i + 1) / num_hidden_layers if stochastic_depth_prob > 0 else 0.0

            block = ResidualBlock(
                in_dim=current_dim,
                out_dim=out_dim,
                use_spectral_norm=use_spectral_norm,
                stochastic_depth_prob=layer_drop_prob,
            )
            self.hidden_blocks.append(block)
            current_dim = out_dim

        # Legacy hidden attribute for backwards compatibility
        self.hidden = None  # Now using hidden_blocks

        # Output layer: single scalar value
        self.output = maybe_spectral_norm(nn.Linear(32, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Shape (batch, input_dim) sparse/dense input features

        Returns:
            Shape (batch, 1) values in [-1, 1]
        """
        # Multi-head or single accumulator projection
        if self.head_projections is not None:
            # Split features into chunks for each head
            chunk_size = features.shape[-1] // self.num_heads
            chunks = [features[..., i*chunk_size:(i+1)*chunk_size] for i in range(self.num_heads)]
            # Project each chunk through its head
            head_outputs = [proj(chunk) for proj, chunk in zip(self.head_projections, chunks, strict=False)]
            # Concatenate head outputs
            acc = torch.cat(head_outputs, dim=-1)
        else:
            # Single accumulator (original behavior)
            acc = self.accumulator(features)

        # Optional batch normalization before activation
        if self.acc_batch_norm is not None:
            acc = self.acc_batch_norm(acc)

        # ClippedReLU activation
        acc = torch.clamp(acc, 0.0, 1.0)

        # Concatenate "perspectives" - simplified version using same features
        # In full NNUE, we'd have separate accumulators for each player view
        x = torch.cat([acc, acc], dim=-1)

        # Hidden layers with residual blocks (optional stochastic depth)
        if self.hidden_blocks:
            for block in self.hidden_blocks:
                x = block(x)
        elif self.hidden is not None:
            # Legacy path for backwards compatibility
            x = self.hidden(x)

        # Output with tanh for [-1, 1] range
        return torch.tanh(self.output(x))

    def forward_with_hidden(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both value and hidden features.

        Useful for auxiliary task training that needs access to hidden
        representations.

        Args:
            features: Shape (batch, input_dim) sparse/dense input features

        Returns:
            Tuple of (values, hidden):
              - values: Shape (batch, 1) in [-1, 1]
              - hidden: Shape (batch, 32) last hidden layer features
        """
        # Multi-head or single accumulator projection
        if self.head_projections is not None:
            chunk_size = features.shape[-1] // self.num_heads
            chunks = [
                features[..., i * chunk_size : (i + 1) * chunk_size]
                for i in range(self.num_heads)
            ]
            head_outputs = [
                proj(chunk)
                for proj, chunk in zip(self.head_projections, chunks, strict=False)
            ]
            acc = torch.cat(head_outputs, dim=-1)
        else:
            acc = self.accumulator(features)

        if self.acc_batch_norm is not None:
            acc = self.acc_batch_norm(acc)

        acc = torch.clamp(acc, 0.0, 1.0)
        x = torch.cat([acc, acc], dim=-1)

        if self.hidden_blocks:
            for block in self.hidden_blocks:
                x = block(x)
        elif self.hidden is not None:
            x = self.hidden(x)

        hidden = x  # Save hidden features before output layer
        values = torch.tanh(self.output(x))
        return values, hidden

    def forward_single(self, features: np.ndarray) -> float:
        """Convenience method for single-sample inference.

        Args:
            features: Shape (input_dim,) numpy array

        Returns:
            Scalar value in [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features[None, ...]).float()
            device = next(self.parameters()).device
            x = x.to(device)
            value = self.forward(x)
        return float(value.cpu().item())

    def quantize_dynamic(self) -> "RingRiftNNUE":
        """Apply dynamic int8 quantization for faster CPU inference.

        Returns:
            Quantized model (self, modified in-place)
        """
        self.cpu()
        self.eval()

        # Dynamic quantization: quantizes weights to int8, activations at runtime
        quantized = torch.quantization.quantize_dynamic(
            self,
            {nn.Linear},  # Quantize Linear layers
            dtype=torch.qint8,
        )

        logger.info("Applied dynamic int8 quantization to NNUE model")
        return quantized

    @classmethod
    def from_quantized_checkpoint(
        cls,
        checkpoint_path: str,
        board_type: BoardType = BoardType.SQUARE8,
    ) -> "RingRiftNNUE":
        """Load a quantized model from checkpoint.

        Args:
            checkpoint_path: Path to quantized model checkpoint
            board_type: Board type for the model

        Returns:
            Quantized NNUE model
        """
        checkpoint = safe_load_checkpoint(checkpoint_path, map_location="cpu", warn_on_unsafe=False)

        # Create base model
        hidden_dim = checkpoint.get("hidden_dim", 256)
        num_hidden_layers = checkpoint.get("num_hidden_layers", 2)
        model = cls(
            board_type=board_type,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )

        # Load state dict (may need to handle quantized format)
        if "quantized_state_dict" in checkpoint:
            # Load quantized weights
            model.load_state_dict(checkpoint["quantized_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()
        return model


# =============================================================================
# Feature Extraction
# =============================================================================

def _pos_to_index(pos: Position, board_size: int, board_type: BoardType) -> int:
    """Convert position to linear index in feature vector."""
    if board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
        # Hex uses axial coordinates, offset by radius
        radius = (board_size - 1) // 2
        cx = pos.x + radius
        cy = pos.y + radius
        return cy * board_size + cx
    else:
        return pos.y * board_size + pos.x


def _rotate_player_perspective(owner: int, player_number: int, num_players: int = 4) -> int:
    """Rotate player number so that player_number becomes player 1.

    This ensures the current player's features are always in plane 0,
    making the evaluation symmetric/perspective-aware.

    Example with player_number=2:
        owner=2 -> 1 (current player)
        owner=1 -> 4 (opponent, wraps around)
        owner=3 -> 2 (other opponent)
        owner=4 -> 3 (other opponent)
    """
    if owner == 0 or owner == player_number:
        return 1 if owner == player_number else owner
    # Rotate: map owner to (owner - player_number + 1) mod num_players, keeping 1-indexed
    rotated = ((owner - player_number) % num_players) + 1
    return rotated


def _compute_line_length(markers: dict, pos_key: str, player: int, dx: int, dy: int, board_size: int) -> int:
    """Compute line length in a direction from a marker position."""
    parts = pos_key.split(",")
    if len(parts) < 2:
        return 1
    x, y = int(parts[0]), int(parts[1])
    length = 1  # Count self

    # Count forward
    for step in range(1, 6):  # Max line length is 5+
        nx, ny = x + dx * step, y + dy * step
        neighbor_key = f"{nx},{ny}"
        if neighbor_key in markers:
            m = markers[neighbor_key]
            if getattr(m, 'player', 0) == player:
                length += 1
            else:
                break
        else:
            break

    # Count backward
    for step in range(1, 6):
        nx, ny = x - dx * step, y - dy * step
        neighbor_key = f"{nx},{ny}"
        if neighbor_key in markers:
            m = markers[neighbor_key]
            if getattr(m, 'player', 0) == player:
                length += 1
            else:
                break
        else:
            break

    return length


def extract_features_from_gamestate(
    state: GameState,
    player_number: int,
) -> np.ndarray:
    """Extract NNUE V3 features from immutable GameState.

    Feature planes (26 total) - PERSPECTIVE ROTATED:
    - Planes 0-3: Ring presence (plane 0 = current player, 1-3 = opponents)
    - Planes 4-7: Stack height (normalized by 5.0)
    - Planes 8-11: Territory ownership
    - Planes 12-15: Marker presence [V3]
    - Planes 16-19: Cap height (normalized by 5.0) [V3]
    - Planes 20-23: Line threat per direction (H, V, diag, anti-diag) [V3]
    - Planes 24-25: Capture threat (vulnerability, opportunity) [V3]

    Global features (32 total):
    - Indices 0-3: Rings in hand per player
    - Indices 4-11: Game phase one-hot (8 phases)
    - Indices 12-15: Eliminated rings per player
    - Indices 16-19: Territory count per player
    - Indices 20-23: Victory proximity per player [V3]
    - Index 24: Board occupancy [V3]
    - Index 25: Marker density [V3]
    - Index 26: Turn number normalized [V3]
    - Indices 27-30: Ring supply ratio [V3]
    - Index 31: Reserved [V3]

    Args:
        state: The game state to extract features from
        player_number: The player perspective (1-4)

    Returns:
        Flattened feature vector of shape (FEATURE_DIMS[board_type],)
    """
    board = state.board
    board_type = board.type
    board_size = get_board_size(board_type)
    num_positions = board_size * board_size
    num_players = len(state.players) if hasattr(state, 'players') else 2

    # Initialize feature planes (26 planes for V3)
    features = np.zeros(num_positions * FEATURE_PLANES, dtype=np.float32)

    # Collect stacks info for capture threat computation
    stacks_by_pos: dict[str, tuple[int, int, int]] = {}  # pos_key -> (owner, height, cap_height)

    # Extract ring/stack/cap_height from board.stacks
    for pos_key, ring_stack in (board.stacks or {}).items():
        try:
            parts = pos_key.split(",")
            if len(parts) >= 2:
                x, y = int(parts[0]), int(parts[1])
                z = int(parts[2]) if len(parts) == 3 else None
                pos = Position(x=x, y=y, z=z)
                idx = _pos_to_index(pos, board_size, board_type)

                if 0 <= idx < num_positions:
                    # Ring features (planes 0-3) - ROTATED perspective
                    rings = getattr(ring_stack, 'rings', [])
                    for ring_owner in rings:
                        if 1 <= ring_owner <= 4:
                            rotated = _rotate_player_perspective(ring_owner, player_number, num_players)
                            plane_idx = (rotated - 1) * num_positions
                            features[plane_idx + idx] = 1.0

                    # Stack height features (planes 4-7) - ROTATED perspective
                    owner = getattr(ring_stack, 'controlling_player', 0)
                    height = getattr(ring_stack, 'stack_height', 0)
                    cap_height = getattr(ring_stack, 'cap_height', 0)

                    if 1 <= owner <= 4 and height > 0:
                        rotated = _rotate_player_perspective(owner, player_number, num_players)
                        # Stack height (planes 4-7)
                        plane_idx = (4 + rotated - 1) * num_positions
                        features[plane_idx + idx] = min(height / 5.0, 1.0)

                        # Cap height (planes 16-19) [V3]
                        plane_idx = (16 + rotated - 1) * num_positions
                        features[plane_idx + idx] = min(cap_height / 5.0, 1.0)

                        # Store for capture threat computation
                        stacks_by_pos[pos_key] = (owner, height, cap_height)
        except (ValueError, AttributeError):
            continue

    # Extract territory features (planes 8-11) - ROTATED perspective
    for _territory_key, territory in (board.territories or {}).items():
        try:
            pnum = getattr(territory, 'player', 0)
            if 1 <= pnum <= 4:
                rotated = _rotate_player_perspective(pnum, player_number, num_players)
                for pos in (getattr(territory, 'spaces', None) or []):
                    idx = _pos_to_index(pos, board_size, board_type)
                    if 0 <= idx < num_positions:
                        plane_idx = (8 + rotated - 1) * num_positions
                        features[plane_idx + idx] = 1.0
        except (ValueError, AttributeError):
            continue

    # Extract marker features (planes 12-15) [V3] and compute line threats
    markers = getattr(board, 'markers', None) or {}
    marker_count = 0
    line_directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # H, V, diag, anti-diag

    for pos_key, marker in markers.items():
        try:
            parts = pos_key.split(",")
            if len(parts) >= 2:
                x, y = int(parts[0]), int(parts[1])
                z = int(parts[2]) if len(parts) == 3 else None
                pos = Position(x=x, y=y, z=z)
                idx = _pos_to_index(pos, board_size, board_type)
                marker_owner = getattr(marker, 'player', 0)

                if 0 <= idx < num_positions and 1 <= marker_owner <= 4:
                    marker_count += 1
                    rotated = _rotate_player_perspective(marker_owner, player_number, num_players)

                    # Marker presence (planes 12-15)
                    plane_idx = (12 + rotated - 1) * num_positions
                    features[plane_idx + idx] = 1.0

                    # Line threat detection (planes 20-23) - only for current player
                    if rotated == 1:  # Current player's markers
                        for dir_idx, (dx, dy) in enumerate(line_directions):
                            line_len = _compute_line_length(markers, pos_key, marker_owner, dx, dy, board_size)
                            # Normalize: 5+ is max (forms a capture line)
                            plane_idx = (20 + dir_idx) * num_positions
                            features[plane_idx + idx] = max(features[plane_idx + idx], min(line_len / 5.0, 1.0))
        except (ValueError, AttributeError):
            continue

    # Capture threat map (planes 24-25) [V3]
    # Simplified: check for stacks that can be captured based on cap height
    for pos_key, (owner, height, cap_height) in stacks_by_pos.items():
        try:
            parts = pos_key.split(",")
            if len(parts) < 2:
                continue
            x, y = int(parts[0]), int(parts[1])
            idx = y * board_size + x
            if idx < 0 or idx >= num_positions:
                continue

            rotated_owner = _rotate_player_perspective(owner, player_number, num_players)

            # Check adjacent stacks for capture threats
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                neighbor_key = f"{x + dx},{y + dy}"
                if neighbor_key in stacks_by_pos:
                    n_owner, n_height, n_cap = stacks_by_pos[neighbor_key]
                    if n_owner != owner:  # Enemy stack
                        if rotated_owner == 1:  # Our stack
                            if n_cap >= cap_height:
                                # We are vulnerable (plane 24)
                                features[24 * num_positions + idx] = max(
                                    features[24 * num_positions + idx],
                                    min((n_cap - cap_height + 1) / 5.0, 1.0)
                                )
                            if cap_height >= n_cap:
                                # We can capture (plane 25)
                                features[25 * num_positions + idx] = max(
                                    features[25 * num_positions + idx],
                                    min((cap_height - n_cap + 1) / 5.0, 1.0)
                                )
        except (ValueError, AttributeError):
            continue

    # Add global features (32 total for V3)
    global_features = np.zeros(GLOBAL_FEATURES, dtype=np.float32)

    # Rings in hand per player (indices 0-3)
    initial_rings = 18 if num_players == 2 else (12 if num_players == 3 else 9)
    total_rings_remaining = 0
    for p in range(1, min(num_players + 1, 5)):
        rotated = _rotate_player_perspective(p, player_number, num_players)
        player_data = state.players[p - 1] if hasattr(state, 'players') and p <= len(state.players) else None
        rings_in_hand = getattr(player_data, 'rings_in_hand', initial_rings) if player_data else initial_rings
        global_features[rotated - 1] = rings_in_hand / initial_rings
        total_rings_remaining += rings_in_hand

    # Game phase one-hot encoding (indices 4-11)
    phase_map = {
        'ring_placement': 0, 'movement': 1, 'capture': 2, 'chain_capture': 2,
        'line_processing': 3, 'territory_processing': 4, 'forced_elimination': 5,
        'game_over': 6, 'recovery': 1,
    }
    phase_name = getattr(state, 'phase', None)
    if phase_name:
        phase_str = phase_name.value if hasattr(phase_name, 'value') else str(phase_name).lower()
        phase_idx = phase_map.get(phase_str, 7)
        global_features[4 + phase_idx] = 1.0
    else:
        global_features[4] = 1.0

    # Eliminated rings per player (indices 12-15)
    victory_threshold = 3 if num_players == 2 else (2 if num_players <= 3 else 1)
    elimination_threshold = 12

    for p in range(1, min(num_players + 1, 5)):
        rotated = _rotate_player_perspective(p, player_number, num_players)
        player_data = state.players[p - 1] if hasattr(state, 'players') and p <= len(state.players) else None
        eliminated = getattr(player_data, 'eliminated_rings', 0) if player_data else 0
        territory_count = getattr(player_data, 'territory_spaces', 0) if player_data else 0

        global_features[12 + rotated - 1] = min(eliminated / 5.0, 1.0)

        # Territory count (indices 16-19)
        global_features[16 + rotated - 1] = min(territory_count / (victory_threshold * 2), 1.0)

        # Victory proximity (indices 20-23) [V3]
        territory_progress = territory_count / victory_threshold if victory_threshold > 0 else 0
        elim_progress = eliminated / elimination_threshold
        victory_prox = max(territory_progress, elim_progress)
        global_features[20 + rotated - 1] = min(victory_prox, 1.0)

        # Ring supply ratio (indices 27-30) [V3]
        rings_in_hand = getattr(player_data, 'rings_in_hand', 0) if player_data else 0
        if total_rings_remaining > 0:
            global_features[27 + rotated - 1] = rings_in_hand / total_rings_remaining

    # Board occupancy (index 24) [V3]
    global_features[24] = len(stacks_by_pos) / num_positions

    # Marker density (index 25) [V3]
    global_features[25] = marker_count / num_positions

    # Turn number normalized (index 26) [V3]
    turn_count = getattr(state, 'turn_count', 0) or getattr(state, 'move_count', 0) or 0
    global_features[26] = min(turn_count / 100.0, 1.0)

    # Index 31: Reserved for future use

    # Concatenate spatial and global features
    return np.concatenate([features, global_features])


def extract_features_from_mutable(
    state: MutableGameState,
    player_number: int,
) -> np.ndarray:
    """Extract NNUE features from MutableGameState.

    Optimized feature extraction that works directly with MutableGameState
    to avoid conversion overhead during search.

    Feature planes (12 spatial + 20 global) - PERSPECTIVE ROTATED:
    - Planes 0-3: Ring presence (plane 0 = current player, 1-3 = opponents)
    - Planes 4-7: Stack presence (plane 4 = current player, 5-7 = opponents)
    - Planes 8-11: Territory ownership (plane 8 = current player, 9-11 = opponents)
    - Global 0-3: Rings in hand per player (normalized)
    - Global 4-11: Game phase one-hot encoding
    - Global 12-15: Eliminated rings per player (normalized)
    - Global 16-19: Territory count per player (normalized)

    Args:
        state: The mutable game state
        player_number: The player perspective (1-4)

    Returns:
        Flattened feature vector of shape (board_size * board_size * 12 + 20,)
    """
    board_type = state.board_type
    board_size = get_board_size(board_type)
    num_positions = board_size * board_size
    num_players = getattr(state, 'num_players', 2)

    # Initialize feature planes
    features = np.zeros(num_positions * FEATURE_PLANES, dtype=np.float32)

    # Access internal state arrays from MutableGameState
    # These are optimized numpy arrays for fast access
    rings = getattr(state, '_rings', None)
    stacks = getattr(state, '_stacks', None)
    territories = getattr(state, '_territories', None)

    # Fallback to conversion if internal arrays not available
    if rings is None:
        immutable = state.to_immutable()
        return extract_features_from_gamestate(immutable, player_number)

    # Extract ring features from internal arrays - ROTATED perspective
    if rings is not None:
        for y in range(board_size):
            for x in range(board_size):
                idx = y * board_size + x
                owner = int(rings[y, x]) if rings.ndim >= 2 else 0
                if 1 <= owner <= 4:
                    # Rotate so current player is always plane 0
                    rotated = _rotate_player_perspective(
                        owner, player_number, num_players
                    )
                    plane_idx = (rotated - 1) * num_positions
                    features[plane_idx + idx] = 1.0

    # Extract stack features - ROTATED perspective
    if stacks is not None:
        for y in range(board_size):
            for x in range(board_size):
                idx = y * board_size + x
                # Stacks array format: [y, x, 0] = owner, [y, x, 1] = height
                if stacks.ndim >= 3:
                    owner = int(stacks[y, x, 0])
                    height = int(stacks[y, x, 1])
                else:
                    owner = int(stacks[y, x]) if stacks.ndim >= 2 else 0
                    height = 1
                if 1 <= owner <= 4 and height > 0:
                    rotated = _rotate_player_perspective(
                        owner, player_number, num_players
                    )
                    plane_idx = (4 + rotated - 1) * num_positions
                    features[plane_idx + idx] = min(height / 5.0, 1.0)

    # Extract territory features - ROTATED perspective
    if territories is not None:
        for y in range(board_size):
            for x in range(board_size):
                idx = y * board_size + x
                owner = int(territories[y, x]) if territories.ndim >= 2 else 0
                if 1 <= owner <= 4:
                    rotated = _rotate_player_perspective(
                        owner, player_number, num_players
                    )
                    plane_idx = (8 + rotated - 1) * num_positions
                    features[plane_idx + idx] = 1.0

    # Add global features (20 total) - CRITICAL for early-game evaluation
    global_features = np.zeros(GLOBAL_FEATURES, dtype=np.float32)

    # Rings in hand per player (4 values, normalized by initial ring count)
    initial_rings = 18 if num_players == 2 else (12 if num_players == 3 else 9)
    rings_in_hand_arr = getattr(state, '_rings_in_hand', None)
    for p in range(1, min(num_players + 1, 5)):
        rotated = _rotate_player_perspective(p, player_number, num_players)
        if rings_in_hand_arr is not None and p <= len(rings_in_hand_arr):
            rings_in_hand = int(rings_in_hand_arr[p - 1])
        else:
            # Fallback: count rings not on board
            rings_in_hand = initial_rings
        global_features[rotated - 1] = rings_in_hand / initial_rings

    # Game phase one-hot encoding (8 values for 8 canonical phases)
    phase_map = {
        'ring_placement': 0, 'movement': 1, 'capture': 2, 'chain_capture': 2,
        'line_processing': 3, 'territory_processing': 4, 'forced_elimination': 5,
        'game_over': 6, 'recovery': 1,
    }
    phase_name = getattr(state, 'phase', None)
    if phase_name:
        phase_str = phase_name.value if hasattr(phase_name, 'value') else str(phase_name).lower()
        phase_idx = phase_map.get(phase_str, 7)
        global_features[4 + phase_idx] = 1.0
    else:
        global_features[4] = 1.0  # Default to ring_placement phase

    # Eliminated rings per player (4 values, normalized)
    eliminated_arr = getattr(state, '_eliminated_rings', None)
    for p in range(1, min(num_players + 1, 5)):
        rotated = _rotate_player_perspective(p, player_number, num_players)
        if eliminated_arr is not None and p <= len(eliminated_arr):
            eliminated = int(eliminated_arr[p - 1])
        else:
            eliminated = 0
        global_features[12 + rotated - 1] = min(eliminated / 5.0, 1.0)

    # Territory count per player (4 values, normalized by victory threshold)
    victory_threshold = 3 if num_players == 2 else (2 if num_players <= 3 else 1)
    territory_counts = getattr(state, '_territory_counts', None)
    for p in range(1, min(num_players + 1, 5)):
        rotated = _rotate_player_perspective(p, player_number, num_players)
        if territory_counts is not None and p <= len(territory_counts):
            territory_count = int(territory_counts[p - 1])
        else:
            # Fallback: count from territory array
            territory_count = 0
            if territories is not None:
                territory_count = int(np.sum(territories == p))
        global_features[16 + rotated - 1] = min(territory_count / (victory_threshold * 2), 1.0)

    # Concatenate spatial and global features
    return np.concatenate([features, global_features])


# =============================================================================
# Model Loading
# =============================================================================

def get_nnue_model_path(
    board_type: BoardType,
    num_players: int = 2,
    model_id: str | None = None,
) -> Path:
    """Get the default NNUE model checkpoint path.

    Args:
        board_type: The board type for model selection
        num_players: Number of players (2, 3, or 4)
        model_id: Optional specific model ID

    Returns:
        Path to the model checkpoint
    """
    models_dir = Path(__file__).parent.parent.parent / "models" / "nnue"

    if model_id:
        return models_dir / f"{model_id}.pt"

    # Default model naming: nnue_{board_type}_{num_players}p.pt
    board_name = board_type.value.lower()
    primary = models_dir / f"nnue_{board_name}_{num_players}p.pt"

    # Backwards-compatible fallback: older training runs published
    # ``nnue_{board}.pt`` for the default 2-player case.
    if num_players == 2 and not primary.exists():
        legacy = models_dir / f"nnue_{board_name}.pt"
        if legacy.exists():
            return legacy

    return primary


def _migrate_legacy_state_dict(
    state_dict: dict[str, torch.Tensor],
    architecture_version: str,
    target_input_size: int | None = None,
    board_type: BoardType | None = None,
) -> tuple[dict[str, torch.Tensor], int]:
    """Migrate a legacy state dict to the current architecture.

    Handles backwards compatibility for older model checkpoints.

    Args:
        state_dict: The legacy state dict
        architecture_version: Version string from checkpoint
        target_input_size: Expected input size (for padding accumulator weights)
        board_type: Board type for feature version detection

    Returns:
        Tuple of (migrated state dict, detected feature version)
    """
    new_state_dict = dict(state_dict)
    detected_feature_version = CURRENT_NNUE_FEATURE_VERSION

    # Detect feature version from accumulator weight shape
    if "accumulator.weight" in new_state_dict and board_type is not None:
        acc_shape = new_state_dict["accumulator.weight"].shape
        detected_feature_version = detect_feature_version_from_accumulator(acc_shape, board_type)

    # v1.0.0 and v1.1.0 used simple Sequential hidden layers:
    # hidden.0.weight/bias (Linear), hidden.2.weight/bias (Linear)
    # Current v1.3.0 uses ResidualBlocks: hidden_blocks.N.fc.weight/bias
    if architecture_version.startswith("v1.0") or architecture_version.startswith("v1.1"):
        migrated = {}
        for key, value in new_state_dict.items():
            if key.startswith("hidden."):
                # Map hidden.0 -> hidden_blocks.0.fc
                # Map hidden.2 -> hidden_blocks.1.fc
                parts = key.split(".")
                if len(parts) == 3:
                    layer_idx = int(parts[1])
                    param_type = parts[2]  # weight or bias
                    # hidden.0 -> block 0, hidden.2 -> block 1
                    block_idx = layer_idx // 2
                    new_key = f"hidden_blocks.{block_idx}.fc.{param_type}"
                    migrated[new_key] = value
                else:
                    migrated[key] = value
            else:
                migrated[key] = value
        new_state_dict = migrated
        logger.info(f"Migrated hidden layer structure from {architecture_version} to v1.3.0")

    # Handle accumulator weight size mismatch (768 -> 788 for global features)
    # Old models were trained without GLOBAL_FEATURES, new models include them
    if "accumulator.weight" in new_state_dict and target_input_size is not None:
        acc_weight = new_state_dict["accumulator.weight"]
        current_size = acc_weight.shape[1]
        if current_size < target_input_size:
            padding_size = target_input_size - current_size
            # Pad with zeros for the new global features
            padding = torch.zeros(acc_weight.shape[0], padding_size, device=acc_weight.device, dtype=acc_weight.dtype)
            new_state_dict["accumulator.weight"] = torch.cat([acc_weight, padding], dim=1)
            logger.info(
                f"Migrated NNUE features V{detected_feature_version} -> V{CURRENT_NNUE_FEATURE_VERSION}: "
                f"padded accumulator {current_size} -> {target_input_size} (+{padding_size} global features)"
            )

    return new_state_dict, detected_feature_version


def load_nnue_model(
    board_type: BoardType,
    num_players: int = 2,
    model_id: str | None = None,
    device: str | None = None,
    allow_fresh: bool = True,
) -> RingRiftNNUE | None:
    """Load an NNUE model, using cache when possible.

    Args:
        board_type: The board type for model selection
        num_players: Number of players (2, 3, or 4)
        model_id: Optional specific model ID
        device: Device to load model on ('cpu', 'cuda', 'mps')
        allow_fresh: If True and no checkpoint exists, return fresh weights

    Returns:
        Loaded NNUE model or None if unavailable
    """

    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Check cache - include num_players in cache key
    cache_key = (board_type.value, num_players, model_id or "default")
    if cache_key in _NNUE_CACHE:
        model = _NNUE_CACHE[cache_key]
        return model.to(device)

    # Try to load checkpoint
    model_path = get_nnue_model_path(board_type, num_players, model_id)

    model = RingRiftNNUE(board_type=board_type)
    loaded_checkpoint_path: str | None = None
    used_fresh_weights: bool = False

    if model_path.exists():
        try:
            checkpoint = safe_load_checkpoint(model_path, map_location=device, warn_on_unsafe=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                arch_version = checkpoint.get("architecture_version", "v1.0.0")

                # Migrate legacy state dicts if needed (including accumulator weight padding)
                target_input_size = FEATURE_DIMS.get(board_type, FEATURE_DIMS[BoardType.SQUARE8])
                state_dict, detected_version = _migrate_legacy_state_dict(
                    state_dict, arch_version, target_input_size, board_type
                )
                if detected_version != CURRENT_NNUE_FEATURE_VERSION:
                    logger.info(f"Loaded NNUE checkpoint with feature version V{detected_version}")

                # Use strict=False to allow for minor structural differences
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            loaded_checkpoint_path = str(model_path)
            logger.info(f"Loaded NNUE model from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load NNUE model from {model_path}: {e}")
            if not allow_fresh:
                return None
            used_fresh_weights = True
            logger.info("Using fresh NNUE weights")
    else:
        if not allow_fresh:
            logger.debug(f"NNUE model not found at {model_path}")
            return None
        used_fresh_weights = True
        logger.info(f"NNUE model not found at {model_path}, using fresh weights")

    # Attach lightweight observability metadata for API consumers.
    try:
        model.loaded_checkpoint_path = loaded_checkpoint_path
        model.used_fresh_weights = used_fresh_weights
    except AttributeError:
        pass

    model = model.to(device)
    model.eval()

    # Apply torch.compile() optimization for faster inference on CUDA
    try:
        from .gpu_batch import compile_model
        dev_type = device.type if isinstance(device, torch.device) else str(device)
        if dev_type not in ("cpu", "mps"):
            model = compile_model(
                model,
                device=torch.device(device) if isinstance(device, str) else device,
                mode="default",  # Use default mode to avoid CUDA graph issues with dynamic shapes
            )
    except ImportError:
        pass  # gpu_batch not available
    except Exception as e:
        logger.debug(f"torch.compile() skipped for NNUE: {e}")

    # Cache the model
    _NNUE_CACHE[cache_key] = model

    return model


# =============================================================================
# NNUE Evaluator
# =============================================================================

class NNUEEvaluator:
    """Wrapper for NNUE evaluation in Minimax search.

    This class handles:
    - Model loading with fallback to heuristic
    - Feature extraction from both immutable and mutable states
    - Score scaling to centipawn-like values
    - Zero-sum evaluation for correct minimax behavior

    Zero-Sum Evaluation:
    For correct alpha-beta behavior, evaluation must satisfy:
        Eval(state, P1) = -Eval(state, P2)

    This is achieved by computing:
        zero_sum_score = (my_eval - max_opponent_eval) / 2

    This doubles evaluation cost but ensures correct minimax behavior.
    Controlled via RINGRIFT_NNUE_ZERO_SUM_EVAL environment variable.
    """

    # Scale factor to convert [-1, 1] to centipawn-like score
    SCORE_SCALE = 10000.0

    def __init__(
        self,
        board_type: BoardType,
        player_number: int,
        num_players: int = 2,
        model_id: str | None = None,
        allow_fresh: bool = True,
    ):
        import os
        self.board_type = board_type
        self.player_number = player_number
        self.num_players = num_players
        self.model = load_nnue_model(
            board_type=board_type,
            num_players=num_players,
            model_id=model_id,
            allow_fresh=allow_fresh,
        )
        self.available = self.model is not None

        # Zero-sum evaluation for minimax (enabled by default)
        self.use_zero_sum = os.getenv(
            'RINGRIFT_NNUE_ZERO_SUM_EVAL', 'true'
        ).lower() in ('true', '1', 'yes')

    def _raw_evaluate_mutable(
        self, state: MutableGameState, player_number: int
    ) -> float:
        """Raw NNUE evaluation for a specific player (NOT zero-sum)."""
        features = extract_features_from_mutable(state, player_number)
        value = self.model.forward_single(features)
        return value * self.SCORE_SCALE

    def _raw_evaluate_gamestate(
        self, state: GameState, player_number: int
    ) -> float:
        """Raw NNUE evaluation for a specific player (NOT zero-sum)."""
        features = extract_features_from_gamestate(state, player_number)
        value = self.model.forward_single(features)
        return value * self.SCORE_SCALE

    def evaluate_mutable(self, state: MutableGameState) -> float:
        """Evaluate a mutable game state with zero-sum normalization.

        Args:
            state: MutableGameState to evaluate

        Returns:
            Score in centipawn-like scale (positive = good for player).
            When zero-sum is enabled, returns (my_eval - max_opp_eval) / 2.
        """
        if not self.available or self.model is None:
            raise RuntimeError("NNUE model not available")

        my_eval = self._raw_evaluate_mutable(state, self.player_number)

        if not self.use_zero_sum:
            return my_eval

        # Compute zero-sum: (my_eval - max_opponent_eval) / 2
        # This ensures Eval(P1) = -Eval(P2)
        max_opponent_eval = float('-inf')
        for p in range(1, self.num_players + 1):
            if p != self.player_number:
                opp_eval = self._raw_evaluate_mutable(state, p)
                max_opponent_eval = max(max_opponent_eval, opp_eval)

        if max_opponent_eval == float('-inf'):
            return my_eval

        return (my_eval - max_opponent_eval) / 2.0

    def evaluate_gamestate(self, state: GameState) -> float:
        """Evaluate an immutable game state with zero-sum normalization.

        Args:
            state: GameState to evaluate

        Returns:
            Score in centipawn-like scale (positive = good for player).
            When zero-sum is enabled, returns (my_eval - max_opp_eval) / 2.
        """
        if not self.available or self.model is None:
            raise RuntimeError("NNUE model not available")

        my_eval = self._raw_evaluate_gamestate(state, self.player_number)

        if not self.use_zero_sum:
            return my_eval

        # Compute zero-sum: (my_eval - max_opponent_eval) / 2
        max_opponent_eval = float('-inf')
        for p in range(1, self.num_players + 1):
            if p != self.player_number:
                opp_eval = self._raw_evaluate_gamestate(state, p)
                max_opponent_eval = max(max_opponent_eval, opp_eval)

        if max_opponent_eval == float('-inf'):
            return my_eval

        return (my_eval - max_opponent_eval) / 2.0


# =============================================================================
# Batch Feature Extraction for GPU States
# =============================================================================


def extract_features_from_gpu_batch_vectorized(
    stack_owner: "torch.Tensor",      # (batch, H, W)
    stack_height: "torch.Tensor",     # (batch, H, W)
    territory_owner: "torch.Tensor",  # (batch, H, W)
    current_player: "torch.Tensor",   # (batch,)
    num_players: int = 2,
) -> "torch.Tensor":
    """Fully vectorized batch feature extraction from GPU tensors.

    Extracts NNUE features directly from GPU batch state tensors for
    efficient batch evaluation during NNUE-guided selfplay.

    Feature planes (12 total) - PERSPECTIVE ROTATED:
    - Planes 0-3: Ring/stack presence (plane 0 = current player)
    - Planes 4-7: Stack height normalized (plane 4 = current player)
    - Planes 8-11: Territory ownership (plane 8 = current player)

    Args:
        stack_owner: (batch, H, W) tensor of stack owners (0=empty, 1-4=player)
        stack_height: (batch, H, W) tensor of stack heights (0-5)
        territory_owner: (batch, H, W) tensor of territory owners
        current_player: (batch,) tensor of current player per game
        num_players: Number of players in the game

    Returns:
        (batch, feature_dim) tensor of features
    """
    batch_size = stack_owner.shape[0]
    H, W = stack_owner.shape[1], stack_owner.shape[2]
    device = stack_owner.device

    # Initialize features: (batch, 12, H, W)
    features = torch.zeros(batch_size, FEATURE_PLANES, H, W, device=device)

    # Create player indices: (num_players,) = [1, 2, ..., num_players]
    players = torch.arange(1, num_players + 1, device=device)

    # Current player expanded: (batch, 1)
    cp = current_player.unsqueeze(1)

    # Compute rotation for all players: (batch, num_players)
    # For player p: rotated = 0 if p == cp, else ((p - cp) % num_players)
    is_current = (players.unsqueeze(0) == cp)  # (batch, num_players)
    rotation = torch.where(
        is_current,
        torch.zeros(batch_size, num_players, dtype=torch.long, device=device),
        (players.unsqueeze(0) - cp) % num_players
    )  # (batch, num_players)

    # For each player, scatter their features to the rotated plane
    for p_idx, p in enumerate(range(1, num_players + 1)):
        # Masks for this player: (batch, H, W)
        owner_mask = (stack_owner == p).float()
        height_feature = torch.clamp((stack_height * owner_mask) / 5.0, 0, 1)
        territory_mask = (territory_owner == p).float()

        # Get rotation for this player: (batch,)
        rot = rotation[:, p_idx]

        # Scatter to appropriate planes using advanced indexing
        batch_idx = torch.arange(batch_size, device=device)

        # Planes 0-3: Ring presence
        features[batch_idx, rot] = owner_mask

        # Planes 4-7: Stack height
        features[batch_idx, 4 + rot] = height_feature

        # Planes 8-11: Territory
        features[batch_idx, 8 + rot] = territory_mask

    # Flatten to (batch, feature_dim)
    return features.view(batch_size, -1)


class BatchNNUEEvaluator:
    """Batch NNUE evaluator for GPU selfplay.

    Evaluates multiple game states in parallel using the NNUE model.
    """

    def __init__(
        self,
        board_type: BoardType,
        num_players: int = 2,
        model_path: str | None = None,
        device: Optional["torch.device"] = None,
    ):
        self.board_type = board_type
        self.num_players = num_players
        self.device = device or torch.device("cpu")

        # Load NNUE model
        self.model: RingRiftNNUE | None = None
        self.available = False

        if model_path is None:
            model_path = get_nnue_model_path(board_type, num_players)

        if model_path and Path(model_path).exists():
            try:
                self.model = RingRiftNNUE(
                    board_type=board_type,
                    hidden_dim=256,
                    num_hidden_layers=2,
                )
                checkpoint = safe_load_checkpoint(model_path, map_location=self.device, warn_on_unsafe=False)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                    arch_version = checkpoint.get("architecture_version", "v1.0.0")

                    # Apply migration for legacy models (including feature version padding)
                    target_input_size = FEATURE_DIMS.get(board_type, FEATURE_DIMS[BoardType.SQUARE8])
                    state_dict, detected_version = _migrate_legacy_state_dict(
                        state_dict, arch_version, target_input_size, board_type
                    )
                    self.model.load_state_dict(state_dict, strict=False)
                else:
                    self.model.load_state_dict(checkpoint, strict=False)
                self.model.to(self.device)
                self.model.eval()
                self.available = True
                logger.info(f"BatchNNUEEvaluator loaded model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load NNUE model: {e}")

    def evaluate_batch(
        self,
        stack_owner: "torch.Tensor",
        stack_height: "torch.Tensor",
        territory_owner: "torch.Tensor",
        current_player: "torch.Tensor",
    ) -> "torch.Tensor":
        """Evaluate a batch of game states.

        Args:
            stack_owner: (batch, H, W) tensor
            stack_height: (batch, H, W) tensor
            territory_owner: (batch, H, W) tensor
            current_player: (batch,) tensor

        Returns:
            (batch,) tensor of evaluation scores in centipawn-like range
        """
        if not self.available or self.model is None:
            # Return zeros if model not available
            return torch.zeros(stack_owner.shape[0], device=self.device)

        # Extract features
        features = extract_features_from_gpu_batch_vectorized(
            stack_owner, stack_height, territory_owner,
            current_player, self.num_players
        )

        # Evaluate with model
        with torch.no_grad():
            scores = self.model(features).squeeze(-1)  # (batch,)

        return scores * 1000.0  # Scale to centipawn-like range

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

import gc
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..models import BoardType, GameState, Position
from ..rules.mutable_state import MutableGameState

logger = logging.getLogger(__name__)

# =============================================================================
# Model Cache
# =============================================================================

_NNUE_CACHE: Dict[Tuple[str, str], nn.Module] = {}


def clear_nnue_cache() -> None:
    """Clear the NNUE model cache and release memory."""
    global _NNUE_CACHE
    cache_size = len(_NNUE_CACHE)

    for model in _NNUE_CACHE.values():
        try:
            model.cpu()
        except Exception:
            pass

    _NNUE_CACHE.clear()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    gc.collect()

    if cache_size > 0:
        logger.info(f"Cleared NNUE cache ({cache_size} models)")


# =============================================================================
# Feature Dimensions
# =============================================================================

# Input features: 12 planes per position
# - Ring presence per player (4 planes for 4 players max)
# - Stack presence per player (4 planes)
# - Territory ownership per player (4 planes)
FEATURE_PLANES = 12

BOARD_SIZES: Dict[BoardType, int] = {
    BoardType.SQUARE8: 8,
    BoardType.SQUARE19: 19,
    BoardType.HEXAGONAL: 25,  # Embedded in 25x25 grid
}

FEATURE_DIMS: Dict[BoardType, int] = {
    BoardType.SQUARE8: 8 * 8 * FEATURE_PLANES,      # 768 features
    BoardType.SQUARE19: 19 * 19 * FEATURE_PLANES,   # 4332 features
    BoardType.HEXAGONAL: 25 * 25 * FEATURE_PLANES,  # 7500 features
}


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


class RingRiftNNUE(nn.Module):
    """NNUE-style evaluation network for RingRift Minimax search.

    Architecture inspired by Stockfish NNUE with RingRift-specific features.
    Designed for efficient CPU inference during alpha-beta search.

    Input: Sparse feature vector (12 planes flattened)
    Output: Scalar value in [-1, 1]

    The network uses ClippedReLU activations and small hidden layers
    to enable fast inference without GPU.
    """

    ARCHITECTURE_VERSION = "v1.0.0"

    def __init__(
        self,
        board_type: BoardType = BoardType.SQUARE8,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        self.board_type = board_type
        input_dim = get_feature_dim(board_type)

        # Accumulator layer (like Half-King-Piece-Square in chess NNUE)
        # Projects sparse input to dense hidden representation
        self.accumulator = nn.Linear(input_dim, hidden_dim, bias=True)

        # Hidden layers with ClippedReLU
        layers = []
        current_dim = hidden_dim * 2  # Concatenate player perspectives
        for i in range(num_hidden_layers):
            out_dim = 32 if i < num_hidden_layers - 1 else 32
            layers.append(nn.Linear(current_dim, out_dim))
            layers.append(ClippedReLU())
            current_dim = out_dim

        self.hidden = nn.Sequential(*layers)

        # Output layer: single scalar value
        self.output = nn.Linear(32, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Shape (batch, input_dim) sparse/dense input features

        Returns:
            Shape (batch, 1) values in [-1, 1]
        """
        # Accumulator with ClippedReLU
        acc = torch.clamp(self.accumulator(features), 0.0, 1.0)

        # Concatenate "perspectives" - simplified version using same features
        # In full NNUE, we'd have separate accumulators for each player view
        x = torch.cat([acc, acc], dim=-1)

        # Hidden layers
        x = self.hidden(x)

        # Output with tanh for [-1, 1] range
        return torch.tanh(self.output(x))

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


# =============================================================================
# Feature Extraction
# =============================================================================

def _pos_to_index(pos: Position, board_size: int, board_type: BoardType) -> int:
    """Convert position to linear index in feature vector."""
    if board_type == BoardType.HEXAGONAL:
        # Hex uses axial coordinates, offset by radius
        radius = (board_size - 1) // 2
        cx = pos.x + radius
        cy = pos.y + radius
        return cy * board_size + cx
    else:
        return pos.y * board_size + pos.x


def extract_features_from_gamestate(
    state: GameState,
    player_number: int,
) -> np.ndarray:
    """Extract NNUE features from immutable GameState.

    Feature planes (12 total):
    - Planes 0-3: Ring presence for players 1-4
    - Planes 4-7: Stack presence for players 1-4
    - Planes 8-11: Territory ownership for players 1-4

    Args:
        state: The game state to extract features from
        player_number: The player perspective (1-4)

    Returns:
        Flattened feature vector of shape (board_size * board_size * 12,)
    """
    board = state.board
    board_type = board.type
    board_size = get_board_size(board_type)
    num_positions = board_size * board_size

    # Initialize feature planes
    features = np.zeros(num_positions * FEATURE_PLANES, dtype=np.float32)

    # Extract ring/stack positions from board.stacks (Dict[str, RingStack])
    for pos_key, ring_stack in (board.stacks or {}).items():
        try:
            parts = pos_key.split(",")
            if len(parts) >= 2:
                x, y = int(parts[0]), int(parts[1])
                z = int(parts[2]) if len(parts) == 3 else None
                pos = Position(x=x, y=y, z=z)
                idx = _pos_to_index(pos, board_size, board_type)

                if 0 <= idx < num_positions:
                    # Ring features (planes 0-3) - mark presence for each player's rings
                    rings = getattr(ring_stack, 'rings', [])
                    for ring_owner in rings:
                        if 1 <= ring_owner <= 4:
                            plane_idx = (ring_owner - 1) * num_positions
                            features[plane_idx + idx] = 1.0

                    # Stack features (planes 4-7) - use controlling player and height
                    owner = getattr(ring_stack, 'controlling_player', 0)
                    height = getattr(ring_stack, 'stack_height', 0)
                    if 1 <= owner <= 4 and height > 0:
                        plane_idx = (4 + owner - 1) * num_positions
                        # Normalize height to [0, 1]
                        features[plane_idx + idx] = min(height / 5.0, 1.0)
        except (ValueError, AttributeError):
            continue

    # Extract territory features (planes 8-11) from board.territories
    for territory_key, territory in (board.territories or {}).items():
        try:
            pnum = getattr(territory, 'player', 0)
            if 1 <= pnum <= 4:
                for pos in (getattr(territory, 'spaces', None) or []):
                    idx = _pos_to_index(pos, board_size, board_type)
                    if 0 <= idx < num_positions:
                        plane_idx = (8 + pnum - 1) * num_positions
                        features[plane_idx + idx] = 1.0
        except (ValueError, AttributeError):
            continue

    return features


def extract_features_from_mutable(
    state: MutableGameState,
    player_number: int,
) -> np.ndarray:
    """Extract NNUE features from MutableGameState.

    Optimized feature extraction that works directly with MutableGameState
    to avoid conversion overhead during search.

    Args:
        state: The mutable game state
        player_number: The player perspective (1-4)

    Returns:
        Flattened feature vector of shape (board_size * board_size * 12,)
    """
    board_type = state.board_type
    board_size = get_board_size(board_type)
    num_positions = board_size * board_size

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

    # Extract ring features from internal arrays
    if rings is not None:
        for y in range(board_size):
            for x in range(board_size):
                idx = y * board_size + x
                owner = int(rings[y, x]) if rings.ndim >= 2 else 0
                if 1 <= owner <= 4:
                    plane_idx = (owner - 1) * num_positions
                    features[plane_idx + idx] = 1.0

    # Extract stack features
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
                    plane_idx = (4 + owner - 1) * num_positions
                    features[plane_idx + idx] = min(height / 5.0, 1.0)

    # Extract territory features
    if territories is not None:
        for y in range(board_size):
            for x in range(board_size):
                idx = y * board_size + x
                owner = int(territories[y, x]) if territories.ndim >= 2 else 0
                if 1 <= owner <= 4:
                    plane_idx = (8 + owner - 1) * num_positions
                    features[plane_idx + idx] = 1.0

    return features


# =============================================================================
# Model Loading
# =============================================================================

def get_nnue_model_path(
    board_type: BoardType,
    num_players: int = 2,
    model_id: Optional[str] = None,
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


def load_nnue_model(
    board_type: BoardType,
    num_players: int = 2,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    allow_fresh: bool = True,
) -> Optional[RingRiftNNUE]:
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
    global _NNUE_CACHE

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

    if model_path.exists():
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded NNUE model from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load NNUE model from {model_path}: {e}")
            if not allow_fresh:
                return None
            logger.info("Using fresh NNUE weights")
    else:
        if not allow_fresh:
            logger.debug(f"NNUE model not found at {model_path}")
            return None
        logger.info(f"NNUE model not found at {model_path}, using fresh weights")

    model = model.to(device)
    model.eval()

    # Apply torch.compile() optimization for faster inference on CUDA
    try:
        from .gpu_batch import compile_model
        if device not in ("cpu", "mps"):
            model = compile_model(
                model,
                device=torch.device(device) if isinstance(device, str) else device,
                mode="reduce-overhead",
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
    """

    # Scale factor to convert [-1, 1] to centipawn-like score
    SCORE_SCALE = 10000.0

    def __init__(
        self,
        board_type: BoardType,
        player_number: int,
        num_players: int = 2,
        model_id: Optional[str] = None,
        allow_fresh: bool = True,
    ):
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

    def evaluate_mutable(self, state: MutableGameState) -> float:
        """Evaluate a mutable game state.

        Args:
            state: MutableGameState to evaluate

        Returns:
            Score in centipawn-like scale (positive = good for player)
        """
        if not self.available or self.model is None:
            raise RuntimeError("NNUE model not available")

        features = extract_features_from_mutable(state, self.player_number)
        value = self.model.forward_single(features)

        # Scale to centipawn-like score
        return value * self.SCORE_SCALE

    def evaluate_gamestate(self, state: GameState) -> float:
        """Evaluate an immutable game state.

        Args:
            state: GameState to evaluate

        Returns:
            Score in centipawn-like scale (positive = good for player)
        """
        if not self.available or self.model is None:
            raise RuntimeError("NNUE model not available")

        features = extract_features_from_gamestate(state, self.player_number)
        value = self.model.forward_single(features)

        # Scale to centipawn-like score
        return value * self.SCORE_SCALE

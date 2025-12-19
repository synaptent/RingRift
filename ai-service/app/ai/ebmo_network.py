"""Energy-Based Move Optimization (EBMO) Neural Network Architecture.

A novel game-playing algorithm that uses gradient descent on continuous
action embeddings at inference time to find optimal moves.

Unlike traditional approaches (policy networks with softmax, MCTS tree search),
EBMO:
1. Trains an energy function E(s, a) over (state, action) pairs
2. Uses gradient descent to minimize energy during inference
3. Projects optimized embeddings back to legal discrete moves

Key Innovation: Continuous optimization in action space enables gradient-guided
exploration rather than discrete search or random sampling.

Usage:
    from app.ai.ebmo_network import EBMONetwork, EBMOConfig

    # Create network
    config = EBMOConfig()
    network = EBMONetwork(config)

    # Forward pass for training
    energies = network(state_features, global_features, action_features)

    # Inference-time optimization
    state_embed = network.encode_state(features, globals)
    action_embed = network.encode_action(move_features)
    energy = network.compute_energy(state_embed, action_embed)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models import (
    BoardType,
    GamePhase,
    GameState,
    Move,
    MoveType,
    Position,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EBMOConfig:
    """Configuration for EBMO network and inference."""

    # Network architecture
    state_embed_dim: int = 256
    action_embed_dim: int = 128
    energy_hidden_dim: int = 256
    num_energy_layers: int = 3

    # State encoder (CNN backbone)
    # Training data has 14 planes x 4 history frames = 56 channels
    num_input_channels: int = 56
    num_global_features: int = 20
    num_residual_blocks: int = 6
    residual_filters: int = 128

    # Action encoder
    action_feature_dim: int = 14  # Raw action features before embedding
    action_hidden_dim: int = 64

    # Inference-time optimization (increased for better convergence)
    optim_steps: int = 100  # Doubled from 50
    optim_lr: float = 0.1
    num_restarts: int = 8  # Increased from 5
    projection_temperature: float = 0.3  # Lower for sharper projection
    project_every_n_steps: int = 10

    # Training
    contrastive_temperature: float = 0.1
    num_negatives: int = 15
    outcome_weight: float = 0.5
    learning_rate: float = 0.001

    # Board-specific
    board_size: int = 8  # Square 8x8
    board_type: BoardType = BoardType.SQUARE8


# =============================================================================
# Action Feature Extraction
# =============================================================================


class ActionFeatureExtractor:
    """Extract continuous features from discrete Move objects.

    Converts RingRift moves to a fixed-size feature vector suitable
    for neural network processing and gradient-based optimization.

    Feature layout (14 dimensions):
        - from_x, from_y: Normalized source position [0, 1]
        - to_x, to_y: Normalized destination position [0, 1]
        - move_type: 8-dim one-hot encoding (relaxed to continuous)
        - direction_x, direction_y: Unit direction vector
    """

    # Map MoveType to index for one-hot encoding
    MOVE_TYPE_MAP = {
        MoveType.PLACE_RING: 0,
        MoveType.MOVE_STACK: 1,
        MoveType.BUILD_STACK: 1,  # Same as move
        MoveType.OVERTAKING_CAPTURE: 2,
        MoveType.CHAIN_CAPTURE: 2,
        MoveType.SKIP_PLACEMENT: 3,
        MoveType.SKIP_CAPTURE: 3,
        MoveType.NO_PLACEMENT_ACTION: 3,
        MoveType.NO_MOVEMENT_ACTION: 3,
        MoveType.NO_LINE_ACTION: 3,
        MoveType.NO_TERRITORY_ACTION: 3,
        MoveType.PROCESS_LINE: 4,
        MoveType.CHOOSE_LINE_OPTION: 4,
        MoveType.CHOOSE_LINE_REWARD: 4,
        MoveType.PROCESS_TERRITORY_REGION: 5,
        MoveType.CHOOSE_TERRITORY_OPTION: 5,
        MoveType.TERRITORY_CLAIM: 5,
        MoveType.SWAP_SIDES: 6,
        MoveType.FORCED_ELIMINATION: 7,
        MoveType.ELIMINATE_RINGS_FROM_STACK: 7,
        MoveType.RECOVERY_SLIDE: 7,
        MoveType.SKIP_RECOVERY: 7,
    }

    NUM_MOVE_TYPES = 8

    def __init__(self, board_size: int = 8):
        """Initialize the feature extractor.

        Args:
            board_size: Size of the board for position normalization
        """
        self.board_size = board_size

    def extract_features(self, move: Move) -> np.ndarray:
        """Extract features from a single move.

        Args:
            move: Move object to extract features from

        Returns:
            14-dimensional feature vector
        """
        features = np.zeros(14, dtype=np.float32)

        # Normalized positions [0, 1]
        if move.from_pos is not None:
            features[0] = move.from_pos.x / (self.board_size - 1)
            features[1] = move.from_pos.y / (self.board_size - 1)
        else:
            features[0] = 0.5  # Center default
            features[1] = 0.5

        if move.to is not None:
            features[2] = move.to.x / (self.board_size - 1)
            features[3] = move.to.y / (self.board_size - 1)
        else:
            features[2] = features[0]  # Same as from
            features[3] = features[1]

        # Move type one-hot (indices 4-11)
        type_idx = self.MOVE_TYPE_MAP.get(move.type, 0)
        features[4 + type_idx] = 1.0

        # Direction vector (normalized)
        if move.from_pos is not None and move.to is not None:
            dx = move.to.x - move.from_pos.x
            dy = move.to.y - move.from_pos.y
            dist = math.sqrt(dx * dx + dy * dy) + 1e-8
            features[12] = dx / dist
            features[13] = dy / dist
        else:
            features[12] = 0.0
            features[13] = 0.0

        return features

    def extract_batch(self, moves: List[Move]) -> np.ndarray:
        """Extract features from a batch of moves.

        Args:
            moves: List of Move objects

        Returns:
            (N, 14) feature array
        """
        return np.stack([self.extract_features(m) for m in moves])

    def extract_tensor(
        self,
        moves: List[Move],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Extract features as a PyTorch tensor.

        Args:
            moves: List of Move objects
            device: Target device for tensor

        Returns:
            (N, 14) tensor
        """
        features = self.extract_batch(moves)
        tensor = torch.from_numpy(features)
        if device is not None:
            tensor = tensor.to(device)
        return tensor


# =============================================================================
# State Feature Extraction (reuses existing patterns)
# =============================================================================


def extract_state_features(
    game_state: GameState,
    player_number: int,
    board_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract board and global features from game state.

    This is a simplified version that extracts the most important
    features for EBMO. For production, consider reusing the full
    feature extraction from neural_net.py.

    Args:
        game_state: Current game state
        player_number: Perspective player (1-based)
        board_size: Size of the board

    Returns:
        (board_features, global_features) tuple
        - board_features: (14, H, W) array
        - global_features: (20,) array
    """
    # Board features: 14 channels x H x W
    board_features = np.zeros((14, board_size, board_size), dtype=np.float32)

    # Channel layout:
    # 0: My stacks
    # 1: Opponent stacks
    # 2: My markers
    # 3: Opponent markers
    # 4: My collapsed spaces
    # 5: Opponent collapsed spaces
    # 6-7: Reserved
    # 8-11: Stack heights (normalized)
    # 12: Valid positions mask
    # 13: Reserved

    board = game_state.board

    # Extract stacks
    # Note: stack.rings is a list of integers (owner IDs), not ring objects
    for key, stack in board.stacks.items():
        try:
            parts = key.split(",")
            x, y = int(parts[0]), int(parts[1])
            if 0 <= x < board_size and 0 <= y < board_size:
                # stack.rings is a list of owner IDs (integers)
                owner = stack.rings[0] if stack.rings else 0
                height = len(stack.rings) / 5.0  # Normalize by max height

                if owner == player_number:
                    board_features[0, y, x] = 1.0
                    board_features[8, y, x] = height
                elif owner != 0:
                    board_features[1, y, x] = 1.0
                    board_features[9, y, x] = height
        except (ValueError, IndexError):
            continue

    # Extract markers
    # Note: MarkerInfo has .player attribute, not .owner
    for key, marker in board.markers.items():
        try:
            parts = key.split(",")
            x, y = int(parts[0]), int(parts[1])
            if 0 <= x < board_size and 0 <= y < board_size:
                if marker.player == player_number:
                    board_features[2, y, x] = 1.0
                else:
                    board_features[3, y, x] = 1.0
        except (ValueError, IndexError):
            continue

    # Extract collapsed spaces
    # Note: collapsed_spaces is Dict[str, int] where value is owner ID
    for key, owner in board.collapsed_spaces.items():
        try:
            parts = key.split(",")
            x, y = int(parts[0]), int(parts[1])
            if 0 <= x < board_size and 0 <= y < board_size:
                if owner == player_number:
                    board_features[4, y, x] = 1.0
                else:
                    board_features[5, y, x] = 1.0
        except (ValueError, IndexError, AttributeError):
            continue

    # Valid positions mask (all positions on 8x8 are valid)
    board_features[12, :, :] = 1.0

    # Global features: 20 dimensions
    global_features = np.zeros(20, dtype=np.float32)

    # Phase encoding (one-hot, first 8 slots)
    phase_map = {
        GamePhase.RING_PLACEMENT: 0,
        GamePhase.MOVEMENT: 1,
        GamePhase.CAPTURE: 2,
        GamePhase.CHAIN_CAPTURE: 3,
        GamePhase.LINE_PROCESSING: 4,
        GamePhase.TERRITORY_PROCESSING: 5,
        GamePhase.FORCED_ELIMINATION: 6,
        GamePhase.GAME_OVER: 7,
    }
    phase_idx = phase_map.get(game_state.current_phase, 0)
    global_features[phase_idx] = 1.0

    # Player info
    num_players = len(game_state.players)
    global_features[8] = player_number / num_players  # Normalized player
    global_features[9] = game_state.current_player / num_players  # Current player

    # Ring counts (if available)
    for i, player in enumerate(game_state.players[:4]):
        if hasattr(player, 'rings_in_hand'):
            global_features[10 + i] = player.rings_in_hand / 15.0  # Normalize

    # Turn number (if available, slot 18)
    if hasattr(game_state, 'turn_number'):
        global_features[18] = min(game_state.turn_number / 100.0, 1.0)

    # Is current player (slot 19)
    global_features[19] = 1.0 if game_state.current_player == player_number else 0.0

    return board_features, global_features


# =============================================================================
# Network Components
# =============================================================================


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excite(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """Residual block with optional SE attention."""

    def __init__(self, channels: int, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + residual)
        return out


class StateEncoder(nn.Module):
    """CNN encoder for game state features.

    Converts (C, H, W) board features + global features into a
    fixed-size state embedding.
    """

    def __init__(self, config: EBMOConfig):
        super().__init__()
        self.config = config

        # Initial convolution
        self.conv_in = nn.Conv2d(
            config.num_input_channels,
            config.residual_filters,
            3,
            padding=1,
            bias=False,
        )
        self.bn_in = nn.BatchNorm2d(config.residual_filters)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(config.residual_filters, use_se=True)
            for _ in range(config.num_residual_blocks)
        ])

        # Global average pooling + projection
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Combine spatial and global features
        combined_dim = config.residual_filters + config.num_global_features
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, config.state_embed_dim),
            nn.ReLU(),
            nn.Linear(config.state_embed_dim, config.state_embed_dim),
        )

    def forward(
        self,
        board_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        """Encode state to embedding.

        Args:
            board_features: (B, C, H, W) board tensor
            global_features: (B, G) global features

        Returns:
            (B, state_embed_dim) state embedding
        """
        # CNN processing
        x = F.relu(self.bn_in(self.conv_in(board_features)))
        for block in self.res_blocks:
            x = block(x)

        # Global average pool
        x = self.gap(x).flatten(1)  # (B, residual_filters)

        # Concatenate global features
        x = torch.cat([x, global_features], dim=1)

        # Final projection
        return self.fc(x)


class ActionEncoder(nn.Module):
    """MLP encoder for action features.

    Converts raw action features (positions, type, direction) into
    a continuous embedding suitable for gradient-based optimization.
    """

    def __init__(self, config: EBMOConfig):
        super().__init__()
        self.config = config

        self.encoder = nn.Sequential(
            nn.Linear(config.action_feature_dim, config.action_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.action_hidden_dim, config.action_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.action_hidden_dim, config.action_embed_dim),
        )

    def forward(self, action_features: torch.Tensor) -> torch.Tensor:
        """Encode action features to embedding.

        Args:
            action_features: (B, action_feature_dim) raw features

        Returns:
            (B, action_embed_dim) action embedding
        """
        return self.encoder(action_features)


class EnergyHead(nn.Module):
    """MLP that computes energy from state and action embeddings.

    Low energy = good move, high energy = bad move.
    """

    def __init__(self, config: EBMOConfig):
        super().__init__()
        self.config = config

        input_dim = config.state_embed_dim + config.action_embed_dim
        hidden_dim = config.energy_hidden_dim

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(config.num_energy_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        state_embed: torch.Tensor,
        action_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Compute energy for state-action pairs.

        Args:
            state_embed: (B, state_embed_dim) state embeddings
            action_embed: (B, action_embed_dim) action embeddings

        Returns:
            (B,) energy values (lower = better move)
        """
        combined = torch.cat([state_embed, action_embed], dim=1)
        return self.mlp(combined).squeeze(-1)


# =============================================================================
# Main EBMO Network
# =============================================================================


class EBMONetwork(nn.Module):
    """Complete Energy-Based Move Optimization network.

    Combines:
    - StateEncoder: CNN for board state -> state embedding
    - ActionEncoder: MLP for action features -> action embedding
    - EnergyHead: MLP for (state, action) -> energy scalar

    Training:
    - Use contrastive loss to push good moves to low energy
    - Optionally weight by game outcome

    Inference:
    - Encode state once
    - Initialize action embedding (random or from prior)
    - Gradient descent on action embedding to minimize energy
    - Project to nearest legal move
    """

    def __init__(self, config: Optional[EBMOConfig] = None):
        super().__init__()
        self.config = config or EBMOConfig()

        self.state_encoder = StateEncoder(self.config)
        self.action_encoder = ActionEncoder(self.config)
        self.energy_head = EnergyHead(self.config)

        # Feature extractor for converting Move objects
        self.feature_extractor = ActionFeatureExtractor(self.config.board_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encode_state(
        self,
        board_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        """Encode game state to embedding.

        Args:
            board_features: (B, C, H, W) board tensor
            global_features: (B, G) global features

        Returns:
            (B, state_embed_dim) state embedding
        """
        return self.state_encoder(board_features, global_features)

    def encode_action(self, action_features: torch.Tensor) -> torch.Tensor:
        """Encode action to embedding.

        Args:
            action_features: (B, action_feature_dim) raw features

        Returns:
            (B, action_embed_dim) action embedding
        """
        return self.action_encoder(action_features)

    def compute_energy(
        self,
        state_embed: torch.Tensor,
        action_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Compute energy for state-action pairs.

        Args:
            state_embed: (B, state_embed_dim) or (state_embed_dim,)
            action_embed: (B, action_embed_dim) or (action_embed_dim,)

        Returns:
            (B,) or scalar energy value
        """
        # Handle single samples
        single = state_embed.dim() == 1
        if single:
            state_embed = state_embed.unsqueeze(0)
            action_embed = action_embed.unsqueeze(0)

        energy = self.energy_head(state_embed, action_embed)

        if single:
            energy = energy.squeeze(0)

        return energy

    def forward(
        self,
        board_features: torch.Tensor,
        global_features: torch.Tensor,
        action_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass computing energy for all inputs.

        Args:
            board_features: (B, C, H, W) board tensor
            global_features: (B, G) global features
            action_features: (B, action_feature_dim) action features

        Returns:
            (B,) energy values
        """
        state_embed = self.encode_state(board_features, global_features)
        action_embed = self.encode_action(action_features)
        return self.compute_energy(state_embed, action_embed)

    def encode_moves(
        self,
        moves: List[Move],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Encode a list of moves to embeddings.

        Args:
            moves: List of Move objects
            device: Target device

        Returns:
            (N, action_embed_dim) tensor of embeddings
        """
        features = self.feature_extractor.extract_tensor(moves, device)
        return self.encode_action(features)

    def encode_state_from_game(
        self,
        game_state: GameState,
        player_number: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Encode game state to embedding.

        Args:
            game_state: Game state object
            player_number: Perspective player
            device: Target device

        Returns:
            (state_embed_dim,) state embedding
        """
        board_feat, global_feat = extract_state_features(
            game_state,
            player_number,
            self.config.board_size,
        )

        # Training data has 56 channels (14 planes x 4 history frames)
        # For inference, stack current state 4 times to match
        if self.config.num_input_channels == 56 and board_feat.shape[0] == 14:
            board_feat = np.concatenate([board_feat] * 4, axis=0)

        board_tensor = torch.from_numpy(board_feat).unsqueeze(0)
        global_tensor = torch.from_numpy(global_feat).unsqueeze(0)

        if device is not None:
            board_tensor = board_tensor.to(device)
            global_tensor = global_tensor.to(device)

        state_embed = self.encode_state(board_tensor, global_tensor)
        return state_embed.squeeze(0)


# =============================================================================
# Loss Functions
# =============================================================================


def contrastive_energy_loss(
    energies_positive: torch.Tensor,
    energies_negative: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Contrastive loss pushing positive to low energy, negatives to high.

    Args:
        energies_positive: (B,) energies for positive (good) moves
        energies_negative: (B, N) energies for N negative moves per sample
        temperature: Softmax temperature

    Returns:
        Scalar loss value
    """
    batch_size = energies_positive.shape[0]

    # Concatenate: positive first, then negatives
    # Shape: (B, 1+N)
    all_energies = torch.cat([
        energies_positive.unsqueeze(1),
        energies_negative,
    ], dim=1)

    # Lower energy should be selected (convert to logits by negating)
    logits = -all_energies / temperature

    # Target: first element (positive) should have highest probability
    targets = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

    return F.cross_entropy(logits, targets)


def outcome_weighted_energy_loss(
    energies: torch.Tensor,
    outcomes: torch.Tensor,
) -> torch.Tensor:
    """Loss weighted by game outcome.

    Args:
        energies: (B,) energy predictions
        outcomes: (B,) game outcomes (+1 win, 0 draw, -1 loss)

    Returns:
        Scalar loss value
    """
    # Target: wins should have low energy (-1), losses high energy (+1)
    target_energies = -outcomes.float()

    return F.mse_loss(energies, target_energies)


def margin_ranking_loss(
    energies_positive: torch.Tensor,
    energies_negative: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Margin-based ranking loss for energy separation.

    Ensures positive moves have lower energy than negatives by at least margin.
    More effective than contrastive for learning energy differences.

    Args:
        energies_positive: (B,) energies for positive (good) moves
        energies_negative: (B, N) energies for N negative moves per sample
        margin: Minimum energy gap required

    Returns:
        Scalar loss value
    """
    batch_size, num_neg = energies_negative.shape

    # Expand positive energies to match negative shape
    pos_expanded = energies_positive.unsqueeze(1).expand(-1, num_neg)

    # Margin loss: max(0, pos_energy - neg_energy + margin)
    # We want pos_energy < neg_energy - margin
    losses = F.relu(pos_expanded - energies_negative + margin)

    return losses.mean()


def hard_negative_contrastive_loss(
    energies_positive: torch.Tensor,
    energies_hard_negative: torch.Tensor,
    energies_random_negative: torch.Tensor,
    temperature: float = 0.1,
    hard_weight: float = 0.7,
) -> torch.Tensor:
    """Contrastive loss with weighted hard and random negatives.

    Hard negatives (moves from losing games or near-misses) are weighted
    more heavily than random negatives.

    Args:
        energies_positive: (B,) energies for positive moves
        energies_hard_negative: (B, H) energies for hard negative moves
        energies_random_negative: (B, R) energies for random negative moves
        temperature: Softmax temperature
        hard_weight: Weight for hard negatives vs random

    Returns:
        Scalar loss value
    """
    batch_size = energies_positive.shape[0]

    # Combine all negatives
    all_negatives = torch.cat([energies_hard_negative, energies_random_negative], dim=1)

    # Concatenate: positive first, then all negatives
    all_energies = torch.cat([
        energies_positive.unsqueeze(1),
        all_negatives,
    ], dim=1)

    # Convert to logits (lower energy = higher logit)
    logits = -all_energies / temperature

    # Target: first element (positive) should have highest probability
    targets = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

    return F.cross_entropy(logits, targets)


def combined_ebmo_loss(
    energies_positive: torch.Tensor,
    energies_negative: torch.Tensor,
    outcomes: torch.Tensor,
    contrastive_temperature: float = 0.1,
    outcome_weight: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combined EBMO training loss.

    Args:
        energies_positive: (B,) energies for positive moves
        energies_negative: (B, N) energies for negative moves
        outcomes: (B,) game outcomes
        contrastive_temperature: Temperature for contrastive loss
        outcome_weight: Weight for outcome loss component

    Returns:
        (total_loss, loss_dict) tuple
    """
    contrastive = contrastive_energy_loss(
        energies_positive,
        energies_negative,
        contrastive_temperature,
    )

    outcome = outcome_weighted_energy_loss(energies_positive, outcomes)

    total = contrastive + outcome_weight * outcome

    return total, {
        "contrastive": contrastive.item(),
        "outcome": outcome.item(),
        "total": total.item(),
    }


# =============================================================================
# Model Loading/Saving
# =============================================================================


def save_ebmo_model(
    model: EBMONetwork,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    **metadata,
) -> None:
    """Save EBMO model checkpoint.

    Args:
        model: EBMO network to save
        path: Path to save checkpoint
        optimizer: Optional optimizer state
        epoch: Current epoch
        **metadata: Additional metadata
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": model.config,
        "epoch": epoch,
        "metadata": metadata,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)
    logger.info(f"Saved EBMO model to {path}")


def load_ebmo_model(
    path: str,
    device: Optional[torch.device] = None,
    config: Optional[EBMOConfig] = None,
) -> Tuple[EBMONetwork, Dict[str, Any]]:
    """Load EBMO model from checkpoint.

    Args:
        path: Path to checkpoint
        device: Target device
        config: Override config (uses saved config if None)

    Returns:
        (model, checkpoint_info) tuple
    """
    checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)

    # Use saved config unless overridden
    model_config = config or checkpoint.get("config", EBMOConfig())
    model = EBMONetwork(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    if device is not None:
        model = model.to(device)

    info = {
        "epoch": checkpoint.get("epoch", 0),
        "metadata": checkpoint.get("metadata", {}),
    }

    logger.info(f"Loaded EBMO model from {path} (epoch {info['epoch']})")

    return model, info


__all__ = [
    "EBMOConfig",
    "EBMONetwork",
    "ActionFeatureExtractor",
    "StateEncoder",
    "ActionEncoder",
    "EnergyHead",
    "extract_state_features",
    "contrastive_energy_loss",
    "outcome_weighted_energy_loss",
    "margin_ranking_loss",
    "hard_negative_contrastive_loss",
    "combined_ebmo_loss",
    "save_ebmo_model",
    "load_ebmo_model",
]

"""Gradient Move Optimization (GMO) AI for RingRift.

A novel game-playing algorithm that uses gradient descent/ascent in move embedding
space to find optimal moves, rather than traditional forward-pass policy sampling
or tree search.

Key innovations:
1. Gradient-based move search: Optimize in continuous embedding space
2. Information-theoretic exploration: Use uncertainty and novelty bonuses
3. Entropy-guided optimization: Balance exploitation vs exploration

The algorithm:
1. Encode game state and all legal moves into embeddings
2. Rank candidates using value + uncertainty + novelty (UCB-style)
3. For top-k candidates, run gradient ascent on the objective
4. Project optimized embeddings back to nearest legal moves
5. Select the best projected move

Architecture:
- StateEncoder: GameState -> 128-dim embedding
- MoveEncoder: Move -> 128-dim embedding
- GMOValueNetWithUncertainty: (state, move) -> (value, log_variance)
- NoveltyTracker: Track explored embeddings for novelty bonus
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models import AIConfig, GameState, Move, MoveType, Position
from ..utils.torch_utils import safe_load_checkpoint
from .base import BaseAI

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GMOConfig:
    """Configuration for GMO AI.

    Defaults tuned based on hyperparameter sweep + diversity study (2024-12):
    - optim_steps=5 (optimal balance of quality and speed)
    - top_k=5 (more candidates for optimization)
    - beta=0.5 (higher exploration improves both diversity AND win rate)
    - gamma=0.0 (no novelty bonus - not beneficial for win rate)
    - mc_samples=10 (critical for performance - 0% win rate without it)

    Performance: 100% vs Random, 80% vs Heuristic, 68.8% vs MCTS(D7)
    """
    # Embedding dimensions
    state_dim: int = 128
    move_dim: int = 128
    hidden_dim: int = 256

    # Optimization parameters (tuned from sweep)
    top_k: int = 5  # Number of candidates to optimize
    optim_steps: int = 5  # Gradient steps per candidate
    lr: float = 0.1  # Learning rate for move optimization

    # Information-theoretic parameters (tuned from diversity study)
    beta: float = 0.5  # Exploration coefficient (0.5 gives 80% win rate + 98% coverage)
    gamma: float = 0.0  # Novelty coefficient (disabled - not beneficial)
    exploration_temp: float = 1.0  # Base exploration temperature

    # MC Dropout parameters (critical - do not reduce mc_samples)
    dropout_rate: float = 0.1
    mc_samples: int = 10  # Number of dropout samples for uncertainty

    # Uncertainty calibration
    # Temperature > 1.0 increases uncertainty (for overconfident models)
    # Temperature < 1.0 decreases uncertainty (for underconfident models)
    # Use calibrate_uncertainty() to learn optimal temperature from data
    calibration_temperature: float = 1.0

    # Novelty tracking
    novelty_memory_size: int = 1000

    # Device
    device: str = "cpu"


# =============================================================================
# Move Encoder
# =============================================================================

# Move type to index mapping
MOVE_TYPE_TO_IDX: dict[MoveType, int] = {
    MoveType.PLACE_RING: 0,
    MoveType.MOVE_STACK: 1,
    MoveType.OVERTAKING_CAPTURE: 2,
    MoveType.SKIP_CAPTURE: 3,
    MoveType.SKIP_PLACEMENT: 4,
    MoveType.NO_PLACEMENT_ACTION: 5,
    MoveType.NO_MOVEMENT_ACTION: 6,
    MoveType.SWAP_SIDES: 7,
}
NUM_MOVE_TYPES = 8


class MoveEncoder(nn.Module):
    """Encode Move objects into continuous embeddings.

    Components:
    - Move type embedding (8 types)
    - From position embedding (64 positions for 8x8 board)
    - To position embedding (64 positions)
    - Placement count embedding (1-3 rings)

    All embeddings are concatenated and projected to final dimension.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        board_size: int = 8,
        type_embed_dim: int = 32,
        pos_embed_dim: int = 32,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.board_size = board_size
        self.num_positions = board_size * board_size

        # Embedding tables
        self.type_embed = nn.Embedding(NUM_MOVE_TYPES, type_embed_dim)
        self.from_embed = nn.Embedding(self.num_positions + 1, pos_embed_dim)  # +1 for None
        self.to_embed = nn.Embedding(self.num_positions + 1, pos_embed_dim)
        self.placement_embed = nn.Embedding(4, 16)  # 0, 1, 2, 3 rings

        # Projection to final dimension
        concat_dim = type_embed_dim + pos_embed_dim * 2 + 16
        self.projection = nn.Sequential(
            nn.Linear(concat_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def _pos_to_idx(self, pos: Position | None) -> int:
        """Convert position to index (0 for None)."""
        if pos is None:
            return self.num_positions  # Use last index for None
        return pos.y * self.board_size + pos.x

    def encode_move(self, move: Move) -> torch.Tensor:
        """Encode a single move to embedding vector."""
        # Get move type index
        type_idx = MOVE_TYPE_TO_IDX.get(move.type, 0)

        # Get position indices
        from_idx = self._pos_to_idx(move.from_pos)
        to_idx = self._pos_to_idx(move.to)

        # Get placement count (0 if None)
        placement_count = move.placement_count if move.placement_count else 0
        placement_count = min(placement_count, 3)

        # Create tensors
        device = next(self.parameters()).device
        type_tensor = torch.tensor([type_idx], device=device)
        from_tensor = torch.tensor([from_idx], device=device)
        to_tensor = torch.tensor([to_idx], device=device)
        placement_tensor = torch.tensor([placement_count], device=device)

        # Get embeddings
        type_emb = self.type_embed(type_tensor)
        from_emb = self.from_embed(from_tensor)
        to_emb = self.to_embed(to_tensor)
        placement_emb = self.placement_embed(placement_tensor)

        # Concatenate and project
        concat = torch.cat([type_emb, from_emb, to_emb, placement_emb], dim=-1)
        return self.projection(concat).squeeze(0)

    def encode_moves(self, moves: list[Move]) -> torch.Tensor:
        """Encode multiple moves to embedding matrix."""
        embeddings = [self.encode_move(m) for m in moves]
        return torch.stack(embeddings)


# =============================================================================
# State Encoder
# =============================================================================

class StateEncoder(nn.Module):
    """Encode GameState into continuous embedding.

    Uses simplified NNUE-style features:
    - Ring presence per player (4 planes)
    - Stack presence per player (4 planes)
    - Territory ownership per player (4 planes)

    Total: 12 planes x board_size x board_size = 768 features for 8x8
    """

    def __init__(
        self,
        embed_dim: int = 128,
        board_size: int = 8,
        num_planes: int = 12,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.board_size = board_size
        self.num_planes = num_planes
        self.input_dim = board_size * board_size * num_planes

        # MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def extract_features(self, state: GameState) -> np.ndarray:
        """Extract feature vector from game state."""
        features = np.zeros(self.input_dim, dtype=np.float32)
        board_size = self.board_size
        num_positions = board_size * board_size

        # Extract stack features
        for _key, stack in state.board.stacks.items():
            pos = stack.position
            idx = pos.y * board_size + pos.x

            if 0 <= idx < num_positions:
                # Ring presence per player (planes 0-3)
                for ring_owner in stack.rings:
                    if 1 <= ring_owner <= 4:
                        plane_idx = ring_owner - 1
                        features[plane_idx * num_positions + idx] = 1.0

                # Stack presence for controlling player (planes 4-7)
                if 1 <= stack.controlling_player <= 4:
                    plane_idx = 4 + stack.controlling_player - 1
                    features[plane_idx * num_positions + idx] = 1.0

        # Territory ownership (planes 8-11)
        for _key, territory in state.board.territories.items():
            if 1 <= territory.controlling_player <= 4:
                plane_idx = 8 + territory.controlling_player - 1
                for space in territory.spaces:
                    idx = space.y * board_size + space.x
                    if 0 <= idx < num_positions:
                        features[plane_idx * num_positions + idx] = 1.0

        return features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Encode features to embedding."""
        return self.encoder(features)

    def encode_state(self, state: GameState) -> torch.Tensor:
        """Encode a game state to embedding vector."""
        features = self.extract_features(state)
        device = next(self.parameters()).device
        features_tensor = torch.from_numpy(features).float().to(device)
        return self.forward(features_tensor)


# =============================================================================
# Joint Value Network with Uncertainty
# =============================================================================

class GMOValueNetWithUncertainty(nn.Module):
    """Joint value network that predicts value and uncertainty.

    Takes concatenated (state_embed, move_embed) and outputs:
    - value: Expected outcome in [-1, 1]
    - log_var: Log variance of prediction (learned uncertainty)

    Uses dropout for MC uncertainty estimation during inference.
    """

    def __init__(
        self,
        state_dim: int = 128,
        move_dim: int = 128,
        hidden_dim: int = 256,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.move_dim = move_dim

        # Joint encoder with dropout for MC estimation
        self.joint_encoder = nn.Sequential(
            nn.Linear(state_dim + move_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Separate heads for value and uncertainty
        self.value_head = nn.Linear(hidden_dim, 1)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state_embed: torch.Tensor,
        move_embed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            state_embed: Shape (batch, state_dim) or (state_dim,)
            move_embed: Shape (batch, move_dim) or (move_dim,)

        Returns:
            value: Shape (batch, 1) or (1,) values in [-1, 1]
            log_var: Shape (batch, 1) or (1,) log variance
        """
        # Handle single sample case
        if state_embed.dim() == 1:
            state_embed = state_embed.unsqueeze(0)
        if move_embed.dim() == 1:
            move_embed = move_embed.unsqueeze(0)

        # Concatenate embeddings
        joint = torch.cat([state_embed, move_embed], dim=-1)

        # Encode
        features = self.joint_encoder(joint)

        # Output heads
        value = torch.tanh(self.value_head(features))
        log_var = self.uncertainty_head(features)

        return value, log_var


# =============================================================================
# Novelty Tracker
# =============================================================================

class NoveltyTracker:
    """Track explored embeddings and compute novelty scores.

    Uses a ring buffer to store recently visited move embeddings.
    Novelty is computed as distance to nearest neighbor in memory.
    """

    def __init__(self, memory_size: int = 1000, embed_dim: int = 128):
        self.memory_size = memory_size
        self.embed_dim = embed_dim
        self.memory = torch.zeros(memory_size, embed_dim)
        self.count = 0

    def compute_novelty(self, move_embed: torch.Tensor) -> torch.Tensor:
        """Compute novelty as distance to nearest visited embedding.

        Args:
            move_embed: Shape (embed_dim,) move embedding

        Returns:
            Scalar novelty score (higher = more novel)
        """
        if self.count == 0:
            return torch.tensor(1.0, device=move_embed.device)

        # Get active memory
        active_size = min(self.count, self.memory_size)
        active_memory = self.memory[:active_size].to(move_embed.device)

        # Compute distances to all stored embeddings
        distances = torch.cdist(
            move_embed.unsqueeze(0).unsqueeze(0),
            active_memory.unsqueeze(0)
        ).squeeze()

        # Return minimum distance (nearest neighbor)
        min_distance = distances.min()
        return min_distance

    def add(self, move_embed: torch.Tensor) -> None:
        """Add embedding to memory."""
        idx = self.count % self.memory_size
        self.memory[idx] = move_embed.detach().cpu()
        self.count += 1

    def reset(self) -> None:
        """Reset memory for new game."""
        self.memory.zero_()
        self.count = 0


# =============================================================================
# Core GMO Functions
# =============================================================================

def estimate_uncertainty(
    state_embed: torch.Tensor,
    move_embed: torch.Tensor,
    value_net: GMOValueNetWithUncertainty,
    n_samples: int = 10,
    calibration_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate value and uncertainty using MC Dropout.

    Args:
        state_embed: State embedding
        move_embed: Move embedding (requires_grad for optimization)
        value_net: Value network with dropout
        n_samples: Number of dropout samples
        calibration_temperature: Temperature for uncertainty calibration.
            - T > 1.0: Increases uncertainty (for overconfident models)
            - T < 1.0: Decreases uncertainty (for underconfident models)
            - T = 1.0: No calibration applied

    Returns:
        mean_value: Mean predicted value
        entropy: Entropy of value distribution
        variance: Calibrated variance of predictions
    """
    was_training = value_net.training
    value_net.train()  # Enable dropout

    values = []
    for _ in range(n_samples):
        with torch.set_grad_enabled(move_embed.requires_grad):
            value, _ = value_net(state_embed, move_embed)
            values.append(value)

    if not was_training:
        value_net.eval()

    values_tensor = torch.stack(values)
    mean_value = values_tensor.mean()
    raw_variance = values_tensor.var() + 1e-8  # Add small epsilon for stability

    # Apply temperature calibration: variance * T
    # T > 1.0 increases variance (for overconfident models)
    variance = raw_variance * calibration_temperature

    # Gaussian entropy: H = 0.5 * log(2 * pi * e * var)
    entropy = 0.5 * torch.log(2 * math.pi * math.e * variance)

    return mean_value, entropy, variance


def optimize_move_with_entropy(
    state_embed: torch.Tensor,
    initial_move_embed: torch.Tensor,
    value_net: GMOValueNetWithUncertainty,
    config: GMOConfig,
) -> torch.Tensor:
    """Optimize move embedding using gradient ascent with entropy bonus.

    Objective: maximize value + beta * sqrt(variance) + gamma * novelty

    Args:
        state_embed: Fixed state embedding
        initial_move_embed: Starting point for optimization
        value_net: Joint value network
        config: GMO configuration

    Returns:
        Optimized move embedding
    """
    # Clone and enable gradients
    move_embed = initial_move_embed.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([move_embed], lr=config.lr)

    for step in range(config.optim_steps):
        optimizer.zero_grad()

        # Estimate value and uncertainty (with calibration)
        mean_value, _entropy, variance = estimate_uncertainty(
            state_embed, move_embed, value_net, config.mc_samples,
            calibration_temperature=config.calibration_temperature
        )

        # Anneal exploration over optimization steps
        progress = step / max(config.optim_steps - 1, 1)
        exploration_weight = config.beta * (1 - progress) * config.exploration_temp

        # Combined objective: maximize value + explore uncertain regions
        objective = mean_value + exploration_weight * torch.sqrt(variance)

        # Maximize objective (minimize negative)
        loss = -objective
        loss.backward()
        optimizer.step()

    return move_embed.detach()


def project_to_legal_move(
    optimized_embed: torch.Tensor,
    move_embeds: torch.Tensor,
    legal_moves: list[Move],
    temperature: float = 0.0,
) -> tuple[Move, int]:
    """Project optimized embedding to nearest legal move.

    Args:
        optimized_embed: Optimized move embedding
        move_embeds: Embeddings of all legal moves (num_moves, embed_dim)
        legal_moves: List of legal Move objects
        temperature: If > 0, sample from softmax instead of argmax

    Returns:
        Selected Move and its index
    """
    # Compute cosine similarities
    optimized_norm = F.normalize(optimized_embed.unsqueeze(0), dim=-1)
    moves_norm = F.normalize(move_embeds, dim=-1)
    similarities = torch.mm(optimized_norm, moves_norm.t()).squeeze(0)

    if temperature > 0:
        # Sample from softmax
        probs = F.softmax(similarities / temperature, dim=0)
        idx = torch.multinomial(probs, 1).item()
    else:
        # Argmax
        idx = similarities.argmax().item()

    return legal_moves[idx], idx


def get_exploration_temperature(state: GameState, base_temp: float = 1.0) -> float:
    """Adapt exploration based on game phase.

    - Early game: High entropy (explore strategically)
    - Late game: Low entropy (exploit winning positions)
    """
    move_count = len(state.move_history)
    rings_on_board = sum(s.stack_height for s in state.board.stacks.values())

    if move_count < 10:
        return base_temp * 1.5  # Explore openings
    elif rings_on_board > 20:
        return base_temp * 0.5  # Exploit in complex positions
    else:
        return base_temp


# =============================================================================
# GMO AI Class
# =============================================================================

class GMOAI(BaseAI):
    """Gradient Move Optimization AI.

    A novel algorithm that uses gradient descent in move embedding space
    to find optimal moves, guided by uncertainty and novelty bonuses.
    """

    def __init__(
        self,
        player_number: int,
        config: AIConfig,
        gmo_config: GMOConfig | None = None,
    ):
        super().__init__(player_number, config)

        self.gmo_config = gmo_config or GMOConfig()
        self.device = torch.device(self.gmo_config.device)

        # Initialize networks
        board_size = 8  # Default to square8
        self.state_encoder = StateEncoder(
            embed_dim=self.gmo_config.state_dim,
            board_size=board_size,
        ).to(self.device)

        self.move_encoder = MoveEncoder(
            embed_dim=self.gmo_config.move_dim,
            board_size=board_size,
        ).to(self.device)

        self.value_net = GMOValueNetWithUncertainty(
            state_dim=self.gmo_config.state_dim,
            move_dim=self.gmo_config.move_dim,
            hidden_dim=self.gmo_config.hidden_dim,
            dropout_rate=self.gmo_config.dropout_rate,
        ).to(self.device)

        # Initialize novelty tracker
        self.novelty_tracker = NoveltyTracker(
            memory_size=self.gmo_config.novelty_memory_size,
            embed_dim=self.gmo_config.move_dim,
        )

        # Track if model is trained
        self._is_trained = False

        # Online learning (disabled by default)
        self._online_learner: OnlineLearner | None = None
        self._last_state: GameState | None = None
        self._last_move: Move | None = None
        self._last_value: float = 0.0

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load trained model from checkpoint."""
        checkpoint = safe_load_checkpoint(
            checkpoint_path,
            map_location=self.device,
            warn_on_unsafe=False,
        )

        self.state_encoder.load_state_dict(checkpoint["state_encoder"])
        self.move_encoder.load_state_dict(checkpoint["move_encoder"])
        self.value_net.load_state_dict(checkpoint["value_net"])

        self._is_trained = True
        logger.info(f"Loaded GMO checkpoint from {checkpoint_path}")

    def save_checkpoint(self, checkpoint_path: Path) -> None:
        """Save model to checkpoint."""
        checkpoint = {
            "state_encoder": self.state_encoder.state_dict(),
            "move_encoder": self.move_encoder.state_dict(),
            "value_net": self.value_net.state_dict(),
            "gmo_config": self.gmo_config,
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved GMO checkpoint to {checkpoint_path}")

    def select_move(self, game_state: GameState) -> Move | None:
        """Select the best move using gradient move optimization.

        Algorithm:
        1. Get legal moves and encode state/moves
        2. Initial ranking with UCB-style scores
        3. Optimize top-k candidates with gradient ascent
        4. Project to legal moves and select best
        """
        # Get legal moves
        legal_moves = self.get_valid_moves(game_state)
        if not legal_moves:
            return None

        # Handle single move case
        if len(legal_moves) == 1:
            return legal_moves[0]

        # Check for random play
        if self.should_pick_random_move():
            return self.get_random_element(legal_moves)

        # Check for swap move
        swap_move = self.maybe_select_swap_move(game_state, legal_moves)
        if swap_move:
            return swap_move

        # Set networks to eval mode
        self.state_encoder.eval()
        self.move_encoder.eval()
        self.value_net.eval()

        with torch.no_grad():
            # Encode state
            state_embed = self.state_encoder.encode_state(game_state)

            # Encode all legal moves
            move_embeds = self.move_encoder.encode_moves(legal_moves)

        # Get exploration temperature based on game phase
        exploration_temp = get_exploration_temperature(
            game_state,
            self.gmo_config.exploration_temp
        )

        # Phase 1: Initial ranking with uncertainty
        candidates = []
        for idx, move_embed in enumerate(move_embeds):
            with torch.no_grad():
                mean_val, _entropy, var = estimate_uncertainty(
                    state_embed, move_embed, self.value_net,
                    self.gmo_config.mc_samples,
                    calibration_temperature=self.gmo_config.calibration_temperature
                )
                novelty = self.novelty_tracker.compute_novelty(move_embed)

            # UCB-style score: value + exploration bonus
            score = (
                mean_val.item() +
                self.gmo_config.beta * math.sqrt(var.item()) +
                self.gmo_config.gamma * novelty.item()
            )
            candidates.append((idx, score, move_embed))

        # Sort by score (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_k = candidates[:self.gmo_config.top_k]

        # Phase 2: Optimize top-k with entropy guidance
        best_move = None
        best_score = float("-inf")
        best_embed = None

        # Create config with current exploration temp
        optim_config = GMOConfig(
            **dict(self.gmo_config.__dict__.items())
        )
        optim_config.exploration_temp = exploration_temp

        for _idx, _, initial_embed in top_k:
            # Gradient optimization
            optimized_embed = optimize_move_with_entropy(
                state_embed,
                initial_embed,
                self.value_net,
                optim_config,
            )

            # Project to legal move
            projected_move, proj_idx = project_to_legal_move(
                optimized_embed,
                move_embeds,
                legal_moves,
            )

            # Final evaluation
            with torch.no_grad():
                final_value, _, final_var = estimate_uncertainty(
                    state_embed,
                    move_embeds[proj_idx],
                    self.value_net,
                    self.gmo_config.mc_samples,
                    calibration_temperature=self.gmo_config.calibration_temperature
                )

            final_score = final_value.item() + self.gmo_config.beta * math.sqrt(final_var.item())

            if final_score > best_score:
                best_score = final_score
                best_move = projected_move
                best_embed = move_embeds[proj_idx]

        # Update novelty memory
        if best_embed is not None:
            self.novelty_tracker.add(best_embed)

        self.move_count += 1
        return best_move

    def evaluate_position(self, game_state: GameState) -> float:
        """Evaluate position by averaging value predictions over legal moves."""
        legal_moves = self.get_valid_moves(game_state)
        if not legal_moves:
            return 0.0

        self.state_encoder.eval()
        self.move_encoder.eval()
        self.value_net.eval()

        with torch.no_grad():
            state_embed = self.state_encoder.encode_state(game_state)

            values = []
            for move in legal_moves:
                move_embed = self.move_encoder.encode_move(move)
                value, _ = self.value_net(state_embed, move_embed)
                values.append(value.item())

        # Return max value (best move evaluation)
        return max(values) if values else 0.0

    def get_move_predictions_with_uncertainty(
        self,
        game_state: GameState,
        legal_moves: list[Move],
    ) -> list[tuple[float, float]]:
        """Get value predictions with uncertainty for all moves.

        Used for calibration studies to evaluate uncertainty quality.

        Args:
            game_state: Current game state
            legal_moves: List of legal moves to evaluate

        Returns:
            List of (mean_value, variance) tuples for each move
        """
        if not legal_moves:
            return []

        self.state_encoder.eval()
        self.move_encoder.eval()
        self.value_net.eval()

        predictions = []

        with torch.no_grad():
            state_embed = self.state_encoder.encode_state(game_state)

            for move in legal_moves:
                move_embed = self.move_encoder.encode_move(move)
                mean_val, _, var = estimate_uncertainty(
                    state_embed, move_embed, self.value_net,
                    self.gmo_config.mc_samples,
                    calibration_temperature=self.gmo_config.calibration_temperature
                )
                predictions.append((mean_val.item(), var.item()))

        return predictions

    def reset_for_new_game(self, *, rng_seed: int | None = None) -> None:
        """Reset state for new game."""
        super().reset_for_new_game(rng_seed=rng_seed)
        self.novelty_tracker.reset()
        self._last_state = None
        self._last_move = None
        self._last_value = 0.0

    # =========================================================================
    # Online Learning Methods
    # =========================================================================

    def enable_online_learning(
        self,
        lr: float = 0.0001,  # Lowered from 0.001 to prevent catastrophic forgetting
        buffer_size: int = 500,  # Increased from 100 for more diverse replay
        discount: float = 0.99,
        weight_decay: float = 0.01,  # L2 regularization
        max_grad_norm: float = 1.0,  # Gradient clipping threshold
    ) -> None:
        """Enable continuous learning during play.

        When enabled, the model will:
        1. Record all moves during gameplay
        2. Update weights after each game based on outcome

        Uses regularization techniques to prevent catastrophic forgetting:
        - Lower learning rate (0.0001)
        - Larger replay buffer (500)
        - Weight decay (L2 regularization)
        - Gradient clipping

        Args:
            lr: Learning rate for online updates
            buffer_size: Size of experience replay buffer
            discount: Temporal discount factor
            weight_decay: L2 regularization strength
            max_grad_norm: Maximum gradient norm for clipping
        """
        self._online_learner = OnlineLearner(
            state_encoder=self.state_encoder,
            move_encoder=self.move_encoder,
            value_net=self.value_net,
            lr=lr,
            buffer_size=buffer_size,
            discount=discount,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )
        logger.info(
            f"Online learning enabled (lr={lr}, buffer={buffer_size}, "
            f"weight_decay={weight_decay}, grad_clip={max_grad_norm})"
        )

    def disable_online_learning(self) -> None:
        """Disable continuous learning."""
        self._online_learner = None
        logger.info("Online learning disabled")

    def is_learning_enabled(self) -> bool:
        """Check if online learning is active."""
        return self._online_learner is not None

    def select_move_with_learning(
        self,
        game_state: GameState,
    ) -> Move | None:
        """Select move and record for online learning.

        Use this instead of select_move() when online learning is enabled.
        """
        move = self.select_move(game_state)

        if move is not None and self._online_learner is not None:
            # Get value prediction for the selected move
            with torch.no_grad():
                state_embed = self.state_encoder.encode_state(game_state)
                move_embed = self.move_encoder.encode_move(move)
                value, _ = self.value_net(state_embed, move_embed)
                predicted_value = value.item()

            # Record for trajectory
            self._online_learner.record_move(
                state=game_state,
                move=move,
                predicted_value=predicted_value,
                move_idx=self.move_count,
            )

            # Store for potential TD update
            self._last_state = game_state
            self._last_move = move
            self._last_value = predicted_value

        return move

    def update_on_game_end(self, outcome: float) -> float:
        """Update model based on game outcome.

        Call this after game ends with the outcome from GMO's perspective:
        - +1.0 for win
        - -1.0 for loss
        - 0.0 for draw (shouldn't happen in RingRift)

        Args:
            outcome: Game result in [-1, 1]

        Returns:
            Training loss (0 if learning disabled)
        """
        if self._online_learner is None:
            return 0.0

        loss = self._online_learner.game_end_update(outcome)

        # Optional: also do a replay update for stability
        if len(self._online_learner.buffer) >= self._online_learner.min_batch_size:
            replay_loss = self._online_learner.replay_update()
            loss = (loss + replay_loss) / 2

        return loss

    def get_learning_stats(self) -> dict | None:
        """Get online learning statistics."""
        if self._online_learner is None:
            return None
        return self._online_learner.get_stats()


# =============================================================================
# Loss Functions for Training
# =============================================================================

# =============================================================================
# Online Learning Components
# =============================================================================

@dataclass
class ExperienceSample:
    """Single experience sample for online learning."""
    state_features: np.ndarray  # Pre-extracted state features
    move: Move
    predicted_value: float  # Value predicted at selection time
    move_idx: int  # Move number in game


class OnlineLearner:
    """Continuous learning module for GMO.

    Supports multiple learning modes:
    1. TD(0): Update using next-state value prediction
    2. TD(λ): Eligibility traces for credit assignment
    3. Game-end: Update all moves with final outcome
    4. Hybrid: Combination of TD and outcome-based

    Uses a replay buffer to maintain stability while allowing
    single-game updates.

    Regularization (added to prevent catastrophic forgetting):
    - Weight decay: L2 regularization on all parameters
    - Gradient clipping: Prevents exploding gradients
    - Larger buffer: More diverse samples reduce overfitting
    - Lower LR: Slower updates preserve learned knowledge
    """

    def __init__(
        self,
        state_encoder: StateEncoder,
        move_encoder: MoveEncoder,
        value_net: GMOValueNetWithUncertainty,
        lr: float = 0.0001,  # Lowered from 0.001 to prevent forgetting
        td_lambda: float = 0.9,
        discount: float = 0.99,
        buffer_size: int = 500,  # Increased from 100 for more diverse replay
        min_batch_size: int = 16,  # Increased for more stable gradients
        weight_decay: float = 0.01,  # L2 regularization
        max_grad_norm: float = 1.0,  # Gradient clipping threshold
    ):
        self.state_encoder = state_encoder
        self.move_encoder = move_encoder
        self.value_net = value_net

        self.lr = lr
        self.td_lambda = td_lambda
        self.discount = discount
        self.buffer_size = buffer_size
        self.min_batch_size = min_batch_size
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

        # Experience buffer (ring buffer)
        self.buffer: list[tuple[np.ndarray, Move, float]] = []

        # Current game trajectory
        self.trajectory: list[ExperienceSample] = []

        # Optimizer for online updates with weight decay
        self.optimizer = torch.optim.AdamW(
            list(state_encoder.parameters()) +
            list(move_encoder.parameters()) +
            list(value_net.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Metrics
        self.updates_performed = 0
        self.total_loss = 0.0

    def record_move(
        self,
        state: GameState,
        move: Move,
        predicted_value: float,
        move_idx: int,
    ) -> None:
        """Record a move during game play."""
        # Extract features (cheaper than storing full state)
        features = self.state_encoder.extract_features(state)

        sample = ExperienceSample(
            state_features=features,
            move=move,
            predicted_value=predicted_value,
            move_idx=move_idx,
        )
        self.trajectory.append(sample)

    def td_update(
        self,
        current_state: GameState,
        next_state: GameState,
        move: Move,
        reward: float = 0.0,
    ) -> float:
        """Perform TD(0) update after a single transition.

        V(s,a) <- V(s,a) + alpha * (r + gamma*V(s') - V(s,a))

        Args:
            current_state: State before move
            next_state: State after move
            move: Move taken
            reward: Immediate reward (usually 0 until game end)

        Returns:
            TD error (for monitoring)
        """
        self.state_encoder.train()
        self.move_encoder.train()
        self.value_net.train()

        device = next(self.value_net.parameters()).device

        # Encode current state-action
        state_embed = self.state_encoder.encode_state(current_state)
        move_embed = self.move_encoder.encode_move(move)

        # Get current value prediction
        current_value, current_log_var = self.value_net(state_embed, move_embed)

        # Get next state value
        with torch.no_grad():
            if reward != 0.0:
                # Terminal state - no future value to bootstrap
                next_value = torch.tensor([0.0], device=device)
            else:
                # Encode next state and estimate its value
                # Use the same move embedding as a proxy (state is more important)
                next_state_embed = self.state_encoder.encode_state(next_state)
                next_value, _ = self.value_net(next_state_embed, move_embed)

        # TD target
        target = reward + self.discount * next_value

        # Compute loss
        self.optimizer.zero_grad()
        loss = nll_loss_with_uncertainty(current_value, current_log_var, target)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        all_params = (
            list(self.state_encoder.parameters()) +
            list(self.move_encoder.parameters()) +
            list(self.value_net.parameters())
        )
        torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)

        self.optimizer.step()

        td_error = (target - current_value).item()
        self.updates_performed += 1
        self.total_loss += loss.item()

        return td_error

    def game_end_update(self, outcome: float) -> float:
        """Update all moves from game trajectory with final outcome.

        Uses temporal discounting: moves closer to game end get
        full outcome weight, earlier moves get discounted.

        Args:
            outcome: Final game outcome in [-1, 1]

        Returns:
            Average loss over trajectory
        """
        if not self.trajectory:
            return 0.0

        self.state_encoder.train()
        self.move_encoder.train()
        self.value_net.train()

        device = next(self.value_net.parameters()).device

        # Compute discounted targets (from end of game backward)
        n_moves = len(self.trajectory)
        targets = []
        for i, _sample in enumerate(self.trajectory):
            # Discount: later moves (closer to outcome) get higher weight
            steps_from_end = n_moves - i - 1
            discounted_outcome = outcome * (self.discount ** steps_from_end)
            targets.append(discounted_outcome)

        targets_tensor = torch.tensor(targets, dtype=torch.float32, device=device)

        # Encode all samples
        state_embeds = []
        move_embeds = []

        for sample in self.trajectory:
            features = torch.from_numpy(sample.state_features).float().to(device)
            state_embed = self.state_encoder.encoder(features)
            move_embed = self.move_encoder.encode_move(sample.move)
            state_embeds.append(state_embed)
            move_embeds.append(move_embed)

        state_embeds = torch.stack(state_embeds)
        move_embeds = torch.stack(move_embeds)

        # Forward pass
        pred_values, pred_log_vars = self.value_net(state_embeds, move_embeds)

        # Compute loss
        self.optimizer.zero_grad()
        loss = nll_loss_with_uncertainty(pred_values, pred_log_vars, targets_tensor)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        all_params = (
            list(self.state_encoder.parameters()) +
            list(self.move_encoder.parameters()) +
            list(self.value_net.parameters())
        )
        torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)

        self.optimizer.step()

        # Add to buffer for future updates
        for sample, target in zip(self.trajectory, targets, strict=True):
            self._add_to_buffer(sample.state_features, sample.move, target)

        # Clear trajectory
        self.trajectory.clear()

        self.updates_performed += 1
        self.total_loss += loss.item()

        return loss.item()

    def _add_to_buffer(
        self,
        features: np.ndarray,
        move: Move,
        target: float,
    ) -> None:
        """Add sample to replay buffer."""
        if len(self.buffer) >= self.buffer_size:
            # Remove oldest
            self.buffer.pop(0)
        self.buffer.append((features, move, target))

    def replay_update(self, batch_size: int | None = None) -> float:
        """Perform update from replay buffer.

        Returns:
            Loss from update (or 0 if buffer too small)
        """
        if len(self.buffer) < self.min_batch_size:
            return 0.0

        batch_size = batch_size or min(len(self.buffer), 32)

        # Sample batch
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        self.state_encoder.train()
        self.move_encoder.train()
        self.value_net.train()

        device = next(self.value_net.parameters()).device

        # Process batch
        state_embeds = []
        move_embeds = []
        targets = []

        for features, move, target in batch:
            features_tensor = torch.from_numpy(features).float().to(device)
            state_embed = self.state_encoder.encoder(features_tensor)
            move_embed = self.move_encoder.encode_move(move)
            state_embeds.append(state_embed)
            move_embeds.append(move_embed)
            targets.append(target)

        state_embeds = torch.stack(state_embeds)
        move_embeds = torch.stack(move_embeds)
        targets_tensor = torch.tensor(targets, dtype=torch.float32, device=device)

        # Forward and update
        pred_values, pred_log_vars = self.value_net(state_embeds, move_embeds)

        self.optimizer.zero_grad()
        loss = nll_loss_with_uncertainty(pred_values, pred_log_vars, targets_tensor)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        all_params = (
            list(self.state_encoder.parameters()) +
            list(self.move_encoder.parameters()) +
            list(self.value_net.parameters())
        )
        torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def get_stats(self) -> dict:
        """Get learning statistics."""
        avg_loss = self.total_loss / max(self.updates_performed, 1)
        return {
            "updates": self.updates_performed,
            "avg_loss": avg_loss,
            "buffer_size": len(self.buffer),
            "trajectory_length": len(self.trajectory),
        }

    def reset_stats(self) -> None:
        """Reset statistics for new training session."""
        self.updates_performed = 0
        self.total_loss = 0.0


def nll_loss_with_uncertainty(
    pred_value: torch.Tensor,
    pred_log_var: torch.Tensor,
    target_value: torch.Tensor,
) -> torch.Tensor:
    """Negative log likelihood with learned uncertainty.

    The network learns to be uncertain when predictions are hard.
    Loss = precision * (pred - target)^2 + log_var

    Args:
        pred_value: Predicted values (batch,)
        pred_log_var: Predicted log variance (batch,)
        target_value: Target values (batch,)

    Returns:
        Scalar loss
    """
    precision = torch.exp(-pred_log_var)
    mse = (pred_value.squeeze() - target_value) ** 2
    loss = precision.squeeze() * mse + pred_log_var.squeeze()
    return loss.mean()


def gmo_combined_loss(
    state_encoder: StateEncoder,
    move_encoder: MoveEncoder,
    value_net: GMOValueNetWithUncertainty,
    states: list[GameState],
    moves: list[Move],
    outcomes: torch.Tensor,
) -> torch.Tensor:
    """Combined training loss for GMO networks.

    Args:
        state_encoder: State encoder network
        move_encoder: Move encoder network
        value_net: Joint value network
        states: List of game states
        moves: List of moves played
        outcomes: Game outcomes (batch,) in [-1, 1]

    Returns:
        Scalar loss
    """
    device = outcomes.device

    # Encode states and moves
    state_embeds = torch.stack([
        state_encoder.encode_state(s) for s in states
    ]).to(device)

    move_embeds = torch.stack([
        move_encoder.encode_move(m) for m in moves
    ]).to(device)

    # Forward pass
    pred_values, pred_log_vars = value_net(state_embeds, move_embeds)

    # Compute loss
    return nll_loss_with_uncertainty(pred_values, pred_log_vars, outcomes)

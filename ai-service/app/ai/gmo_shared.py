"""GMO Shared Components - Base classes and utilities for GMO variants.

.. note:: Supporting Deprecated Code (December 2025)
    This module provides shared utilities for the deprecated GMO family.
    The GMO AI variants (gmo_ai, gmo_v2, ig_gmo) are deprecated and will be
    removed in Q2 2026. This shared module remains to support existing code
    during the deprecation period.

This module provides shared abstractions for the GMO AI family:
- gmo_ai.py (GMO v1): Foundation with entropy-guided optimization
- gmo_v2.py (GMO v2): Enhanced with attention and ensemble voting
- ig_gmo.py (IG-GMO): Research variant with mutual information exploration

The shared components include:
1. Abstract base classes for encoders, value networks, and optimizers
2. Factory functions for creating variant-specific implementations
3. Shared utility functions (uncertainty estimation, projection, etc.)
4. Common configuration base class

Usage:
    from app.ai.gmo_shared import (
        GMOBaseConfig,
        StateEncoderBase,
        MoveEncoderBase,
        ValueNetBase,
        NoveltyTracker,
        estimate_uncertainty,
        project_to_legal_move,
    )

    # Create encoder
    encoder = create_state_encoder(variant="v1", embed_dim=128)

    # Use shared utilities
    uncertainty = estimate_uncertainty(
        logits, mode="variance", calibration_temperature=500.0
    )
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from ..models import GameState, Move

logger = logging.getLogger(__name__)


# =============================================================================
# Base Configuration
# =============================================================================

@dataclass
class GMOBaseConfig:
    """Base configuration for all GMO variants.

    Provides common parameters with sensible defaults. Variant-specific
    parameters can be added via subclassing or the variant_config dict.
    """
    # Core dimensions
    state_dim: int = 128
    move_dim: int = 128
    hidden_dim: int = 256

    # Optimization parameters
    top_k: int = 5  # Number of candidates to optimize
    optim_steps: int = 5  # Gradient steps per candidate
    lr: float = 0.1  # Learning rate for move optimization

    # Information-theoretic parameters
    beta: float = 0.5  # Exploration coefficient (UCB-style)
    gamma: float = 0.0  # Novelty coefficient

    # MC Dropout for Bayesian uncertainty
    dropout_rate: float = 0.1
    mc_samples: int = 10  # Number of forward passes for uncertainty

    # Uncertainty calibration
    calibration_temperature: float = 1.0  # Scale uncertainty estimates

    # Novelty tracking
    novelty_memory_size: int = 1000

    # Variant selection (for factory)
    variant: Literal["base", "v2", "ig_gmo"] = "base"

    # Variant-specific overrides (flexible dict)
    variant_config: dict[str, Any] = field(default_factory=dict)

    # Device
    device: str = "cpu"


# =============================================================================
# Abstract Base Classes
# =============================================================================

class StateEncoderBase(nn.Module, ABC):
    """Abstract base class for state encoders.

    State encoders convert GameState objects into continuous embeddings
    that can be used by value networks and move optimizers.
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(self, state: GameState) -> torch.Tensor:
        """Encode game state to embedding.

        Args:
            state: Game state to encode

        Returns:
            State embedding tensor of shape (embed_dim,)
        """
        pass

    def batch_forward(self, states: list[GameState]) -> torch.Tensor:
        """Encode multiple states to embeddings.

        Args:
            states: List of game states

        Returns:
            Batch of embeddings of shape (batch_size, embed_dim)
        """
        embeddings = [self.forward(s) for s in states]
        return torch.stack(embeddings, dim=0)


class MoveEncoderBase(nn.Module, ABC):
    """Abstract base class for move encoders.

    Move encoders convert Move objects into continuous embeddings
    suitable for gradient-based optimization.
    """

    def __init__(self, embed_dim: int = 128, board_size: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.board_size = board_size
        self.num_positions = board_size * board_size

    @abstractmethod
    def forward(self, move: Move) -> torch.Tensor:
        """Encode move to embedding.

        Args:
            move: Move to encode

        Returns:
            Move embedding tensor of shape (embed_dim,)
        """
        pass

    def batch_forward(self, moves: list[Move]) -> torch.Tensor:
        """Encode multiple moves to embeddings.

        Args:
            moves: List of moves

        Returns:
            Batch of embeddings of shape (batch_size, embed_dim)
        """
        embeddings = [self.forward(m) for m in moves]
        return torch.stack(embeddings, dim=0)


class ValueNetBase(nn.Module, ABC):
    """Abstract base class for value networks.

    Value networks estimate the value (expected outcome) of a
    (state, move) pair, optionally with uncertainty estimates.
    """

    def __init__(self, state_dim: int, move_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.move_dim = move_dim
        self.hidden_dim = hidden_dim

    @abstractmethod
    def forward(
        self,
        state_embed: torch.Tensor,
        move_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute value and uncertainty for (state, move) pair.

        Args:
            state_embed: State embedding (batch_size, state_dim)
            move_embed: Move embedding (batch_size, move_dim)

        Returns:
            Tuple of (value, log_variance) tensors
        """
        pass


class MoveOptimizerBase(ABC):
    """Abstract base class for move optimizers.

    Move optimizers perform gradient-based optimization in move
    embedding space to find optimal moves.
    """

    def __init__(self, config: GMOBaseConfig):
        self.config = config

    @abstractmethod
    def optimize(
        self,
        state_embed: torch.Tensor,
        initial_move_embed: torch.Tensor,
        value_net: ValueNetBase,
    ) -> torch.Tensor:
        """Optimize move embedding via gradient ascent.

        Args:
            state_embed: State embedding (fixed during optimization)
            initial_move_embed: Starting point for optimization
            value_net: Value network for computing objective

        Returns:
            Optimized move embedding
        """
        pass


# =============================================================================
# Shared Components
# =============================================================================

class NoveltyTracker:
    """Track explored state-action embeddings for novelty bonus.

    Uses a ring buffer to maintain recent embeddings and compute
    novelty as distance to nearest neighbor.
    """

    def __init__(self, memory_size: int = 1000, embed_dim: int = 128):
        self.memory_size = memory_size
        self.embed_dim = embed_dim
        self.embeddings: deque[np.ndarray] = deque(maxlen=memory_size)

    def add(self, embedding: np.ndarray | torch.Tensor) -> None:
        """Add an embedding to the memory."""
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
        self.embeddings.append(embedding.flatten())

    def compute_novelty(self, embedding: np.ndarray | torch.Tensor) -> float:
        """Compute novelty score as distance to nearest neighbor.

        Returns:
            Novelty score (higher = more novel). Returns 1.0 if memory is empty.
        """
        if len(self.embeddings) == 0:
            return 1.0

        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
        embedding = embedding.flatten()

        # Compute minimum distance to any stored embedding
        memory_array = np.array(list(self.embeddings))
        distances = np.linalg.norm(memory_array - embedding, axis=1)
        min_distance = np.min(distances)

        # Normalize by embedding dimension for scale-invariance
        return min_distance / math.sqrt(self.embed_dim)

    def clear(self) -> None:
        """Clear all stored embeddings."""
        self.embeddings.clear()

    def __len__(self) -> int:
        return len(self.embeddings)


# =============================================================================
# Shared Utility Functions
# =============================================================================

def estimate_uncertainty(
    value_net: nn.Module,
    state_embed: torch.Tensor,
    move_embed: torch.Tensor,
    n_samples: int = 10,
    mode: Literal["variance", "mutual_info"] = "variance",
    calibration_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate value and uncertainty via MC Dropout.

    Args:
        value_net: Value network with dropout layers
        state_embed: State embedding (batch_size, state_dim)
        move_embed: Move embedding (batch_size, move_dim)
        n_samples: Number of MC dropout samples
        mode: "variance" for epistemic uncertainty, "mutual_info" for MI
        calibration_temperature: Scale factor for uncertainty

    Returns:
        Tuple of (mean_value, uncertainty)
    """
    was_training = value_net.training
    value_net.train()  # Enable dropout

    values = []
    for _ in range(n_samples):
        with torch.no_grad():
            value, log_var = value_net(state_embed, move_embed)
            values.append(value)

    value_net.train(was_training)

    values_stack = torch.stack(values, dim=0)  # (n_samples, batch)
    mean_value = values_stack.mean(dim=0)
    epistemic_var = values_stack.var(dim=0)

    if mode == "mutual_info":
        # Aleatoric variance from the model's log_var output
        aleatoric_var = torch.exp(log_var).clamp(min=1e-8)
        # Mutual information approximation
        uncertainty = 0.5 * torch.log(1 + epistemic_var / aleatoric_var)
    else:
        # Simple variance-based uncertainty
        uncertainty = torch.sqrt(epistemic_var)

    # Apply calibration temperature
    uncertainty = uncertainty * calibration_temperature

    return mean_value, uncertainty


def project_to_legal_move(
    optimized_embed: torch.Tensor,
    legal_move_embeds: torch.Tensor,
    method: Literal["cosine", "euclidean"] = "cosine",
) -> int:
    """Project optimized embedding to nearest legal move.

    Args:
        optimized_embed: Optimized move embedding (embed_dim,)
        legal_move_embeds: Embeddings of legal moves (n_moves, embed_dim)
        method: Distance metric for projection

    Returns:
        Index of nearest legal move
    """
    if method == "cosine":
        # Cosine similarity (higher = closer)
        opt_norm = F.normalize(optimized_embed.unsqueeze(0), p=2, dim=-1)
        legal_norm = F.normalize(legal_move_embeds, p=2, dim=-1)
        similarities = torch.mm(opt_norm, legal_norm.t()).squeeze(0)
        return int(similarities.argmax().item())
    else:
        # Euclidean distance (lower = closer)
        distances = torch.norm(legal_move_embeds - optimized_embed, dim=1)
        return int(distances.argmin().item())


def compute_ucb_score(
    value: torch.Tensor,
    uncertainty: torch.Tensor,
    novelty: torch.Tensor | None = None,
    beta: float = 0.5,
    gamma: float = 0.0,
) -> torch.Tensor:
    """Compute UCB-style score combining value, uncertainty, and novelty.

    Args:
        value: Expected value (batch_size,)
        uncertainty: Uncertainty estimate (batch_size,)
        novelty: Novelty bonus (batch_size,) or None
        beta: Exploration coefficient for uncertainty
        gamma: Exploration coefficient for novelty

    Returns:
        UCB score (batch_size,)
    """
    score = value + beta * uncertainty

    if novelty is not None and gamma > 0:
        score = score + gamma * novelty

    return score


def create_position_index(position, board_size: int) -> int:
    """Convert Position to flat index.

    Args:
        position: Position object with x, y attributes or None
        board_size: Board dimension

    Returns:
        Flat index (0 to board_size^2 - 1) or board_size^2 for None
    """
    if position is None:
        return board_size * board_size  # Special index for None
    return position.y * board_size + position.x


# =============================================================================
# Factory Functions
# =============================================================================

def create_gmo_config(
    variant: Literal["base", "v2", "ig_gmo"] = "base",
    **kwargs,
) -> GMOBaseConfig:
    """Create GMO configuration for specified variant.

    Args:
        variant: GMO variant ("base", "v2", or "ig_gmo")
        **kwargs: Override default parameters

    Returns:
        Configured GMOBaseConfig instance
    """
    # Set variant-specific defaults
    if variant == "v2":
        defaults = {
            "state_dim": 256,
            "move_dim": 256,
            "hidden_dim": 512,
            "top_k": 7,
            "optim_steps": 15,
            "beta": 0.3,
            "gamma": 0.1,
            "mc_samples": 12,
            "novelty_memory_size": 2000,
        }
    elif variant == "ig_gmo":
        defaults = {
            "beta": 0.2,
            "gamma": 0.05,
        }
    else:
        defaults = {}

    # Merge defaults with overrides
    config_params = {**defaults, **kwargs, "variant": variant}
    return GMOBaseConfig(**config_params)


# =============================================================================
# Shared Loss Functions
# =============================================================================

def nll_loss_with_uncertainty(
    value: torch.Tensor,
    log_variance: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood loss with learned uncertainty.

    This loss allows the model to learn both value prediction and
    uncertainty estimation jointly.

    Args:
        value: Predicted value (batch_size,)
        log_variance: Log variance prediction (batch_size,)
        target: Target value (batch_size,)

    Returns:
        Scalar loss
    """
    precision = torch.exp(-log_variance)
    mse = (value - target) ** 2
    loss = 0.5 * (precision * mse + log_variance)
    return loss.mean()


def entropy_objective(
    value: torch.Tensor,
    uncertainty: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Entropy-weighted objective for optimization.

    Combines value maximization with uncertainty-based exploration.

    Args:
        value: Expected value
        uncertainty: Uncertainty estimate
        temperature: Controls exploration-exploitation trade-off

    Returns:
        Objective to maximize
    """
    # Higher temperature = more exploration
    entropy_bonus = temperature * torch.log(uncertainty.clamp(min=1e-8))
    return value + entropy_bonus

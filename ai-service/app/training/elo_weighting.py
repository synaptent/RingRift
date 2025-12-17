"""ELO-Weighted Data Sampling for RingRift AI Training.

Weights training samples by opponent strength - games against stronger
opponents get higher weight, avoiding overfitting on weak-opponent games.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import Sampler, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class EloWeightConfig:
    """Configuration for ELO-based sample weighting."""
    base_elo: float = 1500.0  # Reference Elo
    elo_scale: float = 400.0  # Elo difference scaling
    min_weight: float = 0.2  # Minimum sample weight
    max_weight: float = 3.0  # Maximum sample weight
    normalize_weights: bool = True  # Normalize to mean=1


class EloWeightedSampler:
    """Weights training samples based on opponent Elo rating."""

    def __init__(
        self,
        sample_elos: np.ndarray,
        model_elo: float = 1500.0,
        config: Optional[EloWeightConfig] = None,
    ):
        """Initialize the ELO-weighted sampler.

        Args:
            sample_elos: Array of opponent Elo ratings for each sample
            model_elo: Current model's Elo rating
            config: Weighting configuration
        """
        self.sample_elos = sample_elos
        self.model_elo = model_elo
        self.config = config or EloWeightConfig()
        self.weights = self._compute_weights()

    def _compute_weights(self) -> np.ndarray:
        """Compute sample weights from Elo ratings."""
        config = self.config

        # Higher Elo opponents = higher weight
        elo_diff = self.sample_elos - self.model_elo

        # Sigmoid-like transformation: harder games get more weight
        # Games vs stronger opponents (positive diff) get higher weight
        raw_weights = 1.0 / (1.0 + np.exp(-elo_diff / config.elo_scale))

        # Scale to [0, 1] and then to [min_weight, max_weight]
        weights = config.min_weight + raw_weights * (config.max_weight - config.min_weight)

        # Normalize to mean=1 if requested
        if config.normalize_weights:
            weights = weights / weights.mean()

        return weights

    def update_model_elo(self, new_elo: float):
        """Update model Elo and recompute weights."""
        self.model_elo = new_elo
        self.weights = self._compute_weights()

    def sample(self, n_samples: int) -> np.ndarray:
        """Sample indices weighted by Elo."""
        probs = self.weights / self.weights.sum()
        return np.random.choice(len(self.weights), size=n_samples, replace=True, p=probs)

    def get_weight(self, idx: int) -> float:
        """Get weight for a specific sample."""
        return self.weights[idx]


class EloWeightedDataset:
    """Dataset wrapper that applies Elo-based weights during training."""

    def __init__(
        self,
        base_dataset: "Dataset",
        sample_elos: np.ndarray,
        model_elo: float = 1500.0,
    ):
        self.base_dataset = base_dataset
        self.sampler = EloWeightedSampler(sample_elos, model_elo)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        item = self.base_dataset[idx]
        weight = self.sampler.get_weight(idx)

        if isinstance(item, tuple):
            return (*item, weight)
        return item, weight


def compute_elo_weights(
    opponent_elos: np.ndarray,
    model_elo: float = 1500.0,
    elo_scale: float = 400.0,
) -> np.ndarray:
    """Compute sample weights from opponent Elo ratings.

    Args:
        opponent_elos: Opponent Elo for each sample
        model_elo: Current model Elo
        elo_scale: Scaling factor for Elo difference

    Returns:
        Normalized sample weights
    """
    elo_diff = opponent_elos - model_elo
    raw_weights = 1.0 / (1.0 + np.exp(-elo_diff / elo_scale))
    weights = 0.2 + raw_weights * 2.8  # [0.2, 3.0]
    return weights / weights.mean()

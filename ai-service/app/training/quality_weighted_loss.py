"""
Quality-Weighted Loss Functions for Training.

Resurrected from archive/deprecated_ai/ebmo_network.py (December 2025).

These loss functions weight samples based on MCTS visit counts or other
quality signals, enabling the model to focus on high-quality examples.

Key functions:
- quality_weighted_policy_loss: Weight policy loss by MCTS visit fractions
- ranking_loss_from_quality: Pairwise ranking for quality ordering
- compute_quality_weights: Convert visit counts to normalized weights
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_quality_weights(
    visit_counts: torch.Tensor,
    temperature: float = 1.0,
    min_weight: float = 0.1,
) -> torch.Tensor:
    """Convert MCTS visit counts to quality weights.

    Higher visit counts indicate higher-quality moves (MCTS spent more
    time exploring them). We convert to normalized weights for loss scaling.

    Args:
        visit_counts: (B,) tensor of visit counts per sample
        temperature: Scaling factor (higher = more uniform weights)
        min_weight: Minimum weight to prevent zero-weight samples

    Returns:
        (B,) normalized quality weights in [min_weight, 1.0]
    """
    if visit_counts.numel() == 0:
        return torch.ones(0, device=visit_counts.device)

    # Normalize by max to get fractions in [0, 1]
    max_visits = visit_counts.max()
    if max_visits == 0:
        return torch.ones_like(visit_counts)

    quality_fractions = visit_counts.float() / max_visits

    # Apply temperature scaling
    if temperature != 1.0:
        quality_fractions = quality_fractions.pow(1.0 / temperature)

    # Clamp to minimum weight
    weights = torch.clamp(quality_fractions, min=min_weight)

    # Normalize so mean = 1 (preserves effective batch size)
    weights = weights / weights.mean()

    return weights


def quality_weighted_policy_loss(
    policy_log_probs: torch.Tensor,
    policy_targets: torch.Tensor,
    quality_weights: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute policy loss weighted by quality scores.

    Higher quality samples (more MCTS visits) contribute more to the loss,
    focusing learning on moves that MCTS deemed more important.

    Args:
        policy_log_probs: (B, A) log probabilities from model
        policy_targets: (B, A) target probabilities (soft or one-hot)
        quality_weights: (B,) quality weights per sample
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Weighted policy loss
    """
    # Compute per-sample cross entropy
    per_sample_loss = -(policy_targets * policy_log_probs).sum(dim=1)

    # Apply quality weights
    weighted_loss = per_sample_loss * quality_weights

    if reduction == "mean":
        # Mask for valid samples (non-zero target sum)
        valid_mask = policy_targets.sum(dim=1) > 0
        if valid_mask.any():
            return weighted_loss[valid_mask].mean()
        return torch.tensor(0.0, device=policy_log_probs.device)
    elif reduction == "sum":
        return weighted_loss.sum()
    else:
        return weighted_loss


def quality_weighted_value_loss(
    value_pred: torch.Tensor,
    value_target: torch.Tensor,
    quality_weights: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute value loss weighted by quality scores.

    Args:
        value_pred: (B,) or (B, P) predicted values
        value_target: (B,) or (B, P) target values
        quality_weights: (B,) quality weights per sample
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Weighted value loss
    """
    # Compute per-sample MSE
    if value_pred.ndim == 2:
        per_sample_loss = F.mse_loss(value_pred, value_target, reduction="none").mean(dim=1)
    else:
        per_sample_loss = F.mse_loss(value_pred.view(-1), value_target.view(-1), reduction="none")

    # Apply quality weights
    weighted_loss = per_sample_loss * quality_weights

    if reduction == "mean":
        return weighted_loss.mean()
    elif reduction == "sum":
        return weighted_loss.sum()
    else:
        return weighted_loss


def ranking_loss_from_quality(
    policy_log_probs: torch.Tensor,
    quality_scores: torch.Tensor,
    margin: float = 0.5,
    max_pairs: int = 100,
) -> torch.Tensor:
    """Pairwise ranking loss from quality scores.

    Ensures higher-quality moves have higher policy probabilities
    by a specified margin. This provides additional gradient signal
    for relative move ordering.

    Resurrected from archive/deprecated_ai/ebmo_network.py:1054-1093.

    Args:
        policy_log_probs: (B, A) log probabilities from model
        quality_scores: (B,) quality scores in [0, 1] (higher = better)
        margin: Minimum log-prob gap required between pairs
        max_pairs: Maximum number of pairs to sample (for efficiency)

    Returns:
        Scalar ranking loss value
    """
    batch_size = policy_log_probs.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=policy_log_probs.device)

    # Get the max log prob per sample (as a proxy for "chosen action confidence")
    max_log_probs = policy_log_probs.max(dim=1).values

    # Create pairs efficiently (sample if too many)
    n_pairs = batch_size * (batch_size - 1) // 2
    if n_pairs > max_pairs:
        # Sample random pairs
        indices = torch.randint(0, batch_size, (max_pairs, 2), device=policy_log_probs.device)
        # Ensure i != j
        mask = indices[:, 0] != indices[:, 1]
        indices = indices[mask][:max_pairs]
        i_indices, j_indices = indices[:, 0], indices[:, 1]
    else:
        # Use all pairs
        i_indices = []
        j_indices = []
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                i_indices.append(i)
                j_indices.append(j)
        i_indices = torch.tensor(i_indices, device=policy_log_probs.device)
        j_indices = torch.tensor(j_indices, device=policy_log_probs.device)

    if len(i_indices) == 0:
        return torch.tensor(0.0, device=policy_log_probs.device)

    # Get quality scores and log probs for pairs
    qi = quality_scores[i_indices]
    qj = quality_scores[j_indices]
    log_pi = max_log_probs[i_indices]
    log_pj = max_log_probs[j_indices]

    # For pairs where qi > qj, i should have higher log prob
    # Loss = relu(log_pj - log_pi + margin) when qi > qj
    # Loss = relu(log_pi - log_pj + margin) when qj > qi
    higher_i = qi > qj
    higher_j = qj > qi

    loss_i = F.relu(log_pj - log_pi + margin) * higher_i.float()
    loss_j = F.relu(log_pi - log_pj + margin) * higher_j.float()

    total_loss = loss_i + loss_j

    # Average over valid pairs (where quality differs)
    valid_pairs = higher_i | higher_j
    if valid_pairs.any():
        return total_loss[valid_pairs].mean()

    return torch.tensor(0.0, device=policy_log_probs.device)


class QualityWeightedTrainer:
    """Mixin class for quality-weighted training.

    Add this to your training loop to enable quality-weighted losses.

    Usage:
        trainer = QualityWeightedTrainer(
            quality_weight=0.5,  # Blend between uniform and quality-weighted
            ranking_weight=0.1,  # Weight for ranking loss
        )

        # In training loop:
        policy_loss = trainer.compute_weighted_policy_loss(
            policy_log_probs, policy_targets, quality_weights
        )
    """

    def __init__(
        self,
        quality_weight: float = 0.5,
        ranking_weight: float = 0.1,
        ranking_margin: float = 0.5,
        min_quality_weight: float = 0.1,
        temperature: float = 1.0,
    ):
        """Initialize quality-weighted trainer.

        Args:
            quality_weight: Blend factor [0, 1]. 0 = uniform, 1 = fully quality-weighted
            ranking_weight: Weight for ranking loss term
            ranking_margin: Margin for ranking loss
            min_quality_weight: Minimum sample weight
            temperature: Temperature for quality weight computation
        """
        self.quality_weight = quality_weight
        self.ranking_weight = ranking_weight
        self.ranking_margin = ranking_margin
        self.min_quality_weight = min_quality_weight
        self.temperature = temperature

        # Statistics tracking
        self.quality_stats = {
            "mean_weight": 0.0,
            "std_weight": 0.0,
            "ranking_loss": 0.0,
        }

    def compute_weighted_policy_loss(
        self,
        policy_log_probs: torch.Tensor,
        policy_targets: torch.Tensor,
        visit_counts: Optional[torch.Tensor] = None,
        quality_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute quality-weighted policy loss.

        Args:
            policy_log_probs: (B, A) log probabilities
            policy_targets: (B, A) target probabilities
            visit_counts: (B,) MCTS visit counts (optional)
            quality_weights: (B,) pre-computed quality weights (optional)

        Returns:
            Quality-weighted policy loss
        """
        device = policy_log_probs.device
        batch_size = policy_log_probs.shape[0]

        # Compute quality weights if not provided
        if quality_weights is None:
            if visit_counts is not None:
                quality_weights = compute_quality_weights(
                    visit_counts,
                    temperature=self.temperature,
                    min_weight=self.min_quality_weight,
                )
            else:
                quality_weights = torch.ones(batch_size, device=device)

        # Blend between uniform and quality-weighted
        if self.quality_weight < 1.0:
            uniform_weights = torch.ones_like(quality_weights)
            quality_weights = (
                self.quality_weight * quality_weights +
                (1 - self.quality_weight) * uniform_weights
            )

        # Update statistics
        self.quality_stats["mean_weight"] = quality_weights.mean().item()
        self.quality_stats["std_weight"] = quality_weights.std().item()

        # Compute weighted policy loss
        policy_loss = quality_weighted_policy_loss(
            policy_log_probs, policy_targets, quality_weights
        )

        # Add ranking loss if enabled
        if self.ranking_weight > 0 and visit_counts is not None:
            # Normalize visit counts to [0, 1] for ranking
            max_visits = visit_counts.max()
            if max_visits > 0:
                quality_scores = visit_counts.float() / max_visits
                ranking_loss = ranking_loss_from_quality(
                    policy_log_probs, quality_scores, margin=self.ranking_margin
                )
                self.quality_stats["ranking_loss"] = ranking_loss.item()
                policy_loss = policy_loss + self.ranking_weight * ranking_loss

        return policy_loss

    def compute_weighted_value_loss(
        self,
        value_pred: torch.Tensor,
        value_target: torch.Tensor,
        visit_counts: Optional[torch.Tensor] = None,
        quality_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute quality-weighted value loss.

        Args:
            value_pred: (B,) or (B, P) predicted values
            value_target: (B,) or (B, P) target values
            visit_counts: (B,) MCTS visit counts (optional)
            quality_weights: (B,) pre-computed quality weights (optional)

        Returns:
            Quality-weighted value loss
        """
        device = value_pred.device
        batch_size = value_pred.shape[0]

        # Compute quality weights if not provided
        if quality_weights is None:
            if visit_counts is not None:
                quality_weights = compute_quality_weights(
                    visit_counts,
                    temperature=self.temperature,
                    min_weight=self.min_quality_weight,
                )
            else:
                quality_weights = torch.ones(batch_size, device=device)

        # Blend between uniform and quality-weighted
        if self.quality_weight < 1.0:
            uniform_weights = torch.ones_like(quality_weights)
            quality_weights = (
                self.quality_weight * quality_weights +
                (1 - self.quality_weight) * uniform_weights
            )

        return quality_weighted_value_loss(value_pred, value_target, quality_weights)


def create_quality_weighted_sampler(
    visit_counts: np.ndarray,
    temperature: float = 1.0,
    min_weight: float = 0.1,
) -> np.ndarray:
    """Create sample weights for DataLoader based on visit counts.

    Use with torch.utils.data.WeightedRandomSampler for quality-weighted
    batch construction.

    Args:
        visit_counts: (N,) array of visit counts per sample
        temperature: Scaling factor
        min_weight: Minimum weight

    Returns:
        (N,) normalized sample weights
    """
    if len(visit_counts) == 0:
        return np.array([])

    max_visits = visit_counts.max()
    if max_visits == 0:
        return np.ones(len(visit_counts))

    # Normalize to [0, 1]
    weights = visit_counts.astype(np.float32) / max_visits

    # Apply temperature
    if temperature != 1.0:
        weights = np.power(weights, 1.0 / temperature)

    # Apply minimum and normalize
    weights = np.clip(weights, min_weight, None)
    weights = weights / weights.mean()  # Mean = 1

    return weights


# Module exports
__all__ = [
    "compute_quality_weights",
    "quality_weighted_policy_loss",
    "quality_weighted_value_loss",
    "ranking_loss_from_quality",
    "QualityWeightedTrainer",
    "create_quality_weighted_sampler",
]

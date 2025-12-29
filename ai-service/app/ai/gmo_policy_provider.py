"""GMO Policy Provider for MCTS Move Priors.

.. note:: Experimental Feature (2025-12-20)
    This module provides uncertainty-based move priors for MCTS. Enable with:
    RINGRIFT_USE_GMO_POLICY=1

    Integration point: app/ai/mcts_ai.py policy selection logic as alternative
    to NNUE policy. Could enhance MCTS with uncertainty-aware exploration.

This module provides a GMO-based policy prior provider that can be used
with MCTS for move ordering. Instead of training a separate policy network,
this uses GMO's value + uncertainty estimates to rank moves.

Key advantages:
1. Uses uncertainty for exploration (variance-based bonuses)
2. Works without dedicated policy training data
3. Provides calibrated confidence through MC Dropout

Usage with MCTS:
    from app.ai.gmo_policy_provider import GMOPolicyProvider

    # Create provider from trained GMO checkpoint
    provider = GMOPolicyProvider.from_checkpoint("models/gmo/gmo_best.pt")

    # Get move priors for MCTS
    priors = provider.compute_priors(game_state, legal_moves)
    # Returns: {"move_str": probability, ...}

Integration with MCTSAI:
    class MCTSAI:
        def __init__(self, ...):
            self.gmo_policy = GMOPolicyProvider.from_checkpoint(...)

        def _compute_policy(self, moves, state):
            if self.use_gmo_policy:
                return self.gmo_policy.compute_priors(state, moves)
            return self._compute_nnue_policy(moves, state)

.. deprecated:: December 2025
    This experimental module will be removed in Q2 2026. GMO-based policy
    providers underperformed neural network policies in production. For new
    code, use ``app.ai.nnue_policy`` or the trained policy networks.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "app.ai.gmo_policy_provider is deprecated and will be removed in Q2 2026. "
    "Use app.ai.nnue_policy or neural network policies instead.",
    DeprecationWarning,
    stacklevel=2,
)

import logging
import math
from pathlib import Path

import torch

from ..models import GameState, Move
from ..utils.torch_utils import safe_load_checkpoint
from .gmo_ai import (
    GMOConfig,
    GMOValueNetWithUncertainty,
    MoveEncoder,
    StateEncoder,
    estimate_uncertainty,
)

logger = logging.getLogger(__name__)


class GMOPolicyProvider:
    """Provides move priors using GMO value + uncertainty estimates.

    This class wraps trained GMO networks and uses their predictions
    to compute prior probabilities for MCTS. The key insight is that
    GMO's uncertainty estimates (from MC Dropout) naturally provide
    an exploration bonus similar to UCB.

    The prior for each move is computed as:
        score = value + beta * sqrt(variance)
        prior = softmax(scores / temperature)

    Where beta controls the exploration/exploitation tradeoff.
    """

    def __init__(
        self,
        state_encoder: StateEncoder,
        move_encoder: MoveEncoder,
        value_net: GMOValueNetWithUncertainty,
        gmo_config: GMOConfig | None = None,
        device: str = "cpu",
    ):
        """Initialize from existing GMO components.

        Args:
            state_encoder: Trained state encoder network
            move_encoder: Trained move encoder network
            value_net: Trained value network with uncertainty
            gmo_config: GMO configuration for hyperparameters
            device: Device for inference
        """
        self.state_encoder = state_encoder
        self.move_encoder = move_encoder
        self.value_net = value_net
        self.gmo_config = gmo_config or GMOConfig()
        self.device = torch.device(device)

        # Policy-specific parameters
        self.temperature = 1.0  # Softmax temperature for priors
        self.beta = self.gmo_config.beta  # Exploration coefficient
        self.mc_samples = self.gmo_config.mc_samples

        # Ensure networks are on correct device and in eval mode
        self.state_encoder.to(self.device).eval()
        self.move_encoder.to(self.device).eval()
        self.value_net.to(self.device).eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "cpu",
        gmo_config: GMOConfig | None = None,
    ) -> GMOPolicyProvider:
        """Create provider from saved GMO checkpoint.

        Args:
            checkpoint_path: Path to GMO checkpoint file
            device: Device for inference
            gmo_config: Override GMO configuration

        Returns:
            GMOPolicyProvider instance ready for inference
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"GMO checkpoint not found: {path}")

        checkpoint = safe_load_checkpoint(path, map_location=device, warn_on_unsafe=False)

        config = gmo_config or checkpoint.get("gmo_config", GMOConfig())
        if isinstance(config, dict):
            config = GMOConfig(**config)

        # Create networks
        state_encoder = StateEncoder(
            embed_dim=config.state_dim,
            board_size=8,  # Default square8
        ).to(device)

        move_encoder = MoveEncoder(
            embed_dim=config.move_dim,
            board_size=8,
        ).to(device)

        value_net = GMOValueNetWithUncertainty(
            state_dim=config.state_dim,
            move_dim=config.move_dim,
            hidden_dim=config.hidden_dim,
            dropout_rate=config.dropout_rate,
        ).to(device)

        # Load weights
        state_encoder.load_state_dict(checkpoint["state_encoder"])
        move_encoder.load_state_dict(checkpoint["move_encoder"])
        value_net.load_state_dict(checkpoint["value_net"])

        logger.info(f"Loaded GMO policy provider from {path}")

        return cls(
            state_encoder=state_encoder,
            move_encoder=move_encoder,
            value_net=value_net,
            gmo_config=config,
            device=device,
        )

    def compute_priors(
        self,
        game_state: GameState,
        legal_moves: list[Move],
        temperature: float | None = None,
    ) -> dict[str, float]:
        """Compute prior probabilities for legal moves.

        Args:
            game_state: Current game state
            legal_moves: List of legal moves
            temperature: Softmax temperature (default: self.temperature)

        Returns:
            Dict mapping move string to prior probability
        """
        if not legal_moves:
            return {}

        if len(legal_moves) == 1:
            return {str(legal_moves[0]): 1.0}

        temp = temperature if temperature is not None else self.temperature

        # Encode state once
        with torch.no_grad():
            state_embed = self.state_encoder.encode_state(game_state)

        # Score each move using value + uncertainty
        move_scores: dict[str, float] = {}
        max_score = float('-inf')

        for move in legal_moves:
            with torch.no_grad():
                move_embed = self.move_encoder.encode_move(move)

            # Get value and variance using MC Dropout
            mean_val, _, variance = estimate_uncertainty(
                state_embed,
                move_embed,
                self.value_net,
                self.mc_samples,
                calibration_temperature=self.gmo_config.calibration_temperature,
            )

            # UCB-style score: value + beta * sqrt(variance)
            score = mean_val.item() + self.beta * math.sqrt(variance.item())
            move_key = str(move)
            move_scores[move_key] = score

            if score > max_score:
                max_score = score

        # Convert to probabilities using softmax
        if not move_scores:
            return {}

        exp_scores = {}
        total_exp = 0.0

        for key, score in move_scores.items():
            exp_val = math.exp((score - max_score) / temp)
            exp_scores[key] = exp_val
            total_exp += exp_val

        # Normalize
        if total_exp > 0:
            for key in exp_scores:
                exp_scores[key] /= total_exp

        return exp_scores

    def compute_priors_batch(
        self,
        game_state: GameState,
        legal_moves: list[Move],
    ) -> dict[str, float]:
        """Compute priors with batched move encoding for efficiency.

        For large move lists, this is more efficient than individual encoding.
        """
        if not legal_moves:
            return {}

        if len(legal_moves) == 1:
            return {str(legal_moves[0]): 1.0}

        # Encode state
        with torch.no_grad():
            state_embed = self.state_encoder.encode_state(game_state)

            # Batch encode all moves
            move_embeds = self.move_encoder.encode_moves(legal_moves)

        # Score all moves
        move_scores: dict[str, float] = {}
        max_score = float('-inf')

        for idx, move in enumerate(legal_moves):
            move_embed = move_embeds[idx]

            mean_val, _, variance = estimate_uncertainty(
                state_embed,
                move_embed,
                self.value_net,
                self.mc_samples,
                calibration_temperature=self.gmo_config.calibration_temperature,
            )

            score = mean_val.item() + self.beta * math.sqrt(variance.item())
            move_key = str(move)
            move_scores[move_key] = score

            if score > max_score:
                max_score = score

        # Softmax normalization
        exp_scores = {}
        total_exp = 0.0

        for key, score in move_scores.items():
            exp_val = math.exp((score - max_score) / self.temperature)
            exp_scores[key] = exp_val
            total_exp += exp_val

        if total_exp > 0:
            for key in exp_scores:
                exp_scores[key] /= total_exp

        return exp_scores


def create_hybrid_policy(
    nnue_priors: dict[str, float],
    gmo_priors: dict[str, float],
    gmo_weight: float = 0.5,
) -> dict[str, float]:
    """Combine NNUE and GMO priors into hybrid policy.

    Args:
        nnue_priors: Prior probabilities from NNUE policy
        gmo_priors: Prior probabilities from GMO
        gmo_weight: Weight for GMO priors (1 - gmo_weight for NNUE)

    Returns:
        Combined prior distribution
    """
    combined = {}

    # Get union of all moves
    all_moves = set(nnue_priors.keys()) | set(gmo_priors.keys())

    nnue_weight = 1.0 - gmo_weight

    for move_key in all_moves:
        nnue_p = nnue_priors.get(move_key, 0.0)
        gmo_p = gmo_priors.get(move_key, 0.0)
        combined[move_key] = nnue_weight * nnue_p + gmo_weight * gmo_p

    # Re-normalize
    total = sum(combined.values())
    if total > 0:
        for key in combined:
            combined[key] /= total

    return combined

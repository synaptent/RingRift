"""Auxiliary Prediction Tasks for Multi-Task Learning.

Adds auxiliary prediction heads to improve feature representations:
- Game length prediction
- Piece count prediction
- Win/Loss/Draw classification
- Move legality prediction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class AuxTaskConfig:
    """Configuration for auxiliary tasks."""
    enabled: bool = True
    game_length_weight: float = 0.1
    piece_count_weight: float = 0.1
    outcome_weight: float = 0.05
    move_legality_weight: float = 0.05


class GameLengthHead(nn.Module):
    """Predicts remaining game length from position."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PieceCountHead(nn.Module):
    """Predicts piece count delta (material advantage)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class OutcomeHead(nn.Module):
    """Predicts game outcome (win/loss/draw classification)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # Win, Loss, Draw
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AuxiliaryTaskModule(nn.Module):
    """Module containing all auxiliary prediction heads."""

    def __init__(
        self,
        input_dim: int,
        config: Optional[AuxTaskConfig] = None,
    ):
        super().__init__()
        if not HAS_TORCH:
            raise ImportError("PyTorch not available")

        self.config = config or AuxTaskConfig()

        self.game_length_head = GameLengthHead(input_dim)
        self.piece_count_head = PieceCountHead(input_dim)
        self.outcome_head = OutcomeHead(input_dim)

    def forward(
        self,
        features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through all auxiliary heads.

        Args:
            features: Shared features from backbone (batch, input_dim)

        Returns:
            Dictionary of auxiliary predictions
        """
        return {
            "game_length": self.game_length_head(features),
            "piece_count": self.piece_count_head(features),
            "outcome": self.outcome_head(features),
        }

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total auxiliary loss.

        Args:
            predictions: Auxiliary predictions
            targets: Auxiliary targets

        Returns:
            (total_loss, loss_breakdown)
        """
        losses = {}
        total = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        config = self.config

        if "game_length" in targets and config.game_length_weight > 0:
            loss = F.mse_loss(predictions["game_length"], targets["game_length"])
            losses["game_length"] = loss.item()
            total = total + config.game_length_weight * loss

        if "piece_count" in targets and config.piece_count_weight > 0:
            loss = F.mse_loss(predictions["piece_count"], targets["piece_count"])
            losses["piece_count"] = loss.item()
            total = total + config.piece_count_weight * loss

        if "outcome" in targets and config.outcome_weight > 0:
            loss = F.cross_entropy(predictions["outcome"], targets["outcome"])
            losses["outcome"] = loss.item()
            total = total + config.outcome_weight * loss

        losses["total_aux"] = total.item()
        return total, losses


class AuxiliaryTargetExtractor:
    """Extracts auxiliary targets from game data."""

    def __init__(self):
        pass

    def extract_targets(
        self,
        game_states: List[Any],
        game_outcomes: List[int],
        total_game_lengths: List[int],
    ) -> Dict[str, np.ndarray]:
        """Extract auxiliary targets from game data.

        Args:
            game_states: List of game states
            game_outcomes: Final outcomes (1=win, 0=draw, -1=loss)
            total_game_lengths: Total moves in each game

        Returns:
            Dictionary of target arrays
        """
        n = len(game_states)

        targets = {
            "game_length": np.zeros(n, dtype=np.float32),
            "piece_count": np.zeros(n, dtype=np.float32),
            "outcome": np.zeros(n, dtype=np.int64),
        }

        for i, state in enumerate(game_states):
            # Game length (normalized)
            current_move = getattr(state, "turn_number", i)
            remaining = total_game_lengths[i] - current_move
            targets["game_length"][i] = remaining / 100.0  # Normalize

            # Piece count (would extract from state)
            targets["piece_count"][i] = 0.0  # Placeholder

            # Outcome classification (0=loss, 1=draw, 2=win)
            outcome = game_outcomes[i]
            targets["outcome"][i] = outcome + 1  # Map -1,0,1 to 0,1,2

        return targets


def create_auxiliary_module(
    input_dim: int,
    game_length_weight: float = 0.1,
    piece_count_weight: float = 0.1,
) -> AuxiliaryTaskModule:
    """Create an auxiliary task module."""
    config = AuxTaskConfig(
        game_length_weight=game_length_weight,
        piece_count_weight=piece_count_weight,
    )
    return AuxiliaryTaskModule(input_dim, config)

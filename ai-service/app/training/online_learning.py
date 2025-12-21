"""Online Learning Module for RingRift AI.

Provides continuous learning capabilities for AI models during gameplay.
This module consolidates online learning patterns from across the codebase.

Key Features:
- TD-Energy updates: E(s,a) predicts min E(s', a') over next state
- Outcome-weighted contrastive loss: Winner's moves → low energy, loser's → high
- Rolling buffer: Maintains last N games for stability
- Gradient accumulation with conservative learning rates

Available Components:
- OnlineLearningConfig: Unified configuration for online learning
- OnlineLearner: Generic online learning interface
- EBMOOnlineLearner: EBMO-specific implementation (TD-Energy + contrastive)
- EBMOOnlineAI: EBMO AI wrapper with integrated online learning

Usage:
    # Basic EBMO online learning
    from app.training import EBMOOnlineLearner, OnlineLearningConfig

    config = OnlineLearningConfig(
        buffer_size=20,
        learning_rate=1e-5,
        td_weight=0.5,
        outcome_weight=0.5,
    )
    learner = EBMOOnlineLearner(network, config=config)

    # During gameplay
    learner.record_transition(state, move, player, next_state)

    # After game ends
    metrics = learner.update_from_game(winner)

    # Full AI wrapper
    from app.training import EBMOOnlineAI

    ai = EBMOOnlineAI(player_number=1, config=ai_config, model_path="model.pt")
    move = ai.select_move(state)  # Records transitions automatically
    ai.end_game(winner)  # Runs learning update

Integration with Training Pipeline:
    The online learning components integrate with the unified training
    orchestrator through the hot buffer interface. New games from online
    learning can feed into the main training loop.

    from app.training import (
        UnifiedTrainingOrchestrator,
        EBMOOnlineLearner,
        HotDataBuffer,
    )

    # Online learner feeds hot buffer
    hot_buffer.add_game(game_record)

History:
    - December 2025: Consolidated from app.ai.archive.ebmo_online
    - December 2025: Added to main training module
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from app.models import GameState, Move

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class OnlineLearningConfig:
    """Unified configuration for online learning.

    This configuration applies to all online learning implementations.
    Specific implementations may have additional configuration options.

    Attributes:
        buffer_size: Number of games to keep in rolling buffer
        min_games_before_update: Minimum games before first update
        learning_rate: Learning rate (very low for stability)
        batch_size: Games to sample per update
        gradient_clip: Gradient clipping norm (0 to disable)
        td_weight: Weight for TD-energy loss
        outcome_weight: Weight for outcome-contrastive loss
        gamma: Discount factor for TD updates
        td_lambda: Eligibility trace decay
        winner_energy_target: Target energy for winner's moves
        loser_energy_target: Target energy for loser's moves
    """
    # Buffer settings
    buffer_size: int = 20
    min_games_before_update: int = 1

    # Learning rate
    learning_rate: float = 1e-5  # Very low for stability

    # Update settings
    batch_size: int = 8
    gradient_clip: float = 1.0

    # Loss weights
    td_weight: float = 0.5
    outcome_weight: float = 0.5

    # TD settings
    gamma: float = 0.99
    td_lambda: float = 0.9

    # Outcome targets
    winner_energy_target: float = -1.0
    loser_energy_target: float = 1.0

    # Board settings (for feature extraction)
    board_size: int = 8


@dataclass
class OnlineLearningMetrics:
    """Metrics from an online learning update.

    Attributes:
        total_loss: Combined loss value
        td_loss: TD-energy loss component
        outcome_loss: Outcome-weighted loss component
        samples: Number of samples processed
        games: Number of games used
        buffer_size: Current buffer size
    """
    total_loss: float = 0.0
    td_loss: float = 0.0
    outcome_loss: float = 0.0
    samples: int = 0
    games: int = 0
    buffer_size: int = 0
    status: str = "ok"


@dataclass
class Transition:
    """A single state-action transition for learning."""
    state: "GameState"
    move: "Move"
    player: int
    next_state: "GameState | None"
    energy: float | None = None


@dataclass
class GameRecord:
    """Complete record of a played game for learning."""
    transitions: list[Transition]
    winner: int | None
    players: tuple[int, ...]


@dataclass
class OnlineLearningStats:
    """Statistics for online learning session.

    Attributes:
        games_trained: Total games used for training
        total_updates: Total update steps performed
        buffer_size: Current buffer size
        buffer_capacity: Maximum buffer capacity
        avg_recent_loss: Average loss over recent updates
        learning_rate: Current learning rate
    """
    games_trained: int = 0
    total_updates: int = 0
    buffer_size: int = 0
    buffer_capacity: int = 20
    avg_recent_loss: float = 0.0
    learning_rate: float = 1e-5


# =============================================================================
# Abstract Base Class
# =============================================================================

@runtime_checkable
class OnlineLearner(Protocol):
    """Protocol for online learning implementations."""

    def record_transition(
        self,
        state: "GameState",
        move: "Move",
        player: int,
        next_state: "GameState | None" = None,
    ) -> None:
        """Record a state-action transition during gameplay."""
        ...

    def end_game(self, winner: int | None) -> None:
        """Signal end of game and add to buffer."""
        ...

    def update_from_game(self, winner: int | None) -> dict[str, Any]:
        """Complete game and run update step."""
        ...

    def get_stats(self) -> dict[str, Any]:
        """Get training statistics."""
        ...


# =============================================================================
# EBMO Online Learning Implementation
# =============================================================================

# Re-export from archive with proper documentation
# This maintains backward compatibility while moving to the canonical location

try:
    from app.ai.archive.ebmo_online import (
        EBMOOnlineConfig,
        EBMOOnlineLearner,
        EBMOOnlineAI,
        GameRecord as EBMOGameRecord,
        Transition as EBMOTransition,
    )
    HAS_EBMO_ONLINE = True
except ImportError:
    HAS_EBMO_ONLINE = False

    # Provide stub classes for type checking
    class EBMOOnlineConfig:  # type: ignore[no-redef]
        """Stub for EBMOOnlineConfig when import fails."""
        pass

    class EBMOOnlineLearner:  # type: ignore[no-redef]
        """Stub for EBMOOnlineLearner when import fails."""
        pass

    class EBMOOnlineAI:  # type: ignore[no-redef]
        """Stub for EBMOOnlineAI when import fails."""
        pass


# =============================================================================
# Factory Functions
# =============================================================================

def create_online_learner(
    model: torch.nn.Module,
    learner_type: str = "ebmo",
    config: OnlineLearningConfig | None = None,
    device: str | torch.device = "cpu",
) -> OnlineLearner:
    """Create an online learner instance.

    Args:
        model: Neural network model to train
        learner_type: Type of online learner ("ebmo")
        config: Online learning configuration
        device: Device for computation

    Returns:
        OnlineLearner instance

    Raises:
        ValueError: If learner_type is not supported
        ImportError: If required dependencies are not available
    """
    if learner_type == "ebmo":
        if not HAS_EBMO_ONLINE:
            raise ImportError(
                "EBMO online learning not available. "
                "Check app.ai.archive.ebmo_online module."
            )

        ebmo_config = None
        if config is not None:
            ebmo_config = EBMOOnlineConfig(
                buffer_size=config.buffer_size,
                min_games_before_update=config.min_games_before_update,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                gradient_clip=config.gradient_clip,
                td_weight=config.td_weight,
                outcome_weight=config.outcome_weight,
                gamma=config.gamma,
                td_lambda=config.td_lambda,
                winner_energy_target=config.winner_energy_target,
                loser_energy_target=config.loser_energy_target,
                board_size=config.board_size,
            )

        return EBMOOnlineLearner(model, device=device, config=ebmo_config)

    raise ValueError(f"Unknown learner type: {learner_type}")


def get_online_learning_config(
    profile: str = "default",
    **overrides: Any,
) -> OnlineLearningConfig:
    """Get a predefined online learning configuration.

    Args:
        profile: Configuration profile name
        **overrides: Values to override

    Returns:
        OnlineLearningConfig instance

    Available profiles:
        - "default": Balanced settings for general use
        - "conservative": Very slow learning, maximum stability
        - "aggressive": Faster learning, less stable
        - "tournament": Optimized for tournament play
    """
    profiles = {
        "default": OnlineLearningConfig(),
        "conservative": OnlineLearningConfig(
            learning_rate=1e-6,
            buffer_size=50,
            min_games_before_update=5,
            gradient_clip=0.5,
        ),
        "aggressive": OnlineLearningConfig(
            learning_rate=1e-4,
            buffer_size=10,
            min_games_before_update=1,
            batch_size=4,
        ),
        "tournament": OnlineLearningConfig(
            learning_rate=5e-6,
            buffer_size=30,
            min_games_before_update=3,
            td_weight=0.7,
            outcome_weight=0.3,
        ),
    }

    if profile not in profiles:
        raise ValueError(
            f"Unknown profile: {profile}. "
            f"Available: {list(profiles.keys())}"
        )

    config = profiles[profile]

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config key: {key}")

    return config


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    "OnlineLearningConfig",
    "OnlineLearningMetrics",
    "OnlineLearningStats",
    # Data structures
    "Transition",
    "GameRecord",
    # Protocol
    "OnlineLearner",
    # EBMO implementation
    "EBMOOnlineConfig",
    "EBMOOnlineLearner",
    "EBMOOnlineAI",
    # Factory functions
    "create_online_learner",
    "get_online_learning_config",
    # Feature flag
    "HAS_EBMO_ONLINE",
]

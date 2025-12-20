"""Configuration-related test factories.

Provides factories for creating training configs, unified configs, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "create_evaluation_config",
    "create_training_config",
    "create_unified_config",
]


@dataclass
class MockTrainingConfig:
    """Mock training configuration for testing."""
    learning_rate: float = 0.0003
    batch_size: int = 256
    epochs: int = 50
    weight_decay: float = 0.0001
    dropout: float = 0.1
    validation_split: float = 0.15
    num_filters: int = 192
    num_res_blocks: int = 12

    def to_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "validation_split": self.validation_split,
            "num_filters": self.num_filters,
            "num_res_blocks": self.num_res_blocks,
        }


@dataclass
class MockEvaluationConfig:
    """Mock evaluation configuration for testing."""
    shadow_games: int = 15
    full_tournament_games: int = 50
    min_games_for_elo: int = 30
    elo_k_factor: int = 32

    def to_dict(self) -> dict[str, Any]:
        return {
            "shadow_games": self.shadow_games,
            "full_tournament_games": self.full_tournament_games,
            "min_games_for_elo": self.min_games_for_elo,
            "elo_k_factor": self.elo_k_factor,
        }


@dataclass
class MockUnifiedConfig:
    """Mock unified configuration for testing."""
    training: MockTrainingConfig = field(default_factory=MockTrainingConfig)
    evaluation: MockEvaluationConfig = field(default_factory=MockEvaluationConfig)
    configs: list[str] = field(default_factory=lambda: ["square8_2p", "hex8_2p"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "training": self.training.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "configs": self.configs,
        }


def create_training_config(
    learning_rate: float = 0.0003,
    batch_size: int = 256,
    epochs: int = 50,
    weight_decay: float = 0.0001,
    dropout: float = 0.1,
    **kwargs: Any,
) -> MockTrainingConfig:
    """Create a mock training configuration for testing.

    Args:
        learning_rate: Learning rate (default: 0.0003)
        batch_size: Batch size (default: 256)
        epochs: Number of epochs (default: 50)
        weight_decay: Weight decay (default: 0.0001)
        dropout: Dropout rate (default: 0.1)
        **kwargs: Additional config attributes

    Returns:
        MockTrainingConfig instance
    """
    return MockTrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        weight_decay=weight_decay,
        dropout=dropout,
        **kwargs,
    )


def create_evaluation_config(
    shadow_games: int = 15,
    full_tournament_games: int = 50,
    min_games_for_elo: int = 30,
    elo_k_factor: int = 32,
) -> MockEvaluationConfig:
    """Create a mock evaluation configuration for testing.

    Args:
        shadow_games: Number of shadow games
        full_tournament_games: Number of full tournament games
        min_games_for_elo: Minimum games for Elo calculation
        elo_k_factor: Elo K factor

    Returns:
        MockEvaluationConfig instance
    """
    return MockEvaluationConfig(
        shadow_games=shadow_games,
        full_tournament_games=full_tournament_games,
        min_games_for_elo=min_games_for_elo,
        elo_k_factor=elo_k_factor,
    )


def create_unified_config(
    training_config: MockTrainingConfig | None = None,
    evaluation_config: MockEvaluationConfig | None = None,
    configs: list[str] | None = None,
) -> MockUnifiedConfig:
    """Create a mock unified configuration for testing.

    Args:
        training_config: Training config (uses defaults if None)
        evaluation_config: Evaluation config (uses defaults if None)
        configs: List of config keys (uses defaults if None)

    Returns:
        MockUnifiedConfig instance
    """
    return MockUnifiedConfig(
        training=training_config or MockTrainingConfig(),
        evaluation=evaluation_config or MockEvaluationConfig(),
        configs=configs or ["square8_2p", "hex8_2p"],
    )

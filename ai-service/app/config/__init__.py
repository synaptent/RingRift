"""Configuration module for training runs.

Provides typed dataclasses for:
- CMA-ES optimization (CMAESConfig)
- Neural network training (NeuralNetConfig)
- Self-play data generation (SelfPlayConfig)
"""

from app.config.training_config import (
    CMAESConfig,
    NeuralNetConfig,
    SelfPlayConfig,
)

__all__ = [
    "CMAESConfig",
    "NeuralNetConfig",
    "SelfPlayConfig",
]

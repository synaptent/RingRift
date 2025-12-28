"""Board-specific hyperparameter configurations for optimal training.

Each board type has unique characteristics requiring tuned parameters:
- square8: Fast convergence, standard topology
- square19: Large state space, needs higher capacity
- hexagonal: 6-way symmetry, D6 augmentation critical
- hex8: Small hex, balance between square8 and full hex

Usage:
    from app.training.board_hyperparams import get_board_hyperparams

    params = get_board_hyperparams("square8", num_players=2)
    # Use params.learning_rate, params.batch_size, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BoardHyperparams:
    """Board-specific training hyperparameters."""

    # Core training params
    learning_rate: float = 0.0003
    batch_size: int = 256
    epochs: int = 50
    warmup_epochs: int = 5

    # Architecture
    hidden_dim: int = 256
    num_hidden_layers: int = 2

    # Regularization
    # Dec 28, 2025: Using 0.0001 to prevent underfitting while maintaining generalization
    # Higher values (0.001) were causing models to plateau at 1500 Elo
    weight_decay: float = 0.0001  # Balanced regularization
    dropout: float = 0.1
    label_smoothing: float = 0.05

    # Loss weights
    policy_weight: float = 1.0
    value_weight: float = 1.0

    # Augmentation
    augment_rotations: bool = True
    augment_reflections: bool = True
    augmentation_factor: int = 4  # square: 4, hex: 6 or 12

    # Learning rate schedule
    lr_schedule: str = "cosine"  # cosine, step, plateau
    lr_min_ratio: float = 0.01  # min_lr = lr * ratio

    # Advanced
    gradient_clip: float = 1.0
    early_stopping_patience: int = 15
    swa_start_fraction: float = 0.75

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items()}


# Board-specific configurations (empirically tuned)
BOARD_HYPERPARAMS: dict[str, BoardHyperparams] = {
    # Square 8x8 - baseline, fast training
    # Dec 28, 2025: Upgraded architecture for 2000+ Elo
    "square8_2p": BoardHyperparams(
        learning_rate=0.0003,
        batch_size=64,
        epochs=100,
        hidden_dim=512,
        num_hidden_layers=6,
        weight_decay=0.0001,
        label_smoothing=0.05,
        augmentation_factor=4,
        early_stopping_patience=30,
    ),
    # Dec 28, 2025: Upgraded architecture for 2000+ Elo
    "square8_3p": BoardHyperparams(
        learning_rate=0.0003,
        batch_size=64,
        epochs=100,
        hidden_dim=512,
        num_hidden_layers=6,
        weight_decay=0.0001,
        policy_weight=1.2,
        label_smoothing=0.07,
        early_stopping_patience=30,
    ),
    # Dec 28, 2025: Upgraded architecture for 2000+ Elo
    "square8_4p": BoardHyperparams(
        learning_rate=0.0003,
        batch_size=64,
        epochs=100,
        hidden_dim=512,
        num_hidden_layers=6,
        weight_decay=0.0001,
        policy_weight=1.5,
        label_smoothing=0.08,
        early_stopping_patience=30,
    ),

    # Square 19x19 - large state space
    # Dec 28, 2025: Upgraded architecture for 2000+ Elo
    "square19_2p": BoardHyperparams(
        learning_rate=0.0003,
        batch_size=64,
        epochs=100,
        hidden_dim=512,
        num_hidden_layers=6,
        weight_decay=0.0001,
        label_smoothing=0.06,
        warmup_epochs=10,
        gradient_clip=0.5,  # Tighter clipping for stability
        early_stopping_patience=30,
    ),
    # Dec 28, 2025: Upgraded architecture for 2000+ Elo
    "square19_3p": BoardHyperparams(
        learning_rate=0.0003,
        batch_size=64,
        epochs=100,
        hidden_dim=512,
        num_hidden_layers=6,
        weight_decay=0.0001,
        policy_weight=1.3,
        label_smoothing=0.08,
        early_stopping_patience=30,
    ),
    # Dec 28, 2025: Upgraded architecture for 2000+ Elo
    "square19_4p": BoardHyperparams(
        learning_rate=0.0003,
        batch_size=64,
        epochs=100,
        hidden_dim=512,
        num_hidden_layers=6,
        weight_decay=0.0001,
        policy_weight=1.5,
        label_smoothing=0.10,
        early_stopping_patience=30,
    ),

    # Hexagonal (full) - D6 symmetry
    # Dec 28, 2025: Upgraded architecture for 2000+ Elo
    "hexagonal_2p": BoardHyperparams(
        learning_rate=0.0003,
        batch_size=64,
        epochs=100,
        hidden_dim=512,
        num_hidden_layers=6,
        augmentation_factor=6,  # D6 symmetry
        label_smoothing=0.05,
        weight_decay=0.0001,
        early_stopping_patience=30,
    ),
    # Dec 28, 2025: Upgraded architecture for 2000+ Elo
    "hexagonal_3p": BoardHyperparams(
        learning_rate=0.0003,
        batch_size=64,
        epochs=100,
        hidden_dim=512,
        num_hidden_layers=6,
        augmentation_factor=6,
        policy_weight=1.2,
        label_smoothing=0.06,
        weight_decay=0.0001,
        early_stopping_patience=30,
    ),
    # Dec 28, 2025: Upgraded architecture for 2000+ Elo
    "hexagonal_4p": BoardHyperparams(
        learning_rate=0.0003,
        batch_size=64,
        epochs=100,
        hidden_dim=512,
        num_hidden_layers=6,
        augmentation_factor=6,
        policy_weight=1.4,
        label_smoothing=0.08,
        weight_decay=0.0001,
        early_stopping_patience=30,
    ),

    # Hex8 (small hex) - balanced for good generalization
    # Dec 28, 2025: Major architecture upgrade to reach 2000+ Elo
    # - Increased hidden_dim 192→512 and layers 2→6 for deeper representation
    # - Reduced batch_size 256→64 for ~5K sample datasets
    # - Reduced learning_rate 0.0006→0.0003 for stable convergence
    # - Increased patience 10→30 and epochs 40→100 to let it train fully
    # - Reduced weight_decay 0.001→0.0001 to prevent underfitting
    "hex8_2p": BoardHyperparams(
        learning_rate=0.0003,
        batch_size=64,
        epochs=100,
        hidden_dim=512,
        num_hidden_layers=6,
        weight_decay=0.0001,
        augmentation_factor=6,
        label_smoothing=0.05,
        early_stopping_patience=30,
    ),
    # Dec 28, 2025: Upgraded architecture for 2000+ Elo
    "hex8_3p": BoardHyperparams(
        learning_rate=0.0003,
        batch_size=64,
        epochs=100,
        hidden_dim=512,
        num_hidden_layers=6,
        weight_decay=0.0001,
        augmentation_factor=6,
        label_smoothing=0.05,
        early_stopping_patience=30,
    ),
    # Dec 28, 2025: Upgraded architecture for 2000+ Elo
    "hex8_4p": BoardHyperparams(
        learning_rate=0.0003,
        batch_size=64,
        epochs=100,
        hidden_dim=512,
        num_hidden_layers=6,
        weight_decay=0.0001,
        augmentation_factor=6,
        policy_weight=1.2,
        label_smoothing=0.05,
        early_stopping_patience=30,
    ),
}


def get_board_hyperparams(
    board_type: str,
    num_players: int = 2,
    override: dict[str, Any] | None = None,
) -> BoardHyperparams:
    """Get hyperparameters for a specific board/player configuration.

    Args:
        board_type: Board type (square8, square19, hexagonal, hex8)
        num_players: Number of players (2, 3, 4)
        override: Optional dict of overrides

    Returns:
        BoardHyperparams with tuned values
    """
    config_key = f"{board_type}_{num_players}p"

    # Start with closest match
    if config_key in BOARD_HYPERPARAMS:
        params_dict = BOARD_HYPERPARAMS[config_key].to_dict()
    elif f"{board_type}_2p" in BOARD_HYPERPARAMS:
        # Fall back to 2p variant
        params_dict = BOARD_HYPERPARAMS[f"{board_type}_2p"].to_dict()
    else:
        # Use default
        params_dict = BoardHyperparams().to_dict()

    # Apply overrides
    if override:
        params_dict.update(override)

    return BoardHyperparams(**params_dict)


def get_augmentation_factor(board_type: str) -> int:
    """Get appropriate augmentation factor for board type."""
    if board_type in ("hexagonal", "hex8"):
        return 6  # D6 symmetry (can extend to 12 with reflections)
    return 4  # D4 symmetry for square boards


def get_all_config_keys() -> list[str]:
    """Get all available configuration keys."""
    return list(BOARD_HYPERPARAMS.keys())


__all__ = [
    "BoardHyperparams",
    "BOARD_HYPERPARAMS",
    "get_board_hyperparams",
    "get_augmentation_factor",
    "get_all_config_keys",
]

from dataclasses import dataclass, field
from typing import Dict, Optional
from app.models import BoardType


@dataclass
class TrainConfig:
    """Configuration for training run"""
    board_type: BoardType = BoardType.SQUARE8
    episodes_per_iter: int = 4
    epochs_per_iter: int = 4
    # Number of self-play + training + evaluation cycles to run in the
    # high-level training loop. This was previously hard-coded; exposing it
    # here makes iterative retraining a first-class configuration parameter.
    iterations: int = 2
    learning_rate: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 1e-4
    history_length: int = 3
    seed: int = 42
    max_moves_per_game: int = 200
    k_elo: int = 32
    policy_weight: float = 1.0

    # Model identity used to derive checkpoint filenames. This is kept in sync
    # with NeuralNetAI, which expects checkpoints under
    # "<repo_root>/ai-service/models/<nn_model_id>.pth".
    model_id: str = "ringrift_v1"

    # Board-specific model architecture parameters
    # If None, uses defaults from neural_net.get_model_config_for_board()
    num_res_blocks: Optional[int] = None
    num_filters: Optional[int] = None
    policy_size: Optional[int] = None

    # Paths (initialised to repository-root-relative defaults in __post_init__)
    # When instantiated, these will be rewritten as absolute paths anchored at
    # the ai-service repo root so that training artefacts do not depend on the
    # current working directory and do not create nested "ai-service/ai-service"
    # folders.
    data_dir: str = "ai-service/app/training/data"
    model_dir: str = "ai-service/models"
    log_dir: str = "ai-service/app/logs"

    def __post_init__(self):
        import os
        from pathlib import Path

        # Resolve the ai-service repository root as the directory containing
        # this file's "app" parent (i.e., <repo_root>/ai-service).
        this_file = Path(__file__).resolve()
        repo_root = this_file.parents[2]

        # Construct absolute, repo-root-anchored paths so training runs behave
        # consistently whether invoked from <repo_root> or <repo_root>/ai-service.
        self.data_dir = os.fspath(repo_root / "app" / "training" / "data")
        self.model_dir = os.fspath(repo_root / "models")
        self.log_dir = os.fspath(repo_root / "app" / "logs")

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


# =============================================================================
# Board-Specific Training Configuration Presets
# =============================================================================
#
# These presets provide optimized training configurations for each board type,
# tuned for their respective action space sizes and computational requirements.


def get_training_config_for_board(
    board_type: BoardType,
    base_config: Optional[TrainConfig] = None,
) -> TrainConfig:
    """
    Get a training configuration optimized for a specific board type.

    This function returns a TrainConfig with hyperparameters tuned for the
    board type's action space size and computational requirements. If a
    base_config is provided, it will be used as a starting point with
    board-specific overrides applied.

    Parameters
    ----------
    board_type : BoardType
        The board type to configure for (SQUARE8, SQUARE19, or HEXAGONAL).
    base_config : Optional[TrainConfig]
        Base configuration to use. If None, creates a new TrainConfig.

    Returns
    -------
    TrainConfig
        Configuration optimized for the specified board type.
    """
    config = TrainConfig() if base_config is None else base_config
    config.board_type = board_type

    # Import here to avoid circular imports
    from app.ai.neural_net import (
        POLICY_SIZE_8x8,
        POLICY_SIZE_19x19,
        P_HEX,
    )

    if board_type == BoardType.SQUARE8:
        # 8x8 board: smaller action space, faster training
        config.policy_size = POLICY_SIZE_8x8  # 7,000
        config.num_res_blocks = 6  # Fewer blocks for smaller board
        config.num_filters = 64  # Reduced filters
        config.batch_size = 64  # Can use larger batches
        config.learning_rate = 2e-3  # Slightly higher LR
        config.max_moves_per_game = 150  # Shorter games on 8x8
        config.model_id = "ringrift_8x8_v1"

    elif board_type == BoardType.SQUARE19:
        # 19x19 board: large action space, full capacity model
        config.policy_size = POLICY_SIZE_19x19  # 67,000
        config.num_res_blocks = 10  # Full depth
        config.num_filters = 128  # Full width
        config.batch_size = 32  # Standard batch size
        config.learning_rate = 1e-3  # Standard LR
        config.max_moves_per_game = 300  # Longer games on 19x19
        config.model_id = "ringrift_19x19_v1"

    elif board_type == BoardType.HEXAGONAL:
        # Hex board: specialized architecture with masked pooling
        config.policy_size = P_HEX  # 91,876
        config.num_res_blocks = 8  # Balanced depth
        config.num_filters = 128  # Full width for complex patterns
        config.batch_size = 32  # Standard batch size
        config.learning_rate = 1e-3  # Standard LR
        config.max_moves_per_game = 250  # Medium game length
        config.model_id = "ringrift_hex_v1"

    else:
        # Unknown board type: use defaults
        config.policy_size = POLICY_SIZE_19x19
        config.num_res_blocks = 10
        config.num_filters = 128
        config.model_id = "ringrift_v1"

    return config


# Pre-defined training configurations for quick access
BOARD_TRAINING_CONFIGS: Dict[BoardType, Dict[str, any]] = {
    BoardType.SQUARE8: {
        "policy_size": 7000,
        "num_res_blocks": 6,
        "num_filters": 64,
        "batch_size": 64,
        "learning_rate": 2e-3,
        "max_moves_per_game": 150,
        "model_id": "ringrift_8x8_v1",
        "description": "Optimized for 8x8 competitive play - compact and fast",
    },
    BoardType.SQUARE19: {
        "policy_size": 67000,
        "num_res_blocks": 10,
        "num_filters": 128,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "max_moves_per_game": 300,
        "model_id": "ringrift_19x19_v1",
        "description": "Full capacity for 19x19 strategic depth",
    },
    BoardType.HEXAGONAL: {
        "policy_size": 91876,  # Updated for radius-12 hex (25×25 frame)
        "num_res_blocks": 8,
        "num_filters": 128,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "max_moves_per_game": 300,  # Increased for larger 469-cell board
        "model_id": "ringrift_hex_v1",
        "description": "Specialized hex architecture with masked pooling (radius 12)",
    },
}

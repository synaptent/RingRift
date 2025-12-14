from dataclasses import dataclass
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

    # Learning rate schedule settings for better convergence
    warmup_epochs: int = 1  # LR warmup for training stability
    lr_scheduler: str = "cosine"  # Options: 'none', 'step', 'cosine', 'cosine-warm-restarts'
    lr_min: float = 1e-6  # Minimum LR for cosine annealing
    early_stopping_patience: int = 5  # Epochs without improvement before stopping (0=disabled)

    # Model identity used to derive checkpoint filenames. This is kept in sync
    # with NeuralNetAI, which expects checkpoints under
    # "<repo_root>/ai-service/models/<nn_model_id>.pth".
    #
    # NOTE: "v4"/"v5" are model-id / checkpoint lineage prefixes, not Python
    # architecture class names. See docs/MPS_ARCHITECTURE.md.
    #
    # Default to the preferred square8 2p v3-family checkpoint lineage ("v5").
    # V2-family checkpoints ("v4") remain supported as a fallback.
    model_id: str = "ringrift_v5_sq8_2p_2xh100"

    # Board-specific model architecture parameters
    # If None, uses defaults from neural_net.get_model_config_for_board()
    num_res_blocks: Optional[int] = None
    num_filters: Optional[int] = None
    policy_size: Optional[int] = None

    # GPU parallel data generation settings
    # When enabled, uses ParallelGameRunner for 5-10x faster data generation
    # None = auto-detect (True if CUDA available), False = disabled, True = forced
    use_gpu_parallel_datagen: Optional[bool] = None
    gpu_batch_size: int = 20  # Number of games to run in parallel on GPU

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

        # Auto-detect GPU for parallel data generation if not explicitly set
        # Check RINGRIFT_DISABLE_GPU_DATAGEN env var for opt-out
        if self.use_gpu_parallel_datagen is None:
            disable_env = os.environ.get("RINGRIFT_DISABLE_GPU_DATAGEN", "").lower()
            if disable_env in ("1", "true", "yes"):
                self.use_gpu_parallel_datagen = False
            else:
                try:
                    import torch
                    self.use_gpu_parallel_datagen = torch.cuda.is_available()
                except ImportError:
                    self.use_gpu_parallel_datagen = False


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
        # Prefer the canonical v3-capacity defaults unless explicitly overridden.
        config.num_res_blocks = 12
        config.num_filters = 192
        config.batch_size = 64  # Can use larger batches
        config.learning_rate = 2e-3  # Slightly higher LR
        config.max_moves_per_game = 150  # Shorter games on 8x8
        config.model_id = "ringrift_v5_sq8_2p_2xh100"

    elif board_type == BoardType.SQUARE19:
        # 19x19 board: large action space (~67k), requires smaller batch for stability
        config.policy_size = POLICY_SIZE_19x19  # 67,000
        config.num_res_blocks = 12
        config.num_filters = 192
        config.batch_size = 24  # Smaller batch for larger action space
        config.learning_rate = 5e-4  # Lower LR for stability
        config.max_moves_per_game = 300  # Longer games on 19x19
        config.model_id = "ringrift_v4_sq19_2p"

    elif board_type == BoardType.HEXAGONAL:
        # Hex board: largest action space (~92k), requires careful tuning
        config.policy_size = P_HEX  # 91,876
        config.num_res_blocks = 12
        config.num_filters = 192
        config.batch_size = 20  # Smaller batch for largest action space
        config.learning_rate = 5e-4  # Lower LR for stability
        config.max_moves_per_game = 250  # Medium game length
        config.model_id = "ringrift_v4_hex_2p"

    else:
        # Unknown board type: use defaults
        config.policy_size = POLICY_SIZE_19x19
        config.num_res_blocks = 12
        config.num_filters = 192
        config.model_id = "ringrift_v5_sq8_2p_2xh100"

    return config


# Pre-defined training configurations for quick access
BOARD_TRAINING_CONFIGS: Dict[BoardType, Dict[str, any]] = {
    BoardType.SQUARE8: {
        "policy_size": 7000,
        "num_res_blocks": 12,
        "num_filters": 192,
        "batch_size": 64,
        "learning_rate": 2e-3,
        "max_moves_per_game": 150,
        "model_id": "ringrift_v5_sq8_2p_2xh100",
        "description": "Canonical square8 v3-family (12 blocks, 192 filters)",
    },
    BoardType.SQUARE19: {
        "policy_size": 67000,
        "num_res_blocks": 12,
        "num_filters": 192,
        "batch_size": 24,  # Smaller for large action space stability
        "learning_rate": 5e-4,  # Lower for stability
        "max_moves_per_game": 300,
        "model_id": "ringrift_v4_sq19_2p",
        "description": "Full-capacity square19 baseline (v2-family)",
    },
    BoardType.HEXAGONAL: {
        "policy_size": 91876,  # Updated for radius-12 hex (25×25 frame)
        "num_res_blocks": 12,
        "num_filters": 192,
        "batch_size": 20,  # Smallest batch for largest action space
        "learning_rate": 5e-4,  # Lower for stability
        "max_moves_per_game": 300,  # Increased for larger 469-cell board
        "model_id": "ringrift_v4_hex_2p",
        "description": "Full-capacity hex baseline (v2-family)",
    },
}

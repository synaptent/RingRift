from dataclasses import dataclass
from typing import Dict, Optional
from app.models import BoardType


def _get_gpu_memory_gb() -> float:
    """Get available GPU memory in GB. Returns 0 if no GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def _scale_batch_size_for_gpu(base_batch: int, policy_size: int = 7000) -> int:
    """Scale batch size based on available GPU memory.

    H100 (80GB): 16x multiplier for small policies, 8x for large
    A100 (40-80GB): 8x multiplier for small, 4x for large
    Consumer GPUs (8-24GB): 2-4x multiplier
    """
    gpu_mem = _get_gpu_memory_gb()
    if gpu_mem <= 0:
        return base_batch  # No GPU, use default

    # Estimate memory multiplier based on policy size
    # Larger policies need more memory per sample
    if policy_size > 50000:  # Hex, Square19
        mem_per_sample_mb = 0.5  # ~0.5MB per sample for large policies
    elif policy_size > 10000:
        mem_per_sample_mb = 0.2  # ~0.2MB for medium policies
    else:
        mem_per_sample_mb = 0.1  # ~0.1MB for small policies

    # Reserve 8GB for model + overhead, use rest for batches
    available_gb = max(0, gpu_mem - 8)
    available_mb = available_gb * 1024

    # Calculate max batch size that fits in memory
    max_batch = int(available_mb / mem_per_sample_mb)

    # Scale up from base but cap at max
    if gpu_mem >= 90:  # GH200 class (96GB)
        scaled = base_batch * 32
    elif gpu_mem >= 70:  # H100 class (80GB)
        scaled = base_batch * 16
    elif gpu_mem >= 30:  # A100 class
        scaled = base_batch * 8
    elif gpu_mem >= 16:  # RTX 3090/4090 class
        scaled = base_batch * 4
    else:  # Consumer GPUs
        scaled = base_batch * 2

    return min(scaled, max_batch, 8192)  # Cap at 8192


@dataclass
class TrainConfig:
    """Configuration for training run.

    Note: batch_size will be auto-scaled based on GPU memory if
    RINGRIFT_AUTO_BATCH_SCALE=1 (default) or explicitly set via CLI.
    """
    board_type: BoardType = BoardType.SQUARE8
    episodes_per_iter: int = 4
    epochs_per_iter: int = 4
    # Number of self-play + training + evaluation cycles to run in the
    # high-level training loop. This was previously hard-coded; exposing it
    # here makes iterative retraining a first-class configuration parameter.
    iterations: int = 2
    learning_rate: float = 1e-3
    batch_size: int = 32  # Will be auto-scaled in __post_init__ if GPU available
    weight_decay: float = 1e-4
    history_length: int = 3
    seed: int = 42
    max_moves_per_game: int = 10000
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
    gpu_batch_size: int = 50  # Number of games to run in parallel on GPU

    # Data loading prefetch settings for improved GPU utilization
    # Prefetching loads batches in a background thread while GPU is computing
    use_prefetch: bool = True  # Enable/disable prefetching
    prefetch_count: int = 2  # Number of batches to prefetch
    pin_memory: bool = True  # Pin memory for faster CPU->GPU transfers (CUDA only)

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

        # Auto-scale batch size based on GPU memory (opt-out via env var)
        auto_scale = os.environ.get("RINGRIFT_AUTO_BATCH_SCALE", "1").lower()
        if auto_scale not in ("0", "false", "no"):
            # Use policy_size if set, otherwise estimate from board type
            policy_sz = self.policy_size or 7000  # Default to square8 size
            self.batch_size = _scale_batch_size_for_gpu(self.batch_size, policy_sz)


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
        POLICY_SIZE_HEX8,
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
        config.max_moves_per_game = 10000  # Allow games to complete naturally
        config.model_id = "ringrift_v5_sq8_2p_2xh100"

    elif board_type == BoardType.SQUARE19:
        # 19x19 board: large action space (~67k), requires smaller batch for stability
        config.policy_size = POLICY_SIZE_19x19  # 67,000
        config.num_res_blocks = 12
        config.num_filters = 192
        config.batch_size = 24  # Smaller batch for larger action space
        config.learning_rate = 5e-4  # Lower LR for stability
        config.max_moves_per_game = 10000  # Allow games to complete naturally
        config.model_id = "ringrift_v4_sq19_2p"

    elif board_type == BoardType.HEX8:
        # Hex8 board: smaller hex (radius-4, 61 cells), parallel to square8
        config.policy_size = POLICY_SIZE_HEX8  # 4,500
        config.num_res_blocks = 12
        config.num_filters = 192
        config.batch_size = 64  # Similar to square8
        config.learning_rate = 2e-3  # Slightly higher LR like square8
        config.max_moves_per_game = 10000  # Allow games to complete naturally
        config.model_id = "ringrift_v5_hex8_2p"

    elif board_type == BoardType.HEXAGONAL:
        # Hex board: largest action space (~92k), requires careful tuning
        config.policy_size = P_HEX  # 91,876
        config.num_res_blocks = 12
        config.num_filters = 192
        config.batch_size = 20  # Smaller batch for largest action space
        config.learning_rate = 5e-4  # Lower LR for stability
        config.max_moves_per_game = 10000  # Allow games to complete naturally
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
        "max_moves_per_game": 10000,
        "model_id": "ringrift_v5_sq8_2p_2xh100",
        "description": "Canonical square8 v3-family (12 blocks, 192 filters)",
    },
    BoardType.SQUARE19: {
        "policy_size": 67000,
        "num_res_blocks": 12,
        "num_filters": 192,
        "batch_size": 24,  # Smaller for large action space stability
        "learning_rate": 5e-4,  # Lower for stability
        "max_moves_per_game": 10000,
        "model_id": "ringrift_v4_sq19_2p",
        "description": "Full-capacity square19 baseline (v2-family)",
    },
    BoardType.HEX8: {
        "policy_size": 4500,  # Radius-4 hex (9×9 frame, 61 cells)
        "num_res_blocks": 12,
        "num_filters": 192,
        "batch_size": 64,  # Similar to square8 (comparable complexity)
        "learning_rate": 2e-3,  # Similar to square8
        "max_moves_per_game": 10000,
        "model_id": "ringrift_v5_hex8_2p",
        "description": "Hex8 v3-family (radius-4 hexagonal, parallel to square8)",
    },
    BoardType.HEXAGONAL: {
        "policy_size": 91876,  # Updated for radius-12 hex (25×25 frame)
        "num_res_blocks": 12,
        "num_filters": 192,
        "batch_size": 20,  # Smallest batch for largest action space
        "learning_rate": 5e-4,  # Lower for stability
        "max_moves_per_game": 10000,  # Allow games to complete naturally
        "model_id": "ringrift_v4_hex_2p",
        "description": "Full-capacity hex baseline (v2-family)",
    },
}

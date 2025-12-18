from dataclasses import dataclass, field
import os
from typing import Dict, Optional
from app.models import BoardType


# =============================================================================
# GPU Scaling Configuration
# =============================================================================

@dataclass
class GpuScalingConfig:
    """Configuration for GPU-based batch size scaling.

    Extracts magic numbers from _scale_batch_size_for_gpu for configurability.
    Can be overridden via environment variables with RINGRIFT_GPU_ prefix.
    """
    # Memory per sample estimates (MB)
    mem_per_sample_large_policy_mb: float = 0.5   # Policies > 50k (Hex, Square19)
    mem_per_sample_medium_policy_mb: float = 0.2  # Policies 10k-50k
    mem_per_sample_small_policy_mb: float = 0.1   # Policies < 10k

    # Policy size thresholds
    large_policy_threshold: int = 50000
    medium_policy_threshold: int = 10000

    # Memory reserved for model and overhead (GB)
    reserved_memory_gb: float = 8.0

    # GPU memory tiers and their batch multipliers
    gh200_memory_threshold_gb: float = 90.0   # GH200 class (96GB)
    gh200_batch_multiplier: int = 32

    h100_memory_threshold_gb: float = 70.0    # H100 class (80GB)
    h100_batch_multiplier: int = 16

    a100_memory_threshold_gb: float = 30.0    # A100 class (40-80GB)
    a100_batch_multiplier: int = 8

    rtx_memory_threshold_gb: float = 16.0     # RTX 3090/4090 class
    rtx_batch_multiplier: int = 4

    consumer_batch_multiplier: int = 2        # Consumer GPUs (<16GB)

    # Maximum batch size cap
    max_batch_size: int = 8192

    @classmethod
    def from_env(cls) -> "GpuScalingConfig":
        """Create config from environment variables."""
        config = cls()
        env_prefix = "RINGRIFT_GPU_"

        # Override from environment if set
        for field_name in [
            "reserved_memory_gb", "max_batch_size",
            "gh200_batch_multiplier", "h100_batch_multiplier",
            "a100_batch_multiplier", "rtx_batch_multiplier",
        ]:
            env_var = f"{env_prefix}{field_name.upper()}"
            if env_var in os.environ:
                try:
                    value = os.environ[env_var]
                    if "." in value:
                        setattr(config, field_name, float(value))
                    else:
                        setattr(config, field_name, int(value))
                except ValueError:
                    pass
        return config


# Default GPU scaling configuration (can be overridden via from_env())
_gpu_scaling_config: Optional[GpuScalingConfig] = None


def get_gpu_scaling_config() -> GpuScalingConfig:
    """Get the GPU scaling configuration, loading from env if not set."""
    global _gpu_scaling_config
    if _gpu_scaling_config is None:
        _gpu_scaling_config = GpuScalingConfig.from_env()
    return _gpu_scaling_config


def set_gpu_scaling_config(config: GpuScalingConfig) -> None:
    """Set a custom GPU scaling configuration."""
    global _gpu_scaling_config
    _gpu_scaling_config = config


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


def _scale_batch_size_for_gpu(
    base_batch: int,
    policy_size: int = 7000,
    config: Optional[GpuScalingConfig] = None,
) -> int:
    """Scale batch size based on available GPU memory.

    Uses GpuScalingConfig for all thresholds and multipliers, making the
    scaling behavior configurable via environment variables or explicit config.

    Args:
        base_batch: Base batch size to scale from
        policy_size: Size of policy output (affects memory per sample)
        config: Optional custom config; uses get_gpu_scaling_config() if None
    """
    cfg = config or get_gpu_scaling_config()
    gpu_mem = _get_gpu_memory_gb()

    if gpu_mem <= 0:
        return base_batch  # No GPU, use default

    # Estimate memory per sample based on policy size
    if policy_size > cfg.large_policy_threshold:
        mem_per_sample_mb = cfg.mem_per_sample_large_policy_mb
    elif policy_size > cfg.medium_policy_threshold:
        mem_per_sample_mb = cfg.mem_per_sample_medium_policy_mb
    else:
        mem_per_sample_mb = cfg.mem_per_sample_small_policy_mb

    # Reserve memory for model + overhead
    available_gb = max(0, gpu_mem - cfg.reserved_memory_gb)
    available_mb = available_gb * 1024

    # Calculate max batch size that fits in memory
    max_batch = int(available_mb / mem_per_sample_mb)

    # Scale up from base based on GPU tier
    if gpu_mem >= cfg.gh200_memory_threshold_gb:
        scaled = base_batch * cfg.gh200_batch_multiplier
    elif gpu_mem >= cfg.h100_memory_threshold_gb:
        scaled = base_batch * cfg.h100_batch_multiplier
    elif gpu_mem >= cfg.a100_memory_threshold_gb:
        scaled = base_batch * cfg.a100_batch_multiplier
    elif gpu_mem >= cfg.rtx_memory_threshold_gb:
        scaled = base_batch * cfg.rtx_batch_multiplier
    else:
        scaled = base_batch * cfg.consumer_batch_multiplier

    return min(scaled, max_batch, cfg.max_batch_size)


class BatchSizeAutoTuner:
    """
    Automatically find optimal batch size via profiling.

    Uses binary search with actual forward/backward passes to find the
    largest batch size that fits in GPU memory. This achieves 15-30%
    better throughput than static estimates.

    Usage:
        tuner = BatchSizeAutoTuner(model, sample_features, sample_globals, device)
        optimal_batch = tuner.find_optimal_batch_size()
    """

    def __init__(
        self,
        model,
        sample_features,
        sample_globals,
        device,
        policy_size: int = 7000,
    ):
        """
        Initialize the auto-tuner.

        Args:
            model: The neural network model
            sample_features: Sample feature tensor (1, feature_dim)
            sample_globals: Sample globals tensor (1, globals_dim)
            device: Target device
            policy_size: Size of policy output
        """
        import torch
        self._model = model
        self._sample_features = sample_features
        self._sample_globals = sample_globals
        self._device = device
        self._policy_size = policy_size
        self._torch = torch

    def find_optimal_batch_size(
        self,
        min_batch: int = 32,
        max_batch: int = 8192,
        target_memory_fraction: float = 0.85,
    ) -> int:
        """
        Binary search for largest batch that fits in memory.

        Args:
            min_batch: Minimum batch size to try
            max_batch: Maximum batch size to try
            target_memory_fraction: Target GPU memory utilization (0.0-1.0)

        Returns:
            Optimal batch size
        """
        if not self._torch.cuda.is_available():
            return min_batch

        # Get total GPU memory
        props = self._torch.cuda.get_device_properties(self._device)
        total_memory = props.total_memory

        # Binary search for optimal batch size
        low, high = min_batch, max_batch
        best_batch = min_batch

        while low <= high:
            mid = (low + high) // 2
            success, memory_used = self._profile_batch(mid)

            if success:
                memory_ratio = memory_used / total_memory
                if memory_ratio <= target_memory_fraction:
                    best_batch = mid
                    low = mid + 1
                else:
                    # Using too much memory, reduce
                    high = mid - 1
            else:
                # OOM, reduce batch size
                high = mid - 1

        return best_batch

    def _profile_batch(self, batch_size: int) -> tuple:
        """
        Profile a single batch size.

        Returns:
            (success: bool, memory_used_bytes: int)
        """
        self._torch.cuda.reset_peak_memory_stats(self._device)
        self._torch.cuda.empty_cache()

        try:
            # Create batch by repeating samples
            features = self._sample_features.repeat(batch_size, 1, 1, 1)
            globals_vec = self._sample_globals.repeat(batch_size, 1)

            features = features.to(self._device)
            globals_vec = globals_vec.to(self._device)

            # Create dummy targets
            value_targets = self._torch.zeros(batch_size, 1, device=self._device)
            policy_targets = self._torch.zeros(
                batch_size, self._policy_size, device=self._device
            )

            # Forward pass
            self._model.train()
            value_pred, policy_pred = self._model(features, globals_vec)

            # Compute loss
            value_loss = self._torch.nn.functional.mse_loss(value_pred, value_targets)
            policy_loss = self._torch.nn.functional.cross_entropy(
                policy_pred, policy_targets.argmax(dim=1)
            )
            loss = value_loss + policy_loss

            # Backward pass
            loss.backward()

            # Get peak memory
            peak_memory = self._torch.cuda.max_memory_allocated(self._device)

            # Cleanup
            del features, globals_vec, value_targets, policy_targets
            del value_pred, policy_pred, loss
            self._model.zero_grad()
            self._torch.cuda.empty_cache()

            return True, peak_memory

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self._torch.cuda.empty_cache()
                self._model.zero_grad()
                return False, float('inf')
            raise


def auto_tune_batch_size(
    model,
    device,
    feature_shape: tuple,
    globals_shape: tuple,
    policy_size: int = 7000,
    min_batch: int = 32,
    max_batch: int = 8192,
) -> int:
    """
    Convenience function to auto-tune batch size.

    Args:
        model: Neural network model
        device: Target device
        feature_shape: Shape of feature tensor (C, H, W)
        globals_shape: Shape of globals tensor (D,)
        policy_size: Size of policy output
        min_batch: Minimum batch size
        max_batch: Maximum batch size

    Returns:
        Optimal batch size
    """
    import torch

    if not torch.cuda.is_available():
        return min_batch

    # Create sample tensors
    sample_features = torch.zeros(1, *feature_shape)
    sample_globals = torch.zeros(1, *globals_shape)

    tuner = BatchSizeAutoTuner(
        model, sample_features, sample_globals, device, policy_size
    )

    return tuner.find_optimal_batch_size(min_batch, max_batch)


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

    # Policy label smoothing: mix targets with uniform distribution for regularization
    # smoothed = (1 - eps) * target + eps * uniform
    # Helps prevent overconfident predictions and improves generalization
    policy_label_smoothing: float = 0.0  # 0 = disabled, typical values: 0.05-0.1

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
    # NOTE: use_prefetch was removed (unused) - prefetching controlled in PrefetchIterator
    prefetch_count: int = 2  # Number of batches to prefetch
    pin_memory: bool = True  # Pin memory for faster CPU->GPU transfers (CUDA only)

    # Gradient accumulation: simulate larger batch sizes on memory-constrained GPUs
    # Effective batch size = batch_size * gradient_accumulation_steps
    # Loss is scaled by 1/gradient_accumulation_steps to maintain gradient magnitude
    gradient_accumulation_steps: int = 1  # 1 = disabled (default), 2-8 = typical values

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


# NOTE: BOARD_TRAINING_CONFIGS was removed in 2025-12 cleanup.
# Use get_default_config_for_board() for board-specific settings instead.


# =============================================================================
# Unified Training Pipeline Configuration
# =============================================================================


@dataclass
class SelfPlayConfig:
    """Configuration for self-play data generation.

    Controls how training games are generated during the self-play phase
    of the training pipeline.
    """
    # Number of games to generate per iteration
    games_per_iteration: int = 500

    # Parallel game settings
    batch_size: int = 64  # Games per GPU batch
    num_workers: int = 4  # CPU worker processes for game setup

    # Game rules
    max_moves: int = 2000  # Maximum moves before draw (minimum 2000 for all boards)
    repetition_threshold: int = 3  # Draw on N-fold repetition (0 = disabled)
    swap_enabled: bool = False  # Pie rule for 2-player games

    # Move selection
    temperature: float = 1.0  # MCTS/policy temperature
    noise_scale: float = 0.25  # Dirichlet noise scale
    random_opening_moves: int = 0  # Random moves at start for diversity


@dataclass
class DataConfig:
    """Configuration for training data handling."""
    # Data paths
    data_dir: str = "data/training"
    cache_dir: str = "data/cache"

    # Train/validation split
    val_split: float = 0.1

    # Data augmentation
    enable_augmentation: bool = True

    # Prefetch settings (actual prefetching is controlled in PrefetchIterator)
    prefetch_count: int = 2
    pin_memory: bool = True

    # Streaming settings
    use_streaming: bool = False  # Stream from disk vs load all to memory


@dataclass
class CheckpointConfig:
    """Configuration for model checkpointing."""
    # Checkpoint directory
    checkpoint_dir: str = "data/checkpoints"

    # Auto-resume
    auto_resume: bool = True

    # Async checkpointing (non-blocking I/O)
    async_save: bool = True


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation during training."""
    # Evaluation frequency
    eval_interval_steps: int = 1000

    # Evaluation games
    games_per_eval: int = 20

    # Background evaluation (run in separate process for continuous Elo tracking)
    enable_background_eval: bool = True

    # Elo tracking
    initial_elo: float = 1500.0


@dataclass
class TrainingPipelineConfig:
    """Unified configuration for the entire training pipeline.

    This config brings together all aspects of the training pipeline:
    - Model architecture (via TrainConfig)
    - GPU memory management (via GpuScalingConfig)
    - Self-play data generation
    - Data handling
    - Checkpointing
    - Evaluation

    Example usage:
        # Create default config for hex8
        config = TrainingPipelineConfig.for_board_type(BoardType.HEX8)

        # Override specific settings
        config.train.learning_rate = 1e-4
        config.selfplay.games_per_iteration = 1000

        # Save config for reproducibility
        config.save("experiments/hex8_run1/config.yaml")

        # Load config
        config = TrainingPipelineConfig.load("experiments/hex8_run1/config.yaml")
    """
    # Core training config (model architecture, hyperparameters)
    train: TrainConfig = field(default_factory=TrainConfig)

    # GPU scaling config
    gpu: GpuScalingConfig = field(default_factory=GpuScalingConfig)

    # Self-play config
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)

    # Data handling config
    data: DataConfig = field(default_factory=DataConfig)

    # Checkpoint config
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    # Evaluation config
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Experiment metadata
    experiment_name: str = ""
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def for_board_type(
        cls,
        board_type: BoardType,
        num_players: int = 2,
    ) -> "TrainingPipelineConfig":
        """Create a config optimized for a specific board type.

        Args:
            board_type: Board type to configure for
            num_players: Number of players (affects model architecture)

        Returns:
            TrainingPipelineConfig with board-appropriate defaults
        """
        config = cls()
        config.train = get_training_config_for_board(board_type)

        # Adjust selfplay settings based on board complexity
        if board_type in (BoardType.SQUARE19, BoardType.HEXAGONAL):
            # Larger boards need more diverse games
            config.selfplay.games_per_iteration = 1000
            config.selfplay.batch_size = 32  # Smaller batches for memory
            config.selfplay.max_moves = 1000
        elif board_type == BoardType.HEX8:
            config.selfplay.games_per_iteration = 500
            config.selfplay.batch_size = 64
            config.selfplay.max_moves = 500
        else:  # SQUARE8
            config.selfplay.games_per_iteration = 500
            config.selfplay.batch_size = 64
            config.selfplay.max_moves = 500

        # Set experiment metadata
        config.experiment_name = f"{board_type.value}_{num_players}p"
        config.tags = {
            "board_type": board_type.value,
            "num_players": str(num_players),
        }

        return config

    @classmethod
    def from_env(cls) -> "TrainingPipelineConfig":
        """Create config from environment variables.

        Environment variables (all optional, with RINGRIFT_ prefix):
            RINGRIFT_BOARD_TYPE: Board type (square8, square19, hex8, hexagonal)
            RINGRIFT_NUM_PLAYERS: Number of players
            RINGRIFT_LEARNING_RATE: Learning rate
            RINGRIFT_BATCH_SIZE: Batch size
            RINGRIFT_EPOCHS: Number of epochs
            RINGRIFT_DATA_DIR: Training data directory
            RINGRIFT_CHECKPOINT_DIR: Checkpoint directory
        """
        config = cls()

        # Board type
        board_type_str = os.environ.get("RINGRIFT_BOARD_TYPE", "square8").lower()
        board_type_map = {
            "square8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
            "hex8": BoardType.HEX8,
            "hexagonal": BoardType.HEXAGONAL,
        }
        if board_type_str in board_type_map:
            config = cls.for_board_type(board_type_map[board_type_str])

        # Override from env
        if "RINGRIFT_LEARNING_RATE" in os.environ:
            config.train.learning_rate = float(os.environ["RINGRIFT_LEARNING_RATE"])
        if "RINGRIFT_BATCH_SIZE" in os.environ:
            config.train.batch_size = int(os.environ["RINGRIFT_BATCH_SIZE"])
        if "RINGRIFT_EPOCHS" in os.environ:
            config.train.epochs_per_iter = int(os.environ["RINGRIFT_EPOCHS"])
        if "RINGRIFT_DATA_DIR" in os.environ:
            config.data.data_dir = os.environ["RINGRIFT_DATA_DIR"]
        if "RINGRIFT_CHECKPOINT_DIR" in os.environ:
            config.checkpoint.checkpoint_dir = os.environ["RINGRIFT_CHECKPOINT_DIR"]

        # GPU config from env
        config.gpu = GpuScalingConfig.from_env()

        return config

    def save(self, path: str) -> None:
        """Save config to YAML file.

        Args:
            path: Path to save config (creates directories if needed)
        """
        import json
        from pathlib import Path as PathLib

        # Ensure directory exists
        PathLib(path).parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict for serialization
        config_dict = self._to_dict()

        # Save as YAML if available, else JSON
        try:
            import yaml
            with open(path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        except ImportError:
            with open(path, "w") as f:
                json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingPipelineConfig":
        """Load config from YAML or JSON file.

        Args:
            path: Path to config file

        Returns:
            TrainingPipelineConfig loaded from file
        """
        import json

        with open(path) as f:
            content = f.read()

        # Try YAML first, fall back to JSON
        try:
            import yaml
            config_dict = yaml.safe_load(content)
        except ImportError:
            config_dict = json.loads(content)

        return cls._from_dict(config_dict)

    def _to_dict(self) -> Dict:
        """Convert config to dictionary for serialization."""
        from dataclasses import asdict

        def convert(obj):
            if hasattr(obj, "value"):  # Enum
                return obj.value
            return obj

        result = {
            "train": {
                k: convert(v) for k, v in asdict(self.train).items()
            },
            "gpu": asdict(self.gpu),
            "selfplay": asdict(self.selfplay),
            "data": asdict(self.data),
            "checkpoint": asdict(self.checkpoint),
            "evaluation": asdict(self.evaluation),
            "experiment_name": self.experiment_name,
            "description": self.description,
            "tags": self.tags,
        }
        return result

    @classmethod
    def _from_dict(cls, d: Dict) -> "TrainingPipelineConfig":
        """Create config from dictionary."""
        config = cls()

        # Load train config
        if "train" in d:
            train_dict = d["train"]
            if "board_type" in train_dict:
                board_type_map = {
                    "square8": BoardType.SQUARE8,
                    "square19": BoardType.SQUARE19,
                    "hex8": BoardType.HEX8,
                    "hexagonal": BoardType.HEXAGONAL,
                }
                train_dict["board_type"] = board_type_map.get(
                    train_dict["board_type"].lower(),
                    BoardType.SQUARE8
                )
            config.train = TrainConfig(**{
                k: v for k, v in train_dict.items()
                if k in TrainConfig.__dataclass_fields__
            })

        # Load other configs
        if "gpu" in d:
            config.gpu = GpuScalingConfig(**d["gpu"])
        if "selfplay" in d:
            config.selfplay = SelfPlayConfig(**d["selfplay"])
        if "data" in d:
            config.data = DataConfig(**d["data"])
        if "checkpoint" in d:
            config.checkpoint = CheckpointConfig(**d["checkpoint"])
        if "evaluation" in d:
            config.evaluation = EvaluationConfig(**d["evaluation"])

        # Load metadata
        config.experiment_name = d.get("experiment_name", "")
        config.description = d.get("description", "")
        config.tags = d.get("tags", {})

        return config

    def validate(self) -> list:
        """Validate config and return list of warnings/errors.

        Returns:
            List of validation messages (empty if valid)
        """
        issues = []

        # Check batch size vs GPU memory
        if self.train.batch_size > self.gpu.max_batch_size:
            issues.append(
                f"batch_size ({self.train.batch_size}) exceeds "
                f"gpu.max_batch_size ({self.gpu.max_batch_size})"
            )

        # Check learning rate
        if self.train.learning_rate > 0.01:
            issues.append(
                f"learning_rate ({self.train.learning_rate}) is unusually high"
            )

        # Check selfplay settings
        if self.selfplay.repetition_threshold > 0 and self.selfplay.repetition_threshold < 3:
            issues.append(
                f"repetition_threshold ({self.selfplay.repetition_threshold}) "
                "should be >= 3 to avoid false positives"
            )

        return issues

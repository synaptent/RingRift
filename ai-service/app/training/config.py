"""Training-specific configuration classes.

.. deprecated:: 2025-12
    Many configs in this module have canonical versions in app.config.unified_config.
    For new code, prefer importing from the canonical location:

    - EvaluationConfig → app.config.unified_config.EvaluationConfig (shadow/tournament)
    - SelfPlayConfig → app.training.selfplay_config.SelfplayConfig
    - TrainingConfig → app.config.unified_config.TrainingConfig

    This module's configs (GpuScalingConfig, TrainConfig, DataConfig, etc.)
    remain here for training-loop-specific settings not covered by unified_config.

Configuration Import Guide:
    # Canonical unified configs
    from app.config.unified_config import (
        EvaluationConfig,  # Shadow/tournament evaluation
        TrainingConfig,    # Training hyperparameters
        PromotionConfig,   # Model promotion rules
    )

    # Training-loop-specific configs (this module)
    from app.training.config import (
        GpuScalingConfig,           # GPU memory scaling
        TrainConfig,                # NNUE training args (training-loop-specific)
        TrainingEvaluationConfig,   # Training-loop eval settings
        DataConfig,                 # Data loading settings
        CheckpointConfig,           # Checkpoint settings
        TrainingPipelineConfig,     # Full pipeline config
    )

    # Selfplay config (separate module)
    from app.training.selfplay_config import SelfplayConfig
"""
from __future__ import annotations


import os
import warnings
from dataclasses import dataclass, field

from app.models import BoardType

import logging

logger = logging.getLogger(__name__)

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
    # v2.0: Increased multipliers for better GPU utilization based on benchmarks
    # These scale the base batch size (64) for ParallelGameRunner
    gh200_memory_threshold_gb: float = 90.0   # GH200 class (96GB unified memory)
    gh200_batch_multiplier: int = 64          # 64 * 64 = 4096 games (was 32)

    h100_memory_threshold_gb: float = 70.0    # H100 class (80GB VRAM)
    h100_batch_multiplier: int = 32           # 64 * 32 = 2048 games (was 16)

    a100_memory_threshold_gb: float = 30.0    # A100 class (40-80GB)
    a100_batch_multiplier: int = 16           # 64 * 16 = 1024 games (was 8)

    rtx_memory_threshold_gb: float = 16.0     # RTX 3090/4090 class (24GB)
    rtx_batch_multiplier: int = 8             # 64 * 8 = 512 games (was 4)

    consumer_batch_multiplier: int = 4        # Consumer GPUs (<16GB)

    # Maximum batch size cap (increased for large GPUs)
    max_batch_size: int = 16384

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
_gpu_scaling_config: GpuScalingConfig | None = None


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
    except (ImportError, AttributeError, RuntimeError):
        pass
    return 0.0


def get_selfplay_batch_size_for_gpu(gpu_type: str | None = None) -> int:
    """Get selfplay batch size based on GPU type.

    Reads from unified_loop.yaml gpu_batch_overrides if available.

    Args:
        gpu_type: GPU type string (e.g., 'h100', 'gh200', 'rtx_4090')
                  If None, auto-detects from torch.cuda.get_device_name()

    Returns:
        Batch size (number of games to run in parallel)

    December 2025: Added for GPU-specific selfplay optimization.
    """
    from pathlib import Path
    import yaml

    default_batch = 32

    # Auto-detect GPU type if not provided
    if gpu_type is None:
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0).lower()
                # Map device name to config key
                if 'gh200' in device_name or 'grace hopper' in device_name:
                    gpu_type = 'gh200'
                elif 'h100' in device_name:
                    gpu_type = 'h100'
                elif 'a100' in device_name:
                    if torch.cuda.get_device_properties(0).total_memory > 60e9:
                        gpu_type = 'a100_80gb'
                    else:
                        gpu_type = 'a100'
                elif 'l40s' in device_name:
                    gpu_type = 'l40s'
                elif 'a10' in device_name:
                    gpu_type = 'a10'
                elif '5090' in device_name:
                    gpu_type = 'rtx_5090'
                elif '4090' in device_name:
                    gpu_type = 'rtx_4090'
                elif '3090' in device_name:
                    gpu_type = 'rtx_3090'
                elif '4060' in device_name:
                    gpu_type = 'rtx_4060_ti'
                elif '3060' in device_name:
                    gpu_type = 'rtx_3060'
                else:
                    gpu_type = 'default'
        except (ImportError, AttributeError, RuntimeError):
            gpu_type = 'default'

    # Try to load from unified_loop.yaml
    config_paths = [
        Path(__file__).parent.parent.parent / "config" / "unified_loop.yaml",
        Path("config/unified_loop.yaml"),
    ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)

                gpu_mcts = config.get("selfplay", {}).get("gpu_mcts", {})
                overrides = gpu_mcts.get("gpu_batch_overrides", {})

                if gpu_type in overrides:
                    return overrides[gpu_type]
                if 'default' in overrides:
                    return overrides['default']

                # Fallback to base batch_size
                return gpu_mcts.get("batch_size", default_batch)

            except (FileNotFoundError, OSError, PermissionError, yaml.YAMLError, KeyError, AttributeError):
                pass

    return default_batch


def _scale_batch_size_for_gpu(
    base_batch: int,
    policy_size: int = 7000,
    config: GpuScalingConfig | None = None,
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


def get_optimal_batch_size_from_gpu_memory(
    model_params: int = 10_000_000,  # ~10M params default
    feature_channels: int = 56,
    board_size: int = 8,
    num_players: int = 4,
    target_memory_fraction: float = 0.7,
    min_batch: int = 64,
    max_batch: int = 8192,
) -> int:
    """
    Calculate optimal batch size based on available GPU memory.

    This is a fast heuristic that avoids expensive profiling. It estimates
    memory usage per sample and calculates the largest batch that fits
    in the target memory fraction.

    Args:
        model_params: Number of model parameters (used for gradient memory)
        feature_channels: Number of input feature channels
        board_size: Board size (e.g., 8 for 8x8)
        num_players: Number of players (affects value head size)
        target_memory_fraction: Fraction of GPU memory to target (0.7 = 70%)
        min_batch: Minimum batch size
        max_batch: Maximum batch size

    Returns:
        Optimal batch size

    Example:
        >>> batch = get_optimal_batch_size_from_gpu_memory()
        >>> print(f"Optimal batch size: {batch}")
    """
    try:
        import torch
        if not torch.cuda.is_available():
            logger.info("[AutoBatchSize] No CUDA available, using min_batch")
            return min_batch

        # Get total GPU memory
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory

        # Get currently allocated memory
        allocated_memory = torch.cuda.memory_allocated(device)

        # Available memory for training
        available_memory = total_memory - allocated_memory
        target_memory = int(available_memory * target_memory_fraction)

        # Estimate memory per sample (in bytes)
        # Components:
        # 1. Feature tensor: feature_channels * board_size^2 * 4 bytes (float32)
        # 2. Globals tensor: ~50 floats * 4 bytes
        # 3. Policy targets: ~7000 * 4 bytes (for square8)
        # 4. Value targets: num_players * 4 bytes
        # 5. Intermediate activations: ~10x feature size (conservative)
        # 6. Gradients: same as activations

        feature_mem = feature_channels * board_size * board_size * 4
        globals_mem = 50 * 4
        policy_mem = 7000 * 4  # Conservative for largest policy
        value_mem = num_players * 4
        activation_mem = feature_mem * 10  # Conservative multiplier for activations
        gradient_mem = activation_mem  # Gradients roughly equal activations

        bytes_per_sample = feature_mem + globals_mem + policy_mem + value_mem + activation_mem + gradient_mem

        # Model parameter memory (loaded once, not per sample)
        # Parameters + gradients + optimizer states (Adam: 2x for m and v)
        model_overhead = model_params * 4 * 4  # params + grads + 2 optimizer states

        # Subtract model overhead from available memory
        memory_for_batches = max(0, target_memory - model_overhead)

        # Calculate optimal batch size
        optimal_batch = memory_for_batches // bytes_per_sample

        # Round down to power of 2 for efficiency
        if optimal_batch >= 64:
            optimal_batch = 2 ** (optimal_batch.bit_length() - 1)

        # Clamp to bounds
        optimal_batch = max(min_batch, min(max_batch, optimal_batch))

        logger.info(
            f"[AutoBatchSize] GPU: {total_memory / 1e9:.1f}GB total, "
            f"{available_memory / 1e9:.1f}GB available, "
            f"targeting {target_memory / 1e9:.1f}GB ({target_memory_fraction*100:.0f}%)"
        )
        logger.info(
            f"[AutoBatchSize] Estimated {bytes_per_sample / 1024:.1f}KB/sample, "
            f"model overhead {model_overhead / 1e9:.2f}GB"
        )
        logger.info(f"[AutoBatchSize] Calculated optimal batch size: {optimal_batch}")

        return optimal_batch

    except Exception as e:
        logger.warning(f"[AutoBatchSize] Failed to calculate optimal batch size: {e}")
        return min_batch


@dataclass
class TrainConfig:
    """Configuration for training run.

    Note: batch_size will be auto-scaled based on GPU memory if
    RINGRIFT_AUTO_BATCH_SCALE=1 (default) or explicitly set via CLI.
    """
    board_type: BoardType = BoardType.SQUARE8
    # Number of players (2, 3, or 4). Must match the value head output dimension.
    # This is validated in __post_init__ and used for model config validation.
    num_players: int = 2
    episodes_per_iter: int = 4
    epochs_per_iter: int = 50  # Increased from 4 for stronger models (Dec 2025)
    # Number of self-play + training + evaluation cycles to run in the
    # high-level training loop. This was previously hard-coded; exposing it
    # here makes iterative retraining a first-class configuration parameter.
    iterations: int = 2
    learning_rate: float = 1e-3
    # Jan 13, 2026: Increased from 32 to 128 for stronger gradient signals
    # Larger batches provide more stable gradients and faster convergence
    batch_size: int = 128  # Will be auto-scaled in __post_init__ if GPU available
    # Jan 15, 2026: Increased weight decay to 0.001 based on hyperparameter tuning
    # experiment showing 75% reduction in stale epochs and 89% reduction in overfitting
    weight_decay: float = 1e-3
    history_length: int = 3
    # Feature encoding version. v1 matches legacy encoders; v2 adds
    # chain/forced-elimination signals for hex encoders.
    feature_version: int = 2
    seed: int = 42
    max_moves_per_game: int = 10000
    k_elo: int = 32
    policy_weight: float = 1.0
    # Jan 2026: Value loss weight to balance value vs policy learning.
    # Value loss often dominates in magnitude, so weighting it down improves policy.
    # Recommended: 0.4-0.5 for better policy learning, 1.0 for backward compatibility.
    value_weight: float = 1.0
    rank_dist_weight: float = 0.2

    # Entropy regularization to prevent policy collapse
    # Adds -entropy_weight * H(policy) to the loss, encouraging exploration
    # Based on AlphaZero/MuZero entropy bonus for policy diversity
    entropy_weight: float = 0.01  # Default 0.01 (+10-20 Elo from policy diversity)

    # Learning rate schedule settings for better convergence
    warmup_epochs: int = 1  # LR warmup for training stability
    lr_scheduler: str = "cosine"  # Options: 'none', 'step', 'cosine', 'cosine-warm-restarts'
    lr_min: float = 1e-6  # Minimum LR for cosine annealing
    early_stopping_patience: int = 5  # Epochs without loss improvement before stopping (0=disabled) - Jan 15, 2026: 5 is optimal based on hyperparameter tuning
    elo_early_stopping_patience: int = 10  # Epochs without Elo improvement before stopping (0=disabled)
    elo_min_improvement: float = 5.0  # Minimum Elo gain to reset patience counter

    # Policy label smoothing: mix targets with uniform distribution for regularization
    # smoothed = (1 - eps) * target + eps * uniform
    # Helps prevent overconfident predictions and improves generalization
    policy_label_smoothing: float = 0.05  # Enabled Dec 2025 - softens policy targets for better generalization
    # Allow samples with empty policy targets (value-only supervision).
    # When enabled, training will mask policy loss for those samples instead
    # of filtering them out.
    allow_empty_policies: bool = True

    # Jan 15, 2026: Anti-overfitting settings from hyperparameter tuning experiment
    # These settings reduced stale epochs by 75% and overfitting by 89%
    dropout: float = 0.12  # Dropout rate (up from 0.08) - forces robust feature learning
    # Symmetry augmentation: auto-enable for hex boards (D6 group = 6x data)
    # None = auto (hex boards), True = always, False = never
    augment_symmetry: bool | None = None

    # Model identity used to derive checkpoint filenames. This is kept in sync
    # with NeuralNetAI, which expects checkpoints under
    # "<repo_root>/ai-service/models/<nn_model_id>.pth".
    #
    # NOTE: "v4"/"v5" are model-id / checkpoint lineage prefixes, not Python
    # architecture class names. See ai-service/docs/architecture/MPS_ARCHITECTURE.md.
    #
    # Default to the preferred square8 2p v3-family checkpoint lineage ("v5").
    # V2-family checkpoints ("v4") remain supported as a fallback.
    model_id: str = "ringrift_v5_sq8_2p_2xh100"

    # Board-specific model architecture parameters
    # If None, uses defaults from neural_net.get_model_config_for_board()
    num_res_blocks: int | None = None
    num_filters: int | None = None
    policy_size: int | None = None

    # GPU parallel data generation settings
    # When enabled, uses ParallelGameRunner for 5-10x faster data generation
    # None = auto-detect (True if CUDA available), False = disabled, True = forced
    use_gpu_parallel_datagen: bool | None = None
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

    # Gradient surgery for multi-task learning (PCGrad)
    # Projects conflicting gradients between value/policy heads to prevent oscillation
    # Based on "Gradient Surgery for Multi-Task Learning" (Yu et al., 2020)
    enable_gradient_surgery: bool = True  # PCGrad for multi-task learning (+20-50 Elo)

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

        # Validate num_players (must be 2, 3, or 4)
        if self.num_players not in (2, 3, 4):
            raise ValueError(
                f"num_players must be 2, 3, or 4, got {self.num_players}. "
                f"This determines the value head output dimension."
            )

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

        # Jan 15, 2026: Auto-enable symmetry augmentation for hex boards
        # This provides 6x data augmentation via D6 group (rotations + reflections)
        # Experiment showed this reduces overfitting by 89% on hex boards
        if self.augment_symmetry is None:
            self.augment_symmetry = self.board_type in (BoardType.HEX8, BoardType.HEXAGONAL)


# =============================================================================
# Board-Specific Training Configuration Presets
# =============================================================================
#
# These presets provide optimized training configurations for each board type,
# tuned for their respective action space sizes and computational requirements.


def get_training_config_for_board(
    board_type: BoardType,
    base_config: TrainConfig | None = None,
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
        P_HEX,
        POLICY_SIZE_HEX8,
        POLICY_SIZE_8x8,
        POLICY_SIZE_19x19,
    )

    if board_type == BoardType.SQUARE8:
        # 8x8 board: smaller action space, faster training
        config.policy_size = POLICY_SIZE_8x8  # 7,000
        # Prefer the canonical v3-capacity defaults unless explicitly overridden.
        config.num_res_blocks = 12
        config.num_filters = 192
        # Jan 13, 2026: Increased from 64 to 128 for stronger gradient signals
        config.batch_size = 128  # Larger batches for faster convergence
        config.learning_rate = 2e-3  # Slightly higher LR
        config.max_moves_per_game = 10000  # Allow games to complete naturally
        config.model_id = "ringrift_v5_sq8_2p_2xh100"

    elif board_type == BoardType.SQUARE19:
        # 19x19 board: large action space (~67k), requires smaller batch for stability
        config.policy_size = POLICY_SIZE_19x19  # 67,000
        config.num_res_blocks = 12
        config.num_filters = 192
        # Jan 13, 2026: Increased from 24 to 64 for stronger gradient signals
        config.batch_size = 64  # Increased batch for better convergence
        config.learning_rate = 5e-4  # Lower LR for stability
        config.max_moves_per_game = 10000  # Allow games to complete naturally
        config.model_id = "ringrift_v4_sq19_2p"

    elif board_type == BoardType.HEX8:
        # Hex8 board: smaller hex (radius-4, 61 cells), parallel to square8
        config.policy_size = POLICY_SIZE_HEX8  # 4,500
        config.num_res_blocks = 12
        config.num_filters = 192
        # Jan 13, 2026: Increased from 64 to 128 for stronger gradient signals
        config.batch_size = 128  # Larger batches for faster convergence
        config.learning_rate = 2e-3  # Slightly higher LR like square8
        config.max_moves_per_game = 10000  # Allow games to complete naturally
        config.model_id = "ringrift_v5_hex8_2p"

    elif board_type == BoardType.HEXAGONAL:
        # Hex board: largest action space (~92k), requires careful tuning
        config.policy_size = P_HEX  # 91,876
        config.num_res_blocks = 12
        config.num_filters = 192
        # Jan 13, 2026: Increased from 20 to 64 for stronger gradient signals
        config.batch_size = 64  # Increased batch for better convergence
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


def get_model_version_for_board(
    board_type: BoardType,
    data_path: str | None = None,
) -> str:
    """Get the appropriate model version string for a board type.

    This centralizes the board-to-model-version mapping used by both
    train_loop.py and train.py to avoid duplication.

    Parameters
    ----------
    board_type : BoardType
        The board type to get model version for.
    data_path : str | None
        Optional path to NPZ training data. If provided, auto-detects
        model version from the channel count in the data.

    Returns
    -------
    str
        Model version string. Auto-detected from data if available,
        otherwise defaults to 'v4'.

    Notes
    -----
    **Auto-detection** (December 2025):
    When data_path is provided, detects version from channel count:
    - 40 channels → v2 (hex boards only)
    - 64 channels → v4 (hex v3/v4)
    - 56 channels → v4 (all square versions)

    **v4 (default)**: Best theoretical architecture
    - Multi-head self-attention captures long-range dependencies
    - Spatial policy heads (position-aware, better gradients)
    - NAS-optimized hyperparameters
    - 13 attention blocks with 4 heads each
    - 3-layer value head with rank distribution output
    - ~50% fewer params than v3 while being more capable

    For square boards: 5.1M params (square8/square19)
    For hex boards: 5.5-6.2M params (hex8/hexagonal)
    """
    import os as _os
    # Try auto-detection from data if path provided
    if data_path and _os.path.exists(data_path):
        try:
            import numpy as np
            from app.training.encoder_registry import detect_model_version_from_channels

            with np.load(data_path, mmap_mode="r") as data:
                if "features" in data:
                    in_channels = data["features"].shape[1]
                    detected = detect_model_version_from_channels(
                        in_channels, str(board_type)
                    )
                    if detected:
                        logger.info(
                            f"Auto-detected model version {detected} from "
                            f"{in_channels} channels in {data_path}"
                        )
                        return detected
        except Exception as e:
            logger.debug(f"Model version auto-detection failed: {e}")

    return "v4"  # NAS-optimized attention + spatial policy heads (universal default)


# =============================================================================
# Unified Training Pipeline Configuration
# =============================================================================


@dataclass
class SelfPlayConfig:
    """Configuration for self-play data generation in training pipelines.

    .. deprecated:: 2025-12
        Use :class:`app.training.selfplay_config.SelfplayConfig` instead.
        This simplified config is retained for backwards compatibility with
        existing pipeline code. New code should use the canonical version.

    Controls how training games are generated during the self-play phase
    of the training pipeline.

    Migration::

        # Old:
        from app.training.config import SelfPlayConfig
        # New:
        from app.training.selfplay_config import SelfplayConfig
    """
    # Number of games to generate per iteration
    games_per_iteration: int = 500

    # Parallel game settings
    batch_size: int = 64  # Games per GPU batch
    num_workers: int = 4  # CPU worker processes for game setup

    # Game rules
    max_moves: int = 2000  # Maximum moves before draw (minimum 2000 for all boards)
    repetition_threshold: int = 0  # MUST be 0 - S-invariant guarantees no repetition
    swap_enabled: bool = False  # Pie rule for 2-player games

    # Move selection
    temperature: float = 1.0  # MCTS/policy temperature
    noise_scale: float = 0.25  # Dirichlet noise scale
    random_opening_moves: int = 0  # Random moves at start for diversity

    @property
    def num_games(self) -> int:
        """Alias for games_per_iteration for compatibility with SelfplayConfig."""
        return self.games_per_iteration

    def __post_init__(self):
        warnings.warn(
            "SelfPlayConfig from app.training.config is deprecated since Dec 2025. "
            "Use app.training.selfplay_config.SelfplayConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )


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
class TrainingEvaluationConfig:
    """Configuration for model evaluation during training.

    .. note::
        This is the training-loop evaluation config (eval frequency, games).
        For shadow/tournament evaluation config, use:
        ``from app.config.unified_config import EvaluationConfig``
    """

    # Evaluation frequency
    eval_interval_steps: int = 1000

    # Evaluation games
    games_per_eval: int = 20

    # Background evaluation (run in separate process for continuous Elo tracking)
    enable_background_eval: bool = True

    # Elo tracking
    initial_elo: float = 1500.0


# Backwards-compatible alias
EvaluationConfig = TrainingEvaluationConfig


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
    tags: dict[str, str] = field(default_factory=dict)

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

    def _to_dict(self) -> dict:
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
    def _from_dict(cls, d: dict) -> "TrainingPipelineConfig":
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

"""Training configuration dataclasses for train.py.

These dataclasses group related training parameters to reduce the 96-parameter
signature of train_model() and improve code organization.

Usage:
    from app.training.train_config import (
        TrainingDataConfig,
        DistributedConfig,
        CheckpointConfig,
        EnhancementConfig,
        FaultToleranceConfig,
    )

    data_cfg = TrainingDataConfig(
        data_path="data/training/hex8_2p.npz",
        validate_data=True,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingDataConfig:
    """Configuration for training data loading and validation."""

    data_path: str | list[str] = ""
    use_streaming: bool = False
    data_dir: str | None = None
    sampling_weights: str = "uniform"  # uniform, late_game, phase_emphasis, combined

    # Validation (2025-12)
    validate_data: bool = True
    fail_on_invalid_data: bool = False

    # Freshness check (2025-12)
    skip_freshness_check: bool = False
    max_data_age_hours: float = 1.0
    allow_stale_data: bool = False

    # Quality-aware discovery (2025-12)
    discover_synced_data: bool = False
    min_quality_score: float = 0.0
    include_local_data: bool = True
    include_nfs_data: bool = True


@dataclass
class DistributedConfig:
    """Configuration for distributed training with DDP."""

    distributed: bool = False
    local_rank: int = -1
    scale_lr: bool = False
    lr_scale_mode: str = "linear"  # linear, sqrt
    find_unused_parameters: bool = False


@dataclass
class CheckpointConfig:
    """Configuration for model checkpointing."""

    save_path: str = ""
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 5
    save_all_epochs: bool = True
    resume_path: str | None = None
    init_weights_path: str | None = None
    init_weights_strict: bool = False


@dataclass
class LearningRateConfig:
    """Configuration for learning rate scheduling."""

    warmup_epochs: int = 1
    lr_scheduler: str = "cosine"  # none, step, cosine, cosine-warm-restarts
    lr_min: float = 1e-6
    lr_t0: int = 10
    lr_t_mult: int = 2

    # LR finder (2025-12)
    find_lr: bool = False
    lr_finder_min: float = 1e-7
    lr_finder_max: float = 1.0
    lr_finder_iterations: int = 100

    # Cyclic LR
    cyclic_lr: bool = False
    cyclic_lr_period: int = 5


@dataclass
class EnhancementConfig:
    """Configuration for training enhancements (December 2025)."""

    use_integrated_enhancements: bool = True
    enable_curriculum: bool = False
    enable_augmentation: bool = False
    enable_elo_weighting: bool = True
    enable_auxiliary_tasks: bool = True
    enable_batch_scheduling: bool = False
    enable_background_eval: bool = True

    # Hot data buffer
    use_hot_data_buffer: bool = False
    hot_buffer_size: int = 10000
    hot_buffer_mix_ratio: float = 0.3
    external_hot_buffer: Any | None = None


@dataclass
class FaultToleranceConfig:
    """Configuration for fault tolerance (December 2025)."""

    enable_circuit_breaker: bool = True
    enable_anomaly_detection: bool = True
    gradient_clip_mode: str = "adaptive"  # adaptive, fixed, none
    gradient_clip_max_norm: float = 1.0
    anomaly_spike_threshold: float = 3.0
    anomaly_gradient_threshold: float = 100.0
    enable_graceful_shutdown: bool = True


@dataclass
class ModelArchConfig:
    """Configuration for model architecture."""

    model_version: str = "v2"
    model_type: str = "cnn"  # cnn, gnn, hybrid
    num_res_blocks: int | None = None
    num_filters: int | None = None
    dropout: float = 0.08
    freeze_policy: bool = False

    # Advanced features
    spectral_norm: bool = False
    stochastic_depth: bool = False
    stochastic_depth_prob: float = 0.1


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping."""

    patience: int = 5
    elo_patience: int = 10
    elo_min_improvement: float = 5.0


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""

    enabled: bool = False
    amp_dtype: str = "bfloat16"  # bfloat16, float16


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""

    augment_hex_symmetry: bool = False
    policy_label_smoothing: float = 0.0


@dataclass
class HeartbeatConfig:
    """Configuration for heartbeat monitoring."""

    heartbeat_file: str | None = None
    heartbeat_interval: float = 30.0


@dataclass
class FullTrainingConfig:
    """Complete training configuration combining all sub-configs.

    This is the recommended way to configure training - use this instead
    of passing 96 individual parameters to train_model().
    """

    # Core training settings
    board_type: str = "square8"
    num_players: int = 2
    multi_player: bool = False
    epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 0.001

    # Sub-configurations
    data: TrainingDataConfig = field(default_factory=TrainingDataConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    lr: LearningRateConfig = field(default_factory=LearningRateConfig)
    enhancements: EnhancementConfig = field(default_factory=EnhancementConfig)
    fault_tolerance: FaultToleranceConfig = field(default_factory=FaultToleranceConfig)
    model: ModelArchConfig = field(default_factory=ModelArchConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    mixed_precision: MixedPrecisionConfig = field(default_factory=MixedPrecisionConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)

    # Value whitening
    value_whitening: bool = False
    value_whitening_momentum: float = 0.99

    # EMA
    ema: bool = False
    ema_decay: float = 0.999

    # Adaptive warmup
    adaptive_warmup: bool = False

    # Hard example mining
    hard_example_mining: bool = False
    hard_example_top_k: float = 0.3

    # Auto-tune batch size
    auto_tune_batch_size: bool = True

    # Calibration tracking
    track_calibration: bool = False


def config_from_legacy_params(**kwargs) -> FullTrainingConfig:
    """Convert legacy train_model() parameters to FullTrainingConfig.

    This helper enables backwards compatibility with existing code that
    calls train_model() with individual parameters.

    Usage:
        # Old style
        train_model(config, data_path, save_path, distributed=True, ...)

        # New style (equivalent)
        full_config = config_from_legacy_params(
            data_path=data_path,
            save_path=save_path,
            distributed=True,
            ...
        )
    """
    cfg = FullTrainingConfig()

    # Map legacy params to new config structure
    if "data_path" in kwargs:
        cfg.data.data_path = kwargs["data_path"]
    if "validate_data" in kwargs:
        cfg.data.validate_data = kwargs["validate_data"]
    if "skip_freshness_check" in kwargs:
        cfg.data.skip_freshness_check = kwargs["skip_freshness_check"]

    if "distributed" in kwargs:
        cfg.distributed.distributed = kwargs["distributed"]
    if "local_rank" in kwargs:
        cfg.distributed.local_rank = kwargs["local_rank"]
    if "scale_lr" in kwargs:
        cfg.distributed.scale_lr = kwargs["scale_lr"]

    if "save_path" in kwargs:
        cfg.checkpoint.save_path = kwargs["save_path"]
    if "checkpoint_dir" in kwargs:
        cfg.checkpoint.checkpoint_dir = kwargs["checkpoint_dir"]
    if "resume_path" in kwargs:
        cfg.checkpoint.resume_path = kwargs["resume_path"]

    if "warmup_epochs" in kwargs:
        cfg.lr.warmup_epochs = kwargs["warmup_epochs"]
    if "lr_scheduler" in kwargs:
        cfg.lr.lr_scheduler = kwargs["lr_scheduler"]

    if "enable_curriculum" in kwargs:
        cfg.enhancements.enable_curriculum = kwargs["enable_curriculum"]
    if "enable_elo_weighting" in kwargs:
        cfg.enhancements.enable_elo_weighting = kwargs["enable_elo_weighting"]

    if "enable_circuit_breaker" in kwargs:
        cfg.fault_tolerance.enable_circuit_breaker = kwargs["enable_circuit_breaker"]
    if "gradient_clip_max_norm" in kwargs:
        cfg.fault_tolerance.gradient_clip_max_norm = kwargs["gradient_clip_max_norm"]

    if "model_version" in kwargs:
        cfg.model.model_version = kwargs["model_version"]
    if "model_type" in kwargs:
        cfg.model.model_type = kwargs["model_type"]

    if "early_stopping_patience" in kwargs:
        cfg.early_stopping.patience = kwargs["early_stopping_patience"]
    if "elo_early_stopping_patience" in kwargs:
        cfg.early_stopping.elo_patience = kwargs["elo_early_stopping_patience"]

    if "mixed_precision" in kwargs:
        cfg.mixed_precision.enabled = kwargs["mixed_precision"]
    if "amp_dtype" in kwargs:
        cfg.mixed_precision.amp_dtype = kwargs["amp_dtype"]

    if "augment_hex_symmetry" in kwargs:
        cfg.augmentation.augment_hex_symmetry = kwargs["augment_hex_symmetry"]

    if "heartbeat_file" in kwargs:
        cfg.heartbeat.heartbeat_file = kwargs["heartbeat_file"]

    return cfg


__all__ = [
    "TrainingDataConfig",
    "DistributedConfig",
    "CheckpointConfig",
    "LearningRateConfig",
    "EnhancementConfig",
    "FaultToleranceConfig",
    "ModelArchConfig",
    "EarlyStoppingConfig",
    "MixedPrecisionConfig",
    "AugmentationConfig",
    "HeartbeatConfig",
    "FullTrainingConfig",
    "config_from_legacy_params",
]

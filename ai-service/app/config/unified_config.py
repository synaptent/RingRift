"""Unified Configuration Module for RingRift AI Self-Improvement System.

This module provides a SINGLE SOURCE OF TRUTH for all configuration values
used across the distributed training, evaluation, and promotion system.

All scripts should import config from this module instead of hardcoding values
or using scattered environment variables.

Usage:
    from app.config.unified_config import get_config, UnifiedConfig

    config = get_config()  # Loads from config/unified_loop.yaml

    # Access training thresholds
    threshold = config.training.trigger_threshold_games

    # Access evaluation settings
    shadow_games = config.evaluation.shadow_games_per_config

    # Access all 9 board configurations
    for board_config in config.get_all_board_configs():
        print(f"{board_config.board_type}_{board_config.num_players}p")

Environment Variable Overrides:
    - RINGRIFT_CONFIG_PATH: Override config file path
    - RINGRIFT_TRAINING_THRESHOLD: Override training trigger threshold
    - RINGRIFT_ELO_DB: Override Elo database path
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# Import canonical threshold constants
try:
    from app.config.thresholds import (
        ELO_DROP_ROLLBACK,
        ELO_K_FACTOR,
        INITIAL_ELO_RATING,
        MIN_GAMES_FOR_ELO,
        TRAINING_MIN_INTERVAL_SECONDS,
        TRAINING_TRIGGER_GAMES,
    )
except ImportError:
    # Fallback defaults if thresholds module not available
    INITIAL_ELO_RATING = 1500.0
    ELO_K_FACTOR = 32
    ELO_DROP_ROLLBACK = 50.0
    MIN_GAMES_FOR_ELO = 30
    TRAINING_TRIGGER_GAMES = 500
    TRAINING_MIN_INTERVAL_SECONDS = 1200

# Import centralized path utilities
try:
    from app.utils.paths import AI_SERVICE_ROOT, ensure_dir, ensure_parent_dir
except ImportError:
    AI_SERVICE_ROOT = Path(__file__).parent.parent.parent
    def ensure_dir(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path
    def ensure_parent_dir(path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

logger = logging.getLogger(__name__)

# Default config path relative to ai-service root
DEFAULT_CONFIG_PATH = "config/unified_loop.yaml"

# Singleton instance
_config_instance: UnifiedConfig | None = None


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion from remote hosts."""
    poll_interval_seconds: int = 60
    ephemeral_poll_interval_seconds: int = 15  # Aggressive sync for RAM disk hosts
    sync_method: str = "incremental"
    deduplication: bool = True
    min_games_per_sync: int = 5
    remote_db_pattern: str = "data/games/*.db"
    remote_selfplay_patterns: list[str] = field(default_factory=lambda: [
        "data/selfplay/gpu_*/games*.jsonl",
        "data/selfplay/p2p_gpu/*/games*.jsonl",
        "data/games/gpu_selfplay/*/games*.jsonl",
    ])
    # sync_disabled: When true, data sync is disabled on this machine (orchestrator-only mode)
    # Set to true on machines with limited disk space that shouldn't collect data locally
    sync_disabled: bool = False
    use_external_sync: bool = False  # Whether to use external unified_data_sync.py
    checksum_validation: bool = True
    retry_max_attempts: int = 3
    retry_base_delay_seconds: int = 5
    dead_letter_enabled: bool = True
    wal_enabled: bool = True
    wal_db_path: str = "data/sync_wal.db"
    elo_replication_enabled: bool = True
    elo_replication_interval_seconds: int = 60


@dataclass
class TrainingConfig:
    """Configuration for automatic training triggers.

    This is THE SINGLE SOURCE OF TRUTH for training thresholds.
    Do not hardcode these values elsewhere.
    """
    trigger_threshold_games: int = 500  # Canonical threshold
    min_interval_seconds: int = 1200  # 20 minutes
    max_concurrent_jobs: int = 1
    prefer_gpu_hosts: bool = True
    warm_start: bool = True
    validation_split: float = 0.1
    # NNUE-specific thresholds (higher requirements)
    nnue_min_games: int = 10000
    nnue_policy_min_games: int = 5000
    cmaes_min_games: int = 20000
    nnue_training_script: str = "scripts/train_nnue.py"
    nn_training_script: str = "scripts/run_nn_training_baseline.py"
    export_script: str = "scripts/export_replay_dataset.py"
    # Encoder version for hex boards: "v3" uses HexStateEncoderV3 (16 channels)
    hex_encoder_version: str = "v3"
    # Simplified 3-signal trigger system (2024-12)
    use_simplified_triggers: bool = True  # Use 3-signal system instead of 8+ signals
    staleness_hours: float = 6.0  # Hours before config is "stale"
    min_win_rate_threshold: float = 0.45  # Below this triggers urgent training
    bootstrap_threshold: int = 50  # Low threshold for configs with 0 models
    # Verbose logging for training scheduler
    verbose: bool = False
    # Optimized training settings
    batch_size: int = 256  # Higher batch size for better GPU utilization
    sampling_weights: str = "victory_type"  # Balance across victory types
    warmup_epochs: int = 5  # LR warmup for stability
    use_optimized_hyperparams: bool = True  # Load from hyperparameters.json
    # Advanced training optimizations (2024-12 improvements)
    use_spectral_norm: bool = True  # Gradient stability via spectral normalization
    use_lars: bool = False  # LARS optimizer for distributed large-batch training
    use_cyclic_lr: bool = True  # Cyclic LR with triangular waves
    cyclic_lr_period: int = 5  # Cycle period in epochs
    use_gradient_profiling: bool = False  # Track gradient norms for diagnostics
    use_mixed_precision: bool = True  # FP16/BF16 mixed precision training
    amp_dtype: str = "bfloat16"  # Prefer BF16 for stability
    gradient_accumulation: int = 1  # Gradient accumulation steps
    # Knowledge distillation (optional)
    use_knowledge_distill: bool = False  # Train from teacher model
    teacher_model_path: str | None = None  # Path to teacher model
    distill_alpha: float = 0.5  # Blend weight (0=pure label, 1=pure teacher)
    distill_temperature: float = 2.0  # Softening temperature
    # Advanced NNUE policy training options (2024-12)
    use_swa: bool = True  # Stochastic Weight Averaging for better generalization
    swa_start_fraction: float = 0.75  # Start SWA at 75% of epochs
    use_ema: bool = True  # Exponential Moving Average for smoother weights
    ema_decay: float = 0.999  # EMA decay rate
    use_progressive_batch: bool = True  # Progressive batch sizing
    min_batch_size: int = 64  # Starting batch size
    max_batch_size: int = 512  # Maximum batch size
    focal_gamma: float = 2.0  # Focal loss gamma for hard sample mining
    label_smoothing_warmup: int = 5  # Warmup epochs for label smoothing
    policy_label_smoothing: float = 0.05  # Policy label smoothing factor (0.05-0.1 recommended)
    use_hex_augmentation: bool = True  # D6 symmetry augmentation for hex boards
    hex_augment_count: int = 6  # Number of D6 symmetry augmentations (1-12)
    policy_dropout: float = 0.1  # Dropout rate for policy head regularization
    # Learning rate finder (auto-discovers optimal LR before training)
    use_lr_finder: bool = False  # Run LR finder before training
    lr_finder_iterations: int = 100  # Number of iterations for LR sweep
    # 2024-12 Advanced Training Improvements
    use_value_whitening: bool = True  # Value head whitening for stable training
    value_whitening_momentum: float = 0.99  # Momentum for running stats
    use_stochastic_depth: bool = True  # Stochastic depth regularization
    stochastic_depth_prob: float = 0.1  # Drop probability
    use_adaptive_warmup: bool = True  # Adaptive warmup based on dataset size
    use_hard_example_mining: bool = True  # Focus on difficult examples
    hard_example_top_k: float = 0.3  # Top 30% hardest examples
    use_dynamic_batch: bool = False  # Dynamic batch scheduling (optional)
    dynamic_batch_schedule: str = "linear"  # linear, exponential, or step
    transfer_from_model: str | None = None  # Cross-board transfer learning
    transfer_freeze_epochs: int = 5  # Freeze transferred layers for N epochs
    # Advanced optimizer enhancements
    use_lookahead: bool = True  # Lookahead optimizer wrapper
    lookahead_k: int = 5  # Slow weight update interval
    lookahead_alpha: float = 0.5  # Interpolation factor
    use_adaptive_clip: bool = True  # Adaptive gradient clipping
    use_gradient_noise: bool = False  # Gradient noise injection (optional)
    gradient_noise_variance: float = 0.01  # Initial noise variance
    # Architecture search and pretraining
    use_board_nas: bool = True  # Board-specific neural architecture search
    use_self_supervised: bool = False  # Self-supervised pre-training (optional)
    ss_epochs: int = 10  # Self-supervised pre-training epochs
    ss_projection_dim: int = 128  # Projection dimension for contrastive learning
    ss_temperature: float = 0.07  # Contrastive loss temperature
    use_online_bootstrap: bool = True  # Online bootstrapping with soft labels
    bootstrap_temperature: float = 1.5  # Soft label temperature
    bootstrap_start_epoch: int = 10  # Epoch to start bootstrapping
    # Phase 2 Advanced Training (2024-12)
    use_prefetch_gpu: bool = True  # GPU prefetching for improved throughput
    use_attention: bool = False  # Positional attention (experimental)
    attention_heads: int = 4  # Number of attention heads
    use_moe: bool = False  # Mixture of Experts (experimental)
    moe_experts: int = 4  # Number of experts
    moe_top_k: int = 2  # Top-k expert selection
    use_multitask: bool = False  # Multi-task learning heads
    multitask_weight: float = 0.1  # Auxiliary task weight
    use_difficulty_curriculum: bool = True  # Difficulty-aware curriculum
    curriculum_initial_threshold: float = 0.9  # Start with easy samples
    curriculum_final_threshold: float = 0.3  # End with all samples
    use_lamb: bool = False  # LAMB optimizer for large batch
    use_gradient_compression: bool = False  # Gradient compression for distributed
    compression_ratio: float = 0.1  # Keep top 10% of gradients
    use_quantized_eval: bool = True  # Quantized inference for validation
    use_contrastive: bool = False  # Contrastive representation learning
    contrastive_weight: float = 0.1  # Contrastive loss weight
    # Phase 3 Advanced Training (2024-12)
    use_sam: bool = False  # Sharpness-Aware Minimization for better generalization
    sam_rho: float = 0.05  # SAM neighborhood size
    use_td_lambda: bool = False  # TD(lambda) value learning
    td_lambda_value: float = 0.95  # Lambda value for TD learning
    use_dynamic_batch_gradient: bool = False  # Gradient noise-based batch sizing
    dynamic_batch_max: int = 4096  # Maximum batch size for dynamic batching
    use_pruning: bool = False  # Structured pruning after training
    pruning_ratio: float = 0.3  # Fraction of neurons to prune
    use_game_phase_network: bool = False  # Phase-specialized sub-networks
    use_auxiliary_targets: bool = False  # Auxiliary value targets
    auxiliary_weight: float = 0.1  # Auxiliary loss weight
    use_grokking_detection: bool = True  # Monitor for delayed generalization
    use_self_play: bool = False  # Integrated self-play data generation
    self_play_buffer: int = 100000  # Self-play position buffer size
    use_distillation: bool = False  # Knowledge distillation
    distillation_teacher_path: str | None = None  # Path to teacher model
    distillation_temp: float = 4.0  # Distillation temperature
    distillation_alpha: float = 0.7  # Soft vs hard targets weight
    # NNUE policy training script
    nnue_policy_script: str = "scripts/train_nnue_policy.py"
    nnue_curriculum_script: str = "scripts/train_nnue_policy_curriculum.py"
    # =========================================================================
    # Advanced Training Utilities (2025-12)
    # =========================================================================
    # Gradient Checkpointing - trade compute for memory
    use_gradient_checkpointing: bool = False  # Enable for large models
    gradient_checkpoint_layers: list[str] | None = None  # Specific layers to checkpoint
    # PFSP Opponent Pool - Prioritized Fictitious Self-Play
    use_pfsp: bool = True  # Enable diverse opponent selection
    pfsp_max_pool_size: int = 20  # Maximum opponents in pool
    pfsp_hard_opponent_weight: float = 0.7  # Weight for hard opponents (0-1)
    pfsp_diversity_weight: float = 0.2  # Weight for opponent diversity
    pfsp_recency_weight: float = 0.1  # Weight for recent opponents
    # CMA-ES Auto-Tuning - Automatic HP optimization on plateau
    use_cmaes_auto_tuning: bool = True  # Enable auto HP optimization
    cmaes_plateau_patience: int = 10  # Epochs without improvement before triggering
    cmaes_min_epochs_between: int = 50  # Minimum epochs between auto-tunes
    cmaes_max_auto_tunes: int = 3  # Maximum auto-tunes per training run
    cmaes_generations: int = 30  # CMA-ES generations per optimization
    cmaes_population_size: int = 15  # CMA-ES population size
    # LR Finder - Optimal learning rate detection
    lr_finder_min_lr: float = 1e-7  # Minimum LR to test
    lr_finder_max_lr: float = 10.0  # Maximum LR to test
    lr_finder_smooth_factor: float = 0.05  # Loss smoothing factor
    # =========================================================================
    # Phase 4: Training Stability & Acceleration (2024-12)
    # =========================================================================
    # Training Stability Monitor - Auto-detect and recover from instabilities
    use_stability_monitor: bool = True  # Monitor gradient/loss health
    stability_auto_recover: bool = True  # Auto-reduce LR on instability
    gradient_clip_threshold: float = 10.0  # Gradient norm threshold
    loss_spike_threshold: float = 3.0  # Std devs for spike detection
    # Adaptive Precision - Dynamic FP16/BF16/FP32 switching
    use_adaptive_precision: bool = False  # Auto-switch precision
    initial_precision: str = "bf16"  # Starting precision
    precision_auto_downgrade: bool = True  # Downgrade on overflow
    # Progressive Layer Unfreezing - For fine-tuning
    use_progressive_unfreezing: bool = False  # Gradually unfreeze layers
    unfreezing_num_stages: int = 4  # Number of unfreezing stages
    # SWA with Restarts - Better generalization
    use_swa_restarts: bool = True  # SWA with warm restarts
    swa_start_fraction: float = 0.75  # Start SWA at 75% progress
    swa_restart_period: int = 10  # Epochs between restarts
    swa_num_restarts: int = 3  # Number of restarts
    # Smart Checkpointing - Adaptive checkpoint frequency
    use_smart_checkpoints: bool = True  # Adaptive checkpointing
    checkpoint_top_k: int = 3  # Keep top-k checkpoints
    checkpoint_min_interval: int = 1  # Min epochs between checkpoints
    checkpoint_improvement_threshold: float = 0.01  # Save on X% improvement
    # =========================================================================
    # Phase 5: Production Optimization (2024-12)
    # =========================================================================
    # Gradient Accumulation Scheduling - Dynamic accumulation based on memory
    use_adaptive_accumulation: bool = False  # Auto-adjust accumulation steps
    accumulation_target_memory: float = 0.85  # Target GPU memory utilization
    accumulation_max_steps: int = 16  # Maximum accumulation steps
    # Activation Checkpointing - Trade compute for memory
    use_activation_checkpointing: bool = False  # Checkpoint activations
    checkpoint_ratio: float = 0.5  # Fraction of layers to checkpoint
    # Flash Attention - Memory-efficient attention
    use_flash_attention: bool = False  # Use Flash Attention 2
    # Dynamic Loss Scaling - Adaptive mixed precision
    use_dynamic_loss_scaling: bool = False  # Adaptive loss scaling
    # Elastic Training - Dynamic worker scaling
    use_elastic_training: bool = False  # Support worker join/leave
    elastic_min_workers: int = 1  # Minimum workers
    elastic_max_workers: int = 8  # Maximum workers
    # Streaming NPZ - Large dataset support
    use_streaming_npz: bool = False  # Stream from S3/GCS
    streaming_chunk_size: int = 10000  # Samples per chunk
    # Profiling - Performance analysis
    use_profiling: bool = False  # Enable PyTorch Profiler
    profile_dir: str | None = None  # Profiler output directory
    # A/B Testing - Model comparison
    use_ab_testing: bool = False  # Enable A/B model testing
    ab_min_games: int = 100  # Minimum games for significance
    # =========================================================================
    # Bottleneck Fix Integration (2025-12)
    # =========================================================================
    # Streaming Pipeline - Real-time data ingestion with async DB polling
    use_streaming_pipeline: bool = True  # Enable streaming data pipelines
    streaming_poll_interval: float = 5.0  # Seconds between DB polls
    streaming_buffer_size: int = 10000  # Samples in streaming buffer
    selfplay_db_path: Path = field(default_factory=lambda: Path("data/games"))
    # =========================================================================
    # Parallel Selfplay Temperature Scheduling (2025-12)
    # =========================================================================
    # Local parallel selfplay engine selection
    selfplay_engine: str = "gumbel"  # "descent", "mcts", or "gumbel"
    selfplay_num_workers: int | None = None  # Default: CPU count - 1
    selfplay_games_per_batch: int = 20  # Games per local selfplay batch
    # Temperature scheduling for exploration/exploitation tradeoff
    selfplay_temperature: float = 1.0  # Base move selection temperature
    selfplay_use_temperature_decay: bool = True  # Enable temperature decay per game
    selfplay_move_temp_threshold: int = 30  # Use higher temp for first N moves
    selfplay_opening_temperature: float = 1.5  # Temperature for opening moves
    # Gumbel-MCTS specific parameters
    gumbel_simulations: int = 800  # Simulations per move for Gumbel-MCTS
    gumbel_top_k: int = 16  # Top-k actions for sequential halving
    # Value calibration tracking
    track_calibration: bool = True  # Track value head calibration metrics
    calibration_window_size: int = 5000  # Rolling window for calibration
    # Async Shadow Validation - Non-blocking GPU/CPU parity checking
    use_async_validation: bool = True  # Enable async validation
    validation_sample_rate: float = 0.05  # Fraction of moves to validate (5%)
    parity_failure_threshold: float = 0.10  # Block training above 10% failures
    # Data Quality Gate Enforcement (2025-12)
    enforce_data_quality_gate: bool = True  # Block training on quality failures even without feedback controller
    min_data_quality_for_training: float = 0.7  # Minimum quality score to allow training
    validate_training_data: bool = True  # Validate NPZ files before training
    fail_on_invalid_training_data: bool = False  # Hard-fail vs warn on invalid data
    # Connection Pooling - Thread-local DB connection reuse
    use_connection_pool: bool = True  # Enable connection pooling for WAL
    # Training Auto-Recovery (Phase 7)
    training_max_retries: int = 3  # Max retry attempts on failure
    training_retry_backoff_base: float = 60.0  # Base delay between retries (seconds)
    training_retry_backoff_multiplier: float = 2.0  # Exponential backoff multiplier
    # Post-Promotion Warmup (Phase 7)
    warmup_games_after_promotion: int = 100  # Games to collect before retraining
    warmup_time_after_promotion: float = 1800.0  # Min seconds after promotion (30 min)
    # Training Checkpointing (Phase 7)
    checkpoint_enabled: bool = True  # Save training state for crash recovery
    checkpoint_interval_seconds: float = 300.0  # Save checkpoint every 5 minutes
    checkpoint_path: str | None = None  # Path to checkpoint file (default: data/training_checkpoint.json)
    # A/B Testing for Hyperparameters (Phase 7)
    ab_testing_enabled: bool = False  # Enable A/B testing for new configs
    ab_test_fraction: float = 0.3  # Fraction of configs to use test hyperparameters
    # =========================================================================
    # Phase 7: Training Enhancements Integration (2025-12)
    # =========================================================================
    # Training Anomaly Detection - Auto-halt on NaN/Inf/gradient explosion
    halt_on_nan: bool = True  # Halt training on NaN/Inf loss
    halt_on_gradient_explosion: bool = False  # Halt on gradient norm > threshold
    gradient_norm_threshold: float = 100.0  # Gradient explosion threshold
    max_consecutive_anomalies: int = 5  # Max anomalies before forced halt
    # Configurable Validation Intervals - More frequent validation during training
    validation_interval_steps: int | None = 1000  # Validate every N steps (None=epoch only)
    validation_interval_epochs: float | None = None  # Validate every N epochs (overrides steps)
    validation_subset_size: float = 1.0  # Fraction of val data for fast validation
    adaptive_validation_interval: bool = False  # Adjust interval by loss variance
    # Warm Restarts Learning Rate - SGDR (cosine annealing with warm restarts)
    use_warm_restarts: bool = False  # Enable SGDR schedule
    warm_restart_t0: int = 10  # Initial period (epochs)
    warm_restart_t_mult: int = 2  # Period multiplier after each restart
    warm_restart_eta_min: float = 1e-6  # Minimum learning rate
    # Seed Management - Reproducibility
    training_seed: int | None = None  # Random seed (None=random)
    deterministic_training: bool = False  # Enable CuDNN deterministic mode (slower)
    # Data Quality Freshness - Time-based sample weighting
    freshness_decay_hours: float = 24.0  # Freshness half-life in hours
    freshness_weight: float = 0.2  # Freshness weight in quality score (0-1)
    # Hard Example Mining (enhanced) - Buffer and percentile settings
    hard_example_buffer_size: int = 10000  # Max examples to track
    hard_example_percentile: float = 80.0  # Percentile threshold for hardness
    min_samples_before_mining: int = 1000  # Warmup before mining starts
    # =========================================================================
    # Integrated Training Enhancements (December 2025)
    # =========================================================================
    # Master toggle for integrated enhancements system
    use_integrated_enhancements: bool = True
    # Auxiliary Tasks (Multi-Task Learning) - predict game length, piece count
    auxiliary_tasks_enabled: bool = False
    aux_game_length_weight: float = 0.1
    aux_piece_count_weight: float = 0.1
    aux_outcome_weight: float = 0.05
    # Gradient Surgery (PCGrad) - resolve conflicting gradients
    gradient_surgery_enabled: bool = False
    gradient_surgery_method: str = "pcgrad"  # "pcgrad" or "cagrad"
    # Batch Scheduling - dynamic batch size during training
    batch_scheduling_enabled: bool = False
    batch_initial_size: int = 64
    batch_final_size: int = 512
    batch_schedule_type: str = "linear"
    # Background Evaluation - continuous Elo tracking
    background_eval_enabled: bool = False
    eval_interval_steps: int = 1000
    eval_elo_checkpoint_threshold: float = 10.0
    # ELO Weighting - weight samples by opponent strength
    elo_weighting_enabled: bool = True
    elo_base_rating: float = INITIAL_ELO_RATING
    elo_weight_scale: float = 400.0
    elo_min_weight: float = 0.5
    elo_max_weight: float = 2.0
    # Curriculum Learning - progressive difficulty
    curriculum_enabled: bool = True
    curriculum_auto_advance: bool = True
    # Data Augmentation - board symmetry transforms
    augmentation_enabled: bool = True
    augmentation_mode: str = "all"  # "all", "random", "light"
    # Reanalysis - re-evaluate games with current model
    reanalysis_enabled: bool = False
    reanalysis_blend_ratio: float = 0.5
    # =========================================================================
    # Fault Tolerance (2025-12)
    # =========================================================================
    # Circuit breaker for training operations
    enable_circuit_breaker: bool = True  # Enable training circuit breaker
    # Anomaly detection during training
    enable_anomaly_detection: bool = True  # Detect NaN/Inf and loss spikes
    anomaly_spike_threshold: float = 3.0  # Std devs for spike detection
    anomaly_gradient_threshold: float = 100.0  # Gradient explosion threshold
    # Gradient clipping mode
    gradient_clip_mode: str = "adaptive"  # "adaptive" or "fixed"
    gradient_clip_max_norm: float = 1.0  # Max norm for fixed clipping
    # Graceful shutdown
    enable_graceful_shutdown: bool = True  # Emergency checkpoints on SIGTERM

    def __post_init__(self):
        """Validate TrainingConfig fields after initialization."""
        errors = []

        # Validate thresholds
        if self.trigger_threshold_games < 1:
            errors.append(f"trigger_threshold_games must be >= 1, got {self.trigger_threshold_games}")
        if self.min_interval_seconds < 0:
            errors.append(f"min_interval_seconds must be >= 0, got {self.min_interval_seconds}")
        if self.max_concurrent_jobs < 1:
            errors.append(f"max_concurrent_jobs must be >= 1, got {self.max_concurrent_jobs}")

        # Validate batch sizes
        if self.batch_size < 1:
            errors.append(f"batch_size must be >= 1, got {self.batch_size}")
        if self.min_batch_size < 1:
            errors.append(f"min_batch_size must be >= 1, got {self.min_batch_size}")
        if self.max_batch_size < self.min_batch_size:
            errors.append(f"max_batch_size ({self.max_batch_size}) must be >= min_batch_size ({self.min_batch_size})")

        # Validate ratios (0-1)
        ratio_fields = [
            ("swa_start_fraction", self.swa_start_fraction),
            ("ema_decay", self.ema_decay),
            ("distill_alpha", self.distill_alpha),
            ("validation_sample_rate", self.validation_sample_rate),
            ("parity_failure_threshold", self.parity_failure_threshold),
            ("min_data_quality_for_training", self.min_data_quality_for_training),
        ]
        for name, value in ratio_fields:
            if not 0.0 <= value <= 1.0:
                errors.append(f"{name} must be between 0.0 and 1.0, got {value}")

        # Validate retry settings
        if self.training_max_retries < 0:
            errors.append(f"training_max_retries must be >= 0, got {self.training_max_retries}")
        if self.training_retry_backoff_base <= 0:
            errors.append(f"training_retry_backoff_base must be > 0, got {self.training_retry_backoff_base}")
        if self.training_retry_backoff_multiplier < 1.0:
            errors.append(
                "training_retry_backoff_multiplier must be >= 1.0, "
                f"got {self.training_retry_backoff_multiplier}"
            )

        # Validate epoch settings
        if self.warmup_epochs < 0:
            errors.append(f"warmup_epochs must be >= 0, got {self.warmup_epochs}")

        if errors:
            raise ValueError("TrainingConfig validation failed:\n  " + "\n  ".join(errors))


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation (shadow + full tournaments)."""
    shadow_interval_seconds: int = 900  # 15 minutes
    shadow_games_per_config: int = 15
    full_tournament_interval_seconds: int = 3600  # 1 hour
    full_tournament_games: int = 50
    baseline_models: list[str] = field(default_factory=lambda: ["random", "heuristic", "mcts_100", "mcts_500"])
    min_games_for_elo: int = 30
    elo_k_factor: int = 32
    # Adaptive interval settings - go faster when cluster is healthy
    adaptive_interval_enabled: bool = False
    adaptive_interval_min_seconds: int = 120  # Can go as low as 2 min
    adaptive_interval_max_seconds: int = 600  # Cap at 10 min during high load


@dataclass
class PromotionConfig:
    """Configuration for automatic model promotion."""
    auto_promote: bool = True
    elo_threshold: int = 25
    min_games: int = 50
    significance_level: float = 0.05
    sync_to_cluster: bool = True
    hosts_config_path: str | None = None
    cooldown_seconds: int = 1800  # 30 minutes
    max_promotions_per_day: int = 10
    regression_test: bool = True

    # Aliases for compatibility with PromotionCriteria
    @property
    def min_elo_improvement(self) -> float:
        return float(self.elo_threshold)

    @property
    def min_win_rate(self) -> float:
        return 0.52  # Default win rate threshold

    @property
    def statistical_significance(self) -> float:
        return self.significance_level


@dataclass
class CurriculumConfig:
    """Configuration for adaptive curriculum (Elo-weighted training)."""
    adaptive: bool = True
    rebalance_interval_seconds: int = 3600  # 1 hour
    max_weight_multiplier: float = 1.5
    min_weight_multiplier: float = 0.7
    ema_alpha: float = 0.3
    min_games_for_weight: int = 100

    # Event-driven rebalancing (new)
    rebalance_on_elo_change: bool = True
    elo_change_threshold: int = 20  # December 2025: Lowered from 50 for faster curriculum adaptation


@dataclass
class SafeguardsConfig:
    """Process safeguards to prevent uncoordinated sprawl."""
    max_python_processes_per_host: int = 20
    max_selfplay_processes: int = 2
    max_tournament_processes: int = 1
    max_training_processes: int = 1
    single_orchestrator: bool = True
    # Lambda Labs account terminated Dec 2025, now using nebius-backbone-1
    orchestrator_host: str = "nebius-backbone-1"
    kill_orphans_on_start: bool = True
    process_watchdog: bool = True
    watchdog_interval_seconds: int = 60
    max_process_age_hours: int = 4
    max_subprocess_depth: int = 2
    subprocess_timeout_seconds: int = 3600


@dataclass
class BoardConfig:
    """Configuration for a specific board type and player count."""
    board_type: str
    num_players: int

    @property
    def config_key(self) -> str:
        return f"{self.board_type}_{self.num_players}p"


@dataclass
class RegressionConfig:
    """Configuration for regression testing before promotion."""
    hard_block: bool = True
    test_script: str = "scripts/run_regression_tests.py"
    timeout_seconds: int = 600


@dataclass
class AlertingConfig:
    """Alerting thresholds for monitoring."""
    sync_failure_threshold: int = 5
    training_timeout_hours: int = 4
    elo_drop_threshold: int = 50
    games_per_hour_min: int = 100


@dataclass
class SafetyConfig:
    """Safety thresholds to prevent bad models from being promoted."""
    overfit_threshold: float = 0.15  # Max gap between train/val loss
    min_memory_gb: int = 64  # Minimum RAM required to run unified loop
    max_consecutive_failures: int = 3  # Stop after N consecutive failures
    parity_failure_rate_max: float = 0.10  # Block training if parity failures exceed this
    data_quality_score_min: float = 0.70  # Minimum data quality to proceed


@dataclass
class PlateauDetectionConfig:
    """Plateau detection and automatic hyperparameter search."""
    elo_plateau_threshold: float = 15.0  # Elo gain below this triggers plateau detection
    elo_plateau_lookback: int = 5  # Number of evaluations to look back
    win_rate_degradation_threshold: float = 0.40  # Win rate below this triggers retraining
    plateau_count_for_cmaes: int = 2  # Trigger CMA-ES after this many consecutive plateaus
    plateau_count_for_nas: int = 4  # Trigger NAS after this many consecutive plateaus


@dataclass
class ReplayBufferConfig:
    """Prioritized experience replay buffer settings."""
    priority_alpha: float = 0.6  # Priority exponent
    importance_beta: float = 0.4  # Importance sampling exponent
    capacity: int = 100000  # Max experiences in buffer
    rebuild_interval_seconds: int = 7200  # Rebuild buffer every 2 hours


@dataclass
class ClusterConfig:
    """Cluster orchestration settings (previously hardcoded in cluster_orchestrator.py)."""
    # Target performance
    target_selfplay_games_per_hour: int = 1000  # Target selfplay rate across cluster

    # Health monitoring
    health_check_interval_seconds: int = 60  # Seconds between cluster health checks
    sync_interval_seconds: int = 300  # Seconds between data sync with cluster

    # Host sync intervals (in iterations, where 1 iteration ~= 5 minutes)
    sync_interval: int = 6  # Sync every 6 iterations (30 minutes)
    model_sync_interval: int = 12  # Sync models every 12 iterations (1 hour)
    model_sync_enabled: bool = True

    # Elo calibration
    elo_calibration_interval: int = 72  # Every 72 iterations (6 hours)
    elo_calibration_games: int = 50

    # Elo-driven curriculum learning
    elo_curriculum_enabled: bool = True
    elo_match_window: int = 200
    elo_underserved_threshold: int = 100

    # Auto-scaling
    auto_scale_interval: int = 12
    underutilized_cpu_threshold: int = 30
    underutilized_python_jobs: int = 10
    scale_up_games_per_host: int = 50

    # Adaptive game count
    adaptive_games_min: int = 30
    adaptive_games_max: int = 150


@dataclass
class SSHConfig:
    """SSH execution settings (shared across all orchestrators)."""
    max_retries: int = 3
    base_delay_seconds: float = 2.0
    max_delay_seconds: float = 30.0
    connect_timeout_seconds: int = 10
    command_timeout_seconds: int = 3600  # 1 hour max
    # SSH transport specific (from ssh_transport.py)
    transport_command_timeout_seconds: int = 30  # For P2P commands via SSH
    retry_delay_seconds: float = 1.0
    address_cache_ttl_seconds: int = 300  # 5 minutes


@dataclass
class SlurmConfig:
    """Slurm execution settings for stable HPC clusters."""
    enabled: bool = False
    partition_training: str = "gpu-train"
    partition_selfplay: str = "gpu-selfplay"
    partition_tournament: str = "cpu-eval"
    account: str | None = None
    qos: str | None = None
    default_time_training: str = "08:00:00"
    default_time_selfplay: str = "02:00:00"
    default_time_tournament: str = "02:00:00"
    gpus_training: int = 1
    cpus_training: int = 16
    mem_training: str = "64G"
    gpus_selfplay: int = 0
    cpus_selfplay: int = 8
    mem_selfplay: str = "16G"
    gpus_tournament: int = 0
    cpus_tournament: int = 8
    mem_tournament: str = "16G"
    job_dir: str = "data/slurm/jobs"
    log_dir: str = "data/slurm/logs"
    shared_root: str | None = None
    repo_subdir: str = "ai-service"
    venv_activate: str | None = None
    venv_activate_arm64: str | None = None
    setup_commands: list[str] = field(default_factory=list)
    extra_sbatch_args: list[str] = field(default_factory=list)
    poll_interval_seconds: int = 20


@dataclass
class DistributedConfig:
    """Configuration for distributed system components.

    Centralizes settings previously hardcoded across:
    - dynamic_registry.py
    - ssh_transport.py
    - unified_data_sync.py
    """
    # Node health state thresholds
    degraded_failure_threshold: int = 2   # Failures before node marked degraded
    offline_failure_threshold: int = 5    # Failures before node marked offline
    recovery_success_threshold: int = 2   # Successes needed to recover from degraded

    # API check intervals (seconds)
    vast_api_check_interval_seconds: int = 300   # 5 minutes
    aws_api_check_interval_seconds: int = 300    # 5 minutes
    tailscale_check_interval_seconds: int = 120  # 2 minutes

    # P2P orchestrator settings
    p2p_port: int = 8770
    p2p_base_url: str = "http://localhost:8770"

    # Gossip sync
    gossip_port: int = 8771

    # Data server (for aria2 transport)
    data_server_port: int = 8766


@dataclass
class SelfplayDefaults:
    """Default selfplay settings shared across all selfplay scripts.

    Note: This is for APPLICATION-LEVEL defaults loaded from config files.
    For per-run configuration, use :class:`app.training.selfplay_config.SelfplayConfig`.
    """
    # Game generation
    default_games_per_config: int = 50
    min_games_for_training: int = 500
    max_games_per_session: int = 1000

    # Worker management
    max_concurrent_workers: int = 4
    worker_timeout_seconds: int = 7200  # 2 hours
    checkpoint_interval_games: int = 100

    # Quality settings
    mcts_simulations: int = 200
    temperature: float = 0.5
    noise_fraction: float = 0.25


# Backward compatibility alias (deprecated December 2025)
# Use SelfplayDefaults for app-level defaults or
# app.training.selfplay_config.SelfplayConfig for per-run config
SelfplayConfig = SelfplayDefaults


@dataclass
class TournamentConfig:
    """Tournament settings (shared across all tournament scripts)."""
    # Default game counts
    default_games_per_matchup: int = 20
    shadow_games: int = 15
    full_tournament_games: int = 50

    # Time limits
    game_timeout_seconds: int = 300  # 5 minutes
    tournament_timeout_seconds: int = 7200  # 2 hours

    # Elo calculation
    k_factor: int = 32
    initial_elo: int = 1500
    min_games_for_rating: int = 30

    # Baseline models
    baseline_models: list[str] = field(default_factory=lambda: ["random", "heuristic", "mcts_100", "mcts_500"])


@dataclass
class IntegratedEnhancementsConfig:
    """Configuration for integrated training enhancements.

    Centralizes all advanced training feature flags and parameters.
    See app/training/integrated_enhancements.py for implementation.

    This is the CANONICAL location for this config. The scripts/unified_loop/config.py
    version should import from here.
    """
    # Master toggle
    enabled: bool = True

    # Auxiliary Tasks (Multi-Task Learning)
    auxiliary_tasks_enabled: bool = False
    aux_game_length_weight: float = 0.1
    aux_piece_count_weight: float = 0.1
    aux_outcome_weight: float = 0.05

    # Gradient Surgery (PCGrad)
    gradient_surgery_enabled: bool = False
    gradient_surgery_method: str = "pcgrad"  # "pcgrad" or "cagrad"
    gradient_conflict_threshold: float = 0.0  # Threshold for detecting conflicts

    # Batch Scheduling
    batch_scheduling_enabled: bool = False
    batch_initial_size: int = 64
    batch_final_size: int = 512
    batch_warmup_steps: int = 1000
    batch_rampup_steps: int = 10000
    batch_schedule_type: str = "linear"  # "linear", "exponential", "step"

    # Background Evaluation
    background_eval_enabled: bool = False
    eval_interval_steps: int = 1000
    eval_games_per_check: int = 20
    eval_elo_checkpoint_threshold: float = 10.0
    eval_elo_drop_threshold: float = ELO_DROP_ROLLBACK
    eval_auto_checkpoint: bool = True
    eval_checkpoint_dir: str = "data/eval_checkpoints"

    # ELO Weighting (uses thresholds.py constants)
    elo_weighting_enabled: bool = True
    elo_base_rating: float = INITIAL_ELO_RATING
    elo_weight_scale: float = 400.0
    elo_min_weight: float = 0.5
    elo_max_weight: float = 2.0

    # Curriculum Learning
    curriculum_enabled: bool = True
    curriculum_auto_advance: bool = True
    curriculum_checkpoint_path: str = "data/curriculum_state.json"

    # Data Augmentation
    augmentation_enabled: bool = True
    augmentation_mode: str = "all"  # "all", "random", "light"
    augmentation_probability: float = 1.0

    # Reanalysis
    reanalysis_enabled: bool = False
    reanalysis_blend_ratio: float = 0.5
    reanalysis_interval_steps: int = 5000
    reanalysis_batch_size: int = 1000


@dataclass
class HealthConfig:
    """Configuration for component health monitoring."""
    enabled: bool = True
    check_interval_seconds: int = 30
    # Thresholds for component health (seconds since last successful operation)
    data_collector_stale_threshold: int = 300  # 5 minutes
    evaluator_stale_threshold: int = 1800  # 30 minutes
    training_stale_threshold: int = 7200  # 2 hours (training can be slow)
    # Recovery settings
    auto_restart_on_failure: bool = True
    max_restart_attempts: int = 3
    restart_cooldown_seconds: int = 60
    # Alert settings
    alert_on_degraded: bool = True
    alert_on_unhealthy: bool = True


@dataclass
class PBTConfig:
    """Configuration for Population-Based Training.

    Migrated from scripts/unified_loop/config.py for consolidation.
    """
    enabled: bool = False  # Disabled by default - resource intensive
    population_size: int = 8
    exploit_interval_steps: int = 1000
    tunable_params: list[str] = field(default_factory=lambda: ["learning_rate", "batch_size", "temperature"])
    check_interval_seconds: int = 1800  # Check PBT status every 30 min
    auto_start: bool = False  # Auto-start PBT when training completes


@dataclass
class NASConfig:
    """Configuration for Neural Architecture Search.

    Migrated from scripts/unified_loop/config.py for consolidation.
    """
    enabled: bool = False  # Disabled by default - very resource intensive
    strategy: str = "evolutionary"  # evolutionary, random, bayesian
    population_size: int = 20
    generations: int = 50
    check_interval_seconds: int = 3600  # Check NAS status every hour
    auto_start_on_plateau: bool = False  # Start NAS when Elo plateaus


@dataclass
class PERConfig:
    """Configuration for Prioritized Experience Replay.

    Migrated from scripts/unified_loop/config.py for consolidation.
    """
    enabled: bool = True  # Enabled by default - improves training efficiency
    alpha: float = 0.6  # Priority exponent
    beta: float = 0.4  # Importance sampling exponent
    buffer_capacity: int = 100000
    rebuild_interval_seconds: int = 7200  # Rebuild buffer every 2 hours


@dataclass
class DataLoadingConfig:
    """Configuration for data loading and streaming.

    Centralizes settings previously scattered across:
    - data_loader.py (StreamingDataLoader)
    - streaming_pipeline.py (StreamingDataPipeline)
    - datasets.py (RingRiftDataset)
    """
    # Policy encoding
    policy_size: int = 55000  # Policy vector size
    max_policy_size: int = 60000  # Maximum policy size for safety

    # Batch settings
    batch_size: int = 512
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True

    # Filtering
    filter_empty_policies: bool = True
    max_samples: int | None = None

    # Streaming buffer
    buffer_size: int = 10000
    poll_interval_seconds: float = 5.0
    dedupe_window: int = 50000

    # Multi-source loading
    enable_multi_source: bool = True
    max_sources: int = 10

    # Phase-based weighting (consolidated from 3 locations)
    phase_weights: dict[str, float] = field(default_factory=lambda: {
        "ring_placement": 0.8,
        "movement": 1.0,
        "capture": 1.2,
        "late_game": 1.5,
    })

    # Victory type weighting
    victory_type_weights: dict[str, float] = field(default_factory=lambda: {
        "elimination": 1.0,
        "territory": 1.2,
        "ring_out": 0.8,
        "timeout": 0.5,
    })


@dataclass
class QualityConfig:
    """Configuration for game quality scoring and weighting.

    Consolidates quality computation from:
    - game_quality_scorer.py
    - quality_extractor.py
    - streaming_pipeline.py
    - unified_manifest.py

    This is the SINGLE SOURCE OF TRUTH for quality scoring parameters.
    """
    # Game quality component weights (must sum to 1.0)
    outcome_weight: float = 0.25
    length_weight: float = 0.25
    phase_balance_weight: float = 0.20
    diversity_weight: float = 0.15
    source_reputation_weight: float = 0.15

    # Sync priority weights (for quality_extractor.py compatibility)
    sync_elo_weight: float = 0.4
    sync_length_weight: float = 0.3
    sync_decisive_weight: float = 0.3

    # Elo normalization bounds
    min_elo: float = 1200.0
    max_elo: float = 2400.0
    default_elo: float = 1500.0

    # Game length normalization
    min_game_length: int = 10
    max_game_length: int = 200
    optimal_game_length: int = 80

    # Sampling weights
    quality_weight: float = 0.4
    recency_weight: float = 0.3
    priority_weight: float = 0.3

    # Recency decay
    recency_half_life_hours: float = 24.0

    # Thresholds
    min_quality_for_training: float = 0.3
    min_quality_for_priority_sync: float = 0.5
    high_quality_threshold: float = 0.7

    # Decisive outcome credit
    decisive_bonus: float = 1.0
    draw_credit: float = 0.3

    def get_sync_weights(self) -> dict[str, float]:
        """Get weights for sync quality extraction."""
        return {
            "elo_weight": self.sync_elo_weight,
            "length_weight": self.sync_length_weight,
            "decisive_weight": self.sync_decisive_weight,
        }


@dataclass
class StoragePathsConfig:
    """Storage paths for a specific provider."""
    selfplay_games: str = "data/selfplay"
    model_checkpoints: str = "models/checkpoints"
    training_data: str = "data/training"
    elo_database: str = "data/unified_elo.db"
    sync_staging: str = "data/sync_staging"
    local_scratch: str = "/tmp/ringrift"
    nfs_base: str | None = None  # NFS base path if applicable
    use_nfs_for_sync: bool = False
    skip_rsync_to_nfs_nodes: bool = False


@dataclass
class ProviderDetectionConfig:
    """Rules for detecting which storage provider to use."""
    check_path: str = ""
    hostname_patterns: list[str] = field(default_factory=list)
    os_type: str | None = None


@dataclass
class StorageConfig:
    """Storage configuration with provider-specific paths."""
    default: StoragePathsConfig = field(default_factory=StoragePathsConfig)
    lambda_: StoragePathsConfig = field(default_factory=lambda: StoragePathsConfig(
        nfs_base="/lambda/nfs/RingRift",
        selfplay_games="/lambda/nfs/RingRift/selfplay",
        model_checkpoints="/lambda/nfs/RingRift/models",
        training_data="/lambda/nfs/RingRift/training_data",
        elo_database="/lambda/nfs/RingRift/elo/unified_elo.db",
        sync_staging="/lambda/nfs/RingRift/sync_staging",
        local_scratch="/tmp/ringrift",
        use_nfs_for_sync=True,
        skip_rsync_to_nfs_nodes=True,
    ))
    vast: StoragePathsConfig = field(default_factory=lambda: StoragePathsConfig(
        selfplay_games="/workspace/data/selfplay",
        model_checkpoints="/workspace/models",
        training_data="/workspace/data/training",
        elo_database="/workspace/data/unified_elo.db",
        sync_staging="/workspace/data/sync_staging",
        local_scratch="/dev/shm/ringrift",
        use_nfs_for_sync=False,
    ))
    mac: StoragePathsConfig = field(default_factory=lambda: StoragePathsConfig(
        local_scratch="/tmp/ringrift",
    ))
    # Provider detection rules
    provider_detection: dict[str, ProviderDetectionConfig] = field(default_factory=dict)


@dataclass
class FeedbackConfig:
    """Configuration for pipeline feedback controller integration.

    Migrated from scripts/unified_loop/config.py for consolidation.
    """
    enabled: bool = True  # Enable closed-loop feedback
    # Performance-based training triggers
    elo_plateau_threshold: float = 15.0  # Elo gain below this triggers plateau detection
    elo_plateau_lookback: int = 5  # Number of evaluations to look back
    win_rate_degradation_threshold: float = 0.40  # Win rate below this triggers retraining
    # Data quality gates
    max_parity_failure_rate: float = 0.10  # Block training if parity failures exceed this
    min_data_quality_score: float = 0.70  # Minimum data quality to proceed with training
    # CMA-ES/NAS auto-trigger
    plateau_count_for_cmaes: int = 2  # Trigger CMA-ES after this many consecutive plateaus
    plateau_count_for_nas: int = 4  # Trigger NAS after this many consecutive plateaus


@dataclass
class P2PClusterConfig:
    """Configuration for P2P distributed cluster integration.

    Migrated from scripts/unified_loop/config.py for consolidation.
    """
    enabled: bool = False  # Enable P2P cluster coordination
    p2p_base_url: str = "http://localhost:8770"  # P2P orchestrator URL
    auth_token: str | None = None  # Auth token (defaults to RINGRIFT_CLUSTER_AUTH_TOKEN env)
    model_sync_enabled: bool = True  # Auto-sync models to cluster
    model_sync_on_promotion: bool = True  # Auto-sync when model is promoted
    target_selfplay_games_per_hour: int = 1000  # Target selfplay rate across cluster
    auto_scale_selfplay: bool = True  # Auto-scale selfplay workers
    use_distributed_tournament: bool = True  # Use cluster for tournament evaluation
    tournament_nodes_per_eval: int = 3  # Number of nodes per evaluation
    health_check_interval: int = 60  # Seconds between cluster health checks
    unhealthy_threshold: int = 3  # Failures before marking unhealthy
    sync_interval_seconds: int = 300  # Seconds between data sync with cluster
    # Gossip sync integration (P2P data replication)
    gossip_sync_enabled: bool = True  # Gossip-based data replication
    gossip_port: int = 8771  # Port for gossip protocol


@dataclass
class ModelPruningConfig:
    """Configuration for automated model pruning/evaluation.

    Migrated from scripts/unified_loop/config.py for consolidation.
    """
    enabled: bool = True  # Enable automatic model pruning
    threshold: int = 100  # Trigger pruning when model count exceeds this
    check_interval_seconds: int = 3600  # Check model count every hour
    top_quartile_keep: float = 0.25  # Keep top 25% of models
    games_per_baseline: int = 10  # Games per baseline for evaluation
    parallel_workers: int = 50  # Parallel evaluation workers
    archive_models: bool = True  # Archive pruned models instead of deleting
    prefer_high_cpu_hosts: bool = True  # Schedule on high-CPU hosts
    evaluation_timeout_seconds: int = 7200  # 2 hour timeout
    dry_run: bool = False  # If true, log but don't prune
    use_elo_based_culling: bool = True  # Use fast ELO-based culling instead of games


@dataclass
class UnifiedConfig:
    """Master configuration class that loads from unified_loop.yaml.

    This is the SINGLE SOURCE OF TRUTH for all configuration values.
    """
    version: str = "1.2"  # Bumped for config consolidation
    execution_backend: str = "auto"

    # Sub-configurations
    data_ingestion: DataIngestionConfig = field(default_factory=DataIngestionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    safeguards: SafeguardsConfig = field(default_factory=SafeguardsConfig)
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    health: HealthConfig = field(default_factory=HealthConfig)

    # Safety and quality thresholds
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    plateau_detection: PlateauDetectionConfig = field(default_factory=PlateauDetectionConfig)
    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)

    # New unified sections (previously scattered as hardcoded constants)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    ssh: SSHConfig = field(default_factory=SSHConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    selfplay: SelfplayDefaults = field(default_factory=SelfplayDefaults)
    tournament: TournamentConfig = field(default_factory=TournamentConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    # Integrated training enhancements (December 2025)
    enhancements: IntegratedEnhancementsConfig = field(default_factory=IntegratedEnhancementsConfig)

    # Advanced training features (migrated from scripts/unified_loop/config.py)
    pbt: PBTConfig = field(default_factory=PBTConfig)
    nas: NASConfig = field(default_factory=NASConfig)
    per: PERConfig = field(default_factory=PERConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    p2p: P2PClusterConfig = field(default_factory=P2PClusterConfig)
    model_pruning: ModelPruningConfig = field(default_factory=ModelPruningConfig)

    # Storage configuration (provider-specific paths)
    storage: StorageConfig = field(default_factory=StorageConfig)

    # Data loading configuration (December 2025)
    data_loading: DataLoadingConfig = field(default_factory=DataLoadingConfig)

    # Quality scoring configuration (December 2025)
    quality: QualityConfig = field(default_factory=QualityConfig)

    # Paths
    hosts_config_path: str = "config/distributed_hosts.yaml"
    elo_db: str = "data/unified_elo.db"
    data_manifest_db: str = "data/data_manifest.db"
    log_dir: str = "logs/unified_loop"

    # Metrics
    metrics_enabled: bool = True
    metrics_port: int = 9090

    # Board configurations
    _board_configs: list[dict[str, Any]] = field(default_factory=list)

    # Source file for debugging
    _source_path: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> UnifiedConfig:
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        config = cls._from_dict(data)
        config._source_path = str(path)
        return config

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> UnifiedConfig:
        """Create config from dictionary."""
        config = cls()

        def _load_dataclass(dc_type: type[Any], payload: dict[str, Any] | None) -> Any:
            if not payload:
                return dc_type()
            allowed = {f.name for f in fields(dc_type)}
            filtered = {k: v for k, v in payload.items() if k in allowed}
            return dc_type(**filtered)

        # Load version
        config.version = data.get("version", config.version)
        config.execution_backend = data.get("execution_backend", config.execution_backend)

        # Load sub-configurations
        if "data_ingestion" in data:
            data_ingestion = dict(data["data_ingestion"])
            if "sync_interval_seconds" in data_ingestion:
                if "poll_interval_seconds" not in data_ingestion:
                    data_ingestion["poll_interval_seconds"] = data_ingestion["sync_interval_seconds"]
                else:
                    logger.warning(
                        "data_ingestion.sync_interval_seconds is deprecated; "
                        "use data_ingestion.poll_interval_seconds instead."
                    )
                data_ingestion.pop("sync_interval_seconds", None)
            config.data_ingestion = _load_dataclass(DataIngestionConfig, data_ingestion)

        if "training" in data:
            training_data = data["training"]
            config.training = TrainingConfig(
                trigger_threshold_games=training_data.get("trigger_threshold_games", 500),
                min_interval_seconds=training_data.get("min_interval_seconds", 1200),
                max_concurrent_jobs=training_data.get("max_concurrent_jobs", 1),
                prefer_gpu_hosts=training_data.get("prefer_gpu_hosts", True),
                nn_training_script=training_data.get("nn_training_script", "scripts/run_nn_training_baseline.py"),
                export_script=training_data.get("export_script", "scripts/export_replay_dataset.py"),
                hex_encoder_version=training_data.get("hex_encoder_version", "v3"),
                warm_start=training_data.get("warm_start", True),
                validation_split=training_data.get("validation_split", 0.1),
            )

        if "evaluation" in data:
            eval_data = data["evaluation"]
            config.evaluation = EvaluationConfig(
                shadow_interval_seconds=eval_data.get("shadow_interval_seconds", 900),
                shadow_games_per_config=eval_data.get("shadow_games_per_config", 15),
                full_tournament_interval_seconds=eval_data.get("full_tournament_interval_seconds", 3600),
                full_tournament_games=eval_data.get("full_tournament_games", 50),
                baseline_models=eval_data.get("baseline_models", ["random", "heuristic", "mcts_100", "mcts_500"]),
                min_games_for_elo=eval_data.get("min_games_for_elo", 30),
                elo_k_factor=eval_data.get("elo_k_factor", 32),
            )

        if "promotion" in data:
            promo_data = data["promotion"]
            config.promotion = PromotionConfig(
                auto_promote=promo_data.get("auto_promote", True),
                elo_threshold=promo_data.get("elo_threshold", 25),
                min_games=promo_data.get("min_games", 50),
                significance_level=promo_data.get("significance_level", 0.05),
                sync_to_cluster=promo_data.get("sync_to_cluster", True),
                hosts_config_path=promo_data.get("hosts_config_path"),
                cooldown_seconds=promo_data.get("cooldown_seconds", 1800),
                max_promotions_per_day=promo_data.get("max_promotions_per_day", 10),
                regression_test=promo_data.get("regression_test", True),
            )

        if "curriculum" in data:
            curr_data = data["curriculum"]
            config.curriculum = CurriculumConfig(
                adaptive=curr_data.get("adaptive", True),
                rebalance_interval_seconds=curr_data.get("rebalance_interval_seconds", 3600),
                max_weight_multiplier=curr_data.get("max_weight_multiplier", 1.5),
                min_weight_multiplier=curr_data.get("min_weight_multiplier", 0.7),
                ema_alpha=curr_data.get("ema_alpha", 0.3),
                min_games_for_weight=curr_data.get("min_games_for_weight", 100),
            )

        if "safeguards" in data:
            safe_data = data["safeguards"]
            config.safeguards = SafeguardsConfig(
                max_python_processes_per_host=safe_data.get("max_python_processes_per_host", 20),
                max_selfplay_processes=safe_data.get("max_selfplay_processes", 2),
                max_tournament_processes=safe_data.get("max_tournament_processes", 1),
                max_training_processes=safe_data.get("max_training_processes", 1),
                single_orchestrator=safe_data.get("single_orchestrator", True),
                orchestrator_host=safe_data.get("orchestrator_host", "nebius-backbone-1"),
                kill_orphans_on_start=safe_data.get("kill_orphans_on_start", True),
                process_watchdog=safe_data.get("process_watchdog", True),
                watchdog_interval_seconds=safe_data.get("watchdog_interval_seconds", 60),
                max_process_age_hours=safe_data.get("max_process_age_hours", 4),
                max_subprocess_depth=safe_data.get("max_subprocess_depth", 2),
                subprocess_timeout_seconds=safe_data.get("subprocess_timeout_seconds", 3600),
            )

        if "regression" in data:
            reg_data = data["regression"]
            config.regression = RegressionConfig(
                hard_block=reg_data.get("hard_block", True),
                test_script=reg_data.get("test_script", "scripts/run_regression_tests.py"),
                timeout_seconds=reg_data.get("timeout_seconds", 600),
            )

        if "alerting" in data:
            alert_data = data["alerting"]
            config.alerting = AlertingConfig(
                sync_failure_threshold=alert_data.get("sync_failure_threshold", 5),
                training_timeout_hours=alert_data.get("training_timeout_hours", 4),
                elo_drop_threshold=alert_data.get("elo_drop_threshold", 50),
                games_per_hour_min=alert_data.get("games_per_hour_min", 100),
            )

        # Load safety and quality thresholds
        if "safety" in data:
            safety_data = data["safety"]
            config.safety = SafetyConfig(
                overfit_threshold=safety_data.get("overfit_threshold", 0.15),
                min_memory_gb=safety_data.get("min_memory_gb", 64),
                max_consecutive_failures=safety_data.get("max_consecutive_failures", 3),
                parity_failure_rate_max=safety_data.get("parity_failure_rate_max", 0.10),
                data_quality_score_min=safety_data.get("data_quality_score_min", 0.70),
            )

        if "plateau_detection" in data:
            plateau_data = data["plateau_detection"]
            config.plateau_detection = PlateauDetectionConfig(
                elo_plateau_threshold=plateau_data.get("elo_plateau_threshold", 15.0),
                elo_plateau_lookback=plateau_data.get("elo_plateau_lookback", 5),
                win_rate_degradation_threshold=plateau_data.get("win_rate_degradation_threshold", 0.40),
                plateau_count_for_cmaes=plateau_data.get("plateau_count_for_cmaes", 2),
                plateau_count_for_nas=plateau_data.get("plateau_count_for_nas", 4),
            )

        if "replay_buffer" in data:
            rb_data = data["replay_buffer"]
            config.replay_buffer = ReplayBufferConfig(
                priority_alpha=rb_data.get("priority_alpha", 0.6),
                importance_beta=rb_data.get("importance_beta", 0.4),
                capacity=rb_data.get("capacity", 100000),
                rebuild_interval_seconds=rb_data.get("rebuild_interval_seconds", 7200),
            )

        # Load new unified sections (previously hardcoded constants)
        if "cluster" in data:
            cluster_data = data["cluster"]
            config.cluster = ClusterConfig(
                target_selfplay_games_per_hour=cluster_data.get("target_selfplay_games_per_hour", 1000),
                health_check_interval_seconds=cluster_data.get("health_check_interval_seconds", 60),
                sync_interval_seconds=cluster_data.get("sync_interval_seconds", 300),
                sync_interval=cluster_data.get("sync_interval", 6),
                model_sync_interval=cluster_data.get("model_sync_interval", 12),
                model_sync_enabled=cluster_data.get("model_sync_enabled", True),
                elo_calibration_interval=cluster_data.get("elo_calibration_interval", 72),
                elo_calibration_games=cluster_data.get("elo_calibration_games", 50),
                elo_curriculum_enabled=cluster_data.get("elo_curriculum_enabled", True),
                elo_match_window=cluster_data.get("elo_match_window", 200),
                elo_underserved_threshold=cluster_data.get("elo_underserved_threshold", 100),
                auto_scale_interval=cluster_data.get("auto_scale_interval", 12),
                underutilized_cpu_threshold=cluster_data.get("underutilized_cpu_threshold", 30),
                underutilized_python_jobs=cluster_data.get("underutilized_python_jobs", 10),
                scale_up_games_per_host=cluster_data.get("scale_up_games_per_host", 50),
                adaptive_games_min=cluster_data.get("adaptive_games_min", 30),
                adaptive_games_max=cluster_data.get("adaptive_games_max", 150),
            )

        if "ssh" in data:
            ssh_data = data["ssh"]
            config.ssh = SSHConfig(
                max_retries=ssh_data.get("max_retries", 3),
                base_delay_seconds=ssh_data.get("base_delay_seconds", 2.0),
                max_delay_seconds=ssh_data.get("max_delay_seconds", 30.0),
                connect_timeout_seconds=ssh_data.get("connect_timeout_seconds", 10),
                command_timeout_seconds=ssh_data.get("command_timeout_seconds", 3600),
            )

        if "slurm" in data:
            config.slurm = SlurmConfig(**data["slurm"])

        if "selfplay" in data:
            sp_data = data["selfplay"]
            config.selfplay = SelfplayDefaults(
                default_games_per_config=sp_data.get("default_games_per_config", 50),
                min_games_for_training=sp_data.get("min_games_for_training", 500),
                max_games_per_session=sp_data.get("max_games_per_session", 1000),
                max_concurrent_workers=sp_data.get("max_concurrent_workers", 4),
                worker_timeout_seconds=sp_data.get("worker_timeout_seconds", 7200),
                checkpoint_interval_games=sp_data.get("checkpoint_interval_games", 100),
                mcts_simulations=sp_data.get("mcts_simulations", 200),
                temperature=sp_data.get("temperature", 0.5),
                noise_fraction=sp_data.get("noise_fraction", 0.25),
            )

        if "tournament" in data:
            tourn_data = data["tournament"]
            config.tournament = TournamentConfig(
                default_games_per_matchup=tourn_data.get("default_games_per_matchup", 20),
                shadow_games=tourn_data.get("shadow_games", 15),
                full_tournament_games=tourn_data.get("full_tournament_games", 50),
                game_timeout_seconds=tourn_data.get("game_timeout_seconds", 300),
                tournament_timeout_seconds=tourn_data.get("tournament_timeout_seconds", 7200),
                k_factor=tourn_data.get("k_factor", 32),
                initial_elo=tourn_data.get("initial_elo", 1500),
                min_games_for_rating=tourn_data.get("min_games_for_rating", 30),
                baseline_models=tourn_data.get("baseline_models", ["random", "heuristic", "mcts_100", "mcts_500"]),
            )

        # Load advanced training feature configs (migrated from scripts/unified_loop/config.py)
        if "pbt" in data:
            config.pbt = PBTConfig(**data["pbt"])

        if "nas" in data:
            config.nas = NASConfig(**data["nas"])

        if "per" in data:
            config.per = PERConfig(**data["per"])

        if "feedback" in data:
            config.feedback = FeedbackConfig(**data["feedback"])

        if "p2p" in data:
            config.p2p = P2PClusterConfig(**data["p2p"])

        if "model_pruning" in data:
            config.model_pruning = ModelPruningConfig(**data["model_pruning"])

        # Load storage configuration
        if "storage" in data:
            storage_data = data["storage"]
            storage_config = StorageConfig()

            # Load default paths
            if "default" in storage_data:
                storage_config.default = StoragePathsConfig(**storage_data["default"])

            # Load Lambda paths
            if "lambda" in storage_data:
                storage_config.lambda_ = StoragePathsConfig(**storage_data["lambda"])

            # Load Vast paths
            if "vast" in storage_data:
                storage_config.vast = StoragePathsConfig(**storage_data["vast"])

            # Load Mac paths
            if "mac" in storage_data:
                storage_config.mac = StoragePathsConfig(**storage_data["mac"])

            config.storage = storage_config

        # Load provider detection rules
        if "provider_detection" in data:
            detection_data = data["provider_detection"]
            for provider, rules in detection_data.items():
                config.storage.provider_detection[provider] = ProviderDetectionConfig(
                    check_path=rules.get("check_path", ""),
                    hostname_patterns=rules.get("hostname_patterns", []),
                    os_type=rules.get("os_type"),
                )

        # Load paths
        config.hosts_config_path = data.get("hosts_config_path", config.hosts_config_path)
        config.elo_db = data.get("elo_db", config.elo_db)
        config.data_manifest_db = data.get("data_manifest_db", config.data_manifest_db)
        config.log_dir = data.get("log_dir", config.log_dir)
        config.metrics_enabled = data.get("metrics_enabled", config.metrics_enabled)
        config.metrics_port = data.get("metrics_port", config.metrics_port)

        # Load data loading configuration (December 2025)
        if "data_loading" in data:
            dl_data = data["data_loading"]
            config.data_loading = DataLoadingConfig(
                policy_size=dl_data.get("policy_size", 55000),
                max_policy_size=dl_data.get("max_policy_size", 60000),
                batch_size=dl_data.get("batch_size", 512),
                num_workers=dl_data.get("num_workers", 4),
                prefetch_factor=dl_data.get("prefetch_factor", 2),
                pin_memory=dl_data.get("pin_memory", True),
                filter_empty_policies=dl_data.get("filter_empty_policies", True),
                max_samples=dl_data.get("max_samples"),
                buffer_size=dl_data.get("buffer_size", 10000),
                poll_interval_seconds=dl_data.get("poll_interval_seconds", 5.0),
                dedupe_window=dl_data.get("dedupe_window", 50000),
                enable_multi_source=dl_data.get("enable_multi_source", True),
                max_sources=dl_data.get("max_sources", 10),
                phase_weights=dl_data.get("phase_weights", {
                    "ring_placement": 0.8,
                    "movement": 1.0,
                    "capture": 1.2,
                    "late_game": 1.5,
                }),
                victory_type_weights=dl_data.get("victory_type_weights", {
                    "elimination": 1.0,
                    "territory": 1.2,
                    "ring_out": 0.8,
                    "timeout": 0.5,
                }),
            )

        # Load quality configuration (December 2025)
        if "quality" in data:
            q_data = data["quality"]
            config.quality = QualityConfig(
                outcome_weight=q_data.get("outcome_weight", 0.25),
                length_weight=q_data.get("length_weight", 0.25),
                phase_balance_weight=q_data.get("phase_balance_weight", 0.20),
                diversity_weight=q_data.get("diversity_weight", 0.15),
                source_reputation_weight=q_data.get("source_reputation_weight", 0.15),
                sync_elo_weight=q_data.get("sync_elo_weight", 0.4),
                sync_length_weight=q_data.get("sync_length_weight", 0.3),
                sync_decisive_weight=q_data.get("sync_decisive_weight", 0.3),
                min_elo=q_data.get("min_elo", 1200.0),
                max_elo=q_data.get("max_elo", 2400.0),
                default_elo=q_data.get("default_elo", 1500.0),
                min_game_length=q_data.get("min_game_length", 10),
                max_game_length=q_data.get("max_game_length", 200),
                optimal_game_length=q_data.get("optimal_game_length", 80),
                quality_weight=q_data.get("quality_weight", 0.4),
                recency_weight=q_data.get("recency_weight", 0.3),
                priority_weight=q_data.get("priority_weight", 0.3),
                recency_half_life_hours=q_data.get("recency_half_life_hours", 24.0),
                min_quality_for_training=q_data.get("min_quality_for_training", 0.3),
                min_quality_for_priority_sync=q_data.get("min_quality_for_priority_sync", 0.5),
                high_quality_threshold=q_data.get("high_quality_threshold", 0.7),
                decisive_bonus=q_data.get("decisive_bonus", 1.0),
                draw_credit=q_data.get("draw_credit", 0.3),
            )

        # Load board configurations
        config._board_configs = data.get("configurations", [])

        return config

    def get_all_board_configs(self) -> list[BoardConfig]:
        """Get all 9 board configurations."""
        configs = []
        for bc in self._board_configs:
            board_type = bc.get("board_type", "")
            for num_players in bc.get("num_players", []):
                configs.append(BoardConfig(board_type=board_type, num_players=num_players))

        # Fallback to default 9 configs if not specified
        if not configs:
            for board in ["square8", "square19", "hexagonal"]:
                for players in [2, 3, 4]:
                    configs.append(BoardConfig(board_type=board, num_players=players))

        return configs

    def get_elo_db_path(self, base_path: Path | None = None) -> Path:
        """Get absolute path to Elo database."""
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent
        return base_path / self.elo_db

    def validate(self) -> list[str]:
        """Validate configuration values and return list of errors.

        Returns:
            List of error messages. Empty list means valid.
        """
        errors: list[str] = []
        valid_backends = {"auto", "local", "ssh", "p2p", "slurm"}
        backend_value = str(self.execution_backend or "auto").lower()
        if backend_value not in valid_backends:
            errors.append(
                f"execution_backend={self.execution_backend!r} invalid "
                f"(expected one of: {', '.join(sorted(valid_backends))})"
            )

        # Training thresholds
        if self.training.trigger_threshold_games < 10:
            errors.append(f"training.trigger_threshold_games={self.training.trigger_threshold_games} too low (min: 10)")
        if self.training.trigger_threshold_games > 100000:
            errors.append(f"training.trigger_threshold_games={self.training.trigger_threshold_games} too high (max: 100000)")
        if self.training.min_interval_seconds < 60:
            errors.append(f"training.min_interval_seconds={self.training.min_interval_seconds} too low (min: 60)")
        if self.training.validation_split < 0 or self.training.validation_split > 0.5:
            errors.append(f"training.validation_split={self.training.validation_split} out of range (0-0.5)")

        # Evaluation thresholds
        if self.evaluation.shadow_games_per_config < 1:
            errors.append(f"evaluation.shadow_games_per_config={self.evaluation.shadow_games_per_config} too low (min: 1)")
        if self.evaluation.min_games_for_elo < 1:
            errors.append(f"evaluation.min_games_for_elo={self.evaluation.min_games_for_elo} too low (min: 1)")
        if self.evaluation.elo_k_factor < 1 or self.evaluation.elo_k_factor > 100:
            errors.append(f"evaluation.elo_k_factor={self.evaluation.elo_k_factor} out of range (1-100)")

        # Promotion thresholds
        if self.promotion.elo_threshold < 0:
            errors.append(f"promotion.elo_threshold={self.promotion.elo_threshold} cannot be negative")
        if self.promotion.significance_level < 0.001 or self.promotion.significance_level > 0.5:
            errors.append(f"promotion.significance_level={self.promotion.significance_level} out of range (0.001-0.5)")

        # Curriculum weights
        if self.curriculum.max_weight_multiplier < 1.0:
            errors.append(f"curriculum.max_weight_multiplier={self.curriculum.max_weight_multiplier} must be >= 1.0")
        if self.curriculum.min_weight_multiplier > 1.0:
            errors.append(f"curriculum.min_weight_multiplier={self.curriculum.min_weight_multiplier} must be <= 1.0")
        if self.curriculum.min_weight_multiplier > self.curriculum.max_weight_multiplier:
            errors.append("curriculum.min_weight_multiplier > max_weight_multiplier")

        # Safeguards
        if self.safeguards.max_python_processes_per_host < 1:
            errors.append(f"safeguards.max_python_processes_per_host={self.safeguards.max_python_processes_per_host} too low")
        if self.safeguards.max_process_age_hours < 0.5:
            errors.append(f"safeguards.max_process_age_hours={self.safeguards.max_process_age_hours} too low (min: 0.5)")

        # Safety thresholds
        if self.safety.overfit_threshold < 0 or self.safety.overfit_threshold > 1:
            errors.append(f"safety.overfit_threshold={self.safety.overfit_threshold} out of range (0-1)")
        if self.safety.data_quality_score_min < 0 or self.safety.data_quality_score_min > 1:
            errors.append(f"safety.data_quality_score_min={self.safety.data_quality_score_min} out of range (0-1)")

        # Tournament config
        if self.tournament.k_factor < 1 or self.tournament.k_factor > 100:
            errors.append(f"tournament.k_factor={self.tournament.k_factor} out of range (1-100)")
        if self.tournament.initial_elo < 0:
            errors.append(f"tournament.initial_elo={self.tournament.initial_elo} cannot be negative")

        # Selfplay config
        if self.selfplay.mcts_simulations < 1:
            errors.append(f"selfplay.mcts_simulations={self.selfplay.mcts_simulations} too low (min: 1)")
        if self.selfplay.temperature < 0:
            errors.append(f"selfplay.temperature={self.selfplay.temperature} cannot be negative")

        return errors

    def validate_or_raise(self) -> None:
        """Validate and raise ValueError if invalid."""
        errors = self.validate()
        if errors:
            raise ValueError("Config validation failed:\n  " + "\n  ".join(errors))

    def apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        # Training threshold override
        if "RINGRIFT_TRAINING_THRESHOLD" in os.environ:
            self.training.trigger_threshold_games = int(os.environ["RINGRIFT_TRAINING_THRESHOLD"])
            logger.info(f"Training threshold overridden to {self.training.trigger_threshold_games}")

        # Also support the old env var name for backwards compatibility
        if "RINGRIFT_MIN_GAMES_FOR_TRAINING" in os.environ:
            self.training.trigger_threshold_games = int(os.environ["RINGRIFT_MIN_GAMES_FOR_TRAINING"])
            logger.info(f"Training threshold overridden to {self.training.trigger_threshold_games} (via legacy env var)")

        # Elo database override
        if "RINGRIFT_ELO_DB" in os.environ:
            self.elo_db = os.environ["RINGRIFT_ELO_DB"]
            logger.info(f"Elo DB overridden to {self.elo_db}")


def get_config(config_path: str | Path | None = None, force_reload: bool = False) -> UnifiedConfig:
    """Get the unified configuration singleton.

    Args:
        config_path: Optional path to config file. Defaults to config/unified_loop.yaml
        force_reload: Force reload from file even if already loaded

    Returns:
        UnifiedConfig instance (singleton)
    """
    global _config_instance

    if _config_instance is not None and not force_reload:
        return _config_instance

    # Determine config path
    if config_path is None:
        config_path = os.environ.get("RINGRIFT_CONFIG_PATH", DEFAULT_CONFIG_PATH)

    # Make path absolute relative to ai-service root
    config_path = Path(config_path)
    if not config_path.is_absolute():
        ai_service_root = Path(__file__).parent.parent.parent
        config_path = ai_service_root / config_path

    # Load config
    _config_instance = UnifiedConfig.from_yaml(config_path)
    _config_instance.apply_env_overrides()

    # Validate configuration
    validation_errors = _config_instance.validate()
    if validation_errors:
        for error in validation_errors:
            logger.warning(f"Config validation: {error}")

    logger.info(f"Loaded unified config from {config_path}")
    logger.info(f"  Training threshold: {_config_instance.training.trigger_threshold_games}")
    logger.info(f"  Elo DB: {_config_instance.elo_db}")

    return _config_instance


def get_training_threshold() -> int:
    """Convenience function to get the training threshold.

    Use this instead of hardcoding values like MIN_NEW_GAMES_FOR_TRAINING.
    """
    return get_config().training.trigger_threshold_games


def get_elo_db_path() -> Path:
    """Convenience function to get the Elo database path."""
    return get_config().get_elo_db_path()


def get_min_elo_improvement() -> float:
    """Get the minimum Elo improvement required for promotion.

    CANONICAL SOURCE: Use this instead of hardcoding values like 25.0.
    Other modules (model_lifecycle, unified_loop_extensions) should
    use this function to ensure consistency.
    """
    return float(get_config().promotion.elo_threshold)


def get_target_selfplay_rate() -> int:
    """Get the target selfplay games per hour.

    CANONICAL SOURCE: Use this instead of hardcoding values like 1000.
    Other modules (p2p_integration, unified_ai_loop) should
    use this function to ensure consistency.
    """
    return get_config().cluster.target_selfplay_games_per_hour


# =========================================================================
# Data Collection Config Helpers
# =========================================================================

def get_data_collector_poll_interval() -> int:
    """Get the data collector poll interval in seconds.

    CANONICAL SOURCE: Use this instead of hardcoding values like 60.
    Scripts (streaming_data_collector) should use this function.
    Default: 60 seconds
    """
    cfg = get_config()
    return getattr(cfg, 'data_collector_poll_interval', 60)


def get_data_collector_max_failures() -> int:
    """Get the max consecutive failures before disabling a host.

    CANONICAL SOURCE: Use this instead of hardcoding values like 5.
    Scripts (streaming_data_collector) should use this function.
    Default: 5
    """
    cfg = get_config()
    return getattr(cfg, 'data_collector_max_failures', 5)


# =========================================================================
# Tournament Config Helpers
# =========================================================================

def get_shadow_tournament_interval() -> int:
    """Get the shadow tournament interval in seconds.

    CANONICAL SOURCE: Use this instead of hardcoding values like 900 (15 min).
    Scripts (shadow_tournament_service) should use this function.
    Default: 900 seconds (15 minutes)
    """
    cfg = get_config()
    return getattr(cfg, 'shadow_tournament_interval', 900)


def get_full_tournament_interval() -> int:
    """Get the full tournament interval in seconds.

    CANONICAL SOURCE: Use this instead of hardcoding values like 3600 (1 hour).
    Scripts (shadow_tournament_service) should use this function.
    Default: 3600 seconds (1 hour)
    """
    cfg = get_config()
    return getattr(cfg, 'full_tournament_interval', 3600)


def get_tournament_games_per_matchup() -> int:
    """Get the number of games per matchup in tournaments.

    CANONICAL SOURCE: Use this instead of hardcoding values like 20 or 50.
    Scripts (improvement_cycle_manager, shadow_tournament_service) should use this.
    Default: 20
    """
    cfg = get_config()
    return getattr(cfg, 'tournament_games_per_matchup', 20)


def get_regression_elo_threshold() -> float:
    """Get the Elo threshold for detecting regressions.

    CANONICAL SOURCE: Use this instead of hardcoding values like 30.0.
    Scripts (shadow_tournament_service) should use this function.
    Default: 30.0 Elo points
    """
    cfg = get_config()
    return getattr(cfg, 'regression_elo_threshold', 30.0)


def get_promotion_elo_threshold() -> float:
    """Get the Elo gain required for automatic model promotion.

    CANONICAL SOURCE: Use this instead of hardcoding values like 20.0.
    Scripts (model_promotion_manager) should use this function.
    Default: 20.0 Elo points
    """
    cfg = get_config()
    return getattr(cfg, 'promotion_elo_threshold', 20.0)


def get_promotion_min_games() -> int:
    """Get the minimum games required before a model can be promoted.

    CANONICAL SOURCE: Use this instead of hardcoding values like 50.
    Scripts (model_promotion_manager) should use this function.
    Default: 50 games
    """
    cfg = get_config()
    return getattr(cfg, 'promotion_min_games', 50)


def get_promotion_check_interval() -> int:
    """Get the interval between auto-promotion checks in seconds.

    CANONICAL SOURCE: Use this instead of hardcoding values like 300.
    Scripts (model_promotion_manager) should use this function.
    Default: 300 seconds (5 minutes)
    """
    cfg = get_config()
    return getattr(cfg, 'promotion_check_interval', 300)


def get_rollback_elo_threshold() -> float:
    """Get the Elo drop threshold for triggering automatic rollback.

    CANONICAL SOURCE: Use this instead of hardcoding values like 50.0.
    Scripts (model_promotion_manager) should use this function.
    Default: 50.0 Elo points
    """
    cfg = get_config()
    return getattr(cfg, 'rollback_elo_threshold', 50.0)


def get_rollback_min_games() -> int:
    """Get the minimum games before rollback can be considered.

    CANONICAL SOURCE: Use this instead of hardcoding values like 20.
    Scripts (model_promotion_manager) should use this function.
    Default: 20 games
    """
    cfg = get_config()
    return getattr(cfg, 'rollback_min_games', 20)


# Constants for backwards compatibility
# These should be deprecated in favor of get_config()
def _get_legacy_threshold() -> int:
    """Legacy accessor - prefer get_config().training.trigger_threshold_games"""
    return get_config().training.trigger_threshold_games


# Export commonly used constants (computed at import time for backwards compat)
# NOTE: These are evaluated once at import - use get_training_threshold() for dynamic access
ALL_BOARD_CONFIGS: list[tuple[str, int]] = [
    ("square8", 2), ("square8", 3), ("square8", 4),
    ("square19", 2), ("square19", 3), ("square19", 4),
    ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
]


# =============================================================================
# Integrated Enhancements Factory
# =============================================================================

def create_training_manager(
    model: Any | None = None,
    board_type: str = "square8",
    config: UnifiedConfig | None = None,
) -> Any | None:
    """Create an IntegratedTrainingManager from UnifiedConfig.

    Factory function that creates a training enhancement manager using
    the canonical IntegratedEnhancementsConfig from unified config.

    Args:
        model: PyTorch model (optional, can be set later)
        board_type: Board type for augmentation
        config: Optional UnifiedConfig (defaults to get_config())

    Returns:
        IntegratedTrainingManager instance or None if disabled/unavailable

    Example:
        from app.config.unified_config import create_training_manager

        manager = create_training_manager(model=my_model, board_type="square8")
        if manager:
            manager.initialize_all()
            batch_size = manager.get_batch_size()
    """
    if config is None:
        config = get_config()

    if not config.enhancements.enabled:
        return None

    try:
        from app.training.integrated_enhancements import (
            IntegratedEnhancementsConfig as EnhancementsConfig,
            IntegratedTrainingManager,
        )

        # Map UnifiedConfig.enhancements to IntegratedEnhancementsConfig
        enh = config.enhancements
        manager_config = EnhancementsConfig(
            auxiliary_tasks_enabled=enh.auxiliary_tasks_enabled,
            aux_game_length_weight=enh.aux_game_length_weight,
            aux_piece_count_weight=enh.aux_piece_count_weight,
            aux_outcome_weight=enh.aux_outcome_weight,
            gradient_surgery_enabled=enh.gradient_surgery_enabled,
            gradient_surgery_method=enh.gradient_surgery_method,
            batch_scheduling_enabled=enh.batch_scheduling_enabled,
            batch_initial_size=enh.batch_initial_size,
            batch_final_size=enh.batch_final_size,
            batch_warmup_steps=enh.batch_warmup_steps,
            batch_rampup_steps=enh.batch_rampup_steps,
            batch_schedule_type=enh.batch_schedule_type,
            background_eval_enabled=enh.background_eval_enabled,
            eval_interval_steps=enh.eval_interval_steps,
            eval_elo_checkpoint_threshold=enh.eval_elo_checkpoint_threshold,
            elo_weighting_enabled=enh.elo_weighting_enabled,
            elo_base_rating=enh.elo_base_rating,
            elo_weight_scale=enh.elo_weight_scale,
            elo_min_weight=enh.elo_min_weight,
            elo_max_weight=enh.elo_max_weight,
            curriculum_enabled=enh.curriculum_enabled,
            curriculum_auto_advance=enh.curriculum_auto_advance,
            augmentation_enabled=enh.augmentation_enabled,
            augmentation_mode=enh.augmentation_mode,
            reanalysis_enabled=enh.reanalysis_enabled,
            reanalysis_blend_ratio=enh.reanalysis_blend_ratio,
        )

        return IntegratedTrainingManager(manager_config, model, board_type)

    except ImportError as e:
        logger.warning(f"Failed to import integrated enhancements: {e}")
        return None


# =============================================================================
# Storage Path Helpers
# =============================================================================

def detect_storage_provider() -> str:
    """Detect the current storage provider based on environment.

    Returns:
        Provider name: "lambda", "vast", "mac", or "default"
    """
    import fnmatch
    import platform
    import socket

    config = get_config()
    hostname = socket.gethostname()

    # Check detection rules in priority order
    for provider, rules in config.storage.provider_detection.items():
        # Check OS type
        if rules.os_type and platform.system() == rules.os_type:
            return provider

        # Check path existence
        if rules.check_path and Path(rules.check_path).exists():
            return provider

        # Check hostname patterns
        for pattern in rules.hostname_patterns:
            if fnmatch.fnmatch(hostname, pattern):
                return provider

    # Fallback detection without explicit rules
    # Lambda: NFS mount
    if Path("/lambda/nfs").exists():
        return "lambda"

    # Vast: /workspace directory
    if Path("/workspace").exists():
        return "vast"

    # Mac: Darwin OS
    if platform.system() == "Darwin":
        return "mac"

    return "default"


def get_storage_paths(provider: str | None = None) -> StoragePathsConfig:
    """Get storage paths for the current or specified provider.

    Args:
        provider: Optional provider name. Auto-detected if not specified.

    Returns:
        StoragePathsConfig with paths for the provider.

    Example:
        from app.config.unified_config import get_storage_paths

        paths = get_storage_paths()
        selfplay_dir = paths.selfplay_games
        model_dir = paths.model_checkpoints

        # For specific provider
        lambda_paths = get_storage_paths("lambda")
    """
    if provider is None:
        provider = detect_storage_provider()

    config = get_config()

    if provider == "lambda":
        return config.storage.lambda_
    elif provider == "vast":
        return config.storage.vast
    elif provider == "mac":
        return config.storage.mac
    else:
        return config.storage.default


def get_selfplay_dir(provider: str | None = None) -> Path:
    """Get the selfplay games directory for the current provider."""
    return Path(get_storage_paths(provider).selfplay_games)


def get_model_checkpoint_dir(provider: str | None = None) -> Path:
    """Get the model checkpoint directory for the current provider."""
    return Path(get_storage_paths(provider).model_checkpoints)


def get_training_data_dir(provider: str | None = None) -> Path:
    """Get the training data directory for the current provider."""
    return Path(get_storage_paths(provider).training_data)


def get_nfs_base(provider: str | None = None) -> Path | None:
    """Get the NFS base path if available for the provider."""
    paths = get_storage_paths(provider)
    if paths.nfs_base:
        return Path(paths.nfs_base)
    return None


def should_use_nfs_sync(provider: str | None = None) -> bool:
    """Check if NFS should be used for data sync (no rsync needed)."""
    return get_storage_paths(provider).use_nfs_for_sync


def should_skip_rsync_to_node(node_id: str) -> bool:
    """Check if rsync should be skipped for a node (has NFS access).

    Args:
        node_id: The node identifier

    Returns:
        True if node has NFS access and rsync should be skipped
    """
    # Lambda nodes with NFS don't need rsync between each other
    if node_id.startswith("lambda-"):
        return get_storage_paths("lambda").skip_rsync_to_nfs_nodes
    return False


def ensure_storage_dirs(provider: str | None = None) -> None:
    """Ensure all storage directories exist for the provider.

    Creates directories if they don't exist.
    """
    paths = get_storage_paths(provider)

    for path_str in [
        paths.selfplay_games,
        paths.model_checkpoints,
        paths.training_data,
        paths.sync_staging,
        paths.local_scratch,
    ]:
        path = Path(path_str)
        if not path.is_absolute():
            # Make relative paths absolute from ai-service root
            path = AI_SERVICE_ROOT / path

        ensure_dir(path)

    # Also create parent dir for elo database
    elo_path = Path(paths.elo_database)
    if not elo_path.is_absolute():
        elo_path = AI_SERVICE_ROOT / elo_path
    ensure_parent_dir(elo_path)

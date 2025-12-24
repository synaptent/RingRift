"""Unified AI Loop Configuration Classes.

This module contains all configuration dataclasses and event types
for the unified AI improvement loop.
Extracted from unified_ai_loop.py for better modularity.

CONSOLIDATION NOTE (2025-12-18):
================================
Core configuration classes have been migrated to the canonical location:
    app.config.unified_config

Event types consolidated to canonical location:
    app.distributed.data_events

The following classes are now re-exported from canonical:
- PBTConfig, NASConfig, PERConfig, FeedbackConfig, P2PClusterConfig, ModelPruningConfig
- DataEventType, DataEvent (from app.distributed.data_events)

This module retains:
- Extended TrainingConfig with advanced options
- UnifiedLoopConfig (script-specific root config)
- Runtime state classes (ConfigState, FeedbackState, HostState)

For new integrations, prefer importing from app.config.unified_config.
See docs/CONSOLIDATION_ROADMAP.md for consolidation status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from app.training.integrated_enhancements import IntegratedTrainingManager

import yaml

# Import canonical threshold constants
try:
    from app.config.thresholds import (
        ELO_DROP_ROLLBACK,
        INITIAL_ELO_RATING,
    )
except ImportError:
    INITIAL_ELO_RATING = 1500.0
    ELO_DROP_ROLLBACK = 50.0

# Re-export migrated classes from canonical location for backward compatibility
try:
    from app.config.unified_config import (
        FeedbackConfig,
        IntegratedEnhancementsConfig,  # Canonical location as of 2025-12-17
        ModelPruningConfig,
        NASConfig,
        P2PClusterConfig,
        PBTConfig,
        PERConfig,
    )
    _HAS_CANONICAL_ENHANCEMENTS = True
except ImportError:
    # Fallback: define locally if canonical import fails (shouldn't happen in normal use)
    _HAS_CANONICAL_ENHANCEMENTS = False

# Import DataEventType and DataEvent from canonical location
from app.coordination.event_router import DataEvent, DataEventType

# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class DataIngestionConfig:
    """Configuration for streaming data collection.

    When use_external_sync=True, the internal data collector is disabled and
    the loop expects an external unified_data_sync.py service to handle data sync.
    This allows using advanced features like P2P fallback, WAL, and content dedup.
    """
    poll_interval_seconds: int = 60
    ephemeral_poll_interval_seconds: int = 15  # Aggressive sync for RAM disk hosts
    sync_method: str = "incremental"  # "incremental" or "full"
    deduplication: bool = True
    min_games_per_sync: int = 10
    remote_db_pattern: str = "data/games/*.db"
    # External sync: when True, skip internal data collection and rely on external
    # unified_data_sync.py service (provides P2P fallback, WAL, content dedup)
    use_external_sync: bool = False
    # Hardening options
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

    NOTE: Defaults match app/config/unified_config.py (single source of truth)
    """
    trigger_threshold_games: int = 500  # Canonical: 500 (was 1000)
    min_interval_seconds: int = 1200  # Canonical: 20 min (was 30)
    max_concurrent_jobs: int = 1
    prefer_gpu_hosts: bool = True
    # Simplified 3-signal trigger system (2024-12)
    use_simplified_triggers: bool = True  # Use 3-signal system instead of 8+ signals
    staleness_hours: float = 6.0  # Hours before config is "stale"
    min_win_rate_threshold: float = 0.45  # Below this triggers urgent training
    bootstrap_threshold: int = 50  # Low threshold for configs with 0 models
    nnue_training_script: str = "scripts/train_nnue.py"
    nn_training_script: str = "scripts/run_nn_training_baseline.py"
    export_script: str = "scripts/export_replay_dataset.py"
    # Encoder version for hex boards: "v3" uses HexStateEncoderV3 (16 channels)
    hex_encoder_version: str = "v3"
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
    selfplay_engine: str = "gumbel"  # "descent", "mcts", or "gumbel" - GPU-accelerated Gumbel MCTS
    selfplay_num_workers: int | None = None  # Default: CPU count - 1
    selfplay_games_per_batch: int = 20  # Games per local selfplay batch
    # Temperature scheduling for exploration/exploitation tradeoff
    selfplay_temperature: float = 1.0  # Base move selection temperature
    selfplay_use_temperature_decay: bool = True  # Enable temperature decay per game
    selfplay_move_temp_threshold: int = 30  # Use higher temp for first N moves
    selfplay_opening_temperature: float = 1.5  # Temperature for opening moves
    # Gumbel-MCTS specific parameters
    gumbel_simulations: int = 800  # Simulations per move for Gumbel-MCTS (AlphaZero standard)
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
            ('swa_start_fraction', self.swa_start_fraction),
            ('ema_decay', self.ema_decay),
            ('distill_alpha', self.distill_alpha),
            ('validation_sample_rate', self.validation_sample_rate),
            ('parity_failure_threshold', self.parity_failure_threshold),
            ('min_data_quality_for_training', self.min_data_quality_for_training),
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
            errors.append(f"training_retry_backoff_multiplier must be >= 1.0, got {self.training_retry_backoff_multiplier}")

        # Validate epoch settings
        if self.warmup_epochs < 0:
            errors.append(f"warmup_epochs must be >= 0, got {self.warmup_epochs}")

        # Raise all errors at once
        if errors:
            raise ValueError("TrainingConfig validation failed:\n  " + "\n  ".join(errors))


@dataclass
class EvaluationConfig:
    """Configuration for continuous evaluation.

    .. note::
        This is a script-local extension of the canonical EvaluationConfig.
        Base defaults match ``app/config/unified_config.py`` (single source of truth).
        Script-specific additions: adaptive_interval_* settings.

    OPTIMIZED: Reduced default from 900s to 300s since parallel execution is 3x faster.
    """
    shadow_interval_seconds: int = 300  # 5 minutes (reduced from 15)
    shadow_games_per_config: int = 15  # Canonical: 15 (was 10)
    full_tournament_interval_seconds: int = 3600  # 1 hour
    full_tournament_games: int = 50
    baseline_models: list[str] = field(default_factory=lambda: ["random", "heuristic", "mcts_100", "mcts_500"])
    # Adaptive interval settings - go faster when cluster is healthy
    adaptive_interval_enabled: bool = True
    adaptive_interval_min_seconds: int = 120  # Can go as low as 2 min
    adaptive_interval_max_seconds: int = 600  # Cap at 10 min during high load


@dataclass
class PromotionConfig:
    """Configuration for automatic model promotion.

    NOTE: Defaults match app/config/unified_config.py (single source of truth)
    """
    auto_promote: bool = True
    elo_threshold: int = 25  # Canonical: 25 (was 20)
    min_games: int = 50
    significance_level: float = 0.05
    sync_to_cluster: bool = True
    hosts_config_path: str | None = None

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
    """Configuration for adaptive curriculum.

    NOTE: Defaults match app/config/unified_config.py (single source of truth)
    """
    adaptive: bool = True
    rebalance_interval_seconds: int = 3600  # 1 hour
    max_weight_multiplier: float = 1.5  # Canonical: 1.5 (was 2.0)
    min_weight_multiplier: float = 0.7  # Canonical: 0.7 (was 0.5)


# =============================================================================
# MIGRATED CLASSES (now imported from app.config.unified_config)
# =============================================================================
# The following classes were removed and are now re-exported from canonical:
# - PBTConfig
# - NASConfig
# - PERConfig
# - FeedbackConfig
# - P2PClusterConfig
# - ModelPruningConfig
# See imports at top of file for backward-compatible re-exports.
# =============================================================================


# IntegratedEnhancementsConfig is now imported from canonical location (app.config.unified_config)
# The definition below is ONLY used as fallback if canonical import fails (standalone execution)
if not _HAS_CANONICAL_ENHANCEMENTS:
    @dataclass
    class IntegratedEnhancementsConfig:
        """Fallback: Configuration for integrated training enhancements.

        NOTE: This is a fallback definition. The canonical location is:
            app.config.unified_config.IntegratedEnhancementsConfig

        Only used when running standalone without app package.
        """
        enabled: bool = True
        auxiliary_tasks_enabled: bool = False
        aux_game_length_weight: float = 0.1
        aux_piece_count_weight: float = 0.1
        aux_outcome_weight: float = 0.05
        gradient_surgery_enabled: bool = False
        gradient_surgery_method: str = "pcgrad"
        gradient_conflict_threshold: float = 0.0
        batch_scheduling_enabled: bool = False
        batch_initial_size: int = 64
        batch_final_size: int = 512
        batch_warmup_steps: int = 1000
        batch_rampup_steps: int = 10000
        batch_schedule_type: str = "linear"
        background_eval_enabled: bool = False
        eval_interval_steps: int = 1000
        eval_games_per_check: int = 20
        eval_elo_checkpoint_threshold: float = 10.0
        eval_elo_drop_threshold: float = 50.0  # ELO_DROP_ROLLBACK default
        eval_auto_checkpoint: bool = True
        eval_checkpoint_dir: str = "data/eval_checkpoints"
        elo_weighting_enabled: bool = True
        elo_base_rating: float = 1500.0  # INITIAL_ELO_RATING default
        elo_weight_scale: float = 400.0
        elo_min_weight: float = 0.5
        elo_max_weight: float = 2.0
        curriculum_enabled: bool = True
        curriculum_auto_advance: bool = True
        curriculum_checkpoint_path: str = "data/curriculum_state.json"
        augmentation_enabled: bool = True
        augmentation_mode: str = "all"
        augmentation_probability: float = 1.0
        reanalysis_enabled: bool = False
        reanalysis_blend_ratio: float = 0.5
        reanalysis_interval_steps: int = 5000
        reanalysis_batch_size: int = 1000


@dataclass
class UnifiedLoopConfig:
    """Complete configuration for the unified AI loop."""
    data_ingestion: DataIngestionConfig = field(default_factory=DataIngestionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    pbt: PBTConfig = field(default_factory=PBTConfig)
    nas: NASConfig = field(default_factory=NASConfig)
    per: PERConfig = field(default_factory=PERConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    p2p: P2PClusterConfig = field(default_factory=P2PClusterConfig)
    model_pruning: ModelPruningConfig = field(default_factory=ModelPruningConfig)
    enhancements: IntegratedEnhancementsConfig = field(default_factory=IntegratedEnhancementsConfig)

    # Host configuration
    hosts_config_path: str = "config/distributed_hosts.yaml"

    # Database paths
    elo_db: str = "data/unified_elo.db"  # Canonical Elo database
    data_manifest_db: str = "data/data_manifest.db"

    # Logging
    log_dir: str = "logs/unified_loop"
    verbose: bool = False

    # Metrics
    metrics_port: int = 9091  # Note: 9090 is reserved for Prometheus itself
    metrics_enabled: bool = True

    # Operation modes
    dry_run: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> UnifiedLoopConfig:
        """Load configuration from YAML file."""
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        config = cls()

        if "data_ingestion" in data:
            for k, v in data["data_ingestion"].items():
                if hasattr(config.data_ingestion, k):
                    setattr(config.data_ingestion, k, v)

        if "training" in data:
            for k, v in data["training"].items():
                if hasattr(config.training, k):
                    setattr(config.training, k, v)

        if "evaluation" in data:
            for k, v in data["evaluation"].items():
                if hasattr(config.evaluation, k):
                    setattr(config.evaluation, k, v)

        if "promotion" in data:
            for k, v in data["promotion"].items():
                if hasattr(config.promotion, k):
                    setattr(config.promotion, k, v)

        if "curriculum" in data:
            for k, v in data["curriculum"].items():
                if hasattr(config.curriculum, k):
                    setattr(config.curriculum, k, v)

        if "pbt" in data:
            for k, v in data["pbt"].items():
                if hasattr(config.pbt, k):
                    setattr(config.pbt, k, v)

        if "nas" in data:
            for k, v in data["nas"].items():
                if hasattr(config.nas, k):
                    setattr(config.nas, k, v)

        if "per" in data:
            for k, v in data["per"].items():
                if hasattr(config.per, k):
                    setattr(config.per, k, v)

        if "feedback" in data:
            for k, v in data["feedback"].items():
                if hasattr(config.feedback, k):
                    setattr(config.feedback, k, v)

        if "p2p" in data:
            for k, v in data["p2p"].items():
                if hasattr(config.p2p, k):
                    setattr(config.p2p, k, v)

        if "model_pruning" in data:
            for k, v in data["model_pruning"].items():
                if hasattr(config.model_pruning, k):
                    setattr(config.model_pruning, k, v)

        if "enhancements" in data:
            for k, v in data["enhancements"].items():
                if hasattr(config.enhancements, k):
                    setattr(config.enhancements, k, v)

        for key in ["hosts_config_path", "elo_db", "data_manifest_db", "log_dir",
                    "verbose", "metrics_port", "metrics_enabled", "dry_run"]:
            if key in data:
                setattr(config, key, data[key])

        if config.promotion.hosts_config_path is None:
            config.promotion.hosts_config_path = config.hosts_config_path

        return config


# =============================================================================
# Event System - Re-exported from canonical module (2025-12-18)
# =============================================================================
# DataEventType and DataEvent are now imported from the canonical module
# app.distributed.data_events for consolidation and consistency.



# =============================================================================
# State Management
# =============================================================================

@dataclass
class HostState:
    """State for a remote host."""
    name: str
    ssh_host: str
    ssh_user: str = "ubuntu"
    ssh_port: int = 22
    last_sync_time: float = 0.0
    last_game_count: int = 0
    consecutive_failures: int = 0
    enabled: bool = True


@dataclass
class FeedbackState:
    """Consolidated feedback state for training decisions (2025-12).

    Groups all feedback signals into a single structure:
    - Curriculum: weights and staleness
    - Quality: parity failures, data health
    - Elo: current rating and trends
    - Win rate: performance metrics
    """
    # Curriculum feedback (0.5-2.0, weight > 1 = needs more training)
    curriculum_weight: float = 1.0
    curriculum_last_update: float = 0.0

    # Data quality feedback
    parity_failure_rate: float = 0.0  # Rolling average of parity failures (0-1)
    parity_checks_total: int = 0  # Total parity checks performed
    data_quality_score: float = 1.0  # Composite quality metric (0-1)

    # Elo feedback (uses INITIAL_ELO_RATING from app.config.thresholds)
    elo_current: float = INITIAL_ELO_RATING
    elo_trend: float = 0.0  # Positive = improving, negative = declining
    elo_peak: float = INITIAL_ELO_RATING  # Historical peak Elo
    elo_plateau_count: int = 0  # Consecutive evaluations without gain

    # Win rate feedback
    win_rate: float = 0.5  # Latest win rate (0-1)
    win_rate_trend: float = 0.0  # Change over recent evals
    consecutive_high_win_rate: int = 0  # Streak above 70%
    consecutive_low_win_rate: int = 0  # Streak below 50%

    # Training urgency metrics
    urgency_score: float = 0.0  # Composite urgency (0-1, higher = more urgent)
    last_urgency_update: float = 0.0

    def update_parity(self, passed: bool, alpha: float = 0.1) -> None:
        """Update rolling parity failure rate."""
        result = 0.0 if passed else 1.0
        self.parity_failure_rate = alpha * result + (1 - alpha) * self.parity_failure_rate
        self.parity_checks_total += 1

    def update_elo(self, new_elo: float, plateau_threshold: float = 15.0) -> None:
        """Update Elo with trend and plateau detection."""
        old_elo = self.elo_current
        self.elo_trend = new_elo - old_elo
        self.elo_current = new_elo
        self.elo_peak = max(self.elo_peak, new_elo)

        # Plateau detection
        if abs(self.elo_trend) < plateau_threshold:
            self.elo_plateau_count += 1
        else:
            self.elo_plateau_count = 0

    def update_win_rate(self, new_win_rate: float) -> None:
        """Update win rate with trend tracking."""
        old_win_rate = self.win_rate
        self.win_rate_trend = new_win_rate - old_win_rate
        self.win_rate = new_win_rate

        # Track consecutive high/low streaks
        if new_win_rate > 0.7:
            self.consecutive_high_win_rate += 1
            self.consecutive_low_win_rate = 0
        elif new_win_rate < 0.5:
            self.consecutive_low_win_rate += 1
            self.consecutive_high_win_rate = 0
        else:
            self.consecutive_high_win_rate = 0
            self.consecutive_low_win_rate = 0

    def compute_urgency(self) -> float:
        """Compute composite urgency score for training prioritization.

        Returns value 0-1 where higher = more urgent training need.
        """
        import time
        urgency = 0.0

        # Factor 1: Low win rate increases urgency
        if self.win_rate < 0.5:
            urgency += (0.5 - self.win_rate) * 0.4  # Up to 0.2 contribution

        # Factor 2: Declining win rate increases urgency
        if self.win_rate_trend < 0:
            urgency += min(0.2, abs(self.win_rate_trend) * 2)

        # Factor 3: Elo plateau increases urgency (stagnation)
        plateau_factor = min(0.2, self.elo_plateau_count * 0.04)
        urgency += plateau_factor

        # Factor 4: High curriculum weight (needs training)
        if self.curriculum_weight > 1.0:
            urgency += min(0.2, (self.curriculum_weight - 1.0) * 0.2)

        # Factor 5: Good data quality is a prerequisite (reduces urgency if bad)
        if self.parity_failure_rate > 0.1:
            urgency *= 0.5  # De-prioritize if data quality is poor

        self.urgency_score = min(1.0, urgency)
        self.last_urgency_update = time.time()
        return self.urgency_score

    def compute_data_quality(
        self,
        sample_diversity: float = 1.0,
        avg_game_length: float = 50.0,
        min_game_length: float = 10.0,
        max_game_length: float = 200.0,
    ) -> float:
        """Compute composite data quality score (0-1).

        Factors:
        - Parity pass rate (inverse of failure rate)
        - Sample diversity (0-1, higher = more diverse positions)
        - Game length normalization (penalize too short or too long)

        Args:
            sample_diversity: Diversity score from data collection (0-1)
            avg_game_length: Average game length in moves
            min_game_length: Expected minimum reasonable length
            max_game_length: Expected maximum reasonable length

        Returns:
            Composite quality score (0-1)
        """
        quality = 0.0

        # Factor 1: Parity pass rate (40% weight)
        parity_score = 1.0 - self.parity_failure_rate
        quality += parity_score * 0.4

        # Factor 2: Sample diversity (30% weight)
        quality += max(0, min(1.0, sample_diversity)) * 0.3

        # Factor 3: Game length normalization (30% weight)
        # Penalize games that are too short (likely errors) or too long (stalemates)
        if avg_game_length < min_game_length:
            length_score = avg_game_length / min_game_length
        elif avg_game_length > max_game_length:
            length_score = max(0.5, max_game_length / avg_game_length)
        else:
            # Optimal range
            length_score = 1.0
        quality += length_score * 0.3

        self.data_quality_score = min(1.0, max(0.0, quality))
        return self.data_quality_score

    def is_data_quality_acceptable(self, threshold: float = 0.7) -> bool:
        """Check if data quality meets minimum threshold.

        Args:
            threshold: Minimum acceptable quality score (0-1)

        Returns:
            True if data quality is acceptable
        """
        return self.data_quality_score >= threshold

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization/logging."""
        return {
            'curriculum_weight': self.curriculum_weight,
            'parity_failure_rate': self.parity_failure_rate,
            'data_quality_score': self.data_quality_score,
            'elo_current': self.elo_current,
            'elo_trend': self.elo_trend,
            'elo_plateau_count': self.elo_plateau_count,
            'win_rate': self.win_rate,
            'win_rate_trend': self.win_rate_trend,
            'urgency_score': self.urgency_score,
        }


@dataclass
class ConfigState:
    """State for a board/player configuration.

    Note: Uses INITIAL_ELO_RATING from app.config.thresholds as default.
    """
    board_type: str
    num_players: int
    game_count: int = 0
    games_since_training: int = 0
    last_training_time: float = 0.0
    last_evaluation_time: float = 0.0
    last_promotion_time: float = 0.0  # For dynamic threshold calculation
    current_elo: float = INITIAL_ELO_RATING
    elo_trend: float = 0.0  # Positive = improving
    training_weight: float = 1.0
    # Win rate tracking for training feedback (Phase 2.4)
    win_rate: float = 0.5  # Latest win rate from evaluations (0.5 = default/unknown)
    win_rate_trend: float = 0.0  # Change in win rate (positive = improving)
    consecutive_high_win_rate: int = 0  # Count of evals with win_rate > 0.7
    # Consolidated feedback state (2025-12)
    feedback: FeedbackState = field(default_factory=FeedbackState)


# =============================================================================
# Integrated Enhancements Factory
# =============================================================================

def create_integrated_manager_from_config(
    training_config: TrainingConfig,
    model: Any | None = None,
    board_type: str = "square8",
) -> IntegratedTrainingManager | None:
    """Create an IntegratedTrainingManager from TrainingConfig.

    Args:
        training_config: TrainingConfig with enhancement settings
        model: PyTorch model (optional, can be set later)
        board_type: Board type for augmentation

    Returns:
        IntegratedTrainingManager or None if disabled
    """
    if not training_config.use_integrated_enhancements:
        return None

    try:
        from app.training.integrated_enhancements import (
            IntegratedEnhancementsConfig,
            IntegratedTrainingManager,
        )

        # Map TrainingConfig to IntegratedEnhancementsConfig
        enhancement_config = IntegratedEnhancementsConfig(
            # Auxiliary Tasks
            auxiliary_tasks_enabled=training_config.auxiliary_tasks_enabled,
            aux_game_length_weight=training_config.aux_game_length_weight,
            aux_piece_count_weight=training_config.aux_piece_count_weight,
            aux_outcome_weight=training_config.aux_outcome_weight,
            # Gradient Surgery
            gradient_surgery_enabled=training_config.gradient_surgery_enabled,
            gradient_surgery_method=training_config.gradient_surgery_method,
            # Batch Scheduling
            batch_scheduling_enabled=training_config.batch_scheduling_enabled,
            batch_initial_size=training_config.batch_initial_size,
            batch_final_size=training_config.batch_final_size,
            batch_schedule_type=training_config.batch_schedule_type,
            # Background Evaluation
            background_eval_enabled=training_config.background_eval_enabled,
            eval_interval_steps=training_config.eval_interval_steps,
            eval_elo_checkpoint_threshold=training_config.eval_elo_checkpoint_threshold,
            # ELO Weighting
            elo_weighting_enabled=training_config.elo_weighting_enabled,
            elo_base_rating=training_config.elo_base_rating,
            elo_weight_scale=training_config.elo_weight_scale,
            elo_min_weight=training_config.elo_min_weight,
            elo_max_weight=training_config.elo_max_weight,
            # Curriculum Learning
            curriculum_enabled=training_config.curriculum_enabled,
            curriculum_auto_advance=training_config.curriculum_auto_advance,
            # Augmentation
            augmentation_enabled=training_config.augmentation_enabled,
            augmentation_mode=training_config.augmentation_mode,
            # Reanalysis
            reanalysis_enabled=training_config.reanalysis_enabled,
            reanalysis_blend_ratio=training_config.reanalysis_blend_ratio,
        )

        return IntegratedTrainingManager(enhancement_config, model, board_type)

    except ImportError as e:
        import logging
        logging.getLogger(__name__).warning(
            f"[Config] Failed to import integrated enhancements: {e}"
        )
        return None


# =============================================================================
# Integration with app.config.unified_config
# =============================================================================

def sync_with_unified_config(loop_config: UnifiedLoopConfig) -> UnifiedLoopConfig:
    """Sync defaults from app.config.unified_config to keep values aligned.

    This ensures that the unified loop uses the same canonical values as the
    rest of the codebase. Call this after loading a UnifiedLoopConfig.

    Args:
        loop_config: The config to sync

    Returns:
        The same config with any unset values populated from unified_config
    """
    try:
        from app.config.unified_config import get_config as get_unified_config

        unified = get_unified_config()

        # Sync training thresholds if using defaults
        if loop_config.training.trigger_threshold_games == 500:  # Default
            loop_config.training.trigger_threshold_games = unified.training.trigger_threshold_games

        # Sync promotion thresholds if using defaults
        if loop_config.promotion.elo_threshold == 25:  # Default
            loop_config.promotion.elo_threshold = int(unified.promotion.min_elo_improvement)

        return loop_config

    except ImportError:
        # app.config.unified_config not available, use local defaults
        return loop_config


def get_canonical_training_threshold() -> int:
    """Get the canonical training threshold from unified_config.

    Returns:
        Training threshold from unified_config, or default (500)
    """
    try:
        from app.config.unified_config import get_training_threshold
        return get_training_threshold()
    except ImportError:
        return 500  # Default

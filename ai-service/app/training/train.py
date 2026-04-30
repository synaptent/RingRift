"""Training CLI for RingRift neural networks."""
from __future__ import annotations

# Maintenance note: the supported CLI remains here; non-loop helpers have been
# extracted into dedicated training modules.

import logging
import os
import time
import warnings
from pathlib import Path
from typing import (
    Any,
    cast,
)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Training control thresholds (December 2025)
from app.config.thresholds import (
    EARLY_STOPPING_PATIENCE,
    ELO_PATIENCE,
    MIN_TRAINING_EPOCHS,
    TRAINING_RETRY_SLEEP_SECONDS,
)

# Training metrics extracted to dedicated module (December 2025)
from app.training.train_metrics import (
    ANOMALY_DETECTIONS,
    BATCH_SIZE,
    CALIBRATION_ECE,
    CALIBRATION_MCE,
    CIRCUIT_BREAKER_STATE,
    GRADIENT_CLIP_NORM,
    GRADIENT_NORM,
    HAS_METRICS_COLLECTOR,
    HAS_PROMETHEUS,
    TRAINING_DURATION,
    TRAINING_EPOCHS,
    TRAINING_LOSS,
    TRAINING_SAMPLES,
    MetricsCollector,
)

from app.ai.neural_losses import (
    build_rank_targets,
    detect_masked_policy_output,
    masked_policy_kl,
    stable_policy_log_softmax,
    uses_spatial_policy_head,
    validate_hex_policy_indices,
)
from app.ai.neural_net import (
    HEX8_BOARD_SIZE,
    HEX_BOARD_SIZE,
    MAX_PLAYERS,
    HexNeuralNet_v2,
    HexNeuralNet_v3,
    HexNeuralNet_v3_Flat,  # V3 with flat policy heads (training compatible, Dec 2025)
    HexNeuralNet_v4,
    HexNeuralNet_v5_Heavy,
    RingRiftCNN_v2,
    RingRiftCNN_v3,
    RingRiftCNN_v3_Flat,  # V3 with flat policy heads (training compatible, Dec 2025)
    get_policy_size_for_board,
    multi_player_value_loss,
)
from app.models import BoardType
from app.utils.canonical_naming import normalize_board_type
from app.training.checkpoint_manager import (
    finalize_training_checkpoints,
    initialize_checkpoint_services,
    save_best_model_artifacts,
    save_early_stop_artifacts,
    save_periodic_checkpoint,
)
from app.training.checkpoint_inspection import detect_tier_from_checkpoint
from app.training.config import TrainConfig
from app.training.data_pipeline import (
    prepare_dataset_metadata_context,
    prepare_training_data_pipeline,
    validate_training_data,
)
from app.training.distributed import (
    DistributedMetrics,
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    scale_learning_rate,
    seed_everything,
    setup_distributed,
)
from app.training.gradient_surgery import GradientSurgeon, GradientSurgeryConfig
from app.training.loss_monitor import LossMonitor
from app.training.model_factory import (
    prepare_training_model_artifacts,
)
from app.training.seed_utils import seed_all
# December 2025: Modular training step/epoch logic
# These modules extract core training logic for testability and reuse
from app.training.train_step import (
    BatchData,
    LossComponents,
    TrainStepConfig,
    TrainStepContext,
    TrainStepResult,
    parse_batch,
    run_training_step,
    transfer_batch_to_device,
)
from app.training.train_epoch import (
    EarlyStopState,
    EpochConfig,
    EpochContext,
    EpochResult,
    run_all_epochs,
    run_training_epoch,
    run_validation_loop,
)
from app.training.train_components import (
    resolve_train_config,
)

# February 2026: Extracted modules from train_model() for maintainability
from app.training.train_pre_validation import run_pre_training_validation
from app.training.training_epoch_reporting import (
    handle_epoch_reporting_and_feedback,
)
from app.training.training_entrypoints import train_from_file, train_with_config
from app.training.training_run_support import (
    initialize_training_run_support,
    maybe_run_lr_finder,
)
from app.training.training_runtime_setup import initialize_training_runtime_setup
from app.training.train_setup import (
    FaultToleranceConfig,
    TrainingState,
    setup_fault_tolerance,
)
from app.training.parameter_validation import (
    validate_training_compatibility as _validate_training_compatibility,
)

# Data validation (2025-12) - use unified module
try:
    from app.training.unified_data_validator import (
        DataValidator,
        DataValidatorConfig,
        validate_npz_file,
    )
    HAS_DATA_VALIDATION = True
except ImportError:
    HAS_DATA_VALIDATION = False
    DataValidator = None
    DataValidatorConfig = None
    validate_npz_file = None

# Checksum verification for data integrity (December 2025)
try:
    from app.training.data_quality import verify_npz_checksums
    HAS_CHECKSUM_VERIFICATION = True
except ImportError:
    verify_npz_checksums = None
    HAS_CHECKSUM_VERIFICATION = False

# NPZ structure validation for corruption detection (December 2025)
# Catches issues like rsync --partial creating files with unreasonable dimensions
try:
    from app.coordination.npz_validation import (
        validate_npz_structure,
        NPZValidationResult,
    )
    HAS_NPZ_STRUCTURE_VALIDATION = True
except ImportError:
    validate_npz_structure = None
    NPZValidationResult = None
    HAS_NPZ_STRUCTURE_VALIDATION = False

# December 2025: Extracted validation utilities
try:
    from app.training.train_validation import (
        validate_training_data_freshness,
        validate_training_data_files,
        validate_data_checksums,
        FreshnessResult,
    )
    HAS_TRAIN_VALIDATION = True
except ImportError:
    HAS_TRAIN_VALIDATION = False
    validate_training_data_freshness = None
    validate_training_data_files = None
    validate_data_checksums = None
    FreshnessResult = None

# Hot data buffer for priority experience replay (2024-12)
try:
    from app.training.hot_data_buffer import HotDataBuffer
    HAS_HOT_DATA_BUFFER = True
except ImportError:
    HotDataBuffer = None
    HAS_HOT_DATA_BUFFER = False

# Quality bridge for quality-aware data selection (2025-12)
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from app.training.quality_bridge import (
            QualityBridge,
            get_quality_bridge,
        )
    HAS_QUALITY_BRIDGE = True
except ImportError:
    HAS_QUALITY_BRIDGE = False
    get_quality_bridge = None
    QualityBridge = None

# Integrated enhancements (2024-12)
try:
    from app.training.integrated_enhancements import (
        IntegratedEnhancementsConfig,
        IntegratedTrainingManager,
    )
    HAS_INTEGRATED_ENHANCEMENTS = True
except ImportError:
    IntegratedTrainingManager = None
    IntegratedEnhancementsConfig = None
    HAS_INTEGRATED_ENHANCEMENTS = False

# Circuit breaker for training fault tolerance (2025-12)
try:
    from app.distributed.circuit_breaker import CircuitState, get_training_breaker
    from app.coordination.event_router import get_router
    from app.coordination.event_router import DataEvent, DataEventType
    HAS_CIRCUIT_BREAKER = True
    HAS_EVENT_BUS = True
except ImportError:
    get_training_breaker = None
    CircuitState = None
    get_router = None
    DataEvent = None
    DataEventType = None
    HAS_CIRCUIT_BREAKER = False
    HAS_EVENT_BUS = False

# Event emission for training feedback loops (Phase 21.2 - Dec 2025)
try:
    from app.coordination.event_router import (
        emit_training_loss_anomaly,
        emit_training_loss_trend,
    )
    HAS_TRAINING_EVENTS = True
except ImportError:
    emit_training_loss_anomaly = None
    emit_training_loss_trend = None
    HAS_TRAINING_EVENTS = False

# Epoch event emission for curriculum feedback (December 2025)
try:
    from app.training.event_integration import publish_epoch_completed
    HAS_EPOCH_EVENTS = True
except ImportError:
    publish_epoch_completed = None
    HAS_EPOCH_EVENTS = False

# Regression detection for training quality monitoring (2025-12)
try:
    from app.training.regression_detector import (
        RegressionSeverity,
        get_regression_detector,
    )
    HAS_REGRESSION_DETECTOR = True
except ImportError:
    get_regression_detector = None
    RegressionSeverity = None
    HAS_REGRESSION_DETECTOR = False

# Training data freshness checking (2025-12)
try:
    from app.coordination.training_freshness import (
        check_freshness_sync,
        FreshnessConfig,
        FreshnessResult,
    )
    HAS_FRESHNESS_CHECK = True
except ImportError:
    check_freshness_sync = None
    FreshnessConfig = None
    FreshnessResult = None
    HAS_FRESHNESS_CHECK = False

# Training stale data fallback (December 2025)
# Part of 48-hour autonomous operation plan - allows training with stale data
# after configurable sync failures or timeout
try:
    from app.coordination.stale_fallback import (
        get_training_fallback_controller,
        should_allow_stale_training,
    )
    HAS_STALE_FALLBACK = True
except ImportError:
    get_training_fallback_controller = None
    should_allow_stale_training = None
    HAS_STALE_FALLBACK = False

# Training anomaly detection and enhancements (2025-12)
try:
    from app.training.training_enhancements import (
        AdaptiveGradientClipper,
        CheckpointAverager,
        EvaluationFeedbackHandler,
        TrainingAnomalyDetector,
    )
    HAS_TRAINING_ENHANCEMENTS = True
except ImportError:
    TrainingAnomalyDetector = None
    CheckpointAverager = None
    AdaptiveGradientClipper = None
    EvaluationFeedbackHandler = None
    HAS_TRAINING_ENHANCEMENTS = False

# Hard example mining for curriculum learning (2025-12)
try:
    from app.training.enhancements.hard_example_mining import HardExampleMiner
    from app.training.enhancements.per_sample_loss import compute_per_sample_loss
    HAS_HARD_EXAMPLE_MINING = True
except ImportError:
    HardExampleMiner = None
    compute_per_sample_loss = None
    HAS_HARD_EXAMPLE_MINING = False

# Unified training enhancements facade (2025-12)
# Consolidates: hard example mining, per-sample loss, curriculum LR, freshness weighting
try:
    from app.training.enhancements.training_facade import (
        FacadeConfig,
        TrainingEnhancementsFacade,
    )
    HAS_TRAINING_FACADE = True
except ImportError:
    FacadeConfig = None
    TrainingEnhancementsFacade = None
    HAS_TRAINING_FACADE = False

# DataCatalog for cluster-wide training data discovery (2025-12)
try:
    from app.distributed.data_catalog import DataCatalog, get_data_catalog
    HAS_DATA_CATALOG = True
except ImportError:
    DataCatalog = None
    get_data_catalog = None
    HAS_DATA_CATALOG = False

# Quality-weighted training (2025-12) - resurrected from ebmo_network.py
try:
    from app.training.quality_weighted_loss import (
        QualityWeightedTrainer,
        compute_quality_weights,
        quality_weighted_policy_loss,
        ranking_loss_from_quality,
    )
    HAS_QUALITY_WEIGHTING = True
except ImportError:
    QualityWeightedTrainer = None
    compute_quality_weights = None
    quality_weighted_policy_loss = None
    ranking_loss_from_quality = None
    HAS_QUALITY_WEIGHTING = False

# Auto-streaming threshold: datasets larger than this will automatically use
# StreamingDataLoader to avoid OOM. Default 5GB.
AUTO_STREAMING_THRESHOLD_BYTES = int(os.environ.get(
    "RINGRIFT_AUTO_STREAMING_THRESHOLD_GB", "5"
)) * 1024 * 1024 * 1024

from app.training.heuristic_tuning import (
    HEURISTIC_WEIGHT_KEYS,
    _flatten_heuristic_weights,
    _reconstruct_heuristic_profile,
    evaluate_heuristic_candidate,
    run_cmaes_heuristic_optimization,
    temporary_heuristic_profile,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def seed_all_legacy(seed: int = 42) -> None:
    """Backwards-compatible seeding wrapper kept for older callers/tests."""
    seed_all(seed)


# EarlyStopping is now imported from training_enhancements for consolidation
# The EnhancedEarlyStopping class provides backwards compatibility via __call__ method
from app.training.training_enhancements import EarlyStopping

# Checkpointing utilities - use unified module (2025-12)
HAS_UNIFIED_CHECKPOINT = True

# Legacy checkpointing functions (still available for backward compatibility)
# Migrated to import from checkpoint_unified (December 2025)
from app.training.checkpoint_unified import (
    AsyncCheckpointer,
    GracefulShutdownHandler,
    load_checkpoint,
    save_checkpoint,
)
from app.training.datasets import RingRiftDataset
from app.training.fault_tolerance import HeartbeatMonitor

def train_model(
    config: TrainConfig,
    data_path: str | list[str],
    save_path: str,
    early_stopping_patience: int | None = None,
    elo_early_stopping_patience: int | None = None,
    elo_min_improvement: float | None = None,
    checkpoint_dir: str = 'checkpoints',
    checkpoint_interval: int = 5,
    _save_all_epochs: bool = True,  # Save every epoch for Elo-based selection
    warmup_epochs: int | None = None,
    lr_scheduler: str | None = None,
    lr_min: float | None = None,
    lr_t0: int = 10,
    lr_t_mult: int = 2,
    resume_path: str | None = None,
    init_weights_path: str | None = None,
    init_weights_strict: bool = False,
    freeze_policy: bool = False,
    augment_hex_symmetry: bool = False,
    distributed: bool = False,
    local_rank: int = -1,
    scale_lr: bool = False,
    lr_scale_mode: str = 'linear',
    find_unused_parameters: bool = False,
    use_streaming: bool = False,
    data_dir: str | None = None,
    sampling_weights: str = 'uniform',
    multi_player: bool = False,
    num_players: int = 2,
    model_version: str = 'v2',
    num_res_blocks: int | None = None,
    num_filters: int | None = None,
    heartbeat_file: str | None = None,
    heartbeat_interval: float = 30.0,
    # 2024-12 Training Improvements (accept but log for now)
    spectral_norm: bool = False,
    cyclic_lr: bool = False,
    cyclic_lr_period: int = 5,
    mixed_precision: bool = False,
    amp_dtype: str = 'bfloat16',
    value_whitening: bool = False,
    value_whitening_momentum: float = 0.99,
    ema: bool = False,
    ema_decay: float = 0.999,
    stochastic_depth: bool = False,
    stochastic_depth_prob: float = 0.1,
    adaptive_warmup: bool = False,
    # Dec 28, 2025: Enabled by default to reduce overfitting and reach 2000+ Elo
    hard_example_mining: bool = True,  # Focus on difficult examples
    hard_example_top_k: float = 0.3,
    # Outcome-weighted policy loss (2025-12)
    # Weights policy loss by game outcome: winner's moves → higher weight, loser's → lower
    # Inspired by EBMO outcome-contrastive loss for improved move quality learning
    # Dec 28, 2025: Enabled by default to improve move quality learning
    enable_outcome_weighted_policy: bool = True,  # Learn from winning moves
    outcome_weight_scale: float = 0.5,  # How much to scale by outcome (0=no effect, 1=full)
    auto_tune_batch_size: bool = True,  # Enabled by default for 15-30% better throughput
    # January 2026: Conservative memory targeting (50% default, 35% safe mode)
    target_memory_fraction: float | None = None,  # None = use config default (50% or 35% safe mode)
    safe_mode: bool = False,  # Extra conservative batch sizing (35% memory target)
    track_calibration: bool = False,
    # 2024-12 Hot Data Buffer and Integrated Enhancements
    use_hot_data_buffer: bool = False,
    hot_buffer_size: int = 10000,
    hot_buffer_mix_ratio: float = 0.3,
    external_hot_buffer: Any | None = None,  # Pre-populated HotDataBuffer from caller
    use_integrated_enhancements: bool = True,  # December 2025: Enable by default for Elo improvement
    # Dec 28, 2025: Enabled curriculum and augmentation by default to reduce overfitting
    enable_curriculum: bool = True,  # Progressive difficulty during training
    enable_augmentation: bool = True,  # Board symmetry augmentation
    enable_elo_weighting: bool = True,  # December 2025: Enable for sample prioritization (+20-35 Elo)
    enable_auxiliary_tasks: bool = True,  # December 2025: Enable for multi-task learning (+5-15 Elo)
    enable_batch_scheduling: bool = False,
    enable_background_eval: bool = True,  # December 2025: Enable for real-time Elo feedback (+30-50 Elo)
    # Policy label smoothing (2025-12)
    policy_label_smoothing: float = 0.0,
    # Data validation (2025-12)
    validate_data: bool = True,
    fail_on_invalid_data: bool = False,
    # Fault tolerance (2025-12)
    enable_circuit_breaker: bool = True,
    enable_anomaly_detection: bool = True,
    gradient_clip_mode: str = 'adaptive',
    gradient_clip_max_norm: float = 1.0,
    anomaly_spike_threshold: float = 3.0,
    anomaly_gradient_threshold: float = 100.0,
    enable_graceful_shutdown: bool = True,
    # Regularization (2025-12)
    dropout: float = 0.08,
    # Quality-aware data discovery (2025-12)
    discover_synced_data: bool = False,
    min_quality_score: float = 0.0,
    _include_local_data: bool = True,
    _include_nfs_data: bool = True,
    # Learning rate finder (2025-12)
    find_lr: bool = False,
    lr_finder_min: float = 1e-7,
    lr_finder_max: float = 1.0,
    lr_finder_iterations: int = 100,
    # GNN support (2025-12)
    model_type: str = "cnn",  # "cnn", "gnn", or "hybrid"
    # Training data freshness check (2025-12)
    # MANDATORY BY DEFAULT - prevents 95% of stale data training incidents
    # Phase 1.5 of improvement plan: fail early if data is stale
    skip_freshness_check: bool = False,  # Default: check IS enabled
    max_data_age_hours: float = 2000.0,  # Default: data must be <2000 hours old (relaxed)
    allow_stale_data: bool = False,      # Default: FAIL on stale data (not warn)
    # Stale fallback for 48-hour autonomous operation (December 2025)
    # Allows training to proceed with stale data after sync failures or timeout
    disable_stale_fallback: bool = False,  # If True, no automatic fallback
    max_sync_failures: int = 5,            # Failures before fallback allowed
    max_sync_duration: float = 2700.0,     # Seconds (45 min) before fallback
    # Checkpoint averaging (2025-12)
    # Averages last N checkpoints at end of training for +10-20 Elo improvement
    enable_checkpoint_averaging: bool = True,
    num_checkpoints_to_average: int = 5,
    # Best checkpoint selection on overfitting (January 2026)
    # When overfitting detected (val_loss/train_loss > threshold), use best checkpoint
    # instead of averaged. This prevents averaged overfit checkpoints from degrading quality.
    prefer_best_on_overfit: bool = True,
    overfit_divergence_threshold: float = 0.5,  # 50% divergence triggers best checkpoint
    # Quality-weighted training (2025-12) - resurrected from ebmo_network.py
    # December 2025: Enabled by default to improve training signal quality
    # Quality weighting focuses learning on high-quality MCTS-derived moves
    enable_quality_weighting: bool = True,
    quality_weight_blend: float = 0.5,
    quality_ranking_weight: float = 0.1,
    # Auto-promotion after training (January 2026)
    # Runs gauntlet evaluation and promotes if criteria met (Elo parity OR win rate floors)
    auto_promote: bool = False,
    auto_promote_games: int = 30,
    auto_promote_sync: bool = True,
    # Gradient checkpointing (January 2026)
    # Trades compute for memory - recomputes activations during backward pass
    # Enables training large models (e.g., hexagonal) on memory-constrained GPUs
    gradient_checkpointing: bool = False,
) -> dict[str, Any]:
    """
    Train the RingRift neural network model.

    Args:
        config: Training configuration
        data_path: Path(s) to training data (.npz file or list of files)
        save_path: Path to save the best model weights
        early_stopping_patience: Number of epochs without loss improvement before
            stopping (0 to disable early stopping)
        elo_early_stopping_patience: Number of epochs without Elo improvement
            before stopping (works in conjunction with loss patience when both
            are tracked; 0 to disable Elo-based early stopping)
        elo_min_improvement: Minimum Elo improvement (default 5.0) to reset
            the Elo patience counter
        checkpoint_dir: Directory for saving periodic checkpoints
        checkpoint_interval: Save checkpoint every N epochs
        warmup_epochs: Number of epochs for LR warmup (0 to disable)
        lr_scheduler: Type of LR scheduler:
            - 'none': No scheduling (constant LR after warmup)
            - 'step': Step decay by 0.5 every 10 epochs
            - 'cosine': CosineAnnealingLR over remaining epochs
            - 'cosine-warm-restarts': CosineAnnealingWarmRestarts
        lr_min: Minimum learning rate for cosine annealing (default: 1e-6)
        lr_t0: T_0 for CosineAnnealingWarmRestarts (initial restart period)
        lr_t_mult: T_mult for CosineAnnealingWarmRestarts (period multiplier)
        resume_path: Path to checkpoint to resume training from
        augment_hex_symmetry: Enable D6 symmetry augmentation for hex boards
        distributed: Enable distributed training with DDP
        local_rank: Local rank for distributed training (set by torchrun)
        scale_lr: Whether to scale learning rate with world size
        lr_scale_mode: LR scaling mode ('linear' or 'sqrt')
        find_unused_parameters: Enable find_unused_parameters for DDP
        use_streaming: Use StreamingDataLoader for large datasets
        data_dir: Directory containing multiple .npz files (for streaming)
        sampling_weights: Position sampling strategy for non-streaming data:
            'uniform', 'late_game', 'phase_emphasis', or 'combined'
        use_integrated_enhancements: Enable IntegratedTrainingManager for advanced features
        enable_curriculum: Enable curriculum learning (difficulty progression)
        enable_augmentation: Enable data augmentation (symmetry transforms)
        enable_elo_weighting: Enable Elo-based sample weighting
        enable_auxiliary_tasks: Enable auxiliary prediction tasks (outcome classification)
            Requires model support for return_features=True
        enable_batch_scheduling: Enable dynamic batch size scheduling (linear ramp-up)
        enable_background_eval: Enable background Elo evaluation during training
            Provides early stopping based on Elo tracking
        find_lr: Run learning rate finder before training to find optimal LR
        lr_finder_min: Minimum LR for range test (default 1e-7)
        lr_finder_max: Maximum LR for range test (default 1.0)
        lr_finder_iterations: Number of iterations for LR range test (default 100)
    """
    # Resolve optional parameters using TrainConfigResolver (December 2025)
    # Provides consistent precedence: explicit param > config attr > default
    resolved = resolve_train_config(
        config=config,
        early_stopping_patience=early_stopping_patience,
        elo_early_stopping_patience=elo_early_stopping_patience,
        elo_min_improvement=elo_min_improvement,
        warmup_epochs=warmup_epochs,
        lr_scheduler=lr_scheduler,
        lr_min=lr_min,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        distributed=distributed,
        local_rank=local_rank,
        num_players=num_players,
    )

    # Compute data_path_str early — used in error messages throughout
    if isinstance(data_path, list):
        data_path_str = data_path[0] if data_path else ""
    else:
        data_path_str = data_path

    # Extract resolved values for backward compatibility
    early_stopping_patience = resolved.early_stopping_patience
    elo_early_stopping_patience = resolved.elo_early_stopping_patience
    elo_min_improvement = resolved.elo_min_improvement
    warmup_epochs = resolved.warmup_epochs
    lr_scheduler = resolved.lr_scheduler
    lr_min = resolved.lr_min

    # Set up distributed training if enabled
    if distributed:
        # Setup distributed process group
        setup_distributed(local_rank)
        world_size = get_world_size()

        # Seed with rank offset for different random state per process
        seed_everything(config.seed, rank_offset=True)

        # Scale learning rate if requested
        if scale_lr:
            config.learning_rate = scale_learning_rate(
                config.learning_rate, world_size, scale_type=lr_scale_mode
            )
            if is_main_process():
                logger.info(
                    f"Scaled learning rate to {config.learning_rate:.6f} "
                    f"({lr_scale_mode} scaling with world_size={world_size})"
                )
    else:
        seed_all(config.seed)

    # ==========================================================================
    # Pre-Training Validation (extracted to train_pre_validation.py)
    # ==========================================================================
    run_pre_training_validation(
        data_path=data_path,
        config=config,
        num_players=num_players,
        distributed=distributed,
        is_main=not distributed or is_main_process(),
        skip_freshness_check=skip_freshness_check,
        max_data_age_hours=max_data_age_hours,
        allow_stale_data=allow_stale_data,
        disable_stale_fallback=disable_stale_fallback,
        max_sync_failures=max_sync_failures,
        max_sync_duration=max_sync_duration,
        validate_data=validate_data,
        fail_on_invalid_data=fail_on_invalid_data,
        use_streaming=use_streaming,
        check_freshness_sync=check_freshness_sync,
        validate_npz_structure_fn=validate_npz_structure,
        validate_npz_file_fn=validate_npz_file,
        verify_npz_checksums_fn=verify_npz_checksums,
        should_allow_stale_training_fn=should_allow_stale_training,
        HAS_FRESHNESS_CHECK=HAS_FRESHNESS_CHECK,
        HAS_NPZ_STRUCTURE_VALIDATION=HAS_NPZ_STRUCTURE_VALIDATION,
        HAS_DATA_VALIDATION=HAS_DATA_VALIDATION,
        HAS_CHECKSUM_VERIFICATION=HAS_CHECKSUM_VERIFICATION,
        HAS_STALE_FALLBACK=HAS_STALE_FALLBACK,
        DataEventType=DataEventType,
    )

    # ==========================================================================
    # Mandatory NPZ validation (Feb 2026)
    # Catches empty, corrupt, or mismatched data before model/optimizer creation
    # ==========================================================================
    if not use_streaming:
        npz_paths = data_path if isinstance(data_path, list) else [data_path]
        board_type_str = (
            config.board_type.value
            if hasattr(config.board_type, "value")
            else str(config.board_type)
        )
        valid_paths = [p for p in npz_paths if p]
        if not valid_paths:
            raise ValueError(
                "No training data paths specified. "
                "Use --data-path to provide NPZ training data."
            )
        for p in valid_paths:
            try:
                validate_training_data(Path(p), board_type_str, num_players)
            except (ValueError, FileNotFoundError) as e:
                logger.error(f"Training data validation failed: {e}")
                raise

    # Device configuration
    if distributed:
        # In distributed mode, use the local_rank device
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        if is_main_process():
            logger.info(
                f"Distributed training on device: {device} "
                f"(rank {get_rank()}/{get_world_size()})"
            )
    else:
        # Standard single-device selection
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        logger.info(f"Using device: {device}")

    runtime_setup = initialize_training_runtime_setup(
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        distributed=distributed,
        is_main=not distributed or is_main_process(),
        spectral_norm=spectral_norm,
        cyclic_lr=cyclic_lr,
        cyclic_lr_period=cyclic_lr_period,
        mixed_precision=mixed_precision,
        amp_dtype=amp_dtype,
        value_whitening=value_whitening,
        ema=ema,
        ema_decay=ema_decay,
        stochastic_depth=stochastic_depth,
        stochastic_depth_prob=stochastic_depth_prob,
        adaptive_warmup=adaptive_warmup,
        hard_example_mining=hard_example_mining,
        hard_example_top_k=hard_example_top_k,
        use_hot_data_buffer=use_hot_data_buffer,
        hot_buffer_size=hot_buffer_size,
        hot_buffer_mix_ratio=hot_buffer_mix_ratio,
        external_hot_buffer=external_hot_buffer,
        use_integrated_enhancements=use_integrated_enhancements,
        enable_curriculum=enable_curriculum,
        enable_augmentation=enable_augmentation,
        enable_elo_weighting=enable_elo_weighting,
        enable_auxiliary_tasks=enable_auxiliary_tasks,
        enable_batch_scheduling=enable_batch_scheduling,
        enable_background_eval=enable_background_eval,
        enable_checkpoint_averaging=enable_checkpoint_averaging,
        num_checkpoints_to_average=num_checkpoints_to_average,
        enable_quality_weighting=enable_quality_weighting,
        quality_weight_blend=quality_weight_blend,
        quality_ranking_weight=quality_ranking_weight,
        has_hot_data_buffer=HAS_HOT_DATA_BUFFER,
        hot_data_buffer_cls=HotDataBuffer,
        has_quality_bridge=HAS_QUALITY_BRIDGE,
        get_quality_bridge=get_quality_bridge,
        has_integrated_enhancements=HAS_INTEGRATED_ENHANCEMENTS,
        integrated_enhancements_config_cls=IntegratedEnhancementsConfig,
        integrated_training_manager_cls=IntegratedTrainingManager,
        checkpoint_averager_cls=CheckpointAverager,
        has_hard_example_mining=HAS_HARD_EXAMPLE_MINING,
        hard_example_miner_cls=HardExampleMiner,
        has_training_facade=HAS_TRAINING_FACADE,
        training_facade_cls=TrainingEnhancementsFacade,
        facade_config_cls=FacadeConfig,
        has_quality_weighting=HAS_QUALITY_WEIGHTING,
        quality_weighted_trainer_cls=QualityWeightedTrainer,
        gradient_surgeon_cls=GradientSurgeon,
        gradient_surgery_config_cls=GradientSurgeryConfig,
        has_metrics_collector=HAS_METRICS_COLLECTOR,
        metrics_collector_cls=MetricsCollector,
    )
    hot_buffer = runtime_setup.hot_buffer
    enhancements_manager = runtime_setup.enhancements_manager
    checkpoint_averager = runtime_setup.checkpoint_averager
    hard_example_miner = runtime_setup.hard_example_miner
    training_facade = runtime_setup.training_facade
    quality_trainer = runtime_setup.quality_trainer
    amp_enabled = runtime_setup.amp_enabled
    amp_torch_dtype = runtime_setup.amp_torch_dtype
    use_grad_scaler = runtime_setup.use_grad_scaler
    scaler = runtime_setup.scaler
    gradient_surgeon = runtime_setup.gradient_surgeon
    use_gradient_surgery = runtime_setup.use_gradient_surgery
    metrics_collector = runtime_setup.metrics_collector

    metadata_context = prepare_dataset_metadata_context(
        data_path=data_path,
        config=config,
        num_players=num_players,
        model_version=model_version,
        multi_player=multi_player,
        use_streaming=use_streaming,
        distributed=distributed,
        is_main=not distributed or is_main_process(),
        resume_path=resume_path,
        num_filters=num_filters,
        num_res_blocks=num_res_blocks,
        device=device,
        data_path_str=data_path_str,
        BoardType=BoardType,
        HEX_BOARD_SIZE=HEX_BOARD_SIZE,
        HEX8_BOARD_SIZE=HEX8_BOARD_SIZE,
        MAX_PLAYERS=MAX_PLAYERS,
        get_policy_size_for_board=get_policy_size_for_board,
        normalize_board_type=normalize_board_type,
        validate_hex_policy_indices=validate_hex_policy_indices,
        detect_tier_from_checkpoint=detect_tier_from_checkpoint,
    )
    board_size = metadata_context.board_size
    policy_size = metadata_context.policy_size
    encoding_channels = metadata_context.encoding_channels
    hex_num_players = metadata_context.hex_num_players
    use_hex_model = metadata_context.use_hex_model
    use_hex_v3 = metadata_context.use_hex_v3
    use_hex_v4 = metadata_context.use_hex_v4
    use_hex_v5 = metadata_context.use_hex_v5
    use_hex_v5_large = metadata_context.use_hex_v5_large
    detected_num_heuristics = metadata_context.detected_num_heuristics
    config_feature_version = metadata_context.config_feature_version
    hex_radius = metadata_context.hex_radius

    # Determine model architecture size (allow CLI override for scaling up)
    # Default: 11 blocks / 160 filters for v5, 13 blocks / 128 filters for v4,
    # 12 blocks / 192 filters for v3/hex, 6 blocks / 96 filters for v2
    # Note: v5-heavy-large/xl use factory defaults from v5_heavy_large.py
    if use_hex_v5 or model_version in ('v5', 'v5-gnn', 'v5-heavy'):
        effective_blocks = num_res_blocks if num_res_blocks is not None else 11  # 6 SE + 5 attention
        effective_filters = num_filters if num_filters is not None else 160  # v5 default
    elif model_version in ('v5-heavy-large', 'v5-heavy-xl', 'v6', 'v6-xl'):
        # v5-heavy-large/xl use configs from v5_heavy_large.py (256-320 filters)
        # Don't override effective_blocks/filters - factory handles defaults
        effective_blocks = num_res_blocks if num_res_blocks is not None else 20  # 10 SE + 10 attention
        effective_filters = num_filters if num_filters is not None else 256  # Large default
    elif use_hex_v4 or model_version == 'v4':
        effective_blocks = num_res_blocks if num_res_blocks is not None else 13  # NAS optimal
        effective_filters = num_filters if num_filters is not None else 128  # NAS optimal
    elif model_version == 'v3' or use_hex_model:
        effective_blocks = num_res_blocks if num_res_blocks is not None else 12
        effective_filters = num_filters if num_filters is not None else 192
    else:
        effective_blocks = num_res_blocks if num_res_blocks is not None else 6
        effective_filters = num_filters if num_filters is not None else 96

    # Log architecture size if non-default
    if (num_res_blocks is not None or num_filters is not None) and (not distributed or is_main_process()):
        logger.info(
            f"Using custom architecture: {effective_blocks} residual blocks, "
            f"{effective_filters} filters"
        )

    model_artifacts = prepare_training_model_artifacts(
        config=config,
        model_version=model_version,
        model_type=model_type,
        board_size=board_size,
        policy_size=policy_size,
        num_players=num_players,
        encoding_channels=encoding_channels,
        hex_radius=hex_radius,
        hex_num_players=hex_num_players,
        use_hex_model=use_hex_model,
        use_hex_v3=use_hex_v3,
        use_hex_v4=use_hex_v4,
        use_hex_v5=use_hex_v5,
        use_hex_v5_large=use_hex_v5_large,
        detected_num_heuristics=detected_num_heuristics,
        effective_blocks=effective_blocks,
        effective_filters=effective_filters,
        multi_player=multi_player,
        dropout=dropout,
        config_feature_version=config_feature_version,
        distributed=distributed,
        is_main=not distributed or is_main_process(),
        device=device,
        enhancements_manager=enhancements_manager,
        gradient_checkpointing=gradient_checkpointing,
        auto_tune_batch_size=auto_tune_batch_size,
        target_memory_fraction=target_memory_fraction,
        safe_mode=safe_mode,
        save_path=save_path,
        init_weights_path=init_weights_path,
        init_weights_strict=init_weights_strict,
        resume_path=resume_path,
        find_unused_parameters=find_unused_parameters,
        warmup_epochs=warmup_epochs,
        lr_scheduler=lr_scheduler,
        lr_min=lr_min,
        lr_t0=lr_t0,
        lr_t_mult=lr_t_mult,
        freeze_policy=freeze_policy,
        early_stopping_patience=early_stopping_patience,
        elo_early_stopping_patience=elo_early_stopping_patience,
        elo_min_improvement=elo_min_improvement,
        checkpoint_dir=checkpoint_dir,
        data_path_str=data_path_str,
        has_training_enhancements=HAS_TRAINING_ENHANCEMENTS,
        evaluation_feedback_handler_cls=EvaluationFeedbackHandler,
    )
    model = model_artifacts.model
    optimizer = model_artifacts.optimizer
    epoch_scheduler = model_artifacts.epoch_scheduler
    plateau_scheduler = model_artifacts.plateau_scheduler
    eval_feedback_handler = model_artifacts.eval_feedback_handler
    early_stopper = model_artifacts.early_stopper
    start_epoch = model_artifacts.start_epoch

    value_criterion = nn.MSELoss()
    nn.KLDivLoss(reduction="batchmean")
    use_multi_player_loss = multi_player

    checkpoint_services = initialize_checkpoint_services(
        config=config,
        track_calibration=track_calibration,
        is_main=not distributed or is_main_process(),
    )
    async_checkpointer = checkpoint_services.async_checkpointer
    calibration_tracker = checkpoint_services.calibration_tracker

    # Mixed precision scaler configured above (GradScaler only for float16)

    pipeline_context = prepare_training_data_pipeline(
        config=config,
        data_path=data_path,
        data_path_str=data_path_str,
        data_dir=data_dir,
        augment_hex_symmetry=augment_hex_symmetry,
        use_streaming=use_streaming,
        sampling_weights=sampling_weights,
        multi_player=multi_player,
        enable_elo_weighting=enable_elo_weighting,
        min_quality_score=min_quality_score,
        discover_synced_data=discover_synced_data,
        distributed=distributed,
        is_main=not distributed or is_main_process(),
        policy_size=policy_size,
        use_hex_model=use_hex_model,
        use_hex_v3=use_hex_v3,
        model_version=model_version,
        config_feature_version=config_feature_version,
        auto_streaming_threshold_bytes=AUTO_STREAMING_THRESHOLD_BYTES,
        has_data_catalog=HAS_DATA_CATALOG,
        get_data_catalog=get_data_catalog,
    )
    if pipeline_context is None:
        return

    use_streaming = pipeline_context.use_streaming
    train_streaming_loader = pipeline_context.train_streaming_loader
    val_streaming_loader = pipeline_context.val_streaming_loader
    train_loader = pipeline_context.train_loader
    val_loader = pipeline_context.val_loader
    train_sampler = pipeline_context.train_sampler
    val_sampler = pipeline_context.val_sampler
    full_dataset = pipeline_context.full_dataset
    train_size = pipeline_context.train_size
    val_size = pipeline_context.val_size
    value_only_training = pipeline_context.value_only_training
    prepared_total_samples = pipeline_context.total_samples
    prepared_num_data_files = pipeline_context.num_data_files

    # Phase 6: Validate training compatibility before starting
    if full_dataset is not None and (not distributed or is_main_process()):
        try:
            _validate_training_compatibility(model, full_dataset, config)
        except ValueError as e:
            logger.error(f"Training compatibility validation failed: {e}")
            if fail_on_invalid_data:
                raise
            else:
                logger.warning("Continuing despite validation failure (fail_on_invalid_data=False)")

    if not distributed or is_main_process():
        logger.info(
            f"Starting training for {config.epochs_per_iter} epochs..."
        )
        logger.info(f"Train size: {train_size}, Val size: {val_size}")
        if use_streaming:
            logger.info("Using StreamingDataLoader for memory-efficient data")
            if distributed:
                logger.info(
                    f"  Data sharding: rank {get_rank()}/{get_world_size()}, "
                    f"~{train_size // get_world_size()} samples per rank"
                )
        if distributed:
            logger.info(
                f"Distributed training with {get_world_size()} processes"
            )
        if early_stopper is not None:
            elo_info = ""
            if elo_early_stopping_patience > 0:
                elo_info = f", Elo patience: {elo_early_stopping_patience} (min improvement: {elo_min_improvement})"
            logger.info(
                f"Early stopping enabled with loss patience: "
                f"{early_stopping_patience}{elo_info}"
            )
        if warmup_epochs > 0:
            logger.info(f"LR warmup enabled for {warmup_epochs} epochs")
        if lr_scheduler in ('cosine', 'cosine-warm-restarts'):
            logger.info(
                f"LR scheduler: {lr_scheduler} (min_lr={lr_min})"
            )
            if lr_scheduler == 'cosine-warm-restarts':
                logger.info(f"  T_0={lr_t0}, T_mult={lr_t_mult}")
        logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    run_support = initialize_training_run_support(
        config=config,
        num_players=num_players,
        batch_size_metric=BATCH_SIZE,
        has_prometheus=HAS_PROMETHEUS,
        distributed=distributed,
        is_main=not distributed or is_main_process(),
        heartbeat_file=heartbeat_file,
        heartbeat_interval=heartbeat_interval,
        start_epoch=start_epoch,
        checkpoint_dir=checkpoint_dir,
        enable_graceful_shutdown=enable_graceful_shutdown,
        enable_circuit_breaker=enable_circuit_breaker,
        enable_anomaly_detection=enable_anomaly_detection,
        gradient_clip_mode=gradient_clip_mode,
        gradient_clip_max_norm=gradient_clip_max_norm,
        anomaly_spike_threshold=anomaly_spike_threshold,
        anomaly_gradient_threshold=anomaly_gradient_threshold,
        model=model,
        optimizer=optimizer,
        epoch_scheduler=epoch_scheduler,
        early_stopper=early_stopper,
        enhancements_manager=enhancements_manager,
        distributed_metrics_cls=DistributedMetrics,
        heartbeat_monitor_cls=HeartbeatMonitor,
        loss_monitor_cls=LossMonitor,
        fault_tolerance_config_cls=FaultToleranceConfig,
        setup_fault_tolerance_fn=setup_fault_tolerance,
        training_state_cls=TrainingState,
        graceful_shutdown_handler_cls=GracefulShutdownHandler,
        save_checkpoint_fn=save_checkpoint,
        has_event_bus=HAS_EVENT_BUS,
        get_router_fn=get_router,
        data_event_cls=DataEvent,
        data_event_type=DataEventType,
        time_module=time,
    )
    dist_metrics = run_support.dist_metrics
    heartbeat_monitor = run_support.heartbeat_monitor
    best_val_loss = run_support.best_val_loss
    best_train_loss_at_best_val = run_support.best_train_loss_at_best_val
    avg_val_loss = run_support.avg_val_loss
    avg_train_loss = run_support.avg_train_loss
    avg_policy_accuracy = run_support.avg_policy_accuracy
    epoch_losses = run_support.epoch_losses
    epochs_completed = run_support.epochs_completed
    _training_completed_normally = run_support.training_completed_normally
    _training_exception = run_support.training_exception
    _training_start_time = run_support.training_start_time
    _final_checkpoint_path = run_support.final_checkpoint_path
    _total_samples = run_support.total_samples
    _num_data_files = run_support.num_data_files
    config_label = run_support.config_label
    loss_monitor = run_support.loss_monitor
    training_breaker = run_support.training_breaker
    anomaly_detector = run_support.anomaly_detector
    adaptive_clipper = run_support.adaptive_clipper
    fixed_clip_norm = run_support.fixed_clip_norm
    gradient_clip_mode = run_support.gradient_clip_mode
    anomaly_step = run_support.anomaly_step
    training_state = run_support.training_state
    shutdown_handler = run_support.shutdown_handler
    rollback_handler = run_support.rollback_handler
    _last_good_checkpoint_path = run_support.last_good_checkpoint_path
    _last_good_epoch = run_support.last_good_epoch
    _circuit_breaker_rollbacks = run_support.circuit_breaker_rollbacks
    _max_circuit_breaker_rollbacks = run_support.max_circuit_breaker_rollbacks

    maybe_run_lr_finder(
        find_lr=find_lr,
        is_main=not distributed or is_main_process(),
        distributed=distributed,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        device=device,
        lr_finder_min=lr_finder_min,
        lr_finder_max=lr_finder_max,
        lr_finder_iterations=lr_finder_iterations,
    )

    try:
        for epoch in range(start_epoch, config.epochs_per_iter):
            # Circuit breaker check - skip training if circuit is open (2025-12)
            if training_breaker and not training_breaker.can_execute("training_epoch"):
                logger.warning(f"Training circuit OPEN - skipping epoch {epoch} (recovering from failures)")
                # Update circuit breaker state metric (1=open)
                if HAS_PROMETHEUS and CIRCUIT_BREAKER_STATE and (not distributed or is_main_process()):
                    CIRCUIT_BREAKER_STATE.labels(config=config_label, operation='training_epoch').set(1)

                # Attempt checkpoint rollback if we have a good checkpoint (2025-12)
                if _last_good_checkpoint_path and _circuit_breaker_rollbacks < _max_circuit_breaker_rollbacks:
                    _circuit_breaker_rollbacks += 1
                    logger.warning(
                        f"Circuit breaker rollback {_circuit_breaker_rollbacks}/{_max_circuit_breaker_rollbacks}: "
                        f"restoring checkpoint from epoch {_last_good_epoch}"
                    )
                    try:
                        # Load the last good checkpoint
                        loaded_epoch, loaded_loss = load_checkpoint(
                            _last_good_checkpoint_path, model, optimizer,
                            scheduler=epoch_scheduler, device=device
                        )
                        logger.info(f"Rollback successful: restored to epoch {loaded_epoch}, loss {loaded_loss:.4f}")

                        # Reduce learning rate by 50% to stabilize training
                        for param_group in optimizer.param_groups:
                            old_lr = param_group['lr']
                            param_group['lr'] = old_lr * 0.5
                            logger.info(f"Reduced learning rate: {old_lr:.2e} -> {param_group['lr']:.2e}")

                        # Reset circuit breaker to allow retry
                        if training_breaker:
                            training_breaker.record_success("training_epoch")
                    except (OSError, RuntimeError, AttributeError) as e:
                        # File I/O errors, state restoration failures, or missing attributes
                        logger.error(f"Rollback failed: {e}")

                time.sleep(TRAINING_RETRY_SLEEP_SECONDS)  # Configurable pause before retry
                continue

            # Update circuit breaker state metric (0=closed, training can proceed)
            if HAS_PROMETHEUS and CIRCUIT_BREAKER_STATE and training_breaker and (not distributed or is_main_process()):
                CIRCUIT_BREAKER_STATE.labels(config=config_label, operation='training_epoch').set(0)

            # Circuit breaker: Check resources at the start of each epoch
            # This prevents training from overwhelming the system when resources are constrained
            if epoch % 5 == 0:  # Check every 5 epochs to minimize overhead
                try:
                    from app.utils.resource_guard import can_proceed, get_resource_status, wait_for_resources
                    if not can_proceed(check_disk=True, check_mem=True, check_cpu_load=False):
                        status = get_resource_status()
                        logger.warning(
                            f"Resource pressure detected at epoch {epoch}: "
                            f"CPU={status['cpu']['used_percent']:.0f}%, "
                            f"Memory={status['memory']['used_percent']:.0f}%, "
                            f"Disk={status['disk']['used_percent']:.0f}%. "
                            f"Waiting for resources..."
                        )
                        if not wait_for_resources(timeout=300.0, mem_required_gb=2.0):
                            logger.warning("Resources still constrained after 5 min wait, continuing anyway")
                except ImportError:
                    pass  # resource_guard not available

            # Log scheduled batch size if batch scheduling is enabled
            if enhancements_manager is not None and enhancements_manager._batch_scheduler is not None:
                scheduled_batch = enhancements_manager.get_batch_size()
                if not distributed or is_main_process():
                    logger.info(f"Epoch {epoch+1}: scheduled batch size = {scheduled_batch}")

            # Set epoch for distributed sampler or streaming loader
            if distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if use_streaming:
                assert train_streaming_loader is not None
                assert val_streaming_loader is not None
                train_streaming_loader.set_epoch(epoch)
                val_streaming_loader.set_epoch(epoch)

            # Track epoch failure state for circuit breaker (2025-12)

            # Phase 2 Feedback Loop: Check improvement optimizer for training adjustments
            # December 2025: Training now responds to evaluation signals (promotion streaks, regressions)
            # Note: HYPERPARAMETER_UPDATED events are handled in real-time by EvaluationFeedbackHandler
            # (see subscribe() call above). This polling is a fallback for cross-process updates.
            # Check EVERY epoch for pending hyperparameter updates from gauntlet feedback
            try:
                from app.coordination.gauntlet_feedback_controller import get_pending_hyperparameter_updates
                pending_updates = get_pending_hyperparameter_updates(config_label)
                if pending_updates and (not distributed or is_main_process()):
                    logger.info(f"[GauntletFeedback] Applying {len(pending_updates)} hyperparameter update(s) at epoch {epoch}")

                for param, update in pending_updates.items():
                    value = update.get("value")
                    reason = update.get("reason", "gauntlet_feedback")

                    # Learning rate adjustments
                    if param == "learning_rate" and isinstance(value, (int, float)):
                        for param_group in optimizer.param_groups:
                            old_lr = param_group["lr"]
                            param_group["lr"] = float(value)
                        if not distributed or is_main_process():
                            logger.info(
                                f"[GauntletFeedback] LR adjusted: {old_lr:.2e} -> {value:.2e} (reason: {reason})"
                            )
                    elif param == "lr_multiplier" and isinstance(value, (int, float)):
                        for param_group in optimizer.param_groups:
                            old_lr = param_group["lr"]
                            param_group["lr"] = old_lr * float(value)
                        if not distributed or is_main_process():
                            logger.info(
                                f"[GauntletFeedback] LR scaled: {old_lr:.2e} * {value:.2f} (reason: {reason})"
                            )

                    # Temperature scale (exploration reduction for strong models)
                    elif param == "temperature_scale" and isinstance(value, (int, float)):
                        # Temperature affects selfplay, not training directly
                        # Log the update for awareness - actual application happens in selfplay
                        if not distributed or is_main_process():
                            logger.info(
                                f"[GauntletFeedback] Temperature scale updated: {value:.2f} (reason: {reason})"
                            )
                            logger.info(
                                f"  Note: Temperature affects selfplay data generation, not training directly"
                            )

                    # Quality threshold boost (raise quality bar for strong models)
                    elif param == "quality_threshold_boost" and isinstance(value, (int, float)):
                        # Quality threshold affects data filtering in selfplay/training data generation
                        # Store for next training iteration
                        if not distributed or is_main_process():
                            logger.info(
                                f"[GauntletFeedback] Quality threshold boost: +{value:.3f} (reason: {reason})"
                            )
                            logger.info(
                                f"  Note: Quality threshold affects data filtering in future training iterations"
                            )

                    # Epoch multiplier (extend training for weak models)
                    elif param == "epoch_multiplier" and isinstance(value, (int, float)):
                        # Calculate how many additional epochs to run
                        multiplier = float(value)
                        original_epochs = config.epochs_per_iter
                        new_total_epochs = int(original_epochs * multiplier)
                        additional_epochs = new_total_epochs - original_epochs

                        if additional_epochs > 0 and (not distributed or is_main_process()):
                            logger.info(
                                f"[GauntletFeedback] Epoch extension requested: {multiplier:.1f}x "
                                f"({original_epochs} -> {new_total_epochs} epochs, +{additional_epochs}) "
                                f"(reason: {reason})"
                            )
                            logger.info(
                                f"  Note: Epoch extension will be applied in the next training run. "
                                f"Current run continues to {original_epochs} epochs."
                            )

                    # Unknown parameter - log for debugging
                    else:
                        if not distributed or is_main_process():
                            logger.debug(
                                f"[GauntletFeedback] Unknown parameter '{param}' = {value} (reason: {reason})"
                            )
            except ImportError:
                pass  # Gauntlet feedback not available
            except (AttributeError, TypeError, OSError, ConnectionError) as e:
                # Missing attributes, type errors, file I/O, or network issues
                if not distributed or is_main_process():
                    logger.debug(f"[GauntletFeedback] Failed to check updates: {e}")

            # Check improvement optimizer every 5 epochs (less frequent than gauntlet feedback)
            if epoch % 5 == 0:
                try:
                    from app.training.improvement_optimizer import get_training_adjustment

                    adjustment = get_training_adjustment(config_label)
                    if adjustment.get("lr_multiplier", 1.0) != 1.0:
                        lr_mult = adjustment["lr_multiplier"]
                        reason = adjustment.get("reason", "unknown")
                        for param_group in optimizer.param_groups:
                            old_lr = param_group["lr"]
                            param_group["lr"] = old_lr * lr_mult
                        if not distributed or is_main_process():
                            logger.info(
                                f"[ImprovementOptimizer] LR adjustment: {lr_mult:.2f}x (reason: {reason})"
                            )

                    if adjustment.get("regularization_boost", 0.0) > 0:
                        # Add extra weight decay for overfit mitigation
                        reg_boost = adjustment["regularization_boost"]
                        for param_group in optimizer.param_groups:
                            param_group["weight_decay"] = param_group.get("weight_decay", 0) + reg_boost
                        if not distributed or is_main_process():
                            logger.info(f"[ImprovementOptimizer] Regularization boost: +{reg_boost:.4f}")

                except ImportError:
                    pass  # Improvement optimizer not available
                except (AttributeError, TypeError, ValueError) as e:
                    # Missing attributes, type errors, or invalid values
                    if not distributed or is_main_process():
                        logger.debug(f"[ImprovementOptimizer] Check failed: {e}")

            # Training
            model.train()
            train_loss = torch.tensor(0.0, device=device)  # Accumulate on GPU to avoid per-batch .item() sync
            train_batches = 0
            if dist_metrics is not None:
                dist_metrics.reset()

            # Select appropriate data source
            # For multi-player mode with streaming, use iter_with_mp() to get
            # per-sample num_players from the batch.
            use_mp_iter = use_multi_player_loss and use_streaming and train_streaming_loader.has_multi_player_values
            if use_streaming:
                assert train_streaming_loader is not None
                # Use prefetch_loader for background prefetching if enabled
                use_prefetch = getattr(config, 'use_prefetch', True)
                pin_memory = getattr(config, 'pin_memory', True) and device.type == 'cuda'
                prefetch_count = getattr(config, 'prefetch_count', 2)
                # Enable async GPU transfer in prefetch thread (10-20% speedup)
                prefetch_to_device = getattr(config, 'prefetch_to_device', True) and device.type == 'cuda'

                if use_prefetch:
                    train_data_iter = prefetch_loader(
                        train_streaming_loader,
                        prefetch_count=prefetch_count,
                        pin_memory=pin_memory,
                        use_mp=use_mp_iter,
                        transfer_to_device=device if prefetch_to_device else None,
                    )
                elif use_mp_iter:
                    train_data_iter = train_streaming_loader.iter_with_mp()
                else:
                    train_data_iter = iter(train_streaming_loader)
            else:
                assert train_loader is not None
                train_data_iter = iter(train_loader)

            for i, batch_data in enumerate(train_data_iter):
                # Circuit breaker check: skip batches if circuit is open (2025-12)
                if training_breaker and not training_breaker.can_execute("batch_processing"):
                    if i % 100 == 0:  # Log every 100th skipped batch
                        logger.debug(f"Batch {i} skipped: circuit breaker open for batch_processing")
                    continue

                # Handle streaming, streaming with multi-player, and legacy batch formats
                batch_num_players = None  # Per-sample num_players or None
                batch_heuristics = None  # Heuristic features for v5 (if available)
                if use_streaming:
                    if use_multi_player_loss and train_streaming_loader.has_multi_player_values:
                        # Streaming with multi-player values
                        (
                            (features, globals_vec),
                            (value_targets, policy_targets),
                            values_mp_batch,
                            batch_num_players,
                        ) = batch_data
                        # Use values_mp as the value targets for multi-player loss
                        if values_mp_batch is not None:
                            value_targets = values_mp_batch
                    else:
                        (
                            (features, globals_vec),
                            (value_targets, policy_targets),
                        ) = batch_data
                else:
                    # Non-streaming mode: batch structure varies based on dataset config
                    # 4 elems: (features, globals, value, policy)
                    # 5 elems: (features, globals, value, policy, num_players) OR (... , heuristics)
                    # 6 elems: (features, globals, value, policy, num_players, heuristics)
                    batch_len = len(batch_data) if isinstance(batch_data, (list, tuple)) else 0
                    if batch_len == 6:
                        # Full: with num_players and heuristics
                        (
                            features,
                            globals_vec,
                            value_targets,
                            policy_targets,
                            batch_num_players,
                            batch_heuristics,
                        ) = batch_data
                    elif batch_len == 5:
                        # Check if 5th element is num_players (int/long tensor) or heuristics (float)
                        fifth_elem = batch_data[4]
                        if fifth_elem.dtype in (torch.int64, torch.int32, torch.long):
                            (
                                features,
                                globals_vec,
                                value_targets,
                                policy_targets,
                                batch_num_players,
                            ) = batch_data
                        else:
                            # Heuristics without num_players
                            (
                                features,
                                globals_vec,
                                value_targets,
                                policy_targets,
                                batch_heuristics,
                            ) = batch_data
                    else:
                        (
                            features,
                            globals_vec,
                            value_targets,
                            policy_targets,
                        ) = batch_data

                # Data quality metrics (every 500 batches to minimize GPU sync overhead)
                if i % 500 == 0 and i > 0:
                    # Value target distribution: check for P1/P2 balance
                    # Positive values typically indicate P1 advantage, negative P2
                    if value_targets.dim() == 1:
                        mean_val = value_targets.mean().item()
                        pos_ratio = (value_targets > 0).float().mean().item()
                        if abs(mean_val) > 0.15 or abs(pos_ratio - 0.5) > 0.15:
                            logger.debug(
                                f"Data quality: value_mean={mean_val:.3f}, "
                                f"positive_ratio={pos_ratio:.2%} (batch {i})"
                            )
                    # Policy entropy: measure diversity of targets
                    # Low entropy indicates concentrated/biased policy targets
                    policy_sums = policy_targets.sum(dim=1)
                    valid_policy = policy_sums > 0
                    if torch.any(valid_policy):
                        policy_probs = policy_targets[valid_policy] + 1e-8  # Avoid log(0)
                        policy_entropy = -(policy_probs * policy_probs.log()).sum(dim=1).mean().item()
                        if policy_entropy < 1.0:  # Very low entropy indicates potential issue
                            logger.debug(
                                f"Data quality: low policy entropy={policy_entropy:.3f} (batch {i})"
                            )

                # Transfer to device if not already there (prefetch may have done this)
                if features.device != device:
                    features = features.to(device, non_blocking=True)
                    globals_vec = globals_vec.to(device, non_blocking=True)
                    value_targets = value_targets.to(device, non_blocking=True)
                    policy_targets = policy_targets.to(device, non_blocking=True)
                if batch_num_players is not None and batch_num_players.device != device:
                    batch_num_players = batch_num_players.to(device, non_blocking=True)
                if batch_heuristics is not None and batch_heuristics.device != device:
                    batch_heuristics = batch_heuristics.to(device, non_blocking=True)

                # Hot data buffer mixing: replace portion of batch with hot buffer samples (2025-12)
                if hot_buffer is not None and hot_buffer.total_samples >= config.batch_size:
                    try:
                        # Compute how many samples to replace
                        n_hot = int(features.size(0) * hot_buffer_mix_ratio)
                        if n_hot > 0:
                            # Get samples from hot buffer
                            hot_board, hot_global, hot_policy, hot_value = hot_buffer.get_training_batch(
                                batch_size=n_hot, shuffle=True
                            )
                            if len(hot_board) > 0:
                                # Convert to tensors and transfer to device
                                hot_board_t = torch.from_numpy(hot_board).to(device, non_blocking=True)
                                hot_global_t = torch.from_numpy(hot_global).to(device, non_blocking=True)
                                hot_policy_t = torch.from_numpy(hot_policy).to(device, non_blocking=True)
                                hot_value_t = torch.from_numpy(hot_value).to(device, non_blocking=True)

                                # Replace last n_hot samples in the batch with hot buffer samples
                                actual_n_hot = min(n_hot, len(hot_board_t), features.size(0))
                                if actual_n_hot > 0:
                                    features[-actual_n_hot:] = hot_board_t[:actual_n_hot]
                                    globals_vec[-actual_n_hot:] = hot_global_t[:actual_n_hot]
                                    policy_targets[-actual_n_hot:] = hot_policy_t[:actual_n_hot]
                                    # Handle scalar vs vector value targets
                                    if value_targets.dim() == 1:
                                        value_targets[-actual_n_hot:] = hot_value_t[:actual_n_hot]
                                    else:
                                        # Vector values - broadcast hot buffer scalar to first element
                                        value_targets[-actual_n_hot:, 0] = hot_value_t[:actual_n_hot]
                    except (RuntimeError, ValueError, IndexError, AttributeError) as e:
                        # Tensor operation errors, invalid values, index errors, or missing attributes
                        # Don't fail training on hot buffer errors
                        if i % 100 == 0:
                            logger.debug(f"Hot buffer mixing skipped: {e}")

                # Data augmentation: apply random symmetry transforms (2025-12)
                if enhancements_manager is not None and enhancements_manager._augmentor is not None:
                    try:
                        features, policy_targets = enhancements_manager.augment_batch_dense(
                            features, policy_targets
                        )
                    except (RuntimeError, ValueError, AttributeError) as e:
                        # Tensor operation errors, invalid values, or missing attributes
                        # Don't fail training on augmentation errors
                        if i % 100 == 0:
                            logger.debug(f"Data augmentation skipped: {e}")

                # Pad policy targets if smaller than model policy_size (e.g., dataset
                # was generated with a smaller policy space than the model supports)
                if hasattr(model, 'policy_size') and policy_targets.size(1) < model.policy_size:
                    pad_size = model.policy_size - policy_targets.size(1)
                    policy_targets = torch.nn.functional.pad(
                        policy_targets, (0, pad_size), value=0.0
                    )

                policy_valid_mask = policy_targets.sum(dim=1) > 0

                # Phase 1 Diagnostics: Validate policy target normalization
                if torch.any(policy_valid_mask):
                    target_sums = policy_targets[policy_valid_mask].sum(dim=1)
                    if not torch.allclose(target_sums, torch.ones_like(target_sums), atol=1e-4):
                        bad_sums = target_sums[~torch.isclose(target_sums, torch.ones_like(target_sums), atol=1e-4)]
                        logger.error(
                            f"Policy targets not normalized at batch {i}! "
                            f"Expected sum=1.0, got: min={target_sums.min():.6f}, "
                            f"max={target_sums.max():.6f}, "
                            f"num_bad={len(bad_sums)}/{len(target_sums)}"
                        )
                        if target_sums.min() < 0.5 or target_sums.max() > 1.5:
                            raise ValueError(
                                f"Policy targets severely denormalized at batch {i}. "
                                f"Check data export pipeline."
                            )

                # Apply label smoothing to policy targets if configured
                # smoothed = (1 - eps) * target + eps * uniform
                # IMPORTANT: For V3/V4 spatial policy heads, only smooth over positions
                # where the original target > 0. This prevents adding probability mass
                # to invalid hex corners (indices 0-11 for hex8) that the model
                # correctly assigns -1e9 logits to via scatter initialization.
                if config.policy_label_smoothing > 0 and torch.any(policy_valid_mask):
                    eps = config.policy_label_smoothing
                    policy_targets = policy_targets.clone()

                    # Create mask of valid action positions (non-zero in original targets)
                    action_mask = policy_targets > 0  # [B, policy_size]

                    # Count valid actions per sample for proper uniform distribution
                    num_valid_per_sample = action_mask.float().sum(dim=1, keepdim=True).clamp(min=1)

                    # Create per-sample uniform distribution over valid actions only
                    # For positions where target=0, uniform stays 0 (preserves zeros)
                    uniform_over_valid = action_mask.float() / num_valid_per_sample

                    # Apply smoothing: (1-eps)*target + eps*uniform_over_valid
                    policy_targets = (1 - eps) * policy_targets + eps * uniform_over_valid

                # Gradient accumulation: only zero grad at start of accumulation window
                # Dynamic batch scheduling: calculate accumulation steps from batch scheduler
                base_accumulation = getattr(config, 'gradient_accumulation_steps', 1)
                if enhancements_manager is not None and enhancements_manager._batch_scheduler is not None:
                    # Get target batch size from scheduler
                    target_batch_size = enhancements_manager.get_batch_size()
                    actual_batch_size = config.batch_size
                    # Calculate accumulation steps to achieve target effective batch size
                    # accumulation_steps = target / actual (minimum 1)
                    scheduler_accumulation = max(1, target_batch_size // actual_batch_size)
                    accumulation_steps = max(base_accumulation, scheduler_accumulation)
                else:
                    accumulation_steps = base_accumulation
                if i % accumulation_steps == 0:
                    optimizer.zero_grad()

                # Autocast for mixed precision (CUDA only for now).
                with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=amp_torch_dtype):
                    # Check if auxiliary tasks are enabled and model supports return_features
                    use_aux_tasks = (
                        enhancements_manager is not None
                        and enhancements_manager.config.auxiliary_tasks_enabled
                        and enhancements_manager._auxiliary_module is not None
                    )

                    # V5 models accept heuristics parameter
                    model_accepts_heuristics = model_version in ('v5', 'v5-gnn', 'v5-heavy')

                    # Forward pass with optional backbone feature extraction
                    if use_aux_tasks:
                        try:
                            # Jan 10, 2026: Try return_features, fall back if legacy model
                            if model_accepts_heuristics:
                                out = model(features, globals_vec, heuristics=batch_heuristics, return_features=True)
                            else:
                                out = model(features, globals_vec, return_features=True)
                        except TypeError as e:
                            # Legacy checkpoints don't support return_features parameter
                            if "return_features" in str(e):
                                logger.warning(
                                    "Model doesn't support return_features - disabling aux tasks"
                                )
                                use_aux_tasks = False
                                if model_accepts_heuristics:
                                    out = model(features, globals_vec, heuristics=batch_heuristics)
                                else:
                                    out = model(features, globals_vec)
                            else:
                                raise
                        # V3+ models with features return (values, policy, rank_dist, features)
                        if use_aux_tasks and isinstance(out, tuple) and len(out) == 4:
                            value_pred, policy_pred, rank_dist_pred, backbone_features = out
                        elif use_aux_tasks and isinstance(out, tuple) and len(out) == 3:
                            # V2 models with features return (values, policy, features)
                            value_pred, policy_pred, backbone_features = out
                            rank_dist_pred = None
                        else:
                            # Fallback: model doesn't support return_features or aux disabled
                            if isinstance(out, tuple) and len(out) >= 3:
                                value_pred, policy_pred, rank_dist_pred = out[:3]
                            else:
                                value_pred, policy_pred = out[:2]
                                rank_dist_pred = None
                            backbone_features = None
                            use_aux_tasks = False
                    else:
                        if model_accepts_heuristics:
                            out = model(features, globals_vec, heuristics=batch_heuristics)
                        else:
                            out = model(features, globals_vec)
                        # V3 models return (values, policy_logits, rank_dist). We
                        # ignore the rank distribution for v1/v2 training losses.
                        if isinstance(out, tuple) and len(out) == 3:
                            value_pred, policy_pred, rank_dist_pred = out
                        else:
                            value_pred, policy_pred = out
                            rank_dist_pred = None
                        backbone_features = None

                    # Phase 1 Diagnostics: Detect numerical issues in policy predictions
                    if torch.any(torch.isnan(policy_pred)) or torch.any(torch.isinf(policy_pred)):
                        nan_count = torch.isnan(policy_pred).sum().item()
                        inf_count = torch.isinf(policy_pred).sum().item()
                        logger.error(
                            f"NaN/Inf detected in policy_pred! "
                            f"NaNs: {nan_count}, Infs: {inf_count}, "
                            f"Range: [{policy_pred[~torch.isnan(policy_pred)].min():.2e}, "
                            f"{policy_pred[~torch.isnan(policy_pred)].max():.2e}]"
                        )
                        raise ValueError(
                            f"Model produced NaN/Inf in policy predictions at batch {i}. "
                            f"Check model weights and learning rate."
                        )

                    # Check for extreme logits, excluding intentional -1e9 masking for invalid hex cells
                    valid_logits_mask = policy_pred > -1e8  # -1e9 is intentional masking
                    if torch.any(valid_logits_mask):
                        valid_logits = policy_pred[valid_logits_mask]
                        policy_pred_max = valid_logits.abs().max().item()
                        if policy_pred_max > 1e6:
                            logger.warning(
                                f"Extreme policy logits detected at batch {i}: "
                                f"max_abs={policy_pred_max:.2e}, "
                                f"valid_range=[{valid_logits.min():.2e}, {valid_logits.max():.2e}]"
                            )

                    # Apply stable log_softmax to policy prediction for KLDivLoss.
                    # Spatial policy heads keep the valid-action mask semantics while
                    # all heads clamp extreme logits before normalization.
                    if detect_masked_policy_output(policy_pred):
                        # Valid positions are either: (1) target distribution > 0, or
                        # (2) model logits > -1e3 (not masked by spatial scatter)
                        valid_mask = (policy_targets > 0) | (policy_pred > -1e3)
                        policy_log_probs = stable_policy_log_softmax(
                            policy_pred,
                            valid_mask,
                        )
                    else:
                        policy_log_probs = stable_policy_log_softmax(policy_pred)

                    # Use multi-player value loss for vector value targets
                    if use_multi_player_loss:
                        # Use per-sample num_players from batch if available,
                        # otherwise fall back to the fixed num_players argument
                        effective_num_players = (
                            batch_num_players if batch_num_players is not None
                            else num_players
                        )
                        value_loss = multi_player_value_loss(
                            value_pred, value_targets, effective_num_players
                        )
                    else:
                        # Scalar training uses only the first value head,
                        # matching NeuralNetAI.evaluate_batch behaviour.
                        if value_pred.ndim == 2:
                            value_pred_scalar = value_pred[:, 0]
                        else:
                            value_pred_scalar = value_pred
                        value_loss = value_criterion(
                            value_pred_scalar.reshape(-1),
                            value_targets.reshape(-1),
                        )

                    policy_loss = masked_policy_kl(
                        policy_log_probs,
                        policy_targets,
                    )

                    # Outcome-weighted policy loss (2025-12)
                    # Weight policy loss by game outcome: winner's moves get higher weight
                    # This focuses learning on moves that lead to winning outcomes
                    if enable_outcome_weighted_policy and outcome_weight_scale > 0:
                        # Compute per-sample outcome weights from value targets
                        # value_targets > 0 → winning position → weight > 1
                        # value_targets < 0 → losing position → weight < 1
                        with torch.no_grad():
                            if value_targets.ndim == 2:
                                # Multi-player: use mean value per sample
                                outcome_signal = value_targets.mean(dim=1)
                            else:
                                outcome_signal = value_targets.reshape(-1)

                            # Compute weights: 1 + outcome_weight_scale * sign(outcome)
                            # Winners: 1 + scale, Losers: 1 - scale
                            outcome_weights = 1.0 + outcome_weight_scale * outcome_signal.sign()
                            outcome_weights = outcome_weights.clamp(min=0.1)  # Prevent zero/negative weights

                        # Compute per-sample policy loss and apply weights
                        # NOTE: Use torch.where to avoid 0 * -inf = NaN when policy_log_probs
                        # has -inf values from masked_log_softmax (V3/V4 spatial policy heads)
                        per_sample_policy = -torch.where(
                            policy_targets > 0,
                            policy_targets * policy_log_probs,
                            torch.zeros_like(policy_log_probs)
                        ).sum(dim=1)
                        valid_mask = policy_targets.sum(dim=1) > 0
                        if valid_mask.any():
                            weighted_policy = (per_sample_policy[valid_mask] * outcome_weights[valid_mask]).mean()
                            policy_loss = weighted_policy

                    # Quality-weighted training (2025-12) - resurrected from ebmo_network.py
                    # Weights samples by MCTS visit counts to focus on high-quality moves
                    quality_ranking_loss = torch.tensor(0.0, device=device)
                    if quality_trainer is not None:
                        # Use policy targets as quality proxy (MCTS visit-derived probabilities)
                        # Higher entropy in targets = less certain position = lower quality
                        with torch.no_grad():
                            target_entropy = -(policy_targets * (policy_targets + 1e-8).log()).sum(dim=1)
                            # Invert: low entropy = high quality
                            quality_scores = 1.0 / (1.0 + target_entropy)
                            # Normalize to [0, 1]
                            quality_scores = (quality_scores - quality_scores.min()) / (
                                quality_scores.max() - quality_scores.min() + 1e-8
                            )

                        # Compute ranking loss to enforce quality ordering
                        if quality_trainer.ranking_weight > 0:
                            quality_ranking_loss = ranking_loss_from_quality(
                                policy_log_probs,
                                quality_scores,
                                margin=quality_trainer.ranking_margin,
                            )
                            quality_trainer.quality_stats["ranking_loss"] = quality_ranking_loss.item()

                        # January 2026 Sprint 10: Apply quality weights to per-sample losses
                        # Higher quality samples (sharper policy targets) contribute more to loss
                        # Expected improvement: +25-40 Elo by focusing learning on decisive positions
                        if quality_trainer.quality_weight > 0:
                            # Compute quality weights with minimum floor
                            quality_weights = torch.clamp(quality_scores, min=quality_trainer.min_quality_weight)
                            # Normalize to mean 1.0 (preserves effective batch size)
                            quality_weights = quality_weights / quality_weights.mean()
                            # Blend with uniform weights
                            blend = quality_trainer.quality_weight
                            uniform_weights = torch.ones_like(quality_weights)
                            final_weights = blend * quality_weights + (1.0 - blend) * uniform_weights

                            # Apply to policy loss (per-sample then weighted mean)
                            # NOTE: Use torch.where to avoid 0 * -inf = NaN when policy_log_probs
                            # has -inf values from masked_log_softmax (V3/V4 spatial policy heads)
                            per_sample_policy_loss = -torch.where(
                                policy_targets > 0,
                                policy_targets * policy_log_probs,
                                torch.zeros_like(policy_log_probs)
                            ).sum(dim=1)
                            valid_mask = policy_targets.sum(dim=1) > 0
                            if valid_mask.any():
                                policy_loss = (per_sample_policy_loss[valid_mask] * final_weights[valid_mask]).mean()

                            # Apply to value loss (per-sample then weighted mean)
                            if value_pred.ndim == 2:
                                per_sample_value_loss = ((value_pred - value_targets) ** 2).mean(dim=1)
                            else:
                                per_sample_value_loss = (value_pred_scalar.reshape(-1) - value_targets.reshape(-1)) ** 2
                            value_loss = (per_sample_value_loss * final_weights).mean()

                            # Track statistics
                            quality_trainer.quality_stats["mean_weight"] = final_weights.mean().item()
                            quality_trainer.quality_stats["std_weight"] = final_weights.std().item()

                    # Entropy regularization to prevent policy collapse
                    # H(p) = -sum(p * log(p)); higher entropy = more exploration
                    # We add -entropy_weight * H to encourage exploration
                    entropy_bonus = torch.tensor(0.0, device=device)
                    if config.entropy_weight > 0:
                        policy_probs = policy_log_probs.exp()
                        # Entropy: -sum(p * log(p)), clamping log for numerical stability
                        policy_entropy = -(policy_probs * policy_log_probs.clamp(min=-20)).sum(dim=1).mean()
                        # Subtract entropy (maximize entropy = minimize negative entropy)
                        entropy_bonus = -config.entropy_weight * policy_entropy

                    # Collect individual losses for gradient surgery
                    # Jan 2026: Apply value_weight to balance value vs policy learning
                    task_losses: dict[str, torch.Tensor] = {
                        "value": config.value_weight * value_loss,
                        "policy": config.policy_weight * policy_loss + entropy_bonus,
                    }

                    # Rank distribution loss (V3+ multi-player head)
                    rank_loss = None
                    if (
                        rank_dist_pred is not None
                        and use_multi_player_loss
                        and value_targets.ndim == 2
                    ):
                        rank_targets, rank_mask = build_rank_targets(
                            value_targets,
                            effective_num_players,
                            output_players=int(rank_dist_pred.shape[1]),
                        )
                        rank_log_probs = torch.log(
                            rank_dist_pred.clamp_min(1e-8)
                        )
                        per_player_loss = -(
                            rank_targets * rank_log_probs
                        ).sum(dim=-1)
                        if torch.any(rank_mask):
                            rank_loss = per_player_loss[rank_mask].mean()
                            task_losses["rank"] = config.rank_dist_weight * rank_loss

                    # Add quality ranking loss if enabled (2025-12)
                    if quality_trainer is not None and quality_ranking_loss.item() > 0:
                        task_losses["quality_ranking"] = quality_trainer.ranking_weight * quality_ranking_loss

                    # Auxiliary task loss (outcome prediction from value targets)
                    aux_loss = None
                    if use_aux_tasks and backbone_features is not None:
                        # Derive outcome class from value targets:
                        # value > 0.3 → Win (2), value < -0.3 → Loss (0), else Draw (1)
                        # For multi-player games, use mean value per sample (not per player)
                        if value_targets.dim() == 2:
                            # Multi-player: value_targets is (batch, num_players)
                            value_flat = value_targets.mean(dim=1)
                        else:
                            value_flat = value_targets.reshape(-1)
                        outcome_targets = torch.where(
                            value_flat > 0.3,
                            torch.tensor(2, device=device, dtype=torch.long),
                            torch.where(
                                value_flat < -0.3,
                                torch.tensor(0, device=device, dtype=torch.long),
                                torch.tensor(1, device=device, dtype=torch.long),
                            ),
                        )
                        aux_targets = {"outcome": outcome_targets}
                        aux_loss, _aux_breakdown = enhancements_manager.compute_auxiliary_loss(
                            backbone_features, aux_targets
                        )
                        task_losses["aux"] = aux_loss

                    # Compute combined loss for metrics (always needed)
                    loss = sum(task_losses.values())

                    # Training facade: per-sample loss, hard example mining, weighted loss (2025-12)
                    # Uses unified facade when available for +80-165 Elo improvement
                    if training_facade is not None:
                        try:
                            # Create batch indices: batch_idx * batch_size + sample_idx
                            batch_size = features.size(0)
                            batch_indices = torch.arange(
                                i * config.batch_size,
                                i * config.batch_size + batch_size,
                                device=device,
                            )

                            # Compute per-sample losses for mining
                            with torch.no_grad():
                                per_sample_losses = training_facade.compute_per_sample_loss(
                                    policy_logits=policy_pred,
                                    policy_targets=policy_targets,
                                    value_pred=value_pred[:, 0] if value_pred.ndim == 2 else value_pred,
                                    value_targets=value_targets[:, 0] if value_targets.ndim == 2 else value_targets,
                                    reduction="none",
                                )

                                # Compute uncertainty from policy entropy
                                policy_probs = torch.softmax(policy_pred, dim=1)
                                policy_entropy = -(policy_probs * (policy_probs + 1e-8).log()).sum(dim=1)

                            # Record batch and get weighted loss
                            training_facade.record_batch(
                                batch_indices=batch_indices,
                                per_sample_losses=per_sample_losses,
                                uncertainties=policy_entropy,
                            )

                            # Apply hard example weighting to loss (upweights difficult samples)
                            # This focuses training on samples the model struggles with
                            if training_facade.is_mining_active:
                                loss = training_facade.get_weighted_loss(
                                    per_sample_losses=per_sample_losses,
                                    batch_indices=batch_indices,
                                )
                                # Add auxiliary losses back (they use original weighting)
                                for key in ['aux', 'rank']:
                                    if key in task_losses:
                                        loss = loss + task_losses[key]
                        except (RuntimeError, ValueError) as e:
                            # Don't fail training on facade errors
                            if i % 500 == 0:
                                logger.debug(f"[Training Facade] Batch {i} skipped: {e}")

                    # Fallback: standalone hard example miner (2025-12)
                    # Used when training facade is not available
                    elif hard_example_miner is not None and compute_per_sample_loss is not None:
                        try:
                            # Compute per-sample losses (no reduction)
                            with torch.no_grad():
                                per_sample_losses = compute_per_sample_loss(
                                    policy_logits=policy_pred,
                                    policy_targets=policy_targets,
                                    value_pred=value_pred[:, 0] if value_pred.ndim == 2 else value_pred,
                                    value_targets=value_targets[:, 0] if value_targets.ndim == 2 else value_targets,
                                    policy_weight=config.policy_weight,
                                    reduction="none",
                                )

                                # Compute uncertainty from policy entropy (higher entropy = more uncertain)
                                policy_probs = torch.softmax(policy_pred, dim=1)
                                policy_entropy = -(policy_probs * (policy_probs + 1e-8).log()).sum(dim=1)

                                # Create batch indices: batch_idx * batch_size + sample_idx
                                batch_size = features.size(0)
                                batch_indices = torch.arange(
                                    i * config.batch_size,
                                    i * config.batch_size + batch_size,
                                    device=device,
                                )

                                # Record to miner
                                hard_example_miner.record_batch(
                                    indices=batch_indices,
                                    losses=per_sample_losses,
                                    uncertainties=policy_entropy,
                                )
                        except (RuntimeError, ValueError) as e:
                            # Don't fail training on mining errors
                            if i % 500 == 0:
                                logger.debug(f"[Hard Example Mining] Batch {i} skipped: {e}")

                    # Scale loss for gradient accumulation to maintain gradient magnitude
                    if accumulation_steps > 1:
                        loss = loss / accumulation_steps
                        # Also scale individual losses for gradient surgery
                        task_losses = {k: v / accumulation_steps for k, v in task_losses.items()}

                # Circuit breaker protection for backward pass (2025-12)
                # Catches CUDA errors, OOM, and other runtime exceptions
                try:
                    if use_gradient_surgery and gradient_surgeon is not None:
                        # Use gradient surgery to project conflicting gradients
                        # Note: apply_surgery handles model.zero_grad and sets gradients
                        gradient_surgeon.apply_surgery(model, task_losses)
                    elif use_grad_scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    error_msg = str(e).lower()
                    is_cuda_error = (
                        'cuda' in error_msg or 'out of memory' in error_msg or
                        'cublas' in error_msg or 'cudnn' in error_msg
                    )
                    if is_cuda_error:
                        logger.warning(f"CUDA error in batch {i}: {e}")
                        if training_breaker:
                            training_breaker.record_failure("batch_processing", e)
                        # Clear gradients and memory
                        optimizer.zero_grad(set_to_none=True)
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue  # Skip to next batch
                    else:
                        raise  # Re-raise non-CUDA errors

                # Only step optimizer after accumulating gradients
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_data_iter):
                    # Gradient clipping (adaptive or fixed) (2025-12)
                    if use_grad_scaler:
                        scaler.unscale_(optimizer)
                    if adaptive_clipper is not None:
                        grad_norm = adaptive_clipper.update_and_clip(model.parameters())
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=fixed_clip_norm,
                        )

                    # Circuit breaker protection for optimizer step (2025-12)
                    try:
                        if use_grad_scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        # Record successful batch processing
                        if training_breaker:
                            training_breaker.record_success("batch_processing")
                    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                        error_msg = str(e).lower()
                        is_cuda_error = (
                            'cuda' in error_msg or 'out of memory' in error_msg or
                            'cublas' in error_msg or 'cudnn' in error_msg
                        )
                        if is_cuda_error:
                            logger.warning(f"CUDA error in optimizer step at batch {i}: {e}")
                            if training_breaker:
                                training_breaker.record_failure("batch_processing", e)
                            optimizer.zero_grad(set_to_none=True)
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()
                            continue
                        else:
                            raise

                    # Update gradient metrics (every 100 batches to minimize overhead)
                    if i % 100 == 0 and HAS_PROMETHEUS and (not distributed or is_main_process()):
                        if GRADIENT_NORM:
                            GRADIENT_NORM.labels(config=config_label).set(
                                grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm
                            )
                        if adaptive_clipper is not None and GRADIENT_CLIP_NORM:
                            GRADIENT_CLIP_NORM.labels(config=config_label).set(
                                adaptive_clipper.current_max_norm
                            )

                    # Update integrated enhancements step counter
                    if enhancements_manager is not None:
                        enhancements_manager.update_step()

                        # Check if reanalysis should be triggered (2025-12)
                        # Reanalyzes historical data with current model for improved targets
                        if enhancements_manager.should_reanalyze():
                            if not distributed or is_main_process():
                                logger.info(
                                    "[Reanalysis] Triggering MuZero-style reanalysis of training data"
                                )
                                reanalyzed_path = enhancements_manager.process_reanalysis(
                                    data_path_str if data_path_str else None
                                )
                                if reanalyzed_path:
                                    logger.info(f"[Reanalysis] Complete: {reanalyzed_path}")
                                    # Note: Reanalyzed data is saved for next training run
                                    # or can be loaded via ReanalyzedDataset for mixing

                # Anomaly detection: check for NaN/Inf in loss (2025-12)
                if anomaly_detector is not None:
                    loss_val = loss.detach().item()
                    anomaly_step += 1
                    if anomaly_detector.check_loss(loss_val, anomaly_step):
                        anomaly_summary = anomaly_detector.get_summary()
                        consecutive = anomaly_summary.get('consecutive_anomalies', 0)
                        # Dec 29, 2025: Detect NaN/Inf explicitly for event emission
                        is_nan = loss_val != loss_val  # NaN != NaN is True
                        is_inf = not is_nan and (loss_val == float('inf') or loss_val == float('-inf'))
                        anomaly_type = 'nan' if is_nan else ('inf' if is_inf else 'spike')
                        logger.warning(
                            f"Training anomaly detected at batch {i}: type={anomaly_type}, "
                            f"total={anomaly_summary.get('total_anomalies', 0)}, "
                            f"consecutive={consecutive}"
                        )
                        # Update Prometheus anomaly counter
                        if HAS_PROMETHEUS and ANOMALY_DETECTIONS and (not distributed or is_main_process()):
                            ANOMALY_DETECTIONS.labels(config=config_label, type=anomaly_type).inc()

                        # Dec 29, 2025: Emit event for batch-level NaN/Inf (critical anomalies)
                        if (is_nan or is_inf) and HAS_TRAINING_EVENTS and (not distributed or is_main_process()):
                            try:
                                import asyncio
                                config_key = f"{config.board_type.value}_{num_players}p"
                                loop = asyncio.get_running_loop()
                                asyncio.ensure_future(emit_training_loss_anomaly(
                                    config_key=config_key,
                                    current_loss=0.0 if is_nan else loss_val,
                                    avg_loss=0.0,
                                    epoch=epoch + 1,
                                    anomaly_ratio=float('inf'),
                                    source="train.py",
                                    anomaly_type=anomaly_type,
                                    batch=i,
                                ))
                            except RuntimeError:
                                pass  # No event loop - OK in non-async context

                        # Auto-reduce learning rate on repeated anomalies (2025-12)
                        # Reduce by 30% after 3 consecutive anomalies (before circuit breaker)
                        if consecutive >= 3 and consecutive % 3 == 0:
                            for param_group in optimizer.param_groups:
                                old_lr = param_group['lr']
                                new_lr = old_lr * 0.7
                                param_group['lr'] = new_lr
                                if not distributed or is_main_process():
                                    logger.warning(
                                        f"Auto-reduced LR due to {consecutive} consecutive anomalies: "
                                        f"{old_lr:.2e} -> {new_lr:.2e}"
                                    )

                        # Record failure with circuit breaker
                        if training_breaker:
                            training_breaker.record_failure("training_epoch")
                        # Skip this batch to avoid corrupting gradients
                        optimizer.zero_grad()
                        continue

                # Accumulate loss without .item() to avoid GPU sync per batch
                # Detach to prevent gradient accumulation, but keep on GPU
                train_loss += loss.detach()
                train_batches += 1

                # Track metrics for distributed reduction (uses detached tensor)
                if dist_metrics is not None:
                    dist_metrics.add(
                        'train_loss',
                        loss.detach(),
                        features.size(0),
                    )

                # Logging every 50 batches - reduced from 10 to minimize GPU sync overhead
                if i % 50 == 0 and (not distributed or is_main_process()):
                    # Only call .item() for logging, not accumulation
                    logger.info(
                        f"Epoch {epoch+1}, Batch {i}: "
                        f"Loss={loss.detach().item():.4f} "
                        f"(Val={value_loss.detach().item():.4f}, "
                        f"Pol={policy_loss.detach().item():.4f})"
                    )

            # Compute average training loss - call .item() only at end of epoch
            if distributed and dist_metrics is not None:
                # Synchronize metrics across all processes
                train_metrics = dist_metrics.reduce_and_reset(device=device)
                avg_train_loss = train_metrics.get('train_loss', 0.0)
            elif train_batches > 0:
                # Single .item() call at end of epoch for accumulated tensor
                avg_train_loss = (train_loss / train_batches).item()
            else:
                avg_train_loss = 0.0

            # Validation
            model.eval()
            val_loss = torch.tensor(0.0, device=device)  # Accumulate on GPU
            val_batches = 0
            val_policy_correct = 0  # Policy accuracy tracking
            val_policy_total = 0
            if dist_metrics is not None:
                dist_metrics.reset()

            # Select appropriate validation data source
            # For multi-player mode with streaming, use iter_with_mp()
            use_val_mp_iter = use_multi_player_loss and use_streaming and val_streaming_loader.has_multi_player_values
            if use_streaming:
                assert val_streaming_loader is not None
                # Use prefetch_loader for background prefetching if enabled
                if use_prefetch:
                    val_data_iter = prefetch_loader(
                        val_streaming_loader,
                        prefetch_count=prefetch_count,
                        pin_memory=pin_memory,
                        use_mp=use_val_mp_iter,
                        transfer_to_device=device if prefetch_to_device else None,
                    )
                elif use_val_mp_iter:
                    val_data_iter = val_streaming_loader.iter_with_mp()
                else:
                    val_data_iter = iter(val_streaming_loader)
                # Limit validation to ~20% of batches for streaming
                max_val_batches = max(
                    1,
                    len(val_streaming_loader) // 5,
                )
            else:
                assert val_loader is not None
                val_data_iter = iter(val_loader)
                max_val_batches = float('inf')

            with torch.no_grad():
                for val_batch_idx, val_batch in enumerate(val_data_iter):
                    if val_batch_idx >= max_val_batches:
                        break

                    # Handle streaming, streaming with multi-player, and legacy batch formats
                    val_batch_num_players = None
                    val_batch_heuristics = None
                    if use_streaming:
                        if use_multi_player_loss and val_streaming_loader.has_multi_player_values:
                            (
                                (features, globals_vec),
                                (value_targets, policy_targets),
                                values_mp_batch,
                                val_batch_num_players,
                            ) = val_batch
                            if values_mp_batch is not None:
                                value_targets = values_mp_batch
                        else:
                            (
                                (features, globals_vec),
                                (value_targets, policy_targets),
                            ) = val_batch
                    else:
                        # Non-streaming: batch structure varies based on dataset config
                        val_batch_len = len(val_batch) if isinstance(val_batch, (list, tuple)) else 0
                        if val_batch_len == 6:
                            (
                                features,
                                globals_vec,
                                value_targets,
                                policy_targets,
                                val_batch_num_players,
                                val_batch_heuristics,
                            ) = val_batch
                        elif val_batch_len == 5:
                            fifth_elem = val_batch[4]
                            if fifth_elem.dtype in (torch.int64, torch.int32, torch.long):
                                (
                                    features,
                                    globals_vec,
                                    value_targets,
                                    policy_targets,
                                    val_batch_num_players,
                                ) = val_batch
                            else:
                                (
                                    features,
                                    globals_vec,
                                    value_targets,
                                    policy_targets,
                                    val_batch_heuristics,
                                ) = val_batch
                        else:
                            (
                                features,
                                globals_vec,
                                value_targets,
                                policy_targets,
                            ) = val_batch

                    # Transfer to device if not already there (prefetch may have done this)
                    if features.device != device:
                        features = features.to(device, non_blocking=True)
                        globals_vec = globals_vec.to(device, non_blocking=True)
                        value_targets = value_targets.to(device, non_blocking=True)
                        policy_targets = policy_targets.to(device, non_blocking=True)
                    if val_batch_num_players is not None and val_batch_num_players.device != device:
                        val_batch_num_players = val_batch_num_players.to(device, non_blocking=True)
                    if val_batch_heuristics is not None and val_batch_heuristics.device != device:
                        val_batch_heuristics = val_batch_heuristics.to(device, non_blocking=True)

                    # Pad policy targets if smaller than model policy_size
                    if hasattr(model, 'policy_size') and policy_targets.size(1) < model.policy_size:
                        pad_size = model.policy_size - policy_targets.size(1)
                        policy_targets = torch.nn.functional.pad(
                            policy_targets, (0, pad_size), value=0.0
                        )

                    # Autocast for mixed precision validation (matches training)
                    with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=amp_torch_dtype):
                        # For DDP, forward through the wrapped model
                        # V5 models accept heuristics parameter
                        if model_version in ('v5', 'v5-gnn', 'v5-heavy'):
                            out = model(features, globals_vec, heuristics=val_batch_heuristics)
                        else:
                            out = model(features, globals_vec)
                        if isinstance(out, tuple) and len(out) == 3:
                            value_pred, policy_pred, rank_dist_pred = out
                        else:
                            value_pred, policy_pred = out
                            rank_dist_pred = None

                        # Clamp extreme logits before log-softmax while preserving
                        # masked spatial-head normalization.
                        if detect_masked_policy_output(policy_pred):
                            valid_mask = (policy_targets > 0) | (policy_pred > -1e3)
                            policy_log_probs = stable_policy_log_softmax(
                                policy_pred,
                                valid_mask,
                            )
                        else:
                            policy_log_probs = stable_policy_log_softmax(policy_pred)

                        # Policy accuracy: compare predicted move vs target move
                        pred_move = policy_pred.argmax(dim=1)
                        target_move = policy_targets.argmax(dim=1)
                        val_policy_correct += (pred_move == target_move).sum().item()
                        val_policy_total += pred_move.size(0)

                        # Use multi-player value loss for validation too
                        if use_multi_player_loss:
                            effective_val_num_players = (
                                val_batch_num_players if val_batch_num_players is not None
                                else num_players
                            )
                            v_loss = multi_player_value_loss(
                                value_pred, value_targets, effective_val_num_players
                            )
                        else:
                            if value_pred.ndim == 2:
                                value_pred_scalar = value_pred[:, 0]
                            else:
                                value_pred_scalar = value_pred
                            v_loss = value_criterion(
                                value_pred_scalar.reshape(-1),
                                value_targets.reshape(-1),
                            )
                        p_loss = masked_policy_kl(
                            policy_log_probs, policy_targets
                        )
                        # Jan 2026: Apply value_weight for consistency with training
                        loss = (config.value_weight * v_loss) + (config.policy_weight * p_loss)

                        # Rank distribution loss (V3+ multi-player head)
                        if (
                            rank_dist_pred is not None
                            and use_multi_player_loss
                            and value_targets.ndim == 2
                        ):
                            rank_targets, rank_mask = build_rank_targets(
                                value_targets,
                                effective_val_num_players,
                                output_players=int(rank_dist_pred.shape[1]),
                            )
                            rank_log_probs = torch.log(
                                rank_dist_pred.clamp_min(1e-8)
                            )
                            per_player_loss = -(
                                rank_targets * rank_log_probs
                            ).sum(dim=-1)
                            if torch.any(rank_mask):
                                rank_loss = per_player_loss[rank_mask].mean()
                                loss = loss + (config.rank_dist_weight * rank_loss)
                    # Accumulate on GPU without .item() sync
                    val_loss += loss.detach()
                    val_batches += 1

                    # Track metrics for distributed reduction (detached tensor)
                    if dist_metrics is not None:
                        dist_metrics.add(
                            'val_loss', loss.detach(), features.size(0)
                        )

                    # Collect calibration samples (value predictions vs actual outcomes)
                    if calibration_tracker is not None and not use_multi_player_loss:
                        # Get scalar predictions and targets for calibration
                        preds_cpu = value_pred_scalar.detach().cpu().numpy().flatten()
                        targets_cpu = value_targets.detach().cpu().numpy().flatten()
                        # Sample subset to avoid too much overhead
                        sample_size = min(len(preds_cpu), 100)
                        for i in range(sample_size):
                            calibration_tracker.add_sample(
                                float(preds_cpu[i]),
                                float(targets_cpu[i])
                            )

            # Compute average validation loss - single .item() at end
            if distributed and dist_metrics is not None:
                val_metrics = dist_metrics.reduce_and_reset(device=device)
                avg_val_loss = val_metrics.get('val_loss', 0.0)
            elif val_batches > 0:
                avg_val_loss = (val_loss / val_batches).item()
            else:
                avg_val_loss = 0.0

            # Compute policy accuracy
            avg_policy_accuracy = (
                val_policy_correct / val_policy_total if val_policy_total > 0 else 0.0
            )

            # Update training state for emergency checkpoints (2025-12)
            training_state.epoch = epoch
            training_state.avg_val_loss = avg_val_loss
            if avg_val_loss < training_state.best_val_loss:
                training_state.best_val_loss = avg_val_loss

            # Update scheduler at end of epoch
            if epoch_scheduler is not None:
                epoch_scheduler.step()
            elif plateau_scheduler is not None:
                plateau_scheduler.step(avg_val_loss)

            # Apply curriculum LR scaling from training facade (December 2025)
            # Scales LR based on training progress: warmup → 1.0 → max_scale
            if training_facade is not None and training_facade.config.enable_curriculum_lr:
                try:
                    curriculum_scale = training_facade.get_curriculum_lr_scale()
                    if abs(curriculum_scale - 1.0) > 0.01:  # Only apply if meaningfully different
                        base_lr = optimizer.param_groups[0]['lr']
                        adjusted_lr = base_lr * curriculum_scale
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = adjusted_lr
                        if (epoch + 1) % 5 == 0:  # Log every 5 epochs
                            logger.debug(
                                f"[Curriculum LR] Epoch {epoch+1}: scale={curriculum_scale:.3f}, "
                                f"lr={adjusted_lr:.2e}"
                            )
                except (AttributeError, ValueError) as e:
                    logger.debug(f"[Curriculum LR] Failed to apply: {e}")

            # Apply evaluation feedback LR adjustment (December 2025)
            # This responds to EVALUATION_COMPLETED events and adjusts LR based on Elo trends
            if eval_feedback_handler is not None and eval_feedback_handler.should_adjust_lr():
                new_lr = eval_feedback_handler.apply_lr_adjustment(current_epoch=epoch)
                if new_lr is not None and (not distributed or is_main_process()):
                    logger.info(
                        f"[EvaluationFeedback] LR adjusted to {new_lr:.2e} based on Elo trend"
                    )

            # Always log current learning rate
            if not distributed or is_main_process():
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"  Current LR: {current_lr:.6f}")

            if not distributed or is_main_process():
                # Log epoch statistics with hot buffer info
                epoch_log = (
                    f"Epoch [{epoch+1}/{config.epochs_per_iter}], "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, "
                    f"Policy Acc: {avg_policy_accuracy:.1%}"
                )
                if hot_buffer is not None:
                    hot_stats = hot_buffer.get_statistics()
                    epoch_log += f", Hot Buffer: {hot_stats['game_count']}/{hot_stats['max_size']} games"
                logger.info(epoch_log)

                # Publish training progress event to EventBus (2025-12)
                if HAS_EVENT_BUS and get_router is not None:
                    try:
                        router = get_router()
                        event_payload = {
                            "epoch": epoch + 1,
                            "total_epochs": config.epochs_per_iter,
                            "train_loss": float(avg_train_loss),
                            "val_loss": float(avg_val_loss),
                            "policy_accuracy": float(avg_policy_accuracy),
                            "lr": float(optimizer.param_groups[0]['lr']),
                            "config": f"{config.board_type.value}_{num_players}p",
                        }
                        # Add hot buffer stats if available
                        if hot_buffer is not None:
                            event_payload["hot_buffer"] = hot_buffer.get_statistics()
                        router.publish_sync(DataEvent(
                            event_type=DataEventType.TRAINING_PROGRESS,
                            payload=event_payload,
                            source="train",
                        ))
                    except (RuntimeError, ConnectionError, TimeoutError) as e:
                        # Event emission can fail due to async runtime or network issues
                        logger.debug(f"Failed to publish training progress event: {e}")

            epoch_reporting = handle_epoch_reporting_and_feedback(
                epoch=epoch,
                avg_train_loss=avg_train_loss,
                avg_val_loss=avg_val_loss,
                avg_policy_accuracy=avg_policy_accuracy,
                optimizer=optimizer,
                best_val_loss=best_val_loss,
                config=config,
                num_players=num_players,
                distributed=distributed,
                is_main=not distributed or is_main_process(),
                device=device,
                calibration_tracker=calibration_tracker,
                epoch_losses=epoch_losses,
                loss_monitor=loss_monitor,
                training_facade=training_facade,
                hard_example_miner=hard_example_miner,
                metrics_collector=metrics_collector,
                has_regression_detector=HAS_REGRESSION_DETECTOR,
                get_regression_detector=get_regression_detector,
                regression_severity=RegressionSeverity,
                has_epoch_events=HAS_EPOCH_EVENTS,
                publish_epoch_completed=publish_epoch_completed,
                has_training_events=HAS_TRAINING_EVENTS,
                emit_training_loss_anomaly=emit_training_loss_anomaly,
                emit_training_loss_trend=emit_training_loss_trend,
                has_prometheus=HAS_PROMETHEUS,
                training_epochs_metric=TRAINING_EPOCHS,
                training_loss_metric=TRAINING_LOSS,
                training_duration_metric=TRAINING_DURATION,
                calibration_ece_metric=CALIBRATION_ECE,
                calibration_mce_metric=CALIBRATION_MCE,
            )
            skip_checkpoint_on_regression = epoch_reporting.skip_checkpoint_on_regression
            epochs_completed = epoch_reporting.epochs_completed
            epoch_record = epoch_reporting.epoch_record

            # Check early stopping (only on main process for DDP)
            # Get model for checkpointing (unwrap DDP if needed)
            model_to_save = cast(
                nn.Module,
                model.module if distributed else model,
            )

            # Check integrated enhancements early stopping (based on Elo tracking)
            if enhancements_manager is not None and enhancements_manager.should_early_stop():
                if not distributed or is_main_process():
                    logger.info(
                        f"Enhancements manager triggered early stop at epoch {epoch+1} "
                        "(Elo regression detected)"
                    )
                break

            # Check baseline gating - warn if model failing against basic baselines
            if enhancements_manager is not None:
                passes_gating, failed_baselines, consecutive_failures = (
                    enhancements_manager.get_baseline_gating_status()
                )
                if not passes_gating and (not distributed or is_main_process()):
                    logger.warning(
                        f"[BASELINE GATING] Epoch {epoch+1}: Model failed baseline thresholds "
                        f"({', '.join(failed_baselines)}). Consecutive failures: {consecutive_failures}"
                    )
                    if consecutive_failures >= 5:
                        logger.error(
                            f"[BASELINE GATING] {consecutive_failures} consecutive failures! "
                            "Model may be overfitting to neural-vs-neural play. "
                            "Consider: more diverse training data, regularization, or early stopping."
                        )

            if early_stopper is not None:
                # Get current Elo from enhancements manager if available
                current_elo = None
                if enhancements_manager is not None:
                    current_elo = enhancements_manager.get_current_elo()

                # Use should_stop() with Elo support instead of __call__
                should_stop = early_stopper.should_stop(
                    val_loss=avg_val_loss,
                    current_elo=current_elo,
                    model=model_to_save,
                    epoch=epoch,
                )
                # December 2025: Enforce minimum training epochs before early stopping
                # This prevents stopping at 3-5 epochs when 15-20+ are needed for 2000+ Elo
                if should_stop and epoch + 1 < MIN_TRAINING_EPOCHS:
                    if not distributed or is_main_process():
                        logger.info(
                            f"Early stopping suppressed at epoch {epoch+1} (minimum: {MIN_TRAINING_EPOCHS})"
                        )
                    should_stop = False
                if should_stop:
                    if not distributed or is_main_process():
                        elo_info = f", best Elo: {early_stopper.best_elo:.1f}" if early_stopper.best_elo > float('-inf') else ""
                        logger.info(
                            f"Early stopping triggered at epoch {epoch+1} "
                            f"(best loss: {early_stopper.best_loss:.4f}{elo_info})"
                        )

                        # Emit TRAINING_EARLY_STOPPED event (December 2025 - feedback loop)
                        # This triggers curriculum boost for this config
                        try:
                            import asyncio
                            from app.coordination.event_router import emit_training_early_stopped

                            config_key = f"{config.board_type}_{num_players}p"
                            best_elo = early_stopper.best_elo if early_stopper.best_elo > float('-inf') else None
                            epochs_without_improvement = early_stopper.counter if hasattr(early_stopper, 'counter') else 0

                            # Use fire-and-forget emit via event loop
                            try:
                                loop = asyncio.get_running_loop()
                                loop.create_task(emit_training_early_stopped(
                                    config_key=config_key,
                                    epoch=epoch + 1,
                                    best_loss=float(early_stopper.best_loss),
                                    final_loss=float(avg_val_loss),
                                    best_elo=best_elo,
                                    reason="loss_stagnation",
                                    epochs_without_improvement=epochs_without_improvement,
                                ))
                            except RuntimeError:
                                # No running loop - create one for sync emit
                                asyncio.run(emit_training_early_stopped(
                                    config_key=config_key,
                                    epoch=epoch + 1,
                                    best_loss=float(early_stopper.best_loss),
                                    final_loss=float(avg_val_loss),
                                    best_elo=best_elo,
                                    reason="loss_stagnation",
                                    epochs_without_improvement=epochs_without_improvement,
                                ))

                            logger.info(f"[train] Emitted TRAINING_EARLY_STOPPED for {config_key}")
                        except (RuntimeError, ConnectionError, TimeoutError) as e:
                            # Event emission can fail due to async runtime or network issues
                            logger.warning(f"Failed to emit TRAINING_EARLY_STOPPED: {e}")
                        # Restore best weights
                        early_stopper.restore_best_weights(model_to_save)
                        _final_checkpoint_path = save_early_stop_artifacts(
                            model_to_save=model_to_save,
                            optimizer=optimizer,
                            epoch=epoch,
                            checkpoint_dir=checkpoint_dir,
                            save_path=save_path,
                            config=config,
                            num_players=num_players,
                            early_stopper=early_stopper,
                            async_checkpointer=async_checkpointer,
                            epoch_scheduler=epoch_scheduler,
                        )
                    # Mark early stopping as successful completion (for hardened event emission)
                    _training_completed_normally = True
                    break

            # Checkpoint at intervals (only on main process)
            if (
                checkpoint_interval > 0
                and (epoch + 1) % checkpoint_interval == 0
            ) and (not distributed or is_main_process()):
                checkpoint_path = save_periodic_checkpoint(
                    model_to_save=model_to_save,
                    optimizer=optimizer,
                    epoch=epoch,
                    avg_val_loss=avg_val_loss,
                    checkpoint_dir=checkpoint_dir,
                    async_checkpointer=async_checkpointer,
                    epoch_scheduler=epoch_scheduler,
                    early_stopper=early_stopper,
                )
                # Track for circuit breaker rollback (2025-12)
                _last_good_checkpoint_path = checkpoint_path
                _last_good_epoch = epoch

            # Save best model (only on main process)
            # January 2026: Skip saving if significant regression detected
            if avg_val_loss < best_val_loss and not skip_checkpoint_on_regression:
                best_val_loss = avg_val_loss
                best_train_loss_at_best_val = avg_train_loss  # Track for overfitting detection
                if not distributed or is_main_process():
                    save_best_model_artifacts(
                        model_to_save=model_to_save,
                        save_path=save_path,
                        config=config,
                        num_players=num_players,
                        epoch=epoch,
                        train_size=train_size,
                        avg_val_loss=avg_val_loss,
                        avg_train_loss=avg_train_loss,
                        checkpoint_averager=checkpoint_averager,
                    )

            # Knowledge distillation check (2025-12)
            # Distills ensemble knowledge from best checkpoints into current model
            if enhancements_manager is not None:
                # Set checkpoint directory so distillation can find teacher models
                enhancements_manager.set_checkpoint_dir(checkpoint_dir)

                if enhancements_manager.should_distill(epoch + 1):
                    if not distributed or is_main_process():
                        logger.info(
                            f"[Distillation] Triggering ensemble distillation at epoch {epoch+1}"
                        )
                        # Use the training dataloader for distillation
                        distillation_success = enhancements_manager.run_distillation(
                            current_epoch=epoch + 1,
                            dataloader=train_loader,
                        )
                        if distillation_success:
                            logger.info(
                                f"[Distillation] Epoch {epoch+1}: Successfully distilled "
                                "ensemble knowledge into model"
                            )

            # Beat heartbeat at end of each epoch to signal health
            if heartbeat_monitor is not None:
                heartbeat_monitor.beat()

            # Record successful epoch completion with circuit breaker (2025-12)
            if training_breaker:
                training_breaker.record_success("training_epoch")
        else:
            # Final checkpoint at end of training (if not early stopped).
            # This else clause is for the for-loop and executes if no break
            # occurred.
            if not distributed or is_main_process():
                model_to_save_final = cast(
                    nn.Module,
                    model.module if distributed else model,
                )
                final_checkpoint_path = finalize_training_checkpoints(
                    model_to_save_final=model_to_save_final,
                    optimizer=optimizer,
                    config=config,
                    checkpoint_dir=checkpoint_dir,
                    save_path=save_path,
                    num_players=num_players,
                    avg_val_loss=avg_val_loss,
                    best_val_loss=best_val_loss,
                    best_train_loss_at_best_val=best_train_loss_at_best_val,
                    overfit_divergence_threshold=overfit_divergence_threshold,
                    prefer_best_on_overfit=prefer_best_on_overfit,
                    early_stopper=early_stopper,
                    checkpoint_averager=checkpoint_averager,
                    async_checkpointer=async_checkpointer,
                    epoch_scheduler=epoch_scheduler,
                )
                _final_checkpoint_path = final_checkpoint_path  # Track for event emission

                # Log reanalysis summary if enabled (2025-12)
                if enhancements_manager is not None:
                    reanalysis_stats = enhancements_manager.get_reanalysis_stats()
                    if reanalysis_stats.get("enabled") and reanalysis_stats.get("positions_reanalyzed", 0) > 0:
                        logger.info(
                            f"[Reanalysis Summary] "
                            f"Positions: {reanalysis_stats['positions_reanalyzed']}, "
                            f"Games: {reanalysis_stats['games_reanalyzed']}, "
                            f"Blend ratio: {reanalysis_stats['blend_ratio']:.2f}"
                        )

                    # Log distillation summary if enabled (2025-12)
                    distillation_stats = enhancements_manager.get_distillation_stats()
                    if distillation_stats.get("enabled") and distillation_stats.get("last_distillation_epoch", 0) > 0:
                        logger.info(
                            f"[Distillation Summary] "
                            f"Last epoch: {distillation_stats['last_distillation_epoch']}, "
                            f"Teachers: {distillation_stats['available_teachers']}, "
                            f"Temperature: {distillation_stats['temperature']:.1f}"
                        )

                # Publish training completed event (2025-12)
                if HAS_EVENT_BUS and get_router is not None:
                    try:
                        router = get_router()
                        event_payload = {
                            "epochs_completed": epochs_completed,
                            "best_val_loss": float(best_val_loss),
                            "final_train_loss": float(avg_train_loss),
                            "final_val_loss": float(avg_val_loss),
                            "config": f"{config.board_type.value}_{num_players}p",
                            "config_key": f"{config.board_type.value}_{num_players}p",
                            "checkpoint_path": str(final_checkpoint_path),
                            "trigger_evaluation": True,  # Trigger automatic evaluation
                            # model_path for FeedbackLoopController (Dec 2025 integration fix)
                            "model_path": str(save_path),
                            # policy_accuracy for evaluation trigger threshold check
                            "policy_accuracy": float(avg_policy_accuracy),
                            # Feb 2026: Include training data stats for generation tracking
                            "training_samples": _total_samples,
                            "training_games": _num_data_files,
                        }
                        # Add reanalysis and distillation stats to event payload
                        if enhancements_manager is not None:
                            reanalysis_stats = enhancements_manager.get_reanalysis_stats()
                            if reanalysis_stats.get("enabled"):
                                event_payload["reanalysis"] = reanalysis_stats
                            distillation_stats = enhancements_manager.get_distillation_stats()
                            if distillation_stats.get("enabled"):
                                event_payload["distillation"] = distillation_stats
                        router.publish_sync(DataEvent(
                            event_type=DataEventType.TRAINING_COMPLETED,
                            payload=event_payload,
                            source="train",
                        ))
                    except (RuntimeError, ConnectionError, TimeoutError) as e:
                        # Event emission can fail due to async runtime or network issues
                        logger.debug(f"Failed to publish training completed event: {e}")

                # Emit curriculum update event (December 2025)
                # Triggers curriculum reweighting when policy accuracy crosses threshold
                # January 2026 - migrated to event_router
                try:
                    from app.coordination.event_emission_helpers import safe_emit_event

                    config_key = f"{config.board_type.value}_{num_players}p"
                    policy_accuracy_threshold = 0.75

                    # Check if this config should have its curriculum weight increased
                    # High policy accuracy indicates strong learning - boost priority
                    trigger_reweight = avg_policy_accuracy >= policy_accuracy_threshold

                    if trigger_reweight:
                        # Increase curriculum weight for well-performing configs
                        new_weight = 1.0 + (avg_policy_accuracy - 0.5) * 0.5  # 0.75 acc → 1.125 weight
                        safe_emit_event(
                            "CURRICULUM_UPDATED",
                            {
                                "config_key": config_key,
                                "new_weight": new_weight,
                                "trigger": "training_complete",
                                "policy_accuracy": avg_policy_accuracy,
                                "value_loss": avg_val_loss,
                            },
                            context="train.py",
                        )
                        logger.info(
                            f"[Curriculum] Triggered reweight for {config_key}: "
                            f"policy_acc={avg_policy_accuracy:.1%} → weight={new_weight:.3f}"
                        )
                except ImportError:
                    pass  # Event emitters not available
                except (RuntimeError, ConnectionError, TimeoutError, AttributeError) as e:
                    # Event emission failures, network issues, or missing attributes
                    logger.debug(f"Failed to emit curriculum update: {e}")

                # Mark training as completed successfully (for hardened event emission)
                _training_completed_normally = True

                # Feb 28, 2026: Print a clear machine-parseable summary line to stdout.
                # training_executor.py parses this to report training_samples and final_loss
                # in the work result. Previously, it used regex matching on log lines which
                # matched wrong numbers (validation samples instead of total).
                _summary_loss = best_val_loss if best_val_loss != float('inf') else avg_val_loss
                print(
                    f"TRAINING_SUMMARY: loss={_summary_loss:.4f} "
                    f"samples={_total_samples} games={_num_data_files} "
                    f"epochs={config.epochs_per_iter}"
                )

    except (RuntimeError, ValueError, OSError, KeyError) as e:
        # RuntimeError: CUDA/tensor operations, training loop errors
        # ValueError: invalid training parameters or data
        # OSError: checkpoint save/load failures
        # KeyError: missing required data or config keys
        # Capture exception for hardened event emission in finally block
        _training_exception = e
        raise  # Re-raise after capturing
    finally:
        # ==========================================================================
        # Hardened Event Emission (December 2025)
        # ==========================================================================
        # ALWAYS emit training completion/failure event, even if training crashed.
        # This ensures the feedback loop (Training→Evaluation→Curriculum) never breaks.
        if HAS_EVENT_BUS and get_router is not None and (not distributed or is_main_process()):
            try:
                _training_duration = time.time() - _training_start_time
                _config_key = f"{config.board_type.value}_{num_players}p"

                if _training_completed_normally:
                    # Training succeeded - emit TRAINING_COMPLETED (may be duplicate, but guaranteed)
                    router = get_router()
                    payload = {
                        "epochs_completed": epochs_completed,
                        "best_val_loss": float(best_val_loss),
                        "final_train_loss": float(avg_train_loss),
                        "final_val_loss": float(avg_val_loss),
                        "config": _config_key,
                        "config_key": _config_key,
                        "board_type": config.board_type.value,
                        "num_players": num_players,
                        "duration_seconds": _training_duration,
                        "hardened_emit": True,  # Flag indicating this came from finally block
                        "trigger_evaluation": True,  # Trigger automatic evaluation
                        # model_path for FeedbackLoopController (Dec 2025 integration fix)
                        "model_path": str(save_path),
                        # policy_accuracy for evaluation trigger threshold check
                        # Jan 2026: Fixed - use proper None check instead of 'in dir()'
                        "policy_accuracy": float(avg_policy_accuracy) if avg_policy_accuracy is not None else 0.0,
                        # Feb 2026: Include training data stats for generation tracking
                        "training_samples": _total_samples,
                        "training_games": _num_data_files,
                    }
                    # Include checkpoint_path if available (for auto-evaluation)
                    if _final_checkpoint_path:
                        payload["checkpoint_path"] = str(_final_checkpoint_path)
                    router.publish_sync(DataEvent(
                        event_type=DataEventType.TRAINING_COMPLETED,
                        payload=payload,
                        source="train_finally",
                    ))
                    logger.info(f"[train] Hardened TRAINING_COMPLETED emitted for {_config_key}")
                else:
                    # Training failed - emit TRAINING_FAILED
                    router = get_router()
                    error_msg = str(_training_exception) if _training_exception else "Unknown error"
                    router.publish_sync(DataEvent(
                        event_type=DataEventType.TRAINING_FAILED,
                        payload={
                            "config": _config_key,
                            "error": error_msg,
                            "epochs_completed": epochs_completed,
                            "duration_seconds": _training_duration,
                            "best_val_loss": float(best_val_loss) if best_val_loss != float('inf') else None,
                        },
                        source="train_finally",
                    ))
                    logger.warning(f"[train] Hardened TRAINING_FAILED emitted for {_config_key}: {error_msg}")
            except (RuntimeError, ConnectionError, TimeoutError, AttributeError, NameError) as e:
                # Event emission failures, network issues, missing attributes, or undefined vars in finally block
                logger.debug(f"Failed to emit hardened training event: {e}")

        # Shutdown async checkpointer and wait for pending saves
        if async_checkpointer is not None:
            async_checkpointer.shutdown()
            logger.info("Async checkpointer shutdown complete")

        # Stop heartbeat monitor
        if heartbeat_monitor is not None:
            heartbeat_monitor.stop()
            logger.info("Heartbeat monitor stopped")

        # Teardown graceful shutdown handler (2025-12)
        if shutdown_handler is not None:
            shutdown_handler.teardown()
            logger.debug("Graceful shutdown handler teardown complete")

        # Stop integrated enhancements background services
        if enhancements_manager is not None:
            enhancements_manager.stop_background_services()
            logger.info("Integrated enhancements background services stopped")

        # Clean up distributed process group
        if distributed:
            cleanup_distributed()

        # Explicitly shutdown DataLoader workers to prevent process hang.
        # On Linux with num_workers>0, worker processes prevent the main
        # process from exiting (GH200 nodes stuck for 12+ hours).
        for loader in (train_loader, val_loader):
            if loader is not None and hasattr(loader, '_workers'):
                try:
                    loader._iterator = None
                    # Force shutdown of any active worker processes
                    if hasattr(loader, '_shutdown_workers'):
                        loader._shutdown_workers()
                except (AttributeError, OSError, RuntimeError):
                    pass
        # Delete references to trigger __del__ cleanup
        del train_loader, val_loader

    # ==========================================================================
    # Auto-Promotion Hook (January 2026)
    # ==========================================================================
    # If auto-promote is enabled and training completed successfully,
    # run gauntlet evaluation and promote if criteria met.
    if auto_promote and _training_completed_normally:
        logger.info("[AutoPromotion] Starting automated promotion evaluation...")
        try:
            import asyncio
            from app.training.auto_promotion import evaluate_and_promote

            async def _run_auto_promote():
                result = await evaluate_and_promote(
                    model_path=save_path,
                    board_type=config.board_type.value if hasattr(config.board_type, 'value') else str(config.board_type),
                    num_players=num_players,
                    games=auto_promote_games,
                    sync_to_cluster=auto_promote_sync,
                )
                return result

            # Run async promotion in event loop
            try:
                loop = asyncio.get_running_loop()
                promotion_result = asyncio.ensure_future(_run_auto_promote())
            except RuntimeError:
                # No running loop - create one
                promotion_result = asyncio.run(_run_auto_promote())

            if hasattr(promotion_result, 'approved') and promotion_result.approved:
                logger.info(f"[AutoPromotion] Model promoted: {promotion_result.reason}")
            elif hasattr(promotion_result, 'reason'):
                logger.info(f"[AutoPromotion] Promotion rejected: {promotion_result.reason}")

        except ImportError as e:
            logger.warning(f"[AutoPromotion] Auto-promotion module not available: {e}")
        except (RuntimeError, ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"[AutoPromotion] Auto-promotion failed: {e}")

    # Return structured training result for downstream analysis
    return {
        'best_val_loss': float(best_val_loss),
        'final_train_loss': float(avg_train_loss),
        'final_val_loss': float(avg_val_loss),
        'epochs_completed': epochs_completed,
        'epoch_losses': epoch_losses,
    }
# Re-export CLI functions for backwards compatibility
# The actual implementations are in train_cli.py
from app.training.train_cli import main, parse_args

if __name__ == "__main__":
    main()

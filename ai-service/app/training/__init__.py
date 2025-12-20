"""Training module for RingRift AI.

This package provides training infrastructure including:
- Promotion controller for model promotion decisions
- Elo service for rating management
- Model registry for lifecycle tracking
- Tier promotion for difficulty ladder
- Consolidated training components (December 2025)

Architecture Overview (December 2025)
=====================================

1. DATA LOADING & AUGMENTATION
   - data_loader.py: StreamingDataLoader, WeightedStreamingDataLoader
   - hot_data_buffer.py: HotDataBuffer for priority experience replay
   - data_augmentation.py: DataAugmentor (unified factory)
   - elo_weighting.py: EloWeightedSampler for opponent-strength weighting

2. DISTRIBUTED TRAINING
   - distributed_unified.py: CANONICAL - UnifiedDistributedTrainer
     (gradient compression, async SGD, mixed precision)
   - distributed.py: Helper functions + basic DistributedTrainer
     (setup_distributed, cleanup_distributed, is_main_process, etc.)
   - checkpoint_unified.py: UnifiedCheckpointManager
     (adaptive checkpointing, hash verification, lineage tracking)

3. ADVANCED FEATURES
   - integrated_enhancements.py: Unified config combining:
     * auxiliary_tasks.py: Multi-task heads (game length, piece count, outcome)
     * gradient_surgery.py: PCGrad/CAGrad for multi-task gradients
     * batch_scheduling.py: Dynamic batch size scheduling
     * curriculum.py: Progressive difficulty staging
   - unified_orchestrator.py: UnifiedTrainingOrchestrator
     (combines all components with context manager interface)
   - orchestrated_training.py: TrainingOrchestrator
     (unified lifecycle manager for training services)

4. SCHEDULING & CONTROL
   - temperature_scheduling.py: CANONICAL scheduler module
     (TemperatureScheduler, multiple schedule types)
   - promotion_controller.py: Model promotion decisions
   - regression_detector.py: Elo regression detection

5. UTILITIES
   - significance.py: Statistical utilities (wilson_score_interval)
   - value_calibration.py: CalibrationTracker for value head
   - seed_utils.py: Reproducibility helpers
   - checkpointing.py: save_checkpoint, load_checkpoint, AsyncCheckpointer
   - schedulers.py: LR scheduler creation (warmup, cosine, step decay)
   - selfplay_config.py: Unified selfplay configuration
   - datasets.py: RingRiftDataset, WeightedRingRiftDataset

Quick Start
-----------
    # Basic training with unified orchestrator
    from app.training import (
        UnifiedTrainingOrchestrator,
        OrchestratorConfig,
    )

    config = OrchestratorConfig(
        board_type="square8",
        enable_hot_buffer=True,
        enable_curriculum=True,
    )

    with UnifiedTrainingOrchestrator(model, config) as orchestrator:
        for batch in dataloader:
            metrics = orchestrator.train_step(batch)

    # Distributed training
    from app.training import (
        setup_distributed,
        cleanup_distributed,
        is_main_process,
        DistributedMetrics,
    )

    setup_distributed()
    # ... training loop ...
    cleanup_distributed()

    # Model promotion
    from app.training import PromotionController, get_promotion_controller

    controller = get_promotion_controller()
    decision = controller.should_promote(model_id, metrics)
"""

__all__ = [
    # Feature flags
    "HAS_AUGMENTATION",
    "HAS_CHECKPOINT_UNIFIED",
    "HAS_CHECKPOINTING",
    "HAS_CROSSBOARD_STRENGTH",
    "HAS_DATA_LOADERS",
    "HAS_DATASETS",
    "HAS_DISTRIBUTED_HELPERS",
    "HAS_DISTRIBUTED_UNIFIED",
    "HAS_ELO_WEIGHTING",
    "HAS_HOT_BUFFER",
    "HAS_INTEGRATED_ENHANCEMENTS",
    "HAS_ORCHESTRATOR",
    "HAS_PROMOTION_CONTROLLER",
    "HAS_REGRESSION_DETECTOR",
    "HAS_SCHEDULERS",
    "HAS_SEED_UTILS",
    "HAS_SELFPLAY_CONFIG",
    "HAS_SIGNIFICANCE",
    "HAS_TEMPERATURE_SCHEDULING",
    "HAS_TOURNAMENT_UTILS",
    "HAS_TRAINING_ENV",
    "HAS_TRAINING_ORCHESTRATOR",
    "HAS_UNIFIED_MODEL_STORE",
    "HAS_VALUE_CALIBRATION",
    # Promotion controller
    "PromotionController",
    "PromotionCriteria",
    "PromotionDecision",
    "PromotionType",
    "get_promotion_controller",
    # Integrated enhancements
    "IntegratedEnhancementsConfig",
    "IntegratedTrainingManager",
    "create_integrated_manager",
    # Checkpoint unified
    "AsyncCheckpointer",
    "CheckpointType",
    "GracefulShutdownHandler",
    "TrainingProgress",
    "UnifiedCheckpointConfig",
    "UnifiedCheckpointManager",
    "load_checkpoint",
    "save_checkpoint",
    # Distributed unified
    "UnifiedDistributedConfig",
    "UnifiedDistributedTrainer",
    # Distributed helpers
    "DistributedConfig",
    "DistributedMetrics",
    "DistributedTrainer",
    "all_gather_object",
    "broadcast_object",
    "cleanup_distributed",
    "get_device_for_rank",
    "get_distributed_sampler",
    "get_local_rank",
    "get_rank",
    "get_world_size",
    "is_distributed",
    "is_main_process",
    "reduce_tensor",
    "scale_learning_rate",
    "seed_everything",
    "setup_distributed",
    "synchronize",
    "wrap_model_ddp",
    # Temperature scheduling
    "TemperatureConfig",
    "TemperatureScheduler",
    "create_temperature_scheduler",
    # Unified orchestrator
    "OrchestratorConfig",
    "UnifiedTrainingOrchestrator",
    # Training orchestrator
    "TrainingOrchestrator",
    "TrainingOrchestratorConfig",
    "TrainingOrchestratorState",
    "get_training_orchestrator",
    # Regression detector
    "RegressionConfig",
    "RegressionDetector",
    "RegressionEvent",
    "RegressionListener",
    "RegressionSeverity",
    "create_regression_detector",
    "get_regression_detector",
    # Significance
    "wilson_lower_bound",
    "wilson_score_interval",
    # Crossboard strength
    "inversion_count",
    "normalise_tier_name",
    "rank_order_from_elos",
    "spearman_rank_correlation",
    "summarize_crossboard_tier_strength",
    "tier_number",
    # Value calibration
    "CalibrationReport",
    "CalibrationTracker",
    "ValueCalibrator",
    "create_reliability_diagram",
    # Schedulers
    "create_lr_scheduler",
    "get_warmup_scheduler",
    # Selfplay config
    "EngineMode",
    "OutputFormat",
    "SelfplayConfig",
    "create_argument_parser",
    "get_default_config",
    "get_production_config",
    "parse_selfplay_args",
    # Datasets
    "RingRiftDataset",
    "WeightedRingRiftDataset",
    # Data loaders
    "FileHandle",
    "StreamingDataLoader",
    "WeightedStreamingDataLoader",
    # Hot data buffer
    "GameRecord",
    "HotDataBuffer",
    "create_hot_buffer",
    # Data augmentation
    "AugmentedDataLoader",
    "AugmentorConfig",
    "DataAugmentor",
    # Elo weighting
    "EloWeightedSampler",
    "compute_elo_weights",
    # Unified model store
    "ModelInfo",
    "ModelStoreStage",
    "ModelStoreType",
    "UnifiedModelStore",
    "get_model_store",
    "get_production_model",
    "promote_model",
    "register_model",
    # Training environment
    "TrainingEnvConfig",
    "make_env",
    # Seed utilities
    "seed_all",
    # Tournament utilities
    "infer_victory_reason",
]

# Import promotion controller if available
try:
    from app.training.promotion_controller import (
        PromotionController,
        PromotionCriteria,
        PromotionDecision,
        PromotionType,
        get_promotion_controller,
    )
    HAS_PROMOTION_CONTROLLER = True
except ImportError:
    HAS_PROMOTION_CONTROLLER = False

# Import integrated enhancements (December 2025)
try:
    from app.training.integrated_enhancements import (
        IntegratedEnhancementsConfig,
        IntegratedTrainingManager,
        create_integrated_manager,
    )
    HAS_INTEGRATED_ENHANCEMENTS = True
except ImportError:
    HAS_INTEGRATED_ENHANCEMENTS = False

# Import consolidated checkpoint manager (December 2025)
try:
    from app.training.checkpoint_unified import (
        CheckpointType,
        TrainingProgress,
        UnifiedCheckpointConfig,
        UnifiedCheckpointManager,
    )
    HAS_CHECKPOINT_UNIFIED = True
except ImportError:
    HAS_CHECKPOINT_UNIFIED = False

# Import consolidated distributed trainer (December 2025)
try:
    from app.training.distributed_unified import (
        UnifiedDistributedConfig,
        UnifiedDistributedTrainer,
    )
    HAS_DISTRIBUTED_UNIFIED = True
except ImportError:
    HAS_DISTRIBUTED_UNIFIED = False

# Import distributed helper functions (December 2025)
try:
    from app.training.distributed import (
        DistributedConfig,
        DistributedMetrics,
        DistributedTrainer,
        all_gather_object,
        broadcast_object,
        cleanup_distributed,
        get_device_for_rank,
        get_distributed_sampler,
        get_local_rank,
        get_rank,
        get_world_size,
        is_distributed,
        is_main_process,
        reduce_tensor,
        scale_learning_rate,
        seed_everything,
        setup_distributed,
        synchronize,
        wrap_model_ddp,
    )
    HAS_DISTRIBUTED_HELPERS = True
except ImportError:
    HAS_DISTRIBUTED_HELPERS = False

# Import temperature scheduling (December 2025)
try:
    from app.training.temperature_scheduling import (
        TemperatureConfig,
        TemperatureScheduler,
        create_scheduler as create_temperature_scheduler,
    )
    HAS_TEMPERATURE_SCHEDULING = True
except ImportError:
    HAS_TEMPERATURE_SCHEDULING = False

# Import unified orchestrator (December 2025)
try:
    from app.training.unified_orchestrator import (
        OrchestratorConfig,
        UnifiedTrainingOrchestrator,
    )
    HAS_ORCHESTRATOR = True
except ImportError:
    HAS_ORCHESTRATOR = False

# Import training orchestrator (December 2025)
try:
    from app.training.orchestrated_training import (
        TrainingOrchestrator,
        TrainingOrchestratorConfig,
        TrainingOrchestratorState,
        get_training_orchestrator,
    )
    HAS_TRAINING_ORCHESTRATOR = True
except ImportError:
    HAS_TRAINING_ORCHESTRATOR = False

# Import consolidated regression detector (December 2025)
try:
    from app.training.regression_detector import (
        RegressionConfig,
        RegressionDetector,
        RegressionEvent,
        RegressionListener,
        RegressionSeverity,
        create_regression_detector,
        get_regression_detector,
    )
    HAS_REGRESSION_DETECTOR = True
except ImportError:
    HAS_REGRESSION_DETECTOR = False

# Import statistical utilities (December 2025)
try:
    from app.training.significance import (
        wilson_lower_bound,
        wilson_score_interval,
    )
    HAS_SIGNIFICANCE = True
except ImportError:
    HAS_SIGNIFICANCE = False

# Import crossboard strength analysis (December 2025)
try:
    from app.training.crossboard_strength import (
        inversion_count,
        normalise_tier_name,
        rank_order_from_elos,
        spearman_rank_correlation,
        summarize_crossboard_tier_strength,
        tier_number,
    )
    HAS_CROSSBOARD_STRENGTH = True
except ImportError:
    HAS_CROSSBOARD_STRENGTH = False

# Import value calibration utilities (December 2025)
try:
    from app.training.value_calibration import (
        CalibrationReport,
        CalibrationTracker,
        ValueCalibrator,
        create_reliability_diagram,
    )
    HAS_VALUE_CALIBRATION = True
except ImportError:
    HAS_VALUE_CALIBRATION = False

# Import checkpointing utilities (December 2025)
# Migrated to import from checkpoint_unified
try:
    from app.training.checkpoint_unified import (
        AsyncCheckpointer,
        GracefulShutdownHandler,
        load_checkpoint,
        save_checkpoint,
    )
    HAS_CHECKPOINTING = True
except ImportError:
    HAS_CHECKPOINTING = False

# Import LR scheduler utilities (December 2025)
try:
    from app.training.schedulers import (
        create_lr_scheduler,
        get_warmup_scheduler,
    )
    HAS_SCHEDULERS = True
except ImportError:
    HAS_SCHEDULERS = False

# Import selfplay configuration (December 2025)
try:
    from app.training.selfplay_config import (
        EngineMode,
        OutputFormat,
        SelfplayConfig,
        create_argument_parser,
        get_default_config,
        get_production_config,
        parse_selfplay_args,
    )
    HAS_SELFPLAY_CONFIG = True
except ImportError:
    HAS_SELFPLAY_CONFIG = False

# Import dataset classes (December 2025)
try:
    from app.training.datasets import (
        RingRiftDataset,
        WeightedRingRiftDataset,
    )
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

# =============================================================================
# Consolidated Data Loaders (December 2025)
# =============================================================================
# This provides a unified import point for all data loading functionality:
# - StreamingDataLoader: Main disk-backed streaming loader
# - WeightedStreamingDataLoader: Priority-weighted sampling
# - AugmentedDataLoader: With data augmentation
# - HotDataBuffer: In-memory priority buffer with quality auto-calibration

try:
    from app.training.data_loader import (
        FileHandle,
        StreamingDataLoader,
        WeightedStreamingDataLoader,
    )
    HAS_DATA_LOADERS = True
except ImportError:
    HAS_DATA_LOADERS = False

# Import HotDataBuffer with quality integration
try:
    from app.training.hot_data_buffer import (
        GameRecord,
        HotDataBuffer,
        create_hot_buffer,
    )
    HAS_HOT_BUFFER = True
except ImportError:
    HAS_HOT_BUFFER = False

# Import data augmentation loader
try:
    from app.training.data_augmentation import (
        AugmentedDataLoader,
        AugmentorConfig,
        DataAugmentor,
    )
    HAS_AUGMENTATION = True
except ImportError:
    HAS_AUGMENTATION = False

# Import Elo-weighted sampler
try:
    from app.training.elo_weighting import (
        EloWeightedSampler,
        compute_elo_weights,
    )
    HAS_ELO_WEIGHTING = True
except ImportError:
    HAS_ELO_WEIGHTING = False

# =============================================================================
# Unified Model Store (December 2025)
# =============================================================================
try:
    from app.training.unified_model_store import (
        ModelInfo,
        ModelStoreStage,
        ModelStoreType,
        UnifiedModelStore,
        get_model_store,
        get_production_model,
        promote_model,
        register_model,
    )
    HAS_UNIFIED_MODEL_STORE = True
except ImportError:
    HAS_UNIFIED_MODEL_STORE = False

# =============================================================================
# Environment & Seed Utilities (December 2025)
# =============================================================================

# Import training environment configuration
try:
    from app.training.env import (
        TrainingEnvConfig,
        make_env,
    )
    HAS_TRAINING_ENV = True
except ImportError:
    HAS_TRAINING_ENV = False

# Import seed utilities for reproducibility
try:
    from app.training.seed_utils import (
        seed_all,
    )
    HAS_SEED_UTILS = True
except ImportError:
    HAS_SEED_UTILS = False

# Import tournament utilities
try:
    from app.training.tournament import (
        infer_victory_reason,
    )
    HAS_TOURNAMENT_UTILS = True
except ImportError:
    HAS_TOURNAMENT_UTILS = False

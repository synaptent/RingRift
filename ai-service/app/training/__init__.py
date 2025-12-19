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

# Import promotion controller if available
try:
    from app.training.promotion_controller import (
        PromotionController,
        PromotionType,
        PromotionCriteria,
        PromotionDecision,
        get_promotion_controller,
    )
    HAS_PROMOTION_CONTROLLER = True
except ImportError:
    HAS_PROMOTION_CONTROLLER = False

# Import integrated enhancements (December 2025)
try:
    from app.training.integrated_enhancements import (
        IntegratedTrainingManager,
        IntegratedEnhancementsConfig,
        create_integrated_manager,
    )
    HAS_INTEGRATED_ENHANCEMENTS = True
except ImportError:
    HAS_INTEGRATED_ENHANCEMENTS = False

# Import consolidated checkpoint manager (December 2025)
try:
    from app.training.checkpoint_unified import (
        UnifiedCheckpointManager,
        UnifiedCheckpointConfig,
        TrainingProgress,
        CheckpointType,
    )
    HAS_CHECKPOINT_UNIFIED = True
except ImportError:
    HAS_CHECKPOINT_UNIFIED = False

# Import consolidated distributed trainer (December 2025)
try:
    from app.training.distributed_unified import (
        UnifiedDistributedTrainer,
        UnifiedDistributedConfig,
    )
    HAS_DISTRIBUTED_UNIFIED = True
except ImportError:
    HAS_DISTRIBUTED_UNIFIED = False

# Import distributed helper functions (December 2025)
try:
    from app.training.distributed import (
        setup_distributed,
        cleanup_distributed,
        is_distributed,
        is_main_process,
        get_rank,
        get_world_size,
        get_local_rank,
        get_distributed_sampler,
        wrap_model_ddp,
        synchronize,
        reduce_tensor,
        all_gather_object,
        broadcast_object,
        get_device_for_rank,
        seed_everything,
        scale_learning_rate,
        DistributedMetrics,
        DistributedTrainer,
        DistributedConfig,
    )
    HAS_DISTRIBUTED_HELPERS = True
except ImportError:
    HAS_DISTRIBUTED_HELPERS = False

# Import temperature scheduling (December 2025)
try:
    from app.training.temperature_scheduling import (
        TemperatureScheduler,
        TemperatureConfig,
        create_scheduler as create_temperature_scheduler,
    )
    HAS_TEMPERATURE_SCHEDULING = True
except ImportError:
    HAS_TEMPERATURE_SCHEDULING = False

# Import unified orchestrator (December 2025)
try:
    from app.training.unified_orchestrator import (
        UnifiedTrainingOrchestrator,
        OrchestratorConfig,
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

__all__ = [
    "HAS_PROMOTION_CONTROLLER",
    "HAS_INTEGRATED_ENHANCEMENTS",
    "HAS_CHECKPOINT_UNIFIED",
    "HAS_DISTRIBUTED_UNIFIED",
    "HAS_DISTRIBUTED_HELPERS",
    "HAS_TEMPERATURE_SCHEDULING",
    "HAS_ORCHESTRATOR",
    "HAS_TRAINING_ORCHESTRATOR",
]

if HAS_PROMOTION_CONTROLLER:
    __all__.extend([
        "PromotionController",
        "PromotionType",
        "PromotionCriteria",
        "PromotionDecision",
        "get_promotion_controller",
    ])

if HAS_INTEGRATED_ENHANCEMENTS:
    __all__.extend([
        "IntegratedTrainingManager",
        "IntegratedEnhancementsConfig",
        "create_integrated_manager",
    ])

if HAS_CHECKPOINT_UNIFIED:
    __all__.extend([
        "UnifiedCheckpointManager",
        "UnifiedCheckpointConfig",
        "TrainingProgress",
        "CheckpointType",
    ])

if HAS_DISTRIBUTED_UNIFIED:
    __all__.extend([
        "UnifiedDistributedTrainer",
        "UnifiedDistributedConfig",
    ])

if HAS_DISTRIBUTED_HELPERS:
    __all__.extend([
        "setup_distributed",
        "cleanup_distributed",
        "is_distributed",
        "is_main_process",
        "get_rank",
        "get_world_size",
        "get_local_rank",
        "get_distributed_sampler",
        "wrap_model_ddp",
        "synchronize",
        "reduce_tensor",
        "all_gather_object",
        "broadcast_object",
        "get_device_for_rank",
        "seed_everything",
        "scale_learning_rate",
        "DistributedMetrics",
        "DistributedTrainer",
        "DistributedConfig",
    ])

if HAS_TEMPERATURE_SCHEDULING:
    __all__.extend([
        "TemperatureScheduler",
        "TemperatureConfig",
        "create_temperature_scheduler",
    ])

if HAS_ORCHESTRATOR:
    __all__.extend([
        "UnifiedTrainingOrchestrator",
        "OrchestratorConfig",
    ])

if HAS_TRAINING_ORCHESTRATOR:
    __all__.extend([
        "TrainingOrchestrator",
        "TrainingOrchestratorConfig",
        "TrainingOrchestratorState",
        "get_training_orchestrator",
    ])

# Import consolidated regression detector (December 2025)
try:
    from app.training.regression_detector import (
        RegressionDetector,
        RegressionConfig,
        RegressionEvent,
        RegressionSeverity,
        RegressionListener,
        get_regression_detector,
        create_regression_detector,
    )
    HAS_REGRESSION_DETECTOR = True
except ImportError:
    HAS_REGRESSION_DETECTOR = False

__all__.append("HAS_REGRESSION_DETECTOR")

if HAS_REGRESSION_DETECTOR:
    __all__.extend([
        "RegressionDetector",
        "RegressionConfig",
        "RegressionEvent",
        "RegressionSeverity",
        "RegressionListener",
        "get_regression_detector",
        "create_regression_detector",
    ])

# Import statistical utilities (December 2025)
try:
    from app.training.significance import (
        wilson_score_interval,
        wilson_lower_bound,
    )
    HAS_SIGNIFICANCE = True
except ImportError:
    HAS_SIGNIFICANCE = False

__all__.append("HAS_SIGNIFICANCE")

if HAS_SIGNIFICANCE:
    __all__.extend([
        "wilson_score_interval",
        "wilson_lower_bound",
    ])

# Import crossboard strength analysis (December 2025)
try:
    from app.training.crossboard_strength import (
        normalise_tier_name,
        tier_number,
        rank_order_from_elos,
        spearman_rank_correlation,
        inversion_count,
        summarize_crossboard_tier_strength,
    )
    HAS_CROSSBOARD_STRENGTH = True
except ImportError:
    HAS_CROSSBOARD_STRENGTH = False

__all__.append("HAS_CROSSBOARD_STRENGTH")

if HAS_CROSSBOARD_STRENGTH:
    __all__.extend([
        "normalise_tier_name",
        "tier_number",
        "rank_order_from_elos",
        "spearman_rank_correlation",
        "inversion_count",
        "summarize_crossboard_tier_strength",
    ])

# Import value calibration utilities (December 2025)
try:
    from app.training.value_calibration import (
        CalibrationTracker,
        CalibrationReport,
        ValueCalibrator,
        create_reliability_diagram,
    )
    HAS_VALUE_CALIBRATION = True
except ImportError:
    HAS_VALUE_CALIBRATION = False

__all__.append("HAS_VALUE_CALIBRATION")

if HAS_VALUE_CALIBRATION:
    __all__.extend([
        "CalibrationTracker",
        "CalibrationReport",
        "ValueCalibrator",
        "create_reliability_diagram",
    ])

# Import checkpointing utilities (December 2025)
try:
    from app.training.checkpointing import (
        save_checkpoint,
        load_checkpoint,
        AsyncCheckpointer,
        GracefulShutdownHandler,
    )
    HAS_CHECKPOINTING = True
except ImportError:
    HAS_CHECKPOINTING = False

__all__.append("HAS_CHECKPOINTING")

if HAS_CHECKPOINTING:
    __all__.extend([
        "save_checkpoint",
        "load_checkpoint",
        "AsyncCheckpointer",
        "GracefulShutdownHandler",
    ])

# Import LR scheduler utilities (December 2025)
try:
    from app.training.schedulers import (
        get_warmup_scheduler,
        create_lr_scheduler,
    )
    HAS_SCHEDULERS = True
except ImportError:
    HAS_SCHEDULERS = False

__all__.append("HAS_SCHEDULERS")

if HAS_SCHEDULERS:
    __all__.extend([
        "get_warmup_scheduler",
        "create_lr_scheduler",
    ])

# Import selfplay configuration (December 2025)
try:
    from app.training.selfplay_config import (
        SelfplayConfig,
        EngineMode,
        OutputFormat,
        parse_selfplay_args,
        create_argument_parser,
        get_default_config,
        get_production_config,
    )
    HAS_SELFPLAY_CONFIG = True
except ImportError:
    HAS_SELFPLAY_CONFIG = False

__all__.append("HAS_SELFPLAY_CONFIG")

if HAS_SELFPLAY_CONFIG:
    __all__.extend([
        "SelfplayConfig",
        "EngineMode",
        "OutputFormat",
        "parse_selfplay_args",
        "create_argument_parser",
        "get_default_config",
        "get_production_config",
    ])

# Import dataset classes (December 2025)
try:
    from app.training.datasets import (
        RingRiftDataset,
        WeightedRingRiftDataset,
    )
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

__all__.append("HAS_DATASETS")

if HAS_DATASETS:
    __all__.extend([
        "RingRiftDataset",
        "WeightedRingRiftDataset",
    ])

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
        StreamingDataLoader,
        WeightedStreamingDataLoader,
        FileHandle,
    )
    HAS_DATA_LOADERS = True
except ImportError:
    HAS_DATA_LOADERS = False

__all__.append("HAS_DATA_LOADERS")

if HAS_DATA_LOADERS:
    __all__.extend([
        "StreamingDataLoader",
        "WeightedStreamingDataLoader",
        "FileHandle",
    ])

# Import HotDataBuffer with quality integration
try:
    from app.training.hot_data_buffer import (
        HotDataBuffer,
        GameRecord,
        create_hot_buffer,
    )
    HAS_HOT_BUFFER = True
except ImportError:
    HAS_HOT_BUFFER = False

__all__.append("HAS_HOT_BUFFER")

if HAS_HOT_BUFFER:
    __all__.extend([
        "HotDataBuffer",
        "GameRecord",
        "create_hot_buffer",
    ])

# Import data augmentation loader
try:
    from app.training.data_augmentation import (
        AugmentedDataLoader,
        DataAugmentor,
        AugmentorConfig,
    )
    HAS_AUGMENTATION = True
except ImportError:
    HAS_AUGMENTATION = False

__all__.append("HAS_AUGMENTATION")

if HAS_AUGMENTATION:
    __all__.extend([
        "AugmentedDataLoader",
        "DataAugmentor",
        "AugmentorConfig",
    ])

# Import Elo-weighted sampler
try:
    from app.training.elo_weighting import (
        EloWeightedSampler,
        compute_elo_weights,
    )
    HAS_ELO_WEIGHTING = True
except ImportError:
    HAS_ELO_WEIGHTING = False

__all__.append("HAS_ELO_WEIGHTING")

if HAS_ELO_WEIGHTING:
    __all__.extend([
        "EloWeightedSampler",
        "compute_elo_weights",
    ])

# =============================================================================
# Unified Model Store (December 2025)
# =============================================================================
try:
    from app.training.unified_model_store import (
        UnifiedModelStore,
        ModelInfo,
        ModelStoreStage,
        ModelStoreType,
        get_model_store,
        register_model,
        get_production_model,
        promote_model,
    )
    HAS_UNIFIED_MODEL_STORE = True
except ImportError:
    HAS_UNIFIED_MODEL_STORE = False

__all__.append("HAS_UNIFIED_MODEL_STORE")

if HAS_UNIFIED_MODEL_STORE:
    __all__.extend([
        "UnifiedModelStore",
        "ModelInfo",
        "ModelStoreStage",
        "ModelStoreType",
        "get_model_store",
        "register_model",
        "get_production_model",
        "promote_model",
    ])

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

__all__.append("HAS_TRAINING_ENV")

if HAS_TRAINING_ENV:
    __all__.extend([
        "TrainingEnvConfig",
        "make_env",
    ])

# Import seed utilities for reproducibility
try:
    from app.training.seed_utils import (
        seed_all,
    )
    HAS_SEED_UTILS = True
except ImportError:
    HAS_SEED_UTILS = False

__all__.append("HAS_SEED_UTILS")

if HAS_SEED_UTILS:
    __all__.extend([
        "seed_all",
    ])

# Import tournament utilities
try:
    from app.training.tournament import (
        infer_victory_reason,
    )
    HAS_TOURNAMENT_UTILS = True
except ImportError:
    HAS_TOURNAMENT_UTILS = False

__all__.append("HAS_TOURNAMENT_UTILS")

if HAS_TOURNAMENT_UTILS:
    __all__.extend([
        "infer_victory_reason",
    ])

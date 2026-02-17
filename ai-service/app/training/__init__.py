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
   - TrainingOrchestrator: DEPRECATED (archived Dec 2025)
     Use UnifiedTrainingOrchestrator instead

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
    "HAS_CHECKPOINTING",
    "HAS_CHECKPOINT_UNIFIED",
    "HAS_CONFIG_RESOLVER",
    "HAS_CROSSBOARD_STRENGTH",
    "HAS_CURRICULUM",
    "HAS_UNIFIED_CURRICULUM",
    "HAS_DATA_LOADER_FACTORY",
    "HAS_DATA_LOADERS",
    "HAS_DATASETS",
    "HAS_DISTILLATION",
    "HAS_DISTRIBUTED_HELPERS",
    "HAS_DISTRIBUTED_UNIFIED",
    "HAS_EBMO_ONLINE",
    "HAS_ELO_WEIGHTING",
    "HAS_HOT_BUFFER",
    "HAS_INTEGRATED_ENHANCEMENTS",
    "HAS_MODEL_FACTORY",
    "HAS_ONLINE_LEARNING",
    "HAS_ORCHESTRATOR",
    "HAS_PROMOTION_CONTROLLER",
    "HAS_REGRESSION_DETECTOR",
    "HAS_SCHEDULERS",
    "HAS_SEED_UTILS",
    "HAS_SELFPLAY_CONFIG",
    "HAS_SIGNIFICANCE",
    "HAS_TEMPERATURE_SCHEDULING",
    "HAS_TOURNAMENT_UTILS",
    "HAS_TRAIN_SETUP",
    "HAS_TRAINING_ENV",
    "HAS_TRAINING_ORCHESTRATOR",
    "HAS_UNIFIED_MODEL_STORE",
    "HAS_VALUE_CALIBRATION",
    # Checkpoint unified
    "AsyncCheckpointer",
    # Data augmentation
    "AugmentedDataLoader",
    "AugmentorConfig",
    # Value calibration
    "CalibrationReport",
    "CalibrationTracker",
    "CheckpointType",
    # Curriculum learning
    "CurriculumController",
    "CurriculumStage",
    "CurriculumState",
    "create_default_curriculum",
    # Unified curriculum service
    "UnifiedCurriculumService",
    "get_unified_curriculum_service",
    "get_unified_curriculum_weight",
    "get_unified_curriculum_weights",
    "DataAugmentor",
    # Knowledge Distillation
    "DistillationConfig",
    "DistillationTrainer",
    "EnsembleTeacher",
    "SoftTargetLoss",
    "create_distillation_trainer",
    "distill_checkpoint_ensemble",
    # Distributed helpers
    "DistributedConfig",
    "DistributedMetrics",
    "DistributedTrainer",
    # Elo weighting
    "EloWeightedSampler",
    # Online learning
    "EBMOOnlineAI",
    "EBMOOnlineConfig",
    "EBMOOnlineLearner",
    "OnlineLearningConfig",
    "OnlineLearningMetrics",
    "OnlineLearningStats",
    "create_online_learner",
    "get_online_learning_config",
    # Selfplay config
    "EngineMode",
    # Data loaders
    "FileHandle",
    # Hot data buffer
    "GameRecord",
    "GracefulShutdownHandler",
    "HotDataBuffer",
    # Integrated enhancements
    "IntegratedEnhancementsConfig",
    "IntegratedTrainingManager",
    # Unified model store
    "ModelInfo",
    "ModelStoreStage",
    "ModelStoreType",
    # Unified orchestrator
    "OrchestratorConfig",
    "OutputFormat",
    # Promotion controller
    "PromotionController",
    "PromotionCriteria",
    "PromotionDecision",
    "PromotionType",
    # Regression detector
    "RegressionConfig",
    "RegressionDetector",
    "RegressionEvent",
    "RegressionListener",
    "RegressionSeverity",
    # Datasets
    "RingRiftDataset",
    "SelfplayConfig",
    "StreamingDataLoader",
    # Temperature scheduling
    "TemperatureConfig",
    "TemperatureScheduler",
    # Training environment
    "TrainingEnvConfig",
    # Training orchestrator
    "TrainingOrchestrator",
    "TrainingOrchestratorConfig",
    "TrainingOrchestratorState",
    "TrainingProgress",
    "UnifiedCheckpointConfig",
    "UnifiedCheckpointManager",
    # Distributed unified
    "UnifiedDistributedConfig",
    "UnifiedDistributedTrainer",
    "UnifiedModelStore",
    "UnifiedTrainingOrchestrator",
    "ValueCalibrator",
    "WeightedRingRiftDataset",
    "WeightedStreamingDataLoader",
    "all_gather_object",
    "broadcast_object",
    "cleanup_distributed",
    "compute_elo_weights",
    "create_argument_parser",
    "create_hot_buffer",
    "create_integrated_manager",
    # Schedulers
    "create_lr_scheduler",
    "create_regression_detector",
    "create_reliability_diagram",
    "create_temperature_scheduler",
    "get_default_config",
    "get_device_for_rank",
    "get_distributed_sampler",
    "get_local_rank",
    "get_model_store",
    "get_production_config",
    "get_production_model",
    "get_promotion_controller",
    "get_rank",
    "get_regression_detector",
    "get_training_orchestrator",
    "get_warmup_scheduler",
    "get_world_size",
    # Tournament utilities
    "infer_victory_reason",
    # Crossboard strength (December 2025)
    "ALL_BOARD_CONFIGS",
    "aggregate_cross_board_elos",
    "check_promotion_threshold",
    "check_promotion_threshold_strict",
    "config_key",
    "inversion_count",
    "parse_config_key",
    "is_distributed",
    "is_main_process",
    "load_checkpoint",
    "make_env",
    "normalise_tier_name",
    "parse_selfplay_args",
    "promote_model",
    "rank_order_from_elos",
    "reduce_tensor",
    "register_model",
    "save_checkpoint",
    "scale_learning_rate",
    # Seed utilities
    "seed_all",
    "seed_everything",
    "setup_distributed",
    "spearman_rank_correlation",
    "summarize_crossboard_tier_strength",
    "synchronize",
    "tier_number",
    # Significance
    "wilson_lower_bound",
    "wilson_score_interval",
    "wrap_model_ddp",
    # Version suffix stripping (Feb 2026)
    "strip_version_suffix",
    # Modular training components (December 2025)
    # Config resolver
    "ResolvedTrainingParams",
    "get_board_size",
    "get_effective_architecture",
    "resolve_training_params",
    "validate_model_id_for_board",
    # Model factory
    "ModelConfig",
    "compute_in_channels",
    "count_parameters",
    "create_model",
    "get_model_architecture",
    "load_model_weights",
    "log_model_summary",
    # Data loader factory
    "DataLoaderConfig",
    "DataLoaderResult",
    "collect_data_paths",
    "create_data_loaders",
    "create_standard_loaders",
    "create_streaming_loaders",
    "should_use_streaming",
    "validate_dataset_metadata",
    # Train setup
    "FaultToleranceComponents",
    "FaultToleranceConfig",
    "TrainingState",
    "compute_effective_lr",
    "get_device",
    "setup_fault_tolerance",
    "setup_graceful_shutdown",
    # High-tier training (December 2025 - 2000+ Elo)
    "ALL_CONFIGS",
    "CANONICAL_CONFIGS",
    "HAS_HIGH_TIER_CONFIG",
    "HAS_HIGH_TIER_ORCHESTRATOR",
    "HighTierSelfplayConfig",
    "HighTierTrainingConfig",
    "TIER_ELO_TARGETS",
    "create_high_tier_orchestrator",
    "get_crossboard_promotion_config",
    "get_engine_mode_for_tier",
    "get_high_tier_config",
    "get_high_tier_selfplay_config",
    "get_high_tier_training_config",
    "should_use_gumbel_engine",
    # Training configuration objects (December 2025)
    "HAS_TRAIN_CONFIG",
    "HAS_TRAIN_VALIDATION",
    "AugmentationConfig",
    "CheckpointConfig",
    "TrainingDistributedConfig",
    "EarlyStoppingConfig",
    "EnhancementConfig",
    "TrainingFaultToleranceConfig",
    "FullTrainingConfig",
    "HeartbeatConfig",
    "LearningRateConfig",
    "MixedPrecisionConfig",
    "ModelArchConfig",
    "TrainingDataConfig",
    "config_from_legacy_params",
    # Training validation utilities (December 2025)
    "FreshnessResult",
    "ValidationResult",
    "validate_data_checksums",
    "validate_training_data_files",
    "validate_training_data_freshness",
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
# Suppress deprecation warnings for backwards-compatible re-exports
try:
    import warnings as _w
    with _w.catch_warnings():
        _w.filterwarnings("ignore", category=DeprecationWarning)
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
# Suppress deprecation warnings for backwards-compatible re-exports
try:
    with _w.catch_warnings():
        _w.filterwarnings("ignore", category=DeprecationWarning)
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
# ARCHIVED December 26, 2025 - moved to archive/deprecated_training/
# Suppress deprecation warnings for backwards-compatible re-exports
try:
    with _w.catch_warnings():
        _w.filterwarnings("ignore", category=DeprecationWarning)
        from archive.deprecated_training.orchestrated_training import (
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
        ALL_BOARD_CONFIGS,
        aggregate_cross_board_elos,
        check_promotion_threshold,
        check_promotion_threshold_strict,
        config_key,
        inversion_count,
        normalise_tier_name,
        parse_config_key,
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

# Import seed utilities for reproducibility (Lane 3 Consolidation)
try:
    from app.training.seed_utils import (
        derive_ai_seed,
        derive_replay_seed,
        derive_worker_seed,
        get_env_seed,
        get_global_seed,
        get_thread_rng,
        reset_thread_rng,
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

# =============================================================================
# Composite ELO System (December 2025)
# =============================================================================
# Provides composite participant management for (NN, Algorithm) combinations

try:
    from app.training.composite_participant import (
        make_composite_participant_id,
        parse_composite_participant_id,
        get_standard_config,
        is_composite_id,
        get_participant_category,
        extract_nn_id,
        extract_ai_type,
        normalize_nn_id,
        strip_version_suffix,
        STANDARD_ALGORITHM_CONFIGS,
        ParticipantCategory,
    )
    HAS_COMPOSITE_PARTICIPANT = True
except ImportError:
    HAS_COMPOSITE_PARTICIPANT = False

try:
    from app.training.composite_elo_migration import (
        run_migrations as run_composite_elo_migrations,
        migrate_legacy_participants,
        update_nn_performance_summaries,
    )
    HAS_COMPOSITE_ELO_MIGRATION = True
except ImportError:
    HAS_COMPOSITE_ELO_MIGRATION = False

# =============================================================================
# Online Learning (December 2025)
# =============================================================================
# Provides continuous learning during gameplay:
# - EBMO online learning with TD-Energy and outcome-weighted loss
# - Rolling buffer for stability
# - Conservative learning rates to prevent catastrophic forgetting

try:
    from app.training.online_learning import (
        EBMOOnlineAI,
        EBMOOnlineConfig,
        EBMOOnlineLearner,
        OnlineLearner,
        OnlineLearningConfig,
        OnlineLearningMetrics,
        OnlineLearningStats,
        TDStepLearner,
        Transition,
        GameRecord,
        create_online_learner,
        get_online_learning_config,
        HAS_EBMO_ONLINE,
    )
    HAS_ONLINE_LEARNING = True
except ImportError:
    HAS_ONLINE_LEARNING = False
    HAS_EBMO_ONLINE = False

# =============================================================================
# Curriculum Learning (December 2025)
# =============================================================================
# Provides progressive difficulty training with automatic stage advancement

try:
    from app.training.curriculum import (
        CurriculumController,
        CurriculumStage,
        CurriculumState,
        create_default_curriculum,
    )
    HAS_CURRICULUM = True
except ImportError:
    HAS_CURRICULUM = False

# =============================================================================
# Unified Curriculum Service (December 2025 Phase 4)
# =============================================================================
# Single source of truth for curriculum weights, consolidating:
# - CurriculumFeedback: win rate, Elo trend, weak opponents
# - FeedbackAccelerator: momentum state
# - ImprovementOptimizer: promotion success/failure

try:
    from app.training.unified_curriculum_service import (
        UnifiedCurriculumService,
        get_unified_curriculum_service,
        get_unified_curriculum_weight,
        get_unified_curriculum_weights,
    )
    HAS_UNIFIED_CURRICULUM = True
except ImportError:
    HAS_UNIFIED_CURRICULUM = False

# =============================================================================
# Knowledge Distillation (December 2025)
# =============================================================================
# Provides knowledge transfer from teacher models to student models:
# - Soft target loss with temperature scaling
# - Ensemble teacher support for combining multiple models
# - Checkpoint ensemble distillation for model compression

try:
    from app.training.distillation import (
        DistillationConfig,
        DistillationTrainer,
        EnsembleTeacher,
        SoftTargetLoss,
        create_distillation_trainer,
        distill_checkpoint_ensemble,
    )
    HAS_DISTILLATION = True
except (ImportError, AttributeError):
    # Dec 31, 2025: Also catch AttributeError for when torch.nn is None
    # This happens when distillation.py is imported but PyTorch is unavailable
    HAS_DISTILLATION = False

# =============================================================================
# Modular Training Components (December 2025)
# =============================================================================
# Extracted from monolithic train.py for better maintainability:
# - model_factory: Neural network model creation
# - data_loader_factory: Data loader creation and configuration
# - train_setup: Fault tolerance and device setup
# NOTE: training_config_resolver.py was removed Jan 2026 (unused, superseded by
# train_config_resolver.py which is used by train_components.py)

try:
    from app.training.model_factory import (
        ModelConfig,
        compute_in_channels,
        count_parameters,
        create_model,
        get_effective_architecture as get_model_architecture,
        load_model_weights,
        log_model_summary,
    )
    HAS_MODEL_FACTORY = True
except ImportError:
    HAS_MODEL_FACTORY = False

try:
    from app.training.data_loader_factory import (
        DataLoaderConfig,
        DataLoaderResult,
        collect_data_paths,
        create_data_loaders,
        create_standard_loaders,
        create_streaming_loaders,
        should_use_streaming,
        validate_dataset_metadata,
    )
    HAS_DATA_LOADER_FACTORY = True
except ImportError:
    HAS_DATA_LOADER_FACTORY = False

try:
    from app.training.train_setup import (
        FaultToleranceComponents,
        FaultToleranceConfig,
        TrainingState,
        compute_effective_lr,
        get_device,
        setup_fault_tolerance,
        setup_graceful_shutdown,
    )
    HAS_TRAIN_SETUP = True
except ImportError:
    HAS_TRAIN_SETUP = False

# High-tier training config (December 2025 - 2000+ Elo target)
try:
    from app.training.high_tier_config import (
        ALL_CONFIGS,
        CANONICAL_CONFIGS,
        HighTierSelfplayConfig,
        HighTierTrainingConfig,
        TIER_ELO_TARGETS,
        get_crossboard_promotion_config,
        get_engine_mode_for_tier,
        get_high_tier_selfplay_config,
        get_high_tier_training_config,
        should_use_gumbel_engine,
    )
    HAS_HIGH_TIER_CONFIG = True
except ImportError:
    HAS_HIGH_TIER_CONFIG = False

# High-tier orchestrator factory (December 2025)
try:
    from app.training.unified_orchestrator import (
        create_high_tier_orchestrator,
        get_high_tier_config,
    )
    HAS_HIGH_TIER_ORCHESTRATOR = True
except ImportError:
    HAS_HIGH_TIER_ORCHESTRATOR = False

# =============================================================================
# Training Configuration Objects (December 2025)
# =============================================================================
# Structured config objects to reduce train_model() 96-parameter signature
# See TRAIN_REFACTORING.md for full decomposition strategy

try:
    from app.training.train_config import (
        AugmentationConfig,
        CheckpointConfig,
        DistributedConfig as TrainingDistributedConfig,  # Alias to avoid conflict
        EarlyStoppingConfig,
        EnhancementConfig,
        FaultToleranceConfig as TrainingFaultToleranceConfig,  # Alias to avoid conflict
        FullTrainingConfig,
        HeartbeatConfig,
        LearningRateConfig,
        MixedPrecisionConfig,
        ModelArchConfig,
        TrainingDataConfig,
        config_from_legacy_params,
    )
    HAS_TRAIN_CONFIG = True
except ImportError:
    HAS_TRAIN_CONFIG = False

# =============================================================================
# Training Validation Utilities (December 2025)
# =============================================================================
# Extracted validation logic from train.py for testability

try:
    from app.training.train_validation import (
        FreshnessResult,
        ValidationResult,
        validate_data_checksums,
        validate_training_data_files,
        validate_training_data_freshness,
    )
    HAS_TRAIN_VALIDATION = True
except ImportError:
    HAS_TRAIN_VALIDATION = False

"""Training module for RingRift AI.

This package provides training infrastructure including:
- Promotion controller for model promotion decisions
- Elo service for rating management
- Model registry for lifecycle tracking
- Tier promotion for difficulty ladder
- Consolidated training components (December 2025)

Usage:
    # Promotion controller
    from app.training import PromotionController, get_promotion_controller

    # Integrated enhancements
    from app.training import IntegratedTrainingManager, create_integrated_manager

    # Consolidated modules
    from app.training import (
        UnifiedCheckpointManager,
        UnifiedDistributedTrainer,
        UnifiedTrainingOrchestrator,
    )
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

__all__ = [
    "HAS_PROMOTION_CONTROLLER",
    "HAS_INTEGRATED_ENHANCEMENTS",
    "HAS_CHECKPOINT_UNIFIED",
    "HAS_DISTRIBUTED_UNIFIED",
    "HAS_DISTRIBUTED_HELPERS",
    "HAS_TEMPERATURE_SCHEDULING",
    "HAS_ORCHESTRATOR",
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

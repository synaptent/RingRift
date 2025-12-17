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

if HAS_ORCHESTRATOR:
    __all__.extend([
        "UnifiedTrainingOrchestrator",
        "OrchestratorConfig",
    ])

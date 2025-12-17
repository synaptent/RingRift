"""Unified AI Loop Module.

This package provides the unified AI improvement loop coordinator.
It manages data collection, training, evaluation, and model promotion.

Backward Compatibility:
    All types that were previously in unified_ai_loop.py are re-exported
    from this package for backward compatibility.

Module Structure:
    - config.py: Configuration dataclasses and event types
    - evaluation.py: Model evaluation and pruning services
    - curriculum.py: Adaptive curriculum management
    - promotion.py: Model promotion with holdout validation
"""

# Re-export configuration classes
from .config import (
    # Configuration dataclasses
    DataIngestionConfig,
    TrainingConfig,
    EvaluationConfig,
    PromotionConfig,
    CurriculumConfig,
    PBTConfig,
    NASConfig,
    PERConfig,
    FeedbackConfig,
    P2PClusterConfig,
    ModelPruningConfig,
    UnifiedLoopConfig,
    # Event types
    DataEventType,
    DataEvent,
    # State classes
    HostState,
    ConfigState,
)

# Re-export service classes (Phase 2 refactoring)
from .evaluation import ModelPruningService
from .curriculum import AdaptiveCurriculum
from .promotion import ModelPromoter
from .tournament import ShadowTournamentService
from .data_collection import StreamingDataCollector
from .training import TrainingScheduler

__all__ = [
    # Configuration
    'DataIngestionConfig',
    'TrainingConfig',
    'EvaluationConfig',
    'PromotionConfig',
    'CurriculumConfig',
    'PBTConfig',
    'NASConfig',
    'PERConfig',
    'FeedbackConfig',
    'P2PClusterConfig',
    'ModelPruningConfig',
    'UnifiedLoopConfig',
    # Events
    'DataEventType',
    'DataEvent',
    # State
    'HostState',
    'ConfigState',
    # Services (Phase 2)
    'ModelPruningService',
    'AdaptiveCurriculum',
    'ModelPromoter',
    'ShadowTournamentService',
    'StreamingDataCollector',
    'TrainingScheduler',
]

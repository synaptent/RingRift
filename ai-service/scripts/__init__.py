"""Unified AI Loop Module.

This package provides the unified AI improvement loop coordinator.
It manages data collection, training, evaluation, and model promotion.

Backward Compatibility:
    All types that were previously in unified_ai_loop.py are re-exported
    from this package for backward compatibility.

Module Structure:
    - unified_loop/config.py: Configuration dataclasses and event types
    - unified_loop/evaluation.py: Model evaluation and pruning services
    - unified_loop/curriculum.py: Adaptive curriculum management
    - unified_loop/promotion.py: Model promotion with holdout validation
"""

# Re-export configuration classes from unified_loop subpackage
from .unified_loop.config import (
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
from .unified_loop.evaluation import ModelPruningService
from .unified_loop.curriculum import AdaptiveCurriculum
from .unified_loop.promotion import ModelPromoter
from .unified_loop.tournament import ShadowTournamentService
from .unified_loop.data_collection import StreamingDataCollector
from .unified_loop.training import TrainingScheduler

__all__ = [
    'AdaptiveCurriculum',
    'ConfigState',
    'CurriculumConfig',
    'DataEvent',
    # Events
    'DataEventType',
    # Configuration
    'DataIngestionConfig',
    'EvaluationConfig',
    'FeedbackConfig',
    # State
    'HostState',
    'ModelPromoter',
    'ModelPruningConfig',
    # Services (Phase 2)
    'ModelPruningService',
    'NASConfig',
    'P2PClusterConfig',
    'PBTConfig',
    'PERConfig',
    'PromotionConfig',
    'ShadowTournamentService',
    'StreamingDataCollector',
    'TrainingConfig',
    'TrainingScheduler',
    'UnifiedLoopConfig',
]

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

from importlib import import_module
import os

_skip = os.getenv("RINGRIFT_SKIP_SCRIPT_INIT_IMPORTS", "").strip().lower()
_SKIP_IMPORTS = _skip in ("1", "true", "yes", "on")

_EXPORTS = {
    "AdaptiveCurriculum": (".unified_loop.curriculum", "AdaptiveCurriculum"),
    "ConfigState": (".unified_loop.config", "ConfigState"),
    "CurriculumConfig": (".unified_loop.config", "CurriculumConfig"),
    "DataEvent": (".unified_loop.config", "DataEvent"),
    "DataEventType": (".unified_loop.config", "DataEventType"),
    "DataIngestionConfig": (".unified_loop.config", "DataIngestionConfig"),
    "EvaluationConfig": (".unified_loop.config", "EvaluationConfig"),
    "FeedbackConfig": (".unified_loop.config", "FeedbackConfig"),
    "HostState": (".unified_loop.config", "HostState"),
    "ModelPromoter": (".unified_loop.promotion", "ModelPromoter"),
    "ModelPruningConfig": (".unified_loop.config", "ModelPruningConfig"),
    "ModelPruningService": (".unified_loop.evaluation", "ModelPruningService"),
    "NASConfig": (".unified_loop.config", "NASConfig"),
    "P2PClusterConfig": (".unified_loop.config", "P2PClusterConfig"),
    "PBTConfig": (".unified_loop.config", "PBTConfig"),
    "PERConfig": (".unified_loop.config", "PERConfig"),
    "PromotionConfig": (".unified_loop.config", "PromotionConfig"),
    "ShadowTournamentService": (".unified_loop.tournament", "ShadowTournamentService"),
    "StreamingDataCollector": (".unified_loop.data_collection", "StreamingDataCollector"),
    "TrainingConfig": (".unified_loop.config", "TrainingConfig"),
    "TrainingScheduler": (".unified_loop.training", "TrainingScheduler"),
    "UnifiedLoopConfig": (".unified_loop.config", "UnifiedLoopConfig"),
}

__all__ = [] if _SKIP_IMPORTS else list(_EXPORTS)


def __getattr__(name: str):
    if _SKIP_IMPORTS or name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

"""Training pipeline orchestration (December 2025).

DEPRECATED: Import directly from app.coordination.data_pipeline_orchestrator
and app.coordination.pipeline_actions instead.
This module will be removed in Q2 2026.

Consolidates pipeline-related functionality from:
- data_pipeline_orchestrator.py (pipeline orchestration)
- pipeline_actions.py (stage action invokers)

Usage (DEPRECATED):
    from app.coordination.core.pipeline import (
        DataPipelineOrchestrator,
        get_pipeline_orchestrator,
        PipelineStage,
    )

Recommended:
    from app.coordination.data_pipeline_orchestrator import (
        DataPipelineOrchestrator,
        get_pipeline_orchestrator,
        PipelineStage,
    )
"""

from __future__ import annotations

import warnings

warnings.warn(
    "app.coordination.core.pipeline is deprecated. "
    "Import from app.coordination.data_pipeline_orchestrator instead. "
    "This module will be removed in Q2 2026.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from data_pipeline_orchestrator
from app.coordination.data_pipeline_orchestrator import (
    DataPipelineOrchestrator,
    get_pipeline_orchestrator,
    get_pipeline_status,
    get_current_pipeline_stage,
    PipelineStage,
    PipelineStats,
    IterationRecord,
)

# Re-export from pipeline_actions
from app.coordination.pipeline_actions import (
    trigger_npz_export,
    trigger_evaluation,
    trigger_data_sync,
    ActionConfig,
    ActionPriority,
    StageCompletionResult,
)

__all__ = [
    # From data_pipeline_orchestrator
    "DataPipelineOrchestrator",
    "get_pipeline_orchestrator",
    "get_pipeline_status",
    "get_current_pipeline_stage",
    "PipelineStage",
    "PipelineStats",
    "IterationRecord",
    # From pipeline_actions
    "trigger_npz_export",
    "trigger_evaluation",
    "trigger_data_sync",
    "ActionConfig",
    "ActionPriority",
    "StageCompletionResult",
]

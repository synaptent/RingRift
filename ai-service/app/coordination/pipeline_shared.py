"""Shared pipeline enums, constants, and dataclasses."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

MAX_STAGE_RETRIES = 3
STAGE_RETRY_DELAY_SECONDS = 300.0
STAGE_RETRY_BACKOFF_MULTIPLIER = 2.0


class PipelineStage(Enum):
    """Pipeline stages in execution order."""

    IDLE = "idle"
    SELFPLAY = "selfplay"
    DATA_SYNC = "data_sync"
    NPZ_EXPORT = "npz_export"
    NPZ_COMBINATION = "npz_combination"
    TRAINING = "training"
    EVALUATION = "evaluation"
    PROMOTION = "promotion"
    COMPLETE = "complete"


class OperationMode(Enum):
    """Operation mode for graceful degradation."""

    FULL = "full"
    DEGRADED = "degraded"
    LOCAL_ONLY = "local"


@dataclass
class StageTransition:
    """Record of a stage transition."""

    from_stage: PipelineStage
    to_stage: PipelineStage
    iteration: int
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IterationRecord:
    """Record of a complete pipeline iteration."""

    iteration: int
    start_time: float
    end_time: float = 0.0
    success: bool = False
    stages_completed: list[str] = field(default_factory=list)
    games_generated: int = 0
    model_id: str | None = None
    elo_delta: float = 0.0
    promoted: bool = False
    error: str | None = None

    @property
    def duration(self) -> float:
        """Get iteration duration in seconds."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time


@dataclass
class PipelineStats:
    """Aggregate pipeline statistics."""

    iterations_completed: int = 0
    iterations_failed: int = 0
    total_games_generated: int = 0
    total_models_trained: int = 0
    promotions: int = 0
    average_iteration_duration: float = 0.0
    stage_durations: dict[str, float] = field(default_factory=dict)
    last_activity_time: float = 0.0


__all__ = [
    "IterationRecord",
    "MAX_STAGE_RETRIES",
    "OperationMode",
    "PipelineStage",
    "PipelineStats",
    "STAGE_RETRY_BACKOFF_MULTIPLIER",
    "STAGE_RETRY_DELAY_SECONDS",
    "StageTransition",
]

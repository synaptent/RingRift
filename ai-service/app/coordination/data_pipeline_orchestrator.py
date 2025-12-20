"""DataPipelineOrchestrator - Unified pipeline stage coordination (December 2025).

This module provides centralized monitoring and coordination of the self-improvement
pipeline stages. It tracks stage transitions, coordinates downstream triggering,
and provides pipeline-wide observability.

Event Integration:
- Subscribes to SELFPLAY_COMPLETE: Start data sync
- Subscribes to SYNC_COMPLETE: Trigger NPZ export
- Subscribes to NPZ_EXPORT_COMPLETE: Ready for training
- Subscribes to TRAINING_COMPLETE: Start evaluation
- Subscribes to EVALUATION_COMPLETE: Consider promotion
- Subscribes to PROMOTION_COMPLETE: Update curriculum

Pipeline Stages:
    SELFPLAY -> SYNC -> NPZ_EXPORT -> TRAINING -> EVALUATION -> PROMOTION

Usage:
    from app.coordination.data_pipeline_orchestrator import (
        DataPipelineOrchestrator,
        wire_pipeline_events,
        get_pipeline_orchestrator,
    )

    # Wire pipeline events
    orchestrator = wire_pipeline_events()

    # Get pipeline status
    status = orchestrator.get_status()
    print(f"Current stage: {status['current_stage']}")
    print(f"Iterations completed: {status['iterations_completed']}")

    # Get stage timing metrics
    metrics = orchestrator.get_stage_metrics()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline stages in execution order."""

    IDLE = "idle"
    SELFPLAY = "selfplay"
    DATA_SYNC = "data_sync"
    NPZ_EXPORT = "npz_export"
    TRAINING = "training"
    EVALUATION = "evaluation"
    PROMOTION = "promotion"
    COMPLETE = "complete"


@dataclass
class StageTransition:
    """Record of a stage transition."""

    from_stage: PipelineStage
    to_stage: PipelineStage
    iteration: int
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IterationRecord:
    """Record of a complete pipeline iteration."""

    iteration: int
    start_time: float
    end_time: float = 0.0
    success: bool = False
    stages_completed: List[str] = field(default_factory=list)
    games_generated: int = 0
    model_id: Optional[str] = None
    elo_delta: float = 0.0
    promoted: bool = False
    error: Optional[str] = None

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
    stage_durations: Dict[str, float] = field(default_factory=dict)
    last_activity_time: float = 0.0


class DataPipelineOrchestrator:
    """Orchestrates the self-improvement pipeline stages.

    Tracks stage transitions, provides coordination between stages,
    and maintains pipeline-wide metrics and observability.
    """

    def __init__(
        self,
        max_history: int = 100,
        auto_trigger: bool = False,  # Automatically trigger downstream stages
    ):
        """Initialize DataPipelineOrchestrator.

        Args:
            max_history: Maximum iteration records to retain
            auto_trigger: If True, automatically trigger downstream stages
        """
        self.max_history = max_history
        self.auto_trigger = auto_trigger

        # Current pipeline state
        self._current_stage = PipelineStage.IDLE
        self._current_iteration = 0

        # Iteration tracking
        self._iteration_records: Dict[int, IterationRecord] = {}
        self._completed_iterations: List[IterationRecord] = []

        # Stage timing
        self._stage_start_times: Dict[PipelineStage, float] = {}
        self._stage_durations: Dict[PipelineStage, List[float]] = {}

        # Transition history
        self._transitions: List[StageTransition] = []

        # Statistics
        self._total_games = 0
        self._total_models = 0
        self._total_promotions = 0

        # Subscription state
        self._subscribed = False

        # Callbacks for stage transitions
        self._stage_callbacks: Dict[PipelineStage, List[Callable]] = {}

        # Quality distribution tracking (December 2025)
        self._quality_distribution: Dict[str, float] = {}  # level -> percentage
        self._last_quality_update: float = 0.0
        self._cache_invalidation_count: int = 0
        self._pending_cache_refresh: bool = False

        # Optimization tracking (December 2025)
        self._active_optimization: Optional[str] = None  # "cmaes" or "nas"
        self._optimization_run_id: Optional[str] = None
        self._optimization_start_time: float = 0.0

        # Resource constraint tracking (December 2025)
        self._paused: bool = False
        self._pause_reason: Optional[str] = None
        self._pause_time: float = 0.0
        self._resource_constraints: Dict[str, Dict] = {}  # resource_type -> constraint_info
        self._backpressure_active: bool = False

    def subscribe_to_events(self) -> bool:
        """Subscribe to pipeline stage events.

        Returns:
            True if successfully subscribed
        """
        if self._subscribed:
            return True

        try:
            from app.coordination.stage_events import StageEvent, get_event_bus

            bus = get_event_bus()

            # Subscribe to all pipeline stage events
            bus.subscribe(StageEvent.SELFPLAY_COMPLETE, self._on_selfplay_complete)
            bus.subscribe(
                StageEvent.CANONICAL_SELFPLAY_COMPLETE, self._on_selfplay_complete
            )
            bus.subscribe(StageEvent.GPU_SELFPLAY_COMPLETE, self._on_selfplay_complete)
            bus.subscribe(StageEvent.SYNC_COMPLETE, self._on_sync_complete)
            bus.subscribe(StageEvent.NPZ_EXPORT_COMPLETE, self._on_npz_export_complete)
            bus.subscribe(StageEvent.TRAINING_STARTED, self._on_training_started)
            bus.subscribe(StageEvent.TRAINING_COMPLETE, self._on_training_complete)
            bus.subscribe(StageEvent.TRAINING_FAILED, self._on_training_failed)
            bus.subscribe(StageEvent.EVALUATION_COMPLETE, self._on_evaluation_complete)
            bus.subscribe(StageEvent.PROMOTION_COMPLETE, self._on_promotion_complete)
            bus.subscribe(StageEvent.ITERATION_COMPLETE, self._on_iteration_complete)

            self._subscribed = True
            logger.info("[DataPipelineOrchestrator] Subscribed to stage events")
            return True

        except ImportError:
            logger.warning("[DataPipelineOrchestrator] stage_events not available")
            return False
        except Exception as e:
            logger.error(f"[DataPipelineOrchestrator] Failed to subscribe: {e}")
            return False

    def subscribe_to_data_events(self) -> bool:
        """Subscribe to DataEventBus events (December 2025).

        Returns:
            True if successfully subscribed
        """
        try:
            from app.distributed.data_events import DataEventType, get_event_bus

            bus = get_event_bus()

            # Subscribe to quality and cache events
            bus.subscribe(
                DataEventType.QUALITY_DISTRIBUTION_CHANGED,
                self._on_quality_distribution_changed,
            )
            bus.subscribe(
                DataEventType.CACHE_INVALIDATED,
                self._on_cache_invalidated,
            )

            # Subscribe to optimization events (December 2025)
            bus.subscribe(
                DataEventType.CMAES_TRIGGERED,
                self._on_optimization_triggered,
            )
            bus.subscribe(
                DataEventType.NAS_TRIGGERED,
                self._on_optimization_triggered,
            )

            # Subscribe to resource constraint events (December 2025)
            bus.subscribe(
                DataEventType.RESOURCE_CONSTRAINT_DETECTED,
                self._on_resource_constraint_detected,
            )
            bus.subscribe(
                DataEventType.BACKPRESSURE_ACTIVATED,
                self._on_backpressure_activated,
            )
            bus.subscribe(
                DataEventType.BACKPRESSURE_RELEASED,
                self._on_backpressure_released,
            )

            logger.info("[DataPipelineOrchestrator] Subscribed to data events")
            return True

        except ImportError:
            logger.warning("[DataPipelineOrchestrator] data_events not available")
            return False
        except Exception as e:
            logger.error(f"[DataPipelineOrchestrator] Failed to subscribe to data events: {e}")
            return False

    def _transition_to(
        self,
        new_stage: PipelineStage,
        iteration: int,
        success: bool = True,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Record a stage transition."""
        old_stage = self._current_stage

        # Calculate duration of previous stage
        duration = 0.0
        if old_stage in self._stage_start_times:
            duration = time.time() - self._stage_start_times[old_stage]
            # Record stage duration
            if old_stage not in self._stage_durations:
                self._stage_durations[old_stage] = []
            self._stage_durations[old_stage].append(duration)

        # Record transition
        transition = StageTransition(
            from_stage=old_stage,
            to_stage=new_stage,
            iteration=iteration,
            success=success,
            duration_seconds=duration,
            metadata=metadata or {},
        )
        self._transitions.append(transition)

        # Trim history
        if len(self._transitions) > self.max_history * 10:
            self._transitions = self._transitions[-self.max_history * 10 :]

        # Update current state
        self._current_stage = new_stage
        self._current_iteration = iteration
        self._stage_start_times[new_stage] = time.time()

        # Update iteration record
        if iteration in self._iteration_records:
            record = self._iteration_records[iteration]
            record.stages_completed.append(new_stage.value)

        logger.info(
            f"[DataPipelineOrchestrator] Stage transition: {old_stage.value} -> "
            f"{new_stage.value} (iteration {iteration})"
        )

        # Invoke stage callbacks
        for callback in self._stage_callbacks.get(new_stage, []):
            try:
                callback(new_stage, iteration)
            except Exception as e:
                logger.error(f"[DataPipelineOrchestrator] Callback error: {e}")

    def _ensure_iteration_record(self, iteration: int) -> IterationRecord:
        """Ensure an iteration record exists."""
        if iteration not in self._iteration_records:
            self._iteration_records[iteration] = IterationRecord(
                iteration=iteration,
                start_time=time.time(),
            )
        return self._iteration_records[iteration]

    async def _on_selfplay_complete(self, result) -> None:
        """Handle selfplay completion."""
        iteration = result.iteration
        self._ensure_iteration_record(iteration)

        self._iteration_records[iteration].games_generated = result.games_generated
        self._total_games += result.games_generated

        if result.success:
            self._transition_to(
                PipelineStage.DATA_SYNC,
                iteration,
                metadata={"games_generated": result.games_generated},
            )
        else:
            self._transition_to(
                PipelineStage.IDLE,
                iteration,
                success=False,
                metadata={"error": result.error},
            )

    async def _on_sync_complete(self, result) -> None:
        """Handle data sync completion."""
        iteration = result.iteration

        if result.success:
            self._transition_to(
                PipelineStage.NPZ_EXPORT,
                iteration,
                metadata=result.metadata,
            )
        else:
            self._transition_to(
                PipelineStage.IDLE,
                iteration,
                success=False,
                metadata={"error": result.error},
            )

    async def _on_npz_export_complete(self, result) -> None:
        """Handle NPZ export completion."""
        iteration = result.iteration

        if result.success:
            self._transition_to(
                PipelineStage.TRAINING,
                iteration,
                metadata=result.metadata,
            )
        else:
            self._transition_to(
                PipelineStage.IDLE,
                iteration,
                success=False,
                metadata={"error": result.error},
            )

    async def _on_training_started(self, result) -> None:
        """Handle training start."""
        iteration = result.iteration
        self._ensure_iteration_record(iteration)
        self._stage_start_times[PipelineStage.TRAINING] = time.time()

    async def _on_training_complete(self, result) -> None:
        """Handle training completion."""
        iteration = result.iteration
        record = self._ensure_iteration_record(iteration)

        record.model_id = result.model_id
        self._total_models += 1

        if result.success:
            self._transition_to(
                PipelineStage.EVALUATION,
                iteration,
                metadata={
                    "model_id": result.model_id,
                    "train_loss": result.train_loss,
                    "val_loss": result.val_loss,
                },
            )
        else:
            self._transition_to(
                PipelineStage.IDLE,
                iteration,
                success=False,
                metadata={"error": result.error},
            )

    async def _on_training_failed(self, result) -> None:
        """Handle training failure."""
        iteration = result.iteration
        record = self._ensure_iteration_record(iteration)
        record.error = result.error

        self._transition_to(
            PipelineStage.IDLE,
            iteration,
            success=False,
            metadata={"error": result.error},
        )

    async def _on_evaluation_complete(self, result) -> None:
        """Handle evaluation completion."""
        iteration = result.iteration
        record = self._ensure_iteration_record(iteration)

        record.elo_delta = result.elo_delta or 0.0

        if result.success:
            self._transition_to(
                PipelineStage.PROMOTION,
                iteration,
                metadata={
                    "win_rate": result.win_rate,
                    "elo_delta": result.elo_delta,
                },
            )
        else:
            self._transition_to(
                PipelineStage.IDLE,
                iteration,
                success=False,
                metadata={"error": result.error},
            )

    async def _on_promotion_complete(self, result) -> None:
        """Handle promotion completion."""
        iteration = result.iteration
        record = self._ensure_iteration_record(iteration)

        record.promoted = result.promoted
        if result.promoted:
            self._total_promotions += 1

        self._transition_to(
            PipelineStage.COMPLETE,
            iteration,
            metadata={
                "promoted": result.promoted,
                "reason": result.promotion_reason,
            },
        )

    async def _on_iteration_complete(self, result) -> None:
        """Handle iteration completion."""
        iteration = result.iteration
        if iteration in self._iteration_records:
            record = self._iteration_records[iteration]
            record.end_time = time.time()
            record.success = result.success

            # Move to completed history
            self._completed_iterations.append(record)
            if len(self._completed_iterations) > self.max_history:
                self._completed_iterations = self._completed_iterations[
                    -self.max_history :
                ]

            del self._iteration_records[iteration]

        self._transition_to(PipelineStage.IDLE, iteration + 1)

    # =========================================================================
    # Quality and Cache Event Handlers (December 2025)
    # =========================================================================

    async def _on_quality_distribution_changed(self, event) -> None:
        """Handle QUALITY_DISTRIBUTION_CHANGED - track quality changes."""
        payload = event.payload

        # Update quality distribution
        distribution = payload.get("distribution", {})
        if distribution:
            self._quality_distribution = distribution
            self._last_quality_update = time.time()

        high_quality = distribution.get("high", 0.0)
        low_quality = distribution.get("low", 0.0)

        logger.info(
            f"[DataPipelineOrchestrator] Quality distribution updated: "
            f"high={high_quality:.1%}, low={low_quality:.1%}"
        )

        # If quality distribution shifted significantly, may need to adjust training
        if low_quality > 0.5:
            logger.warning(
                "[DataPipelineOrchestrator] Low quality data >50% - "
                "consider curriculum adjustments"
            )

    async def _on_cache_invalidated(self, event) -> None:
        """Handle CACHE_INVALIDATED - track cache changes for data pipeline."""
        payload = event.payload

        invalidation_type = payload.get("invalidation_type", "")
        count = payload.get("count", 0)
        target_id = payload.get("target_id", "")

        self._cache_invalidation_count += count

        # Check if this affects NPZ data caches
        affected_types = ["npz_data", "feature_cache"]
        if invalidation_type == "model":
            # Model cache invalidation may require NPZ re-export
            self._pending_cache_refresh = True
            logger.info(
                f"[DataPipelineOrchestrator] Cache invalidated for model {target_id}: "
                f"{count} entries - NPZ data may need refresh"
            )
        else:
            logger.debug(
                f"[DataPipelineOrchestrator] Cache invalidated: "
                f"{invalidation_type}={target_id}, count={count}"
            )

    async def _on_optimization_triggered(self, event) -> None:
        """Handle CMAES_TRIGGERED or NAS_TRIGGERED - track optimization state."""
        payload = event.payload
        event_type = str(event.event_type.value).lower()

        # Determine optimization type from event
        if "cmaes" in event_type:
            opt_type = "cmaes"
        elif "nas" in event_type:
            opt_type = "nas"
        else:
            opt_type = "unknown"

        run_id = payload.get("run_id", "")
        reason = payload.get("reason", "")

        self._active_optimization = opt_type
        self._optimization_run_id = run_id
        self._optimization_start_time = time.time()

        logger.info(
            f"[DataPipelineOrchestrator] {opt_type.upper()} optimization triggered: "
            f"run_id={run_id}, reason={reason}"
        )

        # Note: Pipeline can continue but training may be coordinated differently
        # during optimization runs (e.g., different hyperparameters being tested)

    # =========================================================================
    # Resource Constraint Event Handlers (December 2025)
    # =========================================================================

    async def _on_resource_constraint_detected(self, event) -> None:
        """Handle RESOURCE_CONSTRAINT_DETECTED - pause pipeline on critical constraints."""
        payload = event.payload

        resource_type = payload.get("resource_type", "unknown")
        severity = payload.get("severity", "warning")
        current_value = payload.get("current_value", 0)
        threshold = payload.get("threshold", 0)
        node_id = payload.get("node_id", "")

        # Track the constraint
        self._resource_constraints[resource_type] = {
            "severity": severity,
            "current_value": current_value,
            "threshold": threshold,
            "node_id": node_id,
            "time": time.time(),
        }

        logger.warning(
            f"[DataPipelineOrchestrator] Resource constraint detected: "
            f"{resource_type}={current_value} (threshold={threshold}, severity={severity})"
        )

        # Pause on critical constraints during resource-intensive stages
        critical_stages = {PipelineStage.TRAINING, PipelineStage.NPZ_EXPORT}
        if severity == "critical" and self._current_stage in critical_stages:
            await self._pause_pipeline(
                reason=f"Critical {resource_type} constraint: {current_value}/{threshold}"
            )

    async def _on_backpressure_activated(self, event) -> None:
        """Handle BACKPRESSURE_ACTIVATED - pause pipeline under heavy load."""
        payload = event.payload

        source = payload.get("source", "unknown")
        level = payload.get("level", "unknown")

        self._backpressure_active = True

        logger.warning(
            f"[DataPipelineOrchestrator] Backpressure activated: source={source}, level={level}"
        )

        # Pause if backpressure is severe
        if level in ("high", "critical"):
            await self._pause_pipeline(reason=f"Backpressure from {source}: {level}")

    async def _on_backpressure_released(self, event) -> None:
        """Handle BACKPRESSURE_RELEASED - potentially resume pipeline."""
        payload = event.payload

        source = payload.get("source", "unknown")

        self._backpressure_active = False

        logger.info(f"[DataPipelineOrchestrator] Backpressure released: source={source}")

        # Auto-resume if paused due to backpressure and no other constraints
        if self._paused and "Backpressure" in (self._pause_reason or ""):
            if not self._has_critical_constraints():
                await self._resume_pipeline()

    async def _pause_pipeline(self, reason: str) -> None:
        """Pause the pipeline due to resource constraints."""
        if self._paused:
            return  # Already paused

        self._paused = True
        self._pause_reason = reason
        self._pause_time = time.time()

        logger.warning(f"[DataPipelineOrchestrator] Pipeline PAUSED: {reason}")

        # Emit event for other coordinators
        try:
            from app.coordination.event_emitters import emit_resource_constraint_detected

            await emit_resource_constraint_detected(
                resource_type="pipeline_pause",
                severity="critical",
                current_value=1,
                threshold=0,
                action_taken=f"pipeline_paused: {reason}",
            )
        except Exception:
            pass

    async def _resume_pipeline(self) -> None:
        """Resume the pipeline after constraint resolution."""
        if not self._paused:
            return

        pause_duration = time.time() - self._pause_time
        logger.info(
            f"[DataPipelineOrchestrator] Pipeline RESUMED after {pause_duration:.1f}s pause"
        )

        self._paused = False
        self._pause_reason = None
        self._pause_time = 0.0

    def _has_critical_constraints(self) -> bool:
        """Check if any critical resource constraints are active."""
        now = time.time()
        for resource_type, constraint in self._resource_constraints.items():
            # Constraints older than 60s are considered stale
            if now - constraint.get("time", 0) > 60:
                continue
            if constraint.get("severity") == "critical":
                return True
        return False

    def is_paused(self) -> bool:
        """Check if pipeline is currently paused."""
        return self._paused

    def get_pause_info(self) -> Optional[Dict[str, Any]]:
        """Get information about current pause state."""
        if not self._paused:
            return None
        return {
            "paused": True,
            "reason": self._pause_reason,
            "duration_seconds": time.time() - self._pause_time,
            "active_constraints": dict(self._resource_constraints),
            "backpressure_active": self._backpressure_active,
        }

    def clear_resource_constraints(self) -> None:
        """Clear all tracked resource constraints."""
        self._resource_constraints.clear()
        logger.info("[DataPipelineOrchestrator] Resource constraints cleared")

    def clear_optimization_state(self) -> None:
        """Clear active optimization tracking."""
        if self._active_optimization:
            duration = time.time() - self._optimization_start_time
            logger.info(
                f"[DataPipelineOrchestrator] Optimization {self._active_optimization} "
                f"completed after {duration:.1f}s"
            )
        self._active_optimization = None
        self._optimization_run_id = None
        self._optimization_start_time = 0.0

    def is_optimization_active(self) -> bool:
        """Check if optimization is currently active."""
        return self._active_optimization is not None

    def get_active_optimization(self) -> Optional[str]:
        """Get the type of active optimization, or None."""
        return self._active_optimization

    def get_quality_distribution(self) -> Dict[str, float]:
        """Get current quality distribution."""
        return dict(self._quality_distribution)

    def needs_cache_refresh(self) -> bool:
        """Check if pipeline needs cache refresh."""
        return self._pending_cache_refresh

    def clear_cache_refresh_flag(self) -> None:
        """Clear the pending cache refresh flag."""
        if self._pending_cache_refresh:
            self._pending_cache_refresh = False
            logger.info("[DataPipelineOrchestrator] Cache refresh flag cleared")

    def start_iteration(self, iteration: int) -> IterationRecord:
        """Manually start a new pipeline iteration.

        Args:
            iteration: The iteration number

        Returns:
            The created IterationRecord
        """
        record = self._ensure_iteration_record(iteration)
        self._transition_to(PipelineStage.SELFPLAY, iteration)
        return record

    def on_stage_enter(
        self, stage: PipelineStage, callback: Callable[[PipelineStage, int], None]
    ) -> None:
        """Register a callback for when a stage is entered.

        Args:
            stage: The stage to watch
            callback: Function(stage, iteration) to call
        """
        if stage not in self._stage_callbacks:
            self._stage_callbacks[stage] = []
        self._stage_callbacks[stage].append(callback)

    def get_current_stage(self) -> PipelineStage:
        """Get the current pipeline stage."""
        return self._current_stage

    def get_current_iteration(self) -> int:
        """Get the current iteration number."""
        return self._current_iteration

    def get_iteration_record(self, iteration: int) -> Optional[IterationRecord]:
        """Get record for a specific iteration."""
        if iteration in self._iteration_records:
            return self._iteration_records[iteration]
        for record in self._completed_iterations:
            if record.iteration == iteration:
                return record
        return None

    def get_recent_transitions(self, limit: int = 20) -> List[StageTransition]:
        """Get recent stage transitions."""
        return self._transitions[-limit:]

    def get_stage_metrics(self) -> Dict[str, Any]:
        """Get timing metrics for each stage."""
        metrics = {}
        for stage, durations in self._stage_durations.items():
            if durations:
                metrics[stage.value] = {
                    "count": len(durations),
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "total_duration": sum(durations),
                }
        return metrics

    def get_stats(self) -> PipelineStats:
        """Get aggregate pipeline statistics."""
        completed = [r for r in self._completed_iterations if r.success]
        failed = [r for r in self._completed_iterations if not r.success]

        avg_duration = (
            sum(r.duration for r in completed) / len(completed) if completed else 0.0
        )

        stage_avg = {}
        for stage, durations in self._stage_durations.items():
            if durations:
                stage_avg[stage.value] = sum(durations) / len(durations)

        return PipelineStats(
            iterations_completed=len(completed),
            iterations_failed=len(failed),
            total_games_generated=self._total_games,
            total_models_trained=self._total_models,
            promotions=self._total_promotions,
            average_iteration_duration=avg_duration,
            stage_durations=stage_avg,
            last_activity_time=time.time(),
        )

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status for monitoring."""
        stats = self.get_stats()

        # Get active iterations
        active_iterations = list(self._iteration_records.keys())

        return {
            "current_stage": self._current_stage.value,
            "current_iteration": self._current_iteration,
            "active_iterations": active_iterations,
            "iterations_completed": stats.iterations_completed,
            "iterations_failed": stats.iterations_failed,
            "total_games_generated": stats.total_games_generated,
            "total_models_trained": stats.total_models_trained,
            "promotions": stats.promotions,
            "average_iteration_duration": round(stats.average_iteration_duration, 1),
            "stage_durations": {
                k: round(v, 1) for k, v in stats.stage_durations.items()
            },
            "subscribed": self._subscribed,
            "auto_trigger": self.auto_trigger,
            # Quality and cache tracking (December 2025)
            "quality_distribution": dict(self._quality_distribution),
            "pending_cache_refresh": self._pending_cache_refresh,
            "cache_invalidation_count": self._cache_invalidation_count,
            # Optimization tracking (December 2025)
            "active_optimization": self._active_optimization,
            "optimization_run_id": self._optimization_run_id,
            # Resource constraint tracking (December 2025)
            "paused": self._paused,
            "pause_reason": self._pause_reason,
            "backpressure_active": self._backpressure_active,
            "resource_constraints": dict(self._resource_constraints),
        }

    def format_pipeline_report(self) -> str:
        """Format a human-readable pipeline status report."""
        lines = ["=" * 60]
        lines.append("DATA PIPELINE STATUS REPORT")
        lines.append("=" * 60)

        lines.append(f"Current Stage: {self._current_stage.value.upper()}")
        lines.append(f"Current Iteration: {self._current_iteration}")
        lines.append("")

        stats = self.get_stats()
        lines.append(f"Iterations Completed: {stats.iterations_completed}")
        lines.append(f"Iterations Failed: {stats.iterations_failed}")
        lines.append(f"Total Games Generated: {stats.total_games_generated:,}")
        lines.append(f"Models Trained: {stats.total_models_trained}")
        lines.append(f"Promotions: {stats.promotions}")
        lines.append("")

        if stats.stage_durations:
            lines.append("Stage Durations (avg):")
            lines.append("-" * 40)
            for stage, duration in stats.stage_durations.items():
                lines.append(f"  {stage}: {duration:.1f}s")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Singleton and convenience functions
# =============================================================================

_pipeline_orchestrator: Optional[DataPipelineOrchestrator] = None


def get_pipeline_orchestrator() -> DataPipelineOrchestrator:
    """Get the global DataPipelineOrchestrator singleton."""
    global _pipeline_orchestrator
    if _pipeline_orchestrator is None:
        _pipeline_orchestrator = DataPipelineOrchestrator()
    return _pipeline_orchestrator


def wire_pipeline_events(auto_trigger: bool = False) -> DataPipelineOrchestrator:
    """Wire pipeline events to the orchestrator.

    Args:
        auto_trigger: If True, automatically trigger downstream stages

    Returns:
        The wired DataPipelineOrchestrator instance
    """
    global _pipeline_orchestrator
    _pipeline_orchestrator = DataPipelineOrchestrator(auto_trigger=auto_trigger)
    _pipeline_orchestrator.subscribe_to_events()
    _pipeline_orchestrator.subscribe_to_data_events()  # December 2025
    return _pipeline_orchestrator


def get_pipeline_status() -> Dict[str, Any]:
    """Convenience function to get pipeline status."""
    return get_pipeline_orchestrator().get_status()


def get_current_pipeline_stage() -> PipelineStage:
    """Convenience function to get current pipeline stage."""
    return get_pipeline_orchestrator().get_current_stage()


__all__ = [
    "DataPipelineOrchestrator",
    "PipelineStage",
    "StageTransition",
    "IterationRecord",
    "PipelineStats",
    "get_pipeline_orchestrator",
    "wire_pipeline_events",
    "get_pipeline_status",
    "get_current_pipeline_stage",
]

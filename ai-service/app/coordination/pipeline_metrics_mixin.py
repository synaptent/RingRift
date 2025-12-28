"""Pipeline metrics mixin - metrics, status, and health reporting for DataPipelineOrchestrator.

December 2025: Extracted from data_pipeline_orchestrator.py as part of mixin-based refactoring.

This mixin provides metrics, status, and health reporting methods:
- get_metrics: Get orchestrator metrics in protocol-compliant format
- get_stats: Get aggregate pipeline statistics
- get_status: Get orchestrator status for monitoring
- get_health_status: Get pipeline health status for monitoring and alerting
- format_pipeline_report: Format a human-readable pipeline status report
- check_stage_timeout: Check if current stage has timed out
- get_stage_metrics: Get timing metrics for each stage
- get_recent_transitions: Get recent stage transitions
- get_circuit_breaker_status: Get circuit breaker status
- reset_circuit_breaker: Manually reset the circuit breaker
- get_current_stage: Get the current pipeline stage
- get_current_iteration: Get the current iteration number
- get_iteration_record: Get record for a specific iteration

Expected attributes from main class:
- _current_stage: PipelineStage
- _current_iteration: int
- _iteration_records: dict
- _completed_iterations: list
- _stage_start_times: dict
- _stage_durations: dict
- _transitions: list
- _total_games: int
- _total_models: int
- _total_promotions: int
- _subscribed: bool
- auto_trigger: bool
- auto_trigger_sync: bool
- auto_trigger_export: bool
- auto_trigger_training: bool
- auto_trigger_evaluation: bool
- auto_trigger_promotion: bool
- _circuit_breaker: PipelineCircuitBreaker | None
- _quality_distribution: dict
- _pending_cache_refresh: bool
- _cache_invalidation_count: int
- _active_optimization: str | None
- _optimization_run_id: str | None
- _paused: bool
- _pause_reason: str | None
- _backpressure_active: bool
- _resource_constraints: dict
- _coordinator_status: CoordinatorStatus
- _start_time: float
- _events_processed: int
- _errors_count: int
- _last_error: str
- max_history: int
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.coordination.data_pipeline_orchestrator import (
        DataPipelineOrchestrator,
        IterationRecord,
        PipelineStage,
        PipelineStats,
        StageTransition,
    )

logger = logging.getLogger(__name__)


class PipelineMetricsMixin:
    """Mixin providing metrics, status, and health reporting for DataPipelineOrchestrator.

    This mixin provides observability into the pipeline state through
    status reporting, metrics collection, and health checks.
    """

    # Type hints for attributes expected from main class
    if TYPE_CHECKING:
        _current_stage: PipelineStage
        _current_iteration: int
        _iteration_records: dict
        _completed_iterations: list
        _stage_start_times: dict
        _stage_durations: dict
        _transitions: list
        _total_games: int
        _total_models: int
        _total_promotions: int
        _subscribed: bool
        auto_trigger: bool
        auto_trigger_sync: bool
        auto_trigger_export: bool
        auto_trigger_training: bool
        auto_trigger_evaluation: bool
        auto_trigger_promotion: bool
        _circuit_breaker: Any
        _quality_distribution: dict
        _pending_cache_refresh: bool
        _cache_invalidation_count: int
        _active_optimization: str | None
        _optimization_run_id: str | None
        _paused: bool
        _pause_reason: str | None
        _backpressure_active: bool
        _resource_constraints: dict
        _coordinator_status: Any
        _start_time: float
        _events_processed: int
        _errors_count: int
        _last_error: str
        max_history: int
        name: str
        uptime_seconds: float

    # =========================================================================
    # Protocol-Compliant Metrics
    # =========================================================================

    def get_metrics(self) -> dict[str, Any]:
        """Get orchestrator metrics in protocol-compliant format.

        Returns:
            Dictionary of metrics including pipeline-specific stats.
        """
        stats = self.get_stats()
        return {
            "name": self.name,
            "status": self._coordinator_status.value,
            "uptime_seconds": self.uptime_seconds,
            "start_time": self._start_time,
            "events_processed": self._events_processed,
            "errors_count": self._errors_count,
            "last_error": self._last_error,
            # Pipeline-specific metrics
            "current_stage": self._current_stage.value,
            "current_iteration": self._current_iteration,
            "iterations_completed": stats.iterations_completed,
            "iterations_failed": stats.iterations_failed,
            "total_games_generated": stats.total_games_generated,
            "total_models_trained": stats.total_models_trained,
            "promotions": stats.promotions,
            "subscribed": self._subscribed,
            "paused": self._paused,
            "circuit_breaker_state": (
                self._circuit_breaker.state.value if self._circuit_breaker else None
            ),
        }

    # =========================================================================
    # Stage Accessors
    # =========================================================================

    def get_current_stage(self) -> "PipelineStage":
        """Get the current pipeline stage."""
        return self._current_stage

    def get_current_iteration(self) -> int:
        """Get the current iteration number."""
        return self._current_iteration

    def get_iteration_record(self, iteration: int) -> "IterationRecord | None":
        """Get record for a specific iteration."""
        if iteration in self._iteration_records:
            return self._iteration_records[iteration]
        for record in self._completed_iterations:
            if record.iteration == iteration:
                return record
        return None

    def get_recent_transitions(self, limit: int = 20) -> "list[StageTransition]":
        """Get recent stage transitions."""
        return self._transitions[-limit:]

    # =========================================================================
    # Stage Metrics
    # =========================================================================

    def get_stage_metrics(self) -> dict[str, Any]:
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

    # =========================================================================
    # Aggregate Statistics
    # =========================================================================

    def get_stats(self) -> "PipelineStats":
        """Get aggregate pipeline statistics."""
        from app.coordination.data_pipeline_orchestrator import PipelineStats

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

    # =========================================================================
    # Status Reporting
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
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
            # Per-stage auto-trigger controls (December 2025)
            "auto_trigger_sync": self.auto_trigger_sync,
            "auto_trigger_export": self.auto_trigger_export,
            "auto_trigger_training": self.auto_trigger_training,
            "auto_trigger_evaluation": self.auto_trigger_evaluation,
            "auto_trigger_promotion": self.auto_trigger_promotion,
            # Circuit breaker status (December 2025)
            "circuit_breaker": self.get_circuit_breaker_status(),
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

    # =========================================================================
    # Health Status
    # =========================================================================

    def get_health_status(self) -> dict[str, Any]:
        """Get pipeline health status for monitoring and alerting.

        Returns a dict with:
        - healthy: bool - overall health status
        - issues: list[str] - any detected issues
        - stage_health: dict - per-stage health info
        - recommendations: list[str] - suggested actions

        Stage timeout thresholds (seconds):
        - IDLE: no timeout
        - DATA_SYNC: 1800 (30 min)
        - NPZ_EXPORT: 3600 (1 hour)
        - TRAINING: 14400 (4 hours)
        - EVALUATION: 7200 (2 hours)
        - PROMOTION: 600 (10 min)
        """
        from app.coordination.data_pipeline_orchestrator import PipelineStage

        stage_timeouts = {
            PipelineStage.IDLE: float("inf"),
            PipelineStage.DATA_SYNC: 1800,
            PipelineStage.NPZ_EXPORT: 3600,
            PipelineStage.TRAINING: 14400,
            PipelineStage.EVALUATION: 7200,
            PipelineStage.PROMOTION: 600,
            PipelineStage.SELFPLAY: 7200,
            PipelineStage.COMPLETE: float("inf"),
        }

        issues: list[str] = []
        recommendations: list[str] = []
        now = time.time()

        # Check for stuck stage
        stage_duration = 0.0
        if self._current_stage in self._stage_start_times:
            stage_duration = now - self._stage_start_times[self._current_stage]
            timeout = stage_timeouts.get(self._current_stage, 3600)

            if stage_duration > timeout:
                issues.append(
                    f"Stage {self._current_stage.value} stuck for "
                    f"{stage_duration / 60:.1f} min (timeout: {timeout / 60:.1f} min)"
                )
                recommendations.append(
                    f"Consider restarting {self._current_stage.value} stage or checking logs"
                )

        # Check circuit breaker
        cb_status = self.get_circuit_breaker_status()
        if cb_status and cb_status.get("state") == "open":
            issues.append("Circuit breaker is OPEN - pipeline blocked due to failures")
            recommendations.append(
                f"Wait {cb_status.get('time_until_retry', 0):.0f}s for auto-recovery or reset manually"
            )
        elif cb_status and cb_status.get("state") == "half_open":
            issues.append("Circuit breaker is HALF_OPEN - testing recovery")

        # Check error rate
        stats = self.get_stats()
        total_iterations = stats.iterations_completed + stats.iterations_failed
        if total_iterations > 5:
            error_rate = stats.iterations_failed / total_iterations
            if error_rate > 0.3:
                issues.append(f"High error rate: {error_rate:.0%} of iterations failed")
                recommendations.append("Review recent failures and fix root cause")

        # Check if paused
        if self._paused:
            issues.append(f"Pipeline paused: {self._pause_reason or 'unknown reason'}")
            pause_duration = now - getattr(self, "_pause_time", now)
            if pause_duration > 600:
                recommendations.append(
                    f"Pipeline paused for {pause_duration / 60:.1f} min - consider resuming"
                )

        # Check backpressure
        if self._backpressure_active:
            issues.append("Backpressure active - pipeline is throttled")

        # Determine overall health
        healthy = len(issues) == 0

        return {
            "healthy": healthy,
            "status": "healthy" if healthy else "degraded" if len(issues) < 3 else "unhealthy",
            "issues": issues,
            "recommendations": recommendations,
            "stage_health": {
                "current_stage": self._current_stage.value,
                "stage_duration_seconds": round(stage_duration, 1),
                "stage_timeout_seconds": stage_timeouts.get(self._current_stage, 3600),
                "pct_timeout_used": round(
                    stage_duration / stage_timeouts.get(self._current_stage, 3600) * 100, 1
                ) if self._current_stage != PipelineStage.IDLE else 0,
            },
            "circuit_breaker": cb_status,
            "stats": {
                "iterations_completed": stats.iterations_completed,
                "iterations_failed": stats.iterations_failed,
                "error_rate": (
                    stats.iterations_failed / total_iterations if total_iterations > 0 else 0
                ),
            },
            "paused": self._paused,
            "backpressure": self._backpressure_active,
            "timestamp": now,
        }

    def check_stage_timeout(self) -> tuple[bool, str | None]:
        """Check if current stage has timed out.

        Returns:
            Tuple of (timed_out: bool, message: str | None)
        """
        health = self.get_health_status()
        stage_health = health.get("stage_health", {})

        pct_used = stage_health.get("pct_timeout_used", 0)
        if pct_used >= 100:
            return True, health.get("issues", ["Stage timed out"])[0]

        return False, None

    # =========================================================================
    # Circuit Breaker
    # =========================================================================

    def get_circuit_breaker_status(self) -> dict[str, Any] | None:
        """Get circuit breaker status."""
        if self._circuit_breaker:
            return self._circuit_breaker.get_status()
        return None

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker."""
        if self._circuit_breaker:
            self._circuit_breaker.reset()
            logger.info("[DataPipelineOrchestrator] Circuit breaker manually reset")

    # =========================================================================
    # Human-Readable Report
    # =========================================================================

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

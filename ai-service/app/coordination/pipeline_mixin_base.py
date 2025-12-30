"""Base class for DataPipelineOrchestrator mixins.

December 2025: Created to consolidate common patterns across the 4 pipeline mixins:
- PipelineStageMixin
- PipelineEventHandlerMixin
- PipelineTriggerMixin
- PipelineMetricsMixin

Provides:
- Protocol defining expected interface from main class
- Common type hints (shared across all pipeline mixins)
- Utility methods for logging and error handling
- Stage result extraction helpers

This follows the same pattern as SyncMixinBase for the sync mixins.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from app.coordination.data_pipeline_orchestrator import PipelineCircuitBreaker
    from app.coordination.stage_events import PipelineStage

logger = logging.getLogger(__name__)


@runtime_checkable
class DataPipelineOrchestratorProtocol(Protocol):
    """Protocol defining the interface expected from DataPipelineOrchestrator.

    All pipeline mixins expect the main class to implement these attributes
    and methods. This protocol documents the contract explicitly.
    """

    # ==========================================================================
    # Core State (used by all 4 mixins)
    # ==========================================================================

    # Current pipeline state
    _current_stage: PipelineStage
    _current_iteration: int
    _current_board_type: str | None
    _current_num_players: int | None

    # Auto-trigger configuration
    auto_trigger: bool
    auto_trigger_sync: bool
    auto_trigger_export: bool
    auto_trigger_training: bool
    auto_trigger_evaluation: bool
    auto_trigger_promotion: bool

    # ==========================================================================
    # Iteration Tracking (PipelineStageMixin, PipelineMetricsMixin)
    # ==========================================================================

    _iteration_records: dict[int, Any]
    _completed_iterations: list[int]
    _stage_start_times: dict[str, float]
    _stage_durations: dict[str, list[float]]
    _transitions: list[dict[str, Any]]
    max_history: int

    # ==========================================================================
    # Metrics (PipelineStageMixin, PipelineMetricsMixin)
    # ==========================================================================

    _total_games: int
    _total_models: int
    _total_promotions: int

    # ==========================================================================
    # Quality Gate (PipelineStageMixin, PipelineTriggerMixin)
    # ==========================================================================

    quality_gate_enabled: bool
    _last_quality_score: float

    # ==========================================================================
    # Circuit Breaker (PipelineTriggerMixin, PipelineMetricsMixin)
    # ==========================================================================

    _circuit_breaker: PipelineCircuitBreaker | None

    # ==========================================================================
    # Event Handling State (PipelineEventHandlerMixin, PipelineMetricsMixin)
    # ==========================================================================

    _paused: bool
    _pause_reason: str | None
    _backpressure_active: bool
    _resource_constraints: dict[str, Any]
    _stage_metadata: dict[str, Any]

    # Quality distribution
    _quality_distribution: dict[str, float]
    _last_quality_update: float

    # Cache coordination
    _cache_invalidation_count: int
    _pending_cache_refresh: bool

    # Optimization tracking
    _active_optimization: str | None
    _optimization_run_id: str | None
    _optimization_start_time: float

    # ==========================================================================
    # Status (PipelineMetricsMixin)
    # ==========================================================================

    _subscribed: bool
    _start_time: float
    _events_processed: int

    # ==========================================================================
    # Core Methods (used by mixins)
    # ==========================================================================

    def get_config_key(self) -> str:
        """Return current board_type + num_players as config key."""
        ...

    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        """Log with context."""
        ...


class PipelineMixinBase:
    """Base class for DataPipelineOrchestrator mixins.

    Provides common utilities and type hints for all pipeline mixins.
    Subclasses can inherit from this to get:
    - Standard logging helpers
    - Stage result extraction
    - Common error handling

    This class does NOT require the main class to inherit from it.
    It's designed to be used alongside DataPipelineOrchestrator via
    multiple inheritance.
    """

    # Type hints for IDE support - these are provided by the main class
    if TYPE_CHECKING:
        _current_stage: Any
        _current_iteration: int
        _current_board_type: str | None
        _current_num_players: int | None
        auto_trigger: bool
        auto_trigger_sync: bool
        auto_trigger_export: bool
        auto_trigger_training: bool
        auto_trigger_evaluation: bool
        auto_trigger_promotion: bool
        _iteration_records: dict
        _circuit_breaker: Any

    # =========================================================================
    # Config Key Helper
    # =========================================================================

    def _get_config_key(self) -> str:
        """Get config key from current board type and player count.

        Returns:
            Config key like "hex8_2p" or "unknown" if not set
        """
        board = getattr(self, "_current_board_type", None)
        players = getattr(self, "_current_num_players", None)
        if board and players:
            return f"{board}_{players}p"
        return "unknown"

    # =========================================================================
    # Stage Result Extraction
    # =========================================================================

    def _extract_stage_result(self, event_or_result: Any) -> Any:
        """Extract StageCompletionResult from RouterEvent or return as-is.

        When subscribers are called via the unified router, they may receive
        either a StageCompletionResult directly or a RouterEvent wrapper.
        This helper extracts the underlying result for consistent handling.

        Args:
            event_or_result: Either a StageCompletionResult or RouterEvent

        Returns:
            The underlying stage result object
        """
        from types import SimpleNamespace

        # If it's a RouterEvent, extract the stage_result
        if hasattr(event_or_result, "stage_result") and event_or_result.stage_result is not None:
            return event_or_result.stage_result

        # If it's a RouterEvent without stage_result, try payload
        if hasattr(event_or_result, "payload") and hasattr(event_or_result, "event_type"):
            # It's a RouterEvent - create a minimal result-like object from payload
            payload = event_or_result.payload or {}
            result = SimpleNamespace()
            result.success = payload.get("success", True)
            result.stage = payload.get("stage", "unknown")
            result.iteration = payload.get("iteration", 0)
            result.timestamp = payload.get("timestamp", time.time())
            result.data = payload.get("data", {})
            result.error = payload.get("error")
            return result

        # Return as-is if it's already a stage result
        return event_or_result

    # =========================================================================
    # Logging Helpers
    # =========================================================================

    def _log_stage_event(
        self,
        event_name: str,
        result: Any,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log a stage event with standard formatting.

        Args:
            event_name: Name of the event (e.g., "SYNC_COMPLETE")
            result: The stage result object
            extra: Additional key-value pairs to include
        """
        config_key = self._get_config_key()
        iteration = getattr(result, "iteration", getattr(self, "_current_iteration", 0))
        success = getattr(result, "success", True)

        log_data = {
            "event": event_name,
            "config_key": config_key,
            "iteration": iteration,
            "success": success,
        }
        if extra:
            log_data.update(extra)

        if success:
            logger.info(f"[{config_key}] {event_name}: iteration={iteration} {extra or ''}")
        else:
            error = getattr(result, "error", "Unknown error")
            logger.warning(f"[{config_key}] {event_name} FAILED: {error}")

    def _log_trigger(self, stage: str, reason: str = "") -> None:
        """Log a stage trigger event.

        Args:
            stage: Stage being triggered (e.g., "EXPORT", "TRAINING")
            reason: Optional reason for the trigger
        """
        config_key = self._get_config_key()
        iteration = getattr(self, "_current_iteration", 0)
        msg = f"[{config_key}] Triggering {stage} (iteration={iteration})"
        if reason:
            msg += f": {reason}"
        logger.info(msg)

    # =========================================================================
    # Circuit Breaker Helpers
    # =========================================================================

    def _is_circuit_open(self) -> bool:
        """Check if the pipeline circuit breaker is open.

        Returns:
            True if circuit breaker is open (operations should be blocked)
        """
        cb = getattr(self, "_circuit_breaker", None)
        if cb is None:
            return False
        # Check for is_open method first (newer API)
        if hasattr(cb, "is_open"):
            return cb.is_open()
        # Fallback to state check
        if hasattr(cb, "state"):
            return cb.state.name == "OPEN"
        return False

    def _record_circuit_failure(self, error: Exception | str) -> None:
        """Record a failure in the circuit breaker.

        Args:
            error: The error that occurred
        """
        cb = getattr(self, "_circuit_breaker", None)
        if cb is not None and hasattr(cb, "record_failure"):
            error_msg = str(error) if isinstance(error, Exception) else error
            cb.record_failure(error_msg)

    def _record_circuit_success(self) -> None:
        """Record a success in the circuit breaker."""
        cb = getattr(self, "_circuit_breaker", None)
        if cb is not None and hasattr(cb, "record_success"):
            cb.record_success()

    # =========================================================================
    # Auto-Trigger Helpers
    # =========================================================================

    def _should_auto_trigger(self, stage: str) -> bool:
        """Check if a stage should be auto-triggered.

        Args:
            stage: Stage name (sync, export, training, evaluation, promotion)

        Returns:
            True if auto-triggering is enabled for this stage
        """
        if not getattr(self, "auto_trigger", False):
            return False

        stage_flags = {
            "sync": "auto_trigger_sync",
            "export": "auto_trigger_export",
            "training": "auto_trigger_training",
            "evaluation": "auto_trigger_evaluation",
            "promotion": "auto_trigger_promotion",
        }

        flag_name = stage_flags.get(stage.lower())
        if flag_name is None:
            return False

        return getattr(self, flag_name, False)


# =============================================================================
# Type Aliases
# =============================================================================

# For backward compatibility
PipelineProtocol = DataPipelineOrchestratorProtocol


__all__ = [
    "DataPipelineOrchestratorProtocol",
    "PipelineMixinBase",
    "PipelineProtocol",
]

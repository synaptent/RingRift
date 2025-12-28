"""DataPipelineOrchestrator - Unified pipeline stage coordination (December 2025).

This module provides centralized monitoring and coordination of the self-improvement
pipeline stages. It tracks stage transitions, coordinates downstream triggering,
and provides pipeline-wide observability.

Pipeline Stages:
    SELFPLAY -> SYNC -> NPZ_EXPORT -> TRAINING -> EVALUATION -> PROMOTION

Module Structure
----------------
Classes:
    PipelineCircuitBreaker  - Fault tolerance for stage failures (lines 82-227)
    PipelineStage           - Enum of pipeline stages (lines 228-241)
    StageTransition         - Record of stage transition (lines 242-254)
    IterationRecord         - Record of complete pipeline iteration (lines 255-277)
    PipelineStats           - Statistics tracking dataclass (lines 278-290)
    DataPipelineOrchestrator - Main orchestrator class (lines 291-3418)

Module Functions (lines 3420-3505):
    get_pipeline_orchestrator()   - Singleton accessor
    wire_pipeline_events()        - Wire all event subscriptions
    get_pipeline_status()         - Get current status dict
    get_current_pipeline_stage()  - Get current PipelineStage
    get_pipeline_health()         - Get health metrics
    is_pipeline_healthy()         - Check overall health

DataPipelineOrchestrator Key Methods
------------------------------------
Lifecycle:
    start()                      - Subscribe to events, start monitoring
    stop()                       - Unsubscribe, cleanup resources
    health_check()               - Return HealthCheckResult

Stage Management:
    get_status()                 - Current pipeline status dict
    get_stage_metrics()          - Timing/count per stage
    get_current_stage()          - Current active stage
    get_pending_stages()         - Stages waiting for triggers

Stage Triggering:
    trigger_export()             - Trigger NPZ export for config
    trigger_training()           - Trigger training for config
    trigger_evaluation()         - Trigger gauntlet evaluation
    trigger_promotion()          - Trigger model promotion

Event Handlers (subscribed automatically):
    _on_selfplay_complete()      - Handle SELFPLAY_COMPLETE
    _on_sync_completed()         - Handle DATA_SYNC_COMPLETED
    _on_npz_export_complete()    - Handle NPZ_EXPORT_COMPLETE
    _on_training_completed()     - Handle TRAINING_COMPLETED
    _on_evaluation_completed()   - Handle EVALUATION_COMPLETED
    _on_promotion_completed()    - Handle PROMOTION_COMPLETE
    _on_new_games_available()    - Handle NEW_GAMES_AVAILABLE
    _on_orphan_games_detected()  - Handle ORPHAN_GAMES_DETECTED
    _on_regression_detected()    - Handle REGRESSION_DETECTED
    _on_promotion_failed()       - Handle PROMOTION_FAILED

Event Integration
-----------------
Subscribes to (30+ events):
    Core Pipeline:
    - SELFPLAY_COMPLETE, DATA_SYNC_COMPLETED, NPZ_EXPORT_COMPLETE
    - TRAINING_COMPLETED, EVALUATION_COMPLETED, PROMOTION_COMPLETE

    Data Events:
    - NEW_GAMES_AVAILABLE, ORPHAN_GAMES_DETECTED, ORPHAN_GAMES_REGISTERED

    Quality/Feedback:
    - QUALITY_SCORE_UPDATED, REGRESSION_DETECTED, PROMOTION_FAILED
    - CURRICULUM_REBALANCED, CURRICULUM_ADVANCED

    Infrastructure:
    - REPAIR_COMPLETED, REPAIR_FAILED

Emits:
    - Pipeline stage transitions (via internal state)
    - Triggers downstream actions via stage handlers

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

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from app.coordination.protocols import (
    CoordinatorStatus,
    HealthCheckResult,
    register_coordinator,
    unregister_coordinator,
)

# December 2025: Import mixin classes for DataPipelineOrchestrator decomposition
from app.coordination.pipeline_event_handler_mixin import PipelineEventHandlerMixin
from app.coordination.pipeline_metrics_mixin import PipelineMetricsMixin
from app.coordination.pipeline_stage_mixin import PipelineStageMixin
from app.coordination.pipeline_trigger_mixin import PipelineTriggerMixin

logger = logging.getLogger(__name__)


# =============================================================================
# Circuit Breaker for Pipeline Fault Tolerance (December 2025)
# =============================================================================

# December 2025: Import canonical CircuitBreaker and CircuitState
try:
    from app.distributed.circuit_breaker import (
        CircuitBreaker as CanonicalCircuitBreaker,
        CircuitState as CircuitBreakerState,
    )
    _HAS_CANONICAL_CIRCUIT_BREAKER = True
except ImportError:
    # Fallback if canonical module unavailable
    _HAS_CANONICAL_CIRCUIT_BREAKER = False
    CanonicalCircuitBreaker = None

    class CircuitBreakerState(Enum):
        """Circuit breaker states (fallback)."""
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"


class PipelineCircuitBreaker:
    """Circuit breaker wrapper for pipeline stage fault tolerance.

    December 2025: Migrated to use canonical CircuitBreaker from
    app.distributed.circuit_breaker with "pipeline" as the target.
    Adds per-stage failure tracking for observability.

    This is a thin wrapper that:
    1. Uses canonical CircuitBreaker with target="pipeline"
    2. Tracks failures per stage for logging/metrics
    3. Provides backward-compatible API
    """

    # Default target name for pipeline-wide circuit
    PIPELINE_TARGET = "pipeline"

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        """Initialize pipeline circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before trying half-open recovery
            half_open_max_calls: Max test calls in half-open state
        """
        self._failures_by_stage: dict[str, int] = {}
        self._success_count: int = 0

        if _HAS_CANONICAL_CIRCUIT_BREAKER and CanonicalCircuitBreaker:
            self._breaker = CanonicalCircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                half_open_max_calls=half_open_max_calls,
                success_threshold=1,
                operation_type="pipeline",
            )
        else:
            # Minimal fallback for missing canonical module
            self._breaker = None
            self._fallback_state = CircuitBreakerState.CLOSED
            self._fallback_failure_count = 0
            self._fallback_last_failure_time = 0.0
            self._failure_threshold = failure_threshold
            self._recovery_timeout = recovery_timeout

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        if self._breaker:
            state = self._breaker.get_state(self.PIPELINE_TARGET)
            return state == CircuitBreakerState.OPEN
        return self._fallback_state == CircuitBreakerState.OPEN

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        if self._breaker:
            state = self._breaker.get_state(self.PIPELINE_TARGET)
            return state == CircuitBreakerState.CLOSED
        return self._fallback_state == CircuitBreakerState.CLOSED

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit state."""
        if self._breaker:
            return self._breaker.get_state(self.PIPELINE_TARGET)
        return self._fallback_state

    def can_execute(self) -> bool:
        """Check if a request can be executed."""
        if self._breaker:
            return self._breaker.can_execute(self.PIPELINE_TARGET)
        return self._fallback_state != CircuitBreakerState.OPEN

    def record_success(self, stage: str = "") -> None:
        """Record a successful execution."""
        self._success_count += 1
        if self._breaker:
            self._breaker.record_success(self.PIPELINE_TARGET)
            if stage:
                logger.debug(f"[PipelineCircuitBreaker] Success in stage: {stage}")
        else:
            self._fallback_state = CircuitBreakerState.CLOSED
            self._fallback_failure_count = 0

    def record_failure(self, stage: str, error: str = "") -> None:
        """Record a failed execution."""
        self._failures_by_stage[stage] = self._failures_by_stage.get(stage, 0) + 1
        if self._breaker:
            # Create an exception to pass to canonical breaker
            err_obj = Exception(f"{stage}: {error}") if error else None
            self._breaker.record_failure(self.PIPELINE_TARGET, err_obj)
            logger.warning(f"[PipelineCircuitBreaker] Failure in {stage}: {error}")
        else:
            self._fallback_failure_count += 1
            self._fallback_last_failure_time = time.time()
            if self._fallback_failure_count >= self._failure_threshold:
                self._fallback_state = CircuitBreakerState.OPEN
                logger.warning(
                    f"[PipelineCircuitBreaker] Threshold reached, opening circuit: {stage} - {error}"
                )

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._failures_by_stage.clear()
        if self._breaker:
            self._breaker.reset(self.PIPELINE_TARGET)
        else:
            self._fallback_state = CircuitBreakerState.CLOSED
            self._fallback_failure_count = 0
        logger.info("[PipelineCircuitBreaker] Manually reset")



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


class DataPipelineOrchestrator(
    PipelineEventHandlerMixin,
    PipelineTriggerMixin,
    PipelineStageMixin,
    PipelineMetricsMixin,
):
    """Orchestrates the self-improvement pipeline stages.

    Tracks stage transitions, provides coordination between stages,
    and maintains pipeline-wide metrics and observability.

    December 2025: Refactored to use mixin pattern for better maintainability.
    Inherits from:
    - PipelineEventHandlerMixin: All _on_data_* event handlers (~1,200 lines)
    - PipelineTriggerMixin: Stage triggering methods (~600 lines)
    - PipelineStageMixin: Stage callback handlers (~500 lines)
    - PipelineMetricsMixin: Metrics, status, and health reporting (~400 lines)
    """

    def __init__(
        self,
        max_history: int = 100,
        auto_trigger: bool = True,  # Automatically trigger downstream stages
        config: "PipelineConfig | None" = None,  # Use coordinator_config if None
    ):
        """Initialize DataPipelineOrchestrator.

        Args:
            max_history: Maximum iteration records to retain
            auto_trigger: If True, automatically trigger downstream stages
            config: Pipeline configuration (uses global config if None)
        """
        # Load config from coordinator_config if not provided
        if config is None:
            try:
                from app.coordination.coordinator_config import get_config
                config = get_config().pipeline
            except ImportError:
                config = None

        self.max_history = max_history
        self.auto_trigger = auto_trigger
        self._config = config

        # Per-stage auto-trigger controls (December 2025)
        self.auto_trigger_sync = getattr(config, "auto_trigger_sync", True) if config else True
        self.auto_trigger_export = getattr(config, "auto_trigger_export", True) if config else True
        self.auto_trigger_training = getattr(config, "auto_trigger_training", True) if config else True
        self.auto_trigger_evaluation = getattr(config, "auto_trigger_evaluation", True) if config else True
        self.auto_trigger_promotion = getattr(config, "auto_trigger_promotion", True) if config else True

        # Quality gate configuration (December 2025 - Phase 14)
        self.quality_gate_enabled = getattr(config, "quality_gate_enabled", True) if config else True
        self.quality_gate_threshold = getattr(config, "quality_gate_threshold", 0.6) if config else 0.6
        self.quality_gate_min_high_quality_pct = getattr(config, "quality_gate_min_high_quality_pct", 0.30) if config else 0.30
        self._quality_check_history: list[float] = []  # Track quality trend
        self._last_quality_score: float = 0.0

        # Circuit breaker for fault tolerance (December 2025)
        # Migrated to PipelineCircuitBreaker wrapping canonical CircuitBreaker
        cb_enabled = getattr(config, "circuit_breaker_enabled", True) if config else True
        cb_threshold = getattr(config, "circuit_breaker_failure_threshold", 3) if config else 3
        # Support both old and new config names for timeout
        cb_timeout = (
            getattr(config, "circuit_breaker_recovery_timeout", None) or
            getattr(config, "circuit_breaker_reset_timeout_seconds", 300.0)
        ) if config else 300.0
        cb_half_open = (
            getattr(config, "circuit_breaker_half_open_max_calls", None) or
            getattr(config, "circuit_breaker_half_open_max_requests", 1)
        ) if config else 1

        self._circuit_breaker = PipelineCircuitBreaker(
            failure_threshold=cb_threshold,
            recovery_timeout=cb_timeout,
            half_open_max_calls=cb_half_open,
        ) if cb_enabled else None

        # Current pipeline state
        self._current_stage = PipelineStage.IDLE
        self._current_iteration = 0

        # Board configuration tracking for auto-trigger
        self._current_board_type: str | None = None
        self._current_num_players: int | None = None

        # Iteration tracking
        self._iteration_records: dict[int, IterationRecord] = {}
        self._completed_iterations: list[IterationRecord] = []

        # Stage timing
        self._stage_start_times: dict[PipelineStage, float] = {}
        self._stage_durations: dict[PipelineStage, list[float]] = {}

        # Transition history
        self._transitions: list[StageTransition] = []

        # Statistics
        self._total_games = 0
        self._total_models = 0
        self._total_promotions = 0

        # Subscription state
        self._subscribed = False
        self._prefer_stage_events = False
        self._data_event_iteration = 0

        # Callbacks for stage transitions
        self._stage_callbacks: dict[PipelineStage, list[Callable]] = {}

        # Stage metadata for tracking promotion candidates, etc. (Phase 7 fix)
        self._stage_metadata: dict[str, Any] = {
            "candidates": 0,
            "last_iteration": 0,
        }

        # Quality distribution tracking (December 2025)
        self._quality_distribution: dict[str, float] = {}  # level -> percentage
        self._last_quality_update: float = 0.0
        self._cache_invalidation_count: int = 0
        self._pending_cache_refresh: bool = False

        # Optimization tracking (December 2025)
        self._active_optimization: str | None = None  # "cmaes" or "nas"
        self._optimization_run_id: str | None = None
        self._optimization_start_time: float = 0.0

        # Resource constraint tracking (December 2025)
        self._paused: bool = False
        self._pause_reason: str | None = None
        self._pause_time: float = 0.0
        self._resource_constraints: dict[str, dict] = {}  # resource_type -> constraint_info
        self._backpressure_active: bool = False

        # ClusterMonitor caching (December 2025 - performance fix)
        self._cluster_monitor: Any = None
        self._cluster_monitor_last_check: float = 0.0
        self._cluster_monitor_ttl: float = 30.0  # Cache for 30 seconds
        self._last_constraint_emit: dict[str, float] = {}  # Dedup event emission

        # CoordinatorProtocol state (December 2025 - Phase 14)
        self._coordinator_status = CoordinatorStatus.INITIALIZING
        self._start_time: float = 0.0
        self._events_processed: int = 0
        self._errors_count: int = 0
        self._last_error: str = ""

    def _get_board_config(
        self, result: Any = None, metadata: dict | None = None
    ) -> tuple[str | None, int | None]:
        """Get board configuration from various sources.

        Tries to infer board_type and num_players from:
        1. Current tracked state
        2. Result object attributes
        3. Metadata dict (config_key, board_type, num_players)
        4. Parsing config_key format (e.g., "hex8_2p")

        Returns:
            Tuple of (board_type, num_players), either may be None
        """
        board_type = self._current_board_type
        num_players = self._current_num_players

        # Already have config from tracked state
        if board_type and num_players:
            return board_type, num_players

        # Try to get from result object
        if result is not None:
            if hasattr(result, "board_type") and result.board_type:
                board_type = result.board_type
            if hasattr(result, "num_players") and result.num_players:
                num_players = result.num_players

        # Try to get from metadata dict
        if metadata:
            if not board_type and "board_type" in metadata:
                board_type = metadata["board_type"]
            if not num_players and "num_players" in metadata:
                num_players = metadata["num_players"]

            # Try to parse from config_key (e.g., "hex8_2p" or "square8_4p")
            if (not board_type or not num_players) and "config_key" in metadata:
                config_key = metadata["config_key"]
                if "_" in config_key and config_key.endswith("p"):
                    parts = config_key.rsplit("_", 1)
                    if len(parts) == 2:
                        if not board_type:
                            board_type = parts[0]
                        if not num_players:
                            try:
                                num_players = int(parts[1].rstrip("p"))
                            except ValueError:
                                pass

        # Log if we successfully inferred missing config
        if (board_type and num_players) and (
            not self._current_board_type or not self._current_num_players
        ):
            logger.debug(
                f"[DataPipelineOrchestrator] Inferred board config: {board_type}_{num_players}p"
            )

        return board_type, num_players

    # =========================================================================
    # CoordinatorProtocol Implementation (December 2025 - Phase 14)
    # =========================================================================

    @property
    def name(self) -> str:
        """Unique name identifying this coordinator."""
        return "DataPipelineOrchestrator"

    @property
    def status(self) -> CoordinatorStatus:
        """Current status of the coordinator."""
        return self._coordinator_status

    @property
    def uptime_seconds(self) -> float:
        """Time since orchestrator started, in seconds."""
        if self._start_time <= 0:
            return 0.0
        return time.time() - self._start_time

    async def start(self) -> None:
        """Start the orchestrator.

        Subscribes to events and begins tracking pipeline state.
        Idempotent - calling on an already running orchestrator is a no-op.
        """
        if self._coordinator_status == CoordinatorStatus.RUNNING:
            return  # Already running

        self._coordinator_status = CoordinatorStatus.RUNNING
        self._start_time = time.time()

        # Subscribe to events
        self.subscribe_to_events()
        self.subscribe_to_data_events()

        # Register with coordinator registry
        register_coordinator(self)

        logger.info(f"[{self.name}] Started")

    async def stop(self) -> None:
        """Stop the orchestrator gracefully.

        Cleans up subscriptions and resources.
        Idempotent - calling on an already stopped orchestrator is a no-op.
        """
        if self._coordinator_status == CoordinatorStatus.STOPPED:
            return  # Already stopped

        self._coordinator_status = CoordinatorStatus.STOPPING

        # Unregister from coordinator registry
        unregister_coordinator(self.name)

        self._coordinator_status = CoordinatorStatus.STOPPED
        logger.info(f"[{self.name}] Stopped")

    async def run_forever(self) -> None:
        """Run the orchestrator forever (for DaemonManager compatibility).

        Starts the orchestrator and loops until stopped.
        """
        await self.start()
        try:
            while self._coordinator_status == CoordinatorStatus.RUNNING:
                await asyncio.sleep(10)
        finally:
            await self.stop()


    def health_check(self) -> HealthCheckResult:
        """Check orchestrator health.

        Returns:
            Health check result with status and pipeline details.
        """
        # Check for error state
        if self._coordinator_status == CoordinatorStatus.ERROR:
            return HealthCheckResult.unhealthy(
                f"Orchestrator in error state: {self._last_error}"
            )

        # Check if stopped
        if self._coordinator_status == CoordinatorStatus.STOPPED:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="Orchestrator is stopped",
            )

        # Check if paused due to resource constraints
        if self._paused:
            return HealthCheckResult.degraded(
                f"Pipeline paused: {self._pause_reason or 'unknown'}",
                pause_duration=time.time() - self._pause_time,
                constraints=dict(self._resource_constraints),
            )

        # Check circuit breaker
        if self._circuit_breaker and self._circuit_breaker.is_open:
            return HealthCheckResult.degraded(
                "Circuit breaker open - pipeline stages may be blocked",
                circuit_breaker=self._circuit_breaker.get_status(),
            )

        # Check if events are subscribed
        if not self._subscribed:
            return HealthCheckResult.degraded(
                "Not subscribed to events - pipeline may not progress automatically"
            )

        # Healthy
        return HealthCheckResult(
            healthy=True,
            status=self._coordinator_status,
            details={
                "uptime_seconds": self.uptime_seconds,
                "events_processed": self._events_processed,
                "current_stage": self._current_stage.value,
                "current_iteration": self._current_iteration,
            },
        )

    def _record_event_processed(self) -> None:
        """Record that an event was processed (for protocol metrics)."""
        self._events_processed += 1

    def _record_error(self, error: str) -> None:
        """Record an error (for protocol metrics)."""
        self._errors_count += 1
        self._last_error = error
        if self._errors_count >= 10:
            self._coordinator_status = CoordinatorStatus.DEGRADED

    def _should_process_stage_data_event(self, event_type: str) -> bool:
        """Gate DataEventType stage handling when StageEvent is active."""
        if self._prefer_stage_events:
            logger.debug(
                f"[DataPipelineOrchestrator] Skipping {event_type} data event "
                "because stage events are active"
            )
            return False
        return True

    def _next_data_event_iteration(self) -> int:
        """Return a best-effort iteration ID for data events."""
        if self._current_iteration <= 0:
            self._data_event_iteration = max(self._data_event_iteration, 1)
            return self._data_event_iteration

        if self._current_stage in (PipelineStage.IDLE, PipelineStage.COMPLETE):
            self._data_event_iteration = max(self._data_event_iteration, self._current_iteration + 1)
            return self._data_event_iteration

        return self._current_iteration

    def _current_iteration_for_data_event(self) -> int:
        """Return current iteration or a fallback for data events."""
        if self._current_iteration > 0:
            return self._current_iteration
        if self._data_event_iteration == 0:
            self._data_event_iteration = 1
        return self._data_event_iteration

    def subscribe_to_events(self) -> bool:
        """Subscribe to pipeline stage events.

        Returns:
            True if ALL subscriptions succeeded, False if any failed

        Note:
            December 28, 2025: Fixed to properly track subscription failures.
            Previously _subscribed was set True in finally block even on failure,
            which hid partial subscription failures.
        """
        if self._subscribed:
            return True

        try:
            # P0.5 (December 2025): Use get_router() instead of deprecated get_stage_event_bus()
            from app.coordination.event_router import StageEvent, get_router

            router = get_router()

            # Track subscriptions for proper error reporting
            subscriptions = [
                (StageEvent.SELFPLAY_COMPLETE, self._on_selfplay_complete, "SELFPLAY_COMPLETE"),
                (StageEvent.CANONICAL_SELFPLAY_COMPLETE, self._on_selfplay_complete, "CANONICAL_SELFPLAY_COMPLETE"),
                (StageEvent.GPU_SELFPLAY_COMPLETE, self._on_selfplay_complete, "GPU_SELFPLAY_COMPLETE"),
                (StageEvent.SYNC_COMPLETE, self._on_sync_complete, "SYNC_COMPLETE"),
                (StageEvent.NPZ_EXPORT_COMPLETE, self._on_npz_export_complete, "NPZ_EXPORT_COMPLETE"),
                (StageEvent.TRAINING_STARTED, self._on_training_started, "TRAINING_STARTED"),
                (StageEvent.TRAINING_COMPLETE, self._on_training_complete, "TRAINING_COMPLETE"),
                (StageEvent.TRAINING_FAILED, self._on_training_failed, "TRAINING_FAILED"),
                (StageEvent.EVALUATION_COMPLETE, self._on_evaluation_complete, "EVALUATION_COMPLETE"),
                (StageEvent.PROMOTION_COMPLETE, self._on_promotion_complete, "PROMOTION_COMPLETE"),
                (StageEvent.ITERATION_COMPLETE, self._on_iteration_complete, "ITERATION_COMPLETE"),
            ]

            successful_subscriptions: list[str] = []
            failed_subscriptions: list[tuple[str, str]] = []  # (event_name, error_message)

            for event, handler, event_name in subscriptions:
                try:
                    router.subscribe(event, handler)
                    successful_subscriptions.append(event_name)
                except Exception as sub_error:
                    failed_subscriptions.append((event_name, str(sub_error)))
                    logger.error(
                        f"[DataPipelineOrchestrator] Failed to subscribe to {event_name}: {sub_error}"
                    )

            # Report subscription results
            if failed_subscriptions:
                failed_names = [name for name, _ in failed_subscriptions]
                logger.error(
                    f"[DataPipelineOrchestrator] Failed to subscribe to {len(failed_subscriptions)} "
                    f"events: {failed_names}. Successfully subscribed to: {successful_subscriptions}"
                )
                # Don't set _subscribed = True since we have failures
                return False

            # All subscriptions succeeded
            self._subscribed = True
            self._prefer_stage_events = True
            logger.info(
                f"[DataPipelineOrchestrator] Subscribed to {len(successful_subscriptions)} stage events"
            )
            return True

        except ImportError as e:
            logger.error(f"[DataPipelineOrchestrator] Cannot subscribe to events - stage_events not available: {e}")
            # This is a critical failure - orchestrator cannot function without events
            raise RuntimeError(f"EventRouter not available - pipeline cannot function: {e}")
        except Exception as e:
            logger.error(
                f"[DataPipelineOrchestrator] Failed to subscribe to stage events: {e}",
                exc_info=True,
            )
            return False

    def subscribe_to_data_events(self) -> bool:
        """Subscribe to DataEventBus events (December 2025).

        Returns:
            True if successfully subscribed
        """
        try:
            from app.coordination.event_router import get_router
            from app.coordination.event_router import DataEventType

            router = get_router()

            # Subscribe to quality and cache events
            router.subscribe(
                DataEventType.QUALITY_DISTRIBUTION_CHANGED.value,
                self._on_quality_distribution_changed,
            )
            router.subscribe(
                DataEventType.CACHE_INVALIDATED.value,
                self._on_cache_invalidated,
            )

            # Subscribe to optimization events (December 2025)
            router.subscribe(
                DataEventType.CMAES_TRIGGERED.value,
                self._on_optimization_triggered,
            )
            router.subscribe(
                DataEventType.NAS_TRIGGERED.value,
                self._on_optimization_triggered,
            )

            # Subscribe to resource constraint events (December 2025)
            router.subscribe(
                DataEventType.RESOURCE_CONSTRAINT_DETECTED.value,
                self._on_resource_constraint_detected,
            )
            router.subscribe(
                DataEventType.BACKPRESSURE_ACTIVATED.value,
                self._on_backpressure_activated,
            )
            router.subscribe(
                DataEventType.BACKPRESSURE_RELEASED.value,
                self._on_backpressure_released,
            )

            # Subscribe to promotion candidate events (December 2025)
            # This wires the previously orphaned PROMOTION_CANDIDATE event
            router.subscribe(
                DataEventType.PROMOTION_CANDIDATE.value,
                self._on_promotion_candidate,
            )

            # Subscribe to database lifecycle events (December 2025 - Phase 4A.3)
            router.subscribe(
                DataEventType.DATABASE_CREATED.value,
                self._on_database_created,
            )

            # Subscribe to training threshold events
            router.subscribe(
                DataEventType.TRAINING_THRESHOLD_REACHED.value,
                self._on_training_threshold_reached,
            )

            # Subscribe to promotion lifecycle events
            router.subscribe(
                DataEventType.PROMOTION_STARTED.value,
                self._on_promotion_started,
            )

            # Subscribe to work queue events
            router.subscribe(
                DataEventType.WORK_QUEUED.value,
                self._on_work_queued,
            )

            # Core pipeline events (DataEventType fallback for StageEvent wiring)
            router.subscribe(
                DataEventType.SELFPLAY_COMPLETE.value,
                self._on_data_selfplay_complete,
            )
            router.subscribe(
                DataEventType.DATA_SYNC_COMPLETED.value,
                self._on_data_sync_completed,
            )
            router.subscribe(
                DataEventType.DATA_SYNC_FAILED.value,
                self._on_data_sync_failed,
            )
            router.subscribe(
                DataEventType.TRAINING_COMPLETED.value,
                self._on_data_training_completed,
            )
            router.subscribe(
                DataEventType.TRAINING_FAILED.value,
                self._on_data_training_failed,
            )
            router.subscribe(
                DataEventType.EVALUATION_COMPLETED.value,
                self._on_data_evaluation_completed,
            )
            router.subscribe(
                DataEventType.EVALUATION_FAILED.value,
                self._on_data_evaluation_failed,
            )
            router.subscribe(
                DataEventType.MODEL_PROMOTED.value,
                self._on_data_model_promoted,
            )

            # Orphan games detection - triggers sync and registration (Dec 2025)
            router.subscribe(
                DataEventType.ORPHAN_GAMES_DETECTED.value,
                self._on_orphan_games_detected,
            )
            router.subscribe(
                DataEventType.ORPHAN_GAMES_REGISTERED.value,
                self._on_orphan_games_registered,
            )

            # December 2025: Subscribe to GAME_SYNCED for export triggering
            # When games are synced to training nodes, this can trigger NPZ export
            router.subscribe(
                DataEventType.GAME_SYNCED.value,
                self._on_game_synced,
            )

            # December 2025: Subscribe to exploration and sync feedback events
            # These events trigger curriculum adjustments and adaptive exploration
            router.subscribe(
                DataEventType.EXPLORATION_BOOST.value,
                self._on_exploration_boost,
            )
            router.subscribe(
                DataEventType.SYNC_TRIGGERED.value,
                self._on_sync_triggered,
            )

            # December 2025: Subscribe to DATA_STALE for training freshness
            # When training data becomes stale, we trigger urgent sync
            router.subscribe(
                DataEventType.DATA_STALE.value,
                self._on_data_stale,
            )

            # December 2025 Phase 11: Additional event subscriptions for improved
            # pipeline coordination and regression handling
            router.subscribe(
                DataEventType.NEW_GAMES_AVAILABLE.value,
                self._on_new_games_available,
            )
            router.subscribe(
                DataEventType.REGRESSION_DETECTED.value,
                self._on_regression_detected,
            )
            router.subscribe(
                DataEventType.PROMOTION_FAILED.value,
                self._on_promotion_failed,
            )

            # December 2025: Subscribe to consolidation events for training pipeline fix
            # Consolidation merges scattered selfplay games into canonical databases
            router.subscribe(
                DataEventType.CONSOLIDATION_STARTED.value,
                self._on_consolidation_started,
            )
            router.subscribe(
                DataEventType.CONSOLIDATION_COMPLETE.value,
                self._on_consolidation_complete,
            )

            # December 2025: Wire previously orphaned critical events
            # These events were being emitted but had no subscribers
            router.subscribe(
                DataEventType.REPAIR_COMPLETED.value,
                self._on_repair_completed,
            )
            router.subscribe(
                DataEventType.REPAIR_FAILED.value,
                self._on_repair_failed,
            )
            router.subscribe(
                DataEventType.QUALITY_SCORE_UPDATED.value,
                self._on_quality_score_updated,
            )
            router.subscribe(
                DataEventType.CURRICULUM_REBALANCED.value,
                self._on_curriculum_rebalanced,
            )
            router.subscribe(
                DataEventType.CURRICULUM_ADVANCED.value,
                self._on_curriculum_advanced,
            )

            # December 2025: Subscribe to S3 backup events for pipeline tracking
            router.subscribe(
                DataEventType.S3_BACKUP_COMPLETED.value,
                self._on_s3_backup_completed,
            )

            # December 2025: Subscribe to sync integrity events for repair triggering
            if hasattr(DataEventType, 'SYNC_CHECKSUM_FAILED'):
                router.subscribe(
                    DataEventType.SYNC_CHECKSUM_FAILED.value,
                    self._on_sync_checksum_failed,
                )

            logger.info("[DataPipelineOrchestrator] Subscribed to data events")
            return True

        except ImportError:
            logger.warning("[DataPipelineOrchestrator] data_events not available")
            return False
        except Exception as e:
            logger.error(
                f"[DataPipelineOrchestrator] Failed to subscribe to data events "
                f"(router={router!r}): {e}",
                exc_info=True,
            )
            return False

    def _transition_to(
        self,
        new_stage: PipelineStage,
        iteration: int,
        success: bool = True,
        metadata: dict | None = None,
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
                callback_name = getattr(callback, "__name__", repr(callback))
                logger.error(
                    f"[DataPipelineOrchestrator] Callback error in {callback_name} "
                    f"for stage={new_stage.value}, iteration={iteration}: {e}",
                    exc_info=True,
                )

    def _ensure_iteration_record(self, iteration: int) -> IterationRecord:
        """Ensure an iteration record exists."""
        if iteration not in self._iteration_records:
            self._iteration_records[iteration] = IterationRecord(
                iteration=iteration,
                start_time=time.time(),
            )
        return self._iteration_records[iteration]




























    async def _check_training_data_quality(self, npz_path: str, iteration: int) -> bool:
        """Check if training data meets quality threshold.

        Evaluates NPZ training data quality and blocks training if:
        - Average quality score < threshold (default 0.6)
        - High quality game percentage < minimum (default 30%)
        - Quality declining for 3 consecutive exports

        Args:
            npz_path: Path to NPZ training data
            iteration: Pipeline iteration number

        Returns:
            True if quality is acceptable for training
        """
        try:
            from pathlib import Path
            import numpy as np
            from app.utils.numpy_utils import safe_load_npz

            # Check if file exists
            if not Path(npz_path).exists():
                logger.warning(f"[QualityGate] NPZ file not found: {npz_path}")
                return True  # Allow training if we can't check

            # Load NPZ and check for quality metadata
            with safe_load_npz(npz_path) as data:
                # Try to get quality scores from NPZ metadata
                if "quality_scores" in data:
                    quality_scores = data["quality_scores"]
                    avg_quality = float(np.mean(quality_scores))
                    high_quality_pct = float(np.mean(quality_scores >= 0.7))
                elif "metadata" in data:
                    # Check metadata for quality info
                    metadata = data["metadata"].item() if data["metadata"].ndim == 0 else dict(data["metadata"])
                    avg_quality = metadata.get("avg_quality", 0.7)
                    high_quality_pct = metadata.get("high_quality_pct", 0.5)
                else:
                    # Estimate quality from data characteristics
                    avg_quality = await self._estimate_data_quality(data, npz_path)
                    high_quality_pct = 0.5  # Default

            self._last_quality_score = avg_quality
            self._quality_check_history.append(avg_quality)

            # Keep only last 10 checks
            if len(self._quality_check_history) > 10:
                self._quality_check_history = self._quality_check_history[-10:]

            logger.info(
                f"[QualityGate] Iteration {iteration}: avg_quality={avg_quality:.3f}, "
                f"high_quality_pct={high_quality_pct:.1%}, threshold={self.quality_gate_threshold}"
            )

            # Check 1: Average quality threshold
            if avg_quality < self.quality_gate_threshold:
                logger.warning(
                    f"[QualityGate] Quality {avg_quality:.3f} below threshold "
                    f"{self.quality_gate_threshold}"
                )
                return False

            # Check 2: High quality percentage
            if high_quality_pct < self.quality_gate_min_high_quality_pct:
                logger.warning(
                    f"[QualityGate] High quality games {high_quality_pct:.1%} below minimum "
                    f"{self.quality_gate_min_high_quality_pct:.1%}"
                )
                return False

            # Check 3: Quality declining trend
            if len(self._quality_check_history) >= 3:
                recent = self._quality_check_history[-3:]
                if all(recent[i] > recent[i + 1] for i in range(len(recent) - 1)):
                    decline_amount = recent[0] - recent[-1]
                    if decline_amount > 0.1:  # More than 10% decline
                        logger.warning(
                            f"[QualityGate] Quality declining: {recent[0]:.3f} -> {recent[-1]:.3f}"
                        )
                        return False

            return True

        except Exception as e:
            logger.warning(f"[QualityGate] Error checking quality: {e}")
            return True  # Allow training if quality check fails

    async def _estimate_data_quality(self, data: "np.lib.npyio.NpzFile", npz_path: str) -> float:
        """Estimate data quality from NPZ contents when no explicit quality scores.

        Args:
            data: Loaded NPZ file
            npz_path: Path to NPZ (for logging)

        Returns:
            Estimated quality score (0-1)
        """
        try:
            import numpy as np

            quality_signals = []

            # Check sample count
            if "features" in data or "X" in data:
                features_key = "features" if "features" in data else "X"
                n_samples = len(data[features_key])
                # More samples = generally better, normalize to 0.3-1.0 range
                sample_score = min(1.0, 0.3 + 0.7 * (n_samples / 50000))
                quality_signals.append(sample_score)

            # Check policy distribution
            if "policy" in data or "policy_targets" in data:
                policy_key = "policy" if "policy" in data else "policy_targets"
                policy = data[policy_key]
                # Check entropy of policies (higher = more diverse = better)
                policy_probs = np.clip(policy, 1e-10, 1.0)
                entropy = -np.sum(policy_probs * np.log(policy_probs), axis=-1).mean()
                max_entropy = np.log(policy.shape[-1])
                entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0.5
                quality_signals.append(min(1.0, 0.3 + 0.7 * entropy_ratio))

            # Check value distribution
            if "value" in data or "value_targets" in data:
                value_key = "value" if "value" in data else "value_targets"
                values = data[value_key]
                # Check if values span a reasonable range (not all same)
                value_std = np.std(values)
                value_score = min(1.0, 0.4 + value_std * 2)  # Higher variance = more diverse
                quality_signals.append(value_score)

            if quality_signals:
                return float(np.mean(quality_signals))
            return 0.6  # Default moderate quality

        except Exception as e:
            logger.debug(f"[QualityGate] Error estimating quality: {e}")
            return 0.6

    async def _emit_training_blocked_by_quality(self, iteration: int, npz_path: str) -> None:
        """Emit event when training is blocked due to quality gate.

        This triggers data regeneration or other corrective actions.

        Args:
            iteration: Pipeline iteration
            npz_path: Path to the NPZ file that failed quality check
        """
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            if router:
                board_type, num_players = self._get_board_config()
                await router.publish(
                    event_type="TRAINING_BLOCKED_BY_QUALITY",
                    payload={
                        "iteration": iteration,
                        "npz_path": npz_path,
                        "board_type": board_type,
                        "num_players": num_players,
                        "quality_score": self._last_quality_score,
                        "threshold": self.quality_gate_threshold,
                        "quality_history": self._quality_check_history[-5:],
                        "recommendation": "trigger_data_regeneration",
                    },
                    source="DataPipelineOrchestrator",
                )
                logger.info(
                    f"[QualityGate] Emitted TRAINING_BLOCKED_BY_QUALITY for iteration {iteration}"
                )

                # Also trigger data regeneration if we have enough info
                if board_type and num_players:
                    await self._trigger_data_regeneration(board_type, num_players, iteration)

        except Exception as e:
            logger.warning(f"[QualityGate] Failed to emit quality block event: {e}")













    def _can_auto_trigger(self) -> bool:
        """Check if auto-triggering is allowed.

        Returns False if:
        - Pipeline is paused
        - Circuit breaker is open
        - Backpressure is active
        - Cluster resources are constrained (December 2025)
        """
        if self._paused:
            logger.debug("[DataPipelineOrchestrator] Auto-trigger blocked: pipeline paused")
            return False

        if self._circuit_breaker and self._circuit_breaker.is_open:
            logger.debug("[DataPipelineOrchestrator] Auto-trigger blocked: circuit breaker open")
            return False

        if self._backpressure_active:
            logger.debug("[DataPipelineOrchestrator] Auto-trigger blocked: backpressure active")
            return False

        # Check unified health manager (December 2025)
        # This provides comprehensive health scoring including node availability,
        # circuit health, error rates, and recovery status
        try:
            from app.coordination.unified_health_manager import should_pause_pipeline

            should_pause, reason = should_pause_pipeline()
            if should_pause:
                logger.debug(
                    f"[DataPipelineOrchestrator] Auto-trigger blocked by health manager: {reason}"
                )
                # Emit event for monitoring/alerting
                try:
                    from app.distributed.data_events import DataEventType, emit_event

                    emit_event(
                        DataEventType.TRAINING_BLOCKED_BY_QUALITY,  # Reuse existing event type
                        payload={
                            "reason": reason,
                            "blocked_by": "health_manager",
                            "source": "data_pipeline_orchestrator",
                        },
                    )
                except (RuntimeError, AttributeError, ValueError) as emit_err:
                    # Dec 2025: Event emission is best-effort but log for debugging
                    logger.debug(f"[DataPipelineOrchestrator] Event emission failed: {emit_err}")
                return False
        except ImportError:
            pass  # UnifiedHealthManager not available, skip this check
        except Exception as e:
            logger.debug(f"[DataPipelineOrchestrator] Health manager check failed: {e}")
            # Continue on health check failure - don't block training

        # Check cluster resources (December 2025)
        if not self._check_cluster_resources():
            logger.debug("[DataPipelineOrchestrator] Auto-trigger blocked: cluster resources constrained")
            return False

        return True

    def _check_cluster_resources(
        self,
        disk_threshold: float = 85.0,
        min_free_disk_gb: float = 50.0,
    ) -> bool:
        """Check if cluster has sufficient resources for training.

        Returns True if resources are adequate, False if constrained.

        Args:
            disk_threshold: Max disk usage percentage before blocking
            min_free_disk_gb: Minimum free disk space required

        December 2025: Added to integrate cluster status with training decisions.
        Uses cached ClusterMonitor with TTL to avoid expensive SSH reconnections.
        """
        try:
            from app.coordination.cluster_status_monitor import ClusterMonitor

            # Use cached ClusterMonitor with TTL (December 2025 - performance fix)
            now = time.time()
            if (self._cluster_monitor is None or
                now - self._cluster_monitor_last_check > self._cluster_monitor_ttl):
                self._cluster_monitor = ClusterMonitor()
                self._cluster_monitor_last_check = now

            status = self._cluster_monitor.get_cluster_status(
                include_game_counts=False,
                include_training_status=True,
                include_disk_usage=True,
            )

            # Check disk usage
            if status.avg_disk_usage > disk_threshold:
                logger.warning(
                    f"[DataPipelineOrchestrator] Cluster disk usage high: "
                    f"{status.avg_disk_usage:.1f}% (threshold: {disk_threshold}%)"
                )
                self._emit_resource_constraint("disk_usage_high", status.avg_disk_usage)
                return False

            # Check free disk space
            if status.total_disk_free_gb < min_free_disk_gb:
                logger.warning(
                    f"[DataPipelineOrchestrator] Cluster disk space low: "
                    f"{status.total_disk_free_gb:.1f}GB free (min: {min_free_disk_gb}GB)"
                )
                self._emit_resource_constraint("disk_space_low", status.total_disk_free_gb)
                return False

            # Check if too many nodes are already training
            training_ratio = status.nodes_training / max(status.active_nodes, 1)
            if training_ratio > 0.8:
                logger.info(
                    f"[DataPipelineOrchestrator] Most nodes busy training: "
                    f"{status.nodes_training}/{status.active_nodes} ({training_ratio:.0%})"
                )
                # This is informational, not blocking - training can queue
                pass

            return True

        except ImportError:
            # ClusterMonitor not available - allow auto-trigger
            return True
        except Exception as e:
            logger.debug(f"[DataPipelineOrchestrator] Resource check failed: {e}")
            # On error, allow auto-trigger (fail open)
            return True

    def _emit_resource_constraint(self, constraint_type: str, value: float) -> None:
        """Emit RESOURCE_CONSTRAINT_DETECTED event with deduplication.

        December 2025: Added cooldown to prevent event spam during sustained constraints.
        """
        # Deduplicate: Don't emit same constraint type within 60 seconds
        now = time.time()
        last_emit = self._last_constraint_emit.get(constraint_type, 0)
        if now - last_emit < 60.0:
            return
        self._last_constraint_emit[constraint_type] = now

        try:
            from app.coordination.event_router import DataEvent, DataEventType, get_event_bus

            event = DataEvent(
                event_type=DataEventType.RESOURCE_CONSTRAINT_DETECTED,
                payload={
                    "constraint_type": constraint_type,
                    "value": value,
                    "timestamp": now,
                    "source": "data_pipeline_orchestrator",
                },
                source="data_pipeline_orchestrator",
            )

            import asyncio
            bus = get_event_bus()
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(bus.publish(event))
            except RuntimeError:
                asyncio.run(bus.publish(event))

        except Exception as e:
            logger.debug(f"[DataPipelineOrchestrator] Best-effort event emit failed: {e}")

    def _record_circuit_success(self, stage: str) -> None:
        """Record a successful stage execution to circuit breaker."""
        if self._circuit_breaker:
            self._circuit_breaker.record_success(stage)

    def _record_circuit_failure(self, stage: str, error: str) -> None:
        """Record a failed stage execution to circuit breaker."""
        if self._circuit_breaker:
            self._circuit_breaker.record_failure(stage, error)












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
        except Exception as e:
            logger.debug(f"[DataPipelineOrchestrator] Failed to emit resource constraint: {e}")

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
        for _resource_type, constraint in self._resource_constraints.items():
            # Constraints older than 60s are considered stale
            if now - constraint.get("time", 0) > 60:
                continue
            if constraint.get("severity") == "critical":
                return True
        return False

    def is_paused(self) -> bool:
        """Check if pipeline is currently paused."""
        return self._paused

    def get_pause_info(self) -> dict[str, Any] | None:
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

    def get_active_optimization(self) -> str | None:
        """Get the type of active optimization, or None."""
        return self._active_optimization

    def get_quality_distribution(self) -> dict[str, float]:
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





def get_pipeline_orchestrator() -> DataPipelineOrchestrator:
    """Get the global DataPipelineOrchestrator singleton."""
    global _pipeline_orchestrator
    if _pipeline_orchestrator is None:
        _pipeline_orchestrator = DataPipelineOrchestrator()
    return _pipeline_orchestrator


def wire_pipeline_events(
    auto_trigger: bool = True,  # Dec 2025: Default to True for full pipeline loop
    training_epochs: int | None = None,
    training_batch_size: int | None = None,
    training_model_version: str | None = None,
) -> DataPipelineOrchestrator:
    """Wire pipeline events to the orchestrator.

    Args:
        auto_trigger: If True, automatically trigger downstream stages (default: True)
        training_epochs: Override default training epochs
        training_batch_size: Override default training batch size
        training_model_version: Override default model version

    Returns:
        The wired DataPipelineOrchestrator instance
    """
    global _pipeline_orchestrator
    _pipeline_orchestrator = DataPipelineOrchestrator(auto_trigger=auto_trigger)

    # Apply training config overrides (December 2025 - CLI connection)
    if training_epochs is not None:
        _pipeline_orchestrator._config.training_epochs = training_epochs
    if training_batch_size is not None:
        _pipeline_orchestrator._config.training_batch_size = training_batch_size
    if training_model_version is not None:
        _pipeline_orchestrator._config.training_model_version = training_model_version

    _pipeline_orchestrator.subscribe_to_events()
    _pipeline_orchestrator.subscribe_to_data_events()  # December 2025
    return _pipeline_orchestrator


def get_pipeline_status() -> dict[str, Any]:
    """Convenience function to get pipeline status."""
    return get_pipeline_orchestrator().get_status()


def get_current_pipeline_stage() -> PipelineStage:
    """Convenience function to get current pipeline stage."""
    return get_pipeline_orchestrator().get_current_stage()


def get_pipeline_health() -> dict[str, Any]:
    """Convenience function to get pipeline health status.

    Returns a dict with:
    - healthy: bool - overall health
    - status: str - "healthy", "degraded", or "unhealthy"
    - issues: list[str] - detected problems
    - recommendations: list[str] - suggested fixes
    """
    return get_pipeline_orchestrator().get_health_status()


def is_pipeline_healthy() -> bool:
    """Quick check if pipeline is healthy.

    Returns True if no issues detected.
    """
    return get_pipeline_orchestrator().get_health_status().get("healthy", False)


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerState",
    "DataPipelineOrchestrator",
    "IterationRecord",
    "PipelineStage",
    "PipelineStats",
    "StageTransition",
    "get_current_pipeline_stage",
    "get_pipeline_health",
    "get_pipeline_orchestrator",
    "get_pipeline_status",
    "is_pipeline_healthy",
    "wire_pipeline_events",
]

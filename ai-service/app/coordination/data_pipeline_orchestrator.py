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

logger = logging.getLogger(__name__)


# =============================================================================
# Circuit Breaker for Pipeline Fault Tolerance (December 2025)
# =============================================================================

# December 2025: Import canonical CircuitState, alias as CircuitBreakerState for compatibility
try:
    from app.distributed.circuit_breaker import CircuitState as CircuitBreakerState
except ImportError:
    # Fallback if canonical module unavailable
    class CircuitBreakerState(Enum):
        """Circuit breaker states."""
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Circuit breaker for pipeline stage fault tolerance.

    Prevents cascading failures by opening the circuit after repeated failures,
    allowing the system to recover before resuming operations.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Circuit is tripped, requests are rejected
    - HALF_OPEN: Testing if service recovered, limited requests allowed

    Note (December 2025): This is a simplified local implementation for pipeline
    stages. For per-host/per-target circuit breaking with Prometheus metrics and
    exponential backoff, use app.distributed.circuit_breaker.CircuitBreaker instead.
    Migration planned for Q1 2026.
    """

    failure_threshold: int = 3
    reset_timeout_seconds: float = 60.0  # Reduced from 5 min to 1 min for faster recovery
    half_open_max_requests: int = 1

    # Internal state
    _state: CircuitBreakerState = field(default=CircuitBreakerState.CLOSED)
    _failure_count: int = 0
    _success_count: int = 0
    _last_failure_time: float = 0.0
    _half_open_requests: int = 0
    _failures_by_stage: dict[str, int] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        if self._state == CircuitBreakerState.OPEN:
            # Check if reset timeout has passed
            if time.time() - self._last_failure_time >= self.reset_timeout_seconds:
                self._transition_to_half_open()
                return False
            return True
        return False

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitBreakerState.CLOSED

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit state."""
        # Check for automatic transition from OPEN to HALF_OPEN
        if self._state == CircuitBreakerState.OPEN:
            if time.time() - self._last_failure_time >= self.reset_timeout_seconds:
                self._transition_to_half_open()
        return self._state

    def can_execute(self) -> bool:
        """Check if a request can be executed."""
        state = self.state  # This may trigger OPEN -> HALF_OPEN transition

        if state == CircuitBreakerState.CLOSED:
            return True
        elif state == CircuitBreakerState.HALF_OPEN:
            return self._half_open_requests < self.half_open_max_requests
        else:  # OPEN
            return False

    def record_success(self, stage: str = "") -> None:
        """Record a successful execution."""
        self._success_count += 1

        if self._state == CircuitBreakerState.HALF_OPEN:
            # Reset to closed on success in half-open state
            self._transition_to_closed()
            logger.info(
                f"[CircuitBreaker] Recovered, transitioning to CLOSED after success in {stage}"
            )

    def record_failure(self, stage: str, error: str = "") -> None:
        """Record a failed execution."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        self._failures_by_stage[stage] = self._failures_by_stage.get(stage, 0) + 1

        if self._state == CircuitBreakerState.HALF_OPEN:
            # Immediately trip back to open on any failure
            self._transition_to_open()
            logger.warning(
                f"[CircuitBreaker] Failed in HALF_OPEN, reopening circuit: {stage} - {error}"
            )
        elif self._state == CircuitBreakerState.CLOSED:
            if self._failure_count >= self.failure_threshold:
                self._transition_to_open()
                logger.warning(
                    f"[CircuitBreaker] Threshold reached ({self._failure_count}), "
                    f"opening circuit: {stage} - {error}"
                )

    def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        self._state = CircuitBreakerState.OPEN
        self._half_open_requests = 0

    def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        self._state = CircuitBreakerState.HALF_OPEN
        self._half_open_requests = 0
        logger.info("[CircuitBreaker] Reset timeout passed, transitioning to HALF_OPEN")

    def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._half_open_requests = 0

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._transition_to_closed()
        self._failures_by_stage.clear()
        logger.info("[CircuitBreaker] Manually reset")

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "failures_by_stage": dict(self._failures_by_stage),
            "last_failure_time": self._last_failure_time,
            "time_until_reset": max(
                0,
                self.reset_timeout_seconds - (time.time() - self._last_failure_time)
            ) if self._state == CircuitBreakerState.OPEN else 0,
        }


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


class DataPipelineOrchestrator:
    """Orchestrates the self-improvement pipeline stages.

    Tracks stage transitions, provides coordination between stages,
    and maintains pipeline-wide metrics and observability.
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
        cb_enabled = getattr(config, "circuit_breaker_enabled", True) if config else True
        cb_threshold = getattr(config, "circuit_breaker_failure_threshold", 3) if config else 3
        cb_timeout = getattr(config, "circuit_breaker_reset_timeout_seconds", 300.0) if config else 300.0
        cb_half_open = getattr(config, "circuit_breaker_half_open_max_requests", 1) if config else 1

        self._circuit_breaker = CircuitBreaker(
            failure_threshold=cb_threshold,
            reset_timeout_seconds=cb_timeout,
            half_open_max_requests=cb_half_open,
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
            True if successfully subscribed
        """
        if self._subscribed:
            return True

        try:
            # P0.5 (December 2025): Use get_router() instead of deprecated get_stage_event_bus()
            from app.coordination.event_router import StageEvent, get_router

            router = get_router()

            # Subscribe to all pipeline stage events
            router.subscribe(StageEvent.SELFPLAY_COMPLETE, self._on_selfplay_complete)
            router.subscribe(
                StageEvent.CANONICAL_SELFPLAY_COMPLETE, self._on_selfplay_complete
            )
            router.subscribe(StageEvent.GPU_SELFPLAY_COMPLETE, self._on_selfplay_complete)
            router.subscribe(StageEvent.SYNC_COMPLETE, self._on_sync_complete)
            router.subscribe(StageEvent.NPZ_EXPORT_COMPLETE, self._on_npz_export_complete)
            router.subscribe(StageEvent.TRAINING_STARTED, self._on_training_started)
            router.subscribe(StageEvent.TRAINING_COMPLETE, self._on_training_complete)
            router.subscribe(StageEvent.TRAINING_FAILED, self._on_training_failed)
            router.subscribe(StageEvent.EVALUATION_COMPLETE, self._on_evaluation_complete)
            router.subscribe(StageEvent.PROMOTION_COMPLETE, self._on_promotion_complete)
            router.subscribe(StageEvent.ITERATION_COMPLETE, self._on_iteration_complete)

            self._subscribed = True
            self._prefer_stage_events = True
            logger.info("[DataPipelineOrchestrator] Subscribed to stage events")
            return True

        except ImportError as e:
            logger.error(f"[DataPipelineOrchestrator] Cannot subscribe to events - stage_events not available: {e}")
            # This is a critical failure - orchestrator cannot function without events
            raise RuntimeError(f"EventRouter not available - pipeline cannot function: {e}")
        except Exception as e:
            logger.error(f"[DataPipelineOrchestrator] Failed to subscribe: {e}")
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
                logger.error(f"[DataPipelineOrchestrator] Callback error: {e}")

    def _ensure_iteration_record(self, iteration: int) -> IterationRecord:
        """Ensure an iteration record exists."""
        if iteration not in self._iteration_records:
            self._iteration_records[iteration] = IterationRecord(
                iteration=iteration,
                start_time=time.time(),
            )
        return self._iteration_records[iteration]

    async def _on_data_selfplay_complete(self, event: Any) -> None:
        """Handle SELFPLAY_COMPLETE data events as a fallback."""
        if not self._should_process_stage_data_event("SELFPLAY_COMPLETE"):
            return

        payload = getattr(event, "payload", {}) or {}
        config_key = payload.get("config_key") or payload.get("config")
        games_generated = payload.get("games_played", payload.get("games_generated", 0))
        metadata = {"config_key": config_key, **payload}

        if not config_key or not games_generated:
            return

        board_type, num_players = self._get_board_config(metadata=metadata)
        iteration = self._next_data_event_iteration()

        result = SimpleNamespace(
            iteration=iteration,
            board_type=board_type,
            num_players=num_players,
            games_generated=games_generated,
            success=payload.get("success", True),
            error=payload.get("error"),
            metadata=metadata,
        )
        await self._on_selfplay_complete(result)

    async def _on_data_sync_completed(self, event: Any) -> None:
        """Handle DATA_SYNC_COMPLETED data events as a fallback."""
        if not self._should_process_stage_data_event("DATA_SYNC_COMPLETED"):
            return

        payload = getattr(event, "payload", {}) or {}
        config_key = payload.get("config") or payload.get("config_key")
        games_synced = payload.get("games_synced", 0) or payload.get("files_synced", 0)
        metadata = {"config_key": config_key, **payload}
        iteration = self._current_iteration_for_data_event()

        result = SimpleNamespace(
            iteration=iteration,
            success=True,
            error=None,
            metadata=metadata,
        )
        if games_synced:
            metadata["games_synced"] = games_synced

        await self._on_sync_complete(result)

    async def _on_data_sync_failed(self, event: Any) -> None:
        """Handle DATA_SYNC_FAILED data events as a fallback."""
        if not self._should_process_stage_data_event("DATA_SYNC_FAILED"):
            return

        payload = getattr(event, "payload", {}) or {}
        metadata = {"config_key": payload.get("config") or payload.get("config_key"), **payload}
        iteration = self._current_iteration_for_data_event()

        result = SimpleNamespace(
            iteration=iteration,
            success=False,
            error=payload.get("error"),
            metadata=metadata,
        )
        await self._on_sync_complete(result)

    async def _on_data_training_completed(self, event: Any) -> None:
        """Handle TRAINING_COMPLETED data events as a fallback."""
        if not self._should_process_stage_data_event("TRAINING_COMPLETED"):
            return

        payload = getattr(event, "payload", {}) or {}
        config_key = payload.get("config") or payload.get("config_key")
        metadata = {"config_key": config_key, **payload}
        iteration = self._current_iteration_for_data_event()

        model_path = payload.get("checkpoint_path") or payload.get("model_path")
        model_id = payload.get("model_id")
        if not model_id and model_path:
            model_id = Path(model_path).stem

        result = SimpleNamespace(
            iteration=iteration,
            board_type=payload.get("board_type"),
            num_players=payload.get("num_players"),
            model_id=model_id,
            model_path=model_path,
            train_loss=payload.get("final_train_loss") or payload.get("train_loss"),
            val_loss=payload.get("final_val_loss") or payload.get("best_val_loss") or payload.get("val_loss"),
            success=True,
            error=None,
            metadata=metadata,
        )
        await self._on_training_complete(result)

    async def _on_data_training_failed(self, event: Any) -> None:
        """Handle TRAINING_FAILED data events as a fallback."""
        if not self._should_process_stage_data_event("TRAINING_FAILED"):
            return

        payload = getattr(event, "payload", {}) or {}
        metadata = {"config_key": payload.get("config") or payload.get("config_key"), **payload}
        iteration = self._current_iteration_for_data_event()

        result = SimpleNamespace(
            iteration=iteration,
            success=False,
            error=payload.get("error"),
            metadata=metadata,
        )
        await self._on_training_failed(result)

    async def _on_data_evaluation_completed(self, event: Any) -> None:
        """Handle EVALUATION_COMPLETED data events as a fallback."""
        if not self._should_process_stage_data_event("EVALUATION_COMPLETED"):
            return

        payload = getattr(event, "payload", {}) or {}
        metadata = {"config_key": payload.get("config") or payload.get("config_key"), **payload}
        iteration = self._current_iteration_for_data_event()

        result = SimpleNamespace(
            iteration=iteration,
            win_rate=payload.get("win_rate", 0.0),
            elo_delta=payload.get("elo_delta", 0.0),
            model_path=payload.get("model_path") or payload.get("checkpoint_path"),
            success=True,
            error=None,
            metadata=metadata,
        )
        await self._on_evaluation_complete(result)

    async def _on_data_evaluation_failed(self, event: Any) -> None:
        """Handle EVALUATION_FAILED data events as a fallback."""
        if not self._should_process_stage_data_event("EVALUATION_FAILED"):
            return

        payload = getattr(event, "payload", {}) or {}
        metadata = {"config_key": payload.get("config") or payload.get("config_key"), **payload}
        iteration = self._current_iteration_for_data_event()

        result = SimpleNamespace(
            iteration=iteration,
            success=False,
            error=payload.get("error"),
            metadata=metadata,
        )
        await self._on_evaluation_complete(result)

    async def _on_data_model_promoted(self, event: Any) -> None:
        """Handle MODEL_PROMOTED data events as a fallback."""
        if not self._should_process_stage_data_event("MODEL_PROMOTED"):
            return

        payload = getattr(event, "payload", {}) or {}
        metadata = {"config_key": payload.get("config") or payload.get("config_key"), **payload}
        iteration = self._current_iteration_for_data_event()

        result = SimpleNamespace(
            iteration=iteration,
            promoted=payload.get("promoted", True),
            promotion_reason=payload.get("promotion_reason") or payload.get("reason"),
            board_type=payload.get("board_type"),
            num_players=payload.get("num_players"),
            metadata=metadata,
        )
        await self._on_promotion_complete(result)

    async def _on_orphan_games_detected(self, event: Any) -> None:
        """Handle ORPHAN_GAMES_DETECTED events - trigger sync to recover orphan games.

        Orphan games are selfplay games stored on ephemeral nodes that haven't been
        synced to training nodes. This event is emitted by OrphanDetectionDaemon
        when it finds unsynced games. We trigger a priority sync to recover them.
        """
        payload = getattr(event, "payload", {}) or {}
        orphan_count = payload.get("orphan_count", 0)
        source_node = payload.get("source_node")
        config_key = payload.get("config_key")

        if orphan_count == 0:
            return

        logger.info(
            f"[DataPipelineOrchestrator] Orphan games detected: "
            f"{orphan_count} games from {source_node} ({config_key})"
        )

        # Record as pending work that needs sync
        self._orphan_games_pending = getattr(self, "_orphan_games_pending", 0) + orphan_count

        # Trigger priority sync if auto-trigger enabled
        if self.auto_trigger and self.auto_trigger_sync:
            try:
                from app.coordination.sync_facade import get_sync_facade

                facade = get_sync_facade()
                if facade:
                    # Priority sync for orphan recovery
                    await facade.trigger_priority_sync(
                        reason="orphan_games_recovery",
                        source_node=source_node,
                        config_key=config_key,
                    )
                    logger.info(
                        f"[DataPipelineOrchestrator] Triggered priority sync for orphan recovery"
                    )
            except Exception as e:
                logger.warning(f"[DataPipelineOrchestrator] Failed to trigger orphan sync: {e}")

    async def _on_orphan_games_registered(self, event: Any) -> None:
        """Handle ORPHAN_GAMES_REGISTERED events - update pipeline state.

        This event is emitted after orphan games are successfully synced and
        registered in the training data catalog. We can now proceed with export.
        """
        payload = getattr(event, "payload", {}) or {}
        registered_count = payload.get("registered_count", 0)
        config_key = payload.get("config_key")
        board_type = payload.get("board_type")
        num_players = payload.get("num_players")

        if registered_count == 0:
            return

        logger.info(
            f"[DataPipelineOrchestrator] Orphan games registered: "
            f"{registered_count} games for {config_key}"
        )

        # Update pending count
        pending = getattr(self, "_orphan_games_pending", 0)
        self._orphan_games_pending = max(0, pending - registered_count)

        # Emit NEW_GAMES_AVAILABLE for downstream consumers (e.g., export triggers)
        try:
            from app.distributed.data_events import emit_data_event, DataEventType

            await emit_data_event(
                event_type=DataEventType.NEW_GAMES_AVAILABLE,
                payload={
                    "board_type": board_type or "unknown",
                    "num_players": num_players or 2,
                    "games_count": registered_count,
                    "source": "orphan_recovery",
                    "config_key": config_key,
                },
            )
        except ImportError:
            pass  # data_events not available
        except Exception as e:
            logger.debug(f"[DataPipelineOrchestrator] Failed to emit new_games_available: {e}")

    async def _on_selfplay_complete(self, result) -> None:
        """Handle selfplay completion."""
        iteration = result.iteration
        self._ensure_iteration_record(iteration)

        # Track board configuration for downstream stages
        self._current_board_type = getattr(result, "board_type", None)
        self._current_num_players = getattr(result, "num_players", None)

        self._iteration_records[iteration].games_generated = result.games_generated
        self._total_games += result.games_generated

        if result.success:
            self._transition_to(
                PipelineStage.DATA_SYNC,
                iteration,
                metadata={"games_generated": result.games_generated},
            )

            # Auto-trigger data sync if enabled (December 2025)
            if self.auto_trigger and self.auto_trigger_sync:
                await self._auto_trigger_sync(iteration)
        else:
            self._transition_to(
                PipelineStage.IDLE,
                iteration,
                success=False,
                metadata={"error": result.error},
            )

    async def _auto_trigger_sync(self, iteration: int) -> None:
        """Auto-trigger data synchronization with prerequisite validation."""
        if not self._can_auto_trigger():
            return

        board_type, num_players = self._get_board_config()
        if not board_type or not num_players:
            logger.warning("[DataPipelineOrchestrator] Cannot auto-trigger sync: missing board config")
            return

        try:
            # Use PipelineTrigger for prerequisite validation (December 2025)
            from app.coordination.pipeline_triggers import PipelineTrigger
            trigger = PipelineTrigger()
            result = await trigger.trigger_sync_after_selfplay(board_type, num_players)

            if result.success:
                self._record_circuit_success("data_sync")
                logger.info(f"[DataPipelineOrchestrator] Sync triggered successfully: {result.message}")
            else:
                self._record_circuit_failure("data_sync", result.error or "Prerequisite check failed")
                logger.warning(f"[DataPipelineOrchestrator] Sync trigger failed: {result.message}")
        except Exception as e:
            logger.error(f"[DataPipelineOrchestrator] Auto-trigger sync failed: {e}")
            self._record_circuit_failure("data_sync", str(e))

    async def _on_sync_complete(self, result) -> None:
        """Handle data sync completion."""
        iteration = result.iteration

        if result.success:
            self._transition_to(
                PipelineStage.NPZ_EXPORT,
                iteration,
                metadata=result.metadata,
            )

            # Auto-trigger NPZ export if enabled (December 2025)
            if self.auto_trigger and self.auto_trigger_export:
                await self._auto_trigger_export(iteration)
        else:
            self._transition_to(
                PipelineStage.IDLE,
                iteration,
                success=False,
                metadata={"error": result.error},
            )

    async def _auto_trigger_export(self, iteration: int) -> None:
        """Auto-trigger NPZ export with prerequisite validation."""
        if not self._can_auto_trigger():
            return

        board_type, num_players = self._get_board_config()
        if not board_type or not num_players:
            logger.warning("[DataPipelineOrchestrator] Cannot auto-trigger export: missing board config")
            return

        try:
            # Use PipelineTrigger for prerequisite validation (December 2025)
            from app.coordination.pipeline_triggers import PipelineTrigger
            trigger = PipelineTrigger()
            result = await trigger.trigger_export_after_sync(board_type, num_players)

            if result.success:
                self._record_circuit_success("npz_export")
                # Store output path for training stage
                self._iteration_records[iteration].metadata = {
                    "npz_path": result.output_path
                }
                logger.info(f"[DataPipelineOrchestrator] Export triggered successfully: {result.message}")
            else:
                self._record_circuit_failure("npz_export", result.error or "Prerequisite check failed")
                logger.warning(f"[DataPipelineOrchestrator] Export trigger failed: {result.message}")
        except Exception as e:
            logger.error(f"[DataPipelineOrchestrator] Auto-trigger export failed: {e}")
            self._record_circuit_failure("npz_export", str(e))

    async def _on_npz_export_complete(self, result) -> None:
        """Handle NPZ export completion."""
        iteration = result.iteration

        if result.success:
            # Quality gate check before training (December 2025 - Phase 14)
            npz_path = getattr(result, "output_path", None) or result.metadata.get("output_path")
            if self.quality_gate_enabled and npz_path:
                quality_ok = await self._check_training_data_quality(npz_path, iteration)
                if not quality_ok:
                    logger.warning(
                        f"[DataPipelineOrchestrator] Quality gate blocked training for "
                        f"iteration {iteration} (quality={self._last_quality_score:.2f})"
                    )
                    await self._emit_training_blocked_by_quality(iteration, npz_path)
                    # Don't transition to training - stay at export complete
                    return

            self._transition_to(
                PipelineStage.TRAINING,
                iteration,
                metadata=result.metadata,
            )

            # Auto-trigger training if enabled (December 2025)
            if self.auto_trigger and self.auto_trigger_training:
                if npz_path:
                    await self._auto_trigger_training(iteration, npz_path)
        else:
            self._transition_to(
                PipelineStage.IDLE,
                iteration,
                success=False,
                metadata={"error": result.error},
            )

    async def _auto_trigger_training(self, iteration: int, npz_path: str) -> None:
        """Auto-trigger neural network training with prerequisite validation."""
        if not self._can_auto_trigger():
            return

        board_type, num_players = self._get_board_config()
        if not board_type or not num_players:
            logger.warning("[DataPipelineOrchestrator] Cannot auto-trigger training: missing board config")
            return

        try:
            # Use PipelineTrigger for prerequisite validation (December 2025)
            from app.coordination.pipeline_triggers import PipelineTrigger
            trigger = PipelineTrigger()
            result = await trigger.trigger_training_after_export(board_type, num_players)

            if result.success:
                self._record_circuit_success("training")
                # Store model path for evaluation stage
                if iteration in self._iteration_records:
                    self._iteration_records[iteration].model_id = result.metadata.get("model_id")
                logger.info(f"[DataPipelineOrchestrator] Training triggered successfully: {result.message}")
            else:
                self._record_circuit_failure("training", result.error or "Prerequisite check failed")
                logger.warning(f"[DataPipelineOrchestrator] Training trigger failed: {result.message}")
        except Exception as e:
            logger.error(f"[DataPipelineOrchestrator] Auto-trigger training failed: {e}")
            self._record_circuit_failure("training", str(e))

    # =========================================================================
    # Quality Gate Methods (December 2025 - Phase 14)
    # =========================================================================

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

    async def _trigger_data_regeneration(
        self, board_type: str, num_players: int, iteration: int
    ) -> None:
        """Trigger regeneration of training data when quality is low.

        Args:
            board_type: Board type
            num_players: Number of players
            iteration: Pipeline iteration
        """
        try:
            from app.coordination.event_router import DataEventType, get_router

            router = get_router()
            if router:
                await router.publish(
                    event_type=DataEventType.SELFPLAY_TARGET_UPDATED,
                    payload={
                        "config": f"{board_type}_{num_players}p",
                        "board_type": board_type,
                        "num_players": num_players,
                        "extra_games": 2000,  # Request more data
                        "reason": "quality_gate_failed",
                        "quality_score": self._last_quality_score,
                        "iteration": iteration,
                    },
                    source="DataPipelineOrchestrator",
                )
                logger.info(
                    f"[QualityGate] Triggered data regeneration for {board_type}_{num_players}p"
                )
        except Exception as e:
            logger.warning(f"[QualityGate] Failed to trigger data regeneration: {e}")

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

            # Auto-trigger evaluation if enabled (December 2025)
            if self.auto_trigger and self.auto_trigger_evaluation:
                model_path = getattr(result, "model_path", None) or result.metadata.get("model_path")
                if model_path:
                    await self._auto_trigger_evaluation(iteration, model_path)
        else:
            self._transition_to(
                PipelineStage.IDLE,
                iteration,
                success=False,
                metadata={"error": result.error},
            )

    async def _auto_trigger_evaluation(self, iteration: int, model_path: str) -> None:
        """Auto-trigger gauntlet evaluation."""
        if not self._can_auto_trigger():
            return

        board_type, num_players = self._get_board_config()
        if not board_type or not num_players:
            logger.warning("[DataPipelineOrchestrator] Cannot auto-trigger evaluation: missing board config")
            return

        try:
            # Phase 7: Use PipelineTrigger for prerequisite validation
            from app.coordination.pipeline_triggers import get_pipeline_trigger

            trigger = get_pipeline_trigger()
            logger.info(f"[DataPipelineOrchestrator] Auto-triggering evaluation for {model_path}")
            result = await trigger.trigger_evaluation_after_training(
                board_type=board_type,
                num_players=num_players,
                model_path=model_path,
            )

            if result.success:
                self._record_circuit_success("evaluation")
                # Store evaluation results for promotion stage
                if iteration in self._iteration_records:
                    self._iteration_records[iteration].elo_delta = result.metadata.get("elo_delta", 0.0)
            else:
                self._record_circuit_failure("evaluation", result.error or "Unknown error")
        except Exception as e:
            logger.error(f"[DataPipelineOrchestrator] Auto-trigger evaluation failed: {e}")
            self._record_circuit_failure("evaluation", str(e))

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

            # Auto-trigger promotion if enabled (December 2025)
            if self.auto_trigger and self.auto_trigger_promotion:
                model_path = getattr(result, "model_path", None) or result.metadata.get("model_path")
                gauntlet_results = result.metadata if hasattr(result, "metadata") else {}
                if model_path:
                    await self._auto_trigger_promotion(iteration, model_path, gauntlet_results)

            # December 27, 2025: Trigger model sync after successful evaluation
            # This ensures evaluated models are distributed to training nodes
            if self.auto_trigger and self.auto_trigger_sync:
                await self._trigger_model_sync_after_evaluation(result)
        else:
            self._transition_to(
                PipelineStage.IDLE,
                iteration,
                success=False,
                metadata={"error": result.error},
            )

    async def _auto_trigger_promotion(
        self, iteration: int, model_path: str, gauntlet_results: dict
    ) -> None:
        """Auto-trigger model promotion."""
        if not self._can_auto_trigger():
            return

        board_type, num_players = self._get_board_config()
        if not board_type or not num_players:
            logger.warning("[DataPipelineOrchestrator] Cannot auto-trigger promotion: missing board config")
            return

        try:
            # Phase 7: Use PipelineTrigger for prerequisite validation
            from app.coordination.pipeline_triggers import get_pipeline_trigger

            # Extract win rates from gauntlet_results
            win_rates = gauntlet_results.get("win_rates", {})
            win_rate_vs_random = win_rates.get("random", gauntlet_results.get("win_rate_vs_random", 0.0))
            win_rate_vs_heuristic = win_rates.get("heuristic", gauntlet_results.get("win_rate_vs_heuristic", 0.0))

            trigger = get_pipeline_trigger()
            logger.info(f"[DataPipelineOrchestrator] Auto-triggering promotion for {model_path}")
            result = await trigger.trigger_promotion_after_evaluation(
                board_type=board_type,
                num_players=num_players,
                model_path=model_path,
                win_rate_vs_random=win_rate_vs_random,
                win_rate_vs_heuristic=win_rate_vs_heuristic,
            )

            if result.success:
                self._record_circuit_success("promotion")
            else:
                # Promotion failure is not a circuit-breaking event
                logger.info(f"[DataPipelineOrchestrator] Promotion skipped: {result.metadata.get('reason', 'Unknown')}")
        except Exception as e:
            logger.error(f"[DataPipelineOrchestrator] Auto-trigger promotion failed: {e}")
            # Don't record as circuit failure - promotion is optional

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

        # Feed back to curriculum (December 2025)
        await self._update_curriculum_on_promotion(result)

        # December 27, 2025: Trigger model sync after successful promotion
        # This ensures promoted models are distributed to all cluster nodes
        if result.promoted and self.auto_trigger and self.auto_trigger_sync:
            await self._trigger_model_sync_after_promotion(result)

    async def _update_curriculum_on_promotion(self, result) -> None:
        """Update curriculum weights based on promotion result.

        This closes the feedback loop: promotion results affect future
        training resource allocation via curriculum weights.

        December 2025: Added to complete the self-improvement loop.
        """
        try:
            from app.training.curriculum_feedback import get_curriculum_feedback

            # Get config key from result or tracked state
            config_key = None
            if hasattr(result, "board_type") and hasattr(result, "num_players"):
                config_key = f"{result.board_type}_{result.num_players}p"
            elif self._current_board_type and self._current_num_players:
                config_key = f"{self._current_board_type}_{self._current_num_players}p"
            elif hasattr(result, "metadata") and result.metadata:
                config_key = result.metadata.get("config_key")

            if not config_key:
                logger.debug("[DataPipelineOrchestrator] No config_key for curriculum update")
                return

            feedback = get_curriculum_feedback()
            feedback.record_promotion(
                config_key=config_key,
                promoted=result.promoted,
                new_elo=getattr(result, "new_elo", None) or result.metadata.get("new_elo"),
                promotion_reason=getattr(result, "promotion_reason", "") or result.metadata.get("reason", ""),
            )

            logger.info(
                f"[DataPipelineOrchestrator] Curriculum updated for {config_key}: "
                f"promoted={result.promoted}"
            )

        except ImportError:
            logger.debug("[DataPipelineOrchestrator] curriculum_feedback not available")
        except Exception as e:
            logger.warning(f"[DataPipelineOrchestrator] Curriculum update failed: {e}")

    async def _trigger_model_sync_after_evaluation(self, result) -> None:
        """Trigger model sync after evaluation completes.

        December 27, 2025: Ensures evaluated models are distributed to training
        nodes so they can be used for further selfplay or comparison.

        Args:
            result: Evaluation result with model_path and metadata
        """
        try:
            from app.coordination.sync_facade import get_sync_facade

            # Get config key for logging
            config_key = None
            if hasattr(result, "metadata") and result.metadata:
                config_key = result.metadata.get("config_key")
            if not config_key and self._current_board_type and self._current_num_players:
                config_key = f"{self._current_board_type}_{self._current_num_players}p"

            facade = get_sync_facade()
            logger.info(
                f"[DataPipelineOrchestrator] Triggering model sync after evaluation "
                f"({config_key or 'unknown config'})"
            )
            await facade.trigger_priority_sync(
                reason="post_evaluation_sync",
                config_key=config_key,
                data_type="models",
            )
        except ImportError:
            logger.debug("[DataPipelineOrchestrator] sync_facade not available for eval sync")
        except Exception as e:
            logger.warning(f"[DataPipelineOrchestrator] Model sync after evaluation failed: {e}")

    async def _trigger_model_sync_after_promotion(self, result) -> None:
        """Trigger model sync after promotion completes.

        December 27, 2025: Ensures promoted models are distributed to all cluster
        nodes so they can use the new best model for selfplay.

        Args:
            result: Promotion result with model info and metadata
        """
        try:
            from app.coordination.sync_facade import get_sync_facade

            # Get config key
            config_key = None
            if hasattr(result, "board_type") and hasattr(result, "num_players"):
                config_key = f"{result.board_type}_{result.num_players}p"
            elif hasattr(result, "metadata") and result.metadata:
                config_key = result.metadata.get("config_key")
            if not config_key and self._current_board_type and self._current_num_players:
                config_key = f"{self._current_board_type}_{self._current_num_players}p"

            facade = get_sync_facade()
            logger.info(
                f"[DataPipelineOrchestrator] Triggering model sync after promotion "
                f"({config_key or 'unknown config'})"
            )
            await facade.trigger_priority_sync(
                reason="post_promotion_sync",
                config_key=config_key,
                data_type="models",
            )
        except ImportError:
            logger.debug("[DataPipelineOrchestrator] sync_facade not available for promotion sync")
        except Exception as e:
            logger.warning(f"[DataPipelineOrchestrator] Model sync after promotion failed: {e}")

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
    # Circuit Breaker and Auto-Trigger Helpers (December 2025)
    # =========================================================================

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
        if (self._paused and "Backpressure" in (self._pause_reason or "")
                and not self._has_critical_constraints()):
            await self._resume_pipeline()

    async def _on_game_synced(self, event) -> None:
        """Handle GAME_SYNCED - games synced to training nodes.

        December 2025: Wire previously orphaned event. When games are synced
        to training nodes (via AutoSyncDaemon), this can trigger NPZ export
        if sufficient games have accumulated.

        Actions:
        - Track total games synced for metrics
        - Optionally trigger export if auto_trigger_export enabled and
          games synced exceed threshold
        """
        payload = event.payload if hasattr(event, 'payload') else event

        node_id = payload.get("node_id", "unknown")
        games_pushed = payload.get("games_pushed", 0)
        target_nodes = payload.get("target_nodes", [])
        is_ephemeral = payload.get("is_ephemeral", False)

        # Track cumulative games synced
        if not hasattr(self, "_games_synced_count"):
            self._games_synced_count = 0
        self._games_synced_count += games_pushed

        logger.debug(
            f"[DataPipelineOrchestrator] Games synced: {games_pushed} from {node_id} "
            f"to {len(target_nodes)} nodes (ephemeral={is_ephemeral})"
        )

        # If in SYNC stage and auto-trigger is enabled, consider triggering export
        if (self._current_stage == PipelineStage.SYNC
                and self.auto_trigger
                and self.auto_trigger_export):
            # Transition to NPZ_EXPORT stage
            self._transition_to(
                PipelineStage.NPZ_EXPORT,
                self._current_iteration,
                metadata={
                    "games_synced": games_pushed,
                    "source_node": node_id,
                    "target_nodes": target_nodes,
                },
            )
            await self._auto_trigger_export(self._current_iteration)

    async def _on_exploration_boost(self, event) -> None:
        """Handle EXPLORATION_BOOST - request to boost exploration temperature.

        December 2025: Wire previously orphaned event. Emitted when curriculum
        feedback detects that exploration diversity is low or training is
        plateauing on certain configurations.

        Actions:
        - Log the boost request for metrics
        - Forward to curriculum integration if available
        - Update exploration multiplier in pipeline metadata
        """
        payload = event.payload if hasattr(event, "payload") else event

        config_key = payload.get("config_key", "unknown")
        boost_factor = payload.get("boost_factor", 1.2)
        reason = payload.get("reason", "exploration_plateau")

        logger.info(
            f"[DataPipelineOrchestrator] Exploration boost requested: "
            f"config={config_key}, factor={boost_factor:.2f}, reason={reason}"
        )

        # Track exploration boost events for pipeline metrics
        if not hasattr(self, "_exploration_boost_count"):
            self._exploration_boost_count = 0
        self._exploration_boost_count += 1

        # Forward to curriculum integration if wired
        try:
            from app.training.curriculum_integration import (
                get_curriculum_integration,
            )
            curriculum = get_curriculum_integration()
            if curriculum and hasattr(curriculum, "apply_exploration_boost"):
                await curriculum.apply_exploration_boost(config_key, boost_factor)
                logger.debug(
                    f"[DataPipelineOrchestrator] Forwarded exploration boost to curriculum"
                )
        except ImportError:
            pass  # Curriculum integration not available
        except Exception as e:
            logger.warning(f"[DataPipelineOrchestrator] Failed to forward exploration boost: {e}")

    async def _on_sync_triggered(self, event) -> None:
        """Handle SYNC_TRIGGERED - data sync initiated due to staleness.

        December 2025: Wire previously orphaned event. Emitted when AutoSyncDaemon
        or SyncFacade triggers a sync due to stale data detection.

        Actions:
        - Log the sync trigger for metrics
        - Update pipeline stage if appropriate
        - Track sync frequency for feedback
        """
        payload = event.payload if hasattr(event, "payload") else event

        reason = payload.get("reason", "stale_data")
        config_key = payload.get("config_key")
        data_age_hours = payload.get("data_age_hours", 0)
        source = payload.get("source", "unknown")

        logger.info(
            f"[DataPipelineOrchestrator] Sync triggered: reason={reason}, "
            f"config={config_key}, age={data_age_hours:.1f}h, source={source}"
        )

        # Track sync trigger frequency
        if not hasattr(self, "_sync_trigger_count"):
            self._sync_trigger_count = 0
        self._sync_trigger_count += 1

        # If we're idle and sync was triggered, transition to SYNC stage
        if self._current_stage == PipelineStage.IDLE and self.auto_trigger:
            self._transition_to(
                PipelineStage.SYNC,
                self._current_iteration,
                metadata={
                    "trigger_reason": reason,
                    "config_key": config_key,
                    "data_age_hours": data_age_hours,
                },
            )

    async def _on_data_stale(self, event) -> None:
        """Handle DATA_STALE - training data has become stale.

        December 2025: Wire this previously orphaned event. Emitted by
        TrainingFreshness or train_cli.py when data age exceeds threshold.

        Actions:
        - Log the stale data alert
        - Trigger priority sync via SyncFacade
        - Track stale data frequency for health monitoring
        """
        payload = event.payload if hasattr(event, "payload") else event

        config_key = payload.get("config_key")
        data_age_hours = payload.get("data_age_hours", 0)
        max_age_hours = payload.get("max_age_hours", 1.0)
        source = payload.get("source", "unknown")

        logger.warning(
            f"[DataPipelineOrchestrator] DATA_STALE received: "
            f"config={config_key}, age={data_age_hours:.1f}h (max={max_age_hours:.1f}h), "
            f"source={source}"
        )

        # Track stale data frequency for health monitoring
        if not hasattr(self, "_stale_data_count"):
            self._stale_data_count = 0
        self._stale_data_count += 1

        # Trigger priority sync if we have SyncFacade available
        try:
            from app.coordination.sync_facade import get_sync_facade

            facade = get_sync_facade()
            if facade:
                from app.core.async_context import fire_and_forget

                async def trigger_sync():
                    await facade.trigger_priority_sync(
                        reason="stale_data",
                        config_key=config_key,
                        data_type="games",
                    )

                fire_and_forget(
                    trigger_sync(),
                    error_callback=lambda exc: logger.debug(
                        f"Priority sync trigger failed: {exc}"
                    ),
                )
                logger.info(
                    f"[DataPipelineOrchestrator] Triggered priority sync for {config_key}"
                )
        except ImportError:
            logger.debug("[DataPipelineOrchestrator] SyncFacade not available")

    async def _on_new_games_available(self, event) -> None:
        """Handle NEW_GAMES_AVAILABLE - new game data ready for processing.

        December 2025 Phase 11: Wire NEW_GAMES_AVAILABLE to trigger NPZ export
        when new games are available. This closes the loop from selfplay -> export.

        Actions:
        - Track new game availability
        - Consider triggering NPZ export if threshold met
        - Update pipeline state for monitoring
        """
        payload = event.payload if hasattr(event, "payload") else event

        config_key = payload.get("config_key")
        game_count = payload.get("count", payload.get("game_count", 0))
        source = payload.get("source", payload.get("host", "unknown"))

        logger.debug(
            f"[DataPipelineOrchestrator] NEW_GAMES_AVAILABLE: "
            f"config={config_key}, count={game_count}, source={source}"
        )

        # Track new games for this iteration
        if not hasattr(self, "_new_games_tracker"):
            self._new_games_tracker: dict[str, int] = {}
        if config_key:
            self._new_games_tracker[config_key] = (
                self._new_games_tracker.get(config_key, 0) + game_count
            )

        # Update stats
        self._stats["total_games"] += game_count

    async def _on_regression_detected(self, event) -> None:
        """Handle REGRESSION_DETECTED - model performance dropped.

        December 2025 Phase 11: Wire REGRESSION_DETECTED to pause training
        progression and track regression events for health monitoring.

        Actions:
        - Log the regression alert
        - Track regression events for health monitoring
        - Consider pausing training progression
        """
        payload = event.payload if hasattr(event, "payload") else event

        config_key = payload.get("config_key")
        severity = payload.get("severity", "unknown")
        elo_change = payload.get("elo_change", 0)
        reason = payload.get("reason", "")

        logger.warning(
            f"[DataPipelineOrchestrator] REGRESSION_DETECTED: "
            f"config={config_key}, severity={severity}, elo_change={elo_change}, "
            f"reason={reason}"
        )

        # Track regression events for health monitoring
        if not hasattr(self, "_regression_count"):
            self._regression_count = 0
        self._regression_count += 1

        # Store last regression for diagnostics
        self._last_regression = {
            "config_key": config_key,
            "severity": severity,
            "elo_change": elo_change,
            "reason": reason,
            "timestamp": time.time(),
        }

        # If severity is severe/critical, consider pausing training
        if severity in ("severe", "critical"):
            logger.warning(
                f"[DataPipelineOrchestrator] Severe regression detected for {config_key}, "
                "training progression may be paused"
            )
            # Update stage metadata to reflect regression
            self._stage_metadata["regression_detected"] = True
            self._stage_metadata["regression_severity"] = severity

    async def _on_promotion_failed(self, event) -> None:
        """Handle PROMOTION_FAILED - model failed promotion criteria.

        December 2025 Phase 11: Wire PROMOTION_FAILED to track failed promotions
        and update pipeline state appropriately.

        Actions:
        - Log the promotion failure
        - Track failed promotions for monitoring
        - Update pipeline state (stay in EVALUATION or reset)
        """
        payload = event.payload if hasattr(event, "payload") else event

        config_key = payload.get("config_key")
        reason = payload.get("reason", payload.get("error", "unknown"))
        model_path = payload.get("model_path", payload.get("model_id", ""))

        logger.warning(
            f"[DataPipelineOrchestrator] PROMOTION_FAILED: "
            f"config={config_key}, reason={reason}, model={model_path}"
        )

        # Track promotion failures for monitoring
        if not hasattr(self, "_promotion_failure_count"):
            self._promotion_failure_count = 0
        self._promotion_failure_count += 1

        # Store last failure for diagnostics
        self._last_promotion_failure = {
            "config_key": config_key,
            "reason": reason,
            "model_path": model_path,
            "timestamp": time.time(),
        }

        # If we're in promotion stage, transition back to evaluation
        if self._current_stage == PipelineStage.PROMOTION:
            self._transition_to(
                PipelineStage.EVALUATION,
                self._current_iteration,
                success=False,
                metadata={"promotion_failed": True, "reason": reason},
            )

    async def _on_promotion_candidate(self, event) -> None:
        """Handle PROMOTION_CANDIDATE - model ready for promotion evaluation.

        This handler was added December 2025 to wire the previously orphaned
        PROMOTION_CANDIDATE event. It's emitted by PromotionController when
        a model exceeds win rate thresholds in evaluation.

        Actions:
        - Logs the candidate for tracking
        - Updates pipeline state if in EVALUATION stage
        - Emits curriculum feedback event for training adjustments
        """
        payload = event.payload if hasattr(event, 'payload') else event

        model_id = payload.get("model_id", "unknown")
        board_type = payload.get("board_type", "unknown")
        num_players = payload.get("num_players", 2)
        win_rate = payload.get("win_rate_vs_heuristic", 0.0)

        logger.info(
            f"[DataPipelineOrchestrator] Promotion candidate: {model_id} "
            f"({board_type}_{num_players}p, {win_rate:.1%} vs heuristic)"
        )

        # Track candidates for this iteration
        if not hasattr(self, "_promotion_candidates"):
            self._promotion_candidates = []
        self._promotion_candidates.append({
            "model_id": model_id,
            "board_type": board_type,
            "num_players": num_players,
            "win_rate": win_rate,
            "timestamp": time.time(),
        })

        # If we're in evaluation stage, update transition tracking
        if self._current_stage == PipelineStage.EVALUATION:
            self._stage_metadata["candidates"] = len(self._promotion_candidates)

    async def _on_database_created(self, event) -> None:
        """Handle DATABASE_CREATED - new game database file created.

        This handler enables immediate registration and pipeline triggering
        when new databases are created, preventing orphaned databases.

        Added: December 2025 - Phase 4A.3
        """
        payload = event.payload if hasattr(event, 'payload') else event
        db_path = payload.get("db_path", "")
        board_type = payload.get("board_type", "")
        num_players = payload.get("num_players", 0)

        logger.info(
            f"[DataPipelineOrchestrator] New database created: {db_path} "
            f"({board_type}_{num_players}p)"
        )

        # Track for sync triggering if threshold is met
        if not hasattr(self, "_new_databases"):
            self._new_databases = []
        self._new_databases.append({
            "db_path": db_path,
            "board_type": board_type,
            "num_players": num_players,
            "timestamp": time.time(),
        })

    async def _on_training_threshold_reached(self, event) -> None:
        """Handle TRAINING_THRESHOLD_REACHED - enough games for training.

        Triggers NPZ export and training when game threshold is met.
        Added: December 2025
        """
        payload = event.payload if hasattr(event, 'payload') else event
        config = payload.get("config", "")
        games = payload.get("games", 0)

        logger.info(
            f"[DataPipelineOrchestrator] Training threshold reached: "
            f"{config} ({games} games)"
        )

        # Auto-trigger export if enabled and we're in appropriate stage
        if self.auto_trigger and self.auto_trigger_export:
            if self._current_stage in [PipelineStage.IDLE, PipelineStage.DATA_SYNC]:
                board_type = config.split("_")[0] if "_" in config else ""
                num_players_str = config.split("_")[1].replace("p", "") if "_" in config else "2"
                try:
                    num_players = int(num_players_str)
                    iteration = self._current_iteration + 1
                    await self._auto_trigger_export(iteration, board_type, num_players)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse config {config}: {e}")

    async def _on_promotion_started(self, event) -> None:
        """Handle PROMOTION_STARTED - promotion process initiated.

        Tracks promotion attempts and updates pipeline stage.
        Added: December 2025
        """
        payload = event.payload if hasattr(event, 'payload') else event
        config = payload.get("config", "")
        model_id = payload.get("model_id", "")

        logger.info(
            f"[DataPipelineOrchestrator] Promotion started: {model_id} ({config})"
        )

        # Transition to promotion stage if in evaluation
        if self._current_stage == PipelineStage.EVALUATION:
            iteration = self._current_iteration
            self._transition_to(
                PipelineStage.PROMOTION,
                iteration,
                metadata={"model_id": model_id, "config": config},
            )

    async def _on_work_queued(self, event) -> None:
        """Handle WORK_QUEUED - work added to distributed queue.

        Logs work queue activity for pipeline observability.
        Added: December 2025
        """
        payload = event.payload if hasattr(event, 'payload') else event
        work_type = payload.get("work_type", "unknown")
        config = payload.get("config", "")

        logger.debug(
            f"[DataPipelineOrchestrator] Work queued: {work_type} ({config})"
        )

        # Track work queue depth for backpressure decisions
        if not hasattr(self, "_queued_work_count"):
            self._queued_work_count = 0
        self._queued_work_count += 1

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

    def get_current_stage(self) -> PipelineStage:
        """Get the current pipeline stage."""
        return self._current_stage

    def get_current_iteration(self) -> int:
        """Get the current iteration number."""
        return self._current_iteration

    def get_iteration_record(self, iteration: int) -> IterationRecord | None:
        """Get record for a specific iteration."""
        if iteration in self._iteration_records:
            return self._iteration_records[iteration]
        for record in self._completed_iterations:
            if record.iteration == iteration:
                return record
        return None

    def get_recent_transitions(self, limit: int = 20) -> list[StageTransition]:
        """Get recent stage transitions."""
        return self._transitions[-limit:]

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
        if cb_status.get("state") == "open":
            issues.append("Circuit breaker is OPEN - pipeline blocked due to failures")
            recommendations.append(
                f"Wait {cb_status.get('time_until_retry', 0):.0f}s for auto-recovery or reset manually"
            )
        elif cb_status.get("state") == "half_open":
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
            pause_duration = now - self._pause_time if self._pause_time else 0
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

_pipeline_orchestrator: DataPipelineOrchestrator | None = None


def get_pipeline_orchestrator() -> DataPipelineOrchestrator:
    """Get the global DataPipelineOrchestrator singleton."""
    global _pipeline_orchestrator
    if _pipeline_orchestrator is None:
        _pipeline_orchestrator = DataPipelineOrchestrator()
    return _pipeline_orchestrator


def wire_pipeline_events(
    auto_trigger: bool = False,
    training_epochs: int | None = None,
    training_batch_size: int | None = None,
    training_model_version: str | None = None,
) -> DataPipelineOrchestrator:
    """Wire pipeline events to the orchestrator.

    Args:
        auto_trigger: If True, automatically trigger downstream stages
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

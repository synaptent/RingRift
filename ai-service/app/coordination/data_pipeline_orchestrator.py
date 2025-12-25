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


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker for pipeline stage fault tolerance.

    Prevents cascading failures by opening the circuit after repeated failures,
    allowing the system to recover before resuming operations.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Circuit is tripped, requests are rejected
    - HALF_OPEN: Testing if service recovered, limited requests allowed
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

        # Callbacks for stage transitions
        self._stage_callbacks: dict[PipelineStage, list[Callable]] = {}

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

    def subscribe_to_events(self) -> bool:
        """Subscribe to pipeline stage events.

        Returns:
            True if successfully subscribed
        """
        if self._subscribed:
            return True

        try:
            from app.coordination.event_router import StageEvent, get_stage_event_bus

            bus = get_stage_event_bus()

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
        """Auto-trigger data synchronization."""
        if not self._can_auto_trigger():
            return

        board_type, num_players = self._get_board_config()
        if not board_type or not num_players:
            logger.warning("[DataPipelineOrchestrator] Cannot auto-trigger sync: missing board config")
            return

        try:
            from app.coordination.pipeline_actions import trigger_data_sync

            logger.info(f"[DataPipelineOrchestrator] Auto-triggering sync for {board_type}_{num_players}p")
            result = await trigger_data_sync(board_type, num_players, iteration)

            if result.success:
                self._record_circuit_success("data_sync")
            else:
                self._record_circuit_failure("data_sync", result.error or "Unknown error")
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
        """Auto-trigger NPZ export."""
        if not self._can_auto_trigger():
            return

        board_type, num_players = self._get_board_config()
        if not board_type or not num_players:
            logger.warning("[DataPipelineOrchestrator] Cannot auto-trigger export: missing board config")
            return

        try:
            from app.coordination.pipeline_actions import trigger_npz_export

            logger.info(f"[DataPipelineOrchestrator] Auto-triggering export for {board_type}_{num_players}p")
            result = await trigger_npz_export(board_type, num_players, iteration)

            if result.success:
                self._record_circuit_success("npz_export")
                # Store output path for training stage
                self._iteration_records[iteration].metadata = {
                    "npz_path": result.output_path
                }
            else:
                self._record_circuit_failure("npz_export", result.error or "Unknown error")
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
        """Auto-trigger neural network training."""
        if not self._can_auto_trigger():
            return

        board_type, num_players = self._get_board_config()
        if not board_type or not num_players:
            logger.warning("[DataPipelineOrchestrator] Cannot auto-trigger training: missing board config")
            return

        try:
            from app.coordination.pipeline_actions import trigger_training

            # Get training config from pipeline config (December 2025)
            training_epochs = getattr(self._config, "training_epochs", 50)
            training_batch_size = getattr(self._config, "training_batch_size", 512)

            logger.info(
                f"[DataPipelineOrchestrator] Auto-triggering training for {board_type}_{num_players}p "
                f"(epochs={training_epochs}, batch_size={training_batch_size})"
            )
            result = await trigger_training(
                board_type, num_players, npz_path, iteration,
                batch_size=training_batch_size,
                epochs=training_epochs,
            )

            if result.success:
                self._record_circuit_success("training")
                # Store model path for evaluation stage
                if iteration in self._iteration_records:
                    self._iteration_records[iteration].model_id = result.metadata.get("model_id")
            else:
                self._record_circuit_failure("training", result.error or "Unknown error")
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

            # Check if file exists
            if not Path(npz_path).exists():
                logger.warning(f"[QualityGate] NPZ file not found: {npz_path}")
                return True  # Allow training if we can't check

            # Load NPZ and check for quality metadata
            with np.load(npz_path, allow_pickle=True) as data:
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
            from app.coordination.event_router import get_router

            router = get_router()
            if router:
                await router.publish(
                    event_type="SELFPLAY_TARGET_UPDATED",
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
            from app.coordination.pipeline_actions import trigger_evaluation

            logger.info(f"[DataPipelineOrchestrator] Auto-triggering evaluation for {model_path}")
            result = await trigger_evaluation(model_path, board_type, num_players, iteration)

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
            from app.coordination.pipeline_actions import trigger_promotion

            logger.info(f"[DataPipelineOrchestrator] Auto-triggering promotion for {model_path}")
            result = await trigger_promotion(
                model_path, gauntlet_results, board_type, num_players, iteration
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
            from app.distributed.cluster_monitor import ClusterMonitor

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


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerState",
    "DataPipelineOrchestrator",
    "IterationRecord",
    "PipelineStage",
    "PipelineStats",
    "StageTransition",
    "get_current_pipeline_stage",
    "get_pipeline_orchestrator",
    "get_pipeline_status",
    "wire_pipeline_events",
]

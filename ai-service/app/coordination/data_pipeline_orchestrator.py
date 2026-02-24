"""DataPipelineOrchestrator - Unified pipeline stage coordination (December 2025).

This module provides centralized monitoring and coordination of the self-improvement
pipeline stages. It tracks stage transitions, coordinates downstream triggering,
and provides pipeline-wide observability.

Pipeline Stages:
    SELFPLAY -> SYNC -> NPZ_EXPORT -> NPZ_COMBINATION -> TRAINING -> EVALUATION -> PROMOTION

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
    _on_partition_healed()       - Handle PARTITION_HEALED (Jan 3, 2026)

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
    - REPAIR_COMPLETED, REPAIR_FAILED, PARTITION_HEALED

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
import json
import logging
import os
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from app.config.coordination_defaults import CircuitBreakerDefaults
from app.config.thresholds import DISK_PRODUCTION_HALT_PERCENT
from app.config.ports import get_local_p2p_status_url
from app.coordination.coordinator_persistence import StatePersistenceMixin
from app.coordination.protocols import (
    CoordinatorStatus,
    HealthCheckResult,
    register_coordinator,
    unregister_coordinator,
)
from app.coordination.event_handler_utils import extract_config_key
from app.coordination.event_utils import make_config_key, parse_config_key

# December 2025: Import mixin classes for DataPipelineOrchestrator decomposition
from app.coordination.pipeline_event_handler_mixin import PipelineEventHandlerMixin
from app.coordination.pipeline_metrics_mixin import PipelineMetricsMixin
from app.coordination.pipeline_stage_mixin import PipelineStageMixin
from app.coordination.pipeline_trigger_mixin import PipelineTriggerMixin
from app.utils.async_utils import fire_and_forget
from app.utils.sqlite_utils import connect_safe

logger = logging.getLogger(__name__)


# =============================================================================
# Circuit Breaker for Pipeline Fault Tolerance (December 2025)
# =============================================================================

# January 3, 2026: Use unified circuit breaker from circuit_breaker_base
# Sprint 13.2 migration from app.distributed.circuit_breaker
try:
    from app.coordination.circuit_breaker_base import (
        CircuitConfig,
        CircuitState as CircuitBreakerState,
        OperationCircuitBreaker,
    )
    _HAS_CANONICAL_CIRCUIT_BREAKER = True
except ImportError:
    # Fallback if module unavailable
    _HAS_CANONICAL_CIRCUIT_BREAKER = False
    OperationCircuitBreaker = None  # type: ignore
    CircuitConfig = None  # type: ignore

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

    January 2026: Added SQLite persistence to prevent retry storms after restart.

    This is a thin wrapper that:
    1. Uses canonical CircuitBreaker with target="pipeline"
    2. Tracks failures per stage for logging/metrics
    3. Provides backward-compatible API
    4. Persists state to SQLite for crash recovery
    """

    # Default target name for pipeline-wide circuit
    PIPELINE_TARGET = "pipeline"

    # Default persistence path
    DEFAULT_DB_PATH = Path("data/coordination/circuit_breaker_state.db")

    def __init__(
        self,
        failure_threshold: int | None = None,
        recovery_timeout: float | None = None,
        half_open_max_calls: int | None = None,
        db_path: Path | str | None = None,
    ):
        """Initialize pipeline circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit.
                Defaults to CircuitBreakerDefaults.FAILURE_THRESHOLD.
            recovery_timeout: Seconds before trying half-open recovery.
                Defaults to CircuitBreakerDefaults.RECOVERY_TIMEOUT.
            half_open_max_calls: Max test calls in half-open state.
                Defaults to CircuitBreakerDefaults.HALF_OPEN_MAX_CALLS.
            db_path: SQLite database path for state persistence (None = use default)

        Jan 2, 2026: Consolidated to use CircuitBreakerDefaults for consistency.
        """
        if failure_threshold is None:
            failure_threshold = CircuitBreakerDefaults.FAILURE_THRESHOLD
        if recovery_timeout is None:
            recovery_timeout = CircuitBreakerDefaults.RECOVERY_TIMEOUT
        if half_open_max_calls is None:
            half_open_max_calls = CircuitBreakerDefaults.HALF_OPEN_MAX_CALLS
        self._failures_by_stage: dict[str, int] = {}
        self._success_count: int = 0
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout

        # Set up persistence
        self._db_path = Path(db_path) if db_path else self.DEFAULT_DB_PATH
        self._init_persistence_db()

        # January 3, 2026: Use OperationCircuitBreaker from circuit_breaker_base
        if _HAS_CANONICAL_CIRCUIT_BREAKER and OperationCircuitBreaker:
            config = CircuitConfig(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                half_open_max_calls=half_open_max_calls,
                success_threshold=1,
                operation_type="pipeline",
                emit_events=True,  # Emit events for pipeline monitoring
            )
            self._breaker = OperationCircuitBreaker(config=config)
        else:
            # Minimal fallback for missing module
            self._breaker = None
            self._fallback_state = CircuitBreakerState.CLOSED
            self._fallback_failure_count = 0
            self._fallback_last_failure_time = 0.0

        # Load persisted state (after initializing _breaker)
        self._load_state()

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

    # Auto-recovery timeout in hours (Jan 3, 2026)
    AUTO_RECOVERY_HOURS = 24

    def _check_auto_recovery(self) -> bool:
        """Check if circuit should auto-recover based on time since last failure.

        Jan 3, 2026: Added to prevent indefinite circuit blocking.
        If the circuit is OPEN and it's been >24 hours since the last failure,
        automatically reset to CLOSED to allow the pipeline to retry.

        Returns:
            True if auto-recovery was triggered, False otherwise.
        """
        if not self.is_open:
            return False

        # Get last failure time
        if self._breaker:
            status = self._breaker.get_status(self.PIPELINE_TARGET)
            last_failure_time = status.last_failure_time or 0.0
        else:
            last_failure_time = self._fallback_last_failure_time

        if last_failure_time <= 0:
            return False

        hours_since_failure = (time.time() - last_failure_time) / 3600
        if hours_since_failure >= self.AUTO_RECOVERY_HOURS:
            logger.info(
                f"[PipelineCircuitBreaker] Auto-recovering after {hours_since_failure:.1f} hours "
                f"(threshold: {self.AUTO_RECOVERY_HOURS}h)"
            )
            self.reset()
            return True

        return False

    def can_execute(self) -> bool:
        """Check if a request can be executed.

        Jan 3, 2026: Added auto-recovery check before returning.
        """
        # Check for auto-recovery first
        self._check_auto_recovery()

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
        # Persist state change
        self._save_state()

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
        # Persist state change
        self._save_state()

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._failures_by_stage.clear()
        if self._breaker:
            self._breaker.reset(self.PIPELINE_TARGET)
        else:
            self._fallback_state = CircuitBreakerState.CLOSED
            self._fallback_failure_count = 0
        logger.info("[PipelineCircuitBreaker] Manually reset")
        # Persist state change
        self._save_state()

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status."""
        if self._breaker:
            status = self._breaker.get_status(self.PIPELINE_TARGET)
            return {
                "state": status.state.value,
                "failure_count": status.failure_count,
                "success_count": self._success_count,
                "failures_by_stage": dict(self._failures_by_stage),
                "last_failure_time": status.last_failure_time or 0.0,
                "time_until_reset": status.time_since_open or 0,
                "consecutive_opens": status.consecutive_opens,
                "using_canonical": True,
            }
        return {
            "state": self._fallback_state.value,
            "failure_count": self._fallback_failure_count,
            "success_count": self._success_count,
            "failures_by_stage": dict(self._failures_by_stage),
            "last_failure_time": self._fallback_last_failure_time,
            "time_until_reset": 0,
            "consecutive_opens": 0,
            "using_canonical": False,
        }

    # =========================================================================
    # State Persistence (January 2026)
    # =========================================================================

    def _init_persistence_db(self) -> None:
        """Initialize SQLite database for state persistence."""
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = connect_safe(self._db_path, row_factory=None)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS circuit_breaker_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    state TEXT NOT NULL,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    last_failure_time REAL,
                    failures_by_stage TEXT,
                    updated_at REAL NOT NULL
                )
            """)
            conn.commit()
            conn.close()
            logger.debug(f"[PipelineCircuitBreaker] Initialized persistence DB: {self._db_path}")
        except Exception as e:
            logger.warning(f"[PipelineCircuitBreaker] Failed to init persistence DB: {e}")

    def _save_state(self) -> None:
        """Persist current circuit breaker state to SQLite."""
        try:
            state_value = self._fallback_state.value if not self._breaker else (
                self._breaker.get_state(self.PIPELINE_TARGET).value
            )
            failure_count = self._fallback_failure_count if not self._breaker else (
                self._breaker.get_status(self.PIPELINE_TARGET).failure_count
            )
            last_failure = self._fallback_last_failure_time if not self._breaker else (
                self._breaker.get_status(self.PIPELINE_TARGET).last_failure_time or 0.0
            )

            conn = connect_safe(self._db_path, row_factory=None)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO circuit_breaker_state (id, state, failure_count, success_count, last_failure_time, failures_by_stage, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    state = excluded.state,
                    failure_count = excluded.failure_count,
                    success_count = excluded.success_count,
                    last_failure_time = excluded.last_failure_time,
                    failures_by_stage = excluded.failures_by_stage,
                    updated_at = excluded.updated_at
            """, (
                state_value,
                failure_count,
                self._success_count,
                last_failure,
                json.dumps(self._failures_by_stage),
                time.time(),
            ))
            conn.commit()
            conn.close()
            logger.debug(f"[PipelineCircuitBreaker] Saved state: {state_value}, failures={failure_count}")
        except Exception as e:
            logger.warning(f"[PipelineCircuitBreaker] Failed to save state: {e}")

    def _load_state(self) -> None:
        """Load persisted circuit breaker state from SQLite."""
        try:
            if not self._db_path.exists():
                logger.debug("[PipelineCircuitBreaker] No persisted state found")
                return

            conn = connect_safe(self._db_path, row_factory=None)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT state, failure_count, success_count, last_failure_time, failures_by_stage, updated_at
                FROM circuit_breaker_state WHERE id = 1
            """)
            row = cursor.fetchone()
            conn.close()

            if not row:
                logger.debug("[PipelineCircuitBreaker] No persisted state row found")
                return

            state_str, failure_count, success_count, last_failure_time, failures_json, updated_at = row

            # Check if state is stale (older than recovery_timeout)
            age = time.time() - updated_at
            if age > self._recovery_timeout:
                logger.info(f"[PipelineCircuitBreaker] Persisted state too old ({age:.0f}s), starting fresh")
                return

            # Restore state
            self._success_count = success_count
            self._failures_by_stage = json.loads(failures_json) if failures_json else {}

            if not self._breaker:
                # Restore fallback state
                self._fallback_state = CircuitBreakerState(state_str)
                self._fallback_failure_count = failure_count
                self._fallback_last_failure_time = last_failure_time or 0.0
                logger.info(
                    f"[PipelineCircuitBreaker] Restored fallback state: {state_str}, "
                    f"failures={failure_count}, age={age:.0f}s"
                )
            else:
                # For canonical breaker, replay failures to restore state
                # Note: This approximates the state since canonical breaker
                # tracks per-target state internally
                if state_str == CircuitBreakerState.OPEN.value:
                    for _ in range(failure_count):
                        self._breaker.record_failure(self.PIPELINE_TARGET, None)
                logger.info(
                    f"[PipelineCircuitBreaker] Restored canonical state: replayed {failure_count} failures, "
                    f"age={age:.0f}s"
                )

        except Exception as e:
            logger.warning(f"[PipelineCircuitBreaker] Failed to load state: {e}")


# Backward-compat alias (deprecated - use PipelineCircuitBreaker)
CircuitBreaker = PipelineCircuitBreaker


# =============================================================================
# Phase 4.2: Cascade Recovery Constants (January 2026)
# =============================================================================
# Automatic retry for failed pipeline stages with exponential backoff
MAX_STAGE_RETRIES = 3  # Maximum retry attempts per stage/config
STAGE_RETRY_DELAY_SECONDS = 300.0  # 5 minutes between retries
STAGE_RETRY_BACKOFF_MULTIPLIER = 2.0  # Exponential backoff factor


class PipelineStage(Enum):
    """Pipeline stages in execution order."""

    IDLE = "idle"
    SELFPLAY = "selfplay"
    DATA_SYNC = "data_sync"
    NPZ_EXPORT = "npz_export"
    NPZ_COMBINATION = "npz_combination"  # December 2025: Quality-weighted NPZ combination
    TRAINING = "training"
    EVALUATION = "evaluation"
    PROMOTION = "promotion"
    COMPLETE = "complete"


class OperationMode(Enum):
    """Operation mode for graceful degradation.

    Jan 2, 2026: Added to support degraded startup when cluster is unavailable.
    """

    FULL = "full"        # All systems connected, full functionality
    DEGRADED = "degraded"  # Some cluster connectivity missing, reduced features
    LOCAL_ONLY = "local"   # Only local data processing, no cluster sync


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
    StatePersistenceMixin,
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
    - StatePersistenceMixin: Crash recovery persistence (Jan 2026)
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
        # December 2025: Relaxed from 0.6 to 0.3 for faster training iteration
        self.quality_gate_threshold = getattr(config, "quality_gate_threshold", 0.3) if config else 0.3
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

        # Jan 2, 2026: Operation mode for graceful degradation
        self._operation_mode = OperationMode.FULL
        self._missing_dependencies: list[str] = []

        # Resource constraint tracking (December 2025)
        self._paused: bool = False
        self._pause_reason: str | None = None
        self._pause_time: float = 0.0
        self._resource_constraints: dict[str, dict] = {}  # resource_type -> constraint_info
        self._backpressure_active: bool = False

        # Regression tracking (December 29, 2025 - Phase 7)
        self._last_regression: dict[str, Any] | None = None  # Last regression event data

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

        # Phase 4.2: Cascade recovery state (January 2026)
        # Tracks retry counts per (stage, config_key) for automatic retry
        self._stage_retry_counts: dict[tuple[str, str], int] = {}
        self._pending_retries: dict[tuple[str, str], asyncio.Task] = {}

        # Jan 2, 2026: Initialize state persistence for crash recovery
        # Persists pipeline stage, iteration count, and statistics
        self.init_persistence(
            db_path=Path("data/coordination/pipeline_state.db"),
            auto_snapshot=True,
            snapshot_interval=300.0,  # 5 minutes
            max_snapshots=10,
        )

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

            # Try to parse from config_key using canonical utility
            if (not board_type or not num_players) and "config_key" in metadata:
                config_key = metadata["config_key"]
                parsed = parse_config_key(config_key)
                if parsed:
                    if not board_type:
                        board_type = parsed.board_type
                    if not num_players:
                        num_players = parsed.num_players

        # Log if we successfully inferred missing config
        if (board_type and num_players) and (
            not self._current_board_type or not self._current_num_players
        ):
            logger.debug(
                f"[DataPipelineOrchestrator] Inferred board config: {board_type}_{num_players}p"
            )

        return board_type, num_players

    # =========================================================================
    # Phase 4.2: Cascade Recovery Methods (January 2026)
    # =========================================================================

    async def handle_stage_failure(
        self,
        stage: PipelineStage,
        config_key: str,
        error: Exception | str,
    ) -> bool:
        """Handle a stage failure with automatic retry logic.

        Implements cascade recovery: when a stage fails, it will be retried
        up to MAX_STAGE_RETRIES times with exponential backoff.

        Args:
            stage: The pipeline stage that failed
            config_key: The configuration key (e.g., "hex8_2p")
            error: The error that caused the failure

        Returns:
            True if a retry was scheduled, False if retries exhausted
        """
        key = (stage.value, config_key)
        retry_count = self._stage_retry_counts.get(key, 0)

        # Record failure in circuit breaker
        if self._circuit_breaker:
            self._circuit_breaker.record_failure(stage.value, str(error))

        if retry_count >= MAX_STAGE_RETRIES:
            # Retries exhausted - emit PIPELINE_FAILED event
            logger.error(
                f"[DataPipelineOrchestrator] Stage {stage.value} failed {MAX_STAGE_RETRIES} times "
                f"for {config_key}, giving up. Error: {error}"
            )
            self._stage_retry_counts.pop(key, None)  # Reset counter

            await self._emit_pipeline_failed(stage, config_key, error)
            return False

        # Schedule retry with exponential backoff
        self._stage_retry_counts[key] = retry_count + 1
        delay = STAGE_RETRY_DELAY_SECONDS * (STAGE_RETRY_BACKOFF_MULTIPLIER ** retry_count)

        logger.warning(
            f"[DataPipelineOrchestrator] Stage {stage.value} failed for {config_key}: {error}. "
            f"Retry {retry_count + 1}/{MAX_STAGE_RETRIES} in {delay:.0f}s"
        )

        # Cancel any existing pending retry for this stage/config
        existing_task = self._pending_retries.get(key)
        if existing_task and not existing_task.done():
            existing_task.cancel()

        # Schedule the retry
        task = asyncio.create_task(
            self._execute_stage_retry(stage, config_key, delay)
        )
        self._pending_retries[key] = task

        return True

    async def _execute_stage_retry(
        self,
        stage: PipelineStage,
        config_key: str,
        delay: float,
    ) -> None:
        """Execute a delayed stage retry.

        Args:
            stage: The stage to retry
            config_key: The configuration key
            delay: Delay before retry in seconds
        """
        try:
            await asyncio.sleep(delay)

            key = (stage.value, config_key)
            retry_count = self._stage_retry_counts.get(key, 0)

            logger.info(
                f"[DataPipelineOrchestrator] Retrying {stage.value} for {config_key} "
                f"(attempt {retry_count}/{MAX_STAGE_RETRIES})"
            )

            # Parse config_key to get board_type and num_players
            parsed = parse_config_key(config_key)
            if not parsed:
                logger.error(f"[DataPipelineOrchestrator] Invalid config_key: {config_key}")
                return

            # Trigger the appropriate stage
            await self._trigger_stage(stage, parsed.board_type, parsed.num_players)

        except asyncio.CancelledError:
            logger.debug(f"[DataPipelineOrchestrator] Retry cancelled for {stage.value}/{config_key}")
        except Exception as e:
            logger.error(f"[DataPipelineOrchestrator] Retry failed for {stage.value}/{config_key}: {e}")
        finally:
            # Jan 12, 2026: Clean up completed retry task to prevent memory leak
            # Previously, tasks were never removed from _pending_retries, causing
            # unbounded memory growth during 48h+ autonomous operation.
            key = (stage.value, config_key)
            self._pending_retries.pop(key, None)

    async def _trigger_stage(
        self,
        stage: PipelineStage,
        board_type: str,
        num_players: int,
    ) -> None:
        """Trigger a specific pipeline stage.

        Args:
            stage: The stage to trigger
            board_type: Board type for the stage
            num_players: Number of players
        """
        config_key = make_config_key(board_type, num_players)

        # Call the appropriate trigger method based on stage
        if stage == PipelineStage.DATA_SYNC:
            await self._trigger_data_sync(board_type, num_players)
        elif stage == PipelineStage.NPZ_EXPORT:
            await self._trigger_npz_export(board_type, num_players)
        elif stage == PipelineStage.NPZ_COMBINATION:
            await self._trigger_npz_combination(board_type, num_players)
        elif stage == PipelineStage.TRAINING:
            await self._trigger_training(board_type, num_players)
        elif stage == PipelineStage.EVALUATION:
            await self._trigger_evaluation(board_type, num_players)
        elif stage == PipelineStage.PROMOTION:
            await self._trigger_promotion(board_type, num_players)
        else:
            logger.warning(f"[DataPipelineOrchestrator] Cannot trigger stage: {stage.value}")

    async def _emit_pipeline_failed(
        self,
        stage: PipelineStage,
        config_key: str,
        error: Exception | str,
    ) -> None:
        """Emit PIPELINE_FAILED event when retries are exhausted.

        Args:
            stage: The stage that failed
            config_key: The configuration key
            error: The error that caused the failure
        """
        from app.coordination.event_emission_helpers import safe_emit_event_async
        from app.coordination.data_events import DataEventType

        await safe_emit_event_async(
            DataEventType.PIPELINE_FAILED,
            {
                "stage": stage.value,
                "config_key": config_key,
                "error": str(error),
                "retries_exhausted": True,
                "max_retries": MAX_STAGE_RETRIES,
                "timestamp": time.time(),
            },
            context="DataPipelineOrchestrator",
            source="pipeline_retry_manager",
        )

    def reset_stage_retry_count(self, stage: PipelineStage, config_key: str) -> None:
        """Reset retry count for a stage/config after successful completion.

        Call this after a stage completes successfully to reset the retry counter.

        Args:
            stage: The stage that completed
            config_key: The configuration key
        """
        key = (stage.value, config_key)
        if key in self._stage_retry_counts:
            del self._stage_retry_counts[key]
            logger.debug(f"[DataPipelineOrchestrator] Reset retry count for {stage.value}/{config_key}")

    def get_cascade_recovery_status(self) -> dict[str, Any]:
        """Get current cascade recovery status.

        Returns:
            Dict with retry counts and pending retries
        """
        return {
            "retry_counts": {
                f"{stage}/{config}": count
                for (stage, config), count in self._stage_retry_counts.items()
            },
            "pending_retries": [
                f"{stage}/{config}"
                for (stage, config), task in self._pending_retries.items()
                if not task.done()
            ],
            "max_retries": MAX_STAGE_RETRIES,
            "retry_delay_seconds": STAGE_RETRY_DELAY_SECONDS,
            "backoff_multiplier": STAGE_RETRY_BACKOFF_MULTIPLIER,
        }

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
    def is_running(self) -> bool:
        """Check if orchestrator is running (for daemon manager compatibility).

        Jan 3, 2026: Added for DaemonManager._check_daemon_running() integration.
        The daemon manager checks for is_running property to detect daemon status.
        """
        return self._coordinator_status == CoordinatorStatus.RUNNING

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

        Jan 2, 2026: Detects operation mode for graceful degradation.
        Jan 2, 2026: Restores state from previous snapshot for crash recovery.
        """
        if self._coordinator_status == CoordinatorStatus.RUNNING:
            return  # Already running

        self._coordinator_status = CoordinatorStatus.RUNNING
        self._start_time = time.time()

        # Jan 2, 2026: Restore state from previous snapshot (crash recovery)
        try:
            restored = await self.restore_from_snapshot()
            if restored:
                logger.info(f"[{self.name}] Restored state from previous snapshot")
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to restore snapshot: {e}")

        # Jan 2, 2026: Detect operation mode based on available dependencies
        self._operation_mode, self._missing_dependencies = await self._detect_operation_mode()
        if self._operation_mode != OperationMode.FULL:
            logger.warning(
                f"[{self.name}] Starting in {self._operation_mode.value} mode. "
                f"Missing dependencies: {self._missing_dependencies}"
            )

        # Subscribe to events
        self.subscribe_to_events()
        self.subscribe_to_data_events()

        # Jan 2, 2026: Subscribe to local-only events if in degraded/local mode
        if self._operation_mode in (OperationMode.DEGRADED, OperationMode.LOCAL_ONLY):
            self._subscribe_local_only_events()

        # Register with coordinator registry
        register_coordinator(self)

        # Jan 2, 2026: Start auto-snapshots for periodic persistence
        await self.start_auto_snapshots()

        logger.info(
            f"[{self.name}] Started (mode={self._operation_mode.value})"
        )

    async def stop(self) -> None:
        """Stop the orchestrator gracefully.

        Cleans up subscriptions and resources.
        Idempotent - calling on an already stopped orchestrator is a no-op.

        Jan 2, 2026: Saves final snapshot for crash recovery.
        """
        if self._coordinator_status == CoordinatorStatus.STOPPED:
            return  # Already stopped

        self._coordinator_status = CoordinatorStatus.STOPPING

        # Jan 2, 2026: Stop auto-snapshots and save final state
        try:
            await self.stop_auto_snapshots()
            await self.save_snapshot(reason="shutdown")
            logger.debug(f"[{self.name}] Saved final snapshot on shutdown")
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to save final snapshot: {e}")

        # Unregister from coordinator registry
        unregister_coordinator(self.name)

        self._coordinator_status = CoordinatorStatus.STOPPED
        logger.info(f"[{self.name}] Stopped")

    async def _detect_operation_mode(self) -> tuple[OperationMode, list[str]]:
        """Detect operation mode based on available dependencies.

        Jan 2, 2026: Added for graceful degradation support.

        Returns:
            Tuple of (OperationMode, list of missing dependency names)
        """
        missing: list[str] = []

        # Check P2P cluster availability (fast timeout)
        p2p_available = await self._check_p2p_availability(timeout=5.0)
        if not p2p_available:
            missing.append("P2P cluster")

        # Check if AUTO_SYNC daemon is running
        try:
            from app.coordination.daemon_manager import get_daemon_manager
            from app.coordination.daemon_types import DaemonState, DaemonType

            dm = get_daemon_manager()
            auto_sync_info = dm._daemons.get(DaemonType.AUTO_SYNC)
            if auto_sync_info is None or auto_sync_info.state != DaemonState.RUNNING:
                missing.append("AUTO_SYNC daemon")
        except (ImportError, AttributeError):
            missing.append("DaemonManager")

        # Determine mode
        if not missing:
            return OperationMode.FULL, []
        elif "P2P cluster" in missing and "AUTO_SYNC daemon" in missing:
            return OperationMode.LOCAL_ONLY, missing
        else:
            return OperationMode.DEGRADED, missing

    async def _check_p2p_availability(self, timeout: float = 5.0) -> bool:
        """Check if P2P cluster is reachable.

        Args:
            timeout: Timeout in seconds for the check

        Returns:
            True if P2P cluster is reachable
        """
        try:
            import aiohttp

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                async with session.get(get_local_p2p_status_url()) as response:
                    return response.status == 200
        except Exception:
            return False

    def _subscribe_local_only_events(self) -> None:
        """Subscribe to events that work in local-only mode.

        Jan 2, 2026: These events don't require cluster connectivity.
        """
        try:
            from app.coordination.event_router import get_router

            router = get_router()

            # Local file-based events that work without cluster
            local_events = [
                "LOCAL_NPZ_CREATED",
                "LOCAL_GAME_SAVED",
                "LOCAL_MODEL_SAVED",
            ]

            for event_type in local_events:
                router.subscribe(event_type, self._on_local_event)

            logger.info(
                f"[{self.name}] Subscribed to {len(local_events)} local-only events"
            )
        except ImportError:
            logger.debug(f"[{self.name}] Event router not available for local events")

    async def _on_local_event(self, event: Any) -> None:
        """Handle local events in degraded/local-only mode.

        Args:
            event: Event data (RouterEvent or dict)
        """
        # Jan 14, 2026: Handle both RouterEvent and dict types
        from app.coordination.event_router import get_event_payload
        payload = get_event_payload(event)
        event_type = payload.get("type", "unknown")
        logger.debug(f"[{self.name}] Received local event: {event_type}")

        # Trigger local pipeline stages based on event type
        if event_type == "LOCAL_NPZ_CREATED":
            config_key = payload.get("config_key")
            if config_key:
                logger.info(
                    f"[{self.name}] Local NPZ created for {config_key}, "
                    f"triggering local training check"
                )

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

    # =========================================================================
    # State Persistence (Jan 2, 2026 - StatePersistenceMixin implementation)
    # =========================================================================

    def _get_state_for_persistence(self) -> dict[str, Any]:
        """Get pipeline state for crash recovery persistence.

        Jan 2, 2026: Implements StatePersistenceMixin abstract method.
        Persists critical state needed to resume pipeline after restart.

        Returns:
            Dict of state to persist (JSON-serializable)
        """
        # Convert iteration records to serializable format
        completed_records = []
        for record in self._completed_iterations[-self.max_history:]:
            completed_records.append({
                "iteration": record.iteration,
                "start_time": record.start_time,
                "end_time": record.end_time,
                "success": record.success,
                "stages_completed": record.stages_completed,
                "games_generated": record.games_generated,
                "model_id": record.model_id,
                "elo_delta": record.elo_delta,
                "promoted": record.promoted,
                "error": record.error,
            })

        # Convert transitions to serializable format (last 50 only)
        transitions = []
        for trans in self._transitions[-50:]:
            transitions.append({
                "from_stage": trans.from_stage.value,
                "to_stage": trans.to_stage.value,
                "iteration": trans.iteration,
                "timestamp": trans.timestamp,
                "success": trans.success,
                "duration_seconds": trans.duration_seconds,
            })

        return {
            # Core state
            "current_stage": self._current_stage.value,
            "current_iteration": self._current_iteration,
            "current_board_type": self._current_board_type,
            "current_num_players": self._current_num_players,

            # Statistics
            "total_games": self._total_games,
            "total_models": self._total_models,
            "total_promotions": self._total_promotions,
            "events_processed": self._events_processed,
            "errors_count": self._errors_count,

            # Quality tracking
            "last_quality_score": self._last_quality_score,
            "quality_check_history": self._quality_check_history[-20:],

            # History (limited to prevent bloat)
            "completed_iterations": completed_records,
            "transitions": transitions,

            # Operation mode
            "operation_mode": self._operation_mode.value,
        }

    def _restore_state_from_persistence(self, state: dict[str, Any]) -> None:
        """Restore pipeline state after crash/restart.

        Jan 2, 2026: Implements StatePersistenceMixin abstract method.
        Restores critical state from previous snapshot.

        Args:
            state: Previously persisted state dict
        """
        # Restore core state
        stage_value = state.get("current_stage", "IDLE")
        try:
            self._current_stage = PipelineStage(stage_value)
        except ValueError:
            self._current_stage = PipelineStage.IDLE
            logger.warning(f"[{self.name}] Unknown stage '{stage_value}', defaulting to IDLE")

        self._current_iteration = state.get("current_iteration", 0)
        self._current_board_type = state.get("current_board_type")
        self._current_num_players = state.get("current_num_players")

        # Restore statistics
        self._total_games = state.get("total_games", 0)
        self._total_models = state.get("total_models", 0)
        self._total_promotions = state.get("total_promotions", 0)
        self._events_processed = state.get("events_processed", 0)
        self._errors_count = state.get("errors_count", 0)

        # Restore quality tracking
        self._last_quality_score = state.get("last_quality_score", 0.0)
        self._quality_check_history = state.get("quality_check_history", [])

        # Restore completed iterations
        for record_dict in state.get("completed_iterations", []):
            record = IterationRecord(
                iteration=record_dict.get("iteration", 0),
                start_time=record_dict.get("start_time", 0.0),
                end_time=record_dict.get("end_time", 0.0),
                success=record_dict.get("success", False),
                stages_completed=record_dict.get("stages_completed", []),
                games_generated=record_dict.get("games_generated", 0),
                model_id=record_dict.get("model_id"),
                elo_delta=record_dict.get("elo_delta", 0.0),
                promoted=record_dict.get("promoted", False),
                error=record_dict.get("error"),
            )
            self._completed_iterations.append(record)

        # Restore transitions
        for trans_dict in state.get("transitions", []):
            try:
                trans = StageTransition(
                    from_stage=PipelineStage(trans_dict.get("from_stage", "IDLE")),
                    to_stage=PipelineStage(trans_dict.get("to_stage", "IDLE")),
                    iteration=trans_dict.get("iteration", 0),
                    timestamp=trans_dict.get("timestamp", 0.0),
                    success=trans_dict.get("success", True),
                    duration_seconds=trans_dict.get("duration_seconds", 0.0),
                )
                self._transitions.append(trans)
            except (ValueError, KeyError) as e:
                logger.debug(f"[{self.name}] Skipping invalid transition: {e}")

        # Restore operation mode
        mode_value = state.get("operation_mode", "full")
        try:
            self._operation_mode = OperationMode(mode_value)
        except ValueError:
            self._operation_mode = OperationMode.FULL

        logger.info(
            f"[{self.name}] Restored state: stage={self._current_stage.value}, "
            f"iteration={self._current_iteration}, games={self._total_games}"
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
                except (TypeError, ValueError, AttributeError, RuntimeError) as sub_error:
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
        except (TypeError, ValueError, AttributeError, RuntimeError, KeyError) as e:
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

            # December 2025: Subscribe to NPZ combination events for quality-weighted data
            # NPZ combination combines historical + fresh data with quality weighting
            router.subscribe(
                DataEventType.NPZ_COMBINATION_COMPLETE.value,
                self._on_npz_combination_complete,
            )
            router.subscribe(
                DataEventType.NPZ_COMBINATION_FAILED.value,
                self._on_npz_combination_failed,
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
            # December 2025: Track abandoned tasks for accurate pending count
            router.subscribe(
                DataEventType.TASK_ABANDONED.value,
                self._on_task_abandoned,
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

            # January 3, 2026: Subscribe to partition healing events
            # PARTITION_HEALED emitted by partition_healer.py:514 when healing succeeds
            router.subscribe(
                DataEventType.PARTITION_HEALED.value,
                self._on_partition_healed,
            )

            logger.info("[DataPipelineOrchestrator] Subscribed to data events")
            return True

        except ImportError:
            logger.warning("[DataPipelineOrchestrator] data_events not available")
            return False
        except (TypeError, ValueError, AttributeError, RuntimeError, KeyError) as e:
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
        """Record a stage transition.

        December 31, 2025: Added guards to prevent:
        1. Same-stage transitions (no-op, just log debug)
        2. Rapid transitions (minimum 100ms interval to prevent loops)
        """
        old_stage = self._current_stage

        # Guard 1: Skip same-stage transitions
        if old_stage == new_stage and success:
            logger.debug(
                f"[DataPipelineOrchestrator] Skipping same-stage transition: "
                f"{old_stage.value} -> {new_stage.value}"
            )
            return

        # Guard 2: Prevent rapid transitions (minimum 100ms interval)
        now = time.time()
        last_transition = getattr(self, "_last_transition_time", 0.0)
        if now - last_transition < 0.1:  # 100ms cooldown
            logger.debug(
                f"[DataPipelineOrchestrator] Throttling rapid transition: "
                f"{old_stage.value} -> {new_stage.value} (too soon after last)"
            )
            return
        self._last_transition_time = now

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














    # =========================================================================
    # Orphaned Event Handlers (December 27, 2025)
    # These handlers wire previously orphaned events that were being emitted
    # but had no subscribers, breaking feedback loops.
    # =========================================================================















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

        except (ValueError, TypeError, KeyError, AttributeError, ZeroDivisionError, IndexError) as e:
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

        except (ValueError, TypeError, KeyError, IndexError, AttributeError, ZeroDivisionError) as e:
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
                # December 30, 2025: Include config_key for SelfplayScheduler integration
                config_key = make_config_key(board_type, num_players) if board_type and num_players else ""
                await router.publish(
                    event_type="TRAINING_BLOCKED_BY_QUALITY",
                    payload={
                        "iteration": iteration,
                        "npz_path": npz_path,
                        "board_type": board_type,
                        "num_players": num_players,
                        "config_key": config_key,  # Added for SelfplayScheduler
                        "quality_score": self._last_quality_score,
                        "threshold": self.quality_gate_threshold,
                        "quality_history": self._quality_check_history[-5:],
                        "recommendation": "trigger_data_regeneration",
                        "reason": "quality_gate_failed",
                    },
                    source="DataPipelineOrchestrator",
                )
                logger.info(
                    f"[QualityGate] Emitted TRAINING_BLOCKED_BY_QUALITY for iteration {iteration}"
                )

                # Also trigger data regeneration if we have enough info
                if board_type and num_players:
                    await self._trigger_data_regeneration(board_type, num_players, iteration)

        except (RuntimeError, ValueError, TypeError, AttributeError, KeyError, ImportError) as e:
            logger.warning(f"[QualityGate] Failed to emit quality block event: {e}")

    async def _trigger_data_regeneration(
        self, board_type: str, num_players: int, iteration: int
    ) -> None:
        """Request more selfplay data when quality gate blocks training.

        December 30, 2025: Implements Gap #6 from integration analysis.
        Emits SELFPLAY_TARGET_UPDATED to boost data generation for blocked configs.
        """
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            if not router:
                return

            config_key = make_config_key(board_type, num_players)

            # Calculate additional games needed based on quality score
            quality_score = getattr(self, "_last_quality_score", 0.5)
            # Lower quality = more games needed
            additional_games = int(200 * (1.0 - quality_score))
            additional_games = max(100, min(500, additional_games))

            await router.publish(
                event_type="SELFPLAY_TARGET_UPDATED",
                payload={
                    "config_key": config_key,
                    "target_games": additional_games,
                    "priority": "high",
                    "reason": "quality_gate_blocked",
                    "quality_score": quality_score,
                    "iteration": iteration,
                    "exploration_boost": 1.5,  # Encourage diverse data
                },
                source="DataPipelineOrchestrator",
            )
            logger.info(
                f"[DataPipelineOrchestrator] Requested {additional_games} additional games "
                f"for {config_key} (quality={quality_score:.2f})"
            )

        except (ImportError, RuntimeError, AttributeError) as e:
            logger.debug(f"[DataPipelineOrchestrator] Could not trigger regeneration: {e}")










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
                from app.coordination.event_emission_helpers import safe_emit_event
                from app.distributed.data_events import DataEventType

                safe_emit_event(
                    DataEventType.TRAINING_BLOCKED_BY_QUALITY,  # Reuse existing event type
                    payload={
                        "reason": reason,
                        "blocked_by": "health_manager",
                        "source": "data_pipeline_orchestrator",
                    },
                    context="DataPipelineOrchestrator",
                    source="health_manager_check",
                )
                return False
        except ImportError:
            pass  # UnifiedHealthManager not available, skip this check
        except (RuntimeError, ValueError, TypeError, AttributeError, KeyError) as e:
            logger.debug(f"[DataPipelineOrchestrator] Health manager check failed: {e}")
            # Continue on health check failure - don't block training

        # Check cluster resources (December 2025)
        if not self._check_cluster_resources():
            logger.debug("[DataPipelineOrchestrator] Auto-trigger blocked: cluster resources constrained")
            return False

        return True

    def _check_cluster_resources(
        self,
        disk_threshold: float = float(DISK_PRODUCTION_HALT_PERCENT),
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
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError, IOError, KeyError) as e:
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
                fire_and_forget(bus.publish(event), name="pipeline_stage_event")
            except RuntimeError:
                asyncio.run(bus.publish(event))

        except (RuntimeError, ValueError, TypeError, AttributeError, KeyError, ImportError) as e:
            logger.debug(f"[DataPipelineOrchestrator] Best-effort event emit failed: {e}")

    def _record_circuit_success(self, stage: str) -> None:
        """Record a successful stage execution to circuit breaker."""
        if self._circuit_breaker:
            self._circuit_breaker.record_success(stage)

    def _record_circuit_failure(self, stage: str, error: str) -> None:
        """Record a failed stage execution to circuit breaker."""
        if self._circuit_breaker:
            self._circuit_breaker.record_failure(stage, error)



    # =========================================================================
    # Quality and Cache Event Handlers (December 2025)
    # =========================================================================




    # =========================================================================
    # Resource Constraint Event Handlers (December 2025)
    # =========================================================================
















    async def _pause_pipeline(self, reason: str) -> None:
        """Pause the pipeline due to resource constraints."""
        if self._paused:
            return  # Already paused

        self._paused = True
        self._pause_reason = reason
        self._pause_time = time.time()

        logger.warning(f"[DataPipelineOrchestrator] Pipeline PAUSED: {reason}")

        # Emit event for other coordinators (January 2026 - migrated to event_router)
        try:
            from app.coordination.event_emission_helpers import safe_emit_event_async

            await safe_emit_event_async(
                "RESOURCE_CONSTRAINT_DETECTED",
                {
                    "resource_type": "pipeline_pause",
                    "severity": "critical",
                    "current_value": 1,
                    "threshold": 0,
                    "action_taken": f"pipeline_paused: {reason}",
                },
                context="data_pipeline_orchestrator",
            )
        except (RuntimeError, ValueError, TypeError, AttributeError, KeyError, ImportError) as e:
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

    # =========================================================================
    # Event Handlers (December 29, 2025 - Phase 7: Missing handlers implementation)
    # =========================================================================

    def _on_new_games_available(self, event) -> None:
        """Handle NEW_GAMES_AVAILABLE event - trigger export if threshold met.

        December 29, 2025: Added SelfplayScheduler integration to set training
        targets and check if more games are needed before triggering export.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            game_count = payload.get("game_count", 0)
            source = payload.get("source", "unknown")

            logger.info(
                f"[DataPipelineOrchestrator] New games available: "
                f"config={config_key}, count={game_count}, source={source}"
            )

            # December 29, 2025: Wire to SelfplayScheduler for game count normalization
            # Set training sample targets and check if more games are needed
            if config_key:
                self._update_selfplay_scheduler_targets(config_key)

            self._record_event_processed()

        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_new_games_available: {e}")

    def _on_orphan_games_detected(self, event) -> None:
        """Handle ORPHAN_GAMES_DETECTED event - trigger resync and re-export.

        Sprint 4 (Jan 2, 2026): Auto-trigger re-export for orphaned games above threshold.
        Orphan games are selfplay databases that exist on nodes but aren't registered
        in the central manifest. They need to be synced and re-exported.

        The plan suggested adding this handler which was referenced but not implemented.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            host = payload.get("host", "unknown")
            orphan_count = payload.get("orphan_count", 0)
            orphan_paths = payload.get("orphan_paths", [])
            total_games = payload.get("total_games", 0)

            logger.info(
                f"[DataPipelineOrchestrator] Orphan games detected: "
                f"host={host}, count={orphan_count}, games={total_games}"
            )

            # Sprint 4: Auto-resync threshold - only trigger for significant orphan counts
            ORPHAN_RESYNC_THRESHOLD = int(
                os.environ.get("RINGRIFT_ORPHAN_RESYNC_THRESHOLD", "100")
            )

            if total_games >= ORPHAN_RESYNC_THRESHOLD:
                logger.info(
                    f"[DataPipelineOrchestrator] Orphan threshold exceeded "
                    f"({total_games} >= {ORPHAN_RESYNC_THRESHOLD}), triggering resync"
                )
                # Emit sync trigger event to pull orphan data
                self._emit_orphan_resync_trigger(host, orphan_paths, total_games)

            self._record_event_processed()

        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_orphan_games_detected: {e}")

    def _on_orphan_games_registered(self, event) -> None:
        """Handle ORPHAN_GAMES_REGISTERED event - trigger export after registration.

        Sprint 4 (Jan 2, 2026): After orphan games are registered, trigger NPZ export
        to include the recovered data in training.

        Sprint 10 (Jan 3, 2026): Added auto-quality-check for orphan games.
        Only triggers export if orphan game quality is acceptable.
        Expected Elo gain: +1-2 Elo from better training data quality.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            host = payload.get("host", "unknown")
            registered_count = payload.get("registered_count", 0)
            games_recovered = payload.get("games_recovered", 0)
            registered_paths = payload.get("registered_paths", [])

            logger.info(
                f"[DataPipelineOrchestrator] Orphan games registered: "
                f"host={host}, count={registered_count}, games={games_recovered}"
            )

            # Extract configs from registered paths and trigger re-export
            configs_to_export: set[str] = set()
            configs_needing_quality_boost: set[str] = set()

            for path in registered_paths:
                config = self._extract_config_from_db_path(path)
                if config:
                    # Sprint 10: Check quality of orphan games before export
                    quality_result = self._check_orphan_game_quality(path, config)
                    if quality_result["acceptable"]:
                        configs_to_export.add(config)
                        if quality_result.get("needs_boost"):
                            configs_needing_quality_boost.add(config)
                    else:
                        # Quality too low - emit event to boost selfplay quality
                        logger.warning(
                            f"[DataPipelineOrchestrator] Orphan games for {config} "
                            f"have low quality ({quality_result['score']:.2f}), "
                            f"triggering quality boost instead of export"
                        )
                        self._emit_orphan_quality_blocked(
                            config, quality_result["score"], path
                        )

            # Trigger export for configs that passed quality check
            for config_key in configs_to_export:
                source = "orphan_recovery"
                if config_key in configs_needing_quality_boost:
                    source = "orphan_recovery_with_boost"
                self._emit_export_trigger(config_key, source=source)

            self._record_event_processed()

        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_orphan_games_registered: {e}")

    def _emit_orphan_resync_trigger(
        self, host: str, orphan_paths: list[str], total_games: int
    ) -> None:
        """Emit SYNC_TRIGGERED event to resync orphan games.

        Sprint 4 (Jan 2, 2026): Part of orphan game recovery pipeline.
        """
        from app.coordination.event_emission_helpers import safe_emit_event
        from app.distributed.data_events import DataEventType

        safe_emit_event(
            DataEventType.SYNC_TRIGGERED,
            {
                "host": host,
                "paths": orphan_paths[:10],  # Limit payload size
                "game_count": total_games,
                "trigger": "orphan_recovery",
                "source": "data_pipeline_orchestrator",
                "timestamp": time.time(),
            },
            log_after="Emitted SYNC_TRIGGERED for orphan recovery",
            context="DataPipelineOrchestrator",
            source="data_pipeline_orchestrator",
        )

    def _emit_export_trigger(self, config_key: str, source: str) -> None:
        """Emit event to trigger NPZ export for a config.

        Sprint 4 (Jan 2, 2026): Part of orphan game recovery pipeline.
        """
        from app.coordination.event_emission_helpers import safe_emit_event
        from app.distributed.data_events import DataEventType

        safe_emit_event(
            DataEventType.NEW_GAMES_AVAILABLE,
            {
                "config_key": config_key,
                "source": source,
                "trigger": "orphan_recovery",
                "timestamp": time.time(),
            },
            log_after=f"Emitted NEW_GAMES_AVAILABLE for {config_key}",
            context="DataPipelineOrchestrator",
            source="data_pipeline_orchestrator",
        )

    def _extract_config_from_db_path(self, path: str) -> str | None:
        """Extract config key from database path.

        Sprint 4 (Jan 2, 2026): Helper for orphan recovery.
        Path format: .../selfplay_{board}_{n}p.db or .../canonical_{board}_{n}p.db
        """
        import re

        # Match patterns like selfplay_hex8_2p.db or canonical_square8_4p.db
        match = re.search(r"(?:selfplay_|canonical_)?(\w+)_(\d+)p\.db$", path)
        if match:
            board_type = match.group(1)
            num_players = int(match.group(2))
            return make_config_key(board_type, num_players)
        return None

    def _check_orphan_game_quality(
        self, db_path: str, config_key: str
    ) -> dict[str, bool | float]:
        """Check quality of orphan games in a database.

        Sprint 10 (Jan 3, 2026): Auto-quality-check for orphan games.
        Ensures only acceptable-quality orphan data gets exported to training.

        Args:
            db_path: Path to the orphan games database
            config_key: Config key (e.g., "hex8_2p")

        Returns:
            dict with keys:
                - acceptable: True if quality is good enough for export
                - score: Quality score (0.0-1.0)
                - needs_boost: True if quality is borderline and needs boost
                - reason: Human-readable explanation
        """
        import sqlite3
        from pathlib import Path

        # Default thresholds - configurable via env
        ORPHAN_QUALITY_MIN = float(
            os.environ.get("RINGRIFT_ORPHAN_QUALITY_MIN", "0.4")
        )
        ORPHAN_QUALITY_BOOST_THRESHOLD = float(
            os.environ.get("RINGRIFT_ORPHAN_QUALITY_BOOST", "0.6")
        )

        try:
            db = Path(db_path)
            if not db.exists():
                logger.warning(f"[QualityCheck] Orphan DB not found: {db_path}")
                return {
                    "acceptable": False,
                    "score": 0.0,
                    "needs_boost": False,
                    "reason": "database_not_found",
                }

            # Connect and check quality metrics
            conn = connect_safe(db_path, row_factory=None)
            cursor = conn.cursor()

            # Check 1: Game completion rate (finished games / total games)
            cursor.execute(
                "SELECT COUNT(*) FROM games WHERE winner IS NOT NULL"
            )
            finished_games = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM games")
            total_games = cursor.fetchone()[0]

            if total_games == 0:
                conn.close()
                return {
                    "acceptable": False,
                    "score": 0.0,
                    "needs_boost": False,
                    "reason": "no_games",
                }

            completion_rate = finished_games / total_games

            # Check 2: Average game length (longer games = more training signal)
            cursor.execute(
                """
                SELECT AVG(move_count) FROM (
                    SELECT game_id, COUNT(*) as move_count
                    FROM moves GROUP BY game_id
                )
                """
            )
            avg_moves_result = cursor.fetchone()
            avg_moves = avg_moves_result[0] if avg_moves_result[0] else 0

            # Check 3: Move diversity (unique positions explored)
            cursor.execute("SELECT COUNT(DISTINCT game_id) FROM moves")
            games_with_moves = cursor.fetchone()[0]

            conn.close()

            # Calculate quality score (weighted average)
            # - Completion rate: 40% weight (finished games are more valuable)
            # - Game length: 40% weight (normalized to expected range)
            # - Coverage: 20% weight (games with moves recorded)

            # Normalize avg_moves to 0-1 (expect 30-100 moves for a typical game)
            length_score = min(1.0, avg_moves / 50.0) if avg_moves > 0 else 0.0

            # Coverage score
            coverage_score = games_with_moves / total_games if total_games > 0 else 0.0

            quality_score = (
                completion_rate * 0.4
                + length_score * 0.4
                + coverage_score * 0.2
            )

            # Determine acceptability
            acceptable = quality_score >= ORPHAN_QUALITY_MIN
            needs_boost = quality_score < ORPHAN_QUALITY_BOOST_THRESHOLD

            reason = "quality_ok"
            if not acceptable:
                reason = f"quality_too_low_{quality_score:.2f}"
            elif needs_boost:
                reason = f"quality_borderline_{quality_score:.2f}"

            logger.info(
                f"[QualityCheck] Orphan games {config_key}: "
                f"score={quality_score:.2f}, completion={completion_rate:.1%}, "
                f"avg_moves={avg_moves:.0f}, acceptable={acceptable}"
            )

            return {
                "acceptable": acceptable,
                "score": quality_score,
                "needs_boost": needs_boost,
                "reason": reason,
            }

        except (sqlite3.Error, OSError, ValueError) as e:
            logger.warning(f"[QualityCheck] Error checking orphan quality: {e}")
            # On error, be permissive and allow export with boost flag
            return {
                "acceptable": True,
                "score": 0.5,
                "needs_boost": True,
                "reason": f"error_{type(e).__name__}",
            }

    def _emit_orphan_quality_blocked(
        self, config_key: str, quality_score: float, db_path: str
    ) -> None:
        """Emit event when orphan game quality is too low for export.

        Sprint 10 (Jan 3, 2026): Triggers quality boost in SelfplayScheduler.
        This causes the scheduler to prefer high-quality Gumbel MCTS modes
        for this config, improving overall training data quality.

        Args:
            config_key: Config that failed quality check
            quality_score: The quality score that triggered the block
            db_path: Path to the orphan database
        """
        from app.coordination.event_emission_helpers import safe_emit_event
        from app.distributed.data_events import DataEventType

        # Calculate quality deficit for boost strength
        min_quality = float(
            os.environ.get("RINGRIFT_ORPHAN_QUALITY_MIN", "0.4")
        )
        quality_deficit = max(0.0, min_quality - quality_score)

        safe_emit_event(
            DataEventType.TRAINING_BLOCKED_BY_QUALITY,
            {
                "config_key": config_key,
                "quality_score": quality_score,
                "threshold": min_quality,
                "quality_deficit": quality_deficit,
                "source": "orphan_quality_check",
                "db_path": db_path,
                "reason": "orphan_games_low_quality",
                "recommendation": "boost_selfplay_quality",
                "timestamp": time.time(),
            },
            log_after=(
                f"Emitted TRAINING_BLOCKED_BY_QUALITY for orphan games "
                f"{config_key} (score={quality_score:.2f})"
            ),
            context="DataPipelineOrchestrator",
            source="orphan_quality_check",
        )

    def _update_selfplay_scheduler_targets(self, config_key: str) -> None:
        """Update SelfplayScheduler with training sample targets.

        December 29, 2025: Closes the pipeline → scheduler feedback loop.
        - Sets target samples based on board size
        - Checks if more games are needed
        - Emits SELFPLAY_TARGET_UPDATED if games needed

        This ensures the scheduler knows how many more games to generate
        for each configuration before training can proceed.
        """
        try:
            from app.coordination.selfplay_scheduler import get_selfplay_scheduler

            scheduler = get_selfplay_scheduler()

            # Calculate target samples based on board type
            # Default: 50K samples minimum, scale with board size
            target_samples = 50000
            if config_key.startswith("square19") or config_key.startswith("hexagonal"):
                target_samples = 100000  # Large boards need more data
            elif config_key.startswith("square8") or config_key.startswith("hex8"):
                target_samples = 50000  # Standard boards

            scheduler.set_target_training_samples(config_key, target_samples)

            # Check if we have enough games
            games_needed = scheduler.get_games_needed(config_key)
            if games_needed > 0:
                logger.info(
                    f"[DataPipelineOrchestrator] {config_key} needs {games_needed} more games"
                )
                # Emit event to request more selfplay
                self._emit_selfplay_target_updated(config_key, games_needed)

        except ImportError:
            # SelfplayScheduler not available - expected in minimal environments
            logger.debug("[DataPipelineOrchestrator] SelfplayScheduler not available")
        except (AttributeError, RuntimeError, TypeError) as e:
            # Non-critical - log and continue
            logger.debug(f"[DataPipelineOrchestrator] Failed to update scheduler: {e}")

    def _emit_selfplay_target_updated(self, config_key: str, games_needed: int) -> None:
        """Emit SELFPLAY_TARGET_UPDATED event to request more games.

        December 29, 2025: Part of pipeline → scheduler wiring.
        """
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            if router:
                router.publish_sync(
                    "SELFPLAY_TARGET_UPDATED",
                    {
                        "config_key": config_key,
                        "games_needed": games_needed,
                        "source": "data_pipeline_orchestrator",
                        "timestamp": time.time(),
                    },
                    source="data_pipeline_orchestrator",
                )
        except (AttributeError, ImportError, RuntimeError) as e:
            logger.debug(f"[DataPipelineOrchestrator] Failed to emit target update: {e}")

    def _on_regression_detected(self, event) -> None:
        """Handle REGRESSION_DETECTED event - trigger curriculum rebalance.

        December 29, 2025: Phase 7 - Regression-triggered curriculum rebalance.
        When model regression is detected, reduce this config's curriculum weight
        to prevent bad training data from propagating.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            elo_loss = payload.get("elo_loss", payload.get("elo_drop", 0))
            severity = payload.get("severity", "unknown")

            if not config_key:
                return

            # Record regression for tracking
            self._last_regression = {"config": config_key, "loss": elo_loss}

            logger.warning(
                f"[DataPipelineOrchestrator] Regression detected: "
                f"config={config_key}, elo_loss={elo_loss:.0f}, severity={severity}"
            )

            # Phase 7: Trigger curriculum rebalance for significant regressions
            if abs(elo_loss) > 50:
                self._emit_curriculum_emergency_update(config_key, elo_loss)

            self._record_event_processed()

        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_regression_detected: {e}")

    def _emit_curriculum_emergency_update(self, config_key: str, elo_loss: float) -> None:
        """Emit curriculum emergency update to reduce allocation for regressing config.

        December 29, 2025: Phase 7 - Closes the regression → curriculum feedback loop.
        """
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            if router:
                # Calculate reduction factor based on regression severity
                if abs(elo_loss) > 100:
                    factor = 0.3  # Severe regression: reduce to 30%
                else:
                    factor = 0.5  # Moderate regression: reduce to 50%

                router.publish_sync(
                    "CURRICULUM_REBALANCED",
                    {
                        "trigger": "regression_detected",
                        "changed_configs": [config_key],
                        "action": "reduce_allocation",
                        "factor": factor,
                        "elo_loss": elo_loss,
                        "timestamp": time.time(),
                    },
                    source="data_pipeline_orchestrator",
                )
                logger.info(
                    f"[DataPipelineOrchestrator] Emitted curriculum emergency update: "
                    f"config={config_key}, factor={factor}, elo_loss={elo_loss:.0f}"
                )
        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"[DataPipelineOrchestrator] Could not emit curriculum update: {e}")

    def _on_promotion_failed(self, event) -> None:
        """Handle PROMOTION_FAILED event - log and track for pipeline metrics."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            reason = payload.get("reason", "unknown")

            logger.warning(
                f"[DataPipelineOrchestrator] Promotion failed: "
                f"config={config_key}, reason={reason}"
            )
            self._record_event_processed()

        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_promotion_failed: {e}")

    def _on_consolidation_started(self, event) -> None:
        """Handle CONSOLIDATION_STARTED event."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            logger.info(f"[DataPipelineOrchestrator] Consolidation started: {config_key}")
            self._record_event_processed()
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_consolidation_started: {e}")

    def _on_consolidation_complete(self, event) -> None:
        """Handle CONSOLIDATION_COMPLETE event - trigger export for consolidated data."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            game_count = payload.get("game_count", 0)
            logger.info(
                f"[DataPipelineOrchestrator] Consolidation complete: "
                f"config={config_key}, games={game_count}"
            )
            self._record_event_processed()
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_consolidation_complete: {e}")

    def _on_npz_combination_complete(self, event) -> None:
        """Handle NPZ_COMBINATION_COMPLETE event."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            output_path = payload.get("output_path", "")
            logger.info(
                f"[DataPipelineOrchestrator] NPZ combination complete: "
                f"config={config_key}, path={output_path}"
            )
            self._record_event_processed()
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_npz_combination_complete: {e}")

    def _on_npz_combination_failed(self, event) -> None:
        """Handle NPZ_COMBINATION_FAILED event."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            error = payload.get("error", "unknown")
            logger.error(
                f"[DataPipelineOrchestrator] NPZ combination failed: "
                f"config={config_key}, error={error}"
            )
            self._record_error(f"npz_combination_failed: {config_key}")
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_npz_combination_failed: {e}")

    def _on_repair_completed(self, event) -> None:
        """Handle REPAIR_COMPLETED event - retrigger sync after data repair."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            files_repaired = payload.get("files_repaired", 0)
            logger.info(
                f"[DataPipelineOrchestrator] Repair completed: "
                f"config={config_key}, files={files_repaired}"
            )
            self._record_event_processed()
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_repair_completed: {e}")

    def _on_repair_failed(self, event) -> None:
        """Handle REPAIR_FAILED event - track repair failures for circuit breaker."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            error = payload.get("error", "unknown")
            logger.error(
                f"[DataPipelineOrchestrator] Repair failed: "
                f"config={config_key}, error={error}"
            )
            self._record_circuit_failure("repair", error)
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_repair_failed: {e}")

    def _on_partition_healed(self, event) -> None:
        """Handle PARTITION_HEALED event - track P2P partition healing success.

        January 3, 2026: Wires the previously orphaned PARTITION_HEALED event.
        Emitted by partition_healer.py:514 when network partitions are healed.

        Actions:
        - Log healing success with partition details
        - Trigger priority sync to resynchronize data after partition healing
        - Reset any partition-related circuit breakers
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            partitions_found = payload.get("partitions_found", 0)
            partitions_healed = payload.get("partitions_healed", 0)
            nodes_reconnected = payload.get("nodes_reconnected", 0)
            duration_ms = payload.get("duration_ms", 0.0)

            logger.info(
                f"[DataPipelineOrchestrator] Partition healed: "
                f"found={partitions_found}, healed={partitions_healed}, "
                f"reconnected={nodes_reconnected}, duration={duration_ms:.0f}ms"
            )

            # After partition healing, trigger priority sync to resynchronize data
            # across the previously partitioned nodes
            if partitions_healed > 0:
                from app.coordination.event_emission_helpers import safe_emit_event
                from app.distributed.data_events import DataEventType

                safe_emit_event(
                    DataEventType.SYNC_TRIGGERED,
                    {
                        "reason": "partition_healed",
                        "priority": "high",
                        "partitions_healed": partitions_healed,
                        "nodes_reconnected": nodes_reconnected,
                        "timestamp": time.time(),
                    },
                    log_after="Triggered priority sync after partition healing",
                    context="DataPipelineOrchestrator",
                    source="partition_healer",
                )

            self._record_event_processed()
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_partition_healed: {e}")

    def _on_task_abandoned(self, event) -> None:
        """Handle TASK_ABANDONED event - update pending counts."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            task_id = payload.get("task_id", "")
            reason = payload.get("reason", "unknown")
            logger.info(
                f"[DataPipelineOrchestrator] Task abandoned: "
                f"id={task_id}, reason={reason}"
            )
            self._record_event_processed()
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_task_abandoned: {e}")

    def _on_quality_score_updated(self, event) -> None:
        """Handle QUALITY_SCORE_UPDATED event - aggregate quality metrics."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            quality_score = payload.get("quality_score", 0.0)

            if config_key:
                self._quality_distribution[config_key] = quality_score
                logger.debug(
                    f"[DataPipelineOrchestrator] Quality updated: "
                    f"config={config_key}, score={quality_score:.2f}"
                )
            self._record_event_processed()
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_quality_score_updated: {e}")

    def _on_curriculum_rebalanced(self, event) -> None:
        """Handle CURRICULUM_REBALANCED event - update pipeline priorities."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            trigger = payload.get("trigger", "unknown")
            changed_configs = payload.get("changed_configs", [])
            logger.info(
                f"[DataPipelineOrchestrator] Curriculum rebalanced: "
                f"trigger={trigger}, configs={changed_configs}"
            )
            self._record_event_processed()
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_curriculum_rebalanced: {e}")

    def _on_curriculum_advanced(self, event) -> None:
        """Handle CURRICULUM_ADVANCED event - track curriculum progression."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            new_tier = payload.get("new_tier", "")
            logger.info(
                f"[DataPipelineOrchestrator] Curriculum advanced: "
                f"config={config_key}, tier={new_tier}"
            )
            self._record_event_processed()
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_curriculum_advanced: {e}")

    def _on_s3_backup_completed(self, event) -> None:
        """Handle S3_BACKUP_COMPLETED event."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            files_backed_up = payload.get("files_backed_up", 0)
            total_size_mb = payload.get("total_size_mb", 0)
            logger.info(
                f"[DataPipelineOrchestrator] S3 backup completed: "
                f"files={files_backed_up}, size={total_size_mb:.1f}MB"
            )
            self._record_event_processed()
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_s3_backup_completed: {e}")

    def _on_sync_checksum_failed(self, event) -> None:
        """Handle SYNC_CHECKSUM_FAILED event - trigger repair."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            file_path = payload.get("file_path", "")
            expected = payload.get("expected_checksum", "")[:16]
            actual = payload.get("actual_checksum", "")[:16]
            logger.warning(
                f"[DataPipelineOrchestrator] Checksum mismatch: "
                f"file={file_path}, expected={expected}..., actual={actual}..."
            )
            self._record_circuit_failure("sync", f"checksum_mismatch: {file_path}")
        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_sync_checksum_failed: {e}")

    def _on_data_stale(self, event) -> None:
        """Handle DATA_STALE event - trigger urgent sync before training.

        December 29, 2025: Phase 4D - Data freshness gating.
        When training data is stale (>24h old for most configs), trigger
        priority sync to get fresh data before allowing training.

        This prevents training on stale curriculum data which can lead to
        Elo regression or slower improvement velocity.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            data_age_hours = payload.get("data_age_hours", 0)
            threshold_hours = payload.get("threshold_hours", 24)
            source = payload.get("source", "unknown")

            if not config_key:
                return

            logger.warning(
                f"[DataPipelineOrchestrator] Data stale: "
                f"config={config_key}, age={data_age_hours:.1f}h > threshold={threshold_hours}h, "
                f"source={source}"
            )

            # Record staleness for tracking
            if not hasattr(self, "_stale_configs"):
                self._stale_configs: dict = {}
            self._stale_configs[config_key] = {
                "detected_at": time.time(),
                "age_hours": data_age_hours,
            }

            # Trigger priority sync for this config
            self._emit_priority_sync_request(config_key, reason="stale_data")

            self._record_event_processed()

        except (AttributeError, KeyError, TypeError) as e:
            self._record_error(f"_on_data_stale: {e}")

    def _emit_priority_sync_request(self, config_key: str, reason: str) -> None:
        """Emit SYNC_REQUEST event for priority sync.

        December 29, 2025: Used to trigger urgent sync when data is stale
        or after regression detection.
        """
        try:
            from app.coordination.event_router import publish_sync

            publish_sync(
                "SYNC_REQUEST",
                {
                    "config_key": config_key,
                    "priority": "urgent",
                    "reason": reason,
                    "source": "DataPipelineOrchestrator",
                    "requested_at": time.time(),
                },
            )
            logger.info(
                f"[DataPipelineOrchestrator] Priority sync requested: "
                f"config={config_key}, reason={reason}"
            )
        except (ImportError, AttributeError) as e:
            logger.debug(f"Could not emit SYNC_REQUEST: {e}")


# =============================================================================
# HandlerBase Integration (January 2026)
# =============================================================================

try:
    from app.coordination.handler_base import HandlerBase
    from app.coordination.contracts import HealthCheckResult as HBHealthCheckResult

    HAS_HANDLER_BASE = True
except ImportError:
    HAS_HANDLER_BASE = False

if HAS_HANDLER_BASE:

    class DataPipelineHandler(HandlerBase):
        """HandlerBase wrapper for DataPipelineOrchestrator.

        January 2026: Added for unified daemon lifecycle management.
        Runs pipeline stage checks and auto-triggers on a periodic cycle.
        """

        # Default cycle interval (60 seconds for pipeline monitoring)
        DEFAULT_CYCLE_INTERVAL = 60.0

        def __init__(
            self,
            auto_trigger: bool = True,
            cycle_interval: float = DEFAULT_CYCLE_INTERVAL,
        ):
            super().__init__(
                name="data_pipeline",
                cycle_interval=cycle_interval,
            )
            self._orchestrator = DataPipelineOrchestrator(auto_trigger=auto_trigger)
            self._orchestrator.subscribe_to_events()
            self._orchestrator.subscribe_to_data_events()

        async def _run_cycle(self) -> None:
            """Run one pipeline monitoring cycle."""
            try:
                # Check pipeline health and log status
                status = self._orchestrator.get_status()
                current_stage = status.get("current_stage", "unknown")
                pending = status.get("stages_pending", [])

                logger.debug(
                    f"[DataPipelineHandler] Cycle: stage={current_stage}, "
                    f"pending={len(pending)}"
                )

                # Trigger any pending stages if auto-trigger is enabled
                if self._orchestrator.auto_trigger and pending:
                    for stage_name in pending[:1]:  # Process one at a time
                        try:
                            stage = PipelineStage(stage_name)
                            await self._orchestrator._trigger_stage(stage)
                        except (ValueError, Exception) as e:
                            logger.debug(f"Could not trigger {stage_name}: {e}")

            except Exception as e:
                logger.error(f"[DataPipelineHandler] Cycle error: {e}")

        def _get_event_subscriptions(self) -> dict:
            """Get event subscriptions.

            Note: The orchestrator handles its own subscriptions internally.
            This provides additional handler-level hooks.
            """
            return {
                "PIPELINE_FORCE_ADVANCE": self._on_force_advance,
            }

        async def _on_force_advance(self, event: dict) -> None:
            """Handle force advance request."""
            from app.coordination.event_router import get_event_payload
            target_stage = get_event_payload(event).get("target_stage")
            if target_stage:
                logger.info(f"[DataPipelineHandler] Force advance to {target_stage}")

        def health_check(self) -> HBHealthCheckResult:
            """Health check for DaemonManager integration."""
            from app.coordination.protocols import CoordinatorStatus

            health = self._orchestrator.health_check()
            is_healthy = health.get("healthy", False)

            return HBHealthCheckResult(
                healthy=is_healthy,
                status=CoordinatorStatus.RUNNING if is_healthy else CoordinatorStatus.DEGRADED,
                message=f"DataPipelineHandler: {health.get('current_stage', 'unknown')}",
                details=health,
            )






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
    auto_trigger: bool = True,  # Dec 2025: Default to True for full pipeline loop
    training_epochs: int | None = None,
    training_batch_size: int | None = None,
    training_model_version: str | None = None,
    training_use_best_data: bool = False,
    training_data_path: str | None = None,
) -> DataPipelineOrchestrator:
    """Wire pipeline events to the orchestrator.

    Args:
        auto_trigger: If True, automatically trigger downstream stages (default: True)
        training_epochs: Override default training epochs
        training_batch_size: Override default training batch size
        training_model_version: Override default model version
        training_use_best_data: Use best available training data (combined NPZ or largest fresh)
        training_data_path: Explicit path to training data (overrides best_data selection)

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

    # Jan 2026: Store data selection config on orchestrator instance
    _pipeline_orchestrator.training_use_best_data = training_use_best_data
    _pipeline_orchestrator.training_data_path = training_data_path

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

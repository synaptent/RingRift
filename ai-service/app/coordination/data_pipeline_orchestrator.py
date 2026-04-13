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
import time
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.config.coordination_defaults import CircuitBreakerDefaults
from app.config.ports import get_local_p2p_status_url
from app.coordination.coordinator_persistence import StatePersistenceMixin
from app.coordination.protocols import (
    CoordinatorStatus,
    HealthCheckResult,
    register_coordinator,
    unregister_coordinator,
)
from app.coordination.event_utils import parse_config_key
from app.coordination.pipeline_shared import (
    IterationRecord,
    OperationMode,
    PipelineStage,
    PipelineStats,
    StageTransition,
)

# December 2025: Import mixin classes for DataPipelineOrchestrator decomposition
from app.coordination.pipeline_event_handler_mixin import PipelineEventHandlerMixin
from app.coordination.pipeline_metrics_mixin import PipelineMetricsMixin
from app.coordination.pipeline_stage_mixin import PipelineStageMixin
from app.coordination.pipeline_trigger_mixin import PipelineTriggerMixin
from app.utils.sqlite_utils import connect_safe

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.coordination.coordinator_config import PipelineConfig


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


from app.coordination.pipeline_stages import PipelineStagesMixin

class DataPipelineOrchestrator(
    PipelineStagesMixin,
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
        except ImportError:
            return False

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                async with session.get(get_local_p2p_status_url()) as response:
                    return response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
            return False

    def _subscribe_local_only_events(self) -> None:
        """Subscribe to events that work in local-only mode.

        Jan 2, 2026: These events don't require cluster connectivity.
        """
        try:
            from app.coordination.event_router import subscribe

            # Local file-based events that work without cluster
            local_events = [
                "LOCAL_NPZ_CREATED",
                "LOCAL_GAME_SAVED",
                "LOCAL_MODEL_SAVED",
            ]

            for event_type in local_events:
                subscribe(event_type, self._on_local_event)

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
            from app.coordination.event_router import StageEvent, subscribe

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
                    subscribe(event, handler)
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
            from app.coordination.event_router import DataEventType, subscribe

            # Subscribe to quality and cache events
            subscribe(
                DataEventType.QUALITY_DISTRIBUTION_CHANGED.value,
                self._on_quality_distribution_changed,
            )
            subscribe(
                DataEventType.CACHE_INVALIDATED.value,
                self._on_cache_invalidated,
            )

            # Subscribe to optimization events (December 2025)
            subscribe(
                DataEventType.CMAES_TRIGGERED.value,
                self._on_optimization_triggered,
            )
            subscribe(
                DataEventType.NAS_TRIGGERED.value,
                self._on_optimization_triggered,
            )

            # Subscribe to resource constraint events (December 2025)
            subscribe(
                DataEventType.RESOURCE_CONSTRAINT_DETECTED.value,
                self._on_resource_constraint_detected,
            )
            subscribe(
                DataEventType.BACKPRESSURE_ACTIVATED.value,
                self._on_backpressure_activated,
            )
            subscribe(
                DataEventType.BACKPRESSURE_RELEASED.value,
                self._on_backpressure_released,
            )

            # Subscribe to promotion candidate events (December 2025)
            # This wires the previously orphaned PROMOTION_CANDIDATE event
            subscribe(
                DataEventType.PROMOTION_CANDIDATE.value,
                self._on_promotion_candidate,
            )

            # Subscribe to database lifecycle events (December 2025 - Phase 4A.3)
            subscribe(
                DataEventType.DATABASE_CREATED.value,
                self._on_database_created,
            )

            # Subscribe to training threshold events
            subscribe(
                DataEventType.TRAINING_THRESHOLD_REACHED.value,
                self._on_training_threshold_reached,
            )

            # Subscribe to promotion lifecycle events
            subscribe(
                DataEventType.PROMOTION_STARTED.value,
                self._on_promotion_started,
            )

            # Subscribe to work queue events
            subscribe(
                DataEventType.WORK_QUEUED.value,
                self._on_work_queued,
            )

            # Core pipeline events (DataEventType fallback for StageEvent wiring)
            subscribe(
                DataEventType.SELFPLAY_COMPLETE.value,
                self._on_data_selfplay_complete,
            )
            subscribe(
                DataEventType.DATA_SYNC_COMPLETED.value,
                self._on_data_sync_completed,
            )
            subscribe(
                DataEventType.DATA_SYNC_FAILED.value,
                self._on_data_sync_failed,
            )
            subscribe(
                DataEventType.TRAINING_COMPLETED.value,
                self._on_data_training_completed,
            )
            subscribe(
                DataEventType.TRAINING_FAILED.value,
                self._on_data_training_failed,
            )
            subscribe(
                DataEventType.EVALUATION_COMPLETED.value,
                self._on_data_evaluation_completed,
            )
            subscribe(
                DataEventType.EVALUATION_FAILED.value,
                self._on_data_evaluation_failed,
            )
            subscribe(
                DataEventType.MODEL_PROMOTED.value,
                self._on_data_model_promoted,
            )

            # Orphan games detection - triggers sync and registration (Dec 2025)
            subscribe(
                DataEventType.ORPHAN_GAMES_DETECTED.value,
                self._on_orphan_games_detected,
            )
            subscribe(
                DataEventType.ORPHAN_GAMES_REGISTERED.value,
                self._on_orphan_games_registered,
            )

            # December 2025: Subscribe to GAME_SYNCED for export triggering
            # When games are synced to training nodes, this can trigger NPZ export
            subscribe(
                DataEventType.GAME_SYNCED.value,
                self._on_game_synced,
            )

            # December 2025: Subscribe to exploration and sync feedback events
            # These events trigger curriculum adjustments and adaptive exploration
            subscribe(
                DataEventType.EXPLORATION_BOOST.value,
                self._on_exploration_boost,
            )
            subscribe(
                DataEventType.SYNC_TRIGGERED.value,
                self._on_sync_triggered,
            )

            # December 2025: Subscribe to DATA_STALE for training freshness
            # When training data becomes stale, we trigger urgent sync
            subscribe(
                DataEventType.DATA_STALE.value,
                self._on_data_stale,
            )

            # December 2025 Phase 11: Additional event subscriptions for improved
            # pipeline coordination and regression handling
            subscribe(
                DataEventType.NEW_GAMES_AVAILABLE.value,
                self._on_new_games_available,
            )
            subscribe(
                DataEventType.REGRESSION_DETECTED.value,
                self._on_regression_detected,
            )
            subscribe(
                DataEventType.PROMOTION_FAILED.value,
                self._on_promotion_failed,
            )

            # December 2025: Subscribe to consolidation events for training pipeline fix
            # Consolidation merges scattered selfplay games into canonical databases
            subscribe(
                DataEventType.CONSOLIDATION_STARTED.value,
                self._on_consolidation_started,
            )
            subscribe(
                DataEventType.CONSOLIDATION_COMPLETE.value,
                self._on_consolidation_complete,
            )

            # December 2025: Subscribe to NPZ combination events for quality-weighted data
            # NPZ combination combines historical + fresh data with quality weighting
            subscribe(
                DataEventType.NPZ_COMBINATION_COMPLETE.value,
                self._on_npz_combination_complete,
            )
            subscribe(
                DataEventType.NPZ_COMBINATION_FAILED.value,
                self._on_npz_combination_failed,
            )

            # December 2025: Wire previously orphaned critical events
            # These events were being emitted but had no subscribers
            subscribe(
                DataEventType.REPAIR_COMPLETED.value,
                self._on_repair_completed,
            )
            subscribe(
                DataEventType.REPAIR_FAILED.value,
                self._on_repair_failed,
            )
            # December 2025: Track abandoned tasks for accurate pending count
            subscribe(
                DataEventType.TASK_ABANDONED.value,
                self._on_task_abandoned,
            )
            subscribe(
                DataEventType.QUALITY_SCORE_UPDATED.value,
                self._on_quality_score_updated,
            )
            subscribe(
                DataEventType.CURRICULUM_REBALANCED.value,
                self._on_curriculum_rebalanced,
            )
            subscribe(
                DataEventType.CURRICULUM_ADVANCED.value,
                self._on_curriculum_advanced,
            )

            # December 2025: Subscribe to S3 backup events for pipeline tracking
            subscribe(
                DataEventType.S3_BACKUP_COMPLETED.value,
                self._on_s3_backup_completed,
            )

            # December 2025: Subscribe to sync integrity events for repair triggering
            if hasattr(DataEventType, 'SYNC_CHECKSUM_FAILED'):
                subscribe(
                    DataEventType.SYNC_CHECKSUM_FAILED.value,
                    self._on_sync_checksum_failed,
                )

            # January 3, 2026: Subscribe to partition healing events
            # PARTITION_HEALED emitted by partition_healer.py:514 when healing succeeds
            subscribe(
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
                f"[DataPipelineOrchestrator] Failed to subscribe to data events: {e}",
                exc_info=True,
            )
            return False
















    # =========================================================================
    # Orphaned Event Handlers (December 27, 2025)
    # These handlers wire previously orphaned events that were being emitted
    # but had no subscribers, breaking feedback loops.
    # =========================================================================















    # =========================================================================
    # Quality Gate Methods (December 2025 - Phase 14)
    # =========================================================================














    # =========================================================================
    # Circuit Breaker and Auto-Trigger Helpers (December 2025)
    # =========================================================================








    # =========================================================================
    # Quality and Cache Event Handlers (December 2025)
    # =========================================================================




    # =========================================================================
    # Resource Constraint Event Handlers (December 2025)
    # =========================================================================






























    # =========================================================================
    # Event Handlers (December 29, 2025 - Phase 7: Missing handlers implementation)
    # =========================================================================






























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
